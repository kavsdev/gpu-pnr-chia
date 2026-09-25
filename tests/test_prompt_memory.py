import json
import pytest
from pnr_node.contract import EvalRequest, EvalResult, PromptConfig, Constraints, failure_signature, score, KNOBS, CORE_KNOBS
from pnr_node.db import ResultsDB
from pnr_node.llm import FakeProvider, Msg
from pnr_node.mock_eval import MockEvaluator
from pnr_node.search import LLMProposer, run_search

def test_default_behavior_unchanged_byte_identical():
    """Gate 1: Default PromptConfig must produce byte-identical prompts to the hardcoded ones."""
    db, ev = ResultsDB(), MockEvaluator()
    _SYSTEM_ORIG = (
        "You tune the global-placement parameters of DREAMPlace for a chip design. Each candidate is "
        "evaluated by a cheap proxy (HPWL) and the best few by a full route that reports routed wirelength, "
        "worst negative slack (wns, ns, negative is bad), and DRC violations. Lower score is better. "
        "Propose new candidates that improve on the history, use what routing revealed about failures, and "
        "stay inside the knob ranges. Reply with JSON only."
    )
    
    prov = FakeProvider(replies=[json.dumps({"candidates": [{"target_density": 0.7}]})])
    prop = LLMProposer(prov, db, arm="llm")
    prop.propose([], 1)
    
    call_sys, call_msgs = prov.calls[0]
    assert call_sys == _SYSTEM_ORIG
    assert "Return {\"reasoning\": \"<2-4 sentences>\", \"candidates\": [{...knob values...}, ...]} with exactly 1 candidates." in call_msgs[0].content

def test_config_driven_prompts():
    """Gate 2: Non-default per-stage prompt overrides are used correctly."""
    prompt_cfg = PromptConfig(
        system_prompts={"proxy": "PROXY_SYSTEM", "route": "ROUTE_SYSTEM"},
        templates={"proxy": "PROXY_TEMPLATE {history}", "route": "ROUTE_TEMPLATE {history}"}
    )
    prov = FakeProvider(replies=[json.dumps({"candidates": []}), json.dumps({"candidates": []})])
    prop = LLMProposer(prov, prompt_cfg=prompt_cfg)
    
    prop.propose([], 1, stage="proxy")
    assert prov.calls[0][0] == "PROXY_SYSTEM"
    assert "PROXY_TEMPLATE" in prov.calls[0][1][0].content
    
    prop.propose([], 1, stage="route")
    assert prov.calls[1][0] == "ROUTE_SYSTEM"
    assert "ROUTE_TEMPLATE" in prov.calls[1][1][0].content

def test_two_call_feedback_mode():
    """Gate 3: two_call mode makes two calls, second contains first's response, both logged."""
    db = ResultsDB()
    prompt_cfg = PromptConfig(feedback_mode="two_call")
    prov = FakeProvider(replies=["CRITIQUE_RESPONSE", json.dumps({"candidates": []})])
    prop = LLMProposer(prov, db=db, prompt_cfg=prompt_cfg)
    
    prop.propose([], 1)
    
    assert len(prov.calls) == 2
    assert "identify patterns" in prov.calls[0][0] # default critique sys
    assert "CRITIQUE_RESPONSE" in prov.calls[1][1][0].content # second call contains first response
    
    calls = db.con.execute("SELECT phase, response FROM llm_calls ORDER BY id").fetchall()
    assert len(calls) == 2
    assert calls[0]["phase"] == "critique"
    assert calls[1]["phase"] == "propose"

def test_failure_signature_and_recall():
    """Failure signature and historical recall."""
    db = ResultsDB()
    # 1. Create a past "failure then improvement" in the DB
    res_fail = EvalResult(True, "route", routed_wl=1000.0, drc_violations=50) # sig: drc:many
    req_fail = EvalRequest("gcd", placer_cfg={"target_density": 0.8}, tool_id="mock")
    db.record_eval("past_arm", 1, req_fail, res_fail)
    
    res_good = EvalResult(True, "route", routed_wl=500.0, drc_violations=0)
    req_good = EvalRequest("gcd", placer_cfg={"target_density": 0.7}, tool_id="mock")
    db.record_eval("past_arm", 2, req_good, res_good)
    
    # 2. Current history has a matching failure
    current_history = [{
        "design": "gcd",
        "cfg": {"target_density": 0.8},
        "result": res_fail,
        "score": score(res_fail),
        "stage": "route"
    }]
    
    prov = FakeProvider(replies=[json.dumps({"candidates": []})])
    prop = LLMProposer(prov, db=db, prompt_cfg=PromptConfig(use_memory=True))
    prop.propose(current_history, 1)
    
    prompt = prov.calls[0][1][0].content
    assert "Historical Recall" in prompt
    assert "Failures matching drc:many" in prompt
    assert "target_density -> lower" in prompt

def test_constraints_and_knob_narrowing():
    """Constraints restrict knobs and appear in prompt."""
    constraints = Constraints(
        id="C1",
        target_period=1.5,
        allowed_knobs=["target_density", "gamma"]
    )
    prov = FakeProvider(replies=[json.dumps({"candidates": [{"target_density": 0.9, "gamma": 5.0, "density_weight": 1e-4}]})])
    prop = LLMProposer(prov, constraints=constraints)
    
    # Knob narrowing check: it should only tune the allowed ones
    assert set(prop.knobs.keys()) == {"target_density", "gamma"}
    
    cands = prop.propose([], 1)
    # The proposer clamps candidates against its narrowed knobs.
    # Knobs not in self.knobs are NOT included in the proposer's output.
    assert "density_weight" not in cands[0]
    assert "target_density" in cands[0]
    
    # But when we create an EvalRequest, it fills missing knobs with defaults.
    req = EvalRequest("gcd", placer_cfg=cands[0])
    assert "density_weight" in req.placer_cfg
    assert req.placer_cfg["density_weight"] == CORE_KNOBS["density_weight"].default
    
    prompt = prov.calls[0][0] # System prompt contains constraints
    assert "Constraints (ID: C1)" in prompt
    assert "Target clock period: 1.5ns" in prompt

def test_failure_signature_buckets():
    """Verify that failure_signature buckets correctly categorize different failure shapes."""
    # 1. Clean result -> None
    res = EvalResult(True, "route", drc_violations=0, wns=0.0)
    assert failure_signature(res) is None
    
    # 2. Minor timing -> wns:clean-ish
    res = EvalResult(True, "route", drc_violations=0, wns=-0.05)
    assert "wns:clean-ish" in failure_signature(res)
    
    # 3. Real minor timing -> wns:minor
    res = EvalResult(True, "route", drc_violations=0, wns=-0.5)
    assert "wns:minor" in failure_signature(res)
    
    # 4. Major timing -> wns:major
    res = EvalResult(True, "route", drc_violations=0, wns=-1.5)
    assert "wns:major" in failure_signature(res)
    
    # 4. Few DRCs -> drc:few
    res = EvalResult(True, "route", drc_violations=5, wns=0.0)
    assert "drc:few" in failure_signature(res)
    
    # 5. Massive DRCs -> drc:massive
    res = EvalResult(True, "route", drc_violations=500, wns=0.0)
    assert "drc:massive" in failure_signature(res)
    
    # 6. Specific DRC classes from provenance
    res = EvalResult(True, "route", drc_violations=10, wns=0.0, provenance={"drc_classes": ["met1", "met2"]})
    assert "drc:met1,met2" in failure_signature(res)
    
    # 7. High overflow -> overflow:high
    res = EvalResult(False, "route", error="timeout", provenance={"dp_overflow": 0.2})
    assert "overflow:high" in failure_signature(res)
    
    # 8. Error message categorization
    res = EvalResult(False, "route", error="NVIDIA driver: crashed")
    assert "err:NVIDIA driver" in failure_signature(res)

def test_eval_request_key_stability():
    """Verify that adding constraints_id doesn't break backward compatibility for existing keys."""
    cfg = {"target_density": 0.8}
    # Original-style request (no constraints_id passed)
    req1 = EvalRequest("gcd", "nangate45", cfg, 0, "route", "tool1")
    # New-style request with empty constraints_id
    req2 = EvalRequest("gcd", "nangate45", cfg, 0, "route", "tool1", "")
    
    assert req1.key() == req2.key()
    
    # Request with real constraints_id should have a different key
    req3 = EvalRequest("gcd", "nangate45", cfg, 0, "route", "tool1", "C1")
    assert req1.key() != req3.key()

def test_prompt_config_fallback_logic():
    """Verify the fallback chain for per-stage prompt overrides."""
    # 1. No override -> default _SYSTEM
    cfg1 = PromptConfig()
    prop1 = LLMProposer(FakeProvider(replies=[json.dumps({"candidates": []})]), prompt_cfg=cfg1)
    prop1.propose([], 1, stage="proxy")
    assert "You tune the global-placement parameters" in prop1.p.calls[0][0]
    
    # 2. Default override used for all stages
    cfg2 = PromptConfig(system_prompts={"default": "GLOBAL_OVERRIDE"})
    prop2 = LLMProposer(FakeProvider(replies=[json.dumps({"candidates": []})]), prompt_cfg=cfg2)
    prop2.propose([], 1, stage="proxy")
    assert prop2.p.calls[0][0] == "GLOBAL_OVERRIDE"
    
    # 3. Stage-specific override takes precedence
    cfg3 = PromptConfig(system_prompts={"default": "GLOBAL", "proxy": "PROXY_ONLY"})
    prop3 = LLMProposer(FakeProvider(replies=[json.dumps({"candidates": []})]), prompt_cfg=cfg3)
    prop3.propose([], 1, stage="proxy")
    assert prop3.p.calls[0][0] == "PROXY_ONLY"
    prop3.propose([], 1, stage="route")
    assert prop3.p.calls[1][0] == "GLOBAL"

def test_run_search_with_constraints_integration():
    """Verify that run_search correctly threads constraints_id into recorded evaluations."""
    db = ResultsDB()
    ev = MockEvaluator()
    constraints = Constraints(id="MY_CONSTRAINTS", target_period=1.0)
    
    # Use a proposer that has the constraints
    prov = FakeProvider(replies=[json.dumps({"candidates": [{"target_density": 0.75}]})])
    prop = LLMProposer(prov, db=db, constraints=constraints)
    
    # Run a small search
    run_search(ev, prop, db, "test_arm", "gcd", rounds=1, batch=1, route_top=1, constraints=constraints)
    
    # Check the recorded evaluation in the DB
    rows = db.con.execute("SELECT req_json FROM evals").fetchall()
    for row in rows:
        req_data = json.loads(row[0])
        assert req_data["constraints_id"] == "MY_CONSTRAINTS"

def test_recall_hints_pareto_isolation():
    """Verify that proxy-only arms never see hints derived from routing successes."""
    db = ResultsDB()
    # Past failure and improvement at 'route' stage
    res_fail = EvalResult(True, "route", routed_wl=1000.0, drc_violations=50) # sig: drc:many
    res_good = EvalResult(True, "route", routed_wl=500.0, drc_violations=0)
    db.record_eval("arm", 1, EvalRequest("gcd", placer_cfg={}), res_fail)
    db.record_eval("arm", 2, EvalRequest("gcd", placer_cfg={"target_density": 0.7}), res_good)
    
    current_history = [{"design": "gcd", "cfg": {}, "result": res_fail, "score": 1000.0, "stage": "proxy"}]
    
    # 1. Arm with show_routed=True sees the hint
    prop1 = LLMProposer(FakeProvider(replies=[json.dumps({"candidates": []})]), db=db, 
                        show_routed=True, prompt_cfg=PromptConfig(use_memory=True))
    prop1.propose(current_history, 1)
    assert "Historical Recall" in prop1.p.calls[0][1][0].content
    
    # 2. Arm with show_routed=False (ablation) MUST NOT see the hint
    prop2 = LLMProposer(FakeProvider(replies=[json.dumps({"candidates": []})]), db=db, 
                        show_routed=False, prompt_cfg=PromptConfig(use_memory=True))
    prop2.propose(current_history, 1)
    assert "Historical Recall" not in prop2.p.calls[0][1][0].content
