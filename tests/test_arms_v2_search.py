"""v2 arms: categorical knobs, TPE, run_search_v2, v2 arms, ArmRunEvaluator, gemini_rest, driver."""
import importlib.util
import json
import os
import stat
import threading
import time

import pytest

from pnr_node.contract import (FLOW_KNOBS, EvalRequest, EvalResult, Knob, clamp_config, default_config, passes,
                               score_timing)
from pnr_node.db import CachedEvaluator, ResultsDB
from pnr_node.llm import (CostMeter, FakeProvider, GeminiREST, LLMBudgetExceeded, LLMResponse, Msg, make_provider,
                          pick_newest_pro)
from pnr_node.mock_eval import FlowMockEvaluator
from pnr_node.search import (LLMProposer, RandomProposer, TPEProposer, _knob_table, config_stats, run_search_v2)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- contract pieces this code relies on
def test_flow_cfg_clamped_and_key_unchanged_when_empty():
    a = EvalRequest("gcd", seed=3, stage="route")
    assert "flow_cfg" not in json.dumps(a.__dict__) or a.flow_cfg == {}
    b = EvalRequest("gcd", seed=3, stage="cts", flow_cfg={"target_density": 9, "rebuf_full": "false", "x": 1})
    assert b.flow_cfg["target_density"] == 0.80 and b.flow_cfg["rebuf_full"] is False and "x" not in b.flow_cfg
    assert b.flow_cfg["rebuf_stops"] == "0.6,0.2"  # missing -> default
    c = EvalRequest("gcd", seed=3, stage="cts", flow_cfg=dict(b.flow_cfg))
    assert b.key() == c.key() != EvalRequest("gcd", seed=3, stage="cts").key()


def test_score_timing_and_passes():
    ok = EvalResult(True, "cts", wns=-0.2, tns=-50.0)
    assert score_timing(ok) == pytest.approx(50 + 20)
    assert score_timing(EvalResult(False, "cts")) == float("inf")
    assert score_timing(EvalResult(True, "route", wns=-0.1, tns=-1, drc_violations=3)) == float("inf")
    ref = {"wns": -0.25, "tns": -60.0, "wl": 100.0}
    assert passes(ok, ref) and not passes(EvalResult(True, "cts", wns=-0.3, tns=-50.0), ref)
    assert not passes(EvalResult(True, "route", wns=-0.1, tns=-1, routed_wl=104.0), ref)


# ---------------------------------------------------------------- proposers
def test_random_proposer_categoricals_valid_and_deterministic():
    a = RandomProposer(5, FLOW_KNOBS).propose([], 30)
    assert a == RandomProposer(5, FLOW_KNOBS).propose([], 30)
    for c in a:
        for k, kb in FLOW_KNOBS.items():
            if kb.choices is not None:
                assert any(c[k] == ch and type(c[k]) is type(ch) for ch in kb.choices)
            else:
                assert kb.lo <= c[k] <= kb.hi
    assert len({c["rebuf_stops"] for c in a}) > 2 and {c["rebuf_full"] for c in a} == {True, False}


def test_knob_table_renders_choices():
    t = _knob_table(FLOW_KNOBS)
    assert '- rebuf_stops: one of ["0.6,0.2", "0.6,0.4,0.2"' in t and "(categorical)" in t
    assert "- rebuf_full: one of [true, false] default true" in t
    assert "- density_weight: [2e-05, 0.0004] default 8e-05 (log scale)" in t


def _hist(cfgs_scores):
    return [{"cfg": clamp_config(c, FLOW_KNOBS), "score": s, "failed": s is None} for c, s in cfgs_scores]


def test_tpe_startup_random_then_model_and_deterministic():
    hist = []
    rng_cfgs = RandomProposer(3, FLOW_KNOBS).propose([], 20)
    for c in rng_cfgs:  # synthetic objective: best near target_density 0.4 and rebuf_stops "0.7,0.3"
        hist.append((c, abs(c["target_density"] - 0.4) * 100 + (0 if c["rebuf_stops"] == "0.7,0.3" else 10)))
    h = _hist(hist)
    t1, t2 = TPEProposer(7), TPEProposer(7)
    assert t1.propose([], 6) == t2.propose([], 6)  # startup = the seeded random proposer
    p1, p2 = t1.propose(h, 8), t2.propose(list(reversed(h)), 8)
    assert p1 == p2  # deterministic and independent of history order
    for c in p1:
        assert clamp_config(c, FLOW_KNOBS) == c
    # the model concentrates proposals on the good region
    near = sum(abs(c["target_density"] - 0.4) < 0.1 for c in p1)
    rnd = sum(abs(c["target_density"] - 0.4) < 0.1 for c in RandomProposer(99, FLOW_KNOBS).propose([], 8))
    assert near >= max(5, rnd)
    assert sum(c["rebuf_stops"] == "0.7,0.3" for c in p1) >= 3  # random: ~8/6


def test_tpe_handles_all_failed_history():
    h = _hist([(c, None) for c in RandomProposer(1, FLOW_KNOBS).propose([], 10)])
    t = TPEProposer(0)
    t.n_proposed = 10
    out = t.propose(h, 3)
    assert len(out) == 3 and all(clamp_config(c, FLOW_KNOBS) == c for c in out)


def test_llm_proposer_timing_mode_prompt_and_categoricals():
    reply = json.dumps({"reasoning": "x", "candidates": [
        {"target_density": 0.5, "rebuf_stops": "0.7,0.3", "rebuf_full": False, "gamma": 5}]})
    fp = FakeProvider(replies=[reply])
    p = LLMProposer(fp, knobs=FLOW_KNOBS, objective="timing", fallback=RandomProposer(0, FLOW_KNOBS))
    hist = [{"cfg": clamp_config({}, FLOW_KNOBS), "score": 70.0, "wns": -0.2, "tns": -50.0, "n_seeds": 2,
             "failed": False, "result": EvalResult(True, "cts", wns=-0.2, tns=-50.0)},
            {"cfg": clamp_config({"target_density": 0.8}, FLOW_KNOBS), "score": None, "failed": True,
             "result": EvalResult(False, "cts", error="diverged")}]
    out = p.propose(hist, 2, stage="cts")
    system, msgs = fp.calls[0]
    assert "timing" in system and "TNS" in system
    assert '"rebuf_stops": "0.6,0.2"' in msgs[0].content and '"tns": -50.0' in msgs[0].content
    assert "diverged" in msgs[0].content and "one of" in msgs[0].content
    assert out[0]["rebuf_stops"] == "0.7,0.3" and out[0]["rebuf_full"] is False and len(out) == 2


# ---------------------------------------------------------------- run_search_v2
def test_run_search_v2_budget_promotion_and_best_over_two_seeds():
    db = ResultsDB()
    ev = FlowMockEvaluator(fail_rate=0.0)
    out = run_search_v2(ev, RandomProposer(0, FLOW_KNOBS), db, "a", "ariane133", budget_evals=11, batch=4,
                        seeds=(0,), promote_top=2, promote_seeds=(1,))
    rows = db.evals("a", "ariane133")
    assert out["evals_used"] == len(rows) == 11  # exact: the last round is truncated to the budget
    assert all(r["stage"] == "cts" and r["flow_cfg"] for r in rows)
    assert sum("promote" in (r["note"] or "") for r in rows) >= 2
    assert out["best_n_seeds"] >= 2 and not out["best_single_seed_fallback"]
    stats = {json.dumps(c["cfg"], sort_keys=True): c for c in config_stats(db, "a", "ariane133", "cts")}
    best = stats[json.dumps(out["best_cfg"], sort_keys=True)]
    assert best["score"] == out["best_score"] == min(c["score"] for c in stats.values()
                                                       if c["n_seeds"] >= 2 and not c["failed"])
    # the stored score column is the timing score
    assert rows[0]["score"] == pytest.approx(score_timing(rows[0]["result"]))


def test_run_search_v2_first_round_runs_even_past_deadline():
    db = ResultsDB()
    out = run_search_v2(FlowMockEvaluator(), RandomProposer(0, FLOW_KNOBS), db, "d", "gcd", budget_evals=40,
                        batch=3, deadline=time.time() - 1)
    assert out["rounds"] == 1 and out["evals_used"] == 5 and out["stop_reason"] == "deadline" and out["stopped_early"]


def test_run_search_v2_deadline_between_rounds():
    db = ResultsDB()

    class Slow(FlowMockEvaluator):
        def evaluate(self, req):
            time.sleep(0.02)
            return super().evaluate(req)

    out = run_search_v2(Slow(), RandomProposer(0, FLOW_KNOBS), db, "d", "gcd", budget_evals=100, batch=2,
                        deadline=time.time() + 0.05)
    assert 1 <= out["rounds"] < 25 and out["stop_reason"] == "deadline"


def test_run_search_v2_failures_count_and_never_promoted():
    class AllFail:
        tool_id = "x"

        def evaluate(self, req):
            return EvalResult(False, req.stage, error="boom")

    db = ResultsDB()
    out = run_search_v2(AllFail(), RandomProposer(0, FLOW_KNOBS), db, "f", "gcd", budget_evals=6, batch=3)
    assert out["evals_used"] == 6 and out["best_cfg"] is None and out["failed_configs"] == 6
    assert not any("promote" in (r["note"] or "") for r in db.evals("f", "gcd"))


def test_run_search_v2_llm_budget_ends_arm():
    meter = CostMeter(cap_usd=0.001, in_per_m=1000.0, out_per_m=1000.0)

    class Metered(FakeProvider):
        def complete(self, *a, **k):
            meter.check()
            meter.add(1, 1)  # $0.002 per call -> over the cap after one call
            return super().complete(*a, **k)

    reply = json.dumps({"candidates": [{"target_density": 0.5}] * 4})
    db = ResultsDB()
    p = LLMProposer(Metered(replies=[reply]), db, arm="l", knobs=FLOW_KNOBS, objective="timing")
    out = run_search_v2(FlowMockEvaluator(), p, db, "l", "gcd", budget_evals=40, batch=4)
    assert out["stop_reason"] == "llm_budget" and out["rounds"] == 1 and out["evals_used"] > 0
    assert meter.calls == 1


def test_run_search_v2_resume_continues_budget():
    db = ResultsDB()
    run_search_v2(FlowMockEvaluator(), RandomProposer(0, FLOW_KNOBS), db, "r", "gcd", budget_evals=5, batch=2)
    out = run_search_v2(FlowMockEvaluator(), RandomProposer(1, FLOW_KNOBS), db, "r", "gcd", budget_evals=9, batch=2)
    assert out["evals_used"] == 9 == len(db.evals("r", "gcd"))


def test_db_reads_old_rows_without_flow_cfg():
    db = ResultsDB()
    req = EvalRequest("gcd", stage="route")
    db.record_eval("old", 1, req, EvalResult(True, "route", routed_wl=1.0, drc_violations=0))
    db.con.execute("UPDATE evals SET req_json=?", (json.dumps({"placer_cfg": default_config(), "seed": 0}),))
    assert db.evals("old", "gcd")[0]["flow_cfg"] == {}


# ---------------------------------------------------------------- v2 arms / experiment
def test_run_experiment_v2_on_mock_with_fake_llm():
    from pnr_node.experiment import run_experiment_v2, summarize_v2

    db = ResultsDB()
    reply = json.dumps({"candidates": [{"target_density": 0.52, "netwt_pct": 15}, {"target_density": 0.5}]})
    res = run_experiment_v2(FlowMockEvaluator(fail_rate=0.0), db, "ariane133", arms=("default", "random", "tpe", "llm"), reps=2,
                            budget_evals=10, batch=3, default_seeds=(0, 1, 2), validate_seeds=(101, 102, 103),
                            workers=4, provider_factory=lambda: FakeProvider(replies=[reply]), log=lambda m: None)
    names = {s["arm"] for s in res["search"]}
    assert names == {"v2_default#r0", "v2_random#r0", "v2_random#r1", "v2_tpe#r0", "v2_tpe#r1", "v2_llm#r0",
                     "v2_llm#r1"}
    assert all(s["evals_used"] == 10 for s in res["search"] if "default" not in s["arm"])
    d = next(s for s in res["search"] if s["arm"] == "v2_default#r0")
    assert d["evals_used"] == 3 and d["best_cfg"] == clamp_config({}, FLOW_KNOBS)
    v = res["validation"]["v2_tpe#r0"]
    assert v["n"] == 3 and set(v["seed_scores"]) == {"101", "102", "103"}
    assert all(r["stage"] == "route" for r in db.evals("v2_tpe#r0|validate", "ariane133"))
    c = res["comparison"]
    assert c["sigma_source"] == "default_validation" and c["sigma_default"] > 0
    assert c["arms"]["default"]["diff_mean"] == 0
    for arm in ("random", "tpe", "llm"):
        row = c["arms"][arm]
        assert row["n_pairs"] == 6 and row["diff_2se"] is not None
        assert isinstance(row["beats_default_2se"], bool) and isinstance(row["clears_2sigma_lottery"], bool)
    json.dumps(res)  # serialisable (no EvalResult objects left)
    assert "| tpe | 2 |" in summarize_v2(res)


def test_llm_diag_hook():
    from pnr_node import experiment

    with pytest.raises(ValueError, match="llm_diag"):
        experiment.run_arm_v2(FlowMockEvaluator(), ResultsDB(), "llm_diag", 0, "gcd", budget_evals=2, batch=1)
    seen = {}

    def factory(name, rep, db, pf):
        seen["name"] = name
        return RandomProposer(rep, FLOW_KNOBS)

    old = dict(experiment.PROPOSERS_V2)
    try:
        experiment.register_arm_v2("llm_diag", factory)
        out = experiment.run_arm_v2(FlowMockEvaluator(), ResultsDB(), "llm_diag", 1, "gcd", budget_evals=2, batch=1)
        assert seen["name"] == "v2_llm_diag#r1" and out["evals_used"] == 2
    finally:
        experiment.PROPOSERS_V2.clear()
        experiment.PROPOSERS_V2.update(old)


# ---------------------------------------------------------------- ArmRunEvaluator with a fake run_arm.sh
FAKE_RUN_ARM = r"""#!/usr/bin/env bash
# fake run_arm.sh: <design> <label> args...; writes $NETWT_WORK/logs/arm_<design>_<label>.log ending in result JSON
D=$1; L=$2; shift 2
mkdir -p "$NETWT_WORK/logs"; LOG=$NETWT_WORK/logs/arm_${D}_${L}.log
echo "$D $L $*" >> "$NETWT_WORK/calls.txt"
sleep ${FAKE_SLEEP:-0}
if [[ "$*" == *'"rebuf_stops": "0.6"'* ]]; then
  { echo '{"ok": false, "stage": "cts", "error": "rejected: not_converged", "provenance": {"req_key": "c"}}'; echo "exit=0"; } > "$LOG"; exit 0; fi
if [[ "$*" == *'"target_density": 0.8'* ]]; then echo "Traceback: boom" > "$LOG"; echo "exit=1" >> "$LOG"; exit 1; fi
{ echo "ORFS noise"; echo '{"ok": true, "stage": "cts", "wns": -0.1, "tns": -9.0, "provenance": {"x": 1}}';
  echo '{"ok": true, "stage": "cts", "wns": -0.2, "tns": -20.0, "provenance": {"seed": 5}}'; echo "exit=0"; } > "$LOG"
"""


@pytest.fixture
def fake_arm(tmp_path, monkeypatch):
    script = tmp_path / "run_arm.sh"
    script.write_text(FAKE_RUN_ARM)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    work = tmp_path / "work"
    monkeypatch.setenv("PNR_RUN_ARM", str(script))
    monkeypatch.setenv("NETWT_WORK", str(work))
    return work


def test_arm_run_evaluator_parses_last_json_line(fake_arm):
    from pnr_node.arm_eval import ArmRunEvaluator

    ev = ArmRunEvaluator(arm="tpe")
    req = EvalRequest("ariane133", seed=5, stage="cts", flow_cfg={"target_density": 0.5}, tool_id=ev.tool_id)
    r = ev.evaluate(req)
    assert r.ok and r.tns == -20.0 and r.wns == -0.2 and r.provenance["seed"] == 5
    assert r.provenance["arm_label"] == f"v2_tpe_{req.key()}" and r.provenance["run_arm_rc"] == 0
    call = (fake_arm / "calls.txt").read_text().strip()
    assert call.startswith(f"ariane133 v2_tpe_{req.key()} --stage cts --seed 5 --flow-cfg ")
    assert json.loads(call.split("--flow-cfg ", 1)[1]) == req.flow_cfg and "--cfg" not in call


def test_arm_run_evaluator_failure_without_json(fake_arm):
    from pnr_node.arm_eval import ArmRunEvaluator

    r = ArmRunEvaluator().evaluate(EvalRequest("gcd", stage="cts", flow_cfg={"target_density": 0.8}))
    assert not r.ok and "rc=1" in r.error and "no result JSON" in r.error and "boom" in r.error


def test_arm_run_evaluator_pool_cap_and_dedupe(fake_arm, monkeypatch):
    from pnr_node.arm_eval import ArmRunEvaluator

    monkeypatch.setenv("FAKE_SLEEP", "0.3")
    base = ArmRunEvaluator(workers=2)
    a, b = base.for_arm("x"), base.for_arm("y")
    reqs = [EvalRequest("gcd", seed=s, stage="cts", flow_cfg={"gamma": 3.0}) for s in range(4)]
    dup = EvalRequest("gcd", seed=0, stage="cts", flow_cfg={"gamma": 3.0})
    t0 = time.time()
    out = {}
    th = [threading.Thread(target=lambda: out.__setitem__("a", a.evaluate_many(reqs))),
          threading.Thread(target=lambda: out.__setitem__("b", b.evaluate_many([dup])))]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    calls = (fake_arm / "calls.txt").read_text().strip().splitlines()
    assert len(calls) == 4  # the duplicate identity ran once
    assert wall >= 0.55  # 4 runs, at most 2 at a time across both views -> >= 2 waves
    assert all(r.ok for r in out["a"] + out["b"])


def test_arm_run_evaluator_with_cache_and_search(fake_arm):
    from pnr_node.arm_eval import ArmRunEvaluator

    db = ResultsDB()
    ev = CachedEvaluator(ArmRunEvaluator(workers=3).for_arm("random"), db)
    out = run_search_v2(ev, RandomProposer(0, FLOW_KNOBS), db, "v2_random#r0", "gcd", budget_evals=4, batch=2)
    assert out["evals_used"] == 4


# ---------------------------------------------------------------- gemini_rest provider (no network)
class _FakeHTTP:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gemini_rest_request_and_usage(monkeypatch):
    import urllib.request

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return _FakeHTTP({"candidates": [{"content": {"parts": [{"text": "think", "thought": True},
                                                                {"text": '{"a": 1}'}]}}],
                          "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100,
                                            "thoughtsTokenCount": 900}, "modelVersion": "gemini-x"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET123")
    meter = CostMeter(cap_usd=1.0)
    p = make_provider({"provider": "gemini_rest", "model": "gemini-3.1-pro-preview", "meter": meter})
    r = p.complete("sys", [Msg("user", "hi")], json_mode=True)
    assert seen["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/"
                           "gemini-3.1-pro-preview:generateContent")
    assert seen["headers"]["X-goog-api-key"] == "SECRET123" and "SECRET123" not in seen["url"]
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert r.text == '{"a": 1}' and r.input_tokens == 1000 and r.output_tokens == 1000
    assert meter.spent_usd == pytest.approx(1000 * 2.5e-6 + 1000 * 15e-6)
    assert "SECRET123" not in repr(p)


def test_gemini_rest_cap_blocks_call(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the API"))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    meter = CostMeter(cap_usd=0.01)
    meter.add(0, 1000)  # $0.015 already spent
    with pytest.raises(LLMBudgetExceeded):
        GeminiREST(meter=meter).complete("", [Msg("user", "x")])


def test_gemini_rest_4xx_not_retried(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    n = {"c": 0}

    def boom(req, timeout=None):
        n["c"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error": {"message": "nope"}}'))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    from pnr_node.llm import LLMError

    with pytest.raises(LLMError, match="HTTP 400.*nope"):
        GeminiREST().complete("", [Msg("user", "x")])
    assert n["c"] == 1


def test_pick_newest_pro():
    gm = ["generateContent"]
    ms = [{"name": "models/gemini-2.5-pro", "supportedGenerationMethods": gm},
          {"name": "models/gemini-3.1-pro-preview", "supportedGenerationMethods": gm},
          {"name": "models/gemini-3.1-pro-preview-customtools", "supportedGenerationMethods": gm},
          {"name": "models/gemini-3-pro-image-preview", "supportedGenerationMethods": gm},
          {"name": "models/gemini-9-pro-tts", "supportedGenerationMethods": gm},
          {"name": "models/gemini-9-pro", "supportedGenerationMethods": ["embedContent"]},
          {"name": "models/gemini-3.1-flash", "supportedGenerationMethods": gm}]
    assert pick_newest_pro(ms) == "gemini-3.1-pro-preview"


# ---------------------------------------------------------------- driver
def _run_arms():
    spec = importlib.util.spec_from_file_location("run_arms", os.path.join(REPO, "scripts", "arms", "run_arms.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_env_file(tmp_path):
    f = tmp_path / "g.env"
    f.write_text("# c\nexport GEMINI_API_KEY='abc'\nOTHER=1\nKEEP=new\n\nbad line\n")
    env = {"KEEP": "old"}
    names = _run_arms().load_env_file(str(f), env)
    assert names == ["GEMINI_API_KEY", "OTHER"] and env == {"GEMINI_API_KEY": "abc", "OTHER": "1", "KEEP": "old"}


def test_run_arms_dry_run(tmp_path, capsys):
    m = _run_arms()
    rc = m.main(["--dry-run", "--design", "ariane133", "--arms", "default,random,tpe,llm", "--reps", "2",
                 "--budget-evals", "8", "--batch", "3", "--llm-provider", "fake", "--db", str(tmp_path / "x.sqlite"),
                 "--out", str(tmp_path / "x.json"), "--deadline-hours", "1"])
    assert rc == 0
    res = json.load(open(tmp_path / "x.json"))
    assert res["dry_run"] and res["llm"]["model"] == "fake" and set(res["comparison"]["arms"]) == {
        "default", "random", "tpe", "llm"}
    assert "| llm | 2 |" in capsys.readouterr().out


def test_rejected_eval_scores_inf_counts_and_search_continues(fake_arm):
    from pnr_node.arm_eval import ArmRunEvaluator

    class Mixed:
        n = 0

        def propose(self, history, n, stage="cts"):
            Mixed.n += 1
            return [{"target_density": 0.6, "rebuf_stops": "0.6"}, {"target_density": 0.5}][:n]

    db = ResultsDB()
    ev = ArmRunEvaluator(workers=2).for_arm("t")
    r = ev.evaluate(EvalRequest("gcd", stage="cts", flow_cfg={"target_density": 0.6, "rebuf_stops": "0.6"}))
    assert not r.ok and "not_converged" in r.error and score_timing(r) == float("inf")
    out = run_search_v2(ev, Mixed(), db, "t", "gcd", budget_evals=6, batch=2, promote_top=1)
    assert out["evals_used"] == 6 and out["rounds"] == 2 and out["failed_configs"] >= 1
    # round 2 re-proposes both round-1 configs; already-seen configs are replaced by random ones, not re-evaluated
    assert out["configs"] == 4 and out["best_n_seeds"] == 2 and out["best_cfg"]["rebuf_stops"] != "0.6"


def test_incumbent_prior_is_free_history_and_never_reevaluated():
    from pnr_node.experiment import default_flow_cfg, incumbent_prior, run_arm_v2
    from pnr_node.mock_eval import FlowMockEvaluator

    seen = []

    class Echo:  # proposes the default first, then whatever
        def propose(self, history, n, stage="cts"):
            seen.append([h.get("is_default") for h in history])
            return [default_flow_cfg(), {"target_density": 0.55}][:n]

    db, ev = ResultsDB(), FlowMockEvaluator()
    run_arm_v2(ev, db, "default", 0, "gcd", budget_evals=4, batch=2, default_seeds=(0, 1, 2))
    prior = incumbent_prior(db, "gcd", "cts", (0, 1))
    assert len(prior) == 1 and prior[0]["n_seeds"] == 2 and prior[0]["score"] is not None
    out = run_search_v2(ev, Echo(), db, "s", "gcd", budget_evals=4, batch=2, promote_top=1, prior=prior)
    assert seen[0] == [True] and out["incumbent_score"] == prior[0]["score"]
    cfgs = [c["cfg"] for c in config_stats(db, "s", "gcd", "cts")]
    assert default_flow_cfg() not in cfgs  # the incumbent was replaced, never spent budget on


def test_orfs_var_args_change_identity_and_reach_run_arm():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "arms"))
    from run_arms import orfs_var_args
    from pnr_node.arm_eval import ArmRunEvaluator
    plain = ArmRunEvaluator()
    flat = ArmRunEvaluator(extra_args=orfs_var_args(["OPENROAD_HIERARCHICAL=0", "SWAP_ARITH_OPERATORS=0"]))
    # no overrides: the identity is exactly the pre-existing one (existing ResultsDBs stay valid)
    assert plain.tool_id == ArmRunEvaluator(extra_args=()).tool_id
    assert flat.tool_id != plain.tool_id
    assert flat.for_arm("x").tool_id == flat.tool_id
    req = EvalRequest("swerv_wrapper", stage="cts", seed=3, flow_cfg={"target_density": 0.45})
    cmd = flat.for_arm("random").command(req)
    assert cmd[-4:] == ["--orfs-var", "OPENROAD_HIERARCHICAL=0", "--orfs-var", "SWAP_ARITH_OPERATORS=0"]
    with pytest.raises(SystemExit):
        orfs_var_args(["OPENROAD_HIERARCHICAL"])


class _ScoreByCfg:
    """Evaluator whose routed TNS is set per flow config (td -> tns); wns 0. None -> failed run."""

    def __init__(self, tns_by_td):
        self.tns_by_td, self.tool_id, self.calls = tns_by_td, "t", []

    def evaluate(self, req):
        self.calls.append(req)
        t = self.tns_by_td[req.flow_cfg["target_density"]]
        if t is None:
            return EvalResult(False, req.stage, error="boom")
        return EvalResult(True, req.stage, wns=0.0, tns=t + 0.1 * req.seed, routed_wl=1.0)

    def evaluate_many(self, reqs):
        return [self.evaluate(r) for r in reqs]


def _seed_search(db, design, arm, td, scores):
    for seed, sc in zip((3, 4, 5), scores):
        q = EvalRequest(design, "nangate45", default_config(), seed, "cts", "t",
                        flow_cfg=clamp_config({"target_density": td}, FLOW_KNOBS))
        db.record_eval(arm, 1, q, EvalResult(True, "cts", wns=0.0, tns=-sc), score_fn=score_timing)


def test_pool_candidates_dedupes_across_arms_and_skips_default_and_validation():
    from pnr_node.experiment import pool_candidates

    db = ResultsDB()
    _seed_search(db, "d", "v2_random#r0", 0.6, (10, 10, 10))
    _seed_search(db, "d", "v2_llm#r1", 0.6, (10, 10, 10))       # same config found by two arm-runs
    _seed_search(db, "d", "v2_llm#r0", 0.5, (20, 20, 20))
    _seed_search(db, "d", "v2_llm#r0", 0.55, (5,))              # single seed: not a finalist
    _seed_search(db, "d", "v2_default#r0", 0.45, (1, 1, 1))     # the default is always in the final round anyway
    _seed_search(db, "d", "v2_llm#r0|validate", 0.7, (0, 0, 0))
    _seed_search(db, "d", "import", 0.71, (0, 0, 0))
    rows = pool_candidates(db, "d", "cts", top=5)
    assert [r["cfg"]["target_density"] for r in rows] == [0.6, 0.5]
    assert rows[0]["found_by"] == ["v2_llm#r1", "v2_random#r0"] and rows[0]["search_score"] == 10
    assert len(pool_candidates(db, "d", "cts", top=1)) == 1


def test_route_select_keeps_default_unless_a_finalist_routes_better():
    from pnr_node.experiment import route_select_v2

    dtd = clamp_config({}, FLOW_KNOBS)["target_density"]
    cands = [{"cfg": clamp_config({"target_density": td}, FLOW_KNOBS)} for td in (0.6, 0.5, 0.55)]
    # the search-stage favourite (0.6) routes worse than the default; 0.5 routes better; 0.55 fails a seed
    ev = _ScoreByCfg({dtd: -50.0, 0.6: -80.0, 0.5: -40.0, 0.55: None})
    db = ResultsDB()
    s = route_select_v2(ev, db, "d", cands, seeds=(8, 9))
    assert len(ev.calls) == 8 and {q.seed for q in ev.calls} == {8, 9} and {q.stage for q in ev.calls} == {"route"}
    assert not s["winner_is_default"] and s["winner_cfg"]["target_density"] == 0.5
    assert s["rows"][0]["is_default"] and s["rows"][3]["complete"] is False and s["rows"][3]["routed_mean"] is None
    assert len(db.evals("v2_select#r0", "d", "route")) == 8
    # a gain inside min_gain is not enough: keep the default
    s2 = route_select_v2(_ScoreByCfg({dtd: -50.0, 0.6: -80.0, 0.5: -40.0, 0.55: None}), ResultsDB(), "d", cands,
                         seeds=(8, 9), min_gain=15.0)
    assert s2["winner_is_default"] and s2["winner_cfg"] == clamp_config({}, FLOW_KNOBS)
    # every finalist worse: the default
    s3 = route_select_v2(_ScoreByCfg({dtd: -50.0, 0.6: -80.0, 0.5: -60.0, 0.55: -70.0}), ResultsDB(), "d", cands,
                         seeds=(8, 9))
    assert s3["winner_is_default"]


def test_final_round_end_to_end_on_mock(tmp_path):
    import subprocess
    import sys

    from pnr_node.experiment import run_experiment_v2, run_final_round_v2, summarize_final_round

    db = ResultsDB(str(tmp_path / "x.sqlite"))
    run_experiment_v2(FlowMockEvaluator(fail_rate=0.0), db, "ariane133", arms=("default", "random", "tpe"), reps=2,
                      budget_evals=8, batch=3, seeds=(3,), promote_seeds=(4, 5), default_seeds=(3, 4, 5),
                      validate=False, workers=2, log=lambda m: None)
    res = run_final_round_v2(FlowMockEvaluator(fail_rate=0.0), db, "ariane133", top=3, select_seeds=(8, 9),
                             validate_seeds=(101, 102))
    sel = res["selection"]
    assert len(res["candidates"]) == 3 and len(sel["rows"]) == 4 and sel["rows"][0]["is_default"]
    assert all(set(r["routed_scores"]) == {"8", "9"} for r in sel["rows"])
    assert set(res["validation"]) == {"v2_default#r0", "v2_select#r0"}
    assert set(res["comparison"]["arms"]) == {"default", "select"}
    json.dumps(res)
    assert "pick:" in summarize_final_round(res)
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "arms", "final_round.py")
    p = subprocess.run([sys.executable, script, "--design", "ariane133", "--db", str(tmp_path / "x.sqlite"),
                        "--out", str(tmp_path / "f.json"), "--dry-run", "--select-seeds", "8,101"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "must not overlap" in p.stderr
    p = subprocess.run([sys.executable, script, "--design", "ariane133", "--db", str(tmp_path / "x.sqlite"),
                        "--out", str(tmp_path / "f.json"), "--dry-run"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.load(open(tmp_path / "f.json"))["final_round"] is True


def test_final_round_records_each_route_as_it_finishes():
    """A round interrupted mid-way keeps the routes that finished (the re-run replays them from the DB)."""
    from pnr_node.experiment import route_select_v2

    dtd = clamp_config({}, FLOW_KNOBS)["target_density"]
    cands = [{"cfg": clamp_config({"target_density": 0.6}, FLOW_KNOBS)}]

    class Dies(_ScoreByCfg):
        def evaluate(self, req):
            if req.flow_cfg["target_density"] == 0.6 and req.seed == 9:
                time.sleep(0.2)
                raise KeyboardInterrupt
            return super().evaluate(req)

    db = ResultsDB()
    with pytest.raises(KeyboardInterrupt):
        route_select_v2(Dies({dtd: -50.0, 0.6: -40.0}), db, "d", cands, seeds=(8, 9))
    assert len(db.evals("v2_select#r0", "d", "route")) == 3
    ev = _ScoreByCfg({dtd: -50.0, 0.6: -40.0})
    s = route_select_v2(CachedEvaluator(ev, db), db, "d", cands, seeds=(8, 9))
    assert len(ev.calls) == 1 and not s["winner_is_default"]


def test_auto_min_gain_from_default_seed_lottery():
    import math
    import statistics

    from pnr_node.experiment import auto_min_gain

    db = ResultsDB()
    assert auto_min_gain(db, "d", "cts", 2) == 0.0  # no default runs yet: any gain
    _seed_search(db, "d", "v2_default#r0", clamp_config({}, FLOW_KNOBS)["target_density"], (40, 50, 60))
    sd = statistics.stdev([40, 50, 60])
    assert auto_min_gain(db, "d", "cts", 2) == pytest.approx(2 * sd)
    assert auto_min_gain(db, "d", "cts", 8) == pytest.approx(2 * sd * math.sqrt(2 / 8))
