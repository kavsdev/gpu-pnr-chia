import json
import os
import pytest
from pnr_node.evidence import atomic_write_json
from pnr_node.db import ResultsDB
from pnr_node.mock_eval import MockEvaluator
from pnr_node.search import LLMProposer, RandomProposer, run_search
from pnr_node.llm import FakeProvider
from pnr_node.contract import CORE_KNOBS, EvalRequest, EvalResult

def test_atomic_write_json_interruption(tmp_path):
    path = tmp_path / "test.json"
    data1 = {"a": 1}
    atomic_write_json(str(path), data1)
    assert path.read_text().strip() != ""
    
    # Simulate interruption by monkeypatching json.dump to raise
    import json as _json
    original_dump = _json.dump
    def faulty_dump(*args, **kwargs):
        raise RuntimeError("Interrupted!")
    
    import pnr_node.evidence
    original_evidence_dump = pnr_node.evidence.json.dump
    pnr_node.evidence.json.dump = faulty_dump
    
    try:
        atomic_write_json(str(path), {"a": 2})
    except RuntimeError:
        pass
    finally:
        pnr_node.evidence.json.dump = original_evidence_dump
        
    # The old file should still be intact and valid
    with open(path) as f:
        assert json.load(f) == data1

def test_experiment_atomic_write(tmp_path, monkeypatch):
    import runpy
    import sys
    import pnr_node.evidence

    called = []
    def mock_atomic(path, obj):
        called.append(path)

    monkeypatch.setattr(pnr_node.evidence, "atomic_write_json", mock_atomic)

    out_path = str(tmp_path / "exp.json")
    monkeypatch.setattr(sys, "argv", ["experiment", "--mock", "--db", str(tmp_path / "exp.db"), "--out", out_path,
                                      "--arms", "default_seeds", "--repeats", "1", "--rounds", "1", "--batch", "1",
                                      "--route-top", "1"])
    runpy.run_module("pnr_node.experiment", run_name="__main__")

    assert called == [out_path]

def test_study_atomic_writes(tmp_path, monkeypatch):
    from pnr_node import study
    import pnr_node.evidence
    
    called = set()
    def mock_atomic(path, obj):
        called.add(os.path.basename(path))
    
    monkeypatch.setattr(pnr_node.evidence, "atomic_write_json", mock_atomic)
    
    # Mock dependencies to make _cmd_place runnable
    class MockEv:
        def __init__(self, *args, **kwargs):
            self.tool_id = "mock"
        def evaluate(self, req):
            from pnr_node.contract import EvalResult
            return EvalResult(True, "proxy", proxy_hpwl=100.0, provenance={"dp_runtime_s": 1.0})
    
    import pnr_node.raw_handoff
    monkeypatch.setattr(pnr_node.raw_handoff, "RawHandoffEvaluator", MockEv)
    
    class Args:
        work = str(tmp_path / "work")
        dp_install = "dp"
        python = "python"
        native = True
        out = str(tmp_path / "out")
        n = 1
        seed = 0
        run_seed = 0
        wide = False
        design = "gcd"
        indir = str(tmp_path / "out")
        noise_sigma_pct = 3.0
    
    # Test _cmd_place
    os.makedirs(Args.out, exist_ok=True)
    study._cmd_place(Args())
    assert "place.json" in called
    
    # Test _cmd_analyze (mocking the input file)
    called.clear()
    with open(os.path.join(Args.indir, "routed.json"), "w") as f:
        json.dump({"design": "gcd", "rows": [{"proxy_hpwl": 100, "routed_wl": 110, "score": 115}]}, f)
    
    study._cmd_analyze(Args())
    assert "analysis.json" in called

def test_pareto_in_history_text():
    db = ResultsDB()
    ev = MockEvaluator()
    design = "gcd"
    arm = "llm_pareto"
    
    # 1. Setup a history where one point is Pareto-optimal but NOT top-k-by-score
    # Row 1: WL=100, WNS=-0.1, Score=low (Winner)
    # Row 2: WL=80, WNS=-0.5, Score=higher (Pareto but not score-best)
    # Row 3: WL=110, WNS=-0.6, Score=highest (Dominated)
    
    def add_eval(wl, wns, drc=0):
        cfg = {"target_density": 0.5 + wl/1000.0}
        req = EvalRequest(design, placer_cfg=cfg, stage="route")
        res = EvalResult(True, "route", routed_wl=wl, wns=wns, drc_violations=drc)
        db.record_eval(arm, 1, req, res)
        return {"cfg": cfg, "result": res, "score": wl * (1 + abs(wns)), "stage": "route", "design": design}

    h1 = add_eval(100.0, -0.1)
    h2 = add_eval(80.0, -0.5)
    h3 = add_eval(120.0, -0.6)
    
    history = [h1, h2, h3]
    
    # top_k=1 ensures only h1 is in the "top" section
    prop = LLMProposer(FakeProvider(), db, arm=arm, show_routed=True, top_k=1)
    text = prop._history_text(history)
    
    assert "Pareto-optimal" in text
    assert str(80.0) in text # h2 should be there as Pareto
    assert str(120.0) not in text # h3 is dominated and not top-1

def test_show_routed_false_no_pareto():
    db = ResultsDB()
    arm = "llm_proxy"
    design = "gcd"
    
    # Add a routed result
    req = EvalRequest(design, stage="route")
    res = EvalResult(True, "route", routed_wl=100.0, wns=-0.1)
    db.record_eval(arm, 1, req, res)
    
    history = [{"cfg": req.placer_cfg, "result": res, "score": 110.0, "stage": "route"}]
    
    prop = LLMProposer(FakeProvider(), db, arm=arm, show_routed=False)
    text = prop._history_text(history)
    
    assert "Pareto-optimal" not in text
    assert "routed_wl" not in text
