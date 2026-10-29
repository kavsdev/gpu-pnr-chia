"""v2 arms interface: FLOW_KNOBS, flow_cfg identity, score_timing/passes, flow_kwargs, stage cts."""
import json
import os
import subprocess

import pytest

from pnr_node.contract import (FLOW_KNOBS, STAGES, EvalRequest, EvalResult, Knob, clamp_config, default_config,
                               passes, score, score_timing)


# ---- contract -------------------------------------------------------------------------------------------------
def test_existing_keys_unchanged():
    """Hard-coded keys computed before the v2 change (8031d339967c61c4 = a real cached ariane133 final-flow route run)."""
    assert EvalRequest("gcd").key() == "0a1bc1a9d96dd768"
    assert EvalRequest("ariane133", seed=1, stage="route", tool_id="099c8374ba70").key() == "8031d339967c61c4"
    assert EvalRequest("tinyRocket", placer_cfg={"gamma": 6.0, "target_density": 0.7}, seed=2, stage="grt",
                       tool_id="abc").key() == "c0783e9653b64557"
    assert EvalRequest("gcd", stage="proxy", constraints_id="c1").key() == "15a9dfa88230bc52"
    # empty / all-default flow_cfg is the default flow: same key
    assert EvalRequest("gcd", flow_cfg={}).key() == "0a1bc1a9d96dd768"
    assert EvalRequest("gcd", flow_cfg=default_config(FLOW_KNOBS)).key() == "0a1bc1a9d96dd768"
    assert EvalRequest("gcd", flow_cfg={"target_density": 0.45}).key() == "0a1bc1a9d96dd768"


def test_flow_cfg_in_key_and_clamped():
    a = EvalRequest("gcd", flow_cfg={"target_density": 0.6})
    assert a.key() != EvalRequest("gcd").key()
    assert a.flow_cfg["target_density"] == 0.6 and a.flow_cfg["rebuf_stops"] == "0.6,0.2"  # filled with defaults
    assert EvalRequest("gcd", flow_cfg={"target_density": 0.99, "bogus": 1}).flow_cfg["target_density"] == 0.80
    assert "bogus" not in EvalRequest("gcd", flow_cfg={"bogus": 1}).flow_cfg
    assert a.key() == EvalRequest("gcd", flow_cfg=dict(a.flow_cfg)).key()


def test_stages_and_flow_knobs():
    assert STAGES == ("proxy", "grt", "route", "cts")
    EvalRequest("gcd", stage="cts")
    d = default_config(FLOW_KNOBS)
    assert d == {"target_density": 0.45, "rebuf_stops": "0.6,0.2", "netwt_pct": 10.0, "netwt_max": 5.0,
                 "rebuf_full": True, "dp_routability": False, "density_weight": 8e-5, "gamma": 4.0,
                 "learning_rate": 0.01}
    assert FLOW_KNOBS["density_weight"].log and FLOW_KNOBS["learning_rate"].log


def test_categorical_clamp():
    rs, rf = FLOW_KNOBS["rebuf_stops"], FLOW_KNOBS["rebuf_full"]
    assert rs.clamp("0.6,0.4,0.2") == "0.6,0.4,0.2"
    assert rs.clamp("0.6, 0.2") == "0.6,0.2"
    assert rs.clamp("0.62,0.22") == "0.6,0.2" and rs.clamp("0.45") == "0.4" and rs.clamp(0.6) == "0.6"
    assert rs.clamp("nonsense") == "0.6,0.2"  # unparseable -> default
    assert rf.clamp(True) is True and rf.clamp(False) is False and rf.clamp("false") is False and rf.clamp(0) is False
    k = Knob("x", 0, 0, 1.0, choices=(0.5, 1.0, 2.0))
    assert k.clamp(1.6) == 2.0 and k.clamp("0.4") == 0.5
    assert clamp_config({"rebuf_full": "true"}, FLOW_KNOBS)["rebuf_full"] is True
    assert FLOW_KNOBS["gamma"].clamp(20) == 8.0  # numeric knobs unchanged


def test_score_timing_and_passes():
    r = EvalResult(True, "cts", wns=-0.2, tns=-50.0)
    assert score_timing(r) == pytest.approx(50.0 + 20.0)
    assert score_timing(r, {"wns": -1, "tns": -1, "wl": 1}) == score_timing(r)
    assert score_timing(EvalResult(False, "cts", wns=-0.1, tns=-1)) == float("inf")
    assert score_timing(EvalResult(True, "cts")) == float("inf")
    assert score_timing(EvalResult(True, "route", wns=-0.1, tns=-1, drc_violations=3)) == float("inf")
    assert score_timing(EvalResult(True, "route", wns=-0.1, tns=-1, drc_violations=0)) == pytest.approx(11.0)
    assert score_timing(EvalResult(True, "cts", wns=-0.3, tns=-50)) > score_timing(r)  # WNS tie-break
    ref = {"wns": -0.25, "tns": -60.0, "wl": 1000.0}
    assert passes(r, ref)
    assert not passes(EvalResult(True, "cts", wns=-0.3, tns=-40.0), ref)
    assert not passes(EvalResult(True, "cts", wns=-0.2, tns=-61.0), ref)
    assert passes(EvalResult(True, "grt", wns=-0.2, tns=-50, grt_wl=1030.0), ref)
    assert not passes(EvalResult(True, "grt", wns=-0.2, tns=-50, grt_wl=1031.0), ref)
    assert passes(EvalResult(True, "cts", wns=-0.2, tns=-50), {"wns": -0.25, "tns": -60.0, "wl": None})
    assert not passes(EvalResult(False, "cts", wns=0, tns=0), ref)
    # the old objective is untouched
    assert score(EvalResult(True, "route", routed_wl=100.0, wns=-0.1, drc_violations=0)) == pytest.approx(200.0)


# ---- raw_handoff -----------------------------------------------------------------------------------------------
_FINAL_ARGS = ["--rebuffer", "--rebuf-full", "--rebuf-overflow", "0.6,0.2", "--timing-driven-netwt", "static",
               "--netwt-scheme", "gpl", "--rebuf-presplit", "--dp-extra", '{"target_density":0.45}']


def _env(tmp_path, monkeypatch):
    """A fake install/python reproducing the environment of the cached final-flow runs (arm_*_final_d045_*)."""
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    monkeypatch.delenv("PNR_IMAGE_DIGEST", raising=False)
    dp = tmp_path / "dp"
    dp.mkdir()
    (dp / "COMMIT").write_text("6627f3327e6cc17db7782c0b90073a498531ca3c+tdgate100+mnw1024+hsfix3\n")
    (dp / "CUDA_ARCH").write_text("8.6\n")
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\necho 2.9.1+cu129\n")
    py.chmod(0o755)
    return ["--native", "--work", str(tmp_path / "w"), "--dp-install", str(dp), "--python", str(py)]


def test_default_flow_cfg_equals_final_flow_cli(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import from_args
    common = _env(tmp_path, monkeypatch)
    ev_cli, req_cli = from_args(common + _FINAL_ARGS + ["--design", "ariane133", "--seed", "1"])
    ev_fc, req_fc = from_args(common + ["--flow-cfg", "{}", "--design", "ariane133", "--seed", "1"])
    ev_full, req_full = from_args(common + ["--flow-cfg", json.dumps(default_config(FLOW_KNOBS)),
                                            "--design", "ariane133", "--seed", "1"])
    attrs = lambda e: {k: v for k, v in vars(e).items() if k not in ("flow_cfg", "_toolchain")}
    assert attrs(ev_cli) == attrs(ev_fc) == attrs(ev_full)
    assert ev_cli.toolchain == ev_fc.toolchain == ev_full.toolchain
    # the real cached ariane133 final-flow route run (logs/arm_ariane133_final_d045_s1_idle.log)
    assert ev_fc.tool_id == ev_full.tool_id == "099c8374ba70"
    assert req_cli.key() == req_fc.key() == req_full.key() == "8031d339967c61c4"


def test_flow_kwargs_mapping():
    from pnr_node.raw_handoff import FINAL_FLOW_KWARGS, flow_kwargs
    kw, dp = flow_kwargs({})
    assert kw == FINAL_FLOW_KWARGS and kw is not FINAL_FLOW_KWARGS
    assert dp == {"density_weight": 8e-5, "gamma": 4.0, "learning_rate": 0.01}
    kw, dp = flow_kwargs({"target_density": 0.6, "rebuf_stops": "0.6", "netwt_pct": 20, "netwt_max": 3,
                          "rebuf_full": False, "dp_routability": True, "gamma": 6})
    # dp_routability is pinned to False (contract.py): a True request clamps to False
    assert kw["dp_extra"] == {"target_density": 0.6} and kw["rebuf_overflow"] == 0.6
    assert kw["netwt_pct"] == 20.0 and kw["netwt_gpl_max"] == 3.0 and kw["rebuf_full"] is False
    assert kw["dp_routability"] is False and dp["gamma"] == 6.0
    assert FINAL_FLOW_KWARGS["dp_extra"] == {"target_density": 0.45}  # not mutated
    assert flow_kwargs({"rebuf_stops": "0.6,0.4,0.2"})[0]["rebuf_overflow"] == [0.6, 0.4, 0.2]


def test_distinct_flows_distinct_identity(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import from_args
    common = _env(tmp_path, monkeypatch)
    ids, keys = set(), set()
    for fc in ({}, {"target_density": 0.6}, {"rebuf_stops": "0.6"}, {"netwt_pct": 20}, {"netwt_max": 3},
               {"rebuf_full": False}, {"rebuf_stops": "0.7,0.3"}):
        ev, req = from_args(common + ["--flow-cfg", json.dumps(fc), "--stage", "cts"])
        ids.add(ev.tool_id)
        keys.add(req.key())
    assert len(ids) == 7 and len(keys) == 7
    # DREAMPlace knobs: same tool_id as the default flow, but placer_cfg (and the key) differ
    ev0, r0 = from_args(common + ["--flow-cfg", "{}", "--stage", "cts"])
    ev1, r1 = from_args(common + ["--flow-cfg", '{"gamma": 6, "learning_rate": 0.02, "density_weight": 1e-4}',
                                  "--stage", "cts"])
    assert ev0.tool_id == ev1.tool_id and r0.key() != r1.key()
    assert r1.placer_cfg["gamma"] == 6.0 and r1.placer_cfg["learning_rate"] == 0.02
    assert r1.placer_cfg["density_weight"] == 1e-4


def test_flow_cfg_excludes_individual_flags(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import from_args
    common = _env(tmp_path, monkeypatch)
    for extra in (["--rebuffer"], ["--dp-extra", '{"a":1}'], ["--netwt-pct", "20"], ["--cfg", '{"gamma":5}']):
        with pytest.raises(SystemExit):
            from_args(common + ["--flow-cfg", "{}"] + extra)
    with pytest.raises(SystemExit):
        from_args(common + ["--flow-cfg", "[1]"])
    from_args(common + ["--flow-cfg", "{}", "--orfs-var", "X=1", "--stage", "cts"])  # non-flow flags are fine


def test_evaluate_applies_flow_and_rejects_other_flow(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import RawHandoffEvaluator, apply_flow
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    ev = RawHandoffEvaluator.for_flow({"gamma": 6}, str(tmp_path / "w"), str(tmp_path), native=True)
    seen = {}
    monkeypatch.setattr(ev, "prepare", lambda d: str(tmp_path / "base"))

    def fake_place(req, base):
        seen["cfg"] = dict(req.placer_cfg)
        from pnr_node.dreamplace_runner import DPResult
        return DPResult(False, error="stop")
    monkeypatch.setattr(ev, "_place", fake_place)
    req = EvalRequest("gcd", stage="cts", flow_cfg={"gamma": 6})
    r = ev.evaluate(req)
    assert seen["cfg"]["gamma"] == 6.0 and r.provenance["req_key"] == apply_flow(req).key()
    assert apply_flow(apply_flow(req)) == apply_flow(req)
    r = ev.evaluate(EvalRequest("gcd", stage="cts", flow_cfg={"gamma": 5}))
    assert not r.ok and "flow" in r.error and "req_key" in r.provenance


def test_stage_cts_target_and_metrics(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    ev = RawHandoffEvaluator(str(tmp_path / "w"), str(tmp_path), native=True)
    base = tmp_path / "base"
    (base / "results" / "nangate45" / "gcd" / "base").mkdir(parents=True)
    (base / "dp").mkdir()
    seen = {}

    def fake_make(work, design, target="", timeout=0):
        seen["target"] = target
        logs = os.path.join(work, "logs", "nangate45", design, "base")
        os.makedirs(logs, exist_ok=True)
        json.dump({"cts__timing__setup__ws": -0.12, "cts__timing__setup__tns": -3.5},
                  open(os.path.join(logs, "4_1_cts.json"), "w"))
        open(os.path.join(logs, "4_1_cts.log"), "w").write("x\nElapsed time: 1:02.50[h:]min:sec. CPU\n")
        open(os.path.join(logs, "3_3_place_gp.log"), "w").write("read_def -incremental\nElapsed time: 0:03.00[h:]min:sec.\n")
        open(os.path.join(logs, "2_1_floorplan.log"), "w").write("Elapsed time: 0:09.00[h:]min:sec.\n")
        return subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(ev, "_patch_scripts", lambda w: None)
    monkeypatch.setattr(ev, "_make", fake_make)
    from pnr_node import raw_handoff
    monkeypatch.setattr(raw_handoff, "components_only_def", lambda a, b: open(b, "w").write("c") and 1)
    req = EvalRequest("gcd", stage="cts")
    res = ev.route_from_def(req, str(base), str(tmp_path / "run"), "gp.def", 1.0, {"dp_runtime_s": 2.0}, 0.0)
    assert seen["target"].endswith("results/nangate45/gcd/base/4_1_cts.odb")
    assert res.ok and res.stage == "cts" and res.wns == -0.12 and res.tns == -3.5
    assert res.grt_wl is None and res.routed_wl is None
    assert res.provenance["stage_wall_s"] == {"3_3_place_gp": 3.0, "4_1_cts": 62.5}
    assert res.provenance["handoff_read_def_in_3_3"] is True and res.provenance["dp_runtime_s"] == 2.0
    d = json.loads(res.to_json())
    assert {"stage", "wns", "tns", "provenance"} <= set(d)
