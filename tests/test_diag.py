"""llm_diag arm: evidence packer (pnr_node/diag.py) and DiagProposer (pnr_node/diag_proposer.py)."""
import json
import os

import pytest

from pnr_node import diag
from pnr_node.contract import FLOW_KNOBS, EvalResult, clamp_config, default_config
from pnr_node.diag_proposer import DiagProposer, _SYSTEM_DIAG, cfg_key, make_factory, register_llm_diag
from pnr_node.llm import make_provider
from pnr_node.search import RandomProposer

NETWT = os.path.expanduser("~/work/netwt")
ARIANE = os.path.join(NETWT, "ariane133", "8031d339967c61c4")
ARIANE_DP = os.path.join(NETWT, "ariane133", "dp-2f7a0ce158168946")


# ---------------------------------------------------------------------------------------------------------------------
# synthetic run dir (trimmed copies of real gcd cts / ariane grt+route files)
# ---------------------------------------------------------------------------------------------------------------------

_PATH_RPT = """==========================================================================
cts final report_clock_skew
--------------------------------------------------------------------------
Clock core_clock
   0.07 source latency a/CK ^
  -0.07 target latency b/CK ^
   0.00 CRPR
--------------
   0.02 setup skew

==========================================================================
cts final report_checks -path_delay max
--------------------------------------------------------------------------
Startpoint: dpath.a_reg.out[5]$_DFFE_PP_
            (rising edge-triggered flip-flop clocked by core_clock)
Endpoint: dpath.a_reg.out[7]$_DFFE_PP_
          (rising edge-triggered flip-flop clocked by core_clock)
Path Group: core_clock
Path Type: max

Fanout     Cap    Slew   Delay    Time   Description
-----------------------------------------------------------------------------
                          0.00    0.00   clock core_clock (rise edge)
     1    4.13    0.00    0.00    0.00 ^ clk (in)
                                         clk (net)
                  0.00    0.00    0.00 ^ clkbuf_0_clk/A (CLKBUF_X3)
     4    9.88    0.01    0.03    0.03 ^ clkbuf_0_clk/Z (CLKBUF_X3)
                                         clknet_0_clk (net)
                  0.01    0.00    0.07 ^ dpath.a_reg.out[5]$_DFFE_PP_/CK (DFF_X1)
     4    9.58    0.02    0.11    0.18 ^ dpath.a_reg.out[5]$_DFFE_PP_/Q (DFF_X1)
                                         n1 (net)
                  0.02    0.02    0.20 ^ _530_/A (INV_X2)
     2    6.84    0.01    0.01    0.21 v _530_/ZN (INV_X2)
                                         n2 (net)
                  0.03    0.04    0.25 v place169/A (BUF_X4)
     7   16.41    0.01    0.03    0.28 ^ place169/Z (BUF_X4)
                                         n3 (net)
                  0.01    0.00    0.28 ^ dpath.a_reg.out[7]$_DFFE_PP_/D (DFF_X1)
                                  0.28   data arrival time

                          0.25    0.25   clock core_clock (rise edge)
                                  0.13   data required time
-----------------------------------------------------------------------------
                                  0.13   data required time
                                 -0.28   data arrival time
-----------------------------------------------------------------------------
                                 -0.15   slack (VIOLATED)


Startpoint: req_msg[2] (input port clocked by vclk_core_clock)
Endpoint: resp_msg[14] (output port clocked by vclk_core_clock)
Path Group: vclk_core_clock
Path Type: max

Fanout     Cap    Slew   Delay    Time   Description
-----------------------------------------------------------------------------
     1    4.13    0.00    0.00    0.00 ^ req_msg[2] (in)
                                         req_msg[2] (net)
                  0.00    0.00    0.00 ^ _1_/A (INV_X1)
     1    1.00    0.01    0.02    0.02 v _1_/ZN (INV_X1)
                                  0.02   data arrival time
-----------------------------------------------------------------------------
                                  0.09   slack (MET)

==========================================================================
cts final report_checks -path_delay min
--------------------------------------------------------------------------
Startpoint: x
Endpoint: y
Path Group: core_clock
Path Type: min
                                 -9.00   slack (VIOLATED)
"""

_CTS_LOG = """[INFO RSZ-0047] Found 1 long wires.
[INFO RSZ-0048] Inserted 1 buffers in 1 nets.
repair_timing -setup_margin 0
[INFO RSZ-0094] Found 52 endpoints with setup violations.
   Iter   | Removed | Resized | Inserted | Cloned |  Pin  |   Area   |    WNS   |   StTNS    |   EnTNS    |  Viol  |  Worst
    final |       0 |     100 |        6 |      0 |     3 |    +0.1% |   -0.153 |       -0.4 |       -6.8 |     51 | dpath.a_reg.out\\[9\\]$_DFFE_PP_/D
[INFO RSZ-0045] Inserted 6 buffers, 0 to split loads.
[INFO RSZ-0051] Resized 100 instances: 100 up, 0 up match, 0 down, 0 VT
[WARNING RSZ-0062] Unable to repair all setup violations.
Took 3 seconds: repair_timing -setup_margin 0 -hold_margin 0
Took 1 seconds: clock_tree_synthesis -sink_clustering_enable
Elapsed time: 0:06.48[h:]min:sec. CPU time: user 6.00 sys 0.40 (98%). Peak memory: 100000KB.
"""

_GRT_LOG = """Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Congestion
---------------------------------------------------------------------------------------
metal1               0             0            0.00%             0 /  0 /  0
metal2          851834        487334           57.21%             3 /  9 /  12
---------------------------------------------------------------------------------------
Total          6542763       2206497           33.10%             3 /  9 /  12

[INFO GRT-0101] Running extra iterations to remove congestion.
[INFO GRT-0102] Start extra iteration 1/30
[INFO GRT-0102] Start extra iteration 2/30
[INFO GRT-0096] Final congestion report:
Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Congestion
---------------------------------------------------------------------------------------
metal1               0             0            0.00%             0 /  0 /  0
metal2          851834        487334           57.21%             0 /  0 /  0
metal3         1234978        645629           52.28%             0 /  0 /  0
---------------------------------------------------------------------------------------
Total          6542763       2206497           33.72%             0 /  0 /  0

[INFO GRT-0018] Total wirelength: 7756899 um
Took 125 seconds: global_route -congestion_report_file x
Elapsed time: 20:45.52[h:]min:sec. CPU time: user 4076.91 sys 175.03 (341%). Peak memory: 3044512KB.
"""


def _result(**over):
    d = {"ok": True, "stage": "cts", "wns": -0.154579, "tns": -6.90976, "runtime_s": 92.2, "error": "",
         "provenance": {"dp_iterations": 347, "dp_overflow": 0.0998878, "dp_runtime_s": 59.83908414840698,
                        "req_key": "bf0d81b0b302a7b4", "stage_wall_s": {"3_3_place_gp": 0.69, "4_1_cts": 6.48},
                        "rebuf": {"overflow_stop": [0.6, 0.2], "stages": [
                            {"stage": 1, "gp": {"hpwl": 11220.66, "iterations": 221, "overflow": 0.597},
                             "repair": {"buffers": 1, "resized": 0, "rebuffer_removed_inserted": [[0, 1], [1, 113]]}},
                            {"stage": 2, "gp": {"hpwl": 14101.44, "iterations": 1000, "overflow": 0.62},
                             "sta": {"wns": -0.1519, "tns": -6.574, "viol": 52}, "weights": {"weighted": 68}},
                            {"stage": 3, "gp": {"hpwl": 17519.66, "iterations": 347, "overflow": 0.0999}}]}}}
    d.update(over)
    return d


@pytest.fixture()
def run_dir(tmp_path):
    rd = tmp_path / "gcd" / "bf0d81b0b302a7b4"
    logs = rd / "orfs" / "logs" / "nangate45" / "gcd" / "base"
    rpts = rd / "orfs" / "reports" / "nangate45" / "gcd" / "base"
    logs.mkdir(parents=True)
    rpts.mkdir(parents=True)
    (logs / "3_3_place_gp.json").write_text(json.dumps(
        {"globalplace__timing__setup__ws": -0.148, "globalplace__timing__setup__tns": -6.609}))
    (logs / "4_1_cts.json").write_text(json.dumps(
        {"cts__timing__setup__ws": -0.154579, "cts__timing__setup__tns": -6.90976, "cts__design__instance__count": 720,
         "cts__design__instance__utilization": 0.84585, "cts__design__instance__count__setup_buffer": 6}))
    (logs / "5_1_grt.json").write_text("{not json")  # corrupt: must be skipped, not raise
    (logs / "4_1_cts.log").write_text(_CTS_LOG)
    (logs / "5_1_grt.log").write_text(_GRT_LOG)
    (logs / "5_2_route.json").write_text(json.dumps(
        {"detailedroute__route__drc_errors": 0, "detailedroute__route__drc_errors__iter:0": 581,
         "detailedroute__route__drc_errors__iter:1": 3}))
    (rpts / "4_cts_final.rpt").write_text(_PATH_RPT)
    (rpts / "congestion-5.rpt").write_text("violation type: Vertical congestion\n" * 3)
    dp = tmp_path / "gcd" / "dp-0000000000000001"
    (dp / "rebuf").mkdir(parents=True)
    (dp / "dp_result.json").write_text(json.dumps({"runtime_s": 59.83908414840698}))
    (dp / "rebuf" / "r1.slack").write_text("# wns -1\nold 1.0 900 sig\n")
    (dp / "rebuf" / "r2.slack").write_text(
        "# wns -0.15 tns -6.5 viol 51\nclk 0.1 36 clk\n_294_ -0.038 23 sig\nnet190 -0.037 17 sig\n"
        "big inf 400 hideg\na 0.5 2 sig\n")
    other = tmp_path / "gcd" / "dp-0000000000000002"
    other.mkdir()
    (other / "dp_result.json").write_text(json.dumps({"runtime_s": 1.0}))
    return str(rd)


def test_evidence_synthetic(run_dir):
    ev = diag.evidence(_result(), run_dir)
    assert ev["wns"] == -0.155 and ev["tns"] == -6.91 and ev["dp"]["iterations"] == 347
    assert ev["setup_series"] == [["gp", -0.148, -6.609], ["cts", -0.155, -6.91]]
    assert ev["stage_wall_s"] == {"gp": 1, "cts": 6}  # provenance wins over log Elapsed
    assert [t["k"] for t in ev["rebuf_trace"]] == [1, 2, 3] and ev["rebuf_trace"][1]["it"] == 1000
    rep = ev["orfs_repairs"]["cts"]
    assert rep["drv_bufs"] == 1 and rep["timing_bufs"] == 6 and rep["resized"] == 100
    assert rep["viol_endpoints_in"] == 52 and rep["repair_timing_final"]["viol"] == 51 and rep["setup_unrepaired"]
    paths = ev["worst_paths"]
    assert [p["slack"] for p in paths] == [-0.15, 0.09]  # min-delay path ignored
    p = paths[0]
    assert p["group"] == "core_clock" and p["depth"] == 3 and p["bufs"] == 1 and p["max_fanout"] == 7
    assert p["cell_ns"] == pytest.approx(0.15) and p["wire_ns"] == pytest.approx(0.06)  # clock tree excluded
    assert ev["clock_skew_setup"] == 0.02
    c = ev["grt_congestion"]
    assert c["overflow_initial"] == 12 and c["overflow_final"] == 0 and c["usage_pct"] == 33.72
    assert c["extra_iters"] == "2/30" and c["grt_wl"] == 7756899 and c["hotspots"] == {"iter5": 3}
    assert ev["drt"] == {"drc": 0, "drc_iter0": 581, "iters": 2}
    h = ev["high_fanout_nets"]  # matched dp dir by dp_runtime_s, last dump, clock excluded
    assert h["dump"] == "r2.slack" and h["neg_slack_nets"] == 2 and h["untimed_high_degree"] == 1
    assert h["top"][0].startswith("_294_ fo23")
    assert "slowest_steps" in ev and ev["slowest_steps"][0].startswith("global_route@grt:125s")
    assert "_parse_errors" not in ev
    txt = diag.render(ev)
    assert "worst setup paths" in txt and diag.tokens(txt) < 1500
    assert diag.evidence(_result(), run_dir) == ev  # deterministic
    assert "\n" not in diag.render(ev, brief=True)


def test_evidence_degrades(tmp_path):
    assert diag.evidence({}, None) == {"ok": False, "stage": None}
    assert diag.evidence({"ok": False, "stage": "cts", "error": "rejected: not_converged(overflow=0.21)\nmore"},
                         str(tmp_path / "missing"))["rejected"] == "rejected: not_converged(overflow=0.21)"
    (tmp_path / "orfs" / "logs" / "p" / "d" / "base").mkdir(parents=True)  # empty tree
    ev = diag.evidence(EvalResult(True, "cts", wns=-0.1, tns=-1.0), str(tmp_path))
    assert ev["wns"] == -0.1 and "setup_series" not in ev
    assert diag.evidence("garbage", None)["ok"] is False
    assert diag.render({}) == "(no evidence)"


def test_run_dir_for(tmp_path, run_dir):
    r = _result()
    r["provenance"]["arm_log"] = str(tmp_path / "logs" / "arm_gcd_v2_x_abc.log")
    assert diag.run_dir_for(r) == run_dir
    assert diag.run_dir_for(r, "gcd", str(tmp_path)) == run_dir
    assert diag.run_dir_for({"provenance": {}}) is None


@pytest.mark.skipif(not os.path.isdir(ARIANE), reason="real ariane133 run dir absent")
def test_evidence_real_ariane():
    d = json.load(open(os.path.join(ARIANE, "result.json")))
    ev = diag.evidence(d, ARIANE)
    assert ev["ok"] and ev["stage"] == "route" and ev["drc_violations"] == 0
    stages = [s for s, _, _ in ev["setup_series"]]
    assert stages[-1] == "fin" and "cts" in stages and "grt" in stages
    assert ev["worst_paths"][0]["slack"] < 0 and ev["paths_report"] == "6_finish.rpt"
    assert "usage_pct" in ev["grt_congestion"] and ev["drt"]["drc"] == 0
    assert len(ev["rebuf_trace"]) == 3 and ev["rebuf_trace"][0]["bufs"] > 1000
    if os.path.isdir(ARIANE_DP):
        assert diag.find_dp_dir(ARIANE, d) == ARIANE_DP and ev["high_fanout_nets"]["nets"] > 100000
    assert "_parse_errors" not in ev
    assert diag.tokens(diag.render(ev)) <= 1500


def _gcd_cts_dirs():
    out = []
    for label in ("v2A_cts_s0", "v2A_cts_td06b_s0"):
        log = os.path.join(NETWT, "logs", f"arm_gcd_{label}.log")
        if not os.path.exists(log):
            continue
        from pnr_node.arm_eval import last_json_line
        d = last_json_line(open(log, errors="replace").read())
        if d:
            out.append((d, os.path.join(NETWT, "gcd", d["provenance"]["req_key"])))
    return out


@pytest.mark.skipif(not _gcd_cts_dirs(), reason="real gcd cts runs absent")
def test_evidence_real_gcd_cts():
    for d, rd in _gcd_cts_dirs():
        ev = diag.evidence(d, rd if os.path.isdir(rd) else None)
        assert ev["stage"] == "cts" and ev["wns"] == round(d["wns"], 3)
        assert ev["rebuf_trace"] and ev["dp"]["iterations"] == d["provenance"]["dp_iterations"]
        if os.path.isdir(rd):
            assert ev["setup_series"][-1][0] == "cts" and ev["paths_report"] == "4_cts_final.rpt"
            assert "_parse_errors" not in ev
        assert diag.tokens(diag.render(ev)) <= 1500


# ---------------------------------------------------------------------------------------------------------------------
# DiagProposer
# ---------------------------------------------------------------------------------------------------------------------

def _hist(i, td, score=None, failed=False, ev=None):
    r = EvalResult(not failed, "cts", wns=None if failed else -0.2 - i * 0.01, tns=None if failed else -300.0 - i,
                   error="rejected: not_converged(overflow=0.21)" if failed else "",
                   provenance={"req_key": f"k{i}", "dp_iterations": 300 + i, "dp_overflow": 0.099})
    return {"cfg": clamp_config({"target_density": td}, FLOW_KNOBS), "design": "ariane133", "stage": "cts",
            "seeds": [0], "n_seeds": 1, "seed_scores": [score if score is not None else float("inf")],
            "failed": failed, "score": None if failed else score, "wns": r.wns, "tns": r.tns, "result": r,
            "rounds": [1], **({"evidence": ev} if ev is not None else {})}


def _prov(replies):
    return make_provider({"provider": "fake", "replies": replies})


def test_parse_clamp_dedupe_fill(tmp_path):
    hist = [_hist(0, 0.45, 330.0), _hist(1, 0.6, 340.0)]
    reply = "```json\n" + json.dumps([
        {"cfg": {"target_density": 0.6}, "hypothesis": "dup of history", "expected": "x"},  # dupe -> dropped
        {"cfg": {"target_density": 5.0, "rebuf_stops": "0.6, 0.4, 0.2", "rebuf_full": "false", "bogus": 1},
         "hypothesis": "more repair rounds", "expected": "TNS -5%"},
        {"cfg": {"target_density": 0.8, "rebuf_stops": "0.6,0.4,0.2", "rebuf_full": False},  # same after clamp
         "hypothesis": "dup in batch", "expected": "y"},
        "not a dict",
    ]) + "\n```"
    log = tmp_path / "hyp.jsonl"
    p = DiagProposer(_prov([reply]), design="ariane133", log_path=str(log), fallback=RandomProposer(3, FLOW_KNOBS))
    out = p.propose(hist, 3)
    assert len(out) == 3
    assert out[0]["target_density"] == 0.8 and out[0]["rebuf_stops"] == "0.6,0.4,0.2" and out[0]["rebuf_full"] is False
    assert "bogus" not in out[0] and set(out[0]) == set(FLOW_KNOBS)
    keys = {cfg_key(c) for c in out}
    assert len(keys) == 3 and not keys & {cfg_key(h["cfg"]) for h in hist}
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["source"] for r in rows] == ["llm", "random_fill", "random_fill"]
    assert rows[0]["hypothesis"] == "more repair rounds" and rows[0]["expected"] == "TNS -5%"
    assert rows[0]["dupes_dropped"] == 2 and rows[0]["round"] == 1 and rows[0]["design"] == "ariane133"
    assert p.invalid == 1
    for c in out:
        assert clamp_config(c, FLOW_KNOBS) == c


def test_garbage_and_wrapped_answers():
    p = DiagProposer(_prov(["no json here at all", '{"candidates": [{"target_density": 0.5}]}']))
    out = p.propose([], 2)
    assert len(out) == 2 and p.invalid == 1
    out = p.propose([], 1)  # bare cfg inside a wrapping dict
    assert out[0]["target_density"] == 0.5


def test_prompt_contents_and_hypothesis_feedback():
    ev_best = {"ok": True, "stage": "cts", "wns": -0.18, "tns": -270.0,
               "worst_paths": [{"slack": -0.18, "from": "a", "to": "b", "group": "core_clock", "depth": 80}]}
    hist = [_hist(0, 0.45, 330.0, ev=ev_best), _hist(1, 0.6, 340.0), _hist(2, 0.35, failed=True)]
    fake = _prov([json.dumps([{"cfg": {"target_density": 0.5}, "hypothesis": "H1 density", "expected": "E1"}]),
                  "[]"])
    ref = {"wns": -0.217, "tns": -324.0, "n": 3, "evidence": {"ok": True, "stage": "cts", "wns": -0.2}}
    p = DiagProposer(fake, design="ariane133", ref=ref)
    p.propose(hist, 1)
    system, msgs = fake.calls[0]
    prompt = msgs[-1].content
    assert system == _SYSTEM_DIAG
    for s in ("rebuf_stops", "fully_rebuffer", "DEFAULT FLOW reference", "score=345.70", "[BEST] score=330.00",
              "[WORST/FAILED] score=FAILED", "not_converged", "worst setup paths", '"hypothesis"', "exactly 1 NEW"):
        assert s in prompt, s
    hist2 = hist + [_hist(3, 0.5, 300.0)]
    p.propose(hist2, 1)
    prompt2 = fake.calls[1][1][-1].content
    assert "H1 density" in prompt2 and "actual score: 300.00" in prompt2


def test_prompt_budget_20_items():
    """Whole prompt (system + user) under ~8k tokens with 20 history items of heavy evidence."""
    big = {"ok": True, "stage": "cts", "wns": -0.2, "tns": -300.0,
           "setup_series": [[s, -0.3, -500.0] for s in ("gp", "rsz", "dpl", "cts")],
           "rebuf_trace": [{"k": k, "ovf": 0.6, "hpwl": 29073770, "it": 351, "bufs": 1442, "resized": 139,
                            "sta_wns": -0.767, "sta_tns": -1783, "sta_viol": 3272} for k in (1, 2, 3)],
           "orfs_repairs": {"cts": {"drv_bufs": 1, "timing_bufs": 72, "resized": 293, "viol_endpoints_in": 2547,
                                    "repair_timing_final": {"wns": -0.177, "tns": -268.0, "viol": 2147,
                                                            "worst_ep": "x" * 80}}},
           "worst_paths": [{"slack": -0.2, "from": "s" * 60, "to": "e" * 60, "group": "core_clock", "depth": 84,
                            "cell_ns": 3.0, "wire_ns": 0.1, "max_fanout": 34, "bufs": 20,
                            "worst_cell": "DFFR_X1 0.120ns fo3"}] * 5,
           "grt_congestion": {"usage_pct": 33.7, "overflow_initial": 0}, "stage_wall_s": {"cts": 395, "grt": 1246},
           "high_fanout_nets": {"dump": "r2.slack", "neg_slack_nets": 1, "nets": 2, "top": ["n" * 50] * 3}}
    hist = [_hist(i, 0.35 + 0.02 * i, 300.0 + i, ev=big) for i in range(18)] + \
        [_hist(18, 0.79, failed=True, ev=big), _hist(19, 0.795, failed=True, ev=big)]
    fake = _prov(["[]"])
    p = DiagProposer(fake, design="ariane133", ref={"wns": -0.2, "tns": -300.0, "evidence": big})
    p.hypotheses = [{"cfg": h["cfg"], "hypothesis": "h" * 300, "expected": "e" * 120} for h in hist[:8]]
    p.propose(hist, 4)
    system, msgs = fake.calls[0]
    total = diag.tokens(system) + diag.tokens(msgs[-1].content)
    assert total <= 8000, total
    assert "[BEST]" in msgs[-1].content
    p2 = DiagProposer(_prov(["[]"]), design="ariane133", max_prompt_tokens=3000)  # forced demotion still renders
    assert diag.tokens(_SYSTEM_DIAG) + diag.tokens(p2.build_prompt(hist, 4)) < 4000


def test_evidence_from_run_dirs_and_cache(run_dir, tmp_path):
    r = EvalResult.from_json(json.dumps(_result()))
    h = {"cfg": default_config(FLOW_KNOBS), "design": "gcd", "stage": "cts", "failed": False, "score": 22.4,
         "seed_scores": [22.4], "result": r}
    fake = _prov(["[]"])
    p = DiagProposer(fake, work_root=str(tmp_path))
    p.propose([h], 1)
    assert "worst setup paths (4_cts_final.rpt)" in fake.calls[0][1][-1].content
    assert len(p._ev_cache) == 1


def test_default_ref_from_db():
    from pnr_node.db import ResultsDB
    from pnr_node.contract import EvalRequest, score_timing
    db = ResultsDB(":memory:")
    for s, (w, t) in enumerate([(-0.2, -300.0), (-0.22, -320.0)]):
        q = EvalRequest("ariane133", seed=s, stage="cts", flow_cfg=default_config(FLOW_KNOBS))
        db.record_eval("v2_default#r0", 1, q, EvalResult(True, "cts", wns=w, tns=t), score_fn=score_timing)
    p = DiagProposer(_prov(["[]"]), db=db)
    ref = p._default_ref("ariane133", "cts")
    assert ref["n"] == 2 and ref["wns"] == pytest.approx(-0.21) and ref["tns"] == pytest.approx(-310.0)
    assert p._default_ref("ariane133", "route") is None
    assert "DEFAULT FLOW reference (all knobs default), mean over 2" in p.build_prompt([_hist(0, 0.5, 1.0)], 1)


def test_registration_hook(tmp_path):
    from pnr_node import experiment
    log = tmp_path / "h.jsonl"
    register_llm_diag(log_path=str(log))
    try:
        prop = experiment.PROPOSERS_V2["llm_diag"]("v2_llm_diag#r1", 1, None, lambda: _prov(["[]"]))
        assert isinstance(prop, DiagProposer) and prop.arm == "v2_llm_diag#r1" and prop.log_path == str(log)
        with pytest.raises(ValueError):
            make_factory()("x", 0, None, None)
    finally:
        experiment.PROPOSERS_V2.pop("llm_diag", None)


def test_llm_diag_arm_end_to_end(tmp_path):
    """The registered arm through run_arm_v2 on the flow mock: budget respected, hypotheses logged."""
    from pnr_node import experiment
    from pnr_node.db import ResultsDB
    from pnr_node.mock_eval import FlowMockEvaluator
    log = tmp_path / "h.jsonl"
    register_llm_diag(log_path=str(log))
    try:
        reply = json.dumps([{"cfg": {"target_density": 0.4 + 0.05 * i}, "hypothesis": f"h{i}", "expected": "e"}
                            for i in range(4)])
        db = ResultsDB(":memory:")
        out = experiment.run_arm_v2(FlowMockEvaluator(), db, "llm_diag", 0, "gcd", budget_evals=6, batch=2,
                                    provider_factory=lambda: _prov([reply]))
        assert out is not None
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        assert rows and rows[0]["source"] == "llm" and rows[0]["arm"].startswith("v2_llm_diag")
    finally:
        experiment.PROPOSERS_V2.pop("llm_diag", None)
