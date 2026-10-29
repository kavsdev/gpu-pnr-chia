"""Net-weight timing-driven DREAMPlace: weight formula, config threading, identity."""
import os
import subprocess

import pytest

from pnr_node import netwt
from pnr_node.contract import EvalRequest, default_config

DUMP = """# wns -0.4 tns -3.0 viol 7
a -0.4 2 sig
b -0.1 3 sig
c 0.2 2 sig
d INF 2 sig
clk -0.4 36 clk
rst -0.4 150 hideg
"""


def _dump(tmp_path, text=DUMP):
    p = tmp_path / "r.slack"
    p.write_text(text)
    return netwt.parse_slacks(str(p))


def test_parse_slacks(tmp_path):
    d = _dump(tmp_path)
    assert (d.wns, d.tns, d.viol) == (-0.4, -3.0, 7)
    assert d.nets[3] == ("d", None, 2, "sig") and d.nets[4][3] == "clk"


def test_first_round_is_lilith_with_gain(tmp_path):
    w, c = netwt.update_weights(_dump(tmp_path), {}, {}, k=4.0, decay=0.5)
    assert w["a"] == pytest.approx(5 ** 0.5)  # nc = 4 * 1: crit = sqrt(1+4) - 1, weight = 1 + crit
    assert w["b"] == pytest.approx(2 ** 0.5)  # nc = 4 * 0.25
    assert w["c"] == w["d"] == 1.0  # positive / unconstrained slack: not weighted
    assert "clk" not in w and "rst" not in w  # clock and high-degree nets stay at DREAMPlace's default 1.0
    # k=1 is DREAMPlace's lilith exactly
    w1, _ = netwt.update_weights(_dump(tmp_path), {}, {}, k=1.0, decay=0.5)
    assert w1["a"] == pytest.approx(2 ** 0.5)


def test_weights_compound_across_rounds_and_clamp(tmp_path):
    d = _dump(tmp_path)
    w1, c1 = netwt.update_weights(d, {}, {}, k=4.0)
    w2, c2 = netwt.update_weights(d, w1, c1, k=4.0)
    assert w2["a"] > w1["a"] and c2["a"] > c1["a"]  # a persistently critical net keeps gaining weight
    w3, _ = netwt.update_weights(d, w2, c2, k=4.0, max_weight=3.0)
    assert w3["a"] == 3.0
    # timing met (WNS >= 0): criticality is frozen, weights still apply it (lilith)
    met = _dump(tmp_path, "# wns 0.1 tns 0 viol 0\na 0.1 2 sig\n")
    w4, c4 = netwt.update_weights(met, w1, c1)
    assert c4["a"] == c1["a"] and w4["a"] == pytest.approx(w1["a"] * (1 + c1["a"]))


def test_non_compounding_weights_converge(tmp_path):
    d = _dump(tmp_path)
    w, c = {}, {}
    for _ in range(20):
        w, c = netwt.update_weights(d, w, c, k=4.0, compound=False)
    assert w["a"] == pytest.approx(5.0, rel=1e-3)  # crit -> k*1, weight -> 1+k, no geometric growth
    w1, _ = netwt.update_weights(d, {}, {}, k=4.0, compound=False)
    wc, _ = netwt.update_weights(d, {}, {}, k=4.0)
    assert w1 == wc  # identical in round 1 (static mode unaffected)


def test_write_weights_skips_unit(tmp_path):
    p = tmp_path / "w.txt"
    assert netwt.write_weights(str(p), {"a": 2.0, "b": 1.0}) == 1
    assert p.read_text() == "a 2\n"


def test_build_config_net_weights_and_warm_start():
    from pnr_node.dreamplace_runner import build_config
    c = build_config([], "in.def", "/out", default_config(), 0)
    assert "net_weights_file" not in c and "random_center_init_flag" not in c
    c = build_config([], "in.def", "/out", default_config(), 0, net_weights_file="/w.txt", warm_start=True)
    assert c["net_weights_file"] == "/w.txt" and c["random_center_init_flag"] == 0
    assert "timing_opt_flag" not in c  # DREAMPlace's own timer stays off
    assert c["deterministic_flag"] == 1


def test_deterministic_gp_is_part_of_identity(tmp_path):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    assert RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True).toolchain["dp_deterministic"] is True


def test_net_weight_is_part_of_evaluator_identity(tmp_path):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    off = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True)
    st = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, net_weight="static")
    it = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, net_weight="iterative", netwt_rounds=3)
    assert "net_weight" not in off.toolchain
    assert st.toolchain["net_weight"]["rounds"] == 1 and it.toolchain["net_weight"]["rounds"] == 3
    assert len({off.tool_id, st.tool_id, it.tool_id}) == 3
    assert EvalRequest("gcd", tool_id=st.tool_id).key() != EvalRequest("gcd", tool_id=it.tool_id).key()
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, net_weight="iterative", netwt_rounds=1)
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, net_weight="static", timing_driven=True)


def test_orfs_vars_select_own_base_and_tool_id(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    plain = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True)
    a = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, orfs_vars={"GPL_TIMING_DRIVEN": "0"})
    b = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, orfs_vars={"PLACE_DENSITY": "0.33"})
    assert "orfs_vars" not in plain.toolchain  # default runs keep their existing cache keys
    assert plain.base_dir("gcd").endswith("/gcd/base")
    assert len({plain.tool_id, a.tool_id, b.tool_id}) == 3
    bases = {e.base_dir("gcd") for e in (plain, a, b)}
    assert len(bases) == 3 and not any("=" in x for x in bases)
    assert " GPL_TIMING_DRIVEN=0" in a._vars() and plain._vars() == ""
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, orfs_vars={"bad name": "1"})


def test_dp_extra_merges_and_is_identity(tmp_path, monkeypatch):
    from pnr_node.dreamplace_runner import build_config
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    c = build_config([], "in.def", "/out", default_config(), 0,
                     extra={"target_density": 0.33, "global_place_stages": {"iteration": 2000}})
    assert c["target_density"] == 0.33 and c["global_place_stages"][0]["iteration"] == 2000
    assert c["global_place_stages"][0]["optimizer"] == "nesterov"
    a = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True)
    b = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, dp_extra={"target_density": 0.33})
    assert "dp_extra" not in a.toolchain and a.tool_id != b.tool_id


def test_rebuffer_identity_and_patch(tmp_path, monkeypatch):
    from pnr_node import raw_handoff
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    off = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True)
    rb = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rebuffer=True)
    rb6 = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rebuffer=True, rebuf_overflow=0.6)
    assert "rebuffer" not in off.toolchain and len({off.tool_id, rb.tool_id, rb6.tool_id}) == 3
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rebuffer=True, net_weight="iterative")
    # without rebuf_from.def the splice must behave exactly as before: dump or import only
    p = raw_handoff._PATCH_NEW
    assert p.index("rebuf_from.def") < p.index("read_def -incremental $dp_def") < p.index("pre_gp.def")


def test_routability_params_from_def(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import RawHandoffEvaluator, routability_params
    d = tmp_path / "x.def"
    d.write_text("VERSION 5.8 ;\nUNITS DISTANCE MICRONS 2000 ;\nDIEAREA ( 0 0 ) ( 604800 604800 ) ;\n")
    p = routability_params(str(d))
    assert p["routability_opt_flag"] == 1 and p["adjust_nctugr_area_flag"] == 0
    assert abs(p["unit_horizontal_capacity"] - 7.656 * 0.19) < 1e-3  # m3/5/7/9 after fastroute.tcl adjustments
    assert abs(p["unit_vertical_capacity"] - 9.395 * 0.19) < 1e-3
    assert p["route_num_bins_x"] == 108 and p["route_num_bins_y"] == 108
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    a = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True)
    b = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, dp_routability=True)
    assert a.tool_id != b.tool_id


def test_rebuf_stops_parse_and_identity(tmp_path, monkeypatch):
    from pnr_node import raw_handoff
    from pnr_node.raw_handoff import RawHandoffEvaluator, _stops
    assert _stops(None) is None and _stops("0.6") == 0.6 and _stops("0.6,0.2") == [0.6, 0.2]
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    one = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rebuffer=True, rebuf_overflow=0.6)
    two = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rebuffer=True, rebuf_overflow=[0.6, 0.2])
    assert one.rebuf_stops == [0.6] and two.rebuf_stops == [0.6, 0.2] and one.tool_id != two.tool_id
    assert one.toolchain["rebuffer"]["overflow"] == 0.6  # single-stop identity unchanged
    assert raw_handoff.RawHandoffEvaluator._rebuf_name(1) == "rebuf_from.def"
    assert two.toolchain["rebuffer"]["scheme"] == "odb-v1"


def test_rp_finish_identity_and_exclusivity(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    off = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True)
    h3 = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rp_finish=0.3)
    h5 = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rp_finish=0.5)
    h3a = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rp_finish=0.3,
                              rp_args="-init_density_penalty 0.01")
    assert "rp_finish" not in off.toolchain  # default identity unchanged
    assert h3.toolchain["rp_finish"]["overflow"] == 0.3
    assert len({off.tool_id, h3.tool_id, h5.tool_id, h3a.tool_id}) == 4
    for bad in ({"rebuffer": True}, {"net_weight": "static"}, {"timing_driven": True}):
        with pytest.raises(ValueError):
            RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rp_finish=0.3, **bad)
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rp_finish=1.5)
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rp_args="-overflow 0.1")


def test_rp_finish_patch_is_marker_gated():
    from pnr_node import raw_handoff
    p = raw_handoff._PATCH_NEW
    # RePlAce only runs inside the import branch, after gp.def is read, and only with the marker present
    i_read, i_mark = p.index("read_def -incremental $dp_def"), p.index("[file exists @W@/dp/rp_finish]")
    i_gpl = p.index("do_placement $rp_args")
    assert i_read < i_mark < i_gpl < p.index("} elseif { ![file exists @W@/dp/rebuf_from.def] } {\n  file mkdir")
    assert "-force_center_initial_place" in p and "-skip_initial_place" in p
    # result/errMsg defaults come before any branch, so the stock error check sees 0 when RePlAce does not run
    assert p.index('set result 0') < p.index("if { [file exists $dp_def] }")
    assert "do_placement $global_placement_args" not in p


def test_rp_finish_cli_and_gp3_stats(monkeypatch):
    import sys
    from pnr_node import raw_handoff
    seen = {}

    class Fake:
        tool_id, root = "x", "/nonexistent"

        def __init__(self, *a, **k):
            seen.update(k)

        def evaluate(self, req):
            from pnr_node.contract import EvalResult
            return EvalResult(False, req.stage, error="fake")

    monkeypatch.setattr(raw_handoff, "RawHandoffEvaluator", Fake)
    monkeypatch.setattr(sys, "argv", ["x", "--rp-finish", "0.3", "--rp-args", "-init_density_penalty 0.01"])
    raw_handoff.main()
    assert seen["rp_finish"] == 0.3 and seen["rp_args"] == "-init_density_penalty 0.01"
    log = """global_placement -density 0.75 -skip_initial_place
        0 |   0.3012 |  3.0e+05 |   +0.00% |  1.0e-09 |
[INFO GPL-0100] Timing-driven iteration 1/2, virtual: false.
[INFO GPL-0101]    Iter: 1, overflow: 0.301, keep resizer changes at: 1, HPWL: 1
[INFO GPL-0040] Routability iteration: 1
[INFO GPL-1001] Global placement finished at iteration 400
[INFO GPL-1018] Final HPWL (um): 400116.85
Took 12 seconds: global_placement -density 0.75
Elapsed time: 0:17.50[h:]min:sec. CPU time: user 1
"""
    s = raw_handoff.gp3_stats(log)
    assert s["ran"] and s["nesterov_iterations"] == 400 and s["start_overflow"] == 0.3012
    assert s["timing_driven_iters"] == [{"iter": 1, "overflow": 0.301}] and s["routability_iters"] == 1
    assert s["gpl_runtime_s"] == 12 and s["stage_elapsed_s"] == 17.5 and s["final_hpwl_um"] == 400116.85
    assert raw_handoff.gp3_stats("")["ran"] is False


GPL_DUMP = "# wns -0.5 tns -1 viol 3\n" + "".join(f"n{i} {s} 2 sig\n" for i, s in enumerate(
    [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])) + \
    "clk -0.9 30 clk\nbig -0.5 150 hideg\nu INF 2 sig\n"


def test_gpl_weights_formula(tmp_path):
    d = _dump(tmp_path, GPL_DUMP)
    # population: 20 sig + 1 hideg (clock and unconstrained out) -> ceil(21 * 10%) = 3 worst: big/n0 (-0.5), n1 (-0.3)
    w = netwt.gpl_weights(d, 10.0, 5.0)
    assert w == {"big": 5.0, "n0": 5.0}  # slack_max = -0.3 itself is not < slack_max -> weight 1, not written
    w = netwt.gpl_weights(d, 20.0, 5.0)  # ceil(4.2) = 5 worst: -0.5,-0.5,-0.3,-0.1,0.0 -> slack_max 0.0
    assert set(w) == {"big", "n0", "n1", "n2"}
    assert w["n0"] == pytest.approx(5.0) and w["n1"] == pytest.approx(1 + 4 * 0.3 / 0.5)
    assert w["n2"] == pytest.approx(1 + 4 * 0.1 / 0.5)
    # weights met timing too (sign irrelevant), and nothing on clocks / no slacks
    w = netwt.gpl_weights(netwt.SlackDump(0.1, 0, 0, [("a", 0.1, 2, "sig"), ("b", 0.3, 2, "sig"),
                                                      ("c", 0.5, 2, "sig")]), 100.0, 1.9)
    assert w == {"a": pytest.approx(1.9), "b": pytest.approx(1.45)}
    assert netwt.gpl_weights(netwt.SlackDump(None, None, None, [("c", -1.0, 2, "clk")])) == {}
    tie = netwt.SlackDump(0, 0, 0, [("a", -0.2, 2, "sig"), ("b", -0.2, 2, "sig"), ("c", 0.3, 2, "sig")])
    assert netwt.gpl_weights(tie, 50.0, 5.0) == {}  # slack_min == slack_max


def test_gpl_scheme_identity_and_validation(tmp_path, monkeypatch):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    kw = dict(native=True, rebuffer=True, rebuf_full=True, rebuf_overflow=[0.6, 0.2], net_weight="static")
    lil = RawHandoffEvaluator(str(tmp_path), str(tmp_path), **kw)
    lil2 = RawHandoffEvaluator(str(tmp_path), str(tmp_path), netwt_scheme="lilith", **kw)
    g = RawHandoffEvaluator(str(tmp_path), str(tmp_path), netwt_scheme="gpl", **kw)
    g3 = RawHandoffEvaluator(str(tmp_path), str(tmp_path), netwt_scheme="gpl", netwt_gpl_max=3.0, **kw)
    assert lil.toolchain == lil2.toolchain and lil.toolchain["net_weight"]["scheme"] == "lilith-opensta-v2"
    assert "pct" not in lil.toolchain["net_weight"]
    assert g.toolchain["net_weight"] == {"mode": "static", "scheme": "gpl-opensta-v1", "pct": 10.0, "max": 5.0}
    assert len({lil.tool_id, g.tool_id, g3.tool_id}) == 3
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, net_weight="static", netwt_scheme="gpl")
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), netwt_scheme="foo", **kw)


def test_gpl_rebuf_weights_only_after_repairs(tmp_path, monkeypatch):
    """GP1 unweighted; GP k>=2 weighted from the slack dump of kept repair k-1 (written by that stage's 3_3 session,
    no separate STA run); weights reset per stage. Each repair runs once: stage k resumes stage k-1's saved odb."""
    from pnr_node import raw_handoff
    from pnr_node.dreamplace_runner import DPResult
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    ev = RawHandoffEvaluator(str(tmp_path / "w"), str(tmp_path), native=True, rebuffer=True, rebuf_full=True,
                             rebuf_overflow=[0.6, 0.2], net_weight="static", netwt_scheme="gpl")
    base = tmp_path / "base"
    (base / "dp").mkdir(parents=True)
    (base / "results" / "nangate45" / "gcd" / "base").mkdir(parents=True)
    gp_calls, makes = [], []

    def fake_sta(*a, **k):
        raise AssertionError("gpl rebuffer must not run a separate STA process")

    def fake_dp(lefs, netlist, out, cfg, seed, net_weights_file=None, warm_start=False, **kw):
        os.makedirs(out, exist_ok=True)
        gp_calls.append((os.path.basename(netlist), net_weights_file and open(net_weights_file).read(), warm_start))
        nset = 1 if net_weights_file else 0
        open(os.path.join(out, "dreamplace.log"), "w").write(f"net_weights_file f: set {nset} nets, 0 unmatched\n")
        dp_def = os.path.join(out, "gp.def")
        open(dp_def, "w").write("x")
        return DPResult(True, def_path=dp_def, hpwl=1.0, overflow=0.1, iterations=1, runtime_s=0.1)

    def fake_make(work, design, target="", timeout=0):
        k = len(makes) + 1
        dp = os.path.join(work, "dp")
        rin = os.path.join(work, "results", "nangate45", design, "base", raw_handoff._REBUF_IN)
        makes.append({"in": open(rin).read() if os.path.exists(rin) else None,
                      "from": open(os.path.join(dp, "rebuf_from.def")).read(),
                      "full": os.path.exists(os.path.join(dp, "rebuf_full")),
                      "sta": os.path.exists(os.path.join(dp, "rebuf_sta")) and
                             os.path.exists(os.path.join(dp, netwt.DUMP_TCL))})
        open(os.path.join(dp, "rebuf.def"), "w").write("netlist")
        open(os.path.join(dp, "rebuf_out.odb"), "w").write(f"state{k}")
        with open(os.path.join(dp, "rebuf.slack"), "w") as f:
            f.write("# wns -0.5 tns -1 viol 2\n" + f"xr{k} -0.5 2 sig\ny 0.0 2 sig\n" +
                    "".join(f"z{i} 1.0 2 sig\n" for i in range(18)))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(ev, "_sta", fake_sta)
    monkeypatch.setattr(ev, "lefs_for", lambda d: [])
    monkeypatch.setattr(ev, "_patch_scripts", lambda w: None)
    monkeypatch.setattr(ev, "_make", fake_make)
    monkeypatch.setattr(ev, "_repair_stats", lambda w, d: {})
    monkeypatch.setattr(raw_handoff, "components_only_def", lambda a, b: open(b, "w").write("c" + os.path.basename(b)))
    monkeypatch.setattr(raw_handoff, "run_dreamplace", fake_dp)
    d = tmp_path / "d"
    r = ev._place_rebuf(EvalRequest("gcd", seed=0), str(base), str(d))
    assert r.ok
    # stage 1 starts from 3_2 (no saved state), stage 2 resumes stage 1's session; each repairs only its own GP
    assert makes == [{"in": None, "from": "crebuf_from.def", "full": True, "sta": True},
                     {"in": "state1", "from": "crebuf_from2.def", "full": True, "sta": True}]
    assert gp_calls[0] == ("pre_gp.def", None, False)
    assert gp_calls[1] == ("rebuf1.def", "xr1 5\n", True) and gp_calls[2] == ("rebuf2.def", "xr2 5\n", True)
    assert r.rebuf["odb"] == str(d / "rebuf" / "rebuf2.odb") and open(r.rebuf["odb"]).read() == "state2"
    assert (d / "rebuf" / "r1.slack").exists() and (d / "rebuf" / "r2.slack").exists()
    st = r.rebuf["trace"]["stages"]
    assert "weights" not in st[0] and st[1]["weights"]["weighted"] == 1 and st[2]["dp_nets_unmatched"] == 0
    assert st[1]["sta"]["wns"] == -0.5


def test_rebuf_import_resumes_saved_session(tmp_path, monkeypatch):
    """The evaluation's 3_3 gets only the saved odb (no repair inputs) and fails if that 3_3 repaired anything."""
    from pnr_node import raw_handoff
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    ev = RawHandoffEvaluator(str(tmp_path / "w"), str(tmp_path), native=True, rebuffer=True, rebuf_full=True,
                             rebuf_overflow=[0.6, 0.2])
    base = tmp_path / "base"
    rd = base / "results" / "nangate45" / "gcd" / "base"
    rd.mkdir(parents=True)
    (base / "dp").mkdir()
    odb = tmp_path / "rebuf2.odb"
    odb.write_text("state2")
    seen = {}

    def fake_make(work, design, target="", timeout=0):
        dp = os.path.join(work, "dp")
        seen["in"] = open(os.path.join(work, "results", "nangate45", design, "base", raw_handoff._REBUF_IN)).read()
        seen["dp"] = sorted(os.listdir(dp))
        return subprocess.CompletedProcess([], 0, "", "")

    stats = {"buffers": 0, "resized": 0, "undefined_comps": 0}
    monkeypatch.setattr(ev, "_patch_scripts", lambda w: None)
    monkeypatch.setattr(ev, "_make", fake_make)
    monkeypatch.setattr(ev, "_repair_stats", lambda w, d: dict(stats))
    monkeypatch.setattr(ev, "collect_grt", lambda w, d: {"grt_wl": 1, "_evidence": {}})
    monkeypatch.setattr(raw_handoff, "components_only_def", lambda a, b: open(b, "w").write("c") and 1)
    req = EvalRequest("gcd", seed=0, stage="grt")
    run = tmp_path / "run"
    res = ev.route_from_def(req, str(base), str(run), "gp.def", 1.0, {}, 0.0, rebuf={"odb": str(odb)})
    assert res.ok and seen == {"in": "state2", "dp": ["gp.def"]}
    stats["buffers"] = 3  # a repair in the import run means the saved session was not what was resumed
    res = ev.route_from_def(req, str(base), str(run), "gp.def", 1.0, {}, 0.0, rebuf={"odb": str(odb)})
    assert not res.ok and "diverged" in res.error


def test_rebuf_patch_resume_prelude(tmp_path):
    """The prelude patch keeps the stock text verbatim for non-resume runs and skips remove_buffers/buffer_ports
    when resuming; a changed upstream prelude fails loudly."""
    from pnr_node import raw_handoff
    from pnr_node.raw_handoff import HandoffError, RawHandoffEvaluator
    ev = RawHandoffEvaluator(str(tmp_path / "w"), str(tmp_path), native=True, rebuffer=True)
    orig = ("source load.tcl\n" + raw_handoff._PRELUDE_OLD + "\nset global_placement_args {}\n" +
            raw_handoff._PATCH_OLD + "\n")
    so = tmp_path / "w" / "_shared" / "scripts_orig"
    so.mkdir(parents=True)
    (so / "global_place.tcl").write_text(orig)
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    ev._patch_scripts(str(work))
    out = (work / "scripts" / "global_place.tcl").read_text()
    resume, stock = out.split("} else {\n", 1)
    assert raw_handoff._REBUF_IN in resume and "load_design 3_3_rebuf_in.odb 2_floorplan.sdc" in resume
    assert "  remove_buffers\n" not in resume and "buffer_ports {*}" not in resume and "set_dont_use" in resume
    assert raw_handoff._PRELUDE_OLD in stock and out.index("write_db") < out.index("pre_gp.def")
    (so / "global_place.tcl").write_text(orig.replace("  remove_buffers\n", "  remove_buffers -x\n"))
    with pytest.raises(HandoffError):
        ev._patch_scripts(str(work))


def test_prepare_rebuilds_base_when_3_2_odb_pruned(tmp_path, monkeypatch):
    """A base whose pre_gp.def survived but whose 3_2 odb was pruned is rebuilt, not trusted (else every 3_3 copy
    silently re-synthesizes inside make)."""
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    ev = RawHandoffEvaluator(str(tmp_path / "w"), str(tmp_path), native=True)
    monkeypatch.setattr(ev, "_setup", lambda: None)
    base = ev.base_dir("gcd")
    rd = os.path.join(base, "results", ev.platform, "gcd", "base")
    os.makedirs(os.path.join(base, "dp"))
    os.makedirs(rd)
    open(os.path.join(base, "dp", "pre_gp.def"), "w").write("x")
    built = []
    monkeypatch.setattr(ev, "_prepare_locked", lambda d, b, m: built.append(d) or b)
    assert ev.prepare("gcd") == base and built == ["gcd"]  # odb missing -> rebuild
    open(os.path.join(rd, "3_2_place_iop.odb"), "w").write("x")
    assert ev.prepare("gcd") == base and built == ["gcd"]  # complete -> cached


def test_rebuf_presplit_identity_and_patch(tmp_path, monkeypatch):
    from pnr_node import raw_handoff
    from pnr_node.raw_handoff import RawHandoffEvaluator
    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    kw = dict(native=True, rebuffer=True, rebuf_full=True, rebuf_overflow=[0.6, 0.2])
    off = RawHandoffEvaluator(str(tmp_path), str(tmp_path), **kw)
    on = RawHandoffEvaluator(str(tmp_path), str(tmp_path), rebuf_presplit=True, **kw)
    assert "presplit" not in off.toolchain["rebuffer"] and on.toolchain["rebuffer"]["presplit"]["thr"] == 2000
    assert off.tool_id != on.tool_id
    with pytest.raises(ValueError):
        RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, rebuf_presplit=True)
    p = raw_handoff._PATCH_NEW
    # presplit is marker-gated and runs after the stage placement is read, before parasitics/repair
    i_read = p.index("read_def -incremental @W@/dp/rebuf_from.def")
    i_ps, i_rd = p.index("presplit_heavy_nets 2000 1000 BUF_X4"), p.index("repair_design_helper")
    assert i_read < p.index("@W@/dp/rebuf_presplit]") < i_ps < i_rd
    assert os.path.exists(os.path.join(os.path.dirname(raw_handoff.__file__), "rebuf_presplit.tcl"))


def test_rebuf_stage_exits_after_outputs():
    """A re-buffer stage run exits right after its own outputs (session, netlist DEF, slack dump): ORFS's step tail
    (estimate_parasitics, report_metrics, write_db 3_3_place_gp.odb) is skipped. Dump and import runs keep it."""
    from pnr_node import raw_handoff
    p = raw_handoff._PATCH_NEW
    stage = p[p.index("if { [file exists @W@/dp/rebuf_from.def] && ![file exists $dp_def] } {"):p.index("set result 0")]
    assert p.count("\n  exit\n") == 1 and stage.rstrip().endswith("exit\n}")
    i_exit = p.index("\n  exit\n")
    for out in ("write_db @W@/dp/rebuf_out.odb", "write_def @W@/dp/rebuf.def", "source @W@/dp/net_slack_dump.tcl"):
        assert p.index(out) < i_exit
    assert i_exit < p.index("read_def -incremental $dp_def") < p.index("write_def @W@/dp/pre_gp.def")
    assert "exit" not in p[p.index("set result 0"):]
