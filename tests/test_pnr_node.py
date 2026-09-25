import json

import pytest

from pnr_node.contract import CORE_KNOBS, KNOBS, EvalRequest, EvalResult, clamp_config, default_config, score
from pnr_node.db import ResultsDB
from pnr_node.llm import FakeProvider, FallbackProvider, LLMError, Msg, extract_json, make_provider
from pnr_node.mock_eval import MockEvaluator
from pnr_node.search import LLMProposer, RandomProposer, run_default, run_search


def test_clamp_and_defaults():
    c = clamp_config({"target_density": 5, "gamma": 2.5, "num_bins": 700, "bogus": 1})
    assert c["target_density"] == KNOBS["target_density"].hi and c["gamma"] == 2.5
    assert c["num_bins"] == KNOBS["num_bins"].hi  # num_bins is now a real (expanded) knob; 700 clamps to its hi
    assert "bogus" not in c and set(c) == set(KNOBS)
    assert clamp_config({}) == default_config()


def test_expanded_knob_defaults_reproduce_the_old_hardcoded_values():
    """Every expanded knob's default is the value build_config used to hardcode, so an arm that does not vary it is
    behaviourally identical to the original 5-knob pipeline -- the whole comparison rests on this."""
    d = default_config()
    assert d["iteration"] == 1000 and d["Llambda_density_weight_iteration"] == 1 and d["Lsub_iteration"] == 1
    assert d["num_bins"] == 0  # 0 (<32) => DREAMPlace's auto grid, i.e. the historical default


def test_core_knobs_restricts_the_action_space():
    """The baseline arm tunes only CORE_KNOBS; clamping to it drops the expanded knobs entirely."""
    assert set(CORE_KNOBS) == {"target_density", "density_weight", "gamma", "stop_overflow", "learning_rate"}
    assert set(CORE_KNOBS).issubset(set(KNOBS)) and len(KNOBS) > len(CORE_KNOBS)
    core = clamp_config({"target_density": 0.7, "iteration": 1500, "num_bins": 128}, CORE_KNOBS)
    assert set(core) == set(CORE_KNOBS) and "iteration" not in core and "num_bins" not in core
    assert clamp_config({}, CORE_KNOBS) == default_config(CORE_KNOBS)


def test_build_config_dp_depth_and_expanded_knobs():
    from pnr_node.dreamplace_runner import build_config

    base = build_config([], "in.def", "/out", default_config(), 0)  # dp_depth="global" default
    assert base["legalize_flag"] == 0 and base["detailed_place_flag"] == 0
    st = base["global_place_stages"][0]
    assert st["iteration"] == 1000 and st["Llambda_density_weight_iteration"] == 1 and st["Lsub_iteration"] == 1
    assert base["num_bins_x"] == 0 and base["num_bins_y"] == 0  # num_bins default 0 -> DREAMPlace auto grid

    legal = build_config([], "in.def", "/out", default_config(), 0, dp_depth="global+legal")
    assert legal["legalize_flag"] == 1 and legal["detailed_place_flag"] == 0
    full = build_config([], "in.def", "/out", default_config(), 0, dp_depth="global+legal+detail")
    assert full["legalize_flag"] == 1 and full["detailed_place_flag"] == 1
    with pytest.raises(ValueError):
        build_config([], "in.def", "/out", default_config(), 0, dp_depth="bogus")
    # DREAMPlace's key is "detailed_place_flag"; the old misspelling was silently ignored (detail placement never ran)
    assert "detail_place_flag" not in full


def test_build_config_timing_uses_heterosta_with_lib_list():
    from pnr_node.dreamplace_runner import build_config

    c = build_config([], "in.def", "/out", default_config(), 0, timing_driven=True,
                     lib_input=["a.lib", "b.lib"], sdc_input="c.sdc")
    # lib_input must be a list (a space-joined string broke DREAMPlace's option parser). Timing mode itself is
    # not functional in this flow; this only pins the config shape.
    assert c["timing_opt_flag"] == 1 and c["timer_engine"] == "heterosta"
    assert c["lib_input"] == ["a.lib", "b.lib"] and c["sdc_input"] == "c.sdc"

    ex = build_config([], "in.def", "/out",
                      clamp_config({"iteration": 1500, "Llambda_density_weight_iteration": 3,
                                    "Lsub_iteration": 2, "num_bins": 128}), 0)
    st = ex["global_place_stages"][0]
    assert st["iteration"] == 1500 and st["Llambda_density_weight_iteration"] == 3 and st["Lsub_iteration"] == 2
    assert ex["num_bins_x"] == 128 and ex["num_bins_y"] == 128


def test_parse_log_runtime_ignores_detailed_placement_time():
    from pnr_node.dreamplace_runner import parse_log

    log = ("[INFO   ] DREAMPlace - iteration  500, ( 500,  0,  0), Obj 1.5E+06, DensityWeight 1.1E-02, "
           "wHPWL 1.780514E+06, Overflow 1.017945E-01, MaxDensity 3.555E+00, gamma 1.0E+01, time 0.459ms\n"
           "[INFO   ] DREAMPlace - legalization takes 0.035 seconds\n"
           "[INFO   ] DREAMPlace - detailed placement takes 3.651 seconds\n"
           "[INFO   ] DREAMPlace - iteration  503, wHPWL 1.885476E+06, time 0.252ms\n"
           "[INFO   ] DREAMPlace - non-linear placement takes 6.62 seconds\n"
           "[INFO   ] DREAMPlace - placement takes 7.130 seconds\n")
    info = parse_log(log)
    assert info["runtime_s"] == 6.62 and info["iterations"] == 501 and info["overflow"] == 0.1017945


def test_request_validation_and_stable_key():
    a = EvalRequest("gcd", placer_cfg={"target_density": 0.7})
    b = EvalRequest("gcd", placer_cfg={"target_density": 0.7})
    assert a.key() == b.key()
    with pytest.raises(ValueError):
        EvalRequest("gcd", stage="bogus")


def test_result_roundtrip_and_score():
    r = EvalResult(True, "route", routed_wl=100.0, wns=-0.1, drc_violations=10)
    assert EvalResult.from_json(r.to_json()) == r
    assert score(r) == pytest.approx(100.0 * (1 + 1.0 + 0.1))
    assert score(EvalResult(False, "route")) == float("inf")


def test_score_treats_unparsed_drc_as_worse_than_any_known_count():
    """Verdict must come from the parsed report, never a bare 'the process exited 0': a route whose
    DRC count failed to parse is UNKNOWN, not clean, and must never silently outscore a real garbage-but-measured
    route or a clean one."""
    clean = EvalResult(True, "route", routed_wl=100.0, drc_violations=0)
    dirty = EvalResult(True, "route", routed_wl=100.0, drc_violations=5000)
    unparsed = EvalResult(True, "route", routed_wl=100.0, drc_violations=None)
    assert score(clean) < score(dirty) < score(unparsed)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! here: {"a": [1, 2]} done') == {"a": [1, 2]}


def test_make_provider_and_unknown():
    assert make_provider({"provider": "fake", "replies": ["x"]}).name == "fake"
    with pytest.raises(ValueError):
        make_provider({"provider": "nope"})


def test_fallback_reports_and_switches():
    class Boom:
        name = "boom"

        def complete(self, *a, **k):
            raise RuntimeError("rate limited")

    seen = []
    fb = FallbackProvider([Boom(), FakeProvider(replies=["ok"])], on_fallback=lambda n, e: seen.append((n, str(e))))
    assert fb.complete("s", [Msg("user", "hi")]).text == "ok"
    assert seen == [("boom", "rate limited")]
    with pytest.raises(LLMError):
        FallbackProvider([Boom()]).complete("s", [Msg("user", "hi")])


def test_two_tier_search_respects_route_budget():
    db, ev = ResultsDB(), MockEvaluator()
    out = run_search(ev, RandomProposer(0), db, "random", "gcd", rounds=3, batch=8, route_top=2)
    assert out["routes_used"] == 6 and ev.calls["route"] == 6
    assert ev.calls["proxy"] == 24  # every candidate is screened cheaply
    assert out["best_score"] is not None
    assert out["stopped_early"] is False  # no deadline given: unbounded, same as before this existed


def test_wall_clock_deadline_stops_launching_new_rounds_but_keeps_finished_work():
    """A minimal wall-clock budget cap: checked only between rounds, so a round already dispatched always finishes -- this
    is what makes it safe insurance for an unattended run rather than a source of lost work."""
    import time

    db, ev = ResultsDB(), MockEvaluator()
    # deadline already in the past: round 1 must never be dispatched.
    out = run_search(ev, RandomProposer(0), db, "capped", "gcd", rounds=5, batch=8, route_top=2,
                     deadline=time.time() - 1)
    assert out["stopped_early"] is True
    assert out["routes_used"] == 0 and ev.calls["proxy"] == 0

    # deadline far in the future behaves exactly like no deadline at all.
    db2, ev2 = ResultsDB(), MockEvaluator()
    out2 = run_search(ev2, RandomProposer(0), db2, "uncapped", "gcd", rounds=3, batch=8, route_top=2,
                      deadline=time.time() + 3600)
    assert out2["stopped_early"] is False and out2["routes_used"] == 6


def test_llm_proposer_uses_history_and_survives_garbage():
    db, ev = ResultsDB(), MockEvaluator()
    good = json.dumps({"reasoning": "x", "candidates": [{"target_density": 0.7}, {"target_density": 0.75}]})
    replies = [good, "not json at all", good]
    prov = FakeProvider(replies=replies)
    prop = LLMProposer(prov, db, arm="llm_routed")
    out = run_search(ev, prop, db, "llm_routed", "gcd", rounds=3, batch=2, route_top=1)
    assert out["routes_used"] == 3
    assert prop.invalid >= 1  # the garbage answer was counted, not hidden
    assert db.con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 3
    # round 2 prompt must contain the routed result of round 1
    assert "routed_wl" in prov.calls[1][1][0].content


def test_default_arm_and_mock_shows_proxy_misleads_at_high_density():
    ev = MockEvaluator()
    hi = EvalRequest("gcd", placer_cfg={"target_density": 1.0})
    lo = EvalRequest("gcd", placer_cfg={"target_density": 0.8})
    p_hi = ev.evaluate(EvalRequest("gcd", placer_cfg=hi.placer_cfg, stage="proxy")).proxy_hpwl
    p_lo = ev.evaluate(EvalRequest("gcd", placer_cfg=lo.placer_cfg, stage="proxy")).proxy_hpwl
    assert p_hi < p_lo  # proxy prefers high density ...
    assert score(ev.evaluate(hi)) > score(ev.evaluate(lo))  # ... routing disagrees
    db = ResultsDB()
    assert run_default(ev, db, "gcd")["routes_used"] == 1


# ---- additions: identity, evidence, placement check, cache, resume, manifest, ablation isolation ----
import os
import tempfile

from pnr_node.db import CachedEvaluator
from pnr_node.dreamplace_runner import check_placement
from pnr_node.evidence import atomic_write_json, make_bundle, verify_bundle, write_bundle
from pnr_node.manifest import manifest
from pnr_node.raw_handoff import components_only_def


def test_tool_id_is_part_of_identity():
    a = EvalRequest("gcd", tool_id="img1")
    b = EvalRequest("gcd", tool_id="img2")
    assert a.key() != b.key() and a.key() == EvalRequest("gcd", tool_id="img1").key()


def test_evidence_bundle_detects_tampering_and_excludes_narrative():
    req = EvalRequest("gcd")
    res = EvalResult(True, "route", routed_wl=3896.0, wns=-0.16)
    b = make_bundle(req, res, "python -m pnr_node.raw_handoff --design gcd", narrative="agent says it is great")
    assert verify_bundle(b) and b["payload"]["label"] == "measured-routed"
    b2 = dict(b, narrative="totally different story")
    assert verify_bundle(b2)  # narrative is outside the hash
    b["payload"]["result"]["routed_wl"] = 1.0
    assert not verify_bundle(b)  # a fabricated number is detected
    assert make_bundle(req, EvalResult(True, "proxy", proxy_hpwl=1.0))["payload"]["label"] == "proxy-estimate"


def test_hmac_key_changes_digest(monkeypatch):
    req, res = EvalRequest("gcd"), EvalResult(True, "route", routed_wl=1.0)
    plain = make_bundle(req, res)["sha256"]
    monkeypatch.setenv("PNR_EVIDENCE_KEY", "secret")
    keyed = make_bundle(req, res)
    assert keyed["sha256"] != plain and verify_bundle(keyed)


def test_atomic_write_and_bundle_file(tmp_path):
    p = str(tmp_path / "x" / "r.json")
    atomic_write_json(p, {"a": 1})
    atomic_write_json(p, {"a": 2})
    import json as _j
    assert _j.load(open(p)) == {"a": 2}
    assert [f for f in os.listdir(os.path.dirname(p)) if f.startswith(".tmp-")] == []
    path = write_bundle(str(tmp_path), EvalRequest("gcd"), EvalResult(True, "route", routed_wl=2.0))
    assert verify_bundle(_j.load(open(path)))


_DEF = """VERSION 5.8 ;
DESIGN t ;
UNITS DISTANCE MICRONS 2000 ;
DIEAREA ( 0 0 ) ( 1000 1000 ) ;
COMPONENTS 3 ;
    - a INV + PLACED ( 10 10 ) N ;
    - b INV + PLACED ( 500 500 ) N ;
    - c INV + PLACED ( %s %s ) N ;
END COMPONENTS
PINS 1 ;
    - clk + NET clk + PLACED ( 0 0 ) N ;
END PINS
END DESIGN
"""


def _write(tmp_path, x, y, name="t.def"):
    p = tmp_path / name
    p.write_text(_DEF % (x, y))
    return str(p)


def test_check_placement_rejects_only_the_invalid(tmp_path):
    assert check_placement(_write(tmp_path, 900, 900), 0.099, 0.1) == []
    r = check_placement(_write(tmp_path, 5000, 900), 0.099, 0.1)
    assert any(x.startswith("cells_outside_die(1/3)") for x in r)
    assert any(x.startswith("not_converged") for x in check_placement(_write(tmp_path, 1, 1), 0.4, 0.1))
    assert any(x.startswith("not_converged") for x in check_placement(_write(tmp_path, 1, 1), None, 0.1))


def test_components_only_def_drops_pins(tmp_path):
    out = str(tmp_path / "co.def")
    assert components_only_def(_write(tmp_path, 1, 1), out) == 3
    txt = open(out).read()
    assert "PINS" not in txt and "COMPONENTS 3" in txt and txt.strip().endswith("END DESIGN")


def test_cache_replays_identical_identity_and_counts_executed_routes():
    db, ev = ResultsDB(), MockEvaluator()
    cev = CachedEvaluator(ev, db)
    run_search(cev, RandomProposer(0), db, "a", "gcd", rounds=1, batch=4, route_top=2)
    n = ev.calls["route"]
    req = EvalRequest("gcd", placer_cfg=db.evals("a", "gcd", "route")[0]["cfg"], stage="route", tool_id="mock")
    # same identity from a different arm is a hit and runs nothing
    res = cev.evaluate(req)
    assert res.provenance.get("cache_hit") and ev.calls["route"] == n and cev.hits == 1
    db.record_eval("b", 1, req, res, "cache_hit")
    assert db.n_routes("b", "gcd") == 1 and db.n_routes("b", "gcd", executed_only=True) == 0


def test_resume_continues_rounds_and_history(tmp_path):
    path = str(tmp_path / "r.db")
    ev = MockEvaluator()
    run_search(ev, RandomProposer(0), ResultsDB(path), "r", "gcd", rounds=2, batch=4, route_top=1)
    db2 = ResultsDB(path)  # a "restarted worker"
    assert db2.max_round("r", "gcd") == 2
    seen = []

    class Spy(RandomProposer):
        def propose(self, history, n):
            seen.append(len(history))
            return super().propose(history, n)

    run_search(ev, Spy(5), db2, "r", "gcd", rounds=1, batch=4, route_top=1)
    assert db2.max_round("r", "gcd") == 3 and seen[0] >= 2  # history restored from the DB


def test_proxy_only_arm_is_blind_to_routing_even_in_ordering():
    db, ev = ResultsDB(), MockEvaluator()
    prov = FakeProvider(replies=[json.dumps({"candidates": [{"target_density": 0.7}]})])
    prop = LLMProposer(prov, db, arm="llm_proxy", show_routed=False)
    run_search(ev, prop, db, "llm_proxy", "gcd", rounds=2, batch=1, route_top=1)
    prompt = prov.calls[1][1][0].content
    assert "routed_wl" not in prompt and "wns" not in prompt and "proxy_hpwl" in prompt


def test_pareto_front_and_manifest():
    db, ev = ResultsDB(), MockEvaluator()
    run_search(ev, RandomProposer(3), db, "p", "gcd", rounds=2, batch=6, route_top=3)
    front = db.pareto("p", "gcd")
    assert front and len(front) <= db.n_routes("p", "gcd")
    m = manifest()
    assert set(KNOBS) == set(m["knobs"]) and any("signoff" in g.lower() for g in m["known_gaps"])


def test_search_uses_batch_path_and_cache_batches():
    class Batch(MockEvaluator):
        def __init__(self):
            super().__init__()
            self.batches = []

        def evaluate_many(self, reqs):
            self.batches.append((reqs[0].stage, len(reqs)))
            return [self.evaluate(r) for r in reqs]

    db, ev = ResultsDB(), Batch()
    run_search(ev, RandomProposer(0), db, "b", "gcd", rounds=1, batch=6, route_top=3)
    assert ev.batches == [("proxy", 6), ("route", 3)]
    cev = CachedEvaluator(ev, db)
    reqs = [EvalRequest("gcd", placer_cfg=h["cfg"], stage="route", tool_id="mock") for h in db.evals("b", "gcd", "route")]
    before = ev.calls["route"]
    out = cev.evaluate_many(reqs + [EvalRequest("gcd", placer_cfg={"target_density": 0.55}, stage="route", tool_id="mock")])
    assert ev.calls["route"] == before + 1 and cev.hits == len(reqs) and len(out) == len(reqs) + 1


def test_node_runs_in_process_without_chia(monkeypatch, tmp_path):
    import pnr_node.node as node

    class Fake:
        tool_id, toolchain = "t1", {"x": 1}

        def evaluate(self, req):
            return EvalResult(True, req.stage, proxy_hpwl=1.0)

    monkeypatch.setattr(node, "_evaluator", lambda cfg: Fake())
    ce = node.ChiaEvaluator({"work_root": str(tmp_path)})
    assert ce.tool_id == "t1"
    out = ce.evaluate_many([EvalRequest("gcd", stage="proxy"), EvalRequest("gcd", stage="proxy", seed=1)])
    assert [r.ok for r in out] == [True, True]


def test_dp_env_unhides_gpu_from_ray(monkeypatch):
    from pnr_node.dreamplace_runner import dp_env
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert "CUDA_VISIBLE_DEVICES" not in dp_env("/x")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert dp_env("/x")["CUDA_VISIBLE_DEVICES"] == "1"  # an explicit choice is respected
    assert dp_env("/x")["PYTHONPATH"] == "/x"


def test_flock_serialises_first_time_prepare(tmp_path):
    import threading, time
    from pnr_node.raw_handoff import flock
    lock = str(tmp_path / "d" / ".prepare.lock")
    marker = tmp_path / "d" / "marker"
    built = []

    def worker(i):
        if marker.exists():
            return
        with flock(lock):
            if marker.exists():
                return
            built.append(i)  # the expensive stock-ORFS prepare
            time.sleep(0.2)
            marker.write_text("x")

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(built) == 1


def test_study_sampling_and_statistics():
    from pnr_node.study import analyze, sample_configs, spearman, topk_recall
    cfgs = sample_configs(20, seed=1)
    assert len(cfgs) == 20 and all(0.4 <= c["target_density"] <= 1.0 for c in cfgs)
    tds = sorted(c["target_density"] for c in cfgs)
    assert all(tds[i + 1] - tds[i] < 0.2 for i in range(19))  # stratified: covers the range
    assert sample_configs(20, seed=1) == cfgs
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(spearman([1, 1, 2, 3], [1, 2, 3, 4])) <= 1.0  # ties handled
    assert topk_recall([1, 2, 3, 4, 5, 6], [1, 2, 3, 6, 5, 4], 3) == 1.0
    assert topk_recall([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], 3) == 0.0
    rows = [{"proxy_hpwl": float(i), "routed_wl": 100.0 + i, "score": 100.0 + i} for i in range(12)]
    a = analyze(rows)
    assert a["spearman_proxy_vs_routed_wl"] == 1.0 and a["top3_recall"] == 1.0 and a["n_routed"] == 12
    assert analyze(rows[:2])["error"]


def test_llm_sees_rejections_and_proxy_arm_sees_only_proxy_failures():
    class Picky(MockEvaluator):
        """Rejects low density at the proxy tier; fails one route outright."""
        def evaluate(self, req):
            if req.placer_cfg["target_density"] < 0.7:
                return EvalResult(False, req.stage, error="rejected: not_converged(overflow=0.9)")
            if req.stage == "route" and req.placer_cfg["target_density"] > 0.95:
                return EvalResult(False, "route", error="orfs make rc=2: boom")
            return super().evaluate(req)

    reply = json.dumps({"candidates": [{"target_density": 0.62}, {"target_density": 0.99}, {"target_density": 0.8}]})
    for show_routed, must, must_not in [(True, "orfs make rc=2", None), (False, "not_converged", "orfs make rc=2")]:
        db, prov = ResultsDB(), FakeProvider(replies=[reply])
        prop = LLMProposer(prov, db, arm="a", show_routed=show_routed)
        run_search(Picky(), prop, db, "a", "gcd", rounds=2, batch=3, route_top=3)
        prompt = prov.calls[1][1][0].content
        assert "Rejected or failed candidates" in prompt and "not_converged" in prompt and must in prompt
        if must_not:
            assert must_not not in prompt


def test_default_sampling_stays_in_validated_box_and_wide_is_opt_in():
    from pnr_node.study import WIDE, sample_configs
    for c in sample_configs(30, 2):
        assert 0.6 <= c["target_density"] <= 1.0 and 1e-5 <= c["density_weight"] <= 1e-3
        assert 0.004 <= c["learning_rate"] <= 0.025 and 2 <= c["gamma"] <= 12 and 0.05 <= c["stop_overflow"] <= 0.15
    wide = sample_configs(60, 2, WIDE)  # clamped into the (narrow) legal knobs by clamp, but sampled from wider draws
    assert len(wide) == 60


def test_grt_funnel_routes_by_grt_not_by_proxy():
    """With grt_top set the detailed route goes to the best global-route estimate, ignoring a misleading proxy."""
    class Trap(MockEvaluator):
        """Proxy prefers high density; grt/route punish it (the situation measured on gcd)."""
        def evaluate(self, req):
            r = super().evaluate(req)
            return r

    db, ev = ResultsDB(), Trap()
    run_search(ev, RandomProposer(4), db, "g", "gcd", rounds=2, batch=8, route_top=2, grt_top=8)
    routed = db.evals("g", "gcd", "route")
    grt = {json.dumps(h["cfg"], sort_keys=True): h["result"].grt_wl for h in db.evals("g", "gcd", "grt")}
    assert len(routed) == 4 and db.n_stage("g", "gcd", "grt") == 16
    for rnd in (1, 2):
        pool = sorted((h for h in db.evals("g", "gcd", "grt") if h["round"] == rnd), key=lambda h: h["result"].grt_wl)
        want = [json.dumps(h["cfg"], sort_keys=True) for h in pool[:2]]
        got = [json.dumps(h["cfg"], sort_keys=True) for h in routed if h["round"] == rnd]
        assert sorted(got) == sorted(want)
    out = run_search(ev, RandomProposer(9), db, "g", "gcd", rounds=1, batch=4, route_top=1, grt_top=4)
    assert out["grt_runs"] == 20 and out["routes_used"] == 5


def test_llm_history_labels_grt_as_estimate_and_proxy_arm_never_sees_it():
    db, ev = ResultsDB(), MockEvaluator()
    reply = json.dumps({"candidates": [{"target_density": 0.7}, {"target_density": 0.9}, {"target_density": 0.8}]})
    prov = FakeProvider(replies=[reply])
    run_search(ev, LLMProposer(prov, db, arm="r", show_routed=True), db, "r", "gcd", rounds=2, batch=3, route_top=1, grt_top=3)
    p = prov.calls[1][1][0].content
    assert "grt_wl_estimate" in p and "global-route estimate" in p and "routed_wl" in p
    prov2 = FakeProvider(replies=[reply])
    db2 = ResultsDB()
    run_search(ev, LLMProposer(prov2, db2, arm="p", show_routed=False), db2, "p", "gcd", rounds=2, batch=3, route_top=1, grt_top=3)
    p2 = prov2.calls[1][1][0].content
    assert "grt_wl" not in p2 and "routed_wl" not in p2 and "wns" not in p2


def test_full_experiment_protocol_on_mock_with_threads():
    from pnr_node.experiment import ARMS, run_experiment, summarize
    reply = json.dumps({"candidates": [{"target_density": 0.7}, {"target_density": 0.75}, {"target_density": 0.85},
                                       {"target_density": 0.65}]})
    db = ResultsDB()
    out = run_experiment(MockEvaluator(), db, "gcd", repeats=2, rounds=2, batch=4, route_top=2, workers=4,
                         provider_factory=lambda: FakeProvider(replies=[reply]), log=lambda m: None)
    names = {s["arm"] for s in out["search"]}
    assert {a for a in ARMS} == {n.split("#")[0] for n in names}
    assert all(s["routes_used"] == 4 for s in out["search"])  # equal detailed-route budget = rounds * route_top
    for n in names:
        v = out["validation"][n]
        assert v["n_ok"] == 4 and v["validated_mean"] is not None  # held-out seeds 101..104
    assert "| default_seeds |" in summarize(out) and "| llm_grt |" in summarize(out)
    # validation seeds are disjoint from the search seed, so validated results are fresh evaluations
    assert db.con.execute("SELECT COUNT(*) FROM evals WHERE arm LIKE '%|validate'").fetchone()[0] == 4 * len(names)


def test_experiment_wall_clock_cap_stops_every_arm_including_default_seeds():
    """default_seeds bypasses run_search entirely (its own evaluate_many call), so it needs its own deadline check
    -- covered separately from the run_search-level test above."""
    from pnr_node.experiment import ARMS, run_experiment

    db = ResultsDB()
    reply = json.dumps({"candidates": [{"target_density": 0.7}]})
    out = run_experiment(MockEvaluator(), db, "gcd", repeats=1, rounds=2, batch=4, route_top=2, workers=4,
                         provider_factory=lambda: FakeProvider(replies=[reply]), log=lambda m: None,
                         max_wall_s=0)  # already-expired budget
    assert out["any_stopped_early"] is True
    assert out["settings"]["max_wall_s"] == 0
    for s in out["search"]:
        assert s["stopped_early"] is True and s.get("routes_used", 0) == 0
    assert {a for a in ARMS} == {n.split("#")[0] for n in {s["arm"] for s in out["search"]}}  # every arm still ran, just capped


def test_arms_report_uses_validated_mean_and_noise_rule():
    from pnr_node.experiment import run_experiment
    from pnr_node.report import arms_report
    reply = json.dumps({"candidates": [{"target_density": 0.7}, {"target_density": 0.8}, {"target_density": 0.6}, {"target_density": 0.9}]})
    db = ResultsDB()
    res = run_experiment(MockEvaluator(), db, "gcd", repeats=2, rounds=2, batch=4, route_top=2, workers=2,
                         provider_factory=lambda: FakeProvider(replies=[reply]), log=lambda m: None)
    md = arms_report(db, res)
    assert "sigma" in md and "| random_grt |" in md and "held-out validated mean" in md and "clears 2 sigma?" in md


def test_collect_grt_falls_back_when_estimate_missing(tmp_path):
    from pnr_node.raw_handoff import RawHandoffEvaluator
    ev = RawHandoffEvaluator(str(tmp_path), "/x", native=True)
    logs = tmp_path / "w" / "logs" / "nangate45" / "aes" / "base"
    logs.mkdir(parents=True)
    (logs / "5_1_grt.json").write_text(json.dumps({"globalroute__global_route__wirelength": 324743,
                                                    "globalroute__timing__setup__ws": -0.05}))
    r = ev.collect_grt(str(tmp_path / "w"), "aes")
    assert r["grt_wl"] == 324743 and r["_evidence"]["grt_metric"] == "globalroute__global_route__wirelength"
    (logs / "5_1_grt.json").write_text(json.dumps({"globalroute__route__wirelength__estimated": 410287.0,
                                                    "globalroute__global_route__wirelength": 716661}))
    r = ev.collect_grt(str(tmp_path / "w"), "aes")
    assert r["grt_wl"] == 410287.0 and r["_evidence"]["grt_metric"].endswith("estimated")


def test_evaluator_backend_selects_librelane_without_touching_callers(tmp_path):
    import pnr_node.node as node

    node._EVALUATORS.clear()
    raw = node._evaluator({"work_root": str(tmp_path), "dp_install": "/x", "native": True})
    from pnr_node.raw_handoff import RawHandoffEvaluator
    assert isinstance(raw, RawHandoffEvaluator)

    ll = node._evaluator({"work_root": str(tmp_path), "dp_install": "/x", "backend": "librelane"})
    from pnr_node.librelane_handoff import LibreLaneEvaluator
    assert isinstance(ll, LibreLaneEvaluator) and ll.platform == "sky130hd"


def test_chia_evaluator_rejects_split_with_librelane_backend(monkeypatch):
    import pnr_node.node as node

    monkeypatch.setattr(node, "HAVE_CHIA", False)  # __init__ still calls toolchain_remote in-process; avoid a real evaluator build
    monkeypatch.setattr(node, "toolchain_remote", lambda cfg: {"tool_id": "t", "toolchain": {}})
    with pytest.raises(ValueError, match="split mode"):
        node.ChiaEvaluator({"work_root": "/x", "backend": "librelane", "split": True})


def test_vm_affinity_pins_route_to_the_vm_that_ran_grt(monkeypatch):
    import pnr_node.node as node

    calls = []

    class Handle:
        def __init__(self, res): self.res = res
        def chia_remote(self, req, cfg):
            calls.append((self.res, req["stage"]))
            vm = "cpu2" if req["stage"] == "grt" else None
            return EvalResult(True, req["stage"], grt_wl=1.0, provenance={"host": {"vm": vm or "cpu2"}}).to_json()

    class Fn:
        def options(self, resources): return Handle(resources)
        def chia_remote(self, req, cfg): return Handle(None).chia_remote(req, cfg)

    monkeypatch.setattr(node, "HAVE_CHIA", True)
    monkeypatch.setattr(node, "get", lambda x: x)
    monkeypatch.setattr(node, "evaluate_remote", Fn())
    tox = Fn()
    monkeypatch.setattr(node, "toolchain_remote", type("T", (), {"chia_remote": lambda self, cfg: {"tool_id": "t", "toolchain": {}}})())
    ev = node.ChiaEvaluator({"work_root": "/x", "vm_affinity": True})
    g = EvalRequest("gcd", stage="grt", tool_id="t")
    ev.evaluate_many([g])
    assert calls[-1] == (None, "grt")                      # grt: no constraint
    ev.evaluate_many([EvalRequest("gcd", stage="route", tool_id="t")])
    assert calls[-1] == ({"pnr": 1, "vm_cpu2": 0.001}, "route") and ev.affinity_hits == 1
    ev.evaluate_many([EvalRequest("gcd", stage="route", tool_id="t", seed=9)])  # different identity: unconstrained
    assert calls[-1] == (None, "route")


def test_repair_reruns_only_infra_failed_arm_runs():
    from pnr_node.experiment import infra_failed_arm_runs, repair, run_experiment

    class Flaky(MockEvaluator):
        """First call for one specific identity fails with an infrastructure error, like a worker without a GPU."""
        def __init__(self): super().__init__(); self.broken = True
        def evaluate(self, req):
            if self.broken and req.stage == "proxy" and req.placer_cfg["target_density"] == 0.7:
                return EvalResult(False, "proxy", error="dreamplace: RuntimeError: Found no NVIDIA driver on your system")
            return super().evaluate(req)

    reply = json.dumps({"candidates": [{"target_density": 0.7}, {"target_density": 0.75}, {"target_density": 0.85}, {"target_density": 0.65}]})
    db, ev = ResultsDB(), Flaky()
    kw = dict(repeats=2, rounds=2, batch=4, route_top=2, workers=2, provider_factory=lambda: FakeProvider(replies=[reply]), log=lambda m: None)
    first = run_experiment(ev, db, "gcd", arms=("default_seeds", "llm_grt"), **kw)
    bad = infra_failed_arm_runs(db)
    assert bad and all(b.startswith("llm_grt") for b in bad)  # the default arm never proposed that config
    ev.broken = False
    fixed = repair(ev, db, "gcd", first, **{k: v for k, v in kw.items() if k != "log"})
    assert infra_failed_arm_runs(db) == []
    assert set(fixed["repaired"]) == set(bad)
    assert {s["arm"] for s in fixed["search"]} == {s["arm"] for s in first["search"]}
    assert len(fixed["search"]) == len(first["search"])


def test_batch_evaluator_places_once_then_routes_in_parallel():
    """Fake inner evaluator: 'proxy' calls are fast/serial, 'route' calls are the slow ones we want overlapped."""
    import threading
    import time

    from pnr_node.batch import BatchEvaluator

    calls = []
    lock = threading.Lock()
    max_concurrent = [0]
    concurrent = [0]

    class FakeInner:
        tool_id = "fake"
        toolchain = {"x": 1}

        def evaluate(self, req):
            with lock:
                calls.append((req.stage, req.placer_cfg["target_density"]))
                concurrent[0] += 1
                max_concurrent[0] = max(max_concurrent[0], concurrent[0])
            if req.stage == "route":
                time.sleep(0.05)  # simulate the long CPU-bound phase
            with lock:
                concurrent[0] -= 1
            if req.stage == "proxy":
                return EvalResult(True, "proxy", proxy_hpwl=100.0)
            return EvalResult(True, "route", routed_wl=200.0)

    be = BatchEvaluator(inner=FakeInner(), route_workers=4)
    assert be.tool_id == "fake" and be.toolchain == {"x": 1}
    reqs = [EvalRequest("gcd", placer_cfg={"target_density": 0.6 + 0.05 * i}, stage="route") for i in range(4)]
    t0 = time.time()
    out = be.evaluate_many(reqs)
    elapsed = time.time() - t0
    assert all(r.ok and r.routed_wl == 200.0 for r in out)
    # every request was placed (stage=proxy) before any of them was routed
    place_calls = [c for c in calls if c[0] == "proxy"]
    route_calls = [c for c in calls if c[0] == "route"]
    assert len(place_calls) == 4 and len(route_calls) == 4
    assert calls.index(place_calls[-1]) < calls.index(route_calls[0])
    # routes actually overlapped (not serialized): with 4x 0.05s routes truly parallel, wall time << 4*0.05s
    assert elapsed < 0.15
    assert max_concurrent[0] >= 2  # real overlap happened, not accidental serialization


def test_batch_evaluator_never_routes_a_failed_placement():
    from pnr_node.batch import BatchEvaluator

    routed_densities = []

    class FakeInner:
        tool_id, toolchain = "fake", {}

        def evaluate(self, req):
            if req.stage == "proxy":
                bad = req.placer_cfg["target_density"] > 0.8
                return EvalResult(not bad, "proxy", proxy_hpwl=1.0, error="rejected" if bad else "")
            routed_densities.append(req.placer_cfg["target_density"])  # only the accepted one should ever land here
            return EvalResult(True, "route", routed_wl=200.0)

    be = BatchEvaluator(inner=FakeInner())
    reqs = [EvalRequest("gcd", placer_cfg={"target_density": 0.9}, stage="route"),
            EvalRequest("gcd", placer_cfg={"target_density": 0.65}, stage="route")]
    out = be.evaluate_many(reqs)
    assert out[0].ok is False and out[0].error == "rejected"
    assert out[1].ok is True and out[1].routed_wl == 200.0
    assert routed_densities == [0.65]  # the rejected 0.9 config never reached the route call


def test_batch_evaluator_proxy_only_requests_skip_routing_entirely():
    from pnr_node.batch import BatchEvaluator

    routed = []

    class FakeInner:
        tool_id, toolchain = "fake", {}

        def evaluate(self, req):
            if req.stage == "route":
                routed.append(req)
            return EvalResult(True, req.stage, proxy_hpwl=1.0, routed_wl=2.0 if req.stage == "route" else None)

    be = BatchEvaluator(inner=FakeInner())
    out = be.evaluate_many([EvalRequest("gcd", stage="proxy"), EvalRequest("gcd", stage="proxy")])
    assert all(o.stage == "proxy" for o in out) and routed == []


def test_batch_evaluator_is_a_drop_in_for_run_search():
    """Same interface as ChiaEvaluator: search.run_search must work unmodified with a BatchEvaluator."""
    from pnr_node.batch import BatchEvaluator

    be = BatchEvaluator(inner=MockEvaluator())
    db = ResultsDB()
    out = run_search(be, RandomProposer(0), db, "batch", "gcd", rounds=1, batch=4, route_top=2)
    assert out["routes_used"] == 2 and out["best_score"] is not None


def test_batch_evaluator_raises_without_work_root_or_inner():
    from pnr_node.batch import BatchEvaluator

    import pytest as _pytest
    with _pytest.raises(ValueError):
        BatchEvaluator()


def test_dp_depth_is_part_of_evaluator_identity(tmp_path):
    """global vs global+legal vs global+legal+detail must never share a cache/identity -- otherwise a second mode
    silently reuses the first's placement DEF and looks 'identical'. dp_depth is folded into the toolchain
    fingerprint (and hence tool_id and the request key) exactly so that cannot happen."""
    from pnr_node.raw_handoff import RawHandoffEvaluator

    a = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, dp_depth="global")
    b = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, dp_depth="global+legal")
    assert a.dp_depth == "global" and b.dp_depth == "global+legal"
    assert a.toolchain["dp_depth"] == "global" and b.toolchain["dp_depth"] == "global+legal"
    assert a.tool_id != b.tool_id
    assert EvalRequest("tinyRocket", tool_id=a.tool_id).key() != EvalRequest("tinyRocket", tool_id=b.tool_id).key()


def test_expanded_context_and_knob_space_vs_core_baseline():
    """The llm_grt_expanded proposer tunes the full KNOBS space and sees tns/power/area + DREAMPlace convergence;
    the core baseline sees neither. This is the only difference the A/B isolates."""
    reply = json.dumps({"candidates": [{"target_density": 0.7}, {"target_density": 0.9}, {"target_density": 0.8}]})

    db, ev = ResultsDB(), MockEvaluator()
    prov = FakeProvider(replies=[reply, reply])
    prop = LLMProposer(prov, db, arm="exp", show_routed=True, knobs=KNOBS, expanded_context=True)
    run_search(ev, prop, db, "exp", "gcd", rounds=2, batch=3, route_top=1, grt_top=3)
    p = prov.calls[1][1][0].content
    assert "num_bins" in p and "iteration" in p  # expanded knob table offered to the LLM
    assert '"tns"' in p and '"power"' in p and '"area"' in p and "dp_overflow" in p  # fuller routed context

    db2, ev2 = ResultsDB(), MockEvaluator()
    prov2 = FakeProvider(replies=[reply, reply])
    prop2 = LLMProposer(prov2, db2, arm="base", show_routed=True, knobs=CORE_KNOBS, expanded_context=False)
    run_search(ev2, prop2, db2, "base", "gcd", rounds=2, batch=3, route_top=1, grt_top=3)
    p2 = prov2.calls[1][1][0].content
    assert "routed_wl" in p2  # baseline still sees the core routed context ...
    assert '"tns"' not in p2 and '"power"' not in p2 and "dp_overflow" not in p2  # ... but not the expanded fields
    assert "num_bins" not in p2 and "iteration" not in p2  # nor the expanded knobs, in table or history


def test_random_proposer_respects_knob_subset():
    from pnr_node.search import RandomProposer

    core = RandomProposer(0, knobs=CORE_KNOBS).propose([], 5)
    assert all(set(c) == set(CORE_KNOBS) for c in core)
    full = RandomProposer(0, knobs=KNOBS).propose([], 5)
    assert all(set(c) == set(KNOBS) for c in full)


# --- split place/route pin to the same VM on a multi-node cluster ---

def _split_ev(monkeypatch, nodes):
    """A ChiaEvaluator in split mode with ray.nodes() mocked to `nodes` and place/route replaced by handles that
    record the resources they were dispatched with."""
    import pnr_node.node as node

    calls = {"place": [], "route": []}

    class Handle:
        def __init__(self, kind, res): self.kind, self.res = kind, res
        def chia_remote(self, req, cfg, *rest):
            calls[self.kind].append((self.res, req["stage"]))
            # get() is identity below, so return exactly what the caller unpacks: route dispatch is read back as an
            # EvalResult json; the place dispatch's return is only forwarded to route as a dependency, never read.
            return EvalResult(True, "route", routed_wl=1.0).to_json() if self.kind == "route" else "place-ref"

    class Fn:
        def __init__(self, kind): self.kind = kind
        def options(self, resources): return Handle(self.kind, resources)
        def chia_remote(self, req, cfg, *rest): return Handle(self.kind, None).chia_remote(req, cfg, *rest)

    monkeypatch.setattr(node, "HAVE_CHIA", True)
    monkeypatch.setattr(node, "_ray_nodes", lambda: nodes)
    monkeypatch.setattr(node, "place_remote", Fn("place"))
    monkeypatch.setattr(node, "route_remote", Fn("route"))
    monkeypatch.setattr(node, "get", lambda h: h)  # no real Ray futures here
    monkeypatch.setattr(node, "toolchain_remote",
                        type("T", (), {"chia_remote": lambda self, cfg: {"tool_id": "t", "toolchain": {}}})())
    ev = node.ChiaEvaluator({"work_root": "/x", "split": True})
    return ev, calls


def test_split_pins_place_and_route_to_the_same_vm_deterministically(monkeypatch):
    nodes = [{"Alive": True, "Resources": {"pnr": 1, "GPU": 1, "vm_a": 1}},
             {"Alive": True, "Resources": {"pnr": 1, "GPU": 1, "vm_b": 1}}]
    ev, calls = _split_ev(monkeypatch, nodes)
    r = EvalRequest("gcd", stage="route", tool_id="t", seed=5)
    ev.evaluate_many([r])
    place_res, route_res = calls["place"][-1][0], calls["route"][-1][0]
    # place and route of one candidate land on the SAME vm, and that vm carries the pnr+vm resource pin
    assert place_res == route_res
    assert place_res["pnr"] == 1 and any(k.startswith("vm_") for k in place_res)
    # deterministic: the identical request maps to the identical vm on a second dispatch
    ev.evaluate_many([EvalRequest("gcd", stage="route", tool_id="t", seed=5)])
    assert calls["place"][-1][0] == place_res


def test_split_maps_proxy_grt_route_of_one_candidate_to_one_vm(monkeypatch):
    nodes = [{"Alive": True, "Resources": {"pnr": 1, "GPU": 1, "vm_a": 1}},
             {"Alive": True, "Resources": {"pnr": 1, "GPU": 1, "vm_b": 1}}]
    ev, _ = _split_ev(monkeypatch, nodes)
    vms = {ev._split_vm(EvalRequest("gcd", stage=s, tool_id="t", seed=5)) for s in ("proxy", "grt", "route")}
    assert len(vms) == 1 and None not in vms  # stage does not change the VM (mapping is on the grt identity)


def test_split_slots_are_weighted_by_each_nodes_pnr_count(monkeypatch):
    # vm_a advertises pnr=3, vm_b pnr=1 -> a appears 3x as often in the slot table as b
    nodes = [{"Alive": True, "Resources": {"pnr": 3, "GPU": 1, "vm_a": 1}},
             {"Alive": True, "Resources": {"pnr": 1, "GPU": 1, "vm_b": 1}},
             {"Alive": False, "Resources": {"pnr": 9, "GPU": 1, "vm_dead": 1}}]  # dead node contributes nothing
    ev, _ = _split_ev(monkeypatch, nodes)
    assert sorted(ev._slots) == ["a", "a", "a", "b"]
    assert "dead" not in ev._slots


def test_split_falls_back_to_unpinned_when_no_vm_resources(monkeypatch):
    # a lone GPU head with no vm_* resource -> no slots -> place/route dispatched unpinned (None), no crash
    ev, calls = _split_ev(monkeypatch, [{"Alive": True, "Resources": {"pnr": 1, "GPU": 1}}])
    assert ev._slots == [] and ev._split_vm(EvalRequest("gcd", stage="route", tool_id="t")) is None
    ev.evaluate_many([EvalRequest("gcd", stage="route", tool_id="t")])
    assert calls["place"][-1][0] is None and calls["route"][-1][0] is None


# --- identical identities in one batch dispatch exactly once ---

def test_chia_evaluate_many_dispatches_duplicate_identity_once(monkeypatch):
    import pnr_node.node as node

    dispatched = []

    class Fake:
        tool_id, toolchain = "t", {}
        def evaluate(self, req):
            dispatched.append(req.key())
            return EvalResult(True, req.stage, proxy_hpwl=1.0)

    monkeypatch.setattr(node, "_evaluator", lambda cfg: Fake())
    ce = node.ChiaEvaluator({"work_root": "/x"})
    r1 = EvalRequest("gcd", stage="proxy", tool_id="t")
    r2 = EvalRequest("gcd", stage="proxy", tool_id="t")   # identical identity
    r3 = EvalRequest("gcd", stage="proxy", tool_id="t", seed=1)  # distinct
    out = ce.evaluate_many([r1, r2, r3])
    assert len(dispatched) == 2                          # r1/r2 collapsed to one dispatch, r3 its own
    assert len(out) == 3 and all(o.ok for o in out)
    assert out[0].proxy_hpwl == out[1].proxy_hpwl == 1.0  # both positions got the one shared result


def test_cached_evaluate_many_dispatches_duplicate_identity_once():
    from pnr_node.db import CachedEvaluator

    class Counting(MockEvaluator):
        seen = 0
        def evaluate_many(self, reqs):
            self.seen += len(reqs)
            return [self.evaluate(r) for r in reqs]

    db, ev = ResultsDB(), Counting()
    cev = CachedEvaluator(ev, db)
    reqs = [EvalRequest("gcd", stage="proxy", tool_id="mock"),
            EvalRequest("gcd", stage="proxy", tool_id="mock"),   # duplicate of the first
            EvalRequest("gcd", stage="proxy", tool_id="mock", seed=2)]
    out = cev.evaluate_many(reqs)
    assert ev.seen == 2                                  # only the two distinct identities reached the inner eval
    assert len(out) == 3 and all(o.ok for o in out)


# --- the GPU arch is part of the toolchain identity ---

def test_gpu_arch_is_part_of_toolchain_identity(tmp_path, monkeypatch):
    """sm_86 (a laptop 3060) and sm_89 (an L4) place slightly differently; routing amplifies it, so their results
    must never share a cache key. The compute capability is folded into the toolchain fingerprint when gpu=True."""
    from pnr_node.raw_handoff import RawHandoffEvaluator

    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    a = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, gpu=True)
    assert a.toolchain["gpu_arch"] == "8.6"

    monkeypatch.setenv("PNR_GPU_ARCH", "8.9")
    b = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, gpu=True)
    assert b.toolchain["gpu_arch"] == "8.9"
    assert a.tool_id != b.tool_id                         # different GPU arch => different identity

    c = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, gpu=False)
    assert "gpu_arch" not in c.toolchain                  # cpu runs carry no gpu_arch (device already distinguishes)


def test_gpu_arch_detection_never_raises_and_defaults_to_unknown(tmp_path, monkeypatch):
    """With no override, and neither torch nor nvidia-smi reachable (subprocess raises FileNotFoundError),
    detection degrades to 'unknown' rather than breaking the toolchain."""
    import pnr_node.raw_handoff as rh

    monkeypatch.delenv("PNR_GPU_ARCH", raising=False)

    def boom(*a, **k):
        raise FileNotFoundError("no such tool")

    monkeypatch.setattr(rh.subprocess, "run", boom)
    ev = rh.RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, gpu=True)
    assert ev._gpu_arch() == "unknown"


def test_dp_build_arch_recorded_when_install_declares_it(tmp_path, monkeypatch):
    """If the DREAMPlace install records the CUDA arch it was compiled for (docker/run_local.sh writes CUDA_ARCH),
    that too is part of the identity -- a binary built for sm_86 must not be confused with one built for sm_89."""
    from pnr_node.raw_handoff import RawHandoffEvaluator

    monkeypatch.setenv("PNR_GPU_ARCH", "8.6")
    (tmp_path / "CUDA_ARCH").write_text("8.6\n")
    ev = RawHandoffEvaluator(str(tmp_path), str(tmp_path), native=True, gpu=True)
    assert ev.toolchain["dp_build_arch"] == "8.6"
