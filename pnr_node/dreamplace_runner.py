"""Run DREAMPlace once and return its placed DEF plus the proxy cost.

Kept free of CHIA/OpenROAD so it can run on the host, in a container, or behind
any evaluator. Knob names follow `contract.KNOBS`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

_ITER = re.compile(r"iteration\s+(\d+),.*?wHPWL\s+([0-9.eE+-]+),\s+Overflow\s+([0-9.eE+-]+)")
# "non-linear placement takes": GP (+ LG + ABCDPlace when enabled). An unanchored "placement takes" matched
# "detailed placement takes" first once detailed placement was on, recording only ABCDPlace's time.
_TOOK = re.compile(r"non-linear placement takes\s+([0-9.]+)\s+seconds")


@dataclass
class DPResult:
    ok: bool
    def_path: str = ""
    hpwl: Optional[float] = None  # DREAMPlace's final weighted HPWL (the cheap proxy)
    overflow: Optional[float] = None
    iterations: Optional[int] = None
    runtime_s: float = 0.0
    error: str = ""


def dp_env(install: str) -> Dict[str, str]:
    """Environment for the DREAMPlace subprocess.

    Ray sets CUDA_VISIBLE_DEVICES="" for tasks that did not request a GPU, and the child inherits it, so
    DREAMPlace saw no GPU inside a CHIA worker. An empty value means "hidden by the scheduler", not a choice.
    """
    env = dict(os.environ, PYTHONPATH=install)
    if env.get("CUDA_VISIBLE_DEVICES") in ("", "NoDevFiles"):
        env.pop("CUDA_VISIBLE_DEVICES")
    return env


# dp_depth -> (legalize_flag, detailed_place_flag). "global" (default): DREAMPlace does global placement only and
# OpenROAD's OpenDP legalizes + detail-places downstream. The other modes let DREAMPlace do those stages itself
# before the handoff; OpenROAD still re-runs them (measured in results/tinyrocket/local_pair/compare.md).
_DP_DEPTH = {"global": (0, 0), "global+legal": (1, 0), "global+legal+detail": (1, 1)}


def build_config(lef_files: List[str], def_in: str, out_dir: str, cfg: Dict[str, float], seed: int,
                 gpu: bool = True, timing_driven: bool = False,
                 lib_input: Optional[List[str]] = None, sdc_input: Optional[str] = None,
                 wire_resistance_per_micron: Optional[float] = None,
                 wire_capacitance_per_micron: Optional[float] = None,
                 dp_depth: str = "global") -> Dict:
    nb = int(cfg.get("num_bins", 0))
    nb = 0 if nb < 32 else nb  # below 32 makes no sense; treat as auto (DREAMPlace picks the grid)
    gp_iter = int(cfg.get("iteration", 1000))
    llambda = int(cfg.get("Llambda_density_weight_iteration", 1))
    lsub = int(cfg.get("Lsub_iteration", 1))
    if dp_depth not in _DP_DEPTH:
        raise ValueError(f"dp_depth must be one of {sorted(_DP_DEPTH)}, got {dp_depth!r}")
    legalize, detail = _DP_DEPTH[dp_depth]
    conf = {
        "lef_input": lef_files,
        "def_input": def_in,
        "result_dir": out_dir,
        "gpu": 1 if gpu else 0,
        "num_threads": 8,
        "global_place_flag": 1,
        "legalize_flag": legalize,  # "global": OpenROAD's OpenDP legalizes; higher dp_depth: DREAMPlace does too
        "detailed_place_flag": detail,  # DREAMPlace's key; the old "detail_place_flag" was silently ignored
        "plot_flag": 0,
        "enable_fillers": 0,
        "ignore_net_degree": 100,
        "random_seed": int(seed),
        "target_density": cfg["target_density"],
        "density_weight": cfg["density_weight"],
        "gamma": cfg["gamma"],
        "stop_overflow": cfg["stop_overflow"],
        "num_bins_x": nb,
        "num_bins_y": nb,
        "routability_opt_flag": int(cfg.get("routability_opt", 0)),
        "global_place_stages": [{
            "num_bins_x": nb, "num_bins_y": nb, "iteration": gp_iter,
            "learning_rate": cfg["learning_rate"], "wirelength": "weighted_average",
            "optimizer": "nesterov", "Llambda_density_weight_iteration": llambda, "Lsub_iteration": lsub,
        }],
    }
    if timing_driven:
        # NOT functional in this flow (paper §6). HeteroSTA rather than OpenTimer because OpenTimer needs a
        # `verilog_input` netlist this flow does not produce; HeteroSTA also fails in this build (its timing graph
        # collapses even on upstream's superblue18). Needs libcudart.so.11.0 and a `HeteroSTA_Lic` license.
        conf["timing_opt_flag"] = 1
        conf["timer_engine"] = "heterosta"
        if lib_input:
            conf["lib_input"] = list(lib_input) if isinstance(lib_input, (list, tuple)) else [lib_input]
        if sdc_input:
            conf["sdc_input"] = sdc_input
        if wire_resistance_per_micron is not None:
            conf["wire_resistance_per_micron"] = wire_resistance_per_micron
        if wire_capacitance_per_micron is not None:
            conf["wire_capacitance_per_micron"] = wire_capacitance_per_micron
    return conf


def parse_log(text: str) -> Dict[str, Optional[float]]:
    last = None
    for m in _ITER.finditer(text):
        last = m
    took = _TOOK.search(text)
    return {
        "iterations": int(last.group(1)) + 1 if last else None,
        "hpwl": float(last.group(2)) if last else None,
        "overflow": float(last.group(3)) if last else None,
        "runtime_s": float(took.group(1)) if took else None,
    }


def run_dreamplace(lef_files: List[str], def_in: str, out_dir: str, cfg: Dict[str, float], seed: int, *,
                   install: str, python: str = "/usr/bin/python3", gpu: bool = True,
                   timeout: int = 1800, timing_driven: bool = False,
                   lib_input: Optional[List[str]] = None, sdc_input: Optional[str] = None,
                   wire_resistance_per_micron: Optional[float] = None,
                   wire_capacitance_per_micron: Optional[float] = None,
                   dp_depth: str = "global") -> DPResult:
    os.makedirs(out_dir, exist_ok=True)
    conf = build_config(lef_files, def_in, out_dir, cfg, seed, gpu,
                        timing_driven=timing_driven, lib_input=lib_input,
                        sdc_input=sdc_input,
                        wire_resistance_per_micron=wire_resistance_per_micron,
                        wire_capacitance_per_micron=wire_capacitance_per_micron,
                        dp_depth=dp_depth)
    conf_path = os.path.join(out_dir, "dreamplace.json")
    with open(conf_path, "w") as f:
        json.dump(conf, f, indent=2)
    env = dp_env(install)
    t0 = time.time()
    try:
        # cwd=install: DREAMPlace opens some data files by relative path (e.g. the FLUTE lookup table
        # thirdparty/flute/lut.ICCAD2015/POWV9.dat), as its README runs it from the installation directory.
        p = subprocess.run([python, os.path.join(install, "dreamplace", "Placer.py"), conf_path], env=env,
                           capture_output=True, text=True, timeout=timeout, cwd=install)
    except subprocess.TimeoutExpired:
        return DPResult(False, error=f"timeout after {timeout}s", runtime_s=time.time() - t0)
    log = p.stdout + p.stderr
    with open(os.path.join(out_dir, "dreamplace.log"), "w") as f:
        f.write(log)
    info = parse_log(log)
    # DREAMPlace (commit 6627f33) writes a single <result_dir>/<design>/<design>.gp.def at the very end, i.e. AFTER
    # legalization and detailed placement when those are enabled (verified on tinyRocket: "detailed placement takes"
    # precedes "writing to ...gp.def"). Other versions may emit .lg.def/.dp.def, so prefer the deepest if present.
    def _find(suffix: str) -> List[str]:
        return [os.path.join(r, n) for r, _, ns in os.walk(out_dir) for n in ns if n.endswith(suffix)]
    _legalize, _detail = _DP_DEPTH.get(dp_depth, (0, 0))
    defs = []
    if _detail:
        defs = _find(".dp.def")
    if not defs and _legalize:
        defs = _find(".lg.def")
    if not defs:
        defs = _find(".gp.def")
    if p.returncode != 0 or not defs or info["hpwl"] is None:
        return DPResult(False, error=(log.strip().splitlines() or ["no output"])[-1][:300],
                        runtime_s=time.time() - t0)
    return DPResult(True, defs[0], info["hpwl"], info["overflow"], info["iterations"],
                    info["runtime_s"] or (time.time() - t0))


_DIE = re.compile(r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)")
_COMP = re.compile(r"\+\s*(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)")


def check_placement(def_path: str, overflow: Optional[float], stop_overflow: float, tol: float = 1.05) -> List[str]:
    """Reasons a DREAMPlace result should not be sent to a route (empty list = fine).

    Mechanical and conservative: only fires on a placement that cannot be a valid starting point, so a
    rejection costs nothing and a pass never claims the placement is good.
    """
    reasons: List[str] = []
    if overflow is None or overflow > stop_overflow * tol:
        reasons.append(f"not_converged(overflow={overflow}, stop_overflow={stop_overflow})")
    die = None
    outside = n = 0
    in_comp = False
    with open(def_path) as f:
        for line in f:
            if die is None:
                m = _DIE.search(line)
                if m:
                    die = tuple(int(g) for g in m.groups())
            if line.startswith("COMPONENTS"):
                in_comp = True
            elif line.startswith("END COMPONENTS"):
                break
            elif in_comp:
                m = _COMP.search(line)
                if m and die:
                    n += 1
                    x, y = int(m.group(1)), int(m.group(2))
                    if x < die[0] or y < die[1] or x > die[2] or y > die[3]:
                        outside += 1
    if die is None:
        reasons.append("no_diearea_in_def")
    if n == 0:
        reasons.append("no_placed_components")
    if outside:
        reasons.append(f"cells_outside_die({outside}/{n})")
    return reasons
