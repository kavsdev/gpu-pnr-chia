"""Reference evaluator: DREAMPlace global placement spliced into stock ORFS/OpenROAD.

How the splice works (the "DEF handoff"):
  * ORFS runs unchanged through stage 3_2 (floorplan, PDN, IO placement).
  * Stage 3_3 (global_place.tcl) is replaced, using a per-run copy of the ORFS scripts directory passed as
    `make SCRIPTS_DIR=...`, by a copy in which the single `do_placement` call is swapped for:
        - dump mode  (no <work>/dp/gp.def): write_def <work>/dp/pre_gp.def   (after buffer_ports)
        - inject mode (gp.def present):     read_def -incremental <work>/dp/gp.def
  * DREAMPlace runs on the dumped DEF; ORFS then continues with resize, OpenDP legalization, CTS, route, report.
So OpenROAD never places the cells itself; every stage after 3_3 is stock ORFS.

Two execution modes with identical behaviour:
  docker  the host runs `docker run openroad/orfs` per ORFS call (dev VM)
  native  ORFS/OpenROAD are on the PATH of the current machine (inside the CHIA worker container)

timing_driven=True: intended to feed DREAMPlace's timing-driven mode (`timing_opt_flag`) the
design's .lib/.sdc/wire-RC data. NOT functional in this flow: the timer never receives the design netlist, so
placement stays wirelength/density-driven (see the paper, §6). Stock ORFS's RePlAce stage is timing-driven, which
is the likely cause of the setup-timing gap against stock.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from typing import Dict, List, Optional

from .contract import FLOW_DP_KNOBS, FLOW_KNOBS, EvalRequest, EvalResult, clamp_config
from dataclasses import asdict, replace

from . import netwt
from .dreamplace_runner import DPResult, check_placement, run_dreamplace
from .evidence import write_bundle

_FLOW = "/OpenROAD-flow-scripts/flow"
_PATCH_OLD = "set result [catch { do_placement $global_placement_args } errMsg]"
# Re-buffer resume (scheme odb-v1): a 3_3 run whose RESULTS_DIR holds 3_3_rebuf_in.odb (the session saved right after
# the previous kept repair) loads that instead of 3_2_place_iop.odb and skips remove_buffers/buffer_ports, which that
# state already went through. Everything else of the prelude (SDC, derate, RC via load_design; PRE hook; dont_use)
# is the same as stock. _PRELUDE_OLD is the exact stock text, so an upstream change fails loudly.
_PRELUDE_OLD = """load_design 3_2_place_iop.odb 2_floorplan.sdc
source_step_tcl PRE GLOBAL_PLACE

set_dont_use $::env(DONT_USE_CELLS)

if { $::env(GPL_TIMING_DRIVEN) } {
  remove_buffers
}

# Do not buffer chip-level designs
# by default, IO ports will be buffered
# to not buffer IO ports, set environment variable
# DONT_BUFFER_PORT = 1
if { ![env_var_exists_and_non_empty FOOTPRINT] } {
  if { !$::env(DONT_BUFFER_PORTS) } {
    puts "Perform port buffering..."
    buffer_ports {*}[env_var_or_empty BUFFER_PORTS_ARGS]
  }
}
"""
_REBUF_IN = "3_3_rebuf_in.odb"  # in RESULTS_DIR: load_design resolves its file argument there
_PRELUDE_NEW = """if { [file exists $::env(RESULTS_DIR)/""" + _REBUF_IN + """] } {
  # re-buffer resume: state after the previous kept repair (already port-buffered, remove_buffers already applied)
  load_design """ + _REBUF_IN + """ 2_floorplan.sdc
  source_step_tcl PRE GLOBAL_PLACE
  set_dont_use $::env(DONT_USE_CELLS)
} else {
""" + _PRELUDE_OLD + "}\n"
_PATCH_NEW = """set dp_def @W@/dp/gp.def
# re-buffer mode: kept repairs on DREAMPlace placements, as RePlAce's timing-driven mode
# does mid-placement. A stage run (rebuf_from.def present, no gp.def) overlays that stage's GP placement on the loaded
# state (3_2 for stage 1, the previous stage's saved session after), runs ONE repair_design (+ fully_rebuffer), and
# saves the session (rebuf_out.odb, resumed by the next stage / the import run) and the netlist DEF for the next GP.
# With rebuf_sta it also dumps per-net slacks of the repaired netlist (gpl net weights) from this session.
if { [file exists @W@/dp/rebuf_from.def] && ![file exists $dp_def] } {
  log_cmd read_def -incremental @W@/dp/rebuf_from.def
  if { [file exists @W@/dp/rebuf_presplit] } {
    # stage 1 only: pre-split very-high-fanout signal nets (ariane133's 20k-load reset) so repair_design's fanout
    # repair does not spend minutes on one net (rebuf_presplit.tcl)
    source @W@/dp/rebuf_presplit.tcl
    presplit_heavy_nets 2000 1000 BUF_X4
  }
  log_cmd estimate_parasitics -placement
  repair_design_helper
  if { [file exists @W@/dp/rebuf_full] } {
    # what gpl's kept timing-driven iteration does after repairDesign (Resizer::findResizeSlacks -> fullyRebuffer)
    log_cmd rsz::fully_rebuffer NULL
  }
  log_cmd write_db @W@/dp/rebuf_out.odb
  log_cmd write_def @W@/dp/rebuf.def
  if { [file exists @W@/dp/rebuf_sta] } {
    log_cmd estimate_parasitics -placement
    set ::env(NW_OUT) @W@/dp/rebuf.slack
    set ::env(NW_IGNORE_DEG) 100
    source @W@/dp/net_slack_dump.tcl
  }
  # stage run done: skip the rest of the step (estimate_parasitics, report_metrics, write_db 3_3_place_gp.odb; ~27 s
  # on ariane133). This 3_3 copy is thrown away: the next stage and the import run resume from rebuf_out.odb, and
  # _place_rebuf reads only dp/ outputs and the 3_3 log. make does not check that its target was written.
  # Stage outputs are byte-identical with and without this (tinyRocket, ariane133), so the rebuffer scheme is unchanged.
  exit
}
set result 0
set errMsg ""
if { [file exists $dp_def] } {
  log_cmd read_def -incremental $dp_def
  if { [file exists @W@/dp/rp_finish] } {
    # hybrid (rp_finish): DREAMPlace stopped early; RePlAce finishes from that placement with ORFS's own args, so its
    # timing-driven repairs/reweighting and routability inflation run. `-incremental` would be a no-op here (every
    # instance is PLACED -> gpl GPL-0156 returns), so Nesterov starts from the odb locations via -skip_initial_place
    # (and without -force_center_initial_place). The marker's content is extra global_placement args.
    set rp_fh [open @W@/dp/rp_finish]
    set rp_extra [string trim [read $rp_fh]]
    close $rp_fh
    set rp_args [lsearch -all -inline -not -exact $global_placement_args -force_center_initial_place]
    lappend rp_args -skip_initial_place {*}$rp_extra
    set result [catch { do_placement $rp_args } errMsg]
  }
} elseif { ![file exists @W@/dp/rebuf_from.def] } {
  file mkdir @W@/dp
  log_cmd write_def @W@/dp/pre_gp.def
}
"""


class HandoffError(RuntimeError):
    pass


# The final flow as RawHandoffEvaluator kwargs. Equal to the CLI
# `--rebuffer --rebuf-full --rebuf-overflow 0.6,0.2 --timing-driven-netwt static --netwt-scheme gpl --rebuf-presplit
# --dp-extra '{"target_density":0.45}'`, so flow_kwargs({}) has the same toolchain/tool_id as those cached runs.
FINAL_FLOW_KWARGS: Dict = {
    "rebuffer": True, "rebuf_full": True, "rebuf_overflow": [0.6, 0.2], "rebuf_presplit": True,
    "net_weight": "static", "netwt_scheme": "gpl", "netwt_pct": netwt.GPL_NETS_PCT,
    "netwt_gpl_max": netwt.GPL_WEIGHT_MAX, "dp_routability": False, "dp_extra": {"target_density": 0.45},
}


def flow_kwargs(flow_cfg: Optional[Dict]) -> tuple:
    """FLOW_KNOBS dict -> (RawHandoffEvaluator kwargs, DREAMPlace placer_cfg overrides). The dict is clamped (missing
    knobs = the final flow's defaults); evaluator kwargs are FINAL_FLOW_KWARGS with the flow knobs applied, so each
    distinct flow config gets its own toolchain/tool_id (DREAMPlace knobs go into placer_cfg, i.e. the request key)."""
    f = clamp_config(flow_cfg or {}, FLOW_KNOBS)
    kw = json.loads(json.dumps(FINAL_FLOW_KWARGS))  # deep copy
    kw["dp_extra"]["target_density"] = float(f["target_density"])
    kw["rebuf_overflow"] = _stops(f["rebuf_stops"])
    kw["netwt_pct"] = float(f["netwt_pct"])
    kw["netwt_gpl_max"] = float(f["netwt_max"])
    kw["rebuf_full"] = bool(f["rebuf_full"])
    kw["dp_routability"] = bool(f["dp_routability"])
    return kw, {k: float(f[k]) for k in FLOW_DP_KNOBS}


def apply_flow(req: EvalRequest) -> EvalRequest:
    """A request whose placer_cfg carries its flow_cfg's DREAMPlace knobs (idempotent; {} flow_cfg = unchanged)."""
    if not req.flow_cfg:
        return req
    cfg = dict(req.placer_cfg)
    cfg.update(flow_kwargs(req.flow_cfg)[1])
    return replace(req, placer_cfg=cfg) if cfg != req.placer_cfg else req


@contextlib.contextmanager
def flock(path: str):
    """Exclusive inter-process lock. Parallel workers share the on-disk base run; the first one builds it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def components_only_def(src: str, dst: str) -> int:
    """Keep the header and the COMPONENTS section of a placed DEF, drop PINS/NETS/rows/tracks.

    `read_def -incremental` re-reads PINS, which appends a second shape to every existing pin and makes
    detailed routing abort with DRT-0302 ("multiple pins on bterm"). Placement is all we want to import.
    Returns the number of components written.
    """
    out, in_comp, n = [], False, 0
    with open(src) as f:
        for line in f:
            tok = line.split(None, 1)[0] if line.strip() else ""
            if tok in ("VERSION", "DIVIDERCHAR", "BUSBITCHARS", "DESIGN", "UNITS", "DIEAREA"):
                out.append(line)
            elif tok == "COMPONENTS":
                in_comp = True
                out.append(line)
            elif in_comp:
                out.append(line)
                if line.lstrip().startswith("- "):
                    n += 1
                if line.startswith("END COMPONENTS"):
                    in_comp = False
    out.append("END DESIGN\n")
    with open(dst, "w") as f:
        f.writelines(out)
    return n


# Nangate45 routing resources as ORFS's global router sees them: LEF track pitch per layer (metal2..metal10, the
# signal layers) and platforms/nangate45/fastroute.tcl's capacity reductions (metal2-3 0.5, metal4+ 0.25).
_NG45_LAYERS = {"metal2": ("V", 0.19, 0.5), "metal3": ("H", 0.14, 0.5), "metal4": ("V", 0.28, 0.25),
                "metal5": ("H", 0.28, 0.25), "metal6": ("V", 0.28, 0.25), "metal7": ("H", 0.8, 0.25),
                "metal8": ("V", 0.8, 0.25), "metal9": ("H", 1.6, 0.25), "metal10": ("V", 1.6, 0.25)}
_NG45_SITE_UM, _NG45_ROW_UM = 0.19, 1.4


def routability_params(def_path: str) -> Dict:
    """DREAMPlace routability-mode params for a nangate45 DEF.

    DREAMPlace scales LEF/DEF coordinates to site widths (scale_factor = 1/site_width), so capacity per DREAMPlace
    unit = usable tracks per um x site width in um. Routing tiles are two rows square (~2.8 um), close to grt's gcell.
    """
    h = sum((1 - adj) / p for d, p, adj in _NG45_LAYERS.values() if d == "H")
    v = sum((1 - adj) / p for d, p, adj in _NG45_LAYERS.values() if d == "V")
    units, die = 2000.0, None
    with open(def_path) as f:
        for line in f:
            if line.startswith("UNITS DISTANCE MICRONS"):
                units = float(line.split()[3])
            elif line.startswith("DIEAREA"):
                n = [float(x) for x in re.findall(r"-?\d+", line)]
                die = ((n[2] - n[0]) / units, (n[3] - n[1]) / units)
                break
    if die is None:
        raise HandoffError(f"no DIEAREA in {def_path}")
    tile = 2 * _NG45_ROW_UM
    return {"routability_opt_flag": 1, "adjust_nctugr_area_flag": 0,
            "unit_horizontal_capacity": round(h * _NG45_SITE_UM, 6), "unit_vertical_capacity": round(v * _NG45_SITE_UM, 6),
            "route_num_bins_x": max(16, round(die[0] / tile)), "route_num_bins_y": max(16, round(die[1] / tile))}


class RawHandoffEvaluator:
    def __init__(self, work_root: str, dp_install: str, orfs_image: str = "openroad/orfs:latest",
                 python: str = "/usr/bin/python3", gpu: bool = True, platform: str = "nangate45",
                 native: bool = False, docker_cpus: Optional[float] = None, evidence_dir: Optional[str] = None,
                 timing_driven: bool = False, dp_depth: str = "global", net_weight: Optional[str] = None,
                 netwt_rounds: int = 3, netwt_k: float = 4.0, netwt_decay: float = 0.5,
                 netwt_max: float = 1024.0, netwt_repair: bool = True,
                 netwt_compound: bool = True, orfs_vars: Optional[Dict[str, str]] = None,
                 dp_extra: Optional[Dict] = None, rebuffer: bool = False,
                 rebuf_overflow=None, dp_routability: bool = False, rebuf_full: bool = False, rebuf_presplit: bool = False,
                 rp_finish: Optional[float] = None, rp_args: str = "",
                 netwt_scheme: str = "lilith", netwt_pct: float = netwt.GPL_NETS_PCT,
                 netwt_gpl_max: float = netwt.GPL_WEIGHT_MAX):
        self.root = os.path.abspath(work_root)
        # Extra ORFS make variables (e.g. GPL_TIMING_DRIVEN=0, PLACE_DENSITY=0.33) passed to every make call. They can
        # change the pre-placement DEF (remove_buffers, 3_1 density/seed), so they select their own base dir and are
        # part of the toolchain fingerprint: bases and results built under different settings never share a cache.
        for k in orfs_vars or {}:
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", k):
                raise ValueError(f"bad ORFS variable name {k!r}")
        self.orfs_vars = dict(sorted((orfs_vars or {}).items()))
        self.dp_extra = dict(dp_extra or {})  # raw DREAMPlace params merged last into its config (build_config)
        # Re-buffer + re-place: GP, then a KEPT repair_design on that placement (ORFS 3_3
        # session), then a warm-started GP of the buffered netlist. `rebuf_overflow` stops the first GP early at that
        # overflow (RePlAce repairs at ~0.6 and keeps placing); None = the first GP runs to convergence.
        if rebuffer and (timing_driven or net_weight == "iterative"):
            raise ValueError("rebuffer supports net_weight None or 'static' only")
        self.rebuffer, self.rebuf_overflow = bool(rebuffer), rebuf_overflow
        # one kept repair per entry: GP stops at that overflow (None = convergence), repair, warm-started GP, ...
        self.rebuf_stops = list(rebuf_overflow) if isinstance(rebuf_overflow, (list, tuple)) else [rebuf_overflow]
        self.rebuf_full = bool(rebuf_full)  # also rsz::fully_rebuffer after repair_design, as gpl's kept TD repair does
        if rebuf_full and not rebuffer:
            raise ValueError("rebuf_full needs rebuffer")
        # pre-split >2000-load signal nets with a 5x5-ish grid of BUF_X4 before the first kept repair (speed only;
        # ariane133 repair 1 480 s -> 85 s loaded, identical post-repair WNS/TNS)
        if rebuf_presplit and not rebuffer:
            raise ValueError("rebuf_presplit needs rebuffer")
        self.rebuf_presplit = bool(rebuf_presplit)
        # DREAMPlace's own routability mode (RUDY + pin-utilization cell inflation) with
        # routing capacities derived from the platform's tracks (routability_params). Needs the install patch
        # docker/dreamplace_routability_lefdef.patch (NCTUgr op was built unconditionally and crashed on LEF/DEF).
        if dp_routability and platform != "nangate45":
            raise ValueError("dp_routability capacities are only derived for nangate45")
        self.dp_routability = bool(dp_routability)
        # Hybrid: DREAMPlace GP stops at overflow `rp_finish`, then ORFS 3_3 runs RePlAce from that placement with
        # its own TD/RD args (+ `rp_args`, extra global_placement flags), see the rp_finish branch of _PATCH_NEW.
        if rp_finish is not None:
            if rebuffer or net_weight or timing_driven:
                raise ValueError("rp_finish is exclusive with rebuffer, net_weight and timing_driven")
            if not 0.0 < float(rp_finish) < 1.0:
                raise ValueError(f"rp_finish must be an overflow in (0, 1), got {rp_finish!r}")
            rp_finish = float(rp_finish)
        elif rp_args:
            raise ValueError("rp_args needs rp_finish")
        if "\n" in rp_args:
            raise ValueError("rp_args must be one line")
        self.rp_finish, self.rp_args = rp_finish, rp_args.strip()
        self.timing_driven = timing_driven
        # Net-weight timing-driven DREAMPlace: OpenSTA times the real netlist and
        # DREAMPlace only consumes per-net weights. "static": weights from the pre-placement DEF, one GP.
        # "iterative": that, then `netwt_rounds - 1` more re-time -> reweight -> warm-started GP rounds.
        if net_weight not in (None, "static", "iterative"):
            raise ValueError(f"net_weight must be None, 'static' or 'iterative', got {net_weight!r}")
        if net_weight and timing_driven:
            raise ValueError("net_weight and timing_driven (DREAMPlace's internal timer) are mutually exclusive")
        if net_weight == "iterative" and netwt_rounds < 2:
            raise ValueError("iterative net weighting needs netwt_rounds >= 2 (use 'static' for one round)")
        self.net_weight = net_weight
        self.netwt_rounds = 1 if net_weight == "static" else int(netwt_rounds)
        self.netwt_k, self.netwt_decay, self.netwt_max = float(netwt_k), float(netwt_decay), float(netwt_max)
        self.netwt_repair = bool(netwt_repair)  # virtual repair_design before reading slacks (net_weights.tcl)
        self.netwt_compound = bool(netwt_compound)  # lilith w *= 1+crit (True) or w = 1+crit per round (False)
        # "gpl": RePlAce's own reweight (netwt.gpl_weights) after each kept rebuffer repair, timed on the repaired
        # netlist without a further virtual repair; no weights for the first GP (stock has none before its first TD
        # iteration). Weights reset at every stage, as in gpl.
        if netwt_scheme not in ("lilith", "gpl"):
            raise ValueError(f"netwt_scheme must be 'lilith' or 'gpl', got {netwt_scheme!r}")
        if netwt_scheme == "gpl" and not (rebuffer and net_weight == "static"):
            raise ValueError("netwt_scheme 'gpl' needs rebuffer and net_weight 'static'")
        self.netwt_scheme, self.netwt_pct, self.netwt_gpl_max = netwt_scheme, float(netwt_pct), float(netwt_gpl_max)
        # How far DREAMPlace itself places before the ORFS handoff: "global" (default, unchanged), "global+legal",
        # "global+legal+detail". Part of the toolchain fingerprint below so the modes never share a cache/identity
        # -- otherwise a second mode would silently reuse the first mode's placement DEF and look "identical".
        self.dp_depth = dp_depth
        self.dp_install, self.image, self.python, self.gpu, self.platform = dp_install, orfs_image, python, gpu, platform
        self.native, self.docker_cpus, self.evidence_dir = native, docker_cpus, evidence_dir
        self.uid = f"{os.getuid()}:{os.getgid()}"
        os.makedirs(self.root, exist_ok=True)
        self._lefs: Optional[List[str]] = None
        self._design_lefs: Dict[str, List[str]] = {}
        self._ready = False
        self._toolchain: Optional[Dict] = None
        self.flow_cfg: Optional[Dict] = None  # set by for_flow(); then evaluate() rejects requests of another flow

    @classmethod
    def for_flow(cls, flow_cfg: Optional[Dict], work_root: str, dp_install: str, **kw) -> "RawHandoffEvaluator":
        """Evaluator for one ARMS_V2 flow config (FLOW_KNOBS dict; {} = the final flow). `kw` are the non-flow
        kwargs (python, native, gpu, ...). DREAMPlace knobs of the flow are applied per request (apply_flow)."""
        fk, _ = flow_kwargs(flow_cfg)
        ev = cls(work_root, dp_install, **kw, **fk)
        ev.flow_cfg = clamp_config(flow_cfg or {}, FLOW_KNOBS)
        return ev

    # ---- toolchain fingerprint -----------------------------------------
    @property
    def toolchain(self) -> Dict:
        """What produced the numbers. Pinned by image digest, not by the floating `latest` tag."""
        if self._toolchain is None:
            if self.native:
                digest = os.environ.get("PNR_IMAGE_DIGEST", "native-unpinned")
            else:
                p = subprocess.run(["docker", "image", "inspect", self.image, "--format", "{{index .RepoDigests 0}}"],
                                   capture_output=True, text=True)
                digest = p.stdout.strip() or "unknown"
            commit = "unknown"
            cf = os.path.join(self.dp_install, "COMMIT")
            if os.path.exists(cf):
                commit = open(cf).read().strip()
            torch = subprocess.run([self.python, "-c", "import torch;print(torch.__version__)"],
                                   capture_output=True, text=True).stdout.strip() or "unknown"
            # `mode` is deliberately not part of the fingerprint: docker and native runs were bit-identical,
            # so they share cache keys. It is recorded under provenance.host instead.
            # CPU and GPU DREAMPlace give slightly different placements (Ariane: HPWL 3.034e7 vs 3.027e7), and routing
            # amplifies small differences, so the device is part of the identity. Mix devices only across experiments.
            tc = {"orfs_image": digest, "dreamplace_commit": commit, "torch": torch,
                  "platform": self.platform, "dreamplace_device": "gpu" if self.gpu else "cpu",
                  "timing_driven": self.timing_driven, "dp_depth": self.dp_depth,
                  # deterministic GP (dreamplace_runner.build_config); older results were non-deterministic
                  "dp_deterministic": True}
            if self.orfs_vars:  # only when set, so default runs keep their existing keys
                tc["orfs_vars"] = self.orfs_vars
            if self.dp_extra:
                tc["dp_extra"] = self.dp_extra
            if self.rebuffer:
                # odb-v1: each kept repair runs once and its session is saved/resumed (replay-v1/-v2-full replayed
                # every earlier repair in each later 3_3; same results expected, but never shared keys)
                tc["rebuffer"] = {"overflow": self.rebuf_overflow, "scheme": "odb-v1"}
                if self.rebuf_full:
                    tc["rebuffer"]["scheme"] = "odb-v1-full"
                if self.rebuf_presplit:  # only when set, so existing keys stay
                    tc["rebuffer"]["presplit"] = {"thr": 2000, "group": 1000, "cell": "BUF_X4", "v": 1}
            if self.dp_routability:
                tc["dp_routability"] = "rudy-pin-nangate45-v1"
            if self.rp_finish is not None:  # only when set, so default runs keep their existing keys
                tc["rp_finish"] = {"overflow": self.rp_finish, "args": self.rp_args, "scheme": "skip-initial-v1"}
            if self.timing_driven:  # the OpenTimer-era timing results came from a non-functional timer; never share keys
                tc["timer_engine"] = "heterosta"
            if self.net_weight:  # weighted and unweighted placements must never share a cache key
                tc["net_weight"] = {"mode": self.net_weight, "rounds": self.netwt_rounds, "k": self.netwt_k,
                                    "decay": self.netwt_decay, "max": self.netwt_max, "repair": self.netwt_repair,
                                    "scheme": "lilith-opensta-v2"}
                if not self.netwt_compound:  # only when non-default, so existing compounding results keep their keys
                    tc["net_weight"]["compound"] = False
                if self.netwt_scheme == "gpl":  # only when used, so lilith results keep their keys
                    tc["net_weight"] = {"mode": self.net_weight, "scheme": "gpl-opensta-v1", "pct": self.netwt_pct,
                                        "max": self.netwt_gpl_max}
            # Same reasoning per GPU arch (sm_86 vs sm_89 place slightly differently), plus the arch DREAMPlace was
            # compiled for when the install records it.
            if self.gpu:
                tc["gpu_arch"] = self._gpu_arch()
                bf = os.path.join(self.dp_install, "CUDA_ARCH")
                if os.path.exists(bf):
                    tc["dp_build_arch"] = open(bf).read().strip()
            self._toolchain = tc
        return self._toolchain

    def _gpu_arch(self) -> str:
        """Compute capability of the GPU DREAMPlace runs on, e.g. "8.6". PNR_GPU_ARCH overrides (set by the build
        script / for tests); otherwise ask the DREAMPlace venv's own torch, then fall back to nvidia-smi. Never
        raises -- an undetectable arch returns "unknown" rather than breaking the toolchain fingerprint."""
        env = os.environ.get("PNR_GPU_ARCH")
        if env:
            return env.strip()
        try:
            out = subprocess.run(
                [self.python, "-c", "import torch;print('%d.%d'%torch.cuda.get_device_capability())"],
                capture_output=True, text=True, timeout=60).stdout.strip()
            if out:
                return out.splitlines()[-1].strip()
        except Exception:
            pass
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
        return "unknown"

    @property
    def tool_id(self) -> str:
        return hashlib.sha256(json.dumps(self.toolchain, sort_keys=True).encode()).hexdigest()[:12]

    def _host(self) -> Dict:
        return {"hostname": socket.gethostname(), "vm": os.environ.get("PNR_VM"), "cpus": os.cpu_count(),
                "mode": "native" if self.native else "docker",
                "docker_cpus": self.docker_cpus, "gpu": self.gpu}

    # ---- execution helpers ---------------------------------------------
    def _wdir(self, work: str) -> str:
        return work if self.native else "/work"

    def _run(self, work: Optional[str], cmd: str, timeout: int = 14400, env_sh: bool = True) -> subprocess.CompletedProcess:
        # 14400s (4h) default: measured Ariane133 post-placement stages (CTS+GRT+DRT+report) alone take ~115 min
        # (results/ariane133/ariane133_stock_flow.md).
        pre = "source /OpenROAD-flow-scripts/env.sh >/dev/null 2>&1; " if env_sh else ""  # env.sh prints to stdout
        if self.native:
            return subprocess.run(["bash", "-lc", pre + cmd], capture_output=True, text=True, timeout=timeout)
        args = ["docker", "run", "--rm", "--user", self.uid, "-e", "HOME=/tmp"]
        if self.docker_cpus:
            args += ["--cpus", str(self.docker_cpus), "-e", f"NUM_CORES={int(self.docker_cpus)}"]
        if work:
            args += ["-v", f"{work}:/work"]
        return subprocess.run(args + [self.image, "bash", "-lc", pre + cmd], capture_output=True, text=True, timeout=timeout)

    def _patch_scripts(self, work: str) -> None:
        """(Re)write the spliced global_place.tcl for this work dir; other scripts keep their mtimes so make
        does not consider finished stages stale."""
        orig = open(os.path.join(self.root, "_shared", "scripts_orig", "global_place.tcl")).read()
        if _PATCH_OLD not in orig or orig.count(_PRELUDE_OLD) != 1:
            raise HandoffError("ORFS global_place.tcl changed upstream: patch anchor not found")
        with open(os.path.join(work, "scripts", "global_place.tcl"), "w") as f:
            f.write(orig.replace(_PRELUDE_OLD, _PRELUDE_NEW).replace(_PATCH_OLD, _PATCH_NEW.replace("@W@", self._wdir(work))))

    def _vars(self) -> str:
        return "".join(f" {k}={shlex.quote(v)}" for k, v in self.orfs_vars.items())

    def base_dir(self, design: str) -> str:
        """<root>/<design>/base, or base-<slug>-<hash> under ORFS var overrides (no '=' in the name: the dir is part
        of make target paths, where '=' would parse as an assignment)."""
        if not self.orfs_vars:
            return os.path.join(self.root, design, "base")
        blob = json.dumps(self.orfs_vars, sort_keys=True)
        slug = re.sub(r"[^a-z0-9.]+", "-", "_".join(f"{k}-{v}" for k, v in self.orfs_vars.items()).lower())[:60]
        return os.path.join(self.root, design, f"base-{slug}-{hashlib.sha256(blob.encode()).hexdigest()[:6]}")

    def _make(self, work: str, design: str, target: str = "", timeout: int = 14400) -> subprocess.CompletedProcess:
        w = self._wdir(work)
        cfg = f"./designs/{self.platform}/{design}/config.mk"
        cmd = f"cd {_FLOW} && make DESIGN_CONFIG={cfg} WORK_HOME={w} SCRIPTS_DIR={w}/scripts{self._vars()} {target}"
        return self._run(work, cmd, timeout=timeout)

    def _mkdirs(self, work: str) -> None:
        for d in ("results", "logs", "reports", "objects", "dp"):
            os.makedirs(os.path.join(work, d), exist_ok=True)

    # ---- one-time setup -------------------------------------------------
    def _setup(self) -> None:
        if self._ready:
            return
        with flock(os.path.join(self.root, "_shared", ".setup.lock")):
            self._setup_locked()

    def _setup_locked(self) -> None:
        shared = os.path.join(self.root, "_shared")
        os.makedirs(shared, exist_ok=True)
        orig = os.path.join(shared, "scripts_orig")
        plat = f"{_FLOW}/platforms/{self.platform}/lef"
        lefs = ["NangateOpenCellLibrary.tech.lef", "NangateOpenCellLibrary.macro.mod.lef"]
        if not os.path.isdir(orig):
            if self.native:
                shutil.copytree(f"{_FLOW}/scripts", orig)
            else:
                p = self._run(shared, f"cp -r {_FLOW}/scripts /work/scripts_orig", env_sh=False)
                if p.returncode != 0:
                    raise HandoffError(f"could not extract ORFS scripts: {p.stderr[-300:]}")
        # Nangate45's metal3 (the signal wire layer ORFS uses via `set_wire_rc -signal -layer metal3` in
        # setRC.tcl) parasitics, resolved from setRC.tcl's own `set_layer_rc -layer metal3 ...` line. Tech-level,
        # not per-design, so a fixed constant is correct regardless of which design is being placed.
        self._wire_rc = {"resistance": 3.57147e-03, "capacitance": 1.03981e-01}
        if self.native:
            self._lefs = [f"{plat}/{n}" for n in lefs]
            self._libs = [f"{_FLOW}/platforms/{self.platform}/lib/NangateOpenCellLibrary_typical.lib"]
        else:
            if not all(os.path.exists(os.path.join(shared, n)) for n in lefs):
                self._run(shared, "cp " + " ".join(f"{plat}/{n}" for n in lefs) + " /work/", env_sh=False)
            self._lefs = [os.path.join(shared, n) for n in lefs]

            lib_base = "NangateOpenCellLibrary_typical.lib"
            if not os.path.exists(os.path.join(shared, lib_base)):
                self._run(shared, f"cp {_FLOW}/platforms/{self.platform}/lib/{lib_base} /work/", env_sh=False)
            self._libs = [os.path.join(shared, lib_base)]
        self._ready = True

    def prepare(self, design: str) -> str:
        """Run stock ORFS to stage 3_2 and dump the pre-placement DEF. Cached per design; safe under concurrency."""
        self._setup()
        base = self.base_dir(design)
        marker = os.path.join(base, "dp", "pre_gp.def")
        # Ready only if the 3_2 odb that every 3_3 copy starts from is still there too. Otherwise (e.g. results pruned)
        # each copied 3_3 would silently re-run synthesis + floorplan inside make, so rebuild the base once here.
        odb = os.path.join(base, "results", self.platform, design, "base", "3_2_place_iop.odb")

        def ready():
            return os.path.exists(marker) and os.path.exists(odb)
        if ready():
            return base
        lock = ".prepare.lock" if not self.orfs_vars else f".prepare.{os.path.basename(base)}.lock"
        with flock(os.path.join(self.root, design, lock)):
            if ready():  # another worker finished while we waited
                return base
            return self._prepare_locked(design, base, marker)

    def _prepare_locked(self, design: str, base: str, marker: str) -> str:
        self._mkdirs(base)
        shutil.copytree(os.path.join(self.root, "_shared", "scripts_orig"), os.path.join(base, "scripts"),
                        dirs_exist_ok=True)
        self._patch_scripts(base)
        res = f"{self._wdir(base)}/results/{self.platform}/{design}/base"
        p = self._make(base, design, f"{res}/3_2_place_iop.odb")
        if p.returncode != 0:
            raise HandoffError(f"ORFS to 3_2 failed: {p.stdout[-400:]} {p.stderr[-400:]}")
        p = self._make(base, design, f"{res}/3_3_place_gp.odb")  # dump mode
        if p.returncode != 0 or not os.path.exists(marker):
            raise HandoffError(f"pre_gp.def dump failed: {p.stdout[-400:]} {p.stderr[-400:]}")
        rd = os.path.join(base, "results", self.platform, design, "base")
        for f in os.listdir(rd):  # drop 3_3+ outputs so each candidate re-runs from 3_3
            if f.startswith(("3_3", "3_4", "3_5", "3_place", "4_", "5_", "6_")):
                os.remove(os.path.join(rd, f))
        return base

    # ---- evaluation -----------------------------------------------------
    def _repro(self, req: EvalRequest) -> str:
        return (f"python -m pnr_node.raw_handoff --design {req.design} --stage {req.stage} --seed {req.seed} "
                f"--cfg '{json.dumps(req.placer_cfg, sort_keys=True)}'" + (" --native" if self.native else "") +
                (f" --flow-cfg '{json.dumps(req.flow_cfg, sort_keys=True)}'" if req.flow_cfg else ""))

    def libs_for(self, design: str) -> List[str]:
        """Platform + design timing libraries (e.g. fakeram macro .lib), via ORFS's own `make print-ADDITIONAL_LIBS`
        so wildcards in config.mk are expanded exactly as the flow does -- mirrors `lefs_for`."""
        self._setup()
        cfg = f"./designs/{self.platform}/{design}/config.mk"
        p = self._run(None, f"cd {_FLOW} && make DESIGN_CONFIG={cfg} print-ADDITIONAL_LIBS")
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("ADDITIONAL_LIBS:")]
        srcs = line[-1].split(":", 1)[1].split() if line else []
        extra: List[str] = []
        if srcs and not self.native:
            dst = os.path.join(self.root, "_shared", f"libs_{design}")
            os.makedirs(dst, exist_ok=True)
            self._run(os.path.join(self.root, "_shared"), f"mkdir -p /work/libs_{design} && cp " + " ".join(srcs) +
                      f" /work/libs_{design}/", env_sh=False)
            extra = [os.path.join(dst, os.path.basename(x)) for x in srcs]
        else:
            extra = srcs
        return self._libs + extra

    def sdc_for(self, design: str) -> str:
        self._setup()
        sdc = f"{_FLOW}/designs/{self.platform}/{design}/constraint.sdc"
        if self.native:
            return sdc
        dst = os.path.join(self.root, "_shared", f"sdc_{design}")
        os.makedirs(dst, exist_ok=True)
        out_sdc = os.path.join(dst, "constraint.sdc")
        if not os.path.exists(out_sdc):
            self._run(dst, f"cp {sdc} /work/constraint.sdc", env_sh=False)
        return out_sdc

    def lefs_for(self, design: str) -> List[str]:
        """Platform tech + macro LEFs plus the design's own ADDITIONAL_LEFS (e.g. fakeram macros), as DREAMPlace needs
        them. Uses ORFS's own `make print-ADDITIONAL_LEFS` so wildcards in config.mk are expanded exactly as the flow does."""
        self._setup()
        cache = self._design_lefs.get(design)
        if cache is not None:
            return self._lefs + cache
        cfg = f"./designs/{self.platform}/{design}/config.mk"
        p = self._run(None, f"cd {_FLOW} && make DESIGN_CONFIG={cfg} print-ADDITIONAL_LEFS")
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("ADDITIONAL_LEFS:")]
        srcs = line[-1].split(":", 1)[1].split() if line else []
        extra: List[str] = []
        if srcs and not self.native:
            dst = os.path.join(self.root, "_shared", f"lefs_{design}")
            os.makedirs(dst, exist_ok=True)
            self._run(os.path.join(self.root, "_shared"), f"mkdir -p /work/lefs_{design} && cp " + " ".join(srcs) +
                      f" /work/lefs_{design}/", env_sh=False)
            extra = [os.path.join(dst, os.path.basename(x)) for x in srcs]
        else:
            extra = srcs
        self._design_lefs[design] = extra
        return self._lefs + extra

    def _dp_extra(self, base: str) -> Dict:
        """Raw DREAMPlace params for every GP of this evaluator: routability settings (if on), then --dp-extra."""
        out: Dict = {}
        if self.dp_routability:
            out.update(routability_params(os.path.join(base, "dp", "pre_gp.def")))
        out.update(self.dp_extra)
        return out

    def _dp_dir(self, req: EvalRequest) -> str:
        """DREAMPlace output is shared by every stage of one (design, knobs, seed, toolchain): stage is not part of it."""
        blob = json.dumps({"d": req.design, "t": req.tech, "c": req.placer_cfg, "s": req.seed, "id": req.tool_id},
                          sort_keys=True)
        return os.path.join(self.root, req.design, "dp-" + hashlib.sha256(blob.encode()).hexdigest()[:16])

    def _place(self, req: EvalRequest, base: str) -> DPResult:
        """DREAMPlace once per identity; later stages (proxy -> grt -> route) reuse the placement DEF."""
        d = self._dp_dir(req)
        cache = os.path.join(d, "dp_result.json")
        if os.path.exists(cache):
            r = DPResult(**json.load(open(cache)))
            # the re-buffer import resumes the saved session: a cached placement without it cannot be evaluated
            if r.ok and os.path.exists(r.def_path) and (not self.rebuffer or os.path.exists((r.rebuf or {}).get("odb") or "")):
                return r
        kwargs = {}
        if self.timing_driven:
            kwargs["timing_driven"] = True
            kwargs["lib_input"] = self.libs_for(req.design)
            kwargs["sdc_input"] = self.sdc_for(req.design)
            kwargs["wire_resistance_per_micron"] = self._wire_rc["resistance"]
            kwargs["wire_capacitance_per_micron"] = self._wire_rc["capacitance"]

        if self.rebuffer:
            r = self._place_rebuf(req, base, d)
        elif self.rp_finish is not None:  # GP stops early (unclamped: the knob range would cap it at 0.15)
            cfg = dict(req.placer_cfg)
            cfg["stop_overflow"] = self.rp_finish
            r = run_dreamplace(self.lefs_for(req.design), os.path.join(base, "dp", "pre_gp.def"),
                               os.path.join(d, "dp_out"), cfg, req.seed, install=self.dp_install,
                               python=self.python, gpu=self.gpu, dp_depth=self.dp_depth, extra=self._dp_extra(base))
        elif self.net_weight:
            r = self._place_netwt(req, base, d)
        else:
            r = run_dreamplace(self.lefs_for(req.design), os.path.join(base, "dp", "pre_gp.def"),
                               os.path.join(d, "dp_out"), req.placer_cfg, req.seed, install=self.dp_install,
                               python=self.python, gpu=self.gpu, dp_depth=self.dp_depth, extra=self._dp_extra(base),
                               **kwargs)
        if r.ok:
            with open(cache, "w") as f:
                json.dump(asdict(r), f)
        return r

    def _sta(self, design: str, base: str, nw: str, tag: str, placed: Optional[str] = None,
             netlist: Optional[str] = None, repair: Optional[bool] = None) -> netwt.SlackDump:
        """Per-net setup slacks from OpenSTA on the pre-placement netlist, optionally overlaid with a placement.

        Runs `net_weights.tcl` through ORFS's `make run`, so liberty, derate, wire RC and SDC are set up exactly as at
        ORFS stage 3_3. Docker mode mounts the design dir (base + all candidate dirs) at /work. `repair` overrides
        netwt_repair (the virtual repair_design) for this call.
        """
        ddir = os.path.join(self.root, design)

        def m(p: str) -> str:
            return p if self.native else "/work/" + os.path.relpath(p, ddir)

        tcl = os.path.join(nw, netwt.TCL)
        for name in (netwt.TCL, netwt.DUMP_TCL):  # docker mode sees only /work, so the scripts travel with the run
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), os.path.join(nw, name))
        out = os.path.join(nw, f"{tag}.slack")
        env = {"NW_DEF": m(netlist or os.path.join(base, "dp", "pre_gp.def")),
               "NW_SDC": m(os.path.join(base, "results", self.platform, design, "base", "2_floorplan.sdc")),
               "NW_OUT": m(out), "NW_IGNORE_DEG": "100", "NW_REPAIR": "1" if (self.netwt_repair if repair is None else repair) else "0"}
        if placed:
            env["NW_PLACED"] = m(placed)
        cfg = f"./designs/{self.platform}/{design}/config.mk"
        cmd = (f"cd {_FLOW} && " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) +
               f" make DESIGN_CONFIG={cfg} WORK_HOME={shlex.quote(m(os.path.join(nw, 'orfs')))}{self._vars()}"
               f" RUN_LOG_NAME_STEM={tag} run RUN_SCRIPT={shlex.quote(m(tcl))}")
        p = self._run(ddir, cmd, timeout=3600)
        with open(os.path.join(nw, f"{tag}.log"), "w") as f:
            f.write(p.stdout + p.stderr)
        if p.returncode != 0 or not os.path.exists(out):
            raise HandoffError(f"net-weight STA {tag} failed: {p.stdout[-300:]} {p.stderr[-300:]}")
        return netwt.parse_slacks(out)

    def _place_netwt(self, req: EvalRequest, base: str, d: str) -> DPResult:
        """Net-weight timing-driven placement: STA -> weights -> GP, `netwt_rounds` times (Flow A's
        positions -> slack -> weights -> positions loop, with OpenSTA on the true netlist instead of OpenTimer).

        Round 1 is weighted by the pre-placement DEF's timing (== "static" mode); each later round re-times the previous
        round's placement, updates the weights (lilith momentum, pnr_node/netwt.py) and warm-starts GP from that
        placement. The last round writes to <d>/dp_out like a plain run; the rest stays under <d>/netwt/.
        """
        nw = os.path.join(d, "netwt")
        os.makedirs(nw, exist_ok=True)
        t0 = time.time()
        lefs = self.lefs_for(req.design)
        kw = {"k": self.netwt_k, "decay": self.netwt_decay, "max_weight": self.netwt_max,
              "compound": self.netwt_compound}
        dump = self._sta(req.design, base, nw, "r0")
        trace = [{"round": 0, "timed": "pre_gp.def", **netwt.timing_stats(dump)}]
        weights, crit = netwt.update_weights(dump, {}, {}, **kw)
        place, dp_rt, r = os.path.join(base, "dp", "pre_gp.def"), 0.0, None
        for rnd in range(1, self.netwt_rounds + 1):
            wf = os.path.join(nw, f"r{rnd}.weights")
            netwt.write_weights(wf, weights)
            out = os.path.join(d, "dp_out") if rnd == self.netwt_rounds else os.path.join(nw, f"r{rnd}_out")
            r = run_dreamplace(lefs, place, out, req.placer_cfg, req.seed, install=self.dp_install,
                               python=self.python, gpu=self.gpu, dp_depth=self.dp_depth, net_weights_file=wf,
                               warm_start=rnd > 1, extra=self._dp_extra(base))
            dp_rt += r.runtime_s
            if not r.ok:
                r.error = f"net-weight round {rnd}: {r.error}"
                return r
            m = re.search(r"net_weights_file .*: set (\d+) nets, (\d+) unmatched",
                          open(os.path.join(out, "dreamplace.log")).read())
            comp = os.path.join(nw, f"r{rnd}_placed.def")
            components_only_def(r.def_path, comp)
            dump = self._sta(req.design, base, nw, f"r{rnd}", placed=comp)
            trace.append({"round": rnd, "timed": f"round {rnd} placement", **netwt.timing_stats(dump),
                          "weights_used": netwt.weight_stats(weights),
                          "dp_nets_set": int(m.group(1)) if m else None,
                          "dp_nets_unmatched": int(m.group(2)) if m else None,
                          "dp_hpwl": r.hpwl, "dp_overflow": r.overflow, "dp_iterations": r.iterations,
                          "dp_runtime_s": r.runtime_s})
            if rnd < self.netwt_rounds:
                weights, crit = netwt.update_weights(dump, weights, crit, **kw)
            place = r.def_path
        r.netwt = {"mode": self.net_weight, "rounds": trace, "dp_runtime_s": dp_rt,
                   "total_runtime_s": time.time() - t0}
        r.runtime_s = time.time() - t0  # GP rounds + STA rounds: the full cost of the timing-driven placement
        with open(os.path.join(nw, "trace.json"), "w") as f:
            json.dump(r.netwt, f, indent=2)
        return r

    @staticmethod
    def _rebuf_name(k: int) -> str:
        return "rebuf_from.def" if k == 1 else f"rebuf_from{k}.def"

    def _place_rebuf(self, req: EvalRequest, base: str, d: str) -> DPResult:
        """For each stop in rebuf_stops: GP (warm-started after the first) to that overflow, then a kept repair_design
        (+ fully_rebuffer) on the placement in an ORFS 3_3 splice; finally a warm-started GP of the buffered netlist
        to convergence. With net_weight="static", each GP gets weights from OpenSTA on the netlist it places."""
        rb = os.path.join(d, "rebuf")
        os.makedirs(rb, exist_ok=True)
        t0 = time.time()
        lefs = self.lefs_for(req.design)
        kw = {"k": self.netwt_k, "decay": self.netwt_decay, "max_weight": self.netwt_max,
              "compound": self.netwt_compound}
        trace: Dict = {"overflow_stop": self.rebuf_overflow, "stages": []}
        netlist, froms, odb = os.path.join(base, "dp", "pre_gp.def"), [], None
        n = len(self.rebuf_stops)
        for k in range(1, n + 2):  # k = 1..n: GP + repair; k = n+1: final GP
            st: Dict = {"stage": k}
            wf = None
            if self.netwt_scheme == "gpl":
                if k >= 2:  # gpl reweight after kept repair k-1, timed on that repaired netlist (dumped by its 3_3)
                    dump = netwt.parse_slacks(os.path.join(rb, f"r{k - 1}.slack"))
                    st["sta"] = netwt.timing_stats(dump)
                    wf = os.path.join(rb, f"r{k}.weights")
                    w = netwt.gpl_weights(dump, self.netwt_pct, self.netwt_gpl_max)
                    netwt.write_weights(wf, w)
                    st["weights"] = netwt.weight_stats(w)
            elif self.net_weight == "static":
                dump = self._sta(req.design, base, rb, f"r{k - 1}", netlist=None if k == 1 else netlist)
                st["sta"] = netwt.timing_stats(dump)
                wf = os.path.join(rb, f"r{k}.weights")
                netwt.write_weights(wf, netwt.update_weights(dump, {}, {}, **kw)[0])
            cfg = dict(req.placer_cfg)
            if k <= n and self.rebuf_stops[k - 1] is not None:
                cfg["stop_overflow"] = float(self.rebuf_stops[k - 1])
            out = os.path.join(d, "dp_out") if k == n + 1 else os.path.join(rb, f"r{k}_out")
            r = run_dreamplace(lefs, netlist, out, cfg, req.seed, install=self.dp_install, python=self.python,
                               gpu=self.gpu, dp_depth=self.dp_depth, net_weights_file=wf, warm_start=k > 1,
                               extra=self._dp_extra(base))
            if not r.ok:
                r.error = f"rebuffer GP{k}: {r.error}"
                return r
            st["gp"] = {"hpwl": r.hpwl, "overflow": r.overflow, "iterations": r.iterations, "runtime_s": r.runtime_s}
            if wf and self.netwt_scheme == "gpl":
                m = re.search(r"net_weights_file .*: set (\d+) nets, (\d+) unmatched",
                              open(os.path.join(out, "dreamplace.log")).read())
                st["dp_nets_set"], st["dp_nets_unmatched"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
            if k == n + 1:
                trace["stages"].append(st)
                break
            # kept repair k, in a throwaway copy of the base run that resumes the session saved after repair k-1
            work = os.path.join(rb, "orfs")
            if os.path.exists(work):
                shutil.rmtree(work)
            shutil.copytree(base, work, symlinks=True)
            self._patch_scripts(work)
            froms.append(os.path.join(rb, self._rebuf_name(k)))
            components_only_def(r.def_path, froms[-1])
            self._stage_rebuf(work, req.design, froms[-1], odb)
            if self.rebuf_presplit and k == 1:
                shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuf_presplit.tcl"),
                            os.path.join(work, "dp", "rebuf_presplit.tcl"))
                open(os.path.join(work, "dp", "rebuf_presplit"), "w").close()
            gpl_sta = self.netwt_scheme == "gpl"
            if gpl_sta:  # per-net slack dump of the repaired netlist, from the same session (was a separate STA run)
                shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), netwt.DUMP_TCL),
                            os.path.join(work, "dp", netwt.DUMP_TCL))
                open(os.path.join(work, "dp", "rebuf_sta"), "w").close()
            res = f"{self._wdir(work)}/results/{self.platform}/{req.design}/base"
            p = self._make(work, req.design, f"{res}/3_3_place_gp.odb", timeout=7200)
            with open(os.path.join(rb, f"orfs_make_{k}.log"), "w") as f:
                f.write(p.stdout + p.stderr)
            netlist, odb = os.path.join(rb, f"rebuf{k}.def"), os.path.join(rb, f"rebuf{k}.odb")
            outs = ["rebuf.def", "rebuf_out.odb"] + (["rebuf.slack"] if gpl_sta else [])
            if p.returncode != 0 or not all(os.path.exists(os.path.join(work, "dp", o)) for o in outs):
                return DPResult(False, error=f"rebuffer repair {k} failed rc={p.returncode}: {p.stderr[-300:]}",
                                runtime_s=time.time() - t0)
            shutil.move(os.path.join(work, "dp", "rebuf.def"), netlist)
            shutil.move(os.path.join(work, "dp", "rebuf_out.odb"), odb)
            if gpl_sta:
                shutil.move(os.path.join(work, "dp", "rebuf.slack"), os.path.join(rb, f"r{k}.slack"))
            st["repair"] = self._repair_stats(work, req.design)
            trace["stages"].append(st)
        trace["total_runtime_s"] = time.time() - t0
        r.runtime_s = trace["total_runtime_s"]
        r.rebuf = {"from": froms, "odb": odb, "trace": trace}
        with open(os.path.join(rb, "trace.json"), "w") as f:
            json.dump(r.rebuf, f, indent=2)
        return r

    def _stage_rebuf(self, work: str, design: str, from_def: Optional[str], odb: Optional[str]) -> None:
        """Set up a 3_3 work dir for the re-buffer flow: the session saved after the previous kept repair (resumed by
        the prelude instead of 3_2), this stage's placement to repair on (None for the import run), and the
        full-rebuffer marker."""
        if odb:
            dst = os.path.join(work, "results", self.platform, design, "base", _REBUF_IN)
            try:
                os.link(odb, dst)  # read-only input; a hard link avoids copying a large odb
            except OSError:
                shutil.copy(odb, dst)
        if from_def:
            shutil.copy(from_def, os.path.join(work, "dp", "rebuf_from.def"))
            if self.rebuf_full:
                open(os.path.join(work, "dp", "rebuf_full"), "w").close()

    def _repair_stats(self, work: str, design: str) -> Dict:
        """Buffers/resizes of the re-buffer repair_design in a 3_3 log, and read_def's unknown-component warnings
        (non-zero on the final import = the replay diverged from the dump run)."""
        log = os.path.join(work, "logs", self.platform, design, "base", "3_3_place_gp.log")
        t = open(log, errors="replace").read() if os.path.exists(log) else ""
        ins = re.findall(r"RSZ-0038\] Inserted (\d+) buffers", t)
        full = re.findall(r"^\s*final \|\s*\S+ \|\s*(\d+) \|\s*(\d+) \|", t, re.M)  # fully_rebuffer table
        rsz = re.findall(r"RSZ-0039\] Resized (\d+) instances", t)
        return {"buffers": sum(map(int, ins)), "resized": sum(map(int, rsz)),
                "rebuffer_removed_inserted": [list(map(int, x)) for x in full],
                "undefined_comps": len(re.findall(r"undefined comp", t, re.I))}

    def _gp3_stats(self, work: str, design: str) -> Dict:
        """What RePlAce did in a hybrid 3_3: its command, Nesterov iterations, TD/RD iterations, wall times."""
        log = os.path.join(work, "logs", self.platform, design, "base", "3_3_place_gp.log")
        return gp3_stats(open(log, errors="replace").read() if os.path.exists(log) else "")

    def evaluate(self, req: EvalRequest) -> EvalResult:
        t0 = time.time()
        req = apply_flow(req)  # flow_cfg's DREAMPlace knobs -> placer_cfg (the run dir key includes them)
        try:
            if req.flow_cfg and self.flow_cfg is not None and req.flow_cfg != clamp_config(self.flow_cfg, FLOW_KNOBS):
                raise ValueError(f"request flow_cfg {req.flow_cfg} != this evaluator's flow {self.flow_cfg}")
            base = self.prepare(req.design)
            run = os.path.join(self.root, req.design, req.key())
            os.makedirs(run, exist_ok=True)
            dp = self._place(req, base)
            prov = {"toolchain": self.toolchain, "tool_id": self.tool_id, "host": self._host(), "seed": req.seed,
                    "req_key": req.key(), "threads": {"cpus": os.cpu_count(), "dreamplace": 8}}
            if req.flow_cfg:
                prov["flow_cfg"] = req.flow_cfg
            if not dp.ok:
                return EvalResult(False, req.stage, error=f"dreamplace: {dp.error}", runtime_s=time.time() - t0,
                                  provenance=prov)
            prov.update({"dp_runtime_s": dp.runtime_s, "dp_iterations": dp.iterations, "dp_overflow": dp.overflow})
            if dp.netwt:
                prov["netwt"] = dp.netwt
            if dp.rebuf:
                prov["rebuf"] = dp.rebuf["trace"]
            stop = self.rp_finish if self.rp_finish is not None else req.placer_cfg["stop_overflow"]
            reasons = check_placement(dp.def_path, dp.overflow, stop)
            if reasons:  # zero-route-cost rejection of a placement that cannot be a valid start
                prov.update({"rejected": True, "reasons": reasons})
                return EvalResult(False, req.stage, proxy_hpwl=dp.hpwl, error="rejected: " + "; ".join(reasons),
                                  runtime_s=time.time() - t0, provenance=prov)
            if req.stage == "proxy":
                return EvalResult(True, "proxy", proxy_hpwl=dp.hpwl, runtime_s=time.time() - t0, provenance=prov)
            return self.route_from_def(req, base, run, dp.def_path, dp.hpwl, prov, t0,
                                       rebuf=dp.rebuf)
        except Exception as e:  # never let one candidate crash the search
            return EvalResult(False, req.stage, error=f"{type(e).__name__}: {e}", runtime_s=time.time() - t0,
                              provenance={"req_key": req.key()})

    def route_from_def(self, req: EvalRequest, base: str, run: str, def_path: str, proxy_hpwl: Optional[float],
                       prov: Dict, t0: float, rebuf: Optional[Dict] = None) -> EvalResult:
        """Second half of an evaluation: import a DREAMPlace placement and run stock ORFS onward.

        stage "cts"   stops after CTS (4_1) and returns the CTS-stage setup WNS/TNS (the v2 search stage).
        stage "grt"   stops after global route (5_1) and returns the global-route wirelength estimate.
        stage "route" runs to the end; if a finished grt run of the same identity exists it CONTINUES from it
                      (make resumes at 5_2), so screening a candidate and then routing it never repeats CTS/grt.
        Separate from evaluate() so placement (GPU) and routing (CPU) can run on different machines: the
        placement DEF is the only thing that crosses.
        """
        work = os.path.join(run, "orfs")
        if os.path.exists(work):
            shutil.rmtree(work)
        res5 = os.path.join("results", self.platform, req.design, "base", "5_1_grt.odb")
        grt_run = os.path.join(self.root, req.design, replace(req, stage="grt").key(), "orfs")
        reuse = req.stage == "route" and os.path.exists(os.path.join(grt_run, res5))
        shutil.copytree(grt_run if reuse else base, work, symlinks=True)  # copy2 keeps mtimes: finished stages stay fresh
        prov["reused_grt"] = reuse
        if not reuse:
            self._patch_scripts(work)
            prov["components_imported"] = components_only_def(def_path, os.path.join(work, "dp", "gp.def"))
            if rebuf:  # 3_3 resumes the session saved after the last kept repair, then imports gp.def
                self._stage_rebuf(work, req.design, None, rebuf["odb"])
            if self.rp_finish is not None:  # 3_3 runs RePlAce from gp.def (marker content: extra gpl args)
                with open(os.path.join(work, "dp", "rp_finish"), "w") as f:
                    f.write(self.rp_args + "\n")
        res4 = os.path.join("results", self.platform, req.design, "base", "4_1_cts.odb")
        target = {"grt": f"{self._wdir(work)}/{res5}", "cts": f"{self._wdir(work)}/{res4}"}.get(req.stage, "")
        p = self._make(work, req.design, target)
        with open(os.path.join(run, "orfs_make.log"), "w") as f:
            f.write(p.stdout + p.stderr)
        if p.returncode != 0:
            return EvalResult(False, req.stage, proxy_hpwl=proxy_hpwl, error=f"orfs make rc={p.returncode}: {p.stderr[-300:]}",
                              runtime_s=time.time() - t0, provenance=prov)
        if self.rp_finish is not None and not reuse:
            prov["rp_finish"] = gs = self._gp3_stats(work, req.design)
            if not gs["ran"]:
                return EvalResult(False, req.stage, proxy_hpwl=proxy_hpwl, runtime_s=time.time() - t0, provenance=prov,
                                  error="rp_finish: no RePlAce global_placement in 3_3")
        if rebuf and not reuse:
            rs = prov["rebuf_import"] = self._repair_stats(work, req.design)
            if rs["undefined_comps"] or rs["buffers"] or rs["resized"]:
                return EvalResult(False, req.stage, proxy_hpwl=proxy_hpwl, runtime_s=time.time() - t0, provenance=prov,
                                  error=f"rebuffer import diverged from the saved session: {rs}")
        prov["stage_wall_s"] = self.stage_walls(work, req.design)
        if req.stage == "cts":
            c = self.collect_cts(work, req.design)
            prov.update(c.pop("_evidence"))
            if c["wns"] is None or c["tns"] is None:
                return EvalResult(False, "cts", proxy_hpwl=proxy_hpwl, runtime_s=time.time() - t0, provenance=prov,
                                  error="cts: no cts__timing__setup__ws/tns in 4_1_cts.json")
            return EvalResult(True, "cts", proxy_hpwl=proxy_hpwl, runtime_s=time.time() - t0, provenance=prov, **c)
        if req.stage == "grt":
            g = self.collect_grt(work, req.design)
            prov.update(g.pop("_evidence"))
            return EvalResult(True, "grt", proxy_hpwl=proxy_hpwl, runtime_s=time.time() - t0, provenance=prov, **g)
        m = self.collect(work, req.design)
        prov.update(m.pop("_evidence"))
        res = EvalResult(True, "route", proxy_hpwl=proxy_hpwl, runtime_s=time.time() - t0, provenance=prov, **m)
        if self.evidence_dir:
            write_bundle(self.evidence_dir, req, res, self._repro(req))
        return res

    def stage_walls(self, work: str, design: str) -> Dict[str, float]:
        """Per-stage wall time (s) of the ORFS steps this run executed (3_3 onward), from each log's `Elapsed time`."""
        logs = os.path.join(work, "logs", self.platform, design, "base")
        out: Dict[str, float] = {}
        for name in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
            if not name.endswith(".log") or not name.startswith(("3_3", "3_4", "3_5", "4_", "5_", "6_")):
                continue
            t = open(os.path.join(logs, name), errors="replace").read()
            el = re.findall(r"^Elapsed time: (?:(\d+):)?(\d+):([0-9.]+)\[h:\]min:sec", t, re.M)
            if el:
                h, m, sec = el[-1]
                out[name[:-4]] = round(int(h or 0) * 3600 + int(m) * 60 + float(sec), 2)
        return out

    def collect_cts(self, work: str, design: str) -> Dict:
        logs = os.path.join(work, "logs", self.platform, design, "base")
        c = json.load(open(os.path.join(logs, "4_1_cts.json")))
        gp3 = os.path.join(logs, "3_3_place_gp.log")
        gp3 = open(gp3).read() if os.path.exists(gp3) else ""
        return {
            "wns": c.get("cts__timing__setup__ws"),
            "tns": c.get("cts__timing__setup__tns"),
            "_evidence": {"handoff_read_def_in_3_3": "read_def -incremental" in gp3,
                          "gpl_lines_in_3_3": len(re.findall(r"GPL-", gp3))},
        }

    def collect_grt(self, work: str, design: str) -> Dict:
        logs = os.path.join(work, "logs", self.platform, design, "base")
        g = json.load(open(os.path.join(logs, "5_1_grt.json")))
        gp3 = open(os.path.join(logs, "3_3_place_gp.log")).read() if os.path.exists(os.path.join(logs, "3_3_place_gp.log")) else ""
        # Preferred: the flow's own estimate (Spearman +0.99 vs final WL on gcd). Some designs (aes) do not emit it;
        # fall back to the global-route wirelength (+0.93 on gcd) and record which one was used.
        est = g.get("globalroute__route__wirelength__estimated")
        metric, val = ("globalroute__route__wirelength__estimated", est) if est is not None else \
            ("globalroute__global_route__wirelength", g.get("globalroute__global_route__wirelength"))
        return {
            "grt_wl": val,
            "wns": g.get("globalroute__timing__setup__ws"),
            "tns": g.get("globalroute__timing__setup__tns"),
            "_evidence": {"handoff_read_def_in_3_3": "read_def -incremental" in gp3,
                          "gpl_lines_in_3_3": len(re.findall(r"GPL-", gp3)), "grt_metric": metric},
        }

    def collect(self, work: str, design: str) -> Dict:
        logs = os.path.join(work, "logs", self.platform, design, "base")
        rep = json.load(open(os.path.join(logs, "6_report.json")))
        rte = json.load(open(os.path.join(logs, "5_2_route.json")))

        def txt(name: str) -> str:
            p = os.path.join(logs, name)
            return open(p).read() if os.path.exists(p) else ""

        gp3 = txt("3_3_place_gp.log")
        disp = re.findall(r"[Aa]verage displacement\s+([0-9.]+)", txt("3_5_place_dp.log"))
        evidence = {
            "handoff_read_def_in_3_3": "read_def -incremental" in gp3,
            "gpl_lines_in_3_3": len(re.findall(r"GPL-", gp3)),
            "opendp_avg_displacement_um": float(disp[-1]) if disp else None,
            "setup_violation_count": rep.get("finish__timing__drv__setup_violation_count"),
        }
        return {
            "routed_wl": rte.get("detailedroute__route__wirelength"),
            "wns": rep.get("finish__timing__setup__ws"),
            "tns": rep.get("finish__timing__setup__tns"),
            "power": rep.get("finish__power__total"),
            "area": rep.get("finish__design__instance__area"),
            "drc_violations": rte.get("detailedroute__route__drc_errors"),
            "_evidence": evidence,
        }


def gp3_stats(t: str) -> Dict:
    """Parse an ORFS 3_3_place_gp.log: RePlAce's command line, iterations, TD/RD events and wall times."""
    cmd = re.findall(r"^global_placement .*$", t, re.M)
    fin = re.search(r"GPL-1001\] Global placement finished at iteration (\d+)", t)
    td = [{"iter": int(i), "overflow": float(o)} for i, o in
          re.findall(r"GPL-0101\]\s+Iter: (\d+), overflow: ([0-9.]+)", t)]
    ov = re.findall(r"^\s*(\d+) \|\s+([0-9.]+) \|\s+[0-9.e+-]+ \|", t, re.M)  # Nesterov table rows
    took = re.search(r"^Took (\d+) seconds: global_placement", t, re.M)
    el = re.search(r"^Elapsed time: (?:(\d+):)?(\d+):([0-9.]+)\[h:\]min:sec", t, re.M)
    rd = re.search(r"GPL-1017\] Routability mode iteration count: (\d+)", t)
    hp = re.search(r"GPL-1018\] Final HPWL \(um\): ([0-9.]+)", t)
    return {"ran": bool(cmd), "command": cmd[-1] if cmd else None,
            "gpl_lines": len(re.findall(r"GPL-", t)),
            "nesterov_iterations": int(fin.group(1)) if fin else None,
            "start_overflow": float(ov[0][1]) if ov else None,
            "timing_driven_iters": td, "routability_iters": len(re.findall(r"GPL-0040\]", t)),
            "routability_mode_iterations": int(rd.group(1)) if rd else None,
            "final_hpwl_um": float(hp.group(1)) if hp else None,
            "gpl_runtime_s": int(took.group(1)) if took else None,
            "stage_elapsed_s": (int(el.group(1) or 0) * 3600 + int(el.group(2)) * 60 + float(el.group(3))) if el else None}


def _stops(v: Optional[str]):
    """'0.6' -> 0.6 (keeps single-repair identities), '0.6,0.2' -> [0.6, 0.2], None -> None."""
    if v is None:
        return None
    xs = [float(x) for x in v.split(",")]
    return xs[0] if len(xs) == 1 else xs


# CLI flags that set the flow; --flow-cfg replaces them (error if both are given).
_FLOW_FLAGS = ("cfg", "timing_driven", "timing_driven_netwt", "netwt_rounds", "netwt_k", "netwt_max", "netwt_scheme",
               "netwt_pct", "netwt_no_compound", "netwt_no_repair", "dp_extra", "rebuffer", "rebuf_overflow",
               "rebuf_full", "rebuf_presplit", "dp_routability", "rp_finish", "rp_args")


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(description="Run one evaluation through the raw DREAMPlace->ORFS handoff")
    ap.add_argument("--design", default="gcd")
    ap.add_argument("--stage", default="route", choices=["proxy", "grt", "route", "cts"])
    ap.add_argument("--work", default=os.path.expanduser("~/work/handoff"))
    ap.add_argument("--dp-install", default=os.path.expanduser("~/work/dreamplace/install2"))
    ap.add_argument("--cfg", default="{}", help="JSON dict of knob overrides")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native", action="store_true", help="ORFS is on this machine's PATH (inside the worker container)")
    ap.add_argument("--docker-cpus", type=float, default=None)
    ap.add_argument("--python", default="/usr/bin/python3")
    ap.add_argument("--evidence-dir", default=None)
    ap.add_argument("--cpu", action="store_true", help="run DREAMPlace on CPU (a different identity from the GPU results)")
    ap.add_argument("--timing-driven", action="store_true", help="enable timing-driven DREAMPlace")
    ap.add_argument("--timing-driven-netwt", default=None, choices=["static", "iterative"],
                    help="net-weight timing-driven DREAMPlace: OpenSTA per-net weights")
    ap.add_argument("--netwt-rounds", type=int, default=3, help="GP rounds in --timing-driven-netwt iterative")
    ap.add_argument("--netwt-k", type=float, default=4.0, help="criticality gain of the net-weight update")
    ap.add_argument("--netwt-max", type=float, default=None,
                    help="net-weight clamp (lilith, default 1024) / gpl timing_driven_net_weight_max (default 5)")
    ap.add_argument("--netwt-scheme", default="lilith", choices=["lilith", "gpl"],
                    help="lilith (default) or gpl: RePlAce's reweight after each kept repair (needs --rebuffer)")
    ap.add_argument("--netwt-pct", type=float, default=netwt.GPL_NETS_PCT,
                    help="gpl scheme: timing_driven_nets_percentage (default 10)")
    ap.add_argument("--netwt-no-compound", action="store_true",
                    help="reset weights to 1+crit each round instead of lilith's w *= 1+crit")
    ap.add_argument("--netwt-no-repair", action="store_true",
                    help="read slacks on the raw pre-repair netlist (default: after a virtual repair_design)")
    ap.add_argument("--orfs-var", action="append", default=[], metavar="K=V",
                    help="extra ORFS make variable for every make call (repeatable); own base dir and tool_id")
    ap.add_argument("--dp-extra", default="{}",
                    help="JSON of raw DREAMPlace params merged last into its config (unclamped; part of tool_id)")
    ap.add_argument("--rebuffer", action="store_true",
                    help="GP -> kept repair_design -> warm-started GP of the buffered netlist")
    ap.add_argument("--rebuf-overflow", default=None,
                    help="overflow(s) at which GP stops for a kept repair, e.g. 0.6 or 0.6,0.2 (one repair each); "
                         "default: one repair after a converged GP")
    ap.add_argument("--rebuf-full", action="store_true",
                    help="with --rebuffer: also rsz::fully_rebuffer after repair_design (gpl's kept TD repair)")
    ap.add_argument("--rebuf-presplit", action="store_true",
                    help="with --rebuffer: pre-split >2000-load signal nets before the first kept repair (speed)")
    ap.add_argument("--dp-routability", action="store_true",
                    help="DREAMPlace routability mode (RUDY/pin cell inflation) with nangate45 track capacities")
    ap.add_argument("--rp-finish", type=float, default=None, metavar="X",
                    help="hybrid: DREAMPlace GP stops at overflow X, then RePlAce finishes in 3_3 with ORFS's args")
    ap.add_argument("--rp-args", default="", help="with --rp-finish: extra global_placement args for RePlAce")
    ap.add_argument("--image", default="openroad/orfs:latest", help="ORFS image for docker mode")
    ap.add_argument("--dp-depth", default="global", choices=["global", "global+legal", "global+legal+detail"],
                    help="how far DREAMPlace places before the ORFS handoff (default: global placement only)")
    ap.add_argument("--flow-cfg", default=None, metavar="JSON",
                    help="ARMS_V2 flow config (FLOW_KNOBS dict; '{}' = the final flow); replaces the individual flow flags")
    return ap


def from_args(argv=None):
    """Parse the CLI into (evaluator, request) without running anything."""
    ap = build_parser()
    a = ap.parse_args(argv)
    common = dict(python=a.python, native=a.native, docker_cpus=a.docker_cpus, evidence_dir=a.evidence_dir,
                  gpu=not a.cpu, dp_depth=a.dp_depth, orfs_image=a.image,
                  orfs_vars=dict(kv.split("=", 1) for kv in a.orfs_var))
    if a.flow_cfg is not None:
        clash = [f"--{d.replace('_', '-')}" for d in _FLOW_FLAGS if getattr(a, d) != ap.get_default(d)]
        if clash:
            ap.error(f"--flow-cfg replaces the individual flow flags; also given: {' '.join(clash)}")
        fc = json.loads(a.flow_cfg)
        if not isinstance(fc, dict):
            ap.error("--flow-cfg must be a JSON object")
        ev = RawHandoffEvaluator.for_flow(fc, a.work, a.dp_install, **common)
        req = apply_flow(EvalRequest(a.design, seed=a.seed, stage=a.stage, tool_id=ev.tool_id, flow_cfg=fc))
        return ev, req
    ev = RawHandoffEvaluator(a.work, a.dp_install, timing_driven=a.timing_driven,
                             net_weight=a.timing_driven_netwt, netwt_rounds=a.netwt_rounds,
                             netwt_k=a.netwt_k, netwt_repair=not a.netwt_no_repair,
                             netwt_max=1024.0 if a.netwt_max is None else a.netwt_max, netwt_compound=not a.netwt_no_compound,
                             dp_extra=json.loads(a.dp_extra),
                             rebuffer=a.rebuffer, rebuf_overflow=_stops(a.rebuf_overflow), dp_routability=a.dp_routability,
                             rebuf_full=a.rebuf_full, rebuf_presplit=a.rebuf_presplit, rp_finish=a.rp_finish, rp_args=a.rp_args,
                             netwt_scheme=a.netwt_scheme, netwt_pct=a.netwt_pct,
                             netwt_gpl_max=netwt.GPL_WEIGHT_MAX if a.netwt_max is None else a.netwt_max, **common)
    req = EvalRequest(a.design, placer_cfg=json.loads(a.cfg), seed=a.seed, stage=a.stage, tool_id=ev.tool_id)
    return ev, req


def main(argv=None) -> None:
    ev, req = from_args(argv)
    r = ev.evaluate(req)
    # The last stdout line is the result JSON (host-side wrappers parse it): always carry the identity fields.
    r.provenance.setdefault("req_key", req.key())
    r.provenance.setdefault("tool_id", ev.tool_id)
    r.provenance.setdefault("dp_runtime_s", None)
    if req.flow_cfg:
        r.provenance.setdefault("flow_cfg", req.flow_cfg)
    run = os.path.join(ev.root, req.design, req.key())
    if os.path.isdir(run):  # so a run dir can be labelled later (scripts/stage_timing.py) without the console log
        with open(os.path.join(run, "result.json"), "w") as f:
            f.write(r.to_json())
    print(r.to_json(), flush=True)


if __name__ == "__main__":
    main()
