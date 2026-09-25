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
import shutil
import socket
import subprocess
import time
from typing import Dict, List, Optional

from .contract import EvalRequest, EvalResult
from dataclasses import asdict, replace

from .dreamplace_runner import DPResult, check_placement, run_dreamplace
from .evidence import write_bundle

_FLOW = "/OpenROAD-flow-scripts/flow"
_PATCH_OLD = "set result [catch { do_placement $global_placement_args } errMsg]"
_PATCH_NEW = """set dp_def @W@/dp/gp.def
if { [file exists $dp_def] } {
  log_cmd read_def -incremental $dp_def
} else {
  file mkdir @W@/dp
  log_cmd write_def @W@/dp/pre_gp.def
}
set result 0
set errMsg ""
"""


class HandoffError(RuntimeError):
    pass


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


class RawHandoffEvaluator:
    def __init__(self, work_root: str, dp_install: str, orfs_image: str = "openroad/orfs:latest",
                 python: str = "/usr/bin/python3", gpu: bool = True, platform: str = "nangate45",
                 native: bool = False, docker_cpus: Optional[float] = None, evidence_dir: Optional[str] = None,
                 timing_driven: bool = False, dp_depth: str = "global"):
        self.root = os.path.abspath(work_root)
        self.timing_driven = timing_driven
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
                  "timing_driven": self.timing_driven, "dp_depth": self.dp_depth}
            if self.timing_driven:  # keeps these keys apart from the earlier OpenTimer-engine timing runs
                tc["timer_engine"] = "heterosta"
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
        if _PATCH_OLD not in orig:
            raise HandoffError("ORFS global_place.tcl changed upstream: patch anchor not found")
        with open(os.path.join(work, "scripts", "global_place.tcl"), "w") as f:
            f.write(orig.replace(_PATCH_OLD, _PATCH_NEW.replace("@W@", self._wdir(work))))

    def _make(self, work: str, design: str, target: str = "", timeout: int = 14400) -> subprocess.CompletedProcess:
        w = self._wdir(work)
        cfg = f"./designs/{self.platform}/{design}/config.mk"
        cmd = f"cd {_FLOW} && make DESIGN_CONFIG={cfg} WORK_HOME={w} SCRIPTS_DIR={w}/scripts {target}"
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
        base = os.path.join(self.root, design, "base")
        marker = os.path.join(base, "dp", "pre_gp.def")
        if os.path.exists(marker):
            return base
        with flock(os.path.join(self.root, design, ".prepare.lock")):
            if os.path.exists(marker):  # another worker finished while we waited
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
                f"--cfg '{json.dumps(req.placer_cfg, sort_keys=True)}'" + (" --native" if self.native else ""))

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
            if r.ok and os.path.exists(r.def_path):
                return r
        kwargs = {}
        if self.timing_driven:
            kwargs["timing_driven"] = True
            kwargs["lib_input"] = self.libs_for(req.design)
            kwargs["sdc_input"] = self.sdc_for(req.design)
            kwargs["wire_resistance_per_micron"] = self._wire_rc["resistance"]
            kwargs["wire_capacitance_per_micron"] = self._wire_rc["capacitance"]

        r = run_dreamplace(self.lefs_for(req.design), os.path.join(base, "dp", "pre_gp.def"), os.path.join(d, "dp_out"),
                           req.placer_cfg, req.seed, install=self.dp_install, python=self.python, gpu=self.gpu,
                           dp_depth=self.dp_depth, **kwargs)
        if r.ok:
            with open(cache, "w") as f:
                json.dump(asdict(r), f)
        return r

    def evaluate(self, req: EvalRequest) -> EvalResult:
        t0 = time.time()
        try:
            base = self.prepare(req.design)
            run = os.path.join(self.root, req.design, req.key())
            os.makedirs(run, exist_ok=True)
            dp = self._place(req, base)
            prov = {"toolchain": self.toolchain, "tool_id": self.tool_id, "host": self._host(), "seed": req.seed,
                    "req_key": req.key(), "threads": {"cpus": os.cpu_count(), "dreamplace": 8}}
            if not dp.ok:
                return EvalResult(False, req.stage, error=f"dreamplace: {dp.error}", runtime_s=time.time() - t0,
                                  provenance=prov)
            prov.update({"dp_runtime_s": dp.runtime_s, "dp_iterations": dp.iterations, "dp_overflow": dp.overflow})
            reasons = check_placement(dp.def_path, dp.overflow, req.placer_cfg["stop_overflow"])
            if reasons:  # zero-route-cost rejection of a placement that cannot be a valid start
                prov.update({"rejected": True, "reasons": reasons})
                return EvalResult(False, req.stage, proxy_hpwl=dp.hpwl, error="rejected: " + "; ".join(reasons),
                                  runtime_s=time.time() - t0, provenance=prov)
            if req.stage == "proxy":
                return EvalResult(True, "proxy", proxy_hpwl=dp.hpwl, runtime_s=time.time() - t0, provenance=prov)
            return self.route_from_def(req, base, run, dp.def_path, dp.hpwl, prov, t0)
        except Exception as e:  # never let one candidate crash the search
            return EvalResult(False, req.stage, error=f"{type(e).__name__}: {e}", runtime_s=time.time() - t0)

    def route_from_def(self, req: EvalRequest, base: str, run: str, def_path: str, proxy_hpwl: Optional[float],
                       prov: Dict, t0: float) -> EvalResult:
        """Second half of an evaluation: import a DREAMPlace placement and run stock ORFS onward.

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
        target = f"{self._wdir(work)}/{res5}" if req.stage == "grt" else ""
        p = self._make(work, req.design, target)
        with open(os.path.join(run, "orfs_make.log"), "w") as f:
            f.write(p.stdout + p.stderr)
        if p.returncode != 0:
            return EvalResult(False, req.stage, proxy_hpwl=proxy_hpwl, error=f"orfs make rc={p.returncode}: {p.stderr[-300:]}",
                              runtime_s=time.time() - t0, provenance=prov)
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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run one evaluation through the raw DREAMPlace->ORFS handoff")
    ap.add_argument("--design", default="gcd")
    ap.add_argument("--stage", default="route", choices=["proxy", "grt", "route"])
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
    ap.add_argument("--dp-depth", default="global", choices=["global", "global+legal", "global+legal+detail"],
                    help="how far DREAMPlace places before the ORFS handoff (default: global placement only)")
    a = ap.parse_args()
    ev = RawHandoffEvaluator(a.work, a.dp_install, python=a.python, native=a.native, docker_cpus=a.docker_cpus,
                             evidence_dir=a.evidence_dir, gpu=not a.cpu, timing_driven=a.timing_driven,
                             dp_depth=a.dp_depth)
    r = ev.evaluate(EvalRequest(a.design, placer_cfg=json.loads(a.cfg), seed=a.seed, stage=a.stage, tool_id=ev.tool_id))
    print(r.to_json())


if __name__ == "__main__":
    main()
