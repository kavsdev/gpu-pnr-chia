"""DREAMPlace-through-LibreLane splice, sky130A (validated on `spm`: real Magic DRC=0, Netgen LVS=0).

Same mechanism as raw_handoff.py's DEF-dump/inject but hooked into LibreLane's Step API instead of shelling out to
ORFS's Makefile: `flows/librelane_sky130/ll_placer_flow.py` runs inside the LibreLane container, exports a DEF,
blocks on this evaluator to run DREAMPlace, re-imports the components-only DEF, and continues the flow.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from typing import Dict, List, Optional
from dataclasses import replace

from .contract import EvalRequest, EvalResult
from .dreamplace_runner import DPResult, check_placement, run_dreamplace
from .evidence import write_bundle

# Default location of the LibreLane-side flow script (ll_placer_flow.py, runs inside the librelane container) when
# used from a clone of this repo. Override via the `ll_script` constructor arg for any other layout.
_DEFAULT_LL_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "flows", "librelane_sky130",
                                                   "ll_placer_flow.py"))


class LibreLaneEvaluator:
    def __init__(self, work_root: str, dp_install: str,
                 python: str = "/usr/bin/python3", gpu: bool = True, platform: str = "sky130hd",
                 evidence_dir: Optional[str] = None,
                 host_mount: Optional[str] = None, dp_image: str = "chia-pnr:dp",
                 librelane_image: str = "ghcr.io/librelane/librelane:3.0.14",
                 ll_script: Optional[str] = None, pdk_root: Optional[str] = None):
        """
        host_mount: host directory bind-mounted read/write into both the DREAMPlace and LibreLane containers
                    (must contain `work_root`). Defaults to the user's home directory.
        dp_image / librelane_image: the two Docker images this evaluator shells out to (see `docker/Dockerfile.dp`
                    for how dp_image is built; librelane_image is pulled as-is from ghcr.io).
        ll_script: path to `ll_placer_flow.py` (runs inside the librelane container). Defaults to the copy that
                    ships in this repo (`flows/librelane_sky130/`).
        pdk_root: host path to the ciel-managed PDK checkout (must be under host_mount so the bind mount sees it),
                  e.g. `~/work/shared/librelane/pdk/ciel`. Passed into the container as `PDK_ROOT` -- without this,
                  ll_script's own default (`<ll_script dir>/pdk/sky130`) does not match ciel's actual layout
                  (`<pdk_root>/sky130/versions/<hash>/sky130A/...`) and the flow fails with "PDK ... was not found".
        """
        self.root = os.path.abspath(work_root)
        self.dp_install, self.python, self.gpu, self.platform = dp_install, python, gpu, platform
        self.evidence_dir = evidence_dir
        self.host_mount = os.path.abspath(host_mount) if host_mount else os.path.expanduser("~")
        self.dp_image, self.librelane_image = dp_image, librelane_image
        self.ll_script = os.path.abspath(ll_script) if ll_script else _DEFAULT_LL_SCRIPT
        self.pdk_root = os.path.abspath(pdk_root) if pdk_root else None
        os.makedirs(self.root, exist_ok=True)
        self._toolchain: Optional[Dict] = None

        self.dp_wrapper = os.path.join(self.root, "dp_wrapper.sh")
        with open(self.dp_wrapper, "w") as f:
            f.write('#!/bin/bash\n')
            f.write(f'docker run --rm --gpus all -v {self.host_mount}:{self.host_mount} -w "$(pwd)" '
                    f'{self.dp_image} /opt/dp-venv/bin/python "$@"\n')
        os.chmod(self.dp_wrapper, 0o755)

    @property
    def toolchain(self) -> Dict:
        if self._toolchain is None:
            self._toolchain = {"image": self.librelane_image, "dp_image": self.dp_image,
                               "platform": self.platform, "dreamplace_device": "gpu" if self.gpu else "cpu"}
        return self._toolchain

    @property
    def tool_id(self) -> str:
        return hashlib.sha256(json.dumps(self.toolchain, sort_keys=True).encode()).hexdigest()[:12]

    def _host(self) -> Dict:
        return {"hostname": socket.gethostname(), "cpus": os.cpu_count(), "gpu": self.gpu}

    def evaluate(self, req: EvalRequest) -> EvalResult:
        t0 = time.time()
        try:
            run_dir = os.path.join(self.root, req.design, req.key())
            os.makedirs(run_dir, exist_ok=True)

            prov = {"toolchain": self.toolchain, "tool_id": self.tool_id, "host": self._host(), "seed": req.seed,
                    "req_key": req.key(), "threads": {"cpus": os.cpu_count(), "dreamplace": 8}}

            cfg_path = os.path.join(run_dir, "ll_req.json")
            with open(cfg_path, "w") as f:
                json.dump({"seed": req.seed, "placer_cfg": req.placer_cfg}, f)

            dp_req_file = os.path.join(run_dir, "dp_req.json")
            dp_done_file = os.path.join(run_dir, "dp_done.json")
            flow_done_file = os.path.join(run_dir, "flow_done.json")

            for f in [dp_req_file, dp_done_file, flow_done_file]:
                if os.path.exists(f):
                    os.remove(f)

            cmd = ["docker", "run", "--rm", "-v", f"{self.host_mount}:{self.host_mount}",
                   "-w", os.path.dirname(self.ll_script)]
            if self.pdk_root:
                cmd += ["-e", f"PDK_ROOT={self.pdk_root}"]
            cmd += [self.librelane_image, "python3", self.ll_script, req.design, run_dir, req.key()]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Wait for dp_req_file or process exit
            while True:
                if proc.poll() is not None:
                    break
                if os.path.exists(dp_req_file):
                    break
                time.sleep(1)

            if os.path.exists(dp_req_file):
                with open(dp_req_file) as f:
                    dp_req = json.load(f)

                r = run_dreamplace(
                    lef_files=dp_req["lefs"],
                    def_in=dp_req["def_in"],
                    out_dir=dp_req["out_dir"],
                    cfg=req.placer_cfg,
                    seed=req.seed,
                    install=self.dp_install,
                    python=self.dp_wrapper,
                    gpu=self.gpu
                )

                if not r.ok:
                    res = {"ok": False, "error": f"dreamplace: {r.error}"}
                else:
                    reasons = check_placement(r.def_path, r.overflow, req.placer_cfg["stop_overflow"])
                    if reasons:
                        res = {"ok": False, "rejected": True, "reasons": reasons, "proxy_hpwl": r.hpwl, "error": "rejected: " + "; ".join(reasons)}
                    else:
                        res = {"ok": True, "def_path": r.def_path, "proxy_hpwl": r.hpwl, "runtime_s": r.runtime_s, "overflow": r.overflow, "iterations": r.iterations}

                # Write done marker so flow can continue
                with open(dp_done_file, "w") as f:
                    json.dump(res, f)

            stdout, _ = proc.communicate()

            with open(os.path.join(run_dir, "librelane.log"), "w") as f:
                f.write(stdout)

            if proc.returncode != 0 and not os.path.exists(flow_done_file):
                # Did it fail at placement rejection?
                if os.path.exists(dp_done_file):
                    with open(dp_done_file) as f:
                        d = json.load(f)
                    if not d.get("ok"):
                        if d.get("rejected"):
                            prov.update({"rejected": True, "reasons": d["reasons"]})
                            return EvalResult(False, req.stage, proxy_hpwl=d.get("proxy_hpwl"), error=d["error"], runtime_s=time.time() - t0, provenance=prov)
                        return EvalResult(False, req.stage, error=d["error"], runtime_s=time.time() - t0, provenance=prov)

                return EvalResult(False, req.stage, error=f"librelane failed: rc={proc.returncode}\n{stdout[-500:]}", runtime_s=time.time() - t0, provenance=prov)

            with open(flow_done_file) as f:
                m = json.load(f)

            if not m.get("ok"):
                return EvalResult(False, req.stage, error=m.get("error", "Unknown flow error"), runtime_s=time.time() - t0, provenance=prov)

            proxy_hpwl = None
            if os.path.exists(dp_done_file):
                with open(dp_done_file) as f:
                    d = json.load(f)
                    proxy_hpwl = d.get("proxy_hpwl")
                    prov.update({"dp_runtime_s": d.get("runtime_s"), "dp_iterations": d.get("iterations"), "dp_overflow": d.get("overflow")})

            prov.update(m.get("_evidence", {}))

            prov["design__lvs_error__count"] = m.get("lvs_violations")

            res = EvalResult(
                ok=True,
                stage=req.stage,
                proxy_hpwl=proxy_hpwl,
                routed_wl=m.get("routed_wl"),
                wns=m.get("wns"),
                tns=m.get("tns"),
                power=m.get("power"),
                area=m.get("area"),
                drc_violations=m.get("drc_violations"),
                runtime_s=time.time() - t0,
                provenance=prov
            )

            if self.evidence_dir:
                write_bundle(self.evidence_dir, req, res, f"python -m pnr_node.librelane_handoff --design {req.design}")
            return res

        except Exception as e:
            return EvalResult(False, req.stage, error=f"{type(e).__name__}: {e}", runtime_s=time.time() - t0)
