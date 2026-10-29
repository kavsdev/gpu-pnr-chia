"""Host-side evaluator for the v2 arms: one `scripts/gap/run_arm.sh` docker run per evaluation.

`evaluate(req)` runs
    run_arm.sh <design> v2_<arm>_<req.key()> --stage <stage> --seed <seed> --flow-cfg '<json>'
and parses the LAST JSON line of $NETWT_WORK/logs/arm_<design>_<label>.log into an EvalResult (raw_handoff prints
the result JSON last; run_arm.sh then appends "exit=<rc>"). `evaluate_many` fans out over a thread pool; a shared
semaphore caps the in-flight runs at `workers` across every arm-run view (`for_arm`), and slot.sh's global slot
lock caps them again machine-wide. Two concurrent requests with the same identity run once (the raw_handoff work
dir is key-named, so a second concurrent run would clobber the first's logs).

Overrides (tests, other hosts): PNR_RUN_ARM = script path, NETWT_WORK = work root (as in run_arm.sh).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .contract import EvalRequest, EvalResult, default_config

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCRIPT = os.path.join(_REPO, "scripts", "gap", "run_arm.sh")


def last_json_line(text: str) -> Optional[Dict[str, Any]]:
    """The last line of `text` that parses as a JSON object (the evaluator's result line)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                return d
    return None


class _Shared:
    def __init__(self, workers: int):
        self.sem = threading.BoundedSemaphore(max(1, workers))
        self.lock = threading.Lock()
        self.inflight: Dict[str, Future] = {}


class ArmRunEvaluator:
    def __init__(self, script: Optional[str] = None, work: Optional[str] = None, workers: int = 3,
                 arm: str = "arm", timeout_s: Optional[float] = None, tool_id: Optional[str] = None,
                 extra_args: Sequence[str] = (), _shared: Optional[_Shared] = None):
        self.script = script or os.environ.get("PNR_RUN_ARM") or DEFAULT_SCRIPT
        self.work = work or os.environ.get("NETWT_WORK") or os.path.expanduser("~/work/netwt")
        self.workers, self.arm, self.timeout_s, self.extra_args = workers, arm, timeout_s, tuple(extra_args)
        self._shared = _shared or _Shared(workers)
        if tool_id is None:
            # Host-side identity for the ResultsDB cache: the launcher script + the DREAMPlace install it mounts.
            # The container's real tool_id is in each result's provenance. Change the install -> new identity.
            try:
                blob = open(self.script, "rb").read()
            except OSError:
                blob = self.script.encode()
            dp = os.environ.get("DP_INSTALL", "~/work/dreamplace/install_td100")
            # Extra run_arm.sh args (e.g. --orfs-var OPENROAD_HIERARCHICAL=0) change the flow, so they are part of
            # the identity; without them the identity is unchanged, so existing ResultsDBs stay valid.
            if self.extra_args:
                blob += json.dumps(list(self.extra_args)).encode()
            tool_id = "run_arm:" + hashlib.sha256(blob + dp.encode()).hexdigest()[:12]
        self.tool_id = tool_id

    def for_arm(self, arm: str) -> "ArmRunEvaluator":
        """A view labelling its runs with `arm`, sharing this evaluator's concurrency cap and in-flight table."""
        return ArmRunEvaluator(self.script, self.work, self.workers, arm, self.timeout_s, self.tool_id,
                               self.extra_args, self._shared)

    @staticmethod
    def _safe(s: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in s)

    def label(self, req: EvalRequest) -> str:
        return f"v2_{self._safe(self.arm)}_{req.key()}"

    def log_path(self, req: EvalRequest) -> str:
        return os.path.join(self.work, "logs", f"arm_{req.design}_{self.label(req)}.log")

    def command(self, req: EvalRequest) -> List[str]:
        cmd = [self.script, req.design, self.label(req), "--stage", req.stage, "--seed", str(req.seed)]
        if req.flow_cfg:
            cmd += ["--flow-cfg", json.dumps(req.flow_cfg, sort_keys=True)]
        if req.placer_cfg != default_config():  # non-flow request: pass the DREAMPlace knobs the old way
            cmd += ["--cfg", json.dumps(req.placer_cfg, sort_keys=True)]
        return cmd + list(self.extra_args)

    def _run(self, req: EvalRequest) -> EvalResult:
        log = self.log_path(req)
        t0 = time.time()
        with self._shared.sem:
            try:
                p = subprocess.run(self.command(req), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   timeout=self.timeout_s, env=dict(os.environ, NETWT_WORK=self.work))
                rc, out = p.returncode, p.stdout or ""
            except subprocess.TimeoutExpired:
                rc, out = None, "TimeoutExpired"
            except OSError as e:
                rc, out = None, f"OSError: {e}"
        wall = time.time() - t0
        try:
            text = open(log).read()
        except OSError:
            text = ""
        d = last_json_line(text)
        prov = {"arm_label": self.label(req), "arm_log": log, "run_arm_rc": rc, "host_wall_s": round(wall, 1)}
        if d is None:
            tail = (text or out)[-400:]
            return EvalResult(False, req.stage, runtime_s=wall, provenance=prov,
                              error=f"run_arm.sh rc={rc}: no result JSON in {log}: {tail}")
        try:
            res = EvalResult.from_json(json.dumps(d))
        except TypeError as e:
            return EvalResult(False, req.stage, runtime_s=wall, provenance=prov, error=f"bad result JSON: {e}")
        res.provenance = dict(res.provenance or {}, **prov)
        return res

    def evaluate(self, req: EvalRequest) -> EvalResult:
        k = req.key()
        with self._shared.lock:
            fut = self._shared.inflight.get(k)
            mine = fut is None
            if mine:
                fut = Future()
                self._shared.inflight[k] = fut
        if not mine:
            return fut.result()
        try:
            res = self._run(req)
            fut.set_result(res)
            return res
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._shared.lock:
                self._shared.inflight.pop(k, None)

    def evaluate_many(self, reqs: Sequence[EvalRequest]) -> List[EvalResult]:
        if not reqs:
            return []
        with ThreadPoolExecutor(max(1, min(self.workers, len(reqs)))) as ex:
            return list(ex.map(self.evaluate, reqs))
