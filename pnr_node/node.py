"""CHIA node: the place-and-route evaluator as a `ChiaFunction`, plus a driver-side evaluator.

Worker side (inside the `chia-pnr` container): `evaluate_remote` builds a `RawHandoffEvaluator` in native mode
(ORFS/OpenROAD on the PATH, DREAMPlace in its own venv with the GPU) and returns the EvalResult as JSON.
Driver side: `ChiaEvaluator` implements the same `.evaluate(req)` contract as every other evaluator, and
`.evaluate_many(reqs)` dispatches a whole batch at once so the cluster runs candidates in parallel.

Split mode (`cfg["split"] = True`, default False): `place_remote` (GPU-only, `num_gpus=1`) and `route_remote`
(CPU-only) are dispatched as two Ray tasks, so the GPU is released as soon as placement finishes instead of being
held for the whole route. The DEF handoff is a file on the placing VM's local disk, so both halves of a candidate
are pinned to one VM, chosen from its grt identity and weighted by each node's `pnr` slots (see `_split_vm`). With
no `vm_*` resources advertised it falls back to unpinned dispatch and logs a warning.

Without CHIA installed (unit tests, a laptop) the decorator is a no-op and calls run in-process, so the same
code path is testable. Node config (`cfg`) is plain JSON so it can ride in a CHIA config file.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Sequence

from .contract import EvalRequest, EvalResult

_log = logging.getLogger(__name__)


def _ray_nodes():
    """`ray.nodes()`, behind a seam so tests can mock it without a real cluster."""
    import ray

    return ray.nodes()

try:  # pragma: no cover - exercised on the cluster
    from chia.base.ChiaFunction import ChiaFunction, get

    HAVE_CHIA = True
except Exception:  # CHIA not installed here
    HAVE_CHIA = False

    def ChiaFunction(*_a, **_k):  # type: ignore
        return lambda fn: fn

    def get(x):  # type: ignore
        return x


_EVALUATORS: Dict[str, Any] = {}


def _evaluator(cfg: Dict[str, Any]):
    """One evaluator per worker process and config; ORFS base runs are cached on the worker's disk.

    `cfg["backend"]` (default "raw") selects the flow-engine: "raw" builds a `RawHandoffEvaluator` (ORFS/OpenROAD
    on nangate45), "librelane" builds a `LibreLaneEvaluator` (sky130, see librelane_handoff.py). Both implement the
    same `.evaluate(req) -> EvalResult` contract, so this is the only place a caller (evaluate_remote,
    ChiaEvaluator, search.py, experiment.py) needs to know the backend exists -- everything above stays unchanged.
    """
    key = json.dumps(cfg, sort_keys=True)
    if key not in _EVALUATORS:
        if cfg.get("backend", "raw") == "librelane":
            from .librelane_handoff import LibreLaneEvaluator

            _EVALUATORS[key] = LibreLaneEvaluator(
                cfg["work_root"], cfg.get("dp_install", os.environ.get("PNR_DREAMPLACE", "/opt/dreamplace")),
                python=cfg.get("python", "/usr/bin/python3"), gpu=cfg.get("gpu", True),
                platform=cfg.get("platform", "sky130hd"), evidence_dir=cfg.get("evidence_dir"),
                host_mount=cfg.get("host_mount"), dp_image=cfg.get("dp_image", "chia-pnr:dp"),
                librelane_image=cfg.get("librelane_image", "ghcr.io/librelane/librelane:3.0.14"),
                ll_script=cfg.get("ll_script"), pdk_root=cfg.get("pdk_root"))
        else:
            from .raw_handoff import RawHandoffEvaluator

            _EVALUATORS[key] = RawHandoffEvaluator(
                cfg["work_root"],
                cfg.get("dp_install", os.environ.get("PNR_DREAMPLACE", "/opt/dreamplace")),
                python=cfg.get("python", os.environ.get("PNR_DP_PYTHON", "/opt/dp-venv/bin/python")),
                native=cfg.get("native", True), gpu=cfg.get("gpu", True), evidence_dir=cfg.get("evidence_dir"),
                timing_driven=cfg.get("timing_driven", False), dp_depth=cfg.get("dp_depth", "global"))
        # `gpu` is set by the driver for the whole experiment (never auto-detected per worker) so every result in one
        # experiment comes from the same DREAMPlace device.
    return _EVALUATORS[key]


@ChiaFunction(resources={"pnr": 1}, num_gpus=1)
def evaluate_remote(req: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Evaluate one request on a worker. Never raises for a tool failure: returns an ok=False result."""
    return _evaluator(cfg).evaluate(EvalRequest(**req)).to_json()


@ChiaFunction(resources={"pnr": 0.01})
def toolchain_remote(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ev = _evaluator(cfg)
    return {"tool_id": ev.tool_id, "toolchain": ev.toolchain}


@ChiaFunction(resources={"pnr": 1}, num_gpus=1)
def place_remote(req: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """GPU-only half of the split path: DREAMPlace only, then release the GPU. Returns either a terminal
    {"done": True, "result": <EvalResult json>} (proxy stage, rejected, or DREAMPlace failure) or a handoff
    {"done": False, "def_path", "proxy_hpwl", "prov", "placement_time"} for route_remote to continue from."""
    import time
    from .dreamplace_runner import check_placement

    ev = _evaluator(cfg)
    r = EvalRequest(**req)
    t0 = time.time()
    try:
        base = ev.prepare(r.design)
        run = os.path.join(ev.root, r.design, r.key())
        os.makedirs(run, exist_ok=True)
        dp = ev._place(r, base)
        prov = {"toolchain": ev.toolchain, "tool_id": ev.tool_id, "host": ev._host(), "seed": r.seed,
                "req_key": r.key(), "threads": {"cpus": os.cpu_count(), "dreamplace": 8}}
        if not dp.ok:
            res = EvalResult(False, r.stage, error=f"dreamplace: {dp.error}", runtime_s=time.time() - t0, provenance=prov)
            return json.dumps({"done": True, "result": res.to_json()})

        prov.update({"dp_runtime_s": dp.runtime_s, "dp_iterations": dp.iterations, "dp_overflow": dp.overflow})
        reasons = check_placement(dp.def_path, dp.overflow, r.placer_cfg["stop_overflow"])
        if reasons:
            prov.update({"rejected": True, "reasons": reasons})
            res = EvalResult(False, r.stage, proxy_hpwl=dp.hpwl, error="rejected: " + "; ".join(reasons),
                             runtime_s=time.time() - t0, provenance=prov)
            return json.dumps({"done": True, "result": res.to_json()})

        if r.stage == "proxy":
            res = EvalResult(True, "proxy", proxy_hpwl=dp.hpwl, runtime_s=time.time() - t0, provenance=prov)
            return json.dumps({"done": True, "result": res.to_json()})

        return json.dumps({"done": False, "def_path": dp.def_path, "proxy_hpwl": dp.hpwl, "prov": prov,
                           "placement_time": time.time() - t0})
    except Exception as e:
        res = EvalResult(False, r.stage, error=f"{type(e).__name__}: {e}", runtime_s=time.time() - t0)
        return json.dumps({"done": True, "result": res.to_json()})


@ChiaFunction(resources={"pnr": 1})  # deliberately no num_gpus -- the whole point of the split
def route_remote(req: Dict[str, Any], cfg: Dict[str, Any], place_out: str) -> str:
    """CPU-only half of the split path: continues from place_remote's DEF handoff. `place_out` is passed as the
    Ray future returned by place_remote's dispatch, not its resolved value -- Ray resolves the dependency before
    this task runs, which is what lets the driver dispatch both without blocking on placement."""
    import time

    ev = _evaluator(cfg)
    r = EvalRequest(**req)
    po = json.loads(place_out)
    if po["done"]:
        return po["result"]

    t0 = time.time() - po["placement_time"]
    try:
        base = ev.prepare(r.design)
        run = os.path.join(ev.root, r.design, r.key())
        res = ev.route_from_def(r, base, run, po["def_path"], po["proxy_hpwl"], po["prov"], t0)
        return res.to_json()
    except Exception as e:
        res = EvalResult(False, r.stage, error=f"{type(e).__name__}: {e}", runtime_s=time.time() - t0)
        return res.to_json()


class ChiaEvaluator:
    """Driver-side evaluator backed by CHIA workers (or in-process when CHIA is absent).

    vm affinity (cfg["vm_affinity"]=True): a `route` request is pinned to the VM that already ran the `grt` stage of
    the same identity, because stage 5_2 continues from that VM's on-disk working directory (on Ariane that avoids
    repeating ~53 minutes). Needs every worker node type to advertise a `vm_<name>` resource and set PNR_VM=<name>.

    split (cfg["split"], default False): route dispatch through place_remote+route_remote instead of the single
    GPU-holding evaluate_remote. See the module docstring.
    """

    def __init__(self, cfg: Dict[str, Any]):
        if cfg.get("backend", "raw") != "raw" and cfg.get("split", False):
            # place_remote/route_remote call RawHandoffEvaluator-specific ._place/.route_from_def; the librelane
            # backend manages its own GPU container internally and has no split path to opt into.
            raise ValueError("split mode is only supported for backend='raw'")
        self.cfg = cfg
        self.affinity_hits = 0
        self._vm: Dict[str, str] = {}  # grt identity key -> VM name that holds its working directory
        # In split mode both halves of one candidate must run on the same VM (the DEF handoff is a local file).
        # Build a weighted slot list of the VMs advertising pnr+GPU; a candidate maps to one slot by its grt
        # identity. Empty (no vm_* advertised, or single-machine/in-process) => unpinned fallback.
        self._slots: List[str] = self._split_slots() if cfg.get("split", False) else []
        if cfg.get("split", False) and not self._slots:
            _log.warning("split mode: no node advertises vm_*/pnr/GPU; falling back to unpinned single-node dispatch "
                         "(safe on one machine, but on a multi-VM cluster the DEF handoff can miss its route node)")
        info = self._call(toolchain_remote, cfg)
        self.tool_id, self.toolchain = info["tool_id"], info["toolchain"]

    @staticmethod
    def _call(fn, *args):
        return get(fn.chia_remote(*args)) if HAVE_CHIA else fn(*args)

    @staticmethod
    def _split_slots() -> List[str]:
        """VMs that can run the split path, each repeated by its `pnr` count so mapping is throughput-weighted.
        Only meaningful on a real Ray cluster; in-process (unit tests, a laptop head with no vm_* resources) there
        is nothing to pin across, so this is empty and dispatch stays unpinned."""
        if not HAVE_CHIA:
            return []
        try:
            nodes = _ray_nodes()
        except Exception as e:  # no ray, or not connected: single-node fallback
            _log.warning("split mode: could not read ray.nodes() (%s); unpinned dispatch", e)
            return []
        slots: List[str] = []
        for n in nodes:
            res = n["Resources"] if n.get("Alive") else {}
            vms = [k[3:] for k in res if k.startswith("vm_")]
            if vms and res.get("pnr") and res.get("GPU"):
                slots += [vms[0]] * int(res["pnr"])
        return slots

    def _grt_key(self, r: EvalRequest) -> str:
        return replace(r, stage="grt").key()

    def _split_vm(self, r: EvalRequest) -> Optional[str]:
        """The VM this candidate's split place+route are pinned to, or None to leave it unpinned. Keyed on the grt
        identity so proxy/grt/route of the same candidate share a VM (and its on-disk work dir)."""
        if not self._slots:
            return None
        return self._slots[int(self._grt_key(r), 16) % len(self._slots)]

    def _handle_place(self, r: EvalRequest):
        # placement is always on a GPU node (implicitly through place_remote's num_gpus=1); in split mode it is
        # additionally pinned to the candidate's VM so route_remote finds its DEF handoff on the same local disk.
        vm = self._split_vm(r)
        return place_remote.options(resources={"pnr": 1, f"vm_{vm}": 0.001}) if vm else place_remote

    def _handle_route(self, r: EvalRequest):
        vm = self._split_vm(r)
        return route_remote.options(resources={"pnr": 1, f"vm_{vm}": 0.001}) if vm else route_remote

    def _remember(self, r: EvalRequest, res: EvalResult) -> None:
        vm = (res.provenance.get("host") or {}).get("vm")
        if vm and res.ok and r.stage in ("grt", "route"):
            self._vm[self._grt_key(r)] = vm

    def evaluate(self, req: EvalRequest) -> EvalResult:
        return self.evaluate_many([req])[0]

    def evaluate_many(self, reqs: Sequence[EvalRequest]) -> List[EvalResult]:
        # Dispatch each distinct req.key() exactly once, then fan the result out to every position that shares
        # it. Two positions with the same identity (e.g. an LLM rep whose best config equals the default) would
        # otherwise run at the same time in the same key-named work dir and clobber each other's ORFS logs.
        order: List[EvalRequest] = []
        rep: Dict[str, EvalRequest] = {}
        for r in reqs:
            k = r.key()
            if k not in rep:
                rep[k] = r
                order.append(r)
        uniq = self._dispatch(order)
        by_key = {r.key(): res for r, res in zip(order, uniq)}
        for r, res in zip(order, uniq):
            self._remember(r, res)
        return [by_key[r.key()] for r in reqs]

    def _dispatch(self, reqs: Sequence[EvalRequest]) -> List[EvalResult]:
        """Run one batch of DISTINCT requests (dedup is the caller's job). Never raises for a tool/worker failure."""
        if not self.cfg.get("split", False):
            if not HAVE_CHIA:
                return [EvalResult.from_json(evaluate_remote(asdict(r), self.cfg)) for r in reqs]
            refs = []
            for r in reqs:
                vm = self._vm.get(self._grt_key(r)) if (self.cfg.get("vm_affinity") and r.stage == "route") else None
                if vm:
                    self.affinity_hits += 1
                    refs.append(evaluate_remote.options(resources={"pnr": 1, f"vm_{vm}": 0.001})
                                .chia_remote(asdict(r), self.cfg))
                else:
                    refs.append(evaluate_remote.chia_remote(asdict(r), self.cfg))
            out = []
            for r, ref in zip(reqs, refs):
                try:
                    out.append(EvalResult.from_json(get(ref)))
                except Exception as e:  # a lost worker must not kill the search
                    out.append(EvalResult(False, r.stage, error=f"worker: {type(e).__name__}: {e}"))
            return out

        if not HAVE_CHIA:
            out = []
            for r in reqs:
                place_out = place_remote(asdict(r), self.cfg)
                if json.loads(place_out)["done"]:
                    out.append(EvalResult.from_json(json.loads(place_out)["result"]))
                else:
                    out.append(EvalResult.from_json(route_remote(asdict(r), self.cfg, place_out)))
            return out

        place_refs = [self._handle_place(r).chia_remote(asdict(r), self.cfg) for r in reqs]
        route_refs = [self._handle_route(r).chia_remote(asdict(r), self.cfg, pref)
                      for r, pref in zip(reqs, place_refs)]
        out = []
        for r, ref in zip(reqs, route_refs):
            try:
                out.append(EvalResult.from_json(get(ref)))
            except Exception as e:
                out.append(EvalResult(False, r.stage, error=f"worker: {type(e).__name__}: {e}"))
        return out
