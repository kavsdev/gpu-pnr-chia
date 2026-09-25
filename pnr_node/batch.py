"""Decoupled batch place-then-route on one machine (the local complement to `node.ChiaEvaluator`'s cluster dispatch).

With one GPU, placement is sequential, but routing is pure CPU work. `BatchEvaluator` places a batch one candidate
at a time (priming the inner evaluator's on-disk placement cache, see `RawHandoffEvaluator._dp_dir`), then routes
every placed candidate in parallel; phase 2's `evaluate()` calls find their placement cached and go straight to
routing. Same `.evaluate()` / `.evaluate_many()` shape as every other evaluator. `inner` lets tests use a fake.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, List, Optional, Protocol, Sequence

from .contract import EvalRequest, EvalResult


class _Evaluator(Protocol):
    tool_id: str

    def evaluate(self, req: EvalRequest) -> EvalResult: ...


class BatchEvaluator:
    """Place a whole batch (GPU, sequential), then route every placeable candidate (CPU, parallel)."""

    def __init__(self, work_root: Optional[str] = None, dp_install: Optional[str] = None, *, route_workers: int = 4,
                place_stage: str = "proxy", inner: Optional[_Evaluator] = None, **evaluator_kwargs: Any):
        if inner is None:
            from .raw_handoff import RawHandoffEvaluator

            if work_root is None or dp_install is None:
                raise ValueError("work_root and dp_install are required unless `inner` is supplied")
            inner = RawHandoffEvaluator(work_root, dp_install, **evaluator_kwargs)
        self._ev = inner
        self.route_workers = route_workers
        self.place_stage = place_stage
        self.tool_id = getattr(inner, "tool_id", "")
        self.toolchain = getattr(inner, "toolchain", {})

    def evaluate(self, req: EvalRequest) -> EvalResult:
        return self.evaluate_many([req])[0]

    def evaluate_many(self, reqs: Sequence[EvalRequest]) -> List[EvalResult]:
        out: List[Optional[EvalResult]] = [None] * len(reqs)

        # Phase 1: GPU-bound placement, one at a time (this machine has one GPU). Also serves proxy-only requests.
        to_route: List[int] = []
        for i, req in enumerate(reqs):
            p = self._ev.evaluate(req if req.stage == "proxy" else replace(req, stage=self.place_stage))
            if req.stage == "proxy":
                out[i] = p
            elif not p.ok:
                out[i] = p  # rejected/failed at the proxy tier: never pays for a route
            else:
                to_route.append(i)

        # Phase 2: CPU-bound routing, in parallel -- no GPU wait, since phase 1 already cached each placement.
        if to_route:
            with ThreadPoolExecutor(self.route_workers) as ex:
                futs = {ex.submit(self._ev.evaluate, reqs[i]): i for i in to_route}
                for fut in as_completed(futs):
                    out[futs[fut]] = fut.result()
        return out
