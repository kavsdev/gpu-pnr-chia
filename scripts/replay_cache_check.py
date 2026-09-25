"""Offline cache-replay check: every stored experiment identity must resolve to a cache hit, zero re-evaluations.

Opens a finished experiment DB (on a temp copy, so the original is never touched), rebuilds each stored
EvalRequest from its req_json, and replays it through CachedEvaluator wrapping an inner evaluator that raises if
called. Any cache miss is therefore loud.

Two identity definitions are replayed:
  1. "as written": the key hashed from the stored req_json exactly as the writing code version serialised it (this
     is what EvalRequest.key() computed when the row was recorded). This exercises the dedup/replay path.
  2. "current code": EvalRequest(**req_json).key() under the code in this checkout. If the request schema has
     gained defaulted fields since the DB was written (e.g. the expanded DREAMPlace knobs filled in by
     clamp_config), the key changes and every lookup misses. A miss re-evaluates; it can never return a wrong hit.

Only successful results are cached by design (a failed run may be transient), so rows whose identity never
succeeded are expected misses and are counted separately, not replayed.

Usage: python scripts/replay_cache_check.py [path/to/experiment.db]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pnr_node.contract import EvalRequest, EvalResult  # noqa: E402
from pnr_node.db import CachedEvaluator, ResultsDB  # noqa: E402

DEFAULT_DB = "results/tinyrocket/repair3_raw/tinyRocket_main.db"


class _StoredIdentity:
    """The request as the DB recorded it: key() hashes the stored req_json the way EvalRequest.key() did then."""

    def __init__(self, d, stage):
        self._d, self.stage, self.design = d, stage, d["design"]

    def key(self) -> str:
        d = dict(self._d)
        if not d.get("constraints_id"):
            d.pop("constraints_id", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


class _CountingInner:
    tool_id = ""

    def __init__(self):
        self.calls = 0

    def evaluate(self, req):
        self.calls += 1
        return EvalResult(False, req.stage, error="not evaluated (replay check)")


class _MustNotRun:
    tool_id = ""

    def evaluate(self, req):
        raise RuntimeError(f"cache miss -> would re-evaluate {req.design}/{req.stage} key={req.key()}")


def main(path: str) -> int:
    with tempfile.TemporaryDirectory() as td:
        copy = Path(td) / "replay.db"
        shutil.copy(path, copy)
        db = ResultsDB(str(copy))
        rows = db.con.execute("SELECT req_key, req_json, result_json, note FROM evals ORDER BY id").fetchall()

        ok_keys, stored, current = set(), {}, {}
        for r in rows:
            d = json.loads(r["req_json"])
            stored.setdefault(r["req_key"], _StoredIdentity(d, d["stage"]))
            current.setdefault(r["req_key"], EvalRequest(**d))
            if EvalResult.from_json(r["result_json"]).ok:
                ok_keys.add(r["req_key"])
        n = len(ok_keys)
        self_consistent = sum(stored[k].key() == k for k in stored)
        noted_hits = sum((r["note"] or "") == "cache_hit" for r in rows)

        # 1. as written: must be all hits, zero calls to the inner evaluator.
        cached = CachedEvaluator(_MustNotRun(), db)
        stages = {}
        for k in sorted(ok_keys):
            res = cached.evaluate(stored[k])
            assert res.provenance.get("cache_hit") is True
            stages[stored[k].stage] = stages.get(stored[k].stage, 0) + 1
        assert cached.hits == n, (cached.hits, n)

        # 2. current code: the unmodified path -- rebuilt EvalRequest objects through CachedEvaluator. The inner
        #    evaluator only counts calls (a miss here means the key schema changed since the DB was written).
        counter = _CountingInner()
        cur = CachedEvaluator(counter, db)
        for k in sorted(ok_keys):
            cur.evaluate(current[k])
        cur_hits = cur.hits

    print(f"db: {path}")
    print(f"rows: {len(rows)}  unique identities: {len(stored)}  successful identities: {n}  "
          f"failed-only identities (not cacheable by design): {len(stored) - n}")
    print(f"rows already recorded as cache_hit during the live run: {noted_hits}")
    print(f"successful identities by stage: {dict(sorted(stages.items()))}")
    print(f"stored req_json hash == stored req_key: {self_consistent}/{len(stored)}")
    print(f"[as written]  {cached.hits}/{n} cache hits, 0 re-evaluations")
    print(f"[current code] {cur_hits}/{n} cache hits, {counter.calls} re-evaluations, via EvalRequest.key() + CachedEvaluator"
          + ("" if cur_hits == n else "  (key schema changed since this DB was written; misses re-evaluate)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB))
