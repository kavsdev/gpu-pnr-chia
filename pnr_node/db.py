"""SQLite results database: every evaluation and every LLM call, with provenance."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from .contract import EvalRequest, EvalResult, score

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, arm TEXT, round INTEGER, design TEXT,
  stage TEXT, req_key TEXT, req_json TEXT, result_json TEXT, score REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, arm TEXT, round INTEGER, provider TEXT, model TEXT,
  system TEXT, prompt TEXT, response TEXT, input_tokens INTEGER, output_tokens INTEGER, latency_s REAL, phase TEXT
);
CREATE INDEX IF NOT EXISTS evals_arm ON evals(arm, design);
"""


class _Locked:
    """sqlite connection shared by several arm-run threads: statements and commits are serialised."""

    def __init__(self, con: sqlite3.Connection):
        self._c, self._l = con, threading.RLock()

    def execute(self, *a):
        with self._l:
            return self._c.execute(*a)

    def executescript(self, *a):
        with self._l:
            return self._c.executescript(*a)

    def commit(self):
        with self._l:
            return self._c.commit()


class ResultsDB:
    def __init__(self, path: str = ":memory:"):
        con = sqlite3.connect(path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        self.con = _Locked(con)
        self.con.executescript(_SCHEMA)
        # Migration: add 'phase' to llm_calls if missing from an existing database
        try:
            self.con.execute("ALTER TABLE llm_calls ADD COLUMN phase TEXT")
            self.con.commit()
        except sqlite3.OperationalError:
            pass  # Already exists

    def record_eval(self, arm: str, rnd: int, req: EvalRequest, res: EvalResult, note: str = "") -> float:
        s = score(res)
        self.con.execute(
            "INSERT INTO evals(ts,arm,round,design,stage,req_key,req_json,result_json,score,note) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), arm, rnd, req.design, req.stage, req.key(), json.dumps(req.__dict__, default=str),
             res.to_json(), None if s == float("inf") else s, note))
        self.con.commit()
        return s

    def record_llm(self, arm: str, rnd: int, system: str, prompt: str, resp: Any, phase: str = "propose") -> None:
        self.con.execute(
            "INSERT INTO llm_calls(ts,arm,round,provider,model,system,prompt,response,input_tokens,output_tokens,latency_s,phase) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), arm, rnd, resp.provider, resp.model, system, prompt, resp.text,
             resp.input_tokens, resp.output_tokens, resp.latency_s, phase))
        self.con.commit()

    def evals(self, arm: str, design: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM evals WHERE arm=? AND design=?" + (" AND stage=?" if stage else "") + " ORDER BY id"
        rows = self.con.execute(q, (arm, design, stage) if stage else (arm, design)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["cfg"] = json.loads(d["req_json"])["placer_cfg"]
            d["result"] = EvalResult.from_json(d["result_json"])
            out.append(d)
        return out

    def all_evals(self, design: str) -> List[Dict[str, Any]]:
        """Every evaluation for a design across all arms, ordered by time."""
        q = "SELECT * FROM evals WHERE design=? ORDER BY id"
        rows = self.con.execute(q, (design,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["cfg"] = json.loads(d["req_json"])["placer_cfg"]
            d["result"] = EvalResult.from_json(d["result_json"])
            out.append(d)
        return out

    def best_routed(self, arm: str, design: str) -> Optional[Dict[str, Any]]:
        rows = [r for r in self.evals(arm, design, "route") if r["score"] is not None]
        return min(rows, key=lambda r: r["score"]) if rows else None

    def lookup(self, req_key: str) -> Optional[EvalResult]:
        """First successful result for an experiment identity, from any arm (identical identities are bit-identical)."""
        r = self.con.execute("SELECT result_json FROM evals WHERE req_key=? ORDER BY id", (req_key,)).fetchall()
        for row in r:
            res = EvalResult.from_json(row[0])
            if res.ok:
                return res
        return None

    def max_round(self, arm: str, design: str) -> int:
        v = self.con.execute("SELECT MAX(round) FROM evals WHERE arm=? AND design=?", (arm, design)).fetchone()[0]
        return int(v or 0)

    def pareto(self, arm: str, design: str) -> List[Dict[str, Any]]:
        """Non-dominated routed results, minimising wirelength, -WNS (worse slack is worse), power, area, DRC."""
        rows = [r for r in self.evals(arm, design, "route") if r["result"].ok and r["result"].routed_wl is not None]

        def vec(r):
            x = r["result"]
            return (x.routed_wl, -(x.wns if x.wns is not None else 0.0), x.power or 0.0, x.area or 0.0,
                    float(x.drc_violations or 0))

        front = []
        for r in rows:
            v = vec(r)
            if not any(all(a <= b for a, b in zip(vec(o), v)) and vec(o) != v for o in rows):
                front.append(r)
        return front

    def n_stage(self, arm: str, design: str, stage: str) -> int:
        return self.con.execute("SELECT COUNT(*) FROM evals WHERE arm=? AND design=? AND stage=?",
                                (arm, design, stage)).fetchone()[0]

    def n_routes(self, arm: str, design: str, executed_only: bool = False) -> int:
        if executed_only:
            return self.con.execute("SELECT COUNT(*) FROM evals WHERE arm=? AND design=? AND stage='route' "
                                    "AND COALESCE(note,'')!='cache_hit'", (arm, design)).fetchone()[0]
        return self.con.execute("SELECT COUNT(*) FROM evals WHERE arm=? AND design=? AND stage='route'",
                                (arm, design)).fetchone()[0]


class CachedEvaluator:
    """Wrap any evaluator: an experiment identity already measured is replayed instead of re-run.

    Only successful results are replayed (a failed run may be transient). Hits are flagged in provenance
    so budget accounting can report new routes separately from replays.
    """

    def __init__(self, inner, db: "ResultsDB"):
        self.inner, self.db = inner, db
        self.tool_id = getattr(inner, "tool_id", "")
        self.hits = 0

    def evaluate(self, req: EvalRequest) -> EvalResult:
        hit = self.db.lookup(req.key())
        if hit is not None:
            self.hits += 1
            hit.provenance = dict(hit.provenance, cache_hit=True)
            return hit
        return self.inner.evaluate(req)

    def evaluate_many(self, reqs):
        out, miss = [None] * len(reqs), []
        for i, r in enumerate(reqs):
            hit = self.db.lookup(r.key())
            if hit is not None:
                self.hits += 1
                hit.provenance = dict(hit.provenance, cache_hit=True)
                out[i] = hit
            else:
                miss.append(i)
        if miss:
            # Dispatch each distinct identity once. Two positions holding the same req.key() (e.g. an LLM rep
            # whose best config equals the default) would otherwise run at the same time in the same key-named work
            # dir (<work_root>/<design>/<req_key>) and clobber each other's ORFS logs. Fan the one result back out.
            order: List[int] = []
            rep: Dict[str, int] = {}
            for i in miss:
                k = reqs[i].key()
                if k not in rep:
                    rep[k] = i
                    order.append(i)
            sub = [reqs[i] for i in order]
            res = self.inner.evaluate_many(sub) if hasattr(self.inner, "evaluate_many") else [self.inner.evaluate(r) for r in sub]
            by_key = {reqs[i].key(): r for i, r in zip(order, res)}
            for i in miss:
                out[i] = by_key[reqs[i].key()]
        return out
