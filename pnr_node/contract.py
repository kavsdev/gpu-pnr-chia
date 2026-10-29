"""Eval contract shared by every implementation of the place-and-route node.

Dependency-free (stdlib only) so it imports the same on a laptop, in a CHIA
worker container and in unit tests. Any evaluator -- mock, raw OpenROAD,
LibreLane -- takes an EvalRequest and returns an EvalResult.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

# DREAMPlace only / through global route / through detailed route / through CTS (4_1; the v2 search stage)
STAGES = ("proxy", "grt", "route", "cts")


@dataclass(frozen=True)
class Knob:
    """One tunable DREAMPlace parameter with its legal range (the agent's action space)."""

    name: str
    lo: float
    hi: float
    default: float
    integer: bool = False
    log: bool = False
    # Categorical knob: the legal values (str/bool/float, returned exactly as listed here); lo/hi are unused.
    choices: Optional[tuple] = None

    def clamp(self, v: Any) -> Any:
        if self.choices is not None:
            return self._snap(v)
        v = min(max(float(v), self.lo), self.hi)
        return int(round(v)) if self.integer else v

    def _snap(self, v: Any) -> Any:
        """Snap a categorical value to the identical (or nearest) choice; the choice object itself is returned."""
        cs = self.choices
        if all(isinstance(c, bool) for c in cs):
            if isinstance(v, str):
                t = v.strip().lower()
                if t in ("true", "1", "yes", "on"):
                    v = True
                elif t in ("false", "0", "no", "off", ""):
                    v = False
                else:
                    return self.default
            b = bool(v)
            return next(c for c in cs if c is b) if any(c is b for c in cs) else self.default
        for c in cs:  # identical (type-aware: True must not match 1.0)
            if type(c) is type(v) and c == v:
                return c
        if all(isinstance(c, str) for c in cs):
            t = str(v).replace(" ", "").strip()
            if t in cs:
                return t
            vals = _floats(t)  # nearest by numeric content, e.g. "0.6,0.25" -> "0.6,0.2"
            if vals is None:
                return self.default

            def dist(c: str):
                cv = _floats(c) or []
                return (abs(len(cv) - len(vals)), sum(abs(a - b) for a, b in zip(cv, vals)))
            return min(cs, key=dist)
        try:
            x = float(v)
        except (TypeError, ValueError):
            return self.default
        return min(cs, key=lambda c: abs(float(c) - x))


def _floats(s: str) -> Optional[list]:
    try:
        return [float(x) for x in str(s).split(",")]
    except ValueError:
        return None


# The original bounded action space: the region where DREAMPlace converged on gcd (results/gcd/proxy_route_study_gcd.md:
# 60/60 converged here versus 33/160 over the wide ranges 0.4-1.0, 1e-6..1e-2, 1-20, 0.05-0.3, 0.001-0.05).
# Names follow DREAMPlace's params.json.
_CORE = (
    Knob("target_density", 0.6, 1.0, 0.8),
    Knob("density_weight", 1e-5, 1e-3, 8e-5, log=True),
    Knob("gamma", 2.0, 12.0, 4.0),
    Knob("stop_overflow", 0.05, 0.15, 0.1),
    Knob("learning_rate", 0.004, 0.025, 0.01, log=True),
)

# Expanded knobs (opt-in per search arm via a proposer's `knobs=` subset). Each default reproduces the value that
# build_config used to hardcode, so an arm that does not vary a knob is bit-identical to the original 5-knob
# behaviour -- the expanded arm can widen the action space without changing the baseline arm. Ranges were validated
# for convergence on tinyRocket before being admitted here;
# a wider-but-unvalidated range that silently emits non-converging configs would be worse than a smaller real one.
#   iteration:                        global-placement iteration cap (was hardcoded 1000)
#   Llambda_density_weight_iteration: outer density-weight update cadence (was hardcoded 1)
#   Lsub_iteration:                   inner sub-iterations per step (was hardcoded 1)
#   num_bins:                         density-grid resolution; <32 = DREAMPlace's auto grid (the historical default)
_EXPANDED = (
    Knob("iteration", 700, 2000, 1000, integer=True),
    Knob("Llambda_density_weight_iteration", 1, 4, 1, integer=True),
    Knob("Lsub_iteration", 1, 2, 1, integer=True),
    Knob("num_bins", 0, 512, 0, integer=True),
)

CORE_KNOBS: Dict[str, Knob] = {k.name: k for k in _CORE}
KNOBS: Dict[str, Knob] = {k.name: k for k in (_CORE + _EXPANDED)}

# Flow-level action space (identical for every v2 arm). Defaults = the final flow
# (raw_handoff.FINAL_FLOW_KWARGS); raw_handoff.flow_kwargs maps a clamped dict onto RawHandoffEvaluator kwargs
# (target_density -> dp_extra, so the 0.6-1.0 KNOBS range does not apply) and DREAMPlace placer_cfg overrides.
_FLOW = (
    Knob("target_density", 0.35, 0.80, 0.45),
    Knob("rebuf_stops", 0.0, 0.0, "0.6,0.2", choices=("0.6,0.2", "0.6,0.4,0.2", "0.7,0.3", "0.5,0.2", "0.6", "0.4")),
    Knob("netwt_pct", 5.0, 25.0, 10.0),
    Knob("netwt_max", 2.0, 10.0, 5.0),
    Knob("rebuf_full", 0.0, 0.0, True, choices=(True, False)),
    # Pinned off (2026-10-21): with the rebuffer flow the resumed GP has no fillers and routability inflation crashed
    # (DREAMPlace [-0:] slice, docker/dreamplace_filler0.patch); patched, the final GP still diverges (tinyRocket overflow
    # 0.91 at 1000 iterations, gcd 0.56). Kept in the space so keys/tests stay stable.
    Knob("dp_routability", 0.0, 0.0, False, choices=(False,)),
    Knob("density_weight", 2e-5, 4e-4, 8e-5, log=True),
    Knob("gamma", 2.0, 8.0, 4.0),
    Knob("learning_rate", 0.004, 0.025, 0.01, log=True),
)
FLOW_KNOBS: Dict[str, Knob] = {k.name: k for k in _FLOW}
FLOW_DP_KNOBS = ("density_weight", "gamma", "learning_rate")  # applied as DREAMPlace placer_cfg overrides


def default_config(knobs: Optional[Dict[str, Knob]] = None) -> Dict[str, float]:
    return {k: v.default for k, v in (knobs or KNOBS).items()}


def clamp_config(cfg: Dict[str, Any], knobs: Optional[Dict[str, Knob]] = None) -> Dict[str, float]:
    """Drop unknown keys, clamp known ones, fill missing with defaults. `knobs` selects the action space
    (defaults to the full KNOBS; a proposer restricting itself to CORE_KNOBS passes that instead)."""
    knobs = knobs or KNOBS
    out = default_config(knobs)
    for k, v in (cfg or {}).items():
        if k in knobs:
            out[k] = knobs[k].clamp(v)
    return out


@dataclass(frozen=True)
class PromptConfig:
    """Config-driven prompts and feedback modes for the LLM proposer."""

    # Per-stage system prompt overrides. Key: stage ("proxy", "grt", "route") or "default".
    system_prompts: Dict[str, str] = field(default_factory=dict)
    # Per-stage template overrides.
    templates: Dict[str, str] = field(default_factory=dict)
    # "single" (default) or "two_call" (critique then propose)
    feedback_mode: str = "single"
    # Whether to use failure-signature memory (off by default)
    use_memory: bool = False
    # Critique system prompt (used in two_call mode)
    critique_system_prompt: Optional[str] = None


@dataclass(frozen=True)
class Constraints:
    """Design constraints and allowed action space for the search."""

    id: str
    target_period: Optional[float] = None  # ns
    utilization_limit: Optional[float] = None
    area_limit: Optional[float] = None  # um^2
    # Ordered list of PPA priorities, e.g. ["wl", "wns", "power", "drc"]
    priority: list[str] = field(default_factory=list)
    drc_tolerance: int = 0
    # Which knobs the search is allowed to vary (subset of CORE_KNOBS or KNOBS)
    allowed_knobs: Optional[list[str]] = None


@dataclass
class EvalRequest:
    design: str  # e.g. "gcd", "tinyRocket", "ariane133"
    tech: str = "nangate45"
    placer_cfg: Dict[str, float] = field(default_factory=default_config)
    seed: int = 0
    stage: str = "route"  # "proxy" = DREAMPlace only; "route" = full OpenROAD
    tool_id: str = ""  # fingerprint of the toolchain (image digest + DREAMPlace commit); part of the identity
    constraints_id: str = ""  # stable ID for the design constraints
    # Flow-level config (FLOW_KNOBS). {} = the evaluator's own flow; clamped (and filled with defaults)
    # when non-empty. Not part of key() when empty or all-default, so every pre-v2 cache key is unchanged.
    flow_cfg: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        self.placer_cfg = clamp_config(self.placer_cfg)
        if self.flow_cfg:
            self.flow_cfg = clamp_config(self.flow_cfg, FLOW_KNOBS)

    def key(self) -> str:
        """Stable hash of the experiment identity: design, tech, knobs, seed, stage and toolchain.

        Deliberately excludes wall-clock and host. Identical identities gave bit-identical results in our
        determinism check (results/gcd/noise_floor_gcd.md) and were also identical for 2 vs 8 ORFS threads on gcd
        (one config), so a key match is treated as a cache hit. Thread count is recorded in provenance; re-check it
        on larger designs.

        CAVEAT (E5, measured 2026-09-24): bit-identity is verified ONLY on gcd. On tinyRocket the same identity gave
        slightly different routed results across a split vs non-split run (WL 509,808 vs 509,468, +0.07%; WNS -0.272
        vs -0.303 ns). With one sample it is not known whether split changes the result or tinyRocket routing simply
        is not bit-reproducible. Do not rely on cross-machine bit-identity beyond gcd without re-measuring the design.
        """
        d = asdict(self)
        if not d.get("constraints_id"):
            d.pop("constraints_id", None)
        if not d.get("flow_cfg") or d["flow_cfg"] == default_config(FLOW_KNOBS):
            d.pop("flow_cfg", None)  # the default flow keeps its pre-v2 keys (and cached results)
        blob = json.dumps(d, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class EvalResult:
    ok: bool
    stage: str
    proxy_hpwl: Optional[float] = None  # DREAMPlace's own cost. NOT a good ranker (Spearman -0.29 vs routed WL on gcd)
    grt_wl: Optional[float] = None  # global-route wirelength estimate (Spearman +0.99 vs final routed WL on gcd)
    routed_wl: Optional[float] = None  # um, from global/detailed route
    wns: Optional[float] = None  # worst negative slack, ns
    tns: Optional[float] = None
    power: Optional[float] = None  # W
    area: Optional[float] = None  # um^2 instance area
    drc_violations: Optional[int] = None  # detailed-route DRC count (no signoff deck on NanGate45)
    runtime_s: float = 0.0
    error: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "EvalResult":
        d = json.loads(s)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def failure_signature(r: EvalResult) -> Optional[str]:
    """Coarse signature for a failed or poor result to identify recurring failure shapes.
    Normalises away exact values to identify 'classes' of failure (e.g. major timing gap,
    high overflow, or specific DRC rule sets).
    """
    # Successes (clean DRC and clean-ish timing) have no failure signature.
    if r.ok and (r.drc_violations is None or r.drc_violations == 0) and (r.wns is None or r.wns >= -0.01):
        return None

    parts = []
    # 1. DRC rule classes if available in provenance
    drc_classes = r.provenance.get("drc_classes")
    if drc_classes:
        parts.append("drc:" + ",".join(sorted(drc_classes)))
    elif r.drc_violations:
        if r.drc_violations < 10:
            parts.append("drc:few")
        elif r.drc_violations < 100:
            parts.append("drc:many")
        else:
            parts.append("drc:massive")

    # 2. Coarse WNS bucket
    if r.wns is not None:
        if r.wns < -1.0:
            parts.append("wns:major")
        elif r.wns < -0.1:
            parts.append("wns:minor")
        elif r.wns < 0:
            parts.append("wns:clean-ish")

    # 3. Coarse overflow bucket from DREAMPlace
    overflow = r.provenance.get("dp_overflow")
    if overflow is not None:
        if overflow > 0.15:
            parts.append("overflow:high")
        elif overflow > 0.1:
            parts.append("overflow:med")

    # 4. Error message if not OK
    if not r.ok and r.error:
        msg = r.error.split(":")[0].split("\n")[0][:30].strip()
        parts.append(f"err:{msg}")

    return "|".join(parts) if parts else "other:fail"


def score(r: EvalResult) -> float:
    """Scalar objective, lower is better. Failed evals get +inf.

    Deliberately simple and documented so the paper can state it: normalised
    routed wirelength plus a heavy penalty for negative slack and DRC violations.
    """
    if not r.ok:
        return float("inf")
    wl = r.routed_wl if r.routed_wl is not None else (r.grt_wl if r.grt_wl is not None else r.proxy_hpwl)
    if wl is None:
        return float("inf")
    pen = 0.0
    if r.wns is not None and r.wns < 0:
        pen += 10.0 * abs(r.wns)
    if r.drc_violations is None:
        # Verdict must come from the parsed report, never a bare exit code: a route whose DRC
        # count failed to parse is UNKNOWN, not clean. Treat it as worse than any observed violation count so
        # it never silently outranks a real, measured result.
        pen += 1e6
    elif r.drc_violations:
        pen += 0.01 * r.drc_violations
    return float(wl) * (1.0 + pen)


def _wl(r: EvalResult) -> Optional[float]:
    return r.routed_wl if r.routed_wl is not None else r.grt_wl


def score_timing(r: EvalResult, ref: Optional[Dict[str, float]] = None) -> float:
    """Timing-first objective, lower is better: -TNS (ns) + 100 * -WNS (ns).

    +inf if the eval failed, a timing number is missing, or detailed route reported DRC violations. `ref` is accepted
    for a uniform signature but does not change the score (it is only used by passes()).
    """
    if not r.ok or r.wns is None or r.tns is None:
        return float("inf")
    if r.drc_violations is not None and r.drc_violations > 0:
        return float("inf")
    return -float(r.tns) + 100.0 * -float(r.wns)


def passes(r: EvalResult, ref: Dict[str, float]) -> bool:
    """Does `r` match or beat the default-flow reference `ref` ({wns, tns, wl}; mean at the same stage)?
    wns >= ref.wns, tns >= ref.tns, and (grt/routed) WL <= 1.03 * ref.wl when both are known."""
    if score_timing(r) == float("inf"):
        return False
    if r.wns < ref["wns"] or r.tns < ref["tns"]:
        return False
    wl, rwl = _wl(r), ref.get("wl")
    if wl is not None and rwl is not None and wl > 1.03 * rwl:
        return False
    return True
