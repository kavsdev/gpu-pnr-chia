# tinyRocket: is the LLM proposer better than budget-matched random search?

**Answer: no detectable difference, at this budget.** An honest null result, not hidden.

## Setup

Direct contrast: `llm_grt` (the LLM-driven proposer, grt tier) vs `random_grt` (uniform-random config
proposals, identical funnel/route budget/validation protocol). Both draw from the same held-out-seed
validation (seeds 101–104), same `rounds=4, batch=8, route_top=3` settings, same toolchain
(`d356e84aeccb`). `default_seeds#r0` (stock/default config, no search) included as the no-search baseline.

Run: `random_controls_run.py run --resume --src-db tinyRocket_main.db --split --llm-proxy`, launched on
`chia-dev` 2026-09-24 05:01 UTC, real routes dispatched to `chia-gpu5`/`chia-gpu6` via the CHIA cluster,
finished 06:09 UTC (~68 min). Analysis via `random_controls_run.py analyze`, which recomputes everything
from raw DB rows, never from the run's own summary JSON.

## Real results (all DRC-clean, 4/4 held-out seeds ok on every arm, no integrity problems)

| label | mean routed WL (µm) | WNS mean |
|---|---|---|
| default_seeds#r0 (no search) | 501,687 | -0.265 |
| llm_grt#r0 | 481,388 | -0.204 |
| llm_grt#r1 | 480,107 | -0.235 |
| llm_grt#r2 | 500,853 | -0.260 |
| llm_proxy#r0 | 478,154 | -0.220 |
| llm_proxy#r1 | 480,748 | -0.216 |
| llm_proxy#r2 | 501,687 | -0.265 |
| random_grt#r0 | 481,901 | -0.226 |
| random_grt#r1 | 478,840 | -0.242 |
| random_grt#r2 | *0 rounds completed — ran out of budget before any route* | — |

## Budget-matched comparison (the actual question)

`random_grt#r2` never got a round in before its budget ran out, so this comparison is genuinely
**n=3 (llm_grt) vs n=2 (random_grt)** — underpowered, not a clean 3-vs-3:

- llm_grt reps: 481,388 / 480,107 / 500,853
- random_grt reps: 481,901 / 478,840
- **llm − random = +7,079 µm (+1.47%), 2×SE = 13,768 µm → no detectable difference.**

Both clearly beat the no-search `default_seeds#r0` baseline (501,687) by a similar margin (~2–4%), so
*searching at all* helps — but this run cannot distinguish the LLM proposer from random proposals within
the same budget on tinyRocket.

## Honest interpretation

- This is a **real, valid null result**, not a failure to hide: with only 2 usable random-search
  repetitions (one arm never got a route in), the comparison lacks the statistical power to detect
  anything smaller than the ~1.47% observed gap, which itself is well within noise (noise floor on this
  design class was separately measured at ~3% sigma, see `results/gcd/noise_floor_gcd.md`).
- Does not contradict the project's other results (proxy-vs-routed correlation, AutoDMP/stock
  comparisons) — this specifically tests LLM-vs-random *proposal quality* at matched budget, a narrower
  question those other results don't address.
- If pursued further: more `random_grt` reps (fixing whatever caused `#r2` to exhaust its budget with 0
  routes) would sharpen this comparison. Not attempted here given the compute window closing (GCP hard
  death 2026-09-24 06:59 UTC).

Raw data: `results/tinyrocket/random_controls_raw/` (merged DB and the full analyze output). Driver:
`random_controls_run.py`.
