# gcd / NanGate45: 5-arm search comparison (chia-dev + chia-gpu3 cluster, 2026-09-22)

Protocol: `pnr_node/experiment.py`. 4 rounds x batch 8 x route_top 3 = 12 detailed routes per arm-run, 3 repeats per search
arm (1 for `default_seeds`, which IS the repeat mechanism). Search seed 0 throughout. Each arm-run's single best config
(by `score()`, lower=better) is re-routed on 4 held-out seeds (101-104) never seen during search; comparisons use that
validated mean, never the search-time "best" (winner's-curse bias). One run corrupted by a transient infra bug (GPU-less
worker added to the live cluster) was caught and repaired via `pnr_node.experiment --repair` — final numbers below have 0
infra failures across 477 proxy + 257 route + 208 global-route evaluations.

## Seed-lottery baseline
Default knobs (target_density 0.8, density_weight 8e-5, gamma 4.0, stop_overflow 0.1, learning_rate 0.01), routed on 16
seeds: mean score 10583, sigma 518 (4.9%), best 9826, worst 12275. This defines the noise floor; a result is only
credited if it clears this baseline's mean by more than 2 sigma (~1036).

## Result

| arm | runs | validated mean | vs default | clears 2 sigma? |
|---|---|---|---|---|
| default_seeds | 1 | 10471 | (baseline) | - |
| random_proxy | 3 | 11140 | +5.3% (worse) | no |
| random_grt | 3 | 10938 | +3.4% (worse) | no |
| llm_grt | 3 | 10632 | +0.5% (worse, within noise) | no |
| llm_proxy | 3 | 10522 | -0.6% (within noise) | no |

**No arm clears the noise floor on gcd.** This is a genuine negative result, not a bug: the default DREAMPlace
configuration is already close to optimal for this small design (734 cells), so there is little headroom for a search
to find.

## Why the arms differ from each other (chosen configs, `gcd_main.json`)
- **`llm_proxy` converges back to the default** in 2 of 3 runs (`target_density=0.8, density_weight=8e-5, gamma=4.0,
  stop_overflow=0.1, learning_rate=0.01` exactly). Because the proxy prefers high density (it minimizes HPWL, not
  routed WL; see `results/gcd/proxy_route_study_gcd.md`), and the default's proxy score is already competitive, the LLM
  with proxy-only feedback has no signal telling it to move away. It matches the baseline almost exactly because it
  is *effectively re-discovering* the baseline.
- **`random_proxy`/`random_grt` are measurably worse**, because uniform random sampling over the knob box pulls
  target_density toward the top of its range (0.85-0.96 in the chosen configs) more often than the narrow region
  around 0.8 where the default sits, and high density routes worse (the anti-correlation documented in
  `results/gcd/proxy_route_study_gcd.md`).
- **`llm_grt` explores a wider region** (target_density 0.78-0.90, gamma 4.6-6.0) using real routed/global-route
  feedback, but on gcd does not find anything that beats the default outside noise.

## Reading
On this design, agent-driven search with real routed feedback does not do measurably worse than baseline (unlike
blind random search, which does), but it also does not do measurably better: gcd's defaults are already close to
locally optimal, so the interesting test is a design with more placement freedom (macros, larger cell count) where
the default config is less likely to already be near-optimal. See `results/` for aes/tinyRocket/Ariane133, where the
same arms comparison has not yet been run (single-config feasibility only so far).

## Caveats
- n=3 repeats per arm, n=4 held-out validation seeds per arm-run: small samples, sd-across-runs is itself noisy
  (72-370 depending on arm).
- One design only; do not generalize to macro-heavy or larger designs without rerunning.
- The knob box was narrowed to the region where DREAMPlace reliably converges (see `pnr_node/contract.py`), which is
  centered near the default — this may itself limit how far any arm can move from the baseline. A wider, rejection-
  tolerant search (using the pre-route feasibility check to survive more excursions) is untried.
