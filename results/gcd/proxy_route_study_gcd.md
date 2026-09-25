# Proxy-vs-route study on gcd / NanGate45 (2026-09-21)

Purpose: the two-tier funnel only helps if the cheap tier ranks configurations like the full route does.

Method: Latin-hypercube samples of the DREAMPlace knobs (`pnr_node/study.py`); DREAMPlace on the L4 (chia-dev), then every valid
placement imported and routed through stock ORFS (chia-cpu1, 4 parallel x 2 cores), seed 0 for every config. Raw: `place.json`,
`routed.json`, per-run ORFS logs on the VMs (`~/work/study_gcd{2,3}`, `~/work/study_cpu`).

## Feasibility of the knob space (DREAMPlace only)
| sample | n | converged | rejected before any route |
|---|---|---|---|
| full ranges (target_density 0.4-1.0, density_weight 1e-6..1e-2, gamma 1-20, stop_overflow 0.05-0.3, lr 0.001-0.05, bins 0-1024) | 160 | 33 (21%) | 127 (79%) |
| narrow box around the defaults (td 0.6-1.0, dw 1e-5..1e-3, gamma 2-12, stop_ov 0.05-0.15, lr 0.004-0.025, bins auto) | 60 | 60 (100%) | 0 |
`routability_opt=1` crashed DREAMPlace in all 20 tries (no routing capacities for this enablement) and was removed.

## Proxy vs routed, narrow box (n = 60 routed, 0 failures, 0 DRC)
| cheap signal | Spearman vs final routed WL | Spearman vs score | top-3 recall | top-10 recall |
|---|---|---|---|---|
| DREAMPlace HPWL (the "proxy") | **-0.29** | -0.27 | **0.00** | 0.00 |
| Global-route wirelength estimate (stage 5_1) | **+0.99** | +0.97 | 1.00 | 0.90 |
| Global-route wirelength (stage 5_1) | +0.93 | +0.92 | 0.67 | 0.90 |
Routed WL over the 60 configs: min 3720, max 5874, mean 4216, std 393 (range 51% of mean, well above the ~3% seed noise).
Why the proxy misleads: target_density vs HPWL Spearman -0.83, vs routed WL +0.46. Dense placements minimise HPWL and route worse.
Other knobs vs routed WL: gamma +0.15, stop_overflow +0.25, density_weight -0.13, learning_rate +0.02.

## Cost per stage on gcd (median seconds, 2-core containers, 4 in parallel)
CTS 6.9, global route 29.0, detailed route 21.7, everything else < 3 each; stopping after global route saves ~40% of the flow.
Ariane's split is not known yet (its stock run is being measured).

## Caution on "improvement"
Best of 60 random configs = 3720 um. Seed-lottery expectation for the best of 60 draws at the default config
(mean 4048, sigma 130 from n = 5) is about 3749. So the best random config is indistinguishable from a seed lottery, and the mean of
random configs (4216) is worse than the seed-lottery mean (4048). A knob search has to beat ~3750 by a clear margin to count.
n = 60 single-seed points; the seed-lottery numbers come from n = 5, so treat them as rough.
