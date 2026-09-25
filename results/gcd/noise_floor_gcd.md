# Noise floor on gcd / NanGate45 (measured on chia-dev, 2026-09-21)

Same design, same DREAMPlace config, only the DREAMPlace seed changes.

| check | result |
|---|---|
| DREAMPlace, same seed, x5 | HPWL 14138.34 and 324 iterations every time (deterministic on the L4) |
| DREAMPlace, seeds 1..8 | HPWL 14138..14142 (CV 0.01%): seed barely moves the placer's own cost |
| OpenROAD/ORFS route, identical input, x3 (separate work dirs) | bit-identical: WL 3896, WNS -0.160685, TNS -7.17757, power 5.76063 mW, area 863.702 |
| ORFS route, seeds 11..14 (near-identical placements) | WL 4146 / 3956 / 3985 / 4256; WNS -0.164..-0.169; area 866..903 |

Route spread across seeds (n=4, plus seed 0 = 3896, so n=5 WL: 3896 4146 3956 3985 4256): mean ~4048, std ~130 (3%), range 8%.
Power range 7%, area range 4%, WNS range 3%, DRC always 0.

Reading: the flow is deterministic; the spread is chaotic amplification of tiny placement differences by
resize/CTS/legalize/route. Consequences: (1) cache hits are exact; (2) any claimed improvement below ~2 sigma (~6-8% WL on gcd)
is indistinguishable from a seed lottery; (3) the baseline needs a seed-lottery arm at equal route budget; (4) the earlier
"12% better than stock ORFS" (3896 vs 4429) and the density-0.6 result (4015) are NOT established effects.

## Addendum: thread-count check (2026-09-21, after the SCRIPTS_DIR refactor)
The refactored evaluator (per-run scripts copy via `make SCRIPTS_DIR`, native/docker modes, tool fingerprint) was re-run on
gcd, seed 0, default knobs, in a fresh work dir with the ORFS container limited to 2 cores (`--cpus 2`, NUM_CORES=2), versus
the earlier 8-core runs. Result: bit-identical (WL 3896, WNS -0.160685, TNS -7.17757, power 5.76063 mW, area 863.702, DRC 0);
wall time 213 s at 2 cores vs 123 s at 8. So on gcd, OpenROAD thread count did not change the result. n=1 config, one design:
do not generalise to larger designs without a repeat.
