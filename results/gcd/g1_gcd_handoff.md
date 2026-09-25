# G1: DREAMPlace -> OpenROAD handoff on gcd / NanGate45 (PASS)

Machine: chia-dev (g2-standard-8, 1x L4). OpenROAD/ORFS: `openroad/orfs:latest`. DREAMPlace: limbo018/DREAMPlace built with
CUDA 12.9, torch 2.9.1 (CMAKE_CXX_ABI=1). Evaluator: `pnr_node/raw_handoff.py` (stock ORFS to stage 3_2, DREAMPlace GP
spliced in place of ORFS `global_place`, then stock resize / OpenDP / CTS / global+detailed route / report).

| run | proxy HPWL | routed WL (um) | WNS (ns) | TNS (ns) | power (mW) | area (um^2) | DRC | wall |
|---|---|---|---|---|---|---|---|---|
| stock ORFS (RePlAce), 1 run | - | 4429 | -0.1616 | -7.09 | 6.33 | 943.5 | 0 | 215 s |
| DREAMPlace, defaults, seed 0 | 14138 | 3896 | -0.1607 | -7.18 | 5.76 | 863.7 | 0 | 123 s |
| DREAMPlace, density 0.6, gamma 8, seed 3 | 15574 | 4015 | -0.1644 | -7.07 | 5.95 | 893.0 | 0 | 123 s |

Evidence that OpenROAD did not re-place: stage 3_3 log has `read_def -incremental` and 0 GPL lines; only the stock
pre-handoff stage 3_1 mentions GPL; OpenDP average displacement 1.2 um (max 5.7 um in the earlier attempt).
Caveat: single run per config, no repeats yet; baseline and DREAMPlace differ in other ORFS-internal ways (e.g. port
buffering happens before dump in both). Do not claim the 12% wirelength gain as a result until repeated with seeds.

Bug found and fixed: `read_def -incremental` on DREAMPlace's full DEF re-reads PINS and aborts detailed route with
DRT-0302 (multiple pins on bterm). Fix: import a components-only DEF (`components_only_def`). Same applies to any Hammer
hook that uses `read_def -incremental`.
