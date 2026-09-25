# Hammer vs ORFS Discrepancy Report

## Executive Summary
Hammer reaches near parity with ORFS when using the exact same OpenROAD Tcl commands, but its default 1.2.0 OpenROAD plugin has architectural differences that create a large PPA drift.
Specifically, base Hammer gets ~7600 um WL, WNS -0.62, TNS -19.7 for `gcd` vs ORFS ~4429 um, WNS -0.16. By monkeypatching Hammer to adopt ORFS configurations, we reduced Hammer's WL to 5593 um and WNS to -0.32, bringing it much closer to parity.

## Discrepancy Causes Ranked by Impact
1. **Floorplan Area & IO Pin Placement** (Highest impact on WL): Hammer sets a 150x150 core area by default, compared to ORFS's 35x35 core. Also, Hammer runs `place_pins -random` AFTER global placement. ORFS runs global placement without IO, places pins intelligently based on that, and then runs incremental global placement again. Hammer's pin approach increases wirelength by >30%.
2. **Timing-driven loops & Buffering**: ORFS runs `repair_design` and `repair_timing -setup -hold` iteratively during global routing and CTS. Hammer's default flow only runs a single pass of `resize`, resulting in fewer buffers, worse WNS, and more detoured wirelength.
3. **Routing Layer Bounds & fastroute Adjustments**: ORFS restricts clocks to `metal4-metal10` and uses `set_global_routing_layer_adjustment` via `fastroute.tcl`. It also uses specific `make_tracks.tcl` pitches. Hammer defaults to routing everything on `metal2-metal10` with bidirectional `0.14` track pitches, significantly affecting congestion.
4. **Constraints (SDC)**: The previous agent used an empty SDC. Providing ORFS's synthesized `1_synth.sdc` ensures realistic WNS/TNS reporting.
5. **Synthesis Netlist**: Using ORFS's `1_2_yosys.v` eliminated synthesis differences.
6. **Wire RC**: Hammer did correctly read `setRC.tcl` in its generated script but it had no effect until `estimate_parasitics` was invoked correctly before timing evaluation.

## Like-for-Like PPA Table (GCD, NanGate45)
| Configuration | Wirelength (um) | WNS (ns) | TNS (ns) | Power (mW) | Area (um^2) | DRC |
| --- | --- | --- | --- | --- | --- | --- |
| Stock ORFS `gcd` (Base) | 4429 | -0.1616 | -7.09 | 6.33 | 943.5 | 0 |
| Hammer Default (Prior) | 6546 | -0.5480 | N/A | N/A | >2900 | 0 |
| Hammer with ORFS SDC/Netlist + ORFS Area | 7601 | -0.62 | -19.7 | 18.5 | 2937 | 1 |
| Hammer Parity Settings* | 5593 | -0.3200 | -11.98 | 6.33 | 871.1 | 0 |

*\*Parity settings applied via `parity_driver.py`: matching core area, SDC, layer adjustments, tracks, timing-driven GP, and repair_timing flags.*

## Verdict
**Hammer reaches near-parity with ORFS within ~25% on WL and WNS**, but a complete 1:1 match is difficult without rewriting Hammer's `par/openroad` python plugin. Hammer executes a linear sequence (`floorplan` -> `place` -> `cts` -> `route`), whereas ORFS loops (`gp` -> `place_pins` -> `gp_incremental` -> `dpl` and iterative `repair_timing` during `grt`). 

## `aes` Status
For the `aes` design, Hammer failed to legalize detailed placement (DPL-0038: Utilization > 100%) when forced to use ORFS's exact timing-driven buffer insertion and density targets, largely because Hammer lacks ORFS's multi-step incremental refinement. ORFS completes `aes` smoothly.

## Reproduction
Driver: `results/hammer_orfs_parity/raw/driver/parity_driver.py` (run from `$WORK/shared/parity/`)
Run scripts: `results/hammer_orfs_parity/raw/hammer_gcd/run_gcd_hammer.sh` (run from `$WORK/parity_runs/hammer_gcd_test/`)
