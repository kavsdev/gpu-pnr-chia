# Hammer-wrapped OpenROAD vs. direct ORFS integration (gcd, NanGate45)

**Status: disclosed negative result on WL/timing, one design (gcd), independently reproduced (Hammer side n=2).**
Raw evidence pulled into `results/hammer_orfs_parity/raw/` from `chia-cpu1` on 2026-09-24 (before GCP cutoff); an
independent re-run on a fresh VM (`chia-gpu7`, 2026-09-24) is in `raw/repro_gpu7_20260924/`. This file supersedes
the original agent-written `raw/driver/REPORT.md`, which had an overstated verdict.

**Reproduction result (2026-09-24, chia-gpu7):** the Hammer par run reproduced **bit-identically** — routed
WL = 5593, DRC 115→85→49→0. The re-run also **actually measured** Hammer PPA (the cpu1 run left no measurement
file behind), giving WNS -0.32, TNS -11.98, power 6.34 mW, inst_area 871.15 µm² — all matching the original agent
report. This **overturns my initial "fabricated power" flag**: the 6.33 mW cell was *unsupported at the time* (no
raw evidence on cpu1) but is in fact **correct** — Hammer and ORFS have near-identical power. See "What the
reproduction changed."

## Claim as we report it

> Even after hand-tuning Hammer's OpenROAD plugin to match ORFS's configuration as closely as possible, driving
> OpenROAD *through Hammer* produces ~26% worse routed wirelength and roughly 2× worse worst-negative-slack than
> our direct ORFS integration on gcd, and fails outright on aes. We therefore keep the direct ORFS splice as the
> primary path and treat the Hammer wrapper as a parked, disclosed negative result.

This is **not** a default-vs-default benchmark, and DREAMPlace was **never** wired through the Hammer path — the
comparison is *Hammer-native OpenROAD placement* vs. *direct ORFS*, both using OpenROAD as the backend engine.

## Verified numbers (from raw OpenROAD metrics, not the agent prose)

All Hammer values below are now independently re-measured on chia-gpu7 (`raw/repro_gpu7_20260924/`), not agent prose.

| Metric | ORFS (direct) | Hammer (hand-tuned to ORFS) | Delta | Verified from |
|---|---|---|---|---|
| Routed wirelength (µm) | **4429** | **5593** | **+26.3% worse** | both raw route metrics JSON; Hammer reproduced bit-identically on a 2nd VM |
| Detailed-route DRC | **0** | **0** | tie | both converge (ORFS 63→18→10→0; Hammer 115→85→49→0) |
| WNS (ns) | **-0.1616** | **-0.32** | **~2× worse** | ORFS raw `6_report.json`; Hammer `report_checks` in fresh `hammer_measure_ppa.log` |
| Setup TNS (ns) | **-7.086** | **-11.98** | **~1.7× worse** | same |
| Total power (mW) | **6.334** | **6.339** | **≈ tie** | ORFS raw (`6.33e-03 W`); Hammer fresh `report_power` (`Total 6.3387e-03 W`) |
| Stdcell/inst area (µm²) | **943.5** | **871.2** | Hammer smaller | ORFS raw; Hammer fresh `measure_ppa` (`inst_area 871.15`) |

**Bottom line, all raw-data-backed:** Hammer is **~26% worse on WL** and **~2× worse on WNS/TNS**, with **near-identical
power** and a **slightly smaller area** — the smaller area + worse timing are the same story (Hammer inserts fewer
timing-repair buffers than ORFS's iterative `repair_timing`, so it saves area but misses timing). Power is
cell-dominated on gcd, so it barely moves with the wirelength difference — that is why both land at ~6.33 mW.

> Note on the fresh `measure_ppa.log`: `measure_ppa.tcl`'s own summary lines misparse (they print `WNS: 0.0`,
> `total_power: 0.0` — a tokenizer bug in that script). The **true** values are in the raw `report_checks` /
> `report_power` tables it emits just above (`wns max -0.32`, `tns max -11.98`, `Total … 6.3387e-03 W`). Trust the
> report tables, not the script's summary echo.

## Corrections applied vs. the original agent report

**(a) Reframed as hand-tuned-toward-parity, not default-vs-default.** The 5593 number was only reached after
monkeypatching Hammer with six ORFS-specific settings (core area, ORFS SDC, ORFS synthesized netlist, routing-layer
adjustments, track pitches, and timing-driven GP + `repair_timing` flags — see `raw/driver/parity_driver.py`).
Out-of-the-box Hammer is far worse: 6546 µm (prior default) and 7601 µm (ORFS SDC/netlist but Hammer's own flow),
per the report's own table. So the honest framing is "*even forced toward ORFS's config, Hammer is still ~26%
worse*," which is a stronger and more defensible statement than a raw default comparison.

**(b) Power was unverifiable at first, now measured and confirmed (near-identical to ORFS).** When only the cpu1
artifacts existed, the report's Hammer power = 6.33 mW had **zero** raw backing — `6.33` appeared only in ORFS logs
and the cpu1 run had left no `measure_ppa` output, so I flagged it as fabricated and pulled it. **The chia-gpu7
re-run re-measured it properly: Hammer power = 6.339 mW** (`report_power` Total `6.3387e-03 W`), vs ORFS 6.334 mW.
So the value was *right all along* — Hammer and ORFS have effectively identical power on gcd, because total power
here is dominated by standard-cell switching (same netlist, same clock/activity) and wirelength contributes only a
small wire-capacitance term. **Power is therefore not a differentiator and should not be presented as one** — the
real gaps are WL and timing. (Lesson: "no raw evidence" justified distrust, but distrust is not disproof; the fix
was to measure, not to assume.)

**(c) Scoped as a disclosed negative result on one design, now reproduced.** gcd only; the Hammer side is now n=2
(cpu1 + chia-gpu7, bit-identical WL 5593), the ORFS side is the established baseline (reproduced many times this
project). The 26% WL gap is far outside the ~3% seed sigma measured for our DREAMPlace/ORFS path
(`results/gcd/noise_floor_gcd.md`); that noise floor was established for the direct path, not Hammer, so we still
present this as a single-design directional finding, not a multi-seed benchmark. **aes is a crash, not a
measured data point:** under the same forced ORFS-style density/buffering, Hammer's detailed placement hit 105.6%
utilization and aborted with `[ERROR DPL-0038] Utilization greater than 100%, impossible to legalize`
(`raw/hammer_aes_fail/hammer_aes_DPL0038.log:799`); ORFS completes aes cleanly. Report as "Hammer failed to
legalize aes," not as "Hammer aes is worse."

## Why Hammer is worse (mechanistic, from the report + logs, credible)

1. **IO-pin / floorplan handling** (largest WL effect): Hammer places pins with `place_pins -random` *after* global
   placement; ORFS does GP → intelligent pin placement → incremental GP, so pins sit where the netlist wants them.
2. **Single-pass resize vs. ORFS's iterative `repair_design`/`repair_timing`** during GRT and CTS → fewer buffers,
   worse WNS, more detour. Directly visible in the Hammer route log: `RSZ-0062 Unable to repair all setup
   violations`, 35 endpoints left.
3. **Routing-layer bounds / fastroute adjustments** differ from ORFS's `fastroute.tcl` layer adjustments.

These are architectural differences in Hammer's `par/openroad` 1.2.0 plugin, not tuning we failed to apply — which
is exactly why it stays parked rather than pursued.

## What the reproduction changed (2026-09-24, chia-gpu7)

- **Confirmed** WL 5593 / DRC→0 exactly (bit-identical → Hammer side is now n=2, deterministic).
- **Newly raw-verified** WNS -0.32, TNS -11.98 via a real `report_checks` on the routed DB (previously only
  directional via the RSZ-0062 log).
- **Overturned** the "fabricated power" call: measured Hammer power = 6.339 mW ≈ ORFS 6.334 mW. Power is a tie,
  not a gap — dropped as a differentiator for the right reason (measured), not as a suspected fabrication.
- **Confirmed** Hammer inst_area 871.15 µm² (smaller than ORFS 943.5 — fewer timing-repair buffers, consistent
  with worse WNS).

## Caveats / limitations (state these if it goes in the paper)

- One design (gcd, 734 cells, no macros). Hammer side reproduced (n=2, deterministic); still no multi-seed
  statistical claim — but the 26% WL gap dwarfs the ~3% seed sigma, so it is not noise.
- All Hammer PPA is now raw-verified from the fresh run; quote WL/WNS/TNS as hard numbers, and state power as
  "≈ equal (~6.33 mW)".
- Original run had single-agent (Gemini) provenance; this file + the chia-gpu7 re-run are the audited version.
- DREAMPlace was never routed through Hammer, so no claim is made about the full DREAMPlace-via-Hammer pipeline.

## Reproduction

Driver and configs are in `results/hammer_orfs_parity/raw/`:
- `driver/parity_driver.py` — the Hammer monkeypatches that force ORFS-style config.
- `driver/measure_ppa.tcl` — post-route PPA extraction. **Known bug:** its printed summary lines (`WNS:`,
  `total_power:`) misparse to 0.0; read the real values from the `report_checks`/`report_power` tables it emits
  above them.
- `hammer_gcd/{run_gcd_hammer.sh, hammer_wrapper.sh, design.yml, tech.yml, tools.yml, orfs_config.mk}` — Hammer run
  inputs.
- `repro_gpu7_20260924/parity_repro_driver.sh` equivalent: the full independent re-run (re-run Hammer par, then
  measure the routed DB `build/par-rundir/latest`). Reproduce with: stage these inputs under `~/work/parity_runs`
  on any golden-image VM (has `chia-pnr:hammer`), run `run_gcd_hammer.sh`, then run `measure_ppa.tcl` with
  `ODB_FILE=.../par-rundir/latest`.
- ORFS baseline is plain `make DESIGN_CONFIG=.../gcd/config.mk` (nangate45), captured in
  `orfs_gcd_baseline/orfs_gcd.log`.

## Raw evidence index (`results/hammer_orfs_parity/raw/`)

| Path | What it proves |
|---|---|
| `hammer_gcd/metrics-write_design.json` | Hammer routed WL = 5593, DRC converged to 0 (the +26% claim) |
| `hammer_gcd/route_openroad.log` | Hammer detailed route; RSZ-0062 unrepaired setup violations (worse WNS) |
| `orfs_gcd_baseline/5_2_route.json` | ORFS routed WL = 4429, DRC 0 |
| `orfs_gcd_baseline/6_report.json` | ORFS WNS -0.1616, TNS -7.086, power 6.33 mW, area 943.5 µm² |
| `orfs_gcd_baseline/6_report.log` | Human-readable ORFS power/area (`6.33e-03 W`) — matches the independently re-measured Hammer 6.339 mW |
| `hammer_aes_fail/hammer_aes_DPL0038.log` | Hammer aes abort: 105.6% util, DPL-0038 impossible to legalize |
| `driver/REPORT.md` | Original agent report (superseded — audited here; its numbers are correct, its verdict was overstated) |
| `driver/parity_driver.py`, `driver/measure_ppa.tcl` | Reproduction driver + measurement |
| `repro_gpu7_20260924/SUMMARY.txt` | Fresh-run headline extract (WL 5593, DRC 0, area 871.15) |
| `repro_gpu7_20260924/hammer_measure_ppa.log` | **Independent** Hammer measurement: `wns max -0.32`, `tns max -11.98`, `report_power` Total `6.3387e-03 W` |
| `repro_gpu7_20260924/hammer_metrics-write_design.json` | Fresh Hammer route metrics: WL 5593, DRC 115→85→49→0 |
| `repro_gpu7_20260924/flow.log` | Full fresh-run transcript on chia-gpu7 |
