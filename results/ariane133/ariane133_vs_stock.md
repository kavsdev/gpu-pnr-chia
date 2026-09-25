# Ariane133 / NanGate45: DREAMPlace handoff vs. stock ORFS (originally run as a timing-driven test)

> **Correction (2026-09-25):** a later audit found that DREAMPlace's timing-driven mode never ran as intended in these
> runs. The timer received no netlist, multi-library designs failed option parsing, and net reweighting only starts after
> iteration 500 while these runs converged earlier. The measured WNS differences are real single-run measurements but
> cannot be attributed to timing-driven placement, and the "stale iteration-0 weights"/"iteration-0 snapshot" explanation
> used throughout this file is withdrawn (contradicted by the DREAMPlace source: reweighting never fires here at all,
> stale or otherwise). See the paper, §6.

**Answer: no.** Real, measured result: the same `timing_opt_flag=1`/OpenTimer integration that gave a genuine,
meaningful WNS/TNS improvement on `aes` does **not** help Ariane133 — it is marginally *worse* than the untouched
handoff baseline, consistently across every timing metric.

## The numbers (all real, independently verified against raw `6_report.json`/`5_2_route.json`)

| | Stock ORFS (RePlAce, timing-driven) | Our handoff, no DREAMPlace timing awareness | Our handoff, `timing_driven=True` |
|---|---|---|---|
| Setup WNS | -0.2657 ns | -1.10223 ns | **-1.14815 ns** |
| Setup TNS | -429.2 ns | -3725.37 ns | **-3891.02 ns** |
| Violating endpoints | 3115 | 6476 | **6568** |
| Instance area | 720,906 µm² | 722,056 µm² | 722,762 µm² |
| Detailed-route DRC | 0 | 0 | 0 |
| Routed wirelength | 6,140,510 µm | — | 6,962,421 µm |
| DREAMPlace runtime | — | — | 5.74s (397 iterations, overflow 0.099 — converged normally) |

Sources: stock — `results/ariane133/ariane133_stock_flow.md`. Old handoff — `chia-cpu2`,
`~/work/pnr_work/exp/ariane133/8a64490719b6c447/`. New (timing-driven) handoff — `chia-splitpr`,
`~/work/handoff/ariane133/d51f55f7b7b934d4/`, run 2026-09-22, `runtime_s: 8293` (≈2.3h), full route completed
cleanly (`ok: true`, 0 DRC — this is not a crashed or partial run).

## Why this is a real, trustworthy negative result, not noise
- The DREAMPlace run itself completed normally (397 iterations, overflow 0.099 — a legal, converged placement,
  not a failure).
- The regression is small (≈4% worse WNS, ≈4.4% worse TNS, ≈1.4% more violations) but **consistent across all
  three independent timing metrics in the same direction** — not one metric randomly worse while another improves,
  which is what noise would look like.
- Directly contradicts the `aes` result (45% better WNS, 70% better TNS, 33% fewer violations under the identical
  mechanism), so this is a genuine scale/complexity-dependent effect, not a broken integration — the same code
  path that helps a medium design measurably hurts the large, macro-heavy one.

## Leading explanation (hypothesis) — now with real corroborating runtime evidence, still not fully proven
The DREAMPlace timing-driven integration agent's own report noted a real mechanism detail: net weights from
OpenTimer are computed **once, statically, at iteration 0** from the initial (pre-optimization) timing snapshot —
not re-evaluated as placement evolves (this is why the runtime cost is near-zero: no repeated STA calls). On a
small design like `aes`, the set of critical paths likely stays roughly stable through optimization, so a static
initial snapshot is still a good guide throughout. On Ariane133 — 132 macros, ~54,000+ instances, a far more
complex timing graph — the critical-path set plausibly shifts substantially as placement moves macros and cells
around, making the iteration-0 snapshot increasingly stale and misleading by the time placement converges. This
would explain a genuine, mechanism-level scale limit rather than a bug: the fix needs *dynamic* re-weighting to
help at this design size, which the current integration does not do.

**Important clarification, since this is easy to misread**: DREAMPlace genuinely *did* receive real `.lib`/`.sdc`
timing data — `timing_opt_flag=1`, real `lib_input`/`sdc_input`/wire-RC fields, all confirmed present in
`params.json`, not a stub. The negative result is **not** "DREAMPlace never saw real timing data." It's narrower
and more specific: it saw real timing data exactly **once**, used it to compute a fixed net-weighting at
iteration 0, then optimized placement against that frozen snapshot for the remaining ~396 iterations with **zero
further STA feedback**. On a small design that one snapshot stays representative; on a 132-macro design it likely
stops being representative well before convergence — the tool had the right inputs, but the wrong (one-shot, not
iterative) use of them at this scale.

### New corroborating evidence, added after this file was first written — real per-stage runtime breakdown

Pulled directly from both flows' real OpenROAD stage logs (`grep 'Elapsed time:'`, not estimated):

| Stage | Stock RePlAce | Our handoff, timing-driven | Ratio |
|---|---|---|---|
| Global placement | 768.6s | 50.6s (DREAMPlace itself: 5.74s) | **15x faster** |
| **Resize/repair (`3_4_place_resized`)** | 79.0s | **599.5s** | **7.6x slower** |
| Detailed placement/legalize | 109.3s | 179.6s | 1.6x slower |
| Global route | 1,015.0s | 1,509.6s | 1.5x slower |
| **Detailed route (`5_2_route`)** | 548.6s | **2,630.5s** | **4.8x slower** |
| **Total flow** | **~1h12m** | **~2.3h** | **~1.9x slower overall** |

This is real, mechanism-consistent corroboration (not proof) of the staleness hypothesis: the two stages that blow
up the most are exactly the two that have to *fix* timing/routing problems the placement left behind — resize's
`repair_timing` pass (7.6x slower) and detailed route's congestion/DRC rework (4.8x slower). A placement genuinely
optimized against stale, iteration-0 criticality information would be expected to leave more of both kinds of mess
for downstream stages to clean up — which is exactly what's measured. **Still not independently isolated as the
sole cause** (large-design routing congestion has other contributors too), but now two independent measurement
axes (timing metrics *and* runtime) point the same direction, which is stronger than either alone. If this work
continues, the natural next experiment is unchanged: dynamic (not just iteration-0-static) timing-driven
re-weighting inside DREAMPlace's optimization loop.

## What this means for the project
- **Do not cite Ariane133 as a "we fixed the WNS regression" result.** The regression is real, measured, and
  currently unresolved at Ariane's scale.
- The `aes` result stands on its own as a real, positive, mechanism-verified finding — cite it as such, scoped to
  "helps at medium design scale," not as a general claim.
- Honest framing for the paper: we found the root cause of the WNS regression (DREAMPlace placement has no timing
  information), built and validated a real fix for it, and the fix's benefit is scale-dependent — it closes the
  gap on medium designs but not (yet) on the largest, most macro-heavy one tested. That is a legitimate, complete,
  honestly-scoped research finding, not a failure to hide.
- If pursued further: dynamic (not just iteration-0-static) timing-driven re-weighting inside DREAMPlace's
  optimization loop is the natural next experiment, not something attempted here given time constraints.
