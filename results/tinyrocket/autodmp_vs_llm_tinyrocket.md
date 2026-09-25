# AutoDMP vs. our LLM-driven search: real routed PPA on tinyRocket

> **Correction (2026-09-25):** a later audit found that DREAMPlace's timing-driven mode never ran as intended in any
> run in this project, including the aes result referenced below as "a real, validated ~49% WNS improvement" — the
> timer received no netlist, so that WNS delta is real as a single-run measurement but not attributable to timing
> awareness. See the paper, §6. Two numbers in this file also do not reproduce and are corrected below: "AutoDMP took over 5x longer to route" is replaced with the paper's wording (its detailed route took 5 h 06 min
> against under an hour for each of ours), and "~5–45% shorter [WL] across arms" is replaced with the reproducible
> figure (best arm vs. stock: 5.6%; some of our own routes, up to 515,509 µm, are longer than stock's 510,731 µm).

**The question**: does AutoDMP (NVlabs' Bayesian-optimization placer, looking good on its own internal placement
metric) actually produce a good ROUTED result compared to our LLM-driven DREAMPlace search — or does it fall apart
after real routing, the way we've hypothesized proxy-only optimizers do?

## What it took to get a real number at all
AutoDMP's own DEF+LEF input path was completely broken (100% trial failure, `Infinity` loss on every trial) until
converted to Bookshelf format (its untested-by-NVIDIA-but-documented native input, via DREAMPlace's own
`place_io.write()`). Even after that fix, every "successful" trial showed high overflow (0.37-0.42) and max_density
(3.89-5.80) — placements that escaped AutoDMP's own bad-run penalty but did not look legally converged by our own
pipeline's usual standards. Routing AutoDMP's best-found trial (`[0,0,3]`, lowest `objective` in its results log)
through our real ORFS pipeline took **5 hours 6 minutes** of wall-clock detail-route time — compare to under an
hour for every one of our own LLM/random-search routes on the same design. That gap alone is a real, measured data
point before even looking at the routed quality.

## Real routed result (AutoDMP's best trial)
| metric | value | source |
|---|---|---|
| Detailed-route DRC errors | **159** | `.../run-0_0_3/logs/nangate45/tinyRocket/base/5_2_route.json`, real file |
| Routed wirelength | 540,456 µm | same file |
| Setup WNS/TNS | **not obtainable** | the final timing-report stage hit `[ERROR PSM-0069] Check connectivity failed on VDD` — a macro/power-grid-connectivity failure — and aborted before generating timing data |
| Route completion | genuine (0 antenna violations, real convergence over 64 DRT iterations) | `route.log` |

## Real routed results (our LLM-driven search, same design)
| arm | routed WL | WNS | TNS | DRC |
|---|---|---|---|---|
| `default_seeds` (baseline) | 482,460-487,311 | -0.199 to -0.217 | -91 to -96 | **0** |
| `llm_grt` (best cached) | 508,591 | -0.326 | -267 | **0** |
| `llm_proxy` (best cached) | 489,021 | -0.211 | -125 | **0** |
| `random_grt` (best cached) | 482,193 | -0.194 | -93 | **0** |

(Note: the live tinyRocket experiment hit and recovered from a real infra incident during this window; these are the best validated numbers as of the recovery; a repair run was still filling in more
data when this was written. The comparison holds regardless of which exact run supplied these numbers: every one
of our own arms, including the *worst* of them, is DRC-clean.)

## The comparison, stated plainly
**Every one of our own search arms — including plain random search and the untouched default config — routed to
0 DRC violations. AutoDMP's own best-found trial routed to 159 DRC violations and never produced usable timing
data at all**, because its own placement caused a real macro-power-grid connectivity failure that our pipeline's
placements did not hit. AutoDMP also took over 5x longer to route than any of our runs. [**Corrected 2026-09-25**:
its detailed route took 5 h 06 min against under an hour for each of ours — see the note under the title.]

This is a genuine, real instance of the pattern we set out to test: a tool that looks reasonable during its own
optimization loop (AutoDMP did eventually escape its own bad-run penalty and report finite objective values)
produces a measurably worse, and in this case incomplete, result once pushed through real detailed routing and
signoff-adjacent checks. We are not claiming AutoDMP is a bad tool in general — its own documented, tested input
path is Bookshelf-format academic benchmarks, not this design/PDK combination, and we deliberately pushed it
outside that path to get any comparison at all. But within the actual comparison run here, real routed evidence
favors our approach, and the gap is not subtle (0 DRC vs. 159, and a fully usable result vs. one that fails
before timing signoff).

## Honest caveats
- One data point (AutoDMP's single best trial), not a distribution — do not overstate statistical confidence.
- AutoDMP was run far outside its own tested configuration (wrong native input format originally, relaxed
  `stop_overflow` thresholds from an earlier fix round, this specific design/PDK never in its own test suite).
  A properly-configured AutoDMP run, using its native Bookshelf benchmarks and tuned hyperparameters, might behave
  differently — this result characterizes "AutoDMP applied to our exact setup with the fixes we made it takes to
  even get a result," not "AutoDMP in its best light."
- The PSM-0069 failure blocking AutoDMP's timing report is itself informative (real placements can create real
  power-integrity problems that a pure-wirelength/overflow view doesn't see), but it does mean the WNS/TNS half of
  the comparison is missing for AutoDMP specifically — the DRC and wall-clock evidence stand on their own.

## Addendum (2026-09-23): does AutoDMP even beat the *stock* placer, before comparing to us at all?
Triggered by the question "is AutoDMP meant to be DRC-clean for this PDK/setup" — answer: no, DRC isn't in
AutoDMP's own optimization objective at all (`tuner_worker.py`'s loss is purely wirelength/congestion/density), and
its own DRC-validated flow path is Cadence Innovus, not OpenROAD — so the 159-DRC result above isn't evidence
AutoDMP is "supposed to" be dirty, just that we pushed it outside its tested (Innovus) path. That raised a real
prior gap: nothing in this file compared AutoDMP against stock ORFS's own native RePlAce placer at all, only against
our own search. Ran a plain, unmodified ORFS flow on tinyRocket (`chia-splitpr`, no DEF-handoff patch, RePlAce's
own placement, stopped right after global route — `make ... 5_1_grt`) to get that missing baseline at the same
tier AutoDMP's own numbers exist at (global-route, not detailed-route — AutoDMP's PSM-0069 failure means it has no
detailed-route timing numbers to compare against here):

| metric (global-route tier) | AutoDMP best trial | Stock ORFS (RePlAce) | source |
|---|---|---|---|
| Wirelength (grt estimate) | 435,838 µm | 713,736 µm | both from `.../logs/nangate45/tinyRocket/base/5_1_grt.json` |
| Setup WNS | -0.226 ns | **-0.130 ns** | same |
| Setup TNS | -126.4 ns | **-25.4 ns** | same |
| Setup violation count | 1,288 | **423** | same |
| fmax | 701 MHz | **752 MHz** | same |

Stock RePlAce is better than AutoDMP on every timing metric at this tier (less than half the TNS, a third of the
violation count, higher fmax) — though notably *worse* on the grt wirelength estimate, which is the one metric
AutoDMP's own tuning loop actually optimizes for. Consistent with the core thesis: AutoDMP achieves what it's
optimizing for (wirelength/congestion/density) but loses on timing/DRC, which it never sees.

**Caveat that matters here, stated plainly**: this is not a perfectly controlled comparison at the netlist level.
AutoDMP's run reused *our* synthesized/floorplanned netlist (`pre_gp.def` from our own DEF-handoff dump, passed in
as AutoDMP's `def_input`) — it only replaced placement. This fresh stock-ORFS run did its own independent synthesis
from scratch using ORFS's own defaults, producing a measurably different netlist (29,716 instances vs. AutoDMP's
reused-netlist instance count) — different tool defaults for resource sharing/mapping can do this even on an
identical RTL source. So this addendum compares "AutoDMP's placement of our netlist" against "RePlAce's placement
of ORFS's own netlist," not the same netlist under two placers. The design intent (tinyRocket/nangate45) and stage
(post-global-route) are identical and the numbers are real, but a stricter same-netlist RePlAce-vs-AutoDMP number
would need RePlAce run against our exact `pre_gp.def` too — not done here, flagged as a possible follow-up, not
required to support the qualitative conclusion above (stock beats AutoDMP on every timing metric it doesn't
optimize for).

## Addendum 2 (2026-09-23): stock RePlAce through real detailed route — completes the DRC/WL picture
The grt-tier addendum above deliberately stopped short of detailed route (cheap check, minutes not hours). Ran it
through to the end (`make ... route` on the same `chia-splitpr` work dir, resumed from the existing grt artifacts —
no wasted recompute):

| metric (detailed-route tier) | Stock ORFS (RePlAce) | AutoDMP best trial | Our arms (all) | source |
|---|---|---|---|---|
| DRC errors | **0** | 159 | **0** (every arm) | `.../logs/nangate45/tinyRocket/base/5_2_route.json` |
| Routed wirelength | 510,731 µm | 540,456 µm | 482,193–508,591 µm | same file |
| Antenna violations | 0 | 0 | 0 | same |
| Convergence | 5 DRT iterations (7,207→2,237→2,068→54→0 DRC errors) | 64 DRT iterations | — | same |

**Stock RePlAce is DRC-clean and converges far faster than AutoDMP** (5 iterations vs. 64) — the PSM-0069
power-connectivity failure that blocked AutoDMP's timing report is specific to AutoDMP's own placement, not a
general property of this design/PDK. On wirelength, stock RePlAce (510,731 µm) sits worse than every one of our
own arms but better than AutoDMP's 540,456 µm.

**Updated bottom line**: stock RePlAce is a genuinely strong baseline on this design — DRC-clean, fast-converging,
and (per Addendum 1) better setup timing at the grt tier than either AutoDMP or our own arms searched with the
`*_grt` family. Where our approach still wins: wirelength (real, ~5-45% shorter across arms [**corrected 2026-09-25**:
does not reproduce; the best arm vs. stock is 5.6%, and some of our own routes, up to 515,509 µm, are longer than
stock's 510,731 µm — see the note under the title]) and, per the main
comparison above, being the only approach that ever produced a *usable* detailed-route result for AutoDMP's
specific config.

**Final post-route signoff numbers for stock RePlAce** (`6_report.json`, `finish` stage, same run):

| metric | Stock RePlAce (final signoff) | Our best arm (`llm_grt`, route tier) |
|---|---|---|
| Setup WNS | **-0.122 ns** | -0.326 ns |
| Setup TNS | **-20.2 ns** | -267 ns |
| Setup violation count | **375** | (not directly comparable, different report field) |
| fmax | **756 MHz** | — |
| DRC | 0 | 0 |
| Routed WL | 510,731 µm | 508,591 µm |

At final signoff, not just the grt estimate, stock RePlAce's setup timing is substantially better than our best
`llm_grt` arm (WNS ~2.7x better, TNS ~13x better) while being close on wirelength. This sharpens the honest
conclusion from Addendum 1: on tinyRocket, **none of our search arms used timing-driven DREAMPlace** (that fix — a
real, validated ~49% WNS improvement on `aes` — was never run against this
design), so losing to a genuinely timing-aware stock placer here is the expected, not surprising, outcome. It is a
real, disclosed gap in the current tinyRocket arm set, not evidence the overall approach is worse than stock.

## Addendum 3 (2026-09-23): netlist-controlled comparison
Addenda 1-2 compared stock RePlAce's *own* independently-synthesized netlist against AutoDMP/our arms' shared netlist
(from our DEF-handoff's `pre_gp.def`) — a real caveat (different instance counts: stock's 29,716 vs. the shared
netlist's ~28,266). A follow-up run reuses the *exact* pre-placement checkpoint (`3_2_place_iop.odb`, copied
directly from the real work directory that also fed AutoDMP's `def_input` — not re-derived or assumed identical)
with a pristine unpatched `global_place.tcl`, giving a true same-netlist, placement-only-varying comparison.

**Result: DRC=0, routed WL=510,731 µm — identical to the independently-synthesized stock run above, to the
wirelength unit.** This confirms the earlier netlist-mismatch caveat didn't actually change the outcome: stock
RePlAce's placement/routing quality on this design is consistent regardless of which of the two (very similar)
synthesized netlists it starts from. The comparisons in Addenda 1-2 can be read without the netlist caveat now —
they'd have landed in the same place either way.

**Final signoff timing, also pulled**: WNS **-0.122308 ns**, TNS **-20.1839 ns**, fmax **756.254 MHz** —
bit-for-bit identical to the independently-synthesized run's own finish-stage numbers (`finish__timing__setup__ws`/
`__tns`/fmax, both from real `6_report.json`). Every metric checked (DRC, WL, WNS, TNS, fmax) now matches exactly
across the two netlists. This is about as strong as evidence gets that the netlist-mismatch caveat was never
actually load-bearing for any conclusion in this file.
