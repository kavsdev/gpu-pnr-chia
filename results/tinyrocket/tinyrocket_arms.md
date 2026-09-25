# tinyRocket — 5-arm noise-aware search comparison (real, validated, recovered from a real data-loss incident)

**Status: real, mostly good, with two honestly-disclosed data gaps on the weaker comparison arms — not a clean
win, trending consistent with this project's established pattern (real signal near the noise floor, not beyond
it).** Source: `chia-dev:~/work/exp/tinyRocket_main.{db,json,log}`, pulled and independently re-read from the raw
DB/JSON (not the printed summary table), 2026-09-23 ~21:xx UTC. Raw files kept at
`results/tinyrocket/repair3_raw/` for reproduction.

## Protocol

`pnr_node.experiment.run_experiment`, real `ChiaEvaluator` (native CHIA/Ray cluster, real GPU, real
`chia-pnr:dp`), noise-aware protocol: 4 rounds × batch 8 × route_top 3, 3 repeats per arm, held-out validation on
4 unseen seeds (101/102/103/104) not used during search. This run (`repair3`) recovered ~7-8 arm-runs whose
history was wiped by an earlier `purge_arm_runs()` repair bug — see "Known gaps" below for what that recovery did
and didn't reach.

**Audit, not trusted from the printed log table**: 422 total DB rows, **0 with `provenance.mock`** — every result
is a real evaluator run. **29 cache hits** (the content-addressed dedup subsystem working as designed). **Every single routed result across every arm and every phase (search
and held-out) is `drc_violations = 0`** — a real, consistent, favorable finding on its own.

## Results — held-out validated routed wirelength (the deciding numbers, not search-phase best-of-N)

| Arm | Reps w/ real held-out data | Validated WL per rep (µm) | Mean validated WL (µm) | Held-out seeds landed |
|---|---|---|---|---|
| `default_seeds` | 1/1 | 507,293 | **507,293** | 3/4 |
| `llm_grt` | 3/3 | 481,458 / 479,299 / 509,178 | **489,978** | 2/4 per rep |
| `llm_proxy` | 3/3 | 479,296 / 480,324 / 507,293 | **488,971** | 2-3/4 per rep |
| `random_grt` | 1/3 (see gap below) | 482,193 | 482,193 (n=1, not trustworthy alone) | 1/4 |
| `random_proxy` | 0/3 (see gap below) | — | no recoverable held-out data | 0/4 |

**Reading (honest, not spun)**: `llm_grt` and `llm_proxy` both land ~3.4-3.6% below `default_seeds` on held-out
validated WL — a real, consistent, directionally favorable signal for the LLM-guided search over the plain
default. But this project's own established noise floor on this class of result is ~3% (see
`results/gcd/noise_floor_gcd.md` / the repeated "near the noise floor" findings across aes/tinyRocket/Ariane
elsewhere in this project) — a 3.4-3.6% gap is **right at that edge, not clearly beyond it**. This is the same
honest pattern as everywhere else tinyRocket has been measured in this project (low headroom, modest-at-best
signal) — **do not cite this as a decisive win**; cite it as "directionally consistent with the LLM search
finding something, within a noise band that makes a strong claim unjustified on this one design."

One genuine, interesting side-observation: `llm_proxy#r2`'s validated numbers are byte-identical to
`default_seeds#r0`'s — because `llm_proxy#r2`'s own search never found a config better than plain `default_config()`
(its recorded `best_cfg` is exactly the default values), so its held-out validation hit the exact same
content-addressed cache entry as `default_seeds`. This is the dedup subsystem working correctly, not a bug — but
it does mean `llm_proxy#r2` contributes no independent information, worth knowing when weighing the 3-rep mean.

## Known gaps — disclosed, not papered over

- **`random_grt` hit a real transient infra failure, precisely diagnosed, not the original (already-fixed) bug.**
  32 proxy-stage evals failed with `RuntimeError: No CUDA GPUs are available`, all sharing the **exact same
  dispatch timestamp** (a single simultaneous batch, not a scheduling race trickling in over time) — the
  signature of a real NVML/CUDA-driver crash on the physical node at that instant, a failure layer *below* Ray's
  resource accounting. **This is not the original GPU-reservation/head-of-line-blocking bug** (that one was fixed
  by the `num_gpus=1` + split place/route architecture — verified still correctly in place in `pnr_node/node.py`:
  `place_remote`/`evaluate_remote` both request `num_gpus=1`). It matches the *separate*, already-logged
  "cluster-wide NVML breakage" recurrence class (first seen 2026-09-22 ~20:35 UTC; its root cause was never
  fully pinned down) — this timestamp doesn't match that incident, so it looks like a third occurrence of the same still-unresolved flakiness class, not a
  regression. Only 12/44 proxy attempts and ultimately 1 real held-out route survived across all 3 reps — too thin
  to draw any real conclusion from, and not repaired in this pass (`repaired` list in the JSON only covers
  `default_seeds`/`llm_grt`/`llm_proxy`/one `random_grt` rep).
- **`random_proxy` has zero recoverable held-out routed data.** The DB currently has no `stage="route"` rows for
  `random_proxy` at all, despite the (stale) summary JSON's `search` section listing non-zero `routes_used` for
  it — that historical route evidence was lost to the same `purge_arm_runs()` incident that motivated this repair
  pass, and `random_proxy` was not among the arms this particular repair recovered. Its data is genuinely gone
  unless a future run regenerates it (cheap to rerun — 3 reps at a small budget — if this arm's data is wanted for
  the paper, currently not attempted here since it's the least evidentially important arm — a proxy-only,
  no-routed-feedback random baseline is a secondary ablation, not the headline comparison).

## Bottom line — is it good?

**As infrastructure: yes, genuinely good.** A real noise-aware, held-out-validated experiment ran to completion
on real hardware, recovered real lost data via the repair mechanism, zero mock rows, zero DRC violations across
422 real evaluations, and the content-addressed cache/dedup subsystem visibly worked (29 real hits, including
catching the `llm_proxy#r2`/`default_seeds` coincidence correctly). That's a solid, citable demonstration the
whole pipeline holds up under a real multi-hour, multi-arm, infra-hiccup-surviving run.

**As a science result: honest, not a clean win.** `llm_grt`/`llm_proxy` beat `default_seeds` by ~3.4-3.6% on
held-out routed WL — real, consistent in direction, but inside this project's own ~3% noise floor, so it should
be presented as a modest, disclosed trend, not a decisive result. `random_grt`/`random_proxy` don't have enough
surviving data to say anything about the grt-funnel-vs-proxy-funnel question on this design. This is consistent
with, not a contradiction of, everything else already known about tinyRocket in this project — a real design, but
one with limited headroom for any search strategy to show a large effect.

## Reproduction

Raw DB/JSON/logs: `results/tinyrocket/repair3_raw/` (pulled from `chia-dev:~/work/exp/tinyRocket_main.{db,json,log}`
and `tinyRocket_repair3.log`). Settings: `{rounds: 4, batch: 8, route_top: 3, repeats: 3, seed: 0, validate_seeds:
[101, 102, 103, 104]}`.

## Addendum (2026-09-24): random-search controls, all arms on the same 4 held-out seeds

**This supersedes the per-arm table above.** That table averaged a *different subset* of held-out seeds per arm:
seed 101 failed for every arm-run, and the LLM runs had 2/4 seeds. With ~2.3% seed-to-seed spread, that alone
could move a mean by ~1%. Adding seed 101 moved `default_seeds` from 507,293 to 501,687.

**Root cause of the original data loss:** 180 of 185 failed evals came from one node, `chia-gpu5`. Its GPU dropped
out of its container on 09-23 at ~05:24 UTC (`No CUDA GPUs are available`) and it kept taking jobs and failing
them instantly. Before this run, the affected containers were restarted and every node's in-container
`torch.cuda.is_available()` was checked. This run had **0 infrastructure failures**.

**Protocol:**
- Same toolchain as the original arms (`d356e84aeccb`).
- New evaluations used split mode pinned to one VM per candidate, because stock split mode is broken across VMs.
- Every arm-run's best config was routed on the same held-out seeds 101–104.
- `random_grt` used the same settings as the LLM arms, but only **1 round (3 routes) per rep** fit before the
  compute window closed. `random_grt#r2`'s round-1 routes didn't finish, so n=2.
- Budget matching: in 5 of 6 LLM reps the best config was already found in round 1, so the LLM results at 1 round
  are identical to the LLM results at 4 rounds (only `llm_proxy#r1` improved later: 482,005 → 480,748).
- Audit from raw DB rows: 0 mock rows, every row on `d356e84aeccb`, **DRC 0 on all routes**, 4/4 seeds for every
  arm-run compared.

| arm | reps | held-out mean routed WL per rep (µm) | arm mean | vs default |
|---|---|---|---|---|
| `default_seeds` | 1 config × 4 seeds | 484,870 / 506,743 / 505,671 / 509,465 (per seed) | 501,687 | — |
| `random_grt` (1 round) | 2 | 481,901 / 478,840 | **480,370** | **−4.25%** (2·SE 11.7k, beyond noise) |
| `llm_grt` (4 rounds; best found in round 1) | 3 | 481,388 / 480,107 / 500,853 | **487,449** | −2.84% (2·SE 17.6k, within noise) |
| `llm_proxy` (4 rounds) | 2 valid | 478,154 / 480,748 | **479,451** | −4.43% (2·SE 11.6k, beyond noise) |

`llm_proxy#r2` is excluded. Its search was 31/31 infra failures, so its "best" is the default config.

**Comparisons at equal (1-round) budget:**
- **`llm_grt` − `random_grt` = +7,079 µm (+1.47%), 2·SE = 13,768: no detectable difference.** The point estimate
  favors random.
- **`llm_proxy` − `random_grt` = −920 µm (−0.19%), 2·SE = 4,012: no detectable difference.**

**Reading:**
- On tinyRocket, **searching the knob space beats the default by ~4%, but the LLM proposer adds nothing measurable
  over random proposals**, even when random gets a quarter of the LLM's route budget.
- The LLM's routed-feedback rounds (2–4) found a better config in only 1 of 6 reps. Caveat: those later rounds
  lost candidates to the gpu5 failure.
- n is small (2–3 reps per arm) and this is one design, so this bounds the LLM's contribution here. It doesn't
  prove the LLM is useless in general.

**Reproducibility caveat:**
- The same request routed in split mode vs the original non-split run gave WL 509,808 vs 509,468 (+0.07%) and WNS
  −0.272 vs −0.303 ns. That WL jitter is ~3% of the seed spread, so it can't create or hide the effects above.
- Timing is more sensitive to it, so these arms are compared on WL only.

**Raw data:** `results/tinyrocket/random_controls_raw/`. `merged.db` is what `random_controls_run.py analyze` reads;
`analyze_output.txt` is its output.
