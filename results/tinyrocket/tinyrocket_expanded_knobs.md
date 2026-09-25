# tinyRocket / NanGate45: expanded DREAMPlace knobs and context, and placement depth (`dp_depth`)

Four parts: A (four extra knobs), B/C (extra LLM context and wiring), D (`llm_grt` vs `llm_grt_expanded`, the
ablation cited in paper §4) and E (`dp_depth`). Every number below was re-read from the raw result JSON / SQLite DB,
and none of the Part D rows carries `provenance.mock`. DREAMPlace ran on an L4 GPU inside the `chia-pnr:dp`
container in native mode. An earlier attempt at this comparison had run against `MockEvaluator` by mistake; it was
discarded, and this report replaces it.

**Result:** the expanded arm is 1.34% shorter in routed WL (Part D), which is below the ~3% seed-noise floor. We
treat it as a null result.

---

## Part A: expanded knob set (convergence-checked on GPU)

The LLM's action space was 5 knobs (`target_density`, `density_weight`, `gamma`, `stop_overflow`, `learning_rate`).
Part A adds 4 more. Each default reproduces the value `dreamplace_runner.build_config()` used to hardcode, so an arm
that does not vary a knob runs the original configuration (`CORE_KNOBS` vs `KNOBS` in `contract.py`). Before
admitting them, points from each range were run through DREAMPlace (proxy stage, seed 0, tinyRocket) and checked for
convergence (`overflow ≤ stop_overflow`, cells inside the die):

| knob (range, default) | sampled points | result |
|---|---|---|
| `iteration` [700, 2000] def 1000 | 700, 900, 1500, 2000 | all converge (dp_iter ≈ 498–502, HPWL ~1.785M, ~flat) |
| `Llambda_density_weight_iteration` [1, 4] def 1 | 2, 3, 4 | all converge (HPWL 1.772–1.784M) |
| `Lsub_iteration` [1, 2] def 1 | 2 | converges (HPWL 1.692M) but ~2× dp iterations (898 vs ~498) |
| `num_bins` [0, 512] def 0 (=auto) | 64, 128, 256, 512 | all converge (HPWL 1.707 / 1.787 / 1.747 / 1.916M) |

All 15 sampled points converge, so the ranges are safe to expose. On tinyRocket the effects are small:
`iteration` is inert (the design converges at ~498 iterations, below the knob's floor of 700), `num_bins=512`
raises proxy HPWL, and `Lsub_iteration=2` lowers proxy HPWL but roughly doubles DREAMPlace's iteration count, which
is why it is capped at 2. Proxy HPWL is anti-correlated with routed WL on gcd (Spearman −0.29), so lower HPWL here
does not mean better. Part A only establishes that the wider action space converges. `routability_opt` was not
added, because it crashes DREAMPlace (no unit capacities).

## Part B/C: expanded context and wiring

`search.py::LLMProposer` takes a `knobs=` subset, so the baseline `llm_grt` arm tunes the core 5 while
`llm_grt_expanded` tunes all 9. With `expanded_context=True` the LLM also sees `tns`, `power`, `area` and
DREAMPlace's `dp_iterations`/`dp_overflow` for routed candidates; the baseline sees only WL, WNS and DRC.
`default_config()`/`clamp_config()` take an optional knob set. Unit tests cover that the expanded defaults reproduce
the old hardcoded DREAMPlace config and that `dp_depth` is part of the evaluator identity.

## Part D: `llm_grt` vs `llm_grt_expanded`

A held-out-validated A/B through `pnr_node.experiment.run_experiment` and a `ChiaEvaluator` (native mode, GPU,
Gemini proposer). The budget was reduced so the run fit the compute window: 2 repeats × 2 rounds × batch 3 ×
route_top 1, validation seeds {101, 102}. The gcd default is 4 rounds, 3 repeats and 4 seeds. Both arms share the
budget and differ only in action space (5 vs 9 knobs) and routed context (WL/WNS/DRC vs +TNS/power/area/dp_*).

**Search-phase best per arm-run** (selection seed 0; biased by picking the best, so not the deciding number):

| arm-run | best routed WL (µm) | WNS (ns) | TNS (ns) | DRC |
|---|---|---|---|---|
| `llm_grt#r0` | 476,961 | -0.211 | -140.7 | 0 |
| `llm_grt#r1` | 479,263 | -0.237 | -134.9 | 0 |
| `llm_grt_expanded#r0` | 477,051 | -0.233 | -156.7 | 0 |
| `llm_grt_expanded#r1` | 476,220 | -0.192 | -100.1 | 0 |

**Held-out validation** (2 arm-runs × seeds 101 and 102, n=4 routed points per arm, all DRC-clean):

| arm | validated WL mean (µm) | WL sd | validated WNS mean (ns) | WNS sd | DRC |
|---|---|---|---|---|---|
| `llm_grt` (baseline) | 484,922 | 4,982 (1.0%) | -0.2144 | 0.0124 | 0 |
| `llm_grt_expanded` | 478,448 | 2,176 (0.5%) | -0.2014 | 0.0092 | 0 |
| **Δ (expanded − baseline)** | **−6,474 (−1.34%)** | | **+0.0130 (better)** | | — |

Per-run validation points (WL / WNS): `llm_grt#r0` 482467/-0.207, 478021/-0.207; `llm_grt#r1` 490562/-0.236,
488637/-0.207; `llm_grt_expanded#r0` 479127/-0.207, 481663/-0.191; `llm_grt_expanded#r1` 477027/-0.214, 475974/-0.195.

**Reading:** the expanded arm is slightly better on both routed WL (−1.34%) and WNS (+0.013 ns). But (1) 1.34% is
below the ~3% seed-noise floor (`results/gcd/noise_floor_gcd.md`), even though it is ~2.4 standard errors by this
run's own spread; (2) there are only 2 arm-runs × 2 seeds per arm; and (3) tinyRocket has little headroom, and
every other arm comparison on it also landed within noise. We therefore do not claim an improvement.

The score column from `experiment.summarize()` (baseline 1,525,015 vs expanded 1,441,778, ≈ −5.5%) overstates the
gap, because `score = WL × (1 + 10·|WNS| + …)` amplifies the small WNS difference. Use the raw WL and WNS above.

Future work that could settle it: a design with more headroom, more repeats and held-out seeds, or separate arms
for the extra knobs and the extra context.

## Part E: DREAMPlace placement depth (`dp_depth`)

`dp_depth` (`global` default / `global+legal` / `global+legal+detail`) lets DREAMPlace legalize and detail-place
before the ORFS handoff. It is part of the toolchain fingerprint, so the modes get distinct `tool_id`s and never
share a cache entry. Routed results, default knobs, seed 0, tinyRocket:

| `dp_depth` | tool_id | Routed WL (µm) | WNS (ns) | TNS (ns) | DRC | OpenDP avg displ. (µm) | dp_iter |
|---|---|---|---|---|---|---|---|
| `global` (default) | f83da89dd9da | 511,985 | -0.283 | -193.9 | 0 | **1.4** | 497 |
| `global+legal` | 6dcc3385fe50 | 513,856 | -0.319 | -183.8 | 0 | **0.3** | 503 |
| `global+legal+detail`† | 5a0c362bb1c1 | 513,918 | -0.287 | -181.4 | 0 | **0.3** | 498 |

† **Correction (2026-09-25):** when this ran, the runner passed `detail_place_flag`, but DREAMPlace's key is
`detailed_place_flag`. DREAMPlace ignored the unknown key, so ABCDPlace did not run and this row is in effect a second
`global+legal` run. The key is fixed in `dreamplace_runner.py`. The corrected comparison, with ABCDPlace confirmed
running, is the laptop pair in `results/tinyrocket/local_pair/compare.md` (Run A vs A′): OpenDP displacement
1.4 → 0.3 µm, routed WL +0.4%, WNS 8 ps apart, one run each.

**Reading:** pre-legalizing in DREAMPlace cuts OpenDP's average cell displacement from 1.4 to 0.3 µm, because cells
arrive already on rows. Routed WL stays within ~0.4% and WNS/TNS within noise, all DRC-clean. `global` stays the
default.

---

## Provenance

- Part D DB (`partd.db`, 64 eval rows, 0 with `provenance.mock`) and the Part A/E logs were on the cloud VMs and were
  not preserved; the tables above were read from them before the VMs were shut down. Toolchain fingerprint on every
  Part D row: `{orfs_image: native-unpinned, dreamplace_commit: 6627f33..., torch: 2.9.1+cu129,
  dreamplace_device: gpu, timing_driven: False, dp_depth: global}`.

## Reproducing

```
# Part E (dp_depth), one routed evaluation:
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v <repo>:/work/code \
  chia-pnr:dp /opt/dp-venv/bin/python -m pnr_node.raw_handoff --design tinyRocket --stage route --native \
  --work <work_dir> --dp-install /opt/dreamplace --python /opt/dp-venv/bin/python --dp-depth global+legal

# Part D (the A/B), real evaluator + Gemini (GOOGLE_CLOUD_PROJECT set, google-genai installed in the venv).
# The run used validate_seeds=(101, 102) via run_experiment(); the CLI's default is 101-104.
python -m pnr_node.experiment --design tinyRocket --db partd.db --out partd.json \
  --arms llm_grt,llm_grt_expanded --repeats 2 --rounds 2 --batch 3 --route-top 1 --workers 3
```
