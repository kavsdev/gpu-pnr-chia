# pnr-node

A GPU-accelerated place-and-route evaluator node for [CHIA](https://arxiv.org/abs/2606.27350) that splices
[DREAMPlace](https://github.com/limbo018/DREAMPlace)'s GPU global placement into a stock
[OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) (ORFS) flow, and an
LLM-agent search loop that drives it — turning placement-knob search into an optimizer for **real, routed PPA**
(post-detailed-route timing and DRC), not just a placement-stage proxy metric.

## Paper

**[A GPU-Accelerated DREAMPlace/OpenROAD Evaluator Node for CHIA](paper/main.pdf)** — the full writeup, with the
funnel design, protocol and every result below traced to a source. In short: DREAMPlace's own wirelength proxy is
*anti-correlated* with routed wirelength on gcd (Spearman ρ = −0.29), while a cheap global-route estimate is not
(ρ = +0.99), which motivates the proxy/grt/route funnel below. On tinyRocket, knob search shortens routed
wirelength by up to 4.4% with every route DRC-clean, while AutoDMP's best trial has 159 DRC violations — but an LLM
proposer turns out to be indistinguishable from random search once noise is controlled, and stock ORFS keeps
better setup timing and a faster end-to-end flow than our handoff on both tinyRocket and Ariane133.

## Why this exists

Most learned/automated placement work (including the tools we benchmark against below) optimizes a proxy —
wirelength, congestion, density estimated *at placement time* — and stops there. The central bet of this project is
that a proxy which looks good in its own optimization loop can fall apart once it goes through a real router: DRC
violations, timing regressions, and power-connectivity failures that the proxy never saw. So the evaluator here
always closes the loop through a **real OpenROAD detailed route**, not just a placement-stage estimate, and the
search loop is explicitly noise-aware (seed-lottery baselines, held-out validation) so a good number isn't just a
lucky seed.

## How the splice works

ORFS runs completely unmodified through floorplan/PDN/IO placement. The one step that's replaced is global
placement (`global_place.tcl`): instead of running OpenROAD's built-in placer, it either dumps the current
placement to a DEF (`pre_gp.def`) or, once DREAMPlace has placed it on GPU, re-imports the result
(`read_def -incremental gp.def`). Every stage after that — resize, legalization, CTS, global route, detailed route,
signoff reporting — is stock ORFS, untouched. So the numbers you get are real OpenROAD output, not a hand-rolled
approximation of one.

```
ORFS (stock)  →  floorplan, PDN, IO placement            [unmodified]
                        │
                        ▼
              global_place.tcl  →  dump DEF  →  DREAMPlace (GPU)  →  import DEF   [the splice]
                        │
                        ▼
ORFS (stock)  →  resize, legalize, CTS, route, signoff    [unmodified]
```

Implementation: `pnr_node/raw_handoff.py` (`RawHandoffEvaluator`). The evaluator supports a **three-tier funnel** so
a search loop doesn't have to pay for a full route on every candidate:

| Tier | What it measures | Cost |
|---|---|---|
| `proxy` | DREAMPlace's own placement cost | cheapest, but **anti-correlated** with routed quality — do not trust it alone |
| `grt` | Global-route wirelength/timing estimate | cheap, **strongly correlated** with final routed quality (see `results/gcd/proxy_route_study_gcd.md`) |
| `route` | Full detailed route + signoff report | expensive, ground truth |

## What's in this repo

```
pnr_node/                  the evaluator + search framework (pure Python, no hard third-party deps)
├── contract.py              EvalRequest/EvalResult, score() — the objective the search optimizes
├── raw_handoff.py           the DEF-splice mechanism (core of the project)
├── dreamplace_runner.py     DREAMPlace config/invocation. Timing-driven options are wired but NOT functional
│                             (audited: the timer never receives a netlist; see paper §6)
├── node.py                  CHIA @ChiaFunction wiring — GPU-only place / CPU-only route split, ChiaEvaluator
├── batch.py                 machine-local place-once/route-parallel batching
├── study.py                 proxy-vs-route rank-correlation study (place / route / analyze)
├── search.py                the LLM-agent search loop; RandomProposer and LLMProposer both implement
│                             propose(history, n), so either (or a new one) runs in the same loop
├── experiment.py            noise-aware multi-arm experiment runner (seed-lottery baseline, repair(), reports)
├── report.py / db.py        result storage + arm-comparison writeups; db.py's CachedEvaluator wraps any evaluator
├── llm.py                   provider-agnostic LLM layer (Vertex/Gemini API/Anthropic/OpenAI-compatible/CHIA/fake,
│                             lazy-imported) — make_provider() picks one from a config dict
├── manifest.py               capability manifest: stages, metrics, knob ranges, measured noise, known gaps
├── mock_eval.py              fast synthetic evaluator for testing the search loop without real EDA tools —
│                             our own test suite runs the full experiment driver against it
├── evidence.py                writes reproducibility bundles alongside every real result
└── librelane_handoff.py      DREAMPlace-through-LibreLane splice (sky130A), same EvalRequest/EvalResult interface
                               as raw_handoff.py — a second, ChiaEvaluator-compatible backend, not just a script

docker/                    the actual Dockerfiles the evaluators run inside (pulled from the real build, not
                            reconstructed) — base (ORFS + CHIA worker) → hammer (own venv, pydantic conflict with
                            CHIA) → dp (own venv, DREAMPlace + CUDA-pinned torch); build_dreamplace.sh is the real
                            cmake build DREAMPlace's own install directory comes from; run_local.sh builds and runs
                            the same images with no cloud/GCP dependency
packer/                    GCE image template to provision a fresh VM from these Dockerfiles + a public NVIDIA/CUDA
                            base image, so the whole environment is reproducible from code, not a hand-tuned VM

flows/                     flow-side scripts used by the backends and baselines
├── librelane_sky130/        the LibreLane-side flow script `librelane_handoff.py` (above) talks to — sky130A/`spm`,
│                             DREAMPlace actually spliced through LibreLane's Step API
└── autodmp/                  scripts used to get NVlabs' AutoDMP running against our designs for comparison
                               (DEF/LEF→Bookshelf conversion, config, macro-placement fixes) — see results/tinyrocket/

results/                   every real, evidence-linked finding, one folder per design/comparison, including the raw
                            data cited by the paper (search-arm databases, cache-replay logs, the laptop runtime pair)
tests/                     75 tests, `pytest tests/`
```

## Quickstart

```bash
pip install -e ".[test]"
pytest tests/                                    # 75 tests, no EDA tools or GPU required

# Run one real evaluation through the splice (needs docker + the ORFS image + a DREAMPlace install):
python -m pnr_node.raw_handoff --design gcd --stage route \
    --work ~/work/handoff --dp-install ~/work/dreamplace/install2

# Run the full noise-aware, LLM-driven arm comparison (needs a CHIA cluster; --mock runs without one):
python -m pnr_node.experiment --design gcd --db ./gcd.db --out ./gcd_results.json --mock

# Or fully local, no cloud/GCP: see LOCAL_QUICKSTART.md and docker/run_local.sh
```

`pnr_node` itself has **zero hard third-party dependencies** — the CHIA framework and any LLM provider SDK are
imported lazily and degrade gracefully when absent (see `node.py`'s `ChiaFunction` fallback and `llm.py`'s provider
docstring), so the core package and its tests run anywhere with a plain Python 3.10+.

## Results

| Design | What was tested | Result | Detail |
|---|---|---|---|
| gcd | proxy/grt/route correlation study, 60 configs | Proxy (DREAMPlace HPWL) is **anti-correlated** with routed WL (ρ = −0.29, p = 0.023); the global-route estimate is strongly correlated (ρ = +0.99) — the funnel's justification | [`results/gcd/proxy_route_study_gcd.md`](results/gcd/proxy_route_study_gcd.md) |
| gcd | LLM-driven vs. random vs. default-seed search, 5 arms | Negative/neutral — gcd is too small a design to clear the seed-lottery 2σ bar, explained not hidden | [`results/gcd/gcd_arms.md`](results/gcd/gcd_arms.md) |
| tinyRocket | Held-out search (LLM grt-feedback, LLM proxy-only, random grt-feedback) vs. default, 4 held-out seeds | Random search and the proxy-only LLM beat the default by 4.3–4.4% (clearing 2σ); the LLM with grt feedback by 2.8% (not clearing it). 0 DRC on every route. But the **LLM proposer is indistinguishable from random search** here (llm_grt − random_grt = +1.47%, 2·SE = 2.9%: no detectable difference), and stock ORFS keeps better setup timing (held-out WNS −0.20 to −0.27 ns vs. stock −0.122 ns) | [`results/tinyrocket/tinyrocket_arms.md`](results/tinyrocket/tinyrocket_arms.md) addendum, [`results/tinyrocket/random_controls_raw/analyze_output.txt`](results/tinyrocket/random_controls_raw/analyze_output.txt) |
| tinyRocket | Our search vs. NVlabs AutoDMP, real detailed route | Every one of our arms is 0 DRC; AutoDMP's best trial (run through a Bookshelf conversion, outside its tested setup) is 159 DRC, never produces usable timing (a real VDD-connectivity failure), and its detailed route took 5 h 06 min against under an hour for each of ours | [`results/tinyrocket/autodmp_vs_llm_tinyrocket.md`](results/tinyrocket/autodmp_vs_llm_tinyrocket.md) |
| tinyRocket, laptop (RTX 3060) | Our handoff vs. stock ORFS, full flow, one run each | With DREAMPlace legalization + detailed placement on: placement stage 21.8 s vs. 32.2 s (ours faster), but downstream is 1.39× slower and end-to-end is 1.29× slower (791 vs. 611 s; 1.14×, 697 s, with the default global-only handoff); routed WL within noise (+0.3%), WNS worse (−0.30 vs. −0.12 ns), 0 DRC both. The laptop's stock run reproduced the cloud stock-ORFS result bit-for-bit (WL, DRC, WNS) | [`results/tinyrocket/local_pair/compare.md`](results/tinyrocket/local_pair/compare.md) |
| Ariane133 (132 macros, cloud) | Same handoff vs. stock ORFS, full flow | Global placement drops 768.6 s → 50.6 s, but the flow is ~1.9× slower overall (resize 7.6×, detailed route 4.8×) and loses on quality: WNS −1.10 vs. −0.27 ns, routed WL 6.96M vs. 6.14M µm | [`results/ariane133/ariane133_vs_stock.md`](results/ariane133/ariane133_vs_stock.md) |
| gcd | Hammer-wrapped OpenROAD (hand-tuned toward ORFS) vs. direct ORFS integration | Hammer is 26.3% longer in WL (5593 vs. 4429 µm) and about 2× worse WNS (−0.32 vs. −0.162 ns), with equal power. On aes it fails legalization (DPL-0038), so we kept direct ORFS | [`results/hammer_orfs_parity.md`](results/hammer_orfs_parity.md) |
| sky130 `spm` / `test_sram_macro` | Second flow-engine backend: LibreLane with the same DREAMPlace splice | `spm`: Magic DRC = 0, Netgen LVS = 0, WNS = 0. `test_sram_macro` (2 SRAM macros): LVS = 0, 532 DRC violations, all one rule class (`nwell.4`) | [`results/librelane_sky130/REPORT.md`](results/librelane_sky130/REPORT.md) |

**Reading guide**: every result file states its sources (which raw report JSON/CSV/DB backs each number) and its
caveats up front — noise/statistical-confidence limits, tier mismatches, anything that was pushed outside a
compared tool's tested configuration. None of the numbers here are cited without the file that they came from.
Some older result files predate a 2026-09-25 audit that found DREAMPlace's timing-driven mode never functioned as
intended; those files now carry a correction note pointing to paper §6 rather than being rewritten.

## Reproducing the paper

| What | Command |
|---|---|
| Figs 1–2 (overview, protocol) | `make -C paper/tikz` |
| Fig 3 (proxy vs. routed scatter) | `python3 scripts/make_fig_scatter.py` |
| Fig 4 (tinyRocket floorplans) | `python3 scripts/make_fig_floorplans.py` |
| Table 1 (tinyRocket search arms) | `python3 random_controls_run.py analyze --db results/tinyrocket/random_controls_raw/merged.db` |
| §3 cache-replay claim | `python3 scripts/replay_cache_check.py <db>` (e.g. `results/tinyrocket/repair3_raw/tinyRocket_main.db`) |
| §5 laptop runtime pair | `python3 results/tinyrocket/local_pair/collect.py` |
| The paper itself | `make -C paper` (needs a TeX install with Libertine/Biolinum/newtx; outputs `paper/main.pdf`) |

## Reusing the node

The same interfaces the paper's §3 "Reuse beyond this flow" describes:

- **Proposer** — anything implementing `propose(history, n)` runs in the search loop alongside `RandomProposer` and
  `LLMProposer` (`pnr_node/search.py`).
- **LLM provider** — `pnr_node/llm.py`'s `make_provider()` selects Gemini (Vertex or the API-key path), an
  OpenAI-compatible endpoint, Anthropic, or CHIA's own LLM call from a config dict.
- **Evaluator** — anything implementing `evaluate(EvalRequest)` gets caching (`db.py`'s `CachedEvaluator`), the
  three-tier funnel and the noise protocol for free via `experiment.run_experiment`.
- **No search loop needed** — `ChiaEvaluator.evaluate(req)` (`pnr_node/node.py`) runs a single routed evaluation
  directly.
- **Rank-correlation study** — `pnr_node/study.py` is a standalone command (place / route / analyze) for checking
  whether a cheap metric orders candidates like an expensive one on a new design or knob range, before a funnel is
  built on it.
- **Mock evaluator** — `pnr_node/mock_eval.py`'s `MockEvaluator` runs the full experiment driver with no EDA tools
  or GPU; it's what the test suite above uses.

## Limitations

DREAMPlace's timing-driven mode does not work in this flow: the timer never receives a netlist from the ORFS
handoff, and net reweighting is gated to fire after the runs here have already converged. Every result in this repo
therefore uses wirelength-driven placement, while stock ORFS places timing-driven — the likely reason our handoff
loses on setup timing. See paper §6 for what fixing it would take.

## License

MIT — see [`LICENSE`](LICENSE).
