# pnr-node

A GPU-accelerated place-and-route evaluator node for [CHIA](https://arxiv.org/abs/2606.27350). It places with
[DREAMPlace](https://github.com/limbo018/DREAMPlace) on the GPU inside an otherwise stock
[OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) (ORFS) flow, makes that
placement timing-driven, caches every evaluation, and runs a noise-aware agentic search over flow knobs that is
judged on routed timing.

## Paper

**[A GPU-Accelerated DREAMPlace/OpenROAD Evaluator Node for CHIA: Timing-Driven GPU Placement, Cached
Multi-Backend Evaluation, and Noise-Aware Agentic Search](paper/main.pdf)** (A³: Agentic Approaches to Architecture Workshop at MICRO 2026,
hackathon short paper). The tag `a3-camera-ready` is the release that matches the paper, archived at
[doi:10.5281/zenodo.23763972](https://doi.org/10.5281/zenodo.23763972). Its results and the scripts that
recompute them are in [`results/a3/`](results/a3/README.md).

In short:

- **Timing-driven GPU placement.** DREAMPlace is interleaved with OpenROAD buffer repair and OpenSTA net weights.
  On Ariane133 (161k cells, 132 macros) it beats stock ORFS on TNS by 208 ± 169 ns and on WNS by
  0.096 ± 0.068 ns, with placement 1.8× faster (498 vs. 887 s) and comparable end-to-end runtime. On
  swerv_wrapper, jpeg and tinyRocket it routes 2–5% shorter, ties on TNS, and loses 0.059 ns WNS on swerv_wrapper.
- **Noise-aware search.** Candidates are scored after clock-tree synthesis (CTS), the cheapest stage whose timing
  ranks configurations like the routed result. A routed final round adopts a candidate only if it beats the
  default by more than seed noise. Five proposers (default, random, a tree-structured Parzen estimator (TPE), an LLM, and an
  LLM with timing diagnosis) were compared on four designs. The round keeps the default on jpeg and tinyRocket. On swerv_wrapper it adopts an
  LLM-proposed configuration that improves WNS on 11 of 12 test seeds; its score gain is within seed noise.
- **Reusable node.** One request/result contract over two flow backends (ORFS on NanGate45, LibreLane on sky130),
  placement and routing as separate GPU and CPU CHIA functions, and content-addressed caching that replays
  finished experiments without rerunning tools.

## How the flow works

```
ORFS (stock)    floorplan, PDN, IO placement                                   [unmodified]
                      │  design written as DEF
                      ▼
DREAMPlace (GPU)  global placement to overflow 0.6 ─┐
OpenROAD          repair_design + re-buffer,        │  repeated at overflow 0.2,
OpenSTA           per-net slack → net weights ──────┘  last run to 0.1
                      │  cells read back into OpenROAD
                      ▼
ORFS (stock)    resize, legalize, CTS, global and detailed route, reports     [unmodified]
```

Stock ORFS's global placer also pauses twice to repair the netlist and reweight critical nets; the GPU flow does
the same around DREAMPlace. The worst 10% of nets by slack get weights of up to 5. Before the first repair, nets
with more than 2000 loads are split by a 5×5 grid of buffers. Target density is 0.45. Implementation:
`pnr_node/raw_handoff.py` (`RawHandoffEvaluator`, `FINAL_FLOW_KWARGS`), `pnr_node/netwt.py`,
`pnr_node/rebuf_presplit.tcl`.

## What's in this repo

```
pnr_node/                  the evaluator and search framework (pure Python, no hard third-party deps)
├── contract.py              EvalRequest/EvalResult, the flow knobs, score_timing() (the search objective)
├── raw_handoff.py           the ORFS + DREAMPlace flow (handoff, repair stages, net weights, stage stops)
├── netwt.py                 OpenSTA slack → per-net weights; net_weights.tcl, net_slack_dump.tcl
├── rebuf_presplit.tcl       high-fanout net presplit before the first repair
├── dreamplace_runner.py     DREAMPlace config and invocation (its built-in timer stays off)
├── node.py                  CHIA @ChiaFunction wiring: GPU-only place / CPU-only route, ChiaEvaluator
├── arm_eval.py              host-side evaluator: one scripts/gap/run_arm.sh docker run per evaluation
├── search.py                search loop and proposers (random, TPE, LLM), all implementing propose(history, n)
├── diag.py, diag_proposer.py  the LLM proposer with timing diagnosis: per-run evidence and stated hypotheses
├── experiment.py            search driver, validation, and the routed final round
├── db.py, report.py         ResultsDB, CachedEvaluator, reports
├── llm.py                   provider-agnostic LLM layer (Gemini, Anthropic, OpenAI-compatible, CHIA, fake)
├── mock_eval.py             synthetic evaluators for running the search loop without EDA tools
├── study.py                 rank-correlation study (does a cheap metric order candidates like an expensive one?)
├── batch.py, evidence.py, manifest.py   batching, reproducibility bundles, capability manifest
└── librelane_handoff.py     the same DREAMPlace handoff through LibreLane (sky130A)

docker/                    Dockerfiles (base: ORFS + CHIA worker; hammer; dp: DREAMPlace + CUDA torch), the
                            DREAMPlace patches, and run_local.sh, which builds everything on one Linux + NVIDIA box
scripts/arms/              search and final-round drivers, the A3 job file, export and verify scripts
scripts/gap/               run_arm.sh / run_stock.sh (one GPU-flow / stock ORFS run in docker), queue.sh, job files
scripts/                   flow_time.py, stage_timing.py, make_fig_results.py
results/a3/                the A3 paper's results (see its README)
results/                   results of the earlier version of the paper (commit a17c44a)
flows/, packer/            LibreLane and AutoDMP flow scripts; GCE image template
paper/                     LaTeX sources and main.pdf
tests/                     157 tests, `pytest tests/`
```

## Quickstart

```bash
pip install -e ".[test]"
pytest tests/                                   # 157 tests, no EDA tools or GPU required

# Recompute the paper's tables and figure data from results/a3/
python3 scripts/arms/verify_table_a.py
python3 scripts/arms/verify_final_round.py
python3 scripts/arms/fidelity_corr.py
python3 scripts/arms/verify_w.py

# Run the search loop end to end on a synthetic evaluator (no tools, no LLM key)
python3 scripts/arms/run_arms.py --design tinyRocket --arms default,random,tpe,llm,llm_diag --reps 1 \
    --budget-evals 8 --batch 4 --stage cts --seeds 0 --promote-top 1 --promote-seeds 1 --default-seeds 0,1 \
    --validate-seeds 101 --validate-stage route --db d.sqlite --out d.json --llm-provider fake --dry-run

# Build the toolchain locally (Docker + NVIDIA container runtime; see LOCAL_QUICKSTART.md), then run one
# GPU-flow evaluation through the full route
docker/run_local.sh
export DP_INSTALL=$HOME/work/dreamplace/install_local      # run_local.sh's DREAMPlace install
scripts/gap/run_arm.sh gcd demo --seed 1 --stage route --rebuffer --rebuf-full --rebuf-overflow 0.6,0.2 \
    --timing-driven-netwt static --netwt-scheme gpl --rebuf-presplit --dp-extra '{"target_density":0.45}'
```

## Reproducing the paper

| What | Command |
|---|---|
| Tables 1–2, Fig. 3 data, runtime | see [`results/a3/README.md`](results/a3/README.md) |
| Fig. 3 | `python3 scripts/make_fig_results.py` |
| Figs. 1–2 | `make -C paper/tikz` |
| Searches and final rounds | `scripts/arms/jobs_a3.txt` |
| Other routed runs (test seeds, stock ORFS, runtime) | `scripts/gap/queue.sh scripts/gap/jobs_<name>.txt [N]` |
| The paper | `make -C paper` (needs a TeX install with Libertine/Biolinum/newtx) |

The toolchain is the ORFS image `openroad/orfs@sha256:573c1716efa0e286c4f641c26d343e20929d58d27fffbb22be2d0b93f09764f6`
and DREAMPlace commit `6627f33` with the patches in `docker/`; `docker/run_local.sh` builds both.

## Reusing the node

- **Proposer**: anything implementing `propose(history, n)` runs in the search loop (`pnr_node/search.py`).
- **LLM provider**: `pnr_node/llm.py`'s `make_provider()` selects one from a config dict.
- **Evaluator**: anything implementing `evaluate(EvalRequest)` gets caching (`db.py`'s `CachedEvaluator`), the
  search driver and the routed final round (`pnr_node/experiment.py`).
- **Single evaluation**: `ChiaEvaluator.evaluate(req)` (`pnr_node/node.py`) runs one routed evaluation directly.
- **Rank-correlation study**: `pnr_node/study.py` checks whether a cheap metric orders candidates like an
  expensive one on a new design or knob range.
- **Mock evaluators**: `pnr_node/mock_eval.py` runs the full driver with no EDA tools or GPU.

## Limitations

Only global placement runs on the GPU. Timing-driven placement still needs OpenROAD repair and OpenSTA on the
CPU, and these take nearly all of the placement time; CTS and routing take most of the end-to-end runtime. So the
GPU flow matches stock ORFS end to end and is slower on small designs. The evidence covers four NanGate45 designs,
one of them (Ariane133) large and macro-dense. The per-net slack step needs a flat netlist, so hierarchical designs
are flattened. With two or three search runs per proposer and 4–12 test seeds per design, the study cannot rank
the proposers. CTS was validated as the search stage on Ariane133 only. See the paper's §5.

## License

MIT, see [`LICENSE`](LICENSE).
