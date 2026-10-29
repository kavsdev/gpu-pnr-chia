# Results behind the A3 paper

Every number in the A3 camera-ready paper (`paper/main.pdf`) is recomputed from the files in this directory by the
scripts below. No EDA tools or GPU are needed to check them. Scores are `−TNS + 100·(−WNS)` (ns), lower is better.
Every routed run in `routed_runs.json` is a NanGate45 full route with 0 DRC violations.

| paper item | command | data |
|---|---|---|
| Table 1 (GPU flow vs. stock ORFS) | `python3 scripts/arms/verify_table_a.py` | `routed_runs.json` |
| Table 2 (routed final round) | `python3 scripts/arms/verify_final_round.py` | `search/`, `routed_runs.json` |
| Fig. 3a (cost vs. fidelity) | `python3 scripts/arms/fidelity_corr.py` | `fidelity_ariane133.json`, `runtime_ariane133.json` |
| Fig. 3b (seed-104 layouts) | images in `fig3b/` | ORFS final report images |
| Fig. 3c, Ariane133 winner per seed | `python3 scripts/arms/verify_w.py` | `routed_runs.json` |
| §4 Runtime | `runtime_ariane133.json` (written by `scripts/flow_time.py --json`) | per-stage wall times |
| Fig. 3 itself | `python3 scripts/make_fig_results.py` | `fig3b/` |

## Files

- `routed_runs.json`: one record per routed run on a test seed (101+): design, flow (`gpu` or `stock`),
  configuration (`default`, `winner` for the Ariane133 winner, or `stock`), seed, and the evaluator's result
  (TNS, WNS, routed wirelength, DRC, provenance). Two Ariane133 runs failed at CTS (OpenDP DPL-0033) and are kept
  as failed records: the winner on seed 108 and the default on seed 110.
- `search/<design>.sqlite`: the search database. Table `evals` holds every evaluation (request, result, score,
  proposer and round); table `llm_calls` holds every LLM prompt and response.
- `search/<design>_search.json`: the search summary (settings, per-proposer results, cache hits, LLM cost).
  swerv_wrapper has none; its search is in its database. tinyRocket's is the re-validated summary.
- `search/<design>_final_round.json`: the routed final round (finalists, selection-seed scores, test-seed scores).
- `search/<design>_llm_diag_hypotheses.jsonl`: each proposal of the LLM proposer with timing diagnosis and the
  hypothesis it stated.
- `fidelity_ariane133.json`: WNS and TNS after every stage for each routed Ariane133 GPU-flow run.
- `runtime_ariane133.json`: per-stage wall and CPU seconds of the two GPU-flow and two stock runs.
- `fig3b/`: ORFS's `final_placement` and `final_congestion` report images of the seed-104 routes.

Absolute paths inside the records are written as `$WORK`.

## Notes

**Table 1.** The two flows use independent random seeds (DREAMPlace's and RePlAce's), so a shared seed number links
nothing. Differences are unpaired, ± 2 standard errors of independent groups, and win counts are over all run pairs.

**Table 2.** θ = 2σ√(2/n), with σ the default's CTS-score standard deviation over the search seeds and n = 2
selection seeds. `verify_final_round.py` computes it from each search database. The swerv_wrapper, jpeg and
tinyRocket rounds ran with θ = 0 and record it so; the chosen column applies θ to their recorded leads. Each of
those rounds routed its best finalist on the test seeds, which gives the two † rows their test Δscore.

**Fig. 3a.** The paper uses the 13 competitive Ariane133 runs (routed TNS above −700 ns) that existed when CTS was
chosen as the search stage; they are the rows with `fig3a: true`. `fidelity_corr.py` also reports all 57
competitive runs available now, which include the search and test-seed runs. On that set the TNS rank correlation
is 0.65 after global placement, 0.70 after detailed placement, 0.93 after CTS and 0.94 after global route. The
minutes to reach each stage are the means of the two GPU-flow runs in `runtime_ariane133.json`.

**Runtime.** The four runs were measured one at a time on an AMD Ryzen 7 5800H with an RTX 3060 Laptop GPU.

## Reproducing the runs

The runs need the Docker images and DREAMPlace build from `docker/run_local.sh` (ORFS image
`openroad/orfs@sha256:573c1716efa0e286c4f641c26d343e20929d58d27fffbb22be2d0b93f09764f6`, DREAMPlace `6627f33`
with the patches in `docker/`). The DREAMPlace install the runs used also carries three changes that act only when
DREAMPlace's built-in timer is on, which this flow never turns on; provenance records them in the commit string
(`+tdgate100+mnw1024+hsfix3`).

- Searches and final rounds: `scripts/arms/jobs_a3.txt`.
- Routed runs outside them (test seeds 103+, stock ORFS, runtime): `scripts/gap/jobs_*.txt`, run with
  `scripts/gap/queue.sh <jobfile> [N]`.
- `python3 scripts/arms/export_results.py --work <dir>` collects a fresh set of runs into this layout.

`scripts/gap/run_arm.sh` is kept byte-identical to the one the runs used: its hash and the `DP_INSTALL` path are the
search evaluator's identity (`pnr_node/arm_eval.py`), so the databases here replay as cache hits only under that
identity. Set `DP_INSTALL` to the `docker/run_local.sh` install for new runs.

The LLM proposers used `gemini-3.1-pro-preview`, so their proposals will differ on a rerun; the database holds the
prompts and responses of these runs.
