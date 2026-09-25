# Raw data behind the paper's figures and cross-design numbers

Every file under `results/figures_data/raw/` was copied from the cloud run directory that produced it; nothing was
re-typed from a report. The cloud VMs no longer exist, so these copies are the durable source. ORFS metric files
use the standard keys: `detailedroute__route__wirelength` / `detailedroute__route__drc_errors` (in `5_2_route.json`),
the global-route estimate (in `5_1_grt.json`), and `finish__timing__setup__ws` / `__tns` / `finish__power__total`
(in `6_report.json`).

## `proxy_vs_routed_scatter/`: Fig. 3 and the proxy/global-route correlation (§5)

Raw outputs of `pnr_node/study.py` on gcd, backing `results/gcd/proxy_route_study_gcd.md`.

| file | what it is |
|---|---|
| `place_gcd3.json`, `routed_gcd3.json` | the n=60 narrow-range sweep used for the Spearman numbers |
| `analysis_gcd3.json` | precomputed summary (Spearman proxy vs. routed WL = −0.293) |
| `place_gcd2.json` | the n=160 full-range feasibility sweep |

Fig. 3 is regenerated with `python3 scripts/make_fig_scatter.py`.

## `floorplan_render/`: Fig. 4

Placed (`3_5_place_dp`) and routed (`5_2_route`) images for gcd, tinyRocket and Ariane133 under stock RePlAce and
our DREAMPlace splice, plus AutoDMP on tinyRocket. Fig. 4 uses the six tinyRocket images. AutoDMP is not applicable to gcd (no macros) and was not run on
Ariane133. Rendered headless from each run's `.odb` with OpenROAD `save_image` under Xvfb, 2400 px wide. `save_image`
in batch mode needs an explicit `-area {die bbox}` and `-width`; otherwise the viewport is 0×0 and it writes
nothing. Placement images hide power, ground and special nets, so they show cells and macros only.
Fig. 4 is regenerated with `python3 scripts/make_fig_floorplans.py`.

## `autodmp_ours_stock_bars/`: AutoDMP vs. stock ORFS on tinyRocket (§5)

ORFS metric JSONs backing `results/tinyrocket/autodmp_vs_llm_tinyrocket.md`.

| file | run |
|---|---|
| `autodmp_{5_1_grt,5_2_route,6_report}.json` | AutoDMP's best trial: routed WL 540,456 µm, 159 DRC. WNS/TNS/power are null because its final report stage failed with `PSM-0069 Check connectivity failed on VDD`. |
| `stock_{5_1_grt,6_report}.json` | stock ORFS (RePlAce): WNS −0.122 ns, TNS −20.18 ns, WL 510,731 µm, 0 DRC |
| `stock_netlist_ctrl_{5_2_route,6_report}.json` | stock ORFS on the same synthesized netlist as the search arms (netlist control) |

Our search arms' side of this comparison comes from `results/tinyrocket/repair3_raw/tinyRocket_main.db`.

## `design_track_matrix/`: gcd per-arm metrics and the Ariane133 comparison (§5)

| file | what it is |
|---|---|
| `gcd_main.db` | gcd search DB, per-arm WNS/TNS/DRC/WL (arms: default_seeds, llm_grt, llm_proxy, random_grt, random_proxy) |
| `ariane133_stock_{5_2_route,6_report}.json` | stock ORFS on Ariane133: WL 6,140,510 µm, 0 DRC, WNS −0.2657 ns, TNS −429.2 ns |
| `ariane133_baselineA_{5_2_route,6_report}.json` | our DREAMPlace handoff on Ariane133: WL 6,949,677 µm, 0 DRC, WNS −1.102 ns, TNS −3725.4 ns |

## `runtime_breakdown/`: second Ariane133 handoff run, gcd handoff, sky130 `spm`

| file | what it is |
|---|---|
| `ariane133_{5_2_route,6_report}.json` | a second DREAMPlace handoff run on Ariane133 (routed WL 6,962,421 µm, WNS −1.148 ns). This run's per-stage timings are the ones compared against stock in `results/ariane133/ariane133_vs_stock.md`. |
| `gcd_{5_2_route,6_report}.json` | the gcd DREAMPlace handoff run rendered in `floorplan_render/` |
| `spm_flow_done.json` | LibreLane/sky130 `spm` flow summary (WL, Magic DRC, Netgen LVS) |
| `tinyRocket_evalresult_{proxy,grt,route}.json` | one representative `EvalResult` per funnel tier from the tinyRocket DB, including `dp_runtime_s` |

ORFS `6_report.json` files carry PPA only, with no runtimes. Per-stage elapsed times come from the ORFS logs cited in
`results/ariane133/`, and per-evaluation wall times come from `EvalResult` provenance (`dp_runtime_s`, `runtime_s`).
