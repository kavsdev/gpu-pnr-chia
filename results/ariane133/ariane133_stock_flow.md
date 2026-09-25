# Ariane133 / NanGate45: stock ORFS flow, measured (chia-dev, 6 cores, 2026-09-21)

Stock `openroad/orfs` flow, digest sha256:573c1716..., `NUM_CORES=6`. Stages up to 3_5 ran first and the run was stopped there;
it was then resumed in the same work directory from CTS, so CTS onward is a continuation, not a fresh run.
Times are the per-stage "Elapsed time" from the ORFS logs; memory is the log's "Peak memory".

| stage | elapsed | peak memory |
|---|---|---|
| 1_2 yosys synthesis | 19:03 | 0.7 GB |
| 2_1 floorplan | 8:34 | 0.8 GB |
| 2_2-2_4 macro place, tap, PDN | ~0:22 | 0.3 GB |
| 3_1-3_2 global place (skip IO), IO place | ~0:26 | 0.6 GB |
| 3_3 global placement (RePlAce, timing-driven) | 18:17 | 2.0 GB |
| 3_4 resize | 1:41 | 1.0 GB |
| 3_5 detailed placement (legalize) | 2:34 | 1.5 GB |
| 4_1 CTS (incl. timing repair) | 20:23 | 1.7 GB |
| 5_1 global route | 28:18 | 3.1 GB |
| 5_2 detailed route | 49:11 | **7.4 GB** |
| 5_3, 6_1 fill | ~0:10 | 1.1 GB |
| 6_report (final metrics) | 17:59 | - |
| total (ORFS make summary) | 10053 s = 167.6 min | |

Final PPA (measured): instance area 720,906 um^2, setup WNS -0.2657 ns, TNS -429.2 ns, power 0.2519 W, detailed-route wirelength
6,140,510 um, DRC violations 0. The design has 3115 endpoints violating setup at the ORFS clock, which is why CTS timing repair is slow.

## What this means for a search that swaps in DREAMPlace for stage 3_3
- One-time preparation (synth + floorplan + macros + IO, stages 1..3_2): ~29 min per machine, cacheable per design.
- Per candidate: DREAMPlace (13 s on the L4 as reported by the agent; 36 s CPU-only with 4 threads, HPWL 3.034e7 vs 3.027e7)
  then stages 3_4 onward: 1.7 + 2.6 + 20.4 + 28.3 (= 53 min to the global-route screen) + 49.2 (detailed route) + ~18 (final report)
  = about 120 min for a full route, of which the placer is < 1%. The 18-minute stock global placement is what DREAMPlace replaces.
- Memory is 7.4 GB per detailed route; a 64-vCPU / 256 GB VM can hold ~28 concurrent runs by memory.
- So the GPU speeds up only the placer (2.7x on Ariane) and is irrelevant to end-to-end throughput; CPUs are what limit an Ariane search.
Not yet measured: a full Ariane route through the DREAMPlace handoff (pilot running on the CPU VMs), seed-to-seed noise on Ariane.
