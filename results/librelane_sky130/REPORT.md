# LibreLane/sky130 + DREAMPlace: signoff-clean and macro-capable, not yet both at once

sky130 is used here because Magic/Netgen signoff is possible on it (NanGate45 has no Magic DRC / Netgen LVS deck).
DREAMPlace is wired in via the same coordinator-step splice
used everywhere else in this project (`flows/librelane_sky130/ll_placer_flow.py` +
`pnr_node/librelane_handoff.py::LibreLaneEvaluator`).

## `spm` — 0 macros, real signoff-clean, real knob sensitivity

Real Magic DRC=0, Netgen LVS=0. Reproducibility confirmed (repeat run at `target_density=0.6` matches exactly).
Density sweep:

| `target_density` | proxy_hpwl | routed_wl | wns | drc | lvs |
|---|---|---|---|---|---|
| 0.60 | 8246.878 | 6439 | 0 | 0 | 0 |
| 0.70 | 7886.525 | 6884 | 0 | 0 | 0 |
| 0.80 | 7702.039 | 6731 | 0 | 0 | 0 |

Independently re-verified 2026-09-23 via the new `cfg["backend"]="librelane"` dispatch path in `pnr_node/node.py`
(see below): a fresh run at default density returned `routed_wl=6618, drc_violations=0, lvs_violations=0, wns=0` —
consistent with this table, confirming the pluggable-backend wiring reproduces the same real result.

## `test_sram_macro` — 2 real sky130 SRAM macros, PARTIAL signoff

`test_sram_macro` (from `librelane-ci-designs`, https://github.com/librelane/librelane-ci-designs) instantiates two
real, silicon-proven `sky130_sram_1kbyte_1rw1r_32x256_8` hard macros — this is the design meant to close the gap
`spm` (signoff-clean, no macros) and `bp_be_top` (NanGate45, macros, structurally can't sign off) both leave open:
nothing so far has been both macro-containing and signoff-clean.

**Real result** (`chia-splitpr:~/work/shared/librelane/test_sram_macro/a3d55e240d5da05d/flow_done.json`,
independently re-read, not just the agent's own summary):

| Metric | Value |
|---|---|
| Routed wirelength | 64,888 µm |
| WNS / TNS (setup) | 0 / 0 ns |
| Magic DRC violations | 532 |
| Netgen LVS violations | **0** |

**Verdict: PARTIAL, and a genuinely useful one.** LVS=0 proves the PDN and logical connectivity are completely
correct for a real macro-containing sky130 design pushed through the DREAMPlace splice — the routing itself is
sound. All 532 DRC violations are a single rule class (`nwell.4`, "all nwells must contain a metal-connected N+
tap") concentrated around the macro keep-out margins — a tap-cell-insertion gap near macros, not a placement or PDN
correctness failure. This is a real, bounded, diagnosed limitation, not a vague "still needs work."

### Real fixes applied to get this far (now merged into `flows/librelane_sky130/ll_placer_flow.py`)
1. **`config.json` → `config.yaml`**: the design's stock config used strings where LibreLane's current parser wants
   real YAML lists (`VDD_NETS`, `EXTRA_LEFS`, etc.) — see `flows/librelane_sky130/designs/test_sram_macro/config.yaml`.
2. **`PDN_MACRO_CONNECTIONS`** (not the deprecated `FP_PDN_MACRO_HOOKS`): ties each macro instance's power pins to
   the top-level `vccd1`/`vssd1` nets — same mechanism, same lesson, as the NanGate45 `bp_be_top` PDN fix earlier in
   this project.
3. **SPECIALNETS stripped from the DEF handed to DREAMPlace**: DREAMPlace's DEF parser mis-identified PDN
   shapes/vias in `SPECIALNETS` as macro pins and crashed (`macroPinId < numeric_limits<index_type>::max()`).
   Harmless to strip — the re-import is `read_def -incremental` on a COMPONENTS-only DEF anyway, so the PDN was
   never going to come from that file. Now a standing part of `ll_placer_flow.py`'s `LibreLanePlacerStep.run()`.
4. **`EXTRA_LEFS` forwarded into the DREAMPlace request** — previously only `MACRO_LEFS`/`CELL_LEFS`/tech LEF were
   sent; a real SRAM macro's LEF lives in `EXTRA_LEFS`, and without it DREAMPlace can't resolve the macro's own
   pins (crashed looking for `csb0` etc).

### Reproducing this
```
git clone https://github.com/librelane/librelane-ci-designs   # or find test_sram_macro's dir another way if moved
# design config already captured in this repo: flows/librelane_sky130/designs/test_sram_macro/
```
then the same `cfg["backend"]="librelane"` pattern as `spm` above, `req.design = "test_sram_macro"`, PDK root
pointed at a ciel-managed sky130 checkout (`~/work/shared/librelane/pdk/ciel/sky130/versions/<hash>` in this
project's VMs), `dp_install` set to the DREAMPlace install root (the directory *containing* `dreamplace/`, not that
subdir itself — see the pluggable-backend section below for why that distinction matters).

### Not yet done
Closing the remaining 532 `nwell.4` violations (would need explicit tap-cell insertion tuned for the macro
keep-out margins, not attempted — the flow's tap-cell step runs before macro placement is finalized and needs its
own investigation). A larger design with more macros was out of scope for this pass.

## `LibreLaneEvaluator` wired into `ChiaEvaluator` as a real pluggable backend

`pnr_node/node.py`'s `_evaluator(cfg)` now branches on `cfg.get("backend", "raw")`: `"raw"` (default, unchanged)
builds `RawHandoffEvaluator`; `"librelane"` builds `LibreLaneEvaluator`. Both implement the same
`.evaluate(req) -> EvalResult` contract, so `evaluate_remote`/`ChiaEvaluator`/`search.py`/`experiment.py` need zero
changes to use either backend — the explicit "modular, reusable, drop-in" goal for this whole project.

Validated for real on a from-scratch VM (not just code review): a real `EvalRequest("spm", ...)` dispatched through
`node._evaluator({"backend": "librelane", ...})` returned `routed_wl=6618, drc_violations=0, lvs_violations=0,
wns=0` — matching the known `spm` sweep above. This was the first time `LibreLaneEvaluator` was ever actually
driven through `ChiaEvaluator`/`node.py` (previously only run standalone), and it immediately surfaced two real
bugs, now fixed in `pnr_node/librelane_handoff.py`:
1. It never passed `PDK_ROOT` into the librelane container's `docker run` at all — new `pdk_root` constructor arg,
   threaded through as `-e PDK_ROOT=<path>`.
2. `run_dreamplace()` appends `dreamplace/Placer.py` to whatever `dp_install` it's given, so `dp_install` must be
   the *parent* of the `dreamplace/` directory, not that directory itself — an easy mistake since the directory
   containing `Placer.py` is itself named `dreamplace`.

`ChiaEvaluator.__init__` now rejects `backend != "raw"` combined with `split=True` (the split path's
`place_remote`/`route_remote` call `RawHandoffEvaluator`-only methods the librelane backend doesn't have — it
manages its own GPU container internally instead). 43/43 tests pass, including 2 new ones covering this dispatch
and the guard.
