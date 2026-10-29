# Per-net setup-slack dump for net-weight timing-driven DREAMPlace.
#
# Run under ORFS's environment so the STA setup is exactly the one ORFS's own stage 3_3 uses (liberty/corners,
# derate, setRC.tcl wire RC, the design SDC):
#     make DESIGN_CONFIG=... WORK_HOME=<scratch> run RUN_SCRIPT=<this file>
# Inputs (env):
#   NW_DEF      pre-placement DEF dumped by the 3_3 splice (the true post-buffer_ports netlist)
#   NW_PLACED   optional components-only DEF overlaid with read_def -incremental (a DREAMPlace placement to time)
#   NW_SDC      the design SDC ORFS loads at 3_3 (results/.../2_floorplan.sdc)
#   NW_OUT      output file
#   NW_REPAIR   1 (default) = virtual repair_design before reading slacks, 0 = raw pre-repair netlist
#   NW_IGNORE_DEG  nets with more pins than this are flagged "hideg" (DREAMPlace's ignore_net_degree)
# The per-net dump itself is net_slack_dump.tcl (shared with the V2 OpenTimer-vs-OpenSTA check).
# Output: "# wns <ns> tns <ns> viol <n>" header, then one "<net> <worst slack ns|INF> <degree> <sig|clk|hideg>" line
# per signal net. Weights are computed from this in Python (pnr_node/netwt.py) so the formula is testable.
# Clocks stay ideal, as at ORFS 3_3 (pre-CTS): no set_propagated_clock.
source $::env(SCRIPTS_DIR)/load.tcl
source $::env(SCRIPTS_DIR)/read_liberty.tcl
read_lef $::env(TECH_LEF)
read_lef $::env(SC_LEF)
if { [env_var_exists_and_non_empty ADDITIONAL_LEFS] } {
  foreach lef $::env(ADDITIONAL_LEFS) { read_lef $lef }
}
read_def $::env(NW_DEF)
if { [env_var_exists_and_non_empty NW_PLACED] } {
  read_def -incremental $::env(NW_PLACED)
}
read_sdc $::env(NW_SDC)
if { [file exists $::env(PLATFORM_DIR)/derate.tcl] } { source $::env(PLATFORM_DIR)/derate.tcl }
if { [env_var_exists_and_non_empty LAYER_PARASITICS_FILE] } {
  source $::env(LAYER_PARASITICS_FILE)
} else {
  source $::env(PLATFORM_DIR)/setRC.tcl
}
estimate_parasitics -placement
# Virtual repair (NW_REPAIR=1, default): before 3_4's repair_design the netlist has unbuffered high-fanout nets whose
# slews swamp every slack (tinyRocket pre-GP WNS -5.7 ns vs ~-0.1 ns routed). OpenROAD's own timing-driven RePlAce
# reads slacks after a journaled repair_design for the same reason (Resizer::findResizeSlacks). This session is
# throwaway, so repair for real and re-estimate. Buffer nets it creates are not in DREAMPlace's netlist and show up as
# "unmatched" in the hook's log; the original nets keep their names (driver side).
if { ![info exists ::env(NW_REPAIR)] || $::env(NW_REPAIR) != 0 } {
  if { [env_var_exists_and_non_empty DONT_USE_CELLS] } { set_dont_use $::env(DONT_USE_CELLS) }
  repair_design
  estimate_parasitics -placement
}

source [file join [file dirname [info script]] net_slack_dump.tcl]
