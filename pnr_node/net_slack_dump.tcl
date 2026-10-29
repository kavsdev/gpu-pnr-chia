# Per-net worst setup slack dump (sourced after a design is loaded, timed and has parasitics estimated).
# Env: NW_OUT (output file), NW_IGNORE_DEG (default 100). Format: "# wns <ns> tns <ns> viol <n>" then one
# "<net> <worst slack|INF> <degree> <sig|clk|hideg>" line per net with >= 2 pins. Used by net_weights.tcl (the flow) and
# by the V2 OpenTimer-vs-OpenSTA agreement check, so both time nets identically.
set ignore_deg [expr { [info exists ::env(NW_IGNORE_DEG)] && $::env(NW_IGNORE_DEG) ne "" ? $::env(NW_IGNORE_DEG) : 100 }]
set wns [sta::worst_slack -max]
set tns [sta::total_negative_slack -max]
set viol [sta::endpoint_violation_count max]
set fh [open $::env(NW_OUT) w]
puts $fh "# wns $wns tns $tns viol $viol"
# Clock nets: any net touching a register clock pin (is_clock) or a clock source port. Ports have no is_clock.
set clk_src {}
foreach clk [all_clocks] {
  foreach src [get_property $clk sources] { dict set clk_src [get_full_name $src] 1 }
}
set n 0
foreach net [get_nets *] {
  set pins [get_pins -quiet -of_objects $net]
  set ports [get_ports -quiet -of_objects $net]
  set deg [expr { [llength $pins] + [llength $ports] }]
  if { $deg < 2 } { continue }
  set worst INF
  set flag sig
  foreach p $ports {
    if { [dict exists $clk_src [get_full_name $p]] } { set flag clk }
  }
  foreach p $pins {
    if { [get_property $p is_clock] } { set flag clk }
  }
  foreach p [concat $pins $ports] {
    set s [get_property $p slack_max]
    if { $s ne "" && $s ne "INF" && ($worst eq "INF" || $s < $worst) } { set worst $s }
  }
  if { $flag eq "sig" && $deg > $ignore_deg } { set flag hideg }
  puts $fh "[get_full_name $net] $worst $deg $flag"
  incr n
}
close $fh
puts "NETWT wrote $::env(NW_OUT): $n nets wns=$wns tns=$tns viol=$viol"
