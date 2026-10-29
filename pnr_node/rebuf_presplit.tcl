# Pre-split very-high-fanout signal nets (ariane133: the async reset net from input buffer input3251, 20636 loads)
# before repair_design. RSZ's fanout repair (makeRegionRepeaters) costs ~O(remaining parent-net size) per inserted
# repeater, so a 20k-load net takes ~330 s idle (80% of repair_design; stock gpl pays the same inside its TD repair).
# One buffer per spatial grid bin (~group loads) makes each parent small; repair_design then finishes the tree.
# Call after the stage placement is read (read_def -incremental) and before estimate_parasitics/repair_design.
proc presplit_heavy_nets { {thr 2000} {group 1000} {cell BUF_X4} } {
  set block [ord::get_db_block]
  set dbu [$block getDbUnitsPerMicron]
  foreach n [$block getNets] {
    if {[$n getSigType] ne "SIGNAL"} continue
    set its [$n getITerms]
    if {[llength $its] <= $thr} continue
    set clk 0
    foreach it [lrange $its 0 50] {
      if {[[$it getMTerm] getSigType] eq "CLOCK" || [$it isClocked]} { set clk 1; break }
    }
    if {$clk} continue
    set netname [$n getName]
    set loads {}
    foreach p [get_pins -of_objects [get_nets $netname] -filter "direction == input"] {
      set it [sta::sta_to_db_pin $p]
      if {$it eq "NULL" || $it eq ""} continue
      set xy [$it getAvgXY]
      if {![lindex $xy 0]} { set xy [list 1 {*}[[$it getInst] getLocation]] }
      lappend loads [list $p [lindex $xy 1] [lindex $xy 2]]
    }
    set N [llength $loads]
    set g [expr {int(ceil(sqrt(double($N) / $group)))}]
    set xs [lsort -integer [lmap l $loads {lindex $l 1}]]
    set ys [lsort -integer [lmap l $loads {lindex $l 2}]]
    set x0 [lindex $xs 0]; set y0 [lindex $ys 0]
    set dx [expr {max(1, ([lindex $xs end] - $x0 + $g) / $g)}]
    set dy [expr {max(1, ([lindex $ys end] - $y0 + $g) / $g)}]
    array unset bin
    foreach l $loads {
      lappend bin([expr {([lindex $l 1] - $x0) / $dx}],[expr {([lindex $l 2] - $y0) / $dy}]) $l
    }
    set made 0
    foreach k [lsort [array names bin]] {
      set pins {}; set sx 0; set sy 0
      foreach l $bin($k) { lappend pins [lindex $l 0]; incr sx [lindex $l 1]; incr sy [lindex $l 2] }
      set m [llength $pins]
      if {$m < 2} continue
      # -buffer_name/-net_name are required: without them this build segfaults (null base name)
      insert_buffer -buffer_cell $cell -buffer_name presplit -net_name presplit_net -load_pins $pins \
        -location [list [expr {double($sx) / $m / $dbu}] [expr {double($sy) / $m / $dbu}]]
      incr made
    }
    puts "presplit_heavy_nets: $netname loads $N grid ${g}x$g buffers $made"
  }
}
