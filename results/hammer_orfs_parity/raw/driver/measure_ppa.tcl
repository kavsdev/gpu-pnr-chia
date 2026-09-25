read_liberty /OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_lef /OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef
read_lef /OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef
read_db $::env(ODB_FILE)
read_sdc $::env(SDC_FILE)

# Setup RC and Clocks
source /OpenROAD-flow-scripts/flow/platforms/nangate45/setRC.tcl
set_propagated_clock [all_clocks]

estimate_parasitics -global_routing

# Area and Instances
set block [[[::ord::get_db] getChip] getBlock]
set insts [$block getInsts]
set inst_count 0
set inst_area 0.0

foreach inst $insts {
    set master [$inst getMaster]
    set mtype [$master getType]
    set is_filler [string match "CORE_SPACER" $mtype]
    # Skip filler/tap cells by master name, then by master type.
    set mname [$master getName]
    
    if { [string match "FILLCELL*" $mname] || [string match "TAPCELL*" $mname] || [string match "WELLTAP*" $mname] } {
        continue
    }
    
    if { $mtype == "CORE_SPACER" || $mtype == "CORE_WELLTAP" } {
        continue
    }
    
    incr inst_count
    set dx [expr {[$master getWidth] / 2000.0}]
    set dy [expr {[$master getHeight] / 2000.0}]
    set inst_area [expr {$inst_area + ($dx * $dy)}]
}

set wns 0.0
set tns 0.0

# Capture WNS and TNS from report_wns / report_tns
# WNS
set wns_out [report_wns]
if {[regexp {([-0-9.]+)} $wns_out match]} {
    set wns $match
}

# TNS
set tns_out [report_tns]
if {[regexp {([-0-9.]+)} $tns_out match]} {
    set tns $match
}

# Power
set power_out [report_power -digits 4]
set total_power 0.0
foreach line [split $power_out "\n"] {
    if {[string match "Total *" $line]} {
        set tokens [regexp -all -inline {\S+} $line]
        if {[llength $tokens] >= 5} {
            set total_power [lindex $tokens 4]
        }
    }
}

# Routed Wirelength
set total_wl 0.0
foreach net [$block getNets] {
    set wire [$net getWire]
    if { $wire != "NULL" } {
        set wl [$wire length]
        set total_wl [expr {$total_wl + $wl / 2000.0}]  ;# DBU -> um (NanGate45: 2000 DBU/um)
    }
}

puts "=== MEASUREMENT START ==="
puts "inst_count: $inst_count"
puts "inst_area: $inst_area"
puts "WNS: $wns"
puts "TNS: $tns"
puts "total_power: $total_power"
puts "routed_wirelength: $total_wl"
puts "=== MEASUREMENT END ==="
