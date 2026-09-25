#!/opt/hammer/bin/python
"""Custom Hammer CLI driver."""
import json
from typing import List
from hammer.vlsi import CLIDriver, HammerTool
from hammer.vlsi.hooks import HammerToolHookAction
from pydantic.main import BaseModel
from decimal import Decimal

# Patch TechJSON.parse_raw
from hammer.tech import TechJSON
orig_parse_raw = TechJSON.parse_raw

def fixed_parse_raw(cls, b, **kwargs):
    # Needed because Hammer 1.2.0 tech plugin for Nangate45 has invalid grid_unit and macro max_width formats
    # Also fixes tie cell output ports and adds CTS buffer to special_cells
    s = b
    if isinstance(b, bytes):
        s = b.decode("utf-8")
    d = json.loads(s)
    if "grid_unit" in d and "stackups" in d:
        gu = d["grid_unit"]
        for su in d["stackups"]:
            su["grid_unit"] = gu
            if "metals" in su:
                for m in su["metals"]:
                    m["grid_unit"] = gu
                    if "max_width" in m and m["max_width"] is not None:
                        import math
                        m["max_width"] = math.floor(m["max_width"] / float(gu)) * float(gu)
        if "special_cells" in d:
            for sc in d["special_cells"]:
                if sc["cell_type"] in ["tiehicell", "tielocell"]:
                    sc["output_ports"] = ["Z"]
            d["special_cells"].append({"cell_type": "ctsbuffer", "name": ["CLKBUF_X3"]})
    return orig_parse_raw(json.dumps(d), **kwargs)

TechJSON.parse_raw = classmethod(fixed_parse_raw)

import hammer.synthesis.yosys
def new_syn_generic(self) -> bool:
    # Replaces Hammer's default Yosys script with one closer to ORFS (hierarchy check, opt)
    if self.get_setting("synthesis.yosys.latch_map_file") is not None:
        latch_map = f"techmap -map {self.get_setting('synthesis.yosys.latch_map_file')}"
    else:
        latch_map = ""
    
    self.block_append(f"""
    yosys proc
    hierarchy -check -top {self.top_module}
    synth -top {self.top_module}
    opt -purge
    {latch_map}
    """)
    for liberty_file in self.liberty_files_tt.split():
        self.verbose_append(f"dfflibmap -liberty {liberty_file}")
    self.verbose_append("opt")
    self.write_sdc_file()
    return True
hammer.synthesis.yosys.YosysSynth.syn_generic = new_syn_generic

import hammer.par.openroad

def new_global_placement(self) -> bool:
    # Matches ORFS: skip IOs first, set custom layer adjustments, timing/routability driven, incremental placement after IO
    self.block_append("""
    set_global_routing_layer_adjustment metal2-metal3 0.5
    set_global_routing_layer_adjustment metal4-metal10 0.25
    set_routing_layers -signal metal2-metal10 -clock metal4-metal10
    
    # Run global placement without IOs first
    global_placement -skip_io -density 0.65
    
    # Place IO pins
    place_pins -hor_layers metal5 -ver_layers metal6
    
    # Run global placement again to optimize with IO pins
    global_placement -density 0.65 -routability_driven -timing_driven -incremental
    """)
    return True
new_global_placement.__name__ = "global_placement"
hammer.par.openroad.OpenROADPlaceAndRoute.global_placement = new_global_placement

old_floorplan_design = hammer.par.openroad.OpenROADPlaceAndRoute.floorplan_design
def new_floorplan_design(self) -> bool:
    # Removes Hammer's random pin placement and track generation so we can inject ORFS's versions
    res = old_floorplan_design(self)
    for i, line in enumerate(self.output):
        if 'source -verbose' in line:
            self.output[i] = line.replace('source -verbose', 'source')
        if 'place_pins' in line:
            self.output[i] = ""
        if 'make_tracks' in line:
            self.output[i] = ""
    self.output.append("source /OpenROAD-flow-scripts/flow/platforms/nangate45/make_tracks.tcl")
    return res
hammer.par.openroad.OpenROADPlaceAndRoute.floorplan_design = new_floorplan_design

import os
def new_power_straps(self) -> bool:
    # Hammer 1.2.0 pdngen plugin for OpenROAD fails or generates incomplete PDN; we just source power_straps.tcl
    power_straps_tcl_path = os.path.join(self.run_dir, "power_straps.tcl")
    self.write_power_straps_tcl(power_straps_tcl_path)
    self.block_append(f"""
    ################################################################
    # Power distribution network insertion
    source {power_straps_tcl_path}
    # pdngen disabled
    """)
    return True
hammer.par.openroad.OpenROADPlaceAndRoute.power_straps = new_power_straps

def new_detailed_route(self) -> bool:
    # Adds estimate_parasitics before routing and sets bottom/top routing layers appropriately
    metals=self.get_stackup().metals[1:]
    self.block_append(f"""
    ################################################################
    # Detailed routing
    estimate_parasitics -global_routing
    """)
    self.block_append(f"""
    set_propagated_clock [all_clocks]
    set_thread_count {self.get_setting('vlsi.core.max_threads')}
    detailed_route \\
        -output_drc {self.run_dir}/{self.top_module}_route_drc.rpt \\
        -output_maze {self.run_dir}/{self.top_module}_maze.log \\
        -save_guide_updates \\
        -verbose 1
    """)
    return True
hammer.par.openroad.OpenROADPlaceAndRoute.detailed_route = new_detailed_route

def noop_extraction(self) -> bool:
    # OpenRCX is not supported for Nangate45 in this OpenROAD setup (no rules file)
    import os
    with open(os.path.join(self.run_dir, self.top_module + ".par.spef"), "w") as f:
        f.write("*SPEF \"IEEE 1481-1998\"\n")
    return True
hammer.par.openroad.OpenROADPlaceAndRoute.extraction = noop_extraction

def noop_write_gds(self):
    # Skip GDS generation to save time and avoid KLayout dependencies not present here
    with open(self.output_gds_filename, "w") as f:
        f.write("dummy")
    return ""
hammer.par.openroad.OpenROADPlaceAndRoute.write_gds = noop_write_gds

def new_io_placement(self) -> bool:
    self.block_append("""
    # IO placement moved to floorplan
    """)
    return True
new_io_placement.__name__ = "io_placement"
hammer.par.openroad.OpenROADPlaceAndRoute.io_placement = new_io_placement

def new_resize(self) -> bool:
    self.block_append("""
    ################################################################
    # Resizing & Buffering (monkeypatched to match ORFS)
    estimate_parasitics -placement
    repair_design -verbose
    repair_timing -setup_margin 0 -hold_margin 0 -repair_tns 100 -setup -skip_last_gasp -sequence unbuffer,sizeup,swap,vt_swap -verbose
    """)
    return True
new_resize.__name__ = "resize"
hammer.par.openroad.OpenROADPlaceAndRoute.resize = new_resize

def new_detailed_placement(self) -> bool:
    self.block_append("""
    detailed_placement
    """)
    return True
new_detailed_placement.__name__ = "detailed_placement"
hammer.par.openroad.OpenROADPlaceAndRoute.detailed_placement = new_detailed_placement

def _noop_marker(ht: HammerTool) -> bool:
    ht.append("puts \"pnr_driver: custom hook ran\"")
    return True

class PnrCLIDriver(CLIDriver):
    def get_extra_par_hooks(self) -> List[HammerToolHookAction]:
        return [HammerTool.make_pre_insertion_hook("global_placement", _noop_marker)]

if __name__ == "__main__":
    PnrCLIDriver().main()
