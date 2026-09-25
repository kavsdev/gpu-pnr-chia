#!/bin/bash
set -e
# skip syn
# skip syn-to-par
cat << 'EOF' > build/par-input.json
{
    "vlsi.core.synthesis_tool": "hammer.synthesis.yosys",
    "vlsi.core.par_tool": "hammer.par.openroad",
    "vlsi.core.technology": "hammer.technology.nangate45",
    "vlsi.core.node": 45,
    "technology.nangate45.install_dir": "/OpenROAD-flow-scripts/flow/platforms/nangate45",
    "technology.core.stackup": "nangate45_3Ma_2Mb_2Mc_2Md",
    "vlsi.core.top_module": "gcd",
    "vlsi.inputs.supplies.VDD": "1.1 V",
    "vlsi.inputs.supplies.VSS": "0 V",
    "vlsi.inputs.supplies.power_nets": ["VDD"],
    "vlsi.inputs.supplies.ground_nets": ["VSS"],
    "vlsi.inputs.clocks": [
        {
            "name": "core_clock",
            "path": "clk",
            "period": "0.25 ns",
            "uncertainty": "0.01 ns"
        }
    ],
    "vlsi.inputs.placement_constraints": [
        {
            "path": "gcd",
            "type": "toplevel",
            "x": 0,
            "y": 0,
            "width": 35.755,
            "height": 35.755,
            "margins": {
                "left": 1.14,
                "right": 1.175,
                "top": 2.155,
                "bottom": 1.4
            }
        }
    ],
    "vlsi.inputs.pin_mode": "none",
    "par.openroad.floorplan_mode": "manual",
    "par.openroad.floorplan_script_contents": "initialize_floorplan -site FreePDK45_38x28_10R_NP_162NW_34O -die_area {0 0 35.755 35.755} -core_area {1.14 1.4 34.58 33.6}",
    "par.openroad.openrcx_techfiles": [],
    "par.openroad.klayout_techfile_source": "/dev/null",
    "par.openroad.write_reports": false,
    "par.openroad.setrc_file": "/OpenROAD-flow-scripts/flow/platforms/nangate45/setRC.tcl",
    "par.power_straps_mode": "empty",
    "par.inputs.input_files": ["/parity/results/nangate45/gcd/base/1_2_yosys.v"],
    "par.inputs.top_module": "gcd",
    "par.inputs.post_synth_sdc": "/parity/results/nangate45/gcd/base/1_synth.sdc"
}
EOF
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v ${WORK:-$HOME/work}/shared:/shared -v ${WORK:-$HOME/work}/parity_runs:/parity -w /parity/hammer_gcd_test chia-pnr:hammer bash -lc "./pnr_driver.py -p build/par-input.json --obj_dir build par"
