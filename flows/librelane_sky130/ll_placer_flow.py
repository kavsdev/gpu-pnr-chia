import sys
import os
import json
import time
import re
from librelane.common import Path
from librelane.steps.step import Step, ViewsUpdate, MetricsUpdate
from librelane.steps.openroad import GlobalPlacement, _GlobalPlacement
from librelane.flows.classic import Classic
from librelane.state import DesignFormat

# Runs inside ghcr.io/librelane/librelane:3.0.14. sky130A-specific (PDK/pdk_root below, and the Magic-DRC/Netgen-LVS
# metrics keys read at the bottom) -- this is the DREAMPlace-through-LibreLane splice validated on sky130A/spm.
if len(sys.argv) < 4:
    print("Usage: python3 ll_placer_flow.py <design> <run_dir> <tag> [pdk_root]")
    print("  pdk_root defaults to $PDK_ROOT, then a ciel-managed sky130 checkout under this script's directory")
    sys.exit(1)

design = sys.argv[1]
run_dir = sys.argv[2]
tag = sys.argv[3]
pdk_root = sys.argv[4] if len(sys.argv) > 4 else os.environ.get(
    "PDK_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdk", "sky130"))

dp_req_file = os.path.join(run_dir, "dp_req.json")
dp_done_file = os.path.join(run_dir, "dp_done.json")

def components_only_def(src: str, dst: str) -> int:
    out, in_comp, n = [], False, 0
    with open(src) as f:
        for line in f:
            tok = line.split(None, 1)[0] if line.strip() else ""
            if tok in ("VERSION", "DIVIDERCHAR", "BUSBITCHARS", "DESIGN", "UNITS", "DIEAREA"):
                out.append(line)
            elif tok == "COMPONENTS":
                in_comp = True
                out.append(line)
            elif in_comp:
                out.append(line)
                if line.lstrip().startswith("- "):
                    n += 1
                if line.startswith("END COMPONENTS"):
                    in_comp = False
    out.append("END DESIGN\n")
    with open(dst, "w") as f:
        f.writelines(out)
    return n

class LibreLanePlacerStep(Step):
    id = "MyOrg.LibreLaneGlobalPlacement"
    name = "LibreLane Global Placement"
    config_vars = _GlobalPlacement.config_vars

    inputs = [DesignFormat.ODB]
    outputs = [DesignFormat.ODB, DesignFormat.DEF]

    def run(self, state_in, **kwargs):
        in_odb = state_in[DesignFormat.ODB]
        work_dir = self.step_dir

        # 1. Export DEF from ODB
        export_def_tcl = os.path.join(work_dir, "export.tcl")
        pre_gp_def = os.path.join(work_dir, "pre_gp.def")
        with open(export_def_tcl, "w") as f:
            f.write(f"read_db {in_odb}\n")
            f.write(f"write_def {pre_gp_def}\n")
        self.run_subprocess(["openroad", "-exit", export_def_tcl])

        # 2. Extract LEFs. EXTRA_LEFS matters for macro designs (e.g. a real SRAM macro's own LEF lives there, not
        # in MACRO_LEFS) -- without it DREAMPlace can't resolve the macro's pins and crashes looking for them.
        tech_lef = str(self.config["TECH_LEFS"].get("nom_*", list(self.config["TECH_LEFS"].values())[0]))
        cell_lefs = [str(p) for p in self.config["CELL_LEFS"]]
        macro_lefs = [str(p) for p in self.config.get("MACRO_LEFS", [])]
        extra_lefs = [str(p) for p in self.config.get("EXTRA_LEFS", [])]
        lefs = [tech_lef] + cell_lefs + macro_lefs + extra_lefs

        # DREAMPlace's DEF parser mis-identifies SPECIALNETS (the PDN generated before global placement) shapes/vias
        # as macro pins on macro-containing designs and crashes (`macroPinId < numeric_limits<index_type>::max()`).
        # Strip SPECIALNETS from the copy sent to DREAMPlace; PDN is untouched since re-import is `read_def
        # -incremental` on a components-only DEF anyway (see components_only_def below), so nothing is lost.
        def_stripped = os.path.join(work_dir, "pre_gp_stripped.def")
        with open(pre_gp_def) as f_in, open(def_stripped, "w") as f_out:
            in_special = False
            for line in f_in:
                if line.startswith("SPECIALNETS"):
                    in_special = True
                if not in_special:
                    f_out.write(line)
                if in_special and line.startswith("END SPECIALNETS"):
                    in_special = False

        # 3. Create request for host
        dp_out = os.path.join(work_dir, "dp_out")
        os.makedirs(dp_out, exist_ok=True)
        os.chmod(dp_out, 0o777)
        req = {
            "lefs": lefs,
            "def_in": def_stripped,
            "out_dir": os.path.join(work_dir, "dp_out")
        }
        with open(dp_req_file, "w") as f:
            json.dump(req, f)

        # 4. Wait for host to finish DP
        while not os.path.exists(dp_done_file):
            time.sleep(1)

        with open(dp_done_file) as f:
            res = json.load(f)

        if not res.get("ok"):
            raise RuntimeError(res.get("error", "Unknown DREAMPlace error"))

        # 5. Import only components
        components_def_path = os.path.join(work_dir, "components.def")
        n = components_only_def(res["def_path"], components_def_path)

        import_tcl = os.path.join(work_dir, "import.tcl")
        final_odb_path = os.path.join(work_dir, "final.odb")
        final_def_path = os.path.join(work_dir, "final.def")
        with open(import_tcl, "w") as f:
            f.write(f"read_db {in_odb}\n")
            f.write(f"read_def -incremental {components_def_path}\n")
            f.write(f"write_db {final_odb_path}\n")
            f.write(f"write_def {final_def_path}\n")

        out_log = os.path.join(work_dir, "import.log")
        self.run_subprocess(["openroad", "-exit", import_tcl], log_to=Path(out_log))

        gp3 = open(out_log).read()
        evidence = {
            "handoff_read_def_in_3_3": True,
            "gpl_lines_in_3_3": len(re.findall(r"GPL-", gp3)),
        }
        with open(os.path.join(work_dir, "evidence.json"), "w") as f:
            json.dump(evidence, f)

        return {
            DesignFormat.ODB: Path(final_odb_path),
            DesignFormat.DEF: Path(final_def_path),
        }, {}

MyFlow = Classic.Substitute({
    "OpenROAD.GlobalPlacement": LibreLanePlacerStep
})

try:
    flow = MyFlow([f"{design}/config.yaml"], design_dir=design, pdk="sky130A", pdk_root=pdk_root)
    flow.run_dir = Path(run_dir)
    flow.start(tag=tag)

    # After flow finishes, read metrics
    metrics_file = os.path.join(design, "runs", tag, "final", "metrics.json")
    if not os.path.exists(metrics_file):
        raise RuntimeError("Flow finished but metrics.json not found")

    with open(metrics_file) as f:
        metrics = json.load(f)

    ev = {}
    for root, dirs, files in os.walk(os.path.join(design, "runs", tag)):
        if "evidence.json" in files:
            with open(os.path.join(root, "evidence.json")) as f:
                ev = json.load(f)
            break

    out = {
        "ok": True,
        "routed_wl": metrics.get("route__wirelength"),
        "wns": metrics.get("timing__setup__wns"),
        "tns": metrics.get("timing__setup__tns"),
        "power": metrics.get("power__total"),
        "area": metrics.get("design__instance__area"),
        "drc_violations": metrics.get("magic__drc_error__count"),
        "lvs_violations": metrics.get("design__lvs_error__count"),
        "_evidence": ev
    }
except Exception as e:
    out = {"ok": False, "error": str(e)}

flow_done_file = os.path.join(run_dir, "flow_done.json")
with open(flow_done_file, "w") as f:
    json.dump(out, f)
