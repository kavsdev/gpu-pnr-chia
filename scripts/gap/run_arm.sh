#!/usr/bin/env bash
# One DREAMPlace-handoff arm (GAP_INVESTIGATION_SPEC.md), inside chia-pnr:dp, native mode.
# usage: run_arm.sh <design> <label> [raw_handoff args...]     e.g. run_arm.sh tinyRocket off_s1 --stage grt --seed 1
# Log (ends with the result JSON and "exit=<rc>"): $NETWT_WORK/logs/arm_<design>_<label>.log
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${NETWT_WORK:-$HOME/work/netwt}
DP=${DP_INSTALL:-$HOME/work/dreamplace/install_td100}
D=$1; L=$2; shift 2
. "$(dirname "$0")/slot.sh"
mkdir -p "$WORK/logs"; LOG=$WORK/logs/arm_${D}_${L}.log
args=$(printf ' %q' "$@")
sg docker -c "docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -e PNR_GPU_ARCH=8.6 -e LEC_CHECK=0 \
 -e LD_LIBRARY_PATH=/opt/dreamplace_td100/dreamplace/ops/timing_heterosta \
 -v $WORK:$WORK -v $REPO:/repo -v $DP:/opt/dreamplace_td100 \
 -w /repo chia-pnr:dp /home/ray/anaconda3/envs/py_worker/bin/python -m pnr_node.raw_handoff --native \
 --work $WORK --dp-install /opt/dreamplace_td100 --python /opt/dp-venv/bin/python --design $D $args" > "$LOG" 2>&1
rc=$?
echo "exit=$rc" >> "$LOG"
exit $rc
