#!/usr/bin/env bash
# Unmodified ORFS (RePlAce global placement), with optional ORFS variable overrides.
# usage: run_stock.sh <design> <label> [--stage grt|route|floorplan] [--seed N] [--from DIR] [K=V ...]
#   --stage grt   stop after global route (target 5_1_grt.odb); default route = the whole flow
#   --stage floorplan   stop after floorplan (2_floorplan.odb + .sdc, as ORFS `make floorplan`): a --from source
#   --seed N      GPL_RANDOM_SEED=N (gpl's own default is 1; ariane133's config.mk sets 3)
#   --from DIR    reuse synthesis + floorplan (stages 1_*, 2_*) from another stock work dir of the same design;
#                 only valid when the overrides do not affect those stages (GPL_*, PLACE_DENSITY* do not)
# Work dir: $NETWT_WORK/stock_<design>_<label>; log: $NETWT_WORK/logs/stock_<design>_<label>.log (ends "exit=<rc>")
set -u
WORK=${NETWT_WORK:-$HOME/work/netwt}
D=$1; L=$2; shift 2
STAGE=route; VARS=""; FROM=""
while [ $# -gt 0 ]; do
  case $1 in
    --stage) STAGE=$2; shift 2;;
    --seed) VARS="$VARS GPL_RANDOM_SEED=$2"; shift 2;;
    --from) FROM=$2; shift 2;;
    *=*) VARS="$VARS $1"; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
. "$(dirname "$0")/slot.sh"
WH=$WORK/stock_${D}_${L}; SUB=nangate45/$D/base
mkdir -p "$WORK/logs" "$WH"; LOG=$WORK/logs/stock_${D}_${L}.log
if [ -n "$FROM" ]; then
  for k in results logs reports objects; do
    mkdir -p "$WH/$k/$SUB"
    [ -d "$FROM/$k/$SUB" ] || continue
    ( cd "$FROM/$k/$SUB" && ls | grep -E '^(1_|2_|abc\.constr|klayout|copyright|rtlmp|synth_|clock_period\.txt|mem\.json)' | \
      xargs -r -I{} cp -a {} "$WH/$k/$SUB/" )
  done
fi
TARGET=""; [ "$STAGE" = grt ] && TARGET=$WH/results/$SUB/5_1_grt.odb
[ "$STAGE" = floorplan ] && TARGET="$WH/results/$SUB/2_floorplan.odb $WH/results/$SUB/2_floorplan.sdc"  # = ORFS `floorplan`
echo "stock $D $L stage=$STAGE vars=[$VARS] from=[$FROM]" > "$LOG"
sg docker -c "docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e LEC_CHECK=0 -v $WH:$WH chia-pnr:dp bash -lc \
 'source /OpenROAD-flow-scripts/env.sh >/dev/null 2>&1; cd /OpenROAD-flow-scripts/flow && /usr/bin/time -v make \
  DESIGN_CONFIG=./designs/nangate45/$D/config.mk WORK_HOME=$WH $VARS $TARGET'" >> "$LOG" 2>&1
rc=$?
echo "{\"design\": \"$D\", \"label\": \"$L\", \"stage\": \"$STAGE\", \"vars\": \"${VARS# }\", \"from\": \"$FROM\", \"rc\": $rc}" > "$WH/stock_run.json"
echo "exit=$rc" >> "$LOG"
exit $rc
