#!/usr/bin/env bash
# Run a job file (one command per line, relative to the repo root; '#' comments) with N jobs in parallel.
# usage: queue.sh <jobfile> [N=2]      progress: $NETWT_WORK/logs/queue_<jobfile basename>.log
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${NETWT_WORK:-$HOME/work/netwt}
J=$(readlink -f "$1"); N=${2:-2}
Q=$WORK/logs/queue_$(basename "$J" .txt).log
mkdir -p "$WORK/logs"
echo "$(date -Is) start $J N=$N" >> "$Q"
grep -v '^\s*#' "$J" | grep -v '^\s*$' | \
  xargs -d '\n' -P "$N" -I{} bash -c 'cd "$0" && s=$(date +%s) && { {}; } ; rc=$?; echo "$(date -Is) rc=$rc $(( $(date +%s)-s ))s {}" >> "$1"' "$REPO" "$Q"
echo "$(date -Is) done" >> "$Q"
