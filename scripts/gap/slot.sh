# Sourced by run_arm.sh / run_stock.sh: take one of $GAP_SLOTS global run slots (default 4) and wait while free disk
# is below $GAP_MIN_FREE_GB (default 10). The slot is held (flock on fd 9) until the calling script exits.
# First come, first served: each waiter drops a ticket (wait.<ns>.<pid>) and only the oldest live ticket may take a
# slot, so a run cannot starve behind later arrivals (polling alone let some runs wait for hours). Tickets of dead
# waiters are removed by whoever scans next.
GAP_SLOT_DIR=${NETWT_WORK:-$HOME/work/netwt}/slots; mkdir -p "$GAP_SLOT_DIR"
_gap_n=${GAP_SLOTS:-4}; _gap_min=${GAP_MIN_FREE_GB:-10}
_gap_t="$GAP_SLOT_DIR/wait.$(date +%s%N).$$"; : > "$_gap_t"
while :; do
  _gap_first=""
  for _t in $(ls "$GAP_SLOT_DIR" | grep '^wait\.' | sort); do
    if kill -0 "${_t##*.}" 2>/dev/null; then _gap_first="$GAP_SLOT_DIR/$_t"; break; fi
    rm -f "$GAP_SLOT_DIR/$_t"
  done
  _free=$(df --output=avail -BG "$GAP_SLOT_DIR" | tail -1 | tr -dc 0-9)
  if [ "$_gap_first" = "$_gap_t" ] && [ "$_free" -ge "$_gap_min" ]; then
    for _i in $(seq 0 $((_gap_n - 1))); do
      exec 9>"$GAP_SLOT_DIR/slot$_i"
      if flock -n 9; then echo "slot $_i ($(date -Is), ${_free}G free)" >&2; rm -f "$_gap_t"; break 2; fi
      exec 9>&-
    done
  fi
  sleep 10
done
