#!/usr/bin/env bash
# Simple log rotation used by the cron job.
set -euo pipefail

if [ "$#" -lt 1 ] || [ -z "${1:-}" ]; then
  echo "usage: $0 <logfile> [keep]" >&2
  exit 2
fi

logfile=$1
keep=${2:-3}

case "$keep" in
  ''|*[!0-9]*) echo "keep must be a positive integer" >&2; exit 2 ;;
esac
if [ "$keep" -lt 1 ]; then
  echo "keep must be a positive integer" >&2
  exit 2
fi
if [ ! -r "$logfile" ]; then
  echo "cannot read $logfile" >&2
  exit 2
fi

perms=$(stat -c '%a' "$logfile")

# Anything at or beyond the retention limit is stale, even if the live file
# never reached that depth (e.g. keep was lowered between runs).
for f in "$logfile".[0-9]*; do
  [ -e "$f" ] || continue
  num=${f##*.}
  case "$num" in ''|*[!0-9]*) continue ;; esac
  if [ "$num" -ge "$keep" ]; then rm -f "$f"; fi
done

i=$((keep - 1))
while [ "$i" -ge 1 ]; do
  if [ -e "${logfile}.${i}" ]; then
    mv -f "${logfile}.${i}" "${logfile}.$((i + 1))"
  fi
  i=$((i - 1))
done

mv -f "$logfile" "${logfile}.1"
: > "$logfile"
chmod "$perms" "$logfile"
