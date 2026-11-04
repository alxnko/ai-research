#!/usr/bin/env bash
# Probe a command until it succeeds.
set -uo pipefail

if [ "$#" -lt 1 ] || [ -z "${1:-}" ]; then
  echo "usage: $0 <command> [attempts] [sleep_seconds]" >&2
  exit 2
fi

cmd=$1
attempts=${2:-3}
sleep_s=${3:-0}

for value in "$attempts" "$sleep_s"; do
  case "$value" in
    ''|*[!0-9]*) echo "attempts and sleep must be non-negative integers" >&2; exit 2 ;;
  esac
done
if [ "$attempts" -lt 1 ]; then
  echo "attempts must be at least 1" >&2
  exit 2
fi

n=1
while [ "$n" -le "$attempts" ]; do
  sh -c "$cmd" </dev/null
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo ok
    exit 0
  fi
  echo "attempt $n/$attempts: $rc" >&2
  n=$((n + 1))
  if [ "$n" -le "$attempts" ] && [ "$sleep_s" -gt 0 ]; then
    sleep "$sleep_s"
  fi
done

echo fail
exit 1
