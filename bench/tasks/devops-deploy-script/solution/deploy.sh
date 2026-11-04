#!/usr/bin/env bash
# Deploy helper used by CI and by hand.
set -euo pipefail

env=""
tag=""

while [ $# -gt 0 ]; do
  case "$1" in
    --env)
      [ $# -ge 2 ] || { echo "missing value for --env" >&2; exit 2; }
      env="$2"; shift 2 ;;
    --tag)
      [ $# -ge 2 ] || { echo "missing value for --tag" >&2; exit 2; }
      tag="$2"; shift 2 ;;
    *)
      echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$env" ] || { echo "missing --env" >&2; exit 2; }
[ -n "$tag" ] || { echo "missing --tag" >&2; exit 2; }

echo "deploy $env $tag" > deploy.log
echo "deployed $env $tag"
