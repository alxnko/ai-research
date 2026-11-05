#!/usr/bin/env bash
# Deploy helper used by CI and by hand.

env = ""
tag = ""

while [ $# -gt 0 ]; do
  case $1 in
    --env) env=$2; shift;;
    --tag) tag=$2; shift;;
    *) echo "unknown option $1"; exit 0;;
  esac
  shift
done

echo "deploy $env $tag" > deploy.log
echo "deployed $env $tag"
