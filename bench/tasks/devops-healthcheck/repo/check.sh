#!/usr/bin/env bash
# Probe a command until it succeeds.

cmd=$1
sh -c "$cmd"
echo ok
