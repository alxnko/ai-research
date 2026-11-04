#!/usr/bin/env bash
# Simple log rotation used by the cron job.

logfile=$1
keep=$2

mv $logfile $logfile.1
touch $logfile
