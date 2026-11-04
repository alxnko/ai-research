#!/usr/bin/env bash
set -u
pass=0; fail=0
ok()  { pass=$((pass+1)); }
bad() { fail=$((fail+1)); echo "FAIL: $*"; }

SCRATCH=$(mktemp -d)
cp check.sh "$SCRATCH/check.sh"
chmod +x "$SCRATCH/check.sh"
cd "$SCRATCH" || exit 1

counter='"$COUNTFILE"'
run_cmd() { printf '%s' "$1"; }

# 1. success on first try
export COUNTFILE="$SCRATCH/c1"; : > "$COUNTFILE"
out=$(./check.sh 'echo x >> "$COUNTFILE"' 3 2>/dev/null); rc=$?
[ $rc -eq 0 ] && [ "$out" = "ok" ] && ok || bad "first-try rc=$rc out=$out"
[ "$(wc -l < "$COUNTFILE")" -eq 1 ] && ok || bad "ran $(wc -l < "$COUNTFILE") times"

# 2. retries the full count then fails
export COUNTFILE="$SCRATCH/c2"; : > "$COUNTFILE"
out=$(./check.sh 'echo x >> "$COUNTFILE"; exit 7' 4 2>/dev/null); rc=$?
[ $rc -eq 1 ] && [ "$out" = "fail" ] && ok || bad "exhausted rc=$rc out=$out"
[ "$(wc -l < "$COUNTFILE")" -eq 4 ] && ok || bad "expected 4 attempts, got $(wc -l < "$COUNTFILE")"

# 3. one line per attempt on stderr
export COUNTFILE="$SCRATCH/c3"; : > "$COUNTFILE"
err=$(./check.sh 'exit 3' 2 2>&1 >/dev/null)
[ "$(printf '%s\n' "$err" | grep -c 'attempt')" -eq 2 ] && ok || bad "stderr lines: $err"
printf '%s' "$err" | grep -q 'attempt 1/2: 3' && ok || bad "stderr format: $err"

# 4. succeeds after failures
export COUNTFILE="$SCRATCH/c4"; : > "$COUNTFILE"
out=$(./check.sh 'echo x >> "$COUNTFILE"; [ $(wc -l < "$COUNTFILE") -ge 3 ]' 5 2>/dev/null); rc=$?
[ $rc -eq 0 ] && [ "$out" = "ok" ] && ok || bad "late success rc=$rc"
[ "$(wc -l < "$COUNTFILE")" -eq 3 ] && ok || bad "late success attempts=$(wc -l < "$COUNTFILE")"

# 5. defaults: 3 attempts, no sleep
export COUNTFILE="$SCRATCH/c5"; : > "$COUNTFILE"
start=$(date +%s)
out=$(./check.sh 'echo x >> "$COUNTFILE"; exit 1' 2>/dev/null)
elapsed=$(( $(date +%s) - start ))
[ "$out" = "fail" ] && [ "$(wc -l < "$COUNTFILE")" -eq 3 ] && [ "$elapsed" -le 2 ] \
  && ok || bad "defaults: out=$out n=$(wc -l < "$COUNTFILE") t=$elapsed"

# 6. invalid arguments
./check.sh >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "no args should exit 2"
./check.sh 'true' 0 >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "attempts=0 should exit 2"
./check.sh 'true' x >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "attempts=x should exit 2"

# 7. command reading stdin does not hang
out=$(timeout 5 ./check.sh 'cat; exit 1' 1 2>/dev/null); rc=$?
[ $rc -eq 1 ] && [ "$out" = "fail" ] && ok || bad "stdin hang rc=$rc out=$out"

echo "pass=$pass fail=$fail"
[ $fail -eq 0 ]
