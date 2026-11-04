#!/usr/bin/env bash
set -u
pass=0; fail=0
ok()  { pass=$((pass+1)); }
bad() { fail=$((fail+1)); echo "FAIL: $*"; }

SCRATCH=$(mktemp -d)
cp rotate.sh "$SCRATCH/rotate.sh"
chmod +x "$SCRATCH/rotate.sh"
cd "$SCRATCH" || exit 1

# 1. rotates a log and empties the live file
printf 'one\n' > app.log
chmod 640 app.log
./rotate.sh app.log 3 || bad "rotate exited $?"
[ -f app.log.1 ] && grep -qx one app.log.1 && ok || bad "app.log.1 missing content"
[ -s app.log ] && bad "live log not empty" || ok
[ "$(stat -c '%a' app.log)" = "640" ] && ok || bad "permissions not preserved: $(stat -c '%a' app.log)"

# 2. shifts archives up
printf 'two\n' > app.log
./rotate.sh app.log 3 || bad "second rotate exited $?"
grep -qx two app.log.1 && ok || bad "new archive wrong"
grep -qx one app.log.2 && ok || bad "shift to .2 wrong"

# 3. prunes beyond keep
printf 'three\n' > app.log
./rotate.sh app.log 2 || bad "third rotate exited $?"
[ -e app.log.2 ] && ok || bad ".2 should exist"
[ -e app.log.3 ] && bad ".3 should be pruned" || ok

# 4. default keep = 3
rm -f app.log.*
printf 'a\n' > app.log; ./rotate.sh app.log
printf 'b\n' > app.log; ./rotate.sh app.log
printf 'c\n' > app.log; ./rotate.sh app.log
[ -e app.log.3 ] && ok || bad "default keep should be 3"
[ -e app.log.4 ] && bad "default kept too many" || ok

# 5. bad arguments
./rotate.sh >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "no args should exit 2"
./rotate.sh app.log 0 >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "keep=0 should exit 2"
./rotate.sh app.log abc >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "keep=abc should exit 2"
./rotate.sh missing.log >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "missing file should exit 2"

# 6. repeated runs stay safe
for i in 1 2 3 4; do printf "x$i\n" > app.log; ./rotate.sh app.log 2 || bad "repeat $i failed"; done
[ -e app.log.3 ] && bad "repeat run exceeded keep" || ok

# 7. spaces in path
mkdir -p "dir with space"
printf 'spaced\n' > "dir with space/app.log"
./rotate.sh "dir with space/app.log" 2 || bad "spaced path failed"
grep -qx spaced "dir with space/app.log.1" && ok || bad "spaced archive wrong"

echo "pass=$pass fail=$fail"
[ $fail -eq 0 ]
