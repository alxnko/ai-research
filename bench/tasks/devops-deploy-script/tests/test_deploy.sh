#!/usr/bin/env bash
# Hidden acceptance test. Runs in a scratch dir with a fake toolchain.
set -u
pass=0; fail=0
ok()   { pass=$((pass+1)); }
bad()  { fail=$((fail+1)); echo "FAIL: $*"; }

SCRATCH="$(mktemp -d)"
cp deploy.sh "$SCRATCH/deploy.sh"
chmod +x "$SCRATCH/deploy.sh"
cd "$SCRATCH" || exit 1

# 1. happy path
out=$(./deploy.sh --env prod --tag v1.2.3 2>/dev/null); rc=$?
[ $rc -eq 0 ] && ok || bad "happy path rc=$rc"
[ "$out" = "deployed prod v1.2.3" ] && ok || bad "stdout was '$out'"
grep -qx "deploy prod v1.2.3" deploy.log && ok || bad "deploy.log wrong"

# 2. missing tag
./deploy.sh --env prod >/dev/null 2>&1; rc=$?
[ $rc -eq 2 ] && ok || bad "missing --tag should exit 2, got $rc"

# 3. unknown flag
./deploy.sh --env prod --tag v1 --force >/dev/null 2>&1; rc=$?
[ $rc -eq 2 ] && ok || bad "unknown flag should exit 2, got $rc"

# 4. dangling value
./deploy.sh --env >/dev/null 2>&1; rc=$?
[ $rc -ne 0 ] && ok || bad "dangling --env should fail"

# 5. unset variable is an error
grep -q 'set -u' deploy.sh || grep -q 'set -euo' deploy.sh && ok || bad "set -u missing"

# 6. fail fast
grep -q 'set -e' deploy.sh && ok || bad "set -e missing"

# 7. spaces in path
SPACED="$SCRATCH/dir with space"
mkdir -p "$SPACED"; cp deploy.sh "$SPACED/"; cd "$SPACED" || exit 1
out=$(./deploy.sh --env staging --tag v9 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$out" = "deployed staging v9" ] && ok || bad "spaced path rc=$rc out='$out'"

echo "pass=$pass fail=$fail"
[ $fail -eq 0 ]
