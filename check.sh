#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Ruff Lint ==="
uvx ruff@0.17.3 check .

echo "=== 2. Ruff Format Check ==="
uvx ruff@0.17.3 format --check .

echo "=== 3. ty Type Check ==="
uvx ty@0.0.97 check

echo "=== 4. Harness Selfcheck & Smoke Test ==="
python3 bench/selfcheck.py

if command -v bun >/dev/null 2>&1 && [ -f "tsconfig.json" ]; then
  echo "=== 5. Bun Check (TypeScript) ==="
  bun check
fi

echo "All checks passed successfully!"
