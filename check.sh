#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Ruff Lint ==="
uvx ruff check .

echo "=== 2. Ruff Format Check ==="
uvx ruff format --check .

echo "=== 3. ty Type Check ==="
uvx ty check

echo "=== 4. Harness Selfcheck ==="
python3 bench/selfcheck.py

if command -v bun >/dev/null 2>&1 && [ -f "tsconfig.json" ]; then
  echo "=== 5. Bun Check (TypeScript) ==="
  bun check
fi

echo "All checks passed successfully!"
