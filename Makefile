# Convenience entry points, not a second implementation of verification.
# ./check.sh and the experiment-specific scripts remain the authorities.
.DEFAULT_GOAL := help
.PHONY: help setup check selfcheck format report report-json status doctor

help:
	@printf '%s\n' 'AI research developer commands:' \
	  '  make setup         Sync the locked Python environment' \
	  '  make check         Run the full local CI-equivalent check suite' \
	  '  make selfcheck     Run offline harness regression and golden checks' \
	  '  make format        Format project code (not task fixtures)' \
	  '  make report        Analyze EXP-2 (no GPU; incomplete runs are flagged)' \
	  '  make report-json   Machine-readable EXP-2 analysis' \
	  '  make status        Print Git status and current revision' \
	  '  make doctor        Check local prerequisites without changing anything'

setup:
	uv sync --locked

check:
	./check.sh

selfcheck:
	uv run --locked python bench/selfcheck.py

format:
	uvx ruff@0.17.3 format .

report:
	uv run --locked python experiments/analyze_exp2.py

report-json:
	uv run --locked python experiments/analyze_exp2.py --json

status:
	@git status --short
	@git describe --tags --always --dirty

doctor:
	@printf '%s\n' 'Required:'
	@for tool in git uv bwrap systemd-run; do \
	  if command -v "$$tool" >/dev/null 2>&1; then printf '  OK  %s\n' "$$tool"; \
	  else printf '  MISSING  %s\n' "$$tool"; exit 1; fi; \
	done
	@printf '%s\n' 'GPU benchmark prerequisites (optional for code review):'
	@for tool in ollama nvidia-smi; do \
	  if command -v "$$tool" >/dev/null 2>&1; then printf '  OK  %s\n' "$$tool"; \
	  else printf '  NOT INSTALLED  %s\n' "$$tool"; fi; \
	done
