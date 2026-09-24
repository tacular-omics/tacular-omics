# Copilot instructions

The canonical guide for AI agents in this repo is [`CLAUDE.md`](../CLAUDE.md). Read it
first. The rules that matter most:

1. This is a meta-package: no functionality beyond `versions()`. Do not add features;
   they belong in a member package.
2. Every member dependency has a floor and a cap (`peptacular>=4.2,<5`,
   `mzmlpy>=0.9.3,<0.10`). Keep `PACKAGES` in `src/tacular_omics/__init__.py` in step
   with `[project] dependencies`.
3. Before pushing, run what CI runs:
   `uv run ruff check src tests scripts/smoke.py && uv run ruff format --check src tests scripts/smoke.py && uv run ty check src && uv run pytest tests`.
4. Never bump the version, tag or publish: only the tacular-omics overseer releases.
