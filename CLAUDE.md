# tacular-omics — Claude Code Guide

## Project overview

`tacular-omics` is a meta-package: `pip install tacular-omics` installs every core
tacular-omics package at a set of released versions tested together. It has no
functionality of its own beyond `versions()` and `python -m tacular_omics`, which report
the installed member versions.

Place in the tacular-omics graph: the top. It depends on all eleven core packages
(tacular, psimodpy, unimodpy, uniprotptmpy, fastatacular, pefftacular, mzmlpy, tdfpy,
peptacular, paftacular, spxtacular) and nothing depends on it. It is always released
**last** in a batch, after every member it pins is on PyPI.

## Commands

```bash
just install        # uv sync --all-extras
just lint           # ruff check src tests scripts/smoke.py
just format         # ruff isort fix + ruff format -- WRITES FILES
just ty             # ty check src
just test           # pytest tests (imports every member, checks pins)
just check          # lint + ty + test
just smoke          # build the wheel, install it fresh from PyPI, import every member
just check-version  # python scripts/release_version.py check
```

CI (`.github/workflows/ci.yml`): lint, pytest on 3.12-3.14 + macOS/Windows from the
lock, a lowest-direct run (the floors), and a `wheel` job that installs the built wheel
(plain and `[all]`) into a fresh env from PyPI on Linux/macOS/Windows and runs
`scripts/smoke.py`. CI also runs weekly, to catch a member release inside the caps that
breaks the set.

## Architecture

```
src/tacular_omics/
  __init__.py   # __version__ (version source), PACKAGES (member names, dependency order), versions()
  __main__.py   # main(): prints the versions; exit 1 if a member is missing. Also the `tacular-omics` script
tests/test_members.py   # PACKAGES == dependencies, every member imports, installed versions satisfy the pins
scripts/smoke.py        # import every member from an installed wheel (CI wheel job, publish.yml)
scripts/release_version.py  # version sync/check, byte-identical copy of the workspace template
```

## Public API

- `tacular_omics.versions() -> dict[str, str | None]`: installed version of each member
  (`None` if missing), from `importlib.metadata`; imports nothing.
- `tacular_omics.PACKAGES`: member distribution names in dependency order.
- `tacular_omics.__version__`.

## Conventions

- **Pins**: every member in `[project] dependencies` has a floor (the released version
  of the tested set) and a cap (next major, or next minor for 0.x), per the workspace
  sibling pin policy. The `mcp` and `all` extras repeat the same specifiers.
- `PACKAGES` must list exactly the `dependencies` names in the same order (a test
  enforces it). Distribution name == import name for every member.
- Ruff rules E, W, F, I, B, UP; line length 120; `ty check src` clean.

## Gotchas

- Inside the workspace, the members resolve to the local checkouts, so
  `test_installed_versions_satisfy_pins` fails as soon as a local member is bumped past
  a cap. That is the signal to raise the pins in the next tacular-omics release.
- Extras names differ per member: the MCP server extra is `server` for psimodpy,
  unimodpy and uniprotptmpy, and `mcp` elsewhere. tacular, fastatacular, pefftacular and
  spxtacular have no MCP server.

## Releasing

Only the tacular-omics overseer bumps versions or publishes. See `just --list`
(`set-version`, `sync-version`, `check-version`) and the workspace CLAUDE.md release
checklist. Version source: `__version__` in `src/tacular_omics/__init__.py`
(`[tool.hatch.version]`), mirrored in `CITATION.cff` by `scripts/release_version.py`.
A release raises the pins to the member versions just released, then refreshes
`uv.lock` (`just lock tacular-omics` from the workspace root). Publishing is
`publish.yml` on a GitHub release (PyPI trusted publishing, environment `pypi`).

## Workspace note

This repo is also developed inside the tacular-omics uv workspace
(`~/Repos/tacular-omics/packages/tacular-omics`); there `uv run` uses the shared
`.venv` and root `uv.lock`, not this repo's `uv.lock`. See the workspace CLAUDE.md.
