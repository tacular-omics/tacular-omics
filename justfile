default: lint format check test

# Install dependencies
install:
    uv sync --all-extras

# Run linting checks
lint:
    uv run ruff check src tests scripts/smoke.py

# Format code
format:
    uv run ruff check --select I --fix src tests scripts/smoke.py
    uv run ruff format src tests scripts/smoke.py

# Run ty type checker
ty:
    uv run ty check src

# lint + ty + test
check:
    just lint
    just ty
    just test

# Run tests
test:
    uv run pytest tests

# Build the package
build:
    uv build -o dist

# Build, then install the wheel in a fresh env from PyPI and import every member
smoke:
    rm -rf dist && uv build -o dist
    cd "$(mktemp -d)" && uv run --isolated --no-project --with "$(ls {{justfile_directory()}}/dist/*.whl)" python {{justfile_directory()}}/scripts/smoke.py

# --- release (standard tacular-omics recipes; canonical copy in the workspace templates/) ---

# Set the version everywhere and date the [Unreleased] changelog section
set-version version:
    python scripts/release_version.py sync --set {{version}}

# Copy __version__ to CITATION.cff / .zenodo.json after editing it by hand
sync-version:
    python scripts/release_version.py sync

# Fail if version metadata disagrees
check-version:
    python scripts/release_version.py check
