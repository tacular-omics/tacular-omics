"""Smoke test for an installed tacular-omics: import every member and report versions.

Run against a built wheel in a fresh environment, outside the checkout, e.g.
``uv run --isolated --no-project --with dist/*.whl python scripts/smoke.py``.
"""

import importlib
import sys

import tacular_omics

found = tacular_omics.versions()
missing = [name for name, ver in found.items() if ver is None]
if missing:
    sys.exit(f"not installed: {', '.join(missing)}")
for name in tacular_omics.PACKAGES:
    module = importlib.import_module(name)
    print(f"{name:14} {found[name]:10} {module.__file__}")
print(f"tacular-omics {tacular_omics.__version__}: {len(found)} members import")
