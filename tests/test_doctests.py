"""Execute every docstring example in the package.

This package has almost no code of its own, so the example floor is zero and inert. The
module floor is what does the work here: it fails if the walk stops collecting modules,
which is the way a runner like this goes quiet without failing.
"""

import doctest
import importlib
import pkgutil

import tacular_omics

# Measured 2026-11-08: two modules (the package and __main__), no docstring examples yet.
# Raise these when modules or examples are added; never lower one without saying why in the
# commit message.
MIN_MODULES = 2
MIN_EXAMPLES = 0


def _all_modules() -> list[object]:
    """Every importable module in the package, found by walking it rather than by a list."""
    mods: list[object] = [tacular_omics]
    for _finder, name, _ispkg in pkgutil.walk_packages(tacular_omics.__path__, prefix=f"{tacular_omics.__name__}."):
        # An import error fails the test on purpose: a module that stops importing must not
        # drop silently out of the count.
        mods.append(importlib.import_module(name))
    return mods


def test_docstring_examples() -> None:
    mods = _all_modules()
    assert len(mods) >= MIN_MODULES, f"the module walk found only {len(mods)}, expected >= {MIN_MODULES}"
    attempted = failed = 0
    for mod in mods:
        result = doctest.testmod(mod, verbose=False)
        attempted += result.attempted
        failed += result.failed
    assert failed == 0, f"{failed} of {attempted} docstring examples failed"
    assert attempted >= MIN_EXAMPLES, f"only {attempted} examples ran, expected >= {MIN_EXAMPLES}"
