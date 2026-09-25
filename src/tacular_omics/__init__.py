"""tacular-omics: install the tacular-omics proteomics packages together.

This package has no functionality of its own. It depends on every core tacular-omics
package at a tested set of released versions; import the member packages directly.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

__version__ = "0.2.0"

#: Member packages in dependency order (distribution name == import name).
PACKAGES: tuple[str, ...] = (
    "tacular",
    "psimodpy",
    "unimodpy",
    "uniprotptmpy",
    "fastatacular",
    "pefftacular",
    "mzmlpy",
    "tdfpy",
    "peptacular",
    "paftacular",
    "spxtacular",
)


def versions() -> dict[str, str | None]:
    """Return the installed version of each member package, ``None`` if missing.

    Reads installed distribution metadata only; no member package is imported.
    """
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = _dist_version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


__all__ = ["PACKAGES", "__version__", "versions"]
