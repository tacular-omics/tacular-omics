"""``python -m tacular_omics``: print the installed version of every member package."""

import sys

from tacular_omics import __version__, versions


def main() -> int:
    """Print member package versions; exit 1 if any member is not installed."""
    found = versions()
    width = max(len(name) for name in found)
    print(f"{'tacular-omics':<{width}}  {__version__}")
    for name, ver in found.items():
        print(f"{name:<{width}}  {ver or 'NOT INSTALLED'}")
    return 0 if all(found.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
