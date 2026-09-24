import importlib
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import tacular_omics
from tacular_omics.__main__ import main

PYPROJECT = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))


def test_packages_match_dependencies() -> None:
    deps = [Requirement(r).name for r in PYPROJECT["project"]["dependencies"]]
    assert list(tacular_omics.PACKAGES) == deps


@pytest.mark.parametrize("name", tacular_omics.PACKAGES)
def test_member_imports(name: str) -> None:
    importlib.import_module(name)


def test_versions_all_installed() -> None:
    found = tacular_omics.versions()
    assert list(found) == list(tacular_omics.PACKAGES)
    assert all(found.values()), found


def test_installed_versions_satisfy_pins() -> None:
    found = tacular_omics.versions()
    for raw in PYPROJECT["project"]["dependencies"]:
        req = Requirement(raw)
        assert req.specifier.contains(found[req.name], prereleases=True), (req, found[req.name])


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    assert main() == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["tacular-omics", tacular_omics.__version__]
    for name in tacular_omics.PACKAGES:
        assert name in out
