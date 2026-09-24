"""The shared Literal aliases: mzmlpy and tdfpy re-declare ``ToleranceUnit`` and
``Polarity`` (they do not depend on tacular), so they must stay equal to
``tacular.types``: same values, same order."""

from __future__ import annotations

from typing import get_args

import mzmlpy
import pytest
import tacular
import tdfpy
from tacular import types

EXPECTED = {
    "ToleranceUnit": ("da", "ppm"),
    "Polarity": ("positive", "negative"),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_tacular_types_values(name: str) -> None:
    assert get_args(getattr(types, name)) == EXPECTED[name]
    # the top-level re-export is the same alias
    assert get_args(getattr(tacular, name)) == EXPECTED[name]


@pytest.mark.parametrize("module", [mzmlpy, tdfpy], ids=lambda m: m.__name__)
@pytest.mark.parametrize("name", list(EXPECTED))
def test_member_alias_matches_tacular(module, name: str) -> None:
    assert name in module.__all__, f"{module.__name__} does not export {name}"
    assert get_args(getattr(module, name)) == get_args(getattr(types, name))
