"""Pipeline 5: isotope-labelled formulas such as ``[13C6]`` flowing from the CV packages
(unimodpy, psimodpy) and tacular into peptacular, which must give the same mass."""

from __future__ import annotations

import peptacular as pt
import pytest
import tacular as tc
from conftest import HEAVY_BASE

MASS_TOL = 1e-4

# PSI-MOD 02105 lists DiffFormula "O" with DiffMono 31.99; the upstream entry is
# inconsistent, so it says nothing about the packages.
UPSTREAM_INCONSISTENT_PSIMOD = {2105}


def _isotope_unimod_ids() -> list[int]:
    return sorted(
        int(info.id) for info in tc.UNIMOD_LOOKUP.values() if any(k[0].isdigit() for k in info.dict_composition or {})
    )


def _agrees(formula, expected: float) -> bool:
    """True when peptacular parses ``formula`` to ``expected``; parse errors are False."""
    try:
        return abs(pt.chem_mass(formula) - expected) <= MASS_TOL
    except Exception:  # noqa: BLE001 - any parse error means the formula is unusable
        return False


# --------------------------------------------------------------------------- passes


@pytest.mark.parametrize(
    "tag",
    [
        "Formula:[13C6]C-6",
        "UNIMOD:188",
        "+6.020129",
        "U:Label:13C(6)",
        "Label:13C(6)",  # unprefixed works: the colon keeps it off the localisation-score path
    ],
)
def test_heavy_lysine_spellings_agree(tag):
    expected = pt.parse("PEPTIDEK").mass() + 6.020129
    assert pt.parse(f"PEPTIDEK[{tag}]").mass() == pytest.approx(expected, abs=MASS_TOL)


def test_unimodpy_proforma_formula_isotopes(unimod_db):
    checked = 0
    bad = []
    for entry in unimod_db:
        formula = entry.proforma_formula
        if not formula or "[" not in formula:
            continue
        checked += 1
        if not _agrees(formula, entry.delta_mono_mass):
            bad.append((entry.id, formula))
        # and inside a peptide
        base = pt.parse(HEAVY_BASE).mass()
        assert pt.parse(f"{HEAVY_BASE}[Formula:{formula}]").mass() - base == pytest.approx(
            entry.delta_mono_mass, abs=MASS_TOL
        )
    assert checked > 100
    assert not bad


def test_unimodpy_and_tacular_isotope_compositions_agree(unimod_db):
    ids = _isotope_unimod_ids()
    assert len(ids) > 100
    for uid in ids:
        info = tc.UNIMOD_LOOKUP[uid]
        assert pt.chem_mass(dict(info.dict_composition)) == pytest.approx(info.monoisotopic_mass, abs=MASS_TOL), uid
        entry = unimod_db.get_by_id(uid)
        if entry is None:  # tacular's bundled UNIMOD is newer than unimodpy's for a few ids
            continue
        assert pt.chem_mass(entry.proforma_formula) == pytest.approx(info.monoisotopic_mass, abs=MASS_TOL), uid


def test_tacular_psimod_isotope_formulas():
    checked = 0
    for info in tc.PSIMOD_LOOKUP.values():
        if not info.formula or "[" not in info.formula or int(info.id) in UPSTREAM_INCONSISTENT_PSIMOD:
            continue
        checked += 1
        assert pt.chem_mass(info.formula) == pytest.approx(info.monoisotopic_mass, abs=MASS_TOL), info.id
    assert checked > 100


# ------------------------------------------------ regressions (once cross-package bugs)


@pytest.mark.parametrize("uid", [188, 259, 737])
def test_tacular_unimod_isotope_formula_string(uid):
    info = tc.UNIMOD_LOOKUP[uid]
    base = pt.parse(HEAVY_BASE).mass()
    assert pt.parse(f"{HEAVY_BASE}[Formula:{info.formula}]").mass() - base == pytest.approx(
        info.monoisotopic_mass, abs=MASS_TOL
    )


def test_tacular_unimod_isotope_formula_sweep():
    bad = [
        uid
        for uid in _isotope_unimod_ids()
        if not _agrees(tc.UNIMOD_LOOKUP[uid].formula, tc.UNIMOD_LOOKUP[uid].monoisotopic_mass)
    ]
    assert not bad, f"{len(bad)} isotope UNIMOD formulas unusable, e.g. {bad[:5]}"


def _psimod_isotope_entries(psimod_db):
    return [
        e
        for e in psimod_db
        if not e.is_obsolete
        and e.diff_mono is not None
        and e.dict_diff_formula
        and any(k[:1].isdigit() for k in e.dict_diff_formula)
    ]


def test_psimodpy_proforma_diff_formula_isotopes(psimod_db):
    entries = _psimod_isotope_entries(psimod_db)
    assert len(entries) > 100
    bad = [e.id for e in entries if not _agrees(e.proforma_diff_formula, e.diff_mono)]
    assert not bad, f"{len(bad)} entries, e.g. {bad[:5]}"


def test_psimodpy_dict_diff_formula_isotopes(psimod_db):
    entries = _psimod_isotope_entries(psimod_db)
    bad = [e.id for e in entries if not _agrees(dict(e.dict_diff_formula), e.diff_mono)]
    assert not bad, f"{len(bad)} entries, e.g. {bad[:5]}"


def test_psimodpy_and_tacular_agree_on_isotope_formula(psimod_db):
    entry = psimod_db[452]
    info = tc.PSIMOD_LOOKUP[452]
    assert pt.chem_mass(entry.proforma_diff_formula) == pytest.approx(pt.chem_mass(info.formula), abs=MASS_TOL)
