"""Hypothesis: generated modified peptides through peptacular, paftacular, spxtacular
and unimodpy.

The fixed-example suites (test_pipeline_end_to_end, test_cross_package_agreement,
test_fragment_mzpaf) cover a handful of hand-picked peptides. These tests draw the
combinations those miss: one-residue peptides, a modification on the first or last
residue, an N-terminal acetyl next to a modified first residue, two modifications at
once, and every precursor charge 1-4. Each property compares two packages, or a package
against arithmetic done in this file, never a package against itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import paftacular as pft
import peptacular as pt
import pytest
import spxtacular as spx
from hypothesis import given
from hypothesis import strategies as st

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# (Unimod name, Unimod id, residues it may sit on). "^" is the peptide N-terminus.
MODS = [
    ("Phospho", 21, "STY"),
    ("Oxidation", 35, "M"),
    ("Carbamidomethyl", 4, "C"),
    ("Acetyl", 1, "^"),
]

# Monoisotopic residue masses (amino acid minus H2O), typed by hand so the mass check
# does not share peptacular's table (tacular.AA_LOOKUP). Computed from each residue's
# elemental formula with the AME2020 atomic masses below (Wang et al., Chinese Phys. C
# 45, 030003, 2021); they agree with the Unimod amino acid table to its 6 decimals.
_C, _H, _N, _O = 12.0, 1.00782503223, 14.00307400443, 15.99491461957
WATER = 2 * _H + _O
RESIDUE_MASS = {
    "A": 71.037113785,  # C3H5NO
    "C": 103.009184960,  # C3H5NOS
    "D": 115.026943024,  # C4H5NO3
    "E": 129.042593089,  # C5H7NO3
    "F": 147.068413914,  # C9H9NO
    "G": 57.021463721,  # C2H3NO
    "H": 137.058911858,  # C6H7N3O
    "I": 113.084063979,  # C6H11NO
    "K": 128.094963015,  # C6H12N2O
    "L": 113.084063979,  # C6H11NO
    "M": 131.040485088,  # C5H9NOS
    "N": 114.042927441,  # C4H6N2O2
    "P": 97.052763850,  # C5H7NO
    "Q": 128.058577506,  # C5H8N2O2
    "R": 156.101111024,  # C6H12N4O
    "S": 87.032028405,  # C3H5NO2
    "T": 101.047678469,  # C4H7NO2
    "V": 99.068413914,  # C5H9NO
    "W": 186.079312951,  # C11H10N2O
    "Y": 163.063328534,  # C9H9NO2
}

# Same masses summed in a different order: agreement is exact to rounding.
MZ_TOL = 1e-6
MASS_TOL = 1e-6

# mzPAF ion types that paftacular can resolve against a peptide, and the losses a
# phospho/oxidised peptide shows.
FRAGMENT_KW = {
    "ion_types": ["a", "b", "c", "x", "y", "z"],
    "neutral_deltas": ["H2O", "H3PO4"],
    "isotopes": [0, 1],
}


@dataclass(frozen=True)
class Peptide:
    sequence: str
    mods: dict[int, str]  # residue index (-1 = N-terminus) -> Unimod name
    charge: int
    accession_spelling: bool  # write [UNIMOD:21] instead of [Phospho]

    def proforma(self) -> str:
        ids = {name: uid for name, uid, _ in MODS}

        def tag(name: str) -> str:
            return f"[UNIMOD:{ids[name]}]" if self.accession_spelling else f"[{name}]"

        nterm = f"{tag(self.mods[-1])}-" if -1 in self.mods else ""
        body = "".join(aa + (tag(self.mods[i]) if i in self.mods else "") for i, aa in enumerate(self.sequence))
        return f"{nterm}{body}/{self.charge}"


@st.composite
def peptides(draw) -> Peptide:
    sequence = draw(st.text(alphabet=AMINO_ACIDS, min_size=1, max_size=8))
    sites = [(-1, "Acetyl")] + [
        (i, name) for i, aa in enumerate(sequence) for name, _, residues in MODS if aa in residues
    ]
    chosen = draw(st.lists(st.sampled_from(sites), max_size=2, unique_by=lambda s: s[0]))
    return Peptide(
        sequence=sequence,
        mods=dict(chosen),
        charge=draw(st.integers(1, 4)),
        accession_spelling=draw(st.booleans()),
    )


def _fragments(annot: pt.ProFormaAnnotation) -> list[pt.Fragment]:
    return list(annot.fragment(charges=list(range(1, annot.charge_state + 1)), **FRAGMENT_KW))


@pytest.fixture(scope="module")
def unimod_mass(unimod_db) -> dict[str, float]:
    return {name: unimod_db.get_by_id(uid).delta_mono_mass for name, uid, _ in MODS}


@given(peptides())
def test_proforma_round_trip_keeps_mass_and_composition(peptide):
    """parse -> serialize -> parse keeps mass and composition.

    Catches a serializer that drops or moves a modification it did not write itself:
    an N-terminal acetyl next to a modified first residue, a second mod on a short
    peptide, or a UNIMOD accession rewritten to a name that resolves differently.
    """
    first = pt.parse(peptide.proforma())
    text = first.serialize()
    second = pt.parse(text)
    assert second.serialize() == text
    assert second.charge_state == peptide.charge
    assert second.comp() == first.comp(), text
    assert second.neutral_mass() == pytest.approx(first.neutral_mass(), abs=MASS_TOL), text
    assert second.mz() == pytest.approx(first.mz(), abs=MZ_TOL), text


@given(peptides())
def test_mzpaf_label_resolves_to_peptacular_fragment_mz(peptide):
    """Every peptacular fragment's peptide-free mzPAF label, resolved by paftacular
    against the peptide, gives peptacular's m/z.

    Catches an off-by-one in which residues a/b/c vs x/y/z ions span when a mod sits
    on the first or last residue or the N-terminus (the label carries no sequence, so
    paftacular must slice the peptide the same way peptacular did), and charge or
    loss handling that differs at charges 3-4.
    """
    annot = pt.parse(peptide.proforma())
    analyte = annot.serialize().split("/")[0]
    failures = []
    for frag in _fragments(annot):
        text = pft.to_mzpaf(frag, include_sequence=False).serialize()
        got = pft.parse(text).resolve(analyte).mz()
        if abs(got - frag.mz) > MZ_TOL:
            failures.append(f"{text}: paftacular {got}, peptacular {frag.mz}")
    assert not failures, f"{analyte}\n" + "\n".join(failures[:10])


@given(peptides())
def test_score_of_perfect_spectrum_is_maximal(peptide):
    """spxtacular.score() on a spectrum made of exactly peptacular's theoretical fragments
    gives matched_fraction = intensity_fraction = 1 and zero mean ppm error.

    Fragment-to-peak matching itself is covered by test_cross_package_agreement; this
    checks only the score fractions. Catches score() counting neutral-loss or isotope
    variants, or fragments that share one m/z, so that a perfect spectrum scores under 1.
    """
    annot = pt.parse(peptide.proforma())
    frags = _fragments(annot)
    mz = np.unique(np.array([f.mz for f in frags]))
    spectrum = spx.MsnSpectrum(
        mz=mz,
        intensity=np.full(mz.size, 100.0),
        ms_level=2,
        spectrum_type=spx.SpectrumType.CENTROID,
        precursors=[spx.Precursor(precursor_mz=annot.mz(), charge=peptide.charge)],
    )
    result = spx.score(spectrum, frags, tolerance=5.0, tolerance_unit="ppm")
    assert result["matched_fraction"] == pytest.approx(1.0), annot.serialize()
    assert result["intensity_fraction"] == pytest.approx(1.0), annot.serialize()
    assert result["mean_ppm_error"] == pytest.approx(0.0, abs=1e-6), annot.serialize()


@given(peptide=peptides())
def test_neutral_mass_equals_hand_typed_residues_plus_unimod_deltas(peptide, unimod_mass):
    """peptacular's neutral mass equals hand-typed residue masses + mod deltas (unimodpy)
    + H2O, summed here.

    Catches a mod counted twice or not at all (N-terminal acetyl plus a residue mod,
    UNIMOD accession vs name), a wrong residue mass, or a terminal water added per
    residue on one-residue peptides.
    """
    residues = sum(RESIDUE_MASS[aa] for aa in peptide.sequence)
    deltas = sum(unimod_mass[name] for name in peptide.mods.values())
    expected = residues + deltas + WATER

    annot = pt.parse(peptide.proforma())
    assert annot.neutral_mass() == pytest.approx(expected, abs=MASS_TOL), peptide.proforma()
