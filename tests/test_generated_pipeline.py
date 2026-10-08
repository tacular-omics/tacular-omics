"""Hypothesis: generated modified peptides through peptacular, paftacular, spxtacular,
tacular and unimodpy.

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
import tacular as tc
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
    sequence = draw(st.text(alphabet=AMINO_ACIDS, min_size=1, max_size=15))
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
def test_spectrum_of_theoretical_fragments_scores_maximal(peptide):
    """A spectrum built from exactly the theoretical fragments matches each fragment on
    its own peak and scores matched_fraction = intensity_fraction = 1.

    Catches spxtacular dropping fragments it cannot place (one-residue peptides,
    fragments sharing an m/z, high-charge ions below the lowest b/y m/z) or counting
    variants so that a perfect spectrum scores under 1.
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
    matches = spx.match_fragments(spectrum, frags, tolerance=5.0, tolerance_unit="ppm")
    own = {id(m.fragment) for m in matches if abs(spectrum.mz[m.peak_index] - m.fragment.mz) <= MZ_TOL}
    missing = [f"{f.ion_type}{f.position}^{f.charge_state} {f.mz}" for f in frags if id(f) not in own]
    assert not missing, f"{annot.serialize()}: " + ", ".join(missing[:10])

    result = spx.score(spectrum, frags, tolerance=5.0, tolerance_unit="ppm")
    assert result["matched_fraction"] == pytest.approx(1.0)
    assert result["intensity_fraction"] == pytest.approx(1.0)
    assert result["mean_ppm_error"] == pytest.approx(0.0, abs=1e-6)


@given(peptide=peptides())
def test_neutral_mass_equals_tacular_residues_plus_unimod_deltas(peptide, unimod_mass):
    """peptacular's neutral mass equals residues (tacular) + mod deltas (unimodpy) + H2O,
    summed here.

    Catches a mod counted twice or not at all (N-terminal acetyl plus a residue mod,
    UNIMOD accession vs name), a residue mass table in peptacular that drifts from
    tacular's, or a terminal water added per residue on one-residue peptides.
    """
    elements = tc.ELEMENT_LOOKUP
    water = 2 * elements["H"].mass + elements["O"].mass
    residues = sum(tc.AA_LOOKUP[aa].monoisotopic_mass for aa in peptide.sequence)
    deltas = sum(unimod_mass[name] for name in peptide.mods.values())
    expected = residues + deltas + water

    annot = pt.parse(peptide.proforma())
    assert annot.neutral_mass() == pytest.approx(expected, abs=MASS_TOL), peptide.proforma()
