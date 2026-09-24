"""End to end: FASTA protein -> digest -> modified peptides -> fragments -> synthetic
spectrum -> spxtacular annotation -> mzSpecLib on disk -> streamed back.

fastatacular reads the FASTA, peptacular digests and fragments (cross-checked against
``peptacular.fragment_arrays``), spxtacular matches, scores and writes the library,
``MzSpecLibReader`` streams it back, and paftacular recomputes every annotated m/z.
Everything that goes through the file must come back exactly.
"""

from __future__ import annotations

import fastatacular as ft
import numpy as np
import paftacular as pft
import peptacular as pt
import pytest
import spxtacular as spx

# Human serum albumin fragment (UniProt P02768): tryptic peptides with C, M, S and T.
PROTEIN_ID = "sp|P02768|ALBU_HUMAN"
PROTEIN = "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAK"

FRAGMENT_KW = {
    "ion_types": ["b", "y"],
    "charges": [1, 2],
    "neutral_deltas": ["H2O"],
    "isotopes": [0, 1],
}
TOL_PPM = 5.0
# paftacular recomputes the m/z from the annotation; peptacular gives it from the peptide.
RESOLVE_TOL = 1e-9


def _modify(sequence: str) -> str:
    """ProForma with fixed carbamidomethyl C, oxidised M and the first S phosphorylated."""
    out = []
    phospho_done = False
    for aa in sequence:
        if aa == "M":
            out.append("M[Oxidation]")
        elif aa == "S" and not phospho_done:
            out.append("S[Phospho]")
            phospho_done = True
        else:
            out.append(aa)
    prefix = "<[Carbamidomethyl]@C>" if "C" in sequence else ""
    return f"{prefix}{''.join(out)}/2"


@pytest.fixture(scope="module")
def peptides(tmp_path_factory) -> list[pt.ProFormaAnnotation]:
    path = tmp_path_factory.mktemp("fasta") / "protein.fasta"
    ft.write_fasta([ft.SequenceEntry(identifier=PROTEIN_ID, sequence=PROTEIN, pname="Albumin")], path)
    (entry,) = ft.read_fasta(path)
    assert entry.accession == "P02768"
    assert entry.sequence == PROTEIN

    protein = pt.parse(entry.sequence)
    spans = list(protein.digest_spans("trypsin", missed_cleavages=0, min_len=7, max_len=20))
    sequences = [protein[span].sequence for span in spans]
    # the same digest through the functional API
    assert [seq for seq, _ in pt.digest(entry.sequence, "trypsin", min_len=7, max_len=20)] == sequences
    assert len(sequences) >= 4
    # the modifications this test is about all occur
    assert any("C" in s for s in sequences) and any("M" in s for s in sequences)
    assert any("S" in s for s in sequences)
    return [pt.parse(_modify(seq)) for seq in sequences]


def _fragments(peptide: pt.ProFormaAnnotation) -> list[pt.Fragment]:
    frags = list(peptide.fragment(**FRAGMENT_KW))
    assert frags
    return frags


def _spectrum(peptide: pt.ProFormaAnnotation, frags: list[pt.Fragment]) -> spx.MsnSpectrum:
    """One peak per distinct fragment m/z, plus noise far from any fragment."""
    frag_mz = np.array(sorted({f.mz for f in frags}))
    noise = np.linspace(150.0, 1500.0, 25) + 0.4321
    noise = noise[[np.min(np.abs(frag_mz - n)) > 0.05 for n in noise]]
    mz = np.concatenate([frag_mz, noise])
    rng = np.random.default_rng(len(frag_mz))
    intensity = np.concatenate([rng.uniform(1e3, 1e5, len(frag_mz)), np.full(len(noise), 25.0)])
    order = np.argsort(mz)
    return spx.MsnSpectrum(
        mz=mz[order],
        intensity=intensity[order],
        ms_level=2,
        rt=61.5,
        # mzSpecLib peak lists are centroids; the reader marks them so
        spectrum_type=spx.SpectrumType.CENTROID,
        precursors=[spx.Precursor(precursor_mz=peptide.mz(), charge=peptide.charge_state)],
    )


def test_fragment_arrays_agree_with_fragments(peptides):
    for peptide in peptides:
        frags = _fragments(peptide)
        arrays = pt.fragment_arrays(peptide, **FRAGMENT_KW)
        assert set(pt.FRAGMENT_ARRAY_KEYS) <= set(arrays)
        assert len(arrays["mz"]) == len(frags)
        np.testing.assert_array_equal(arrays["mz"], [f.mz for f in frags])
        np.testing.assert_array_equal(arrays["charge_state"], [f.charge_state for f in frags])
        assert list(arrays["ion_type"]) == [str(f.ion_type) for f in frags]
        # the method form gives the same arrays
        method = peptide.fragment_arrays(**FRAGMENT_KW)
        np.testing.assert_array_equal(method["mz"], arrays["mz"])


def test_fasta_to_mzspeclib_roundtrip(peptides, tmp_path):
    entries = []
    expected_labels = []
    for key, peptide in enumerate(peptides, start=1):
        frags = _fragments(peptide)
        spectrum = _spectrum(peptide, frags)

        # every fragment lands on its own peak, with no error
        matches = spx.match_fragments(spectrum, frags, tolerance=TOL_PPM, tolerance_unit="ppm")
        assert {id(m.fragment) for m in matches} == {id(f) for f in frags}
        for m in matches:
            assert spectrum.mz[m.peak_index] == m.fragment.mz
            assert m.ppm_error == pytest.approx(0.0, abs=1e-6)
        result = spx.score(spectrum, frags, tolerance=TOL_PPM, tolerance_unit="ppm")
        assert result["matched_fraction"] == pytest.approx(1.0)
        assert result["mean_ppm_error"] == pytest.approx(0.0, abs=1e-6)

        entry = spx.LibraryEntry.from_spectrum(spectrum, peptide, key=key, peak_annotations=matches)
        assert entry.peak_annotations is not None
        # annotations are grouped per peak: fragment peaks carry the ions that match them,
        # noise peaks carry nothing
        by_peak: dict[int, set[str]] = {}
        for m in matches:
            by_peak.setdefault(m.peak_index, set()).add(pft.to_mzpaf(m.fragment, include_sequence=False).serialize())
        for index, anns in enumerate(entry.peak_annotations):
            got = {a.serialize(include_sequence=False).split("/")[0] for a in anns}
            assert got == by_peak.get(index, set()), (peptide.serialize(), index)
        entries.append(entry)
        expected_labels.append([tuple(a.serialize() for a in anns) for anns in entry.peak_annotations])

    for fmt in ("text", "json"):
        path = spx.write_mzspeclib(entries, tmp_path / f"library.mzspeclib.{fmt}", format=fmt)
        with spx.MzSpecLibReader(path) as reader:
            read = list(reader)
        assert len(read) == len(entries)
        for written, got, labels in zip(entries, read, expected_labels, strict=True):
            assert got.key == written.key
            # peaks: bit-identical
            np.testing.assert_array_equal(got.spectrum.mz, written.spectrum.mz)
            np.testing.assert_array_equal(got.spectrum.intensity, written.spectrum.intensity)
            assert got.spectrum.precursors is not None
            assert got.spectrum.precursors[0].precursor_mz == written.spectrum.precursors[0].precursor_mz
            assert got.spectrum.precursors[0].charge == written.spectrum.precursors[0].charge
            # ProForma: exact text, including the static mod and charge
            assert got.peptidoform is not None and written.peptidoform is not None
            assert got.peptidoform.serialize() == written.peptidoform.serialize()
            assert got.charge == written.charge == 2
            # mzPAF: the same annotations on the same peaks
            assert got.peak_annotations is not None
            assert [tuple(a.serialize() for a in anns) for anns in got.peak_annotations] == labels
            assert got.peak_annotations == written.peak_annotations
            assert got == written


def test_mzpaf_annotations_resolve_to_peptacular_mz(peptides):
    for peptide in peptides:
        analyte = peptide.serialize().split("/")[0]
        for frag in _fragments(peptide):
            text = pft.to_mzpaf(frag, include_sequence=False).serialize()
            resolved = pft.parse(text).resolve(analyte)
            assert resolved.mz() == pytest.approx(frag.mz, abs=RESOLVE_TOL), text
            # the label with the sequence embedded needs no analyte
            embedded = pft.parse(pft.to_mzpaf(frag).serialize())
            assert embedded.mz() == pytest.approx(frag.mz, abs=RESOLVE_TOL), text
