"""Pipeline 3: a synthetic spectrum built from peptacular fragments -> spxtacular
match/annotate -> mzPAF labels (paftacular) -> round-trip; then mzML written to disk,
read back through mzmlpy and spxtacular, gives the same peaks."""

from __future__ import annotations

import shutil
import warnings

import mzmlpy
import numpy as np
import paftacular as pft
import peptacular as pt
import pytest
import spxtacular as spx
from conftest import PACKAGES, SyntheticScan, write_indexed_mzml, write_mzml

PEPTIDE = "PEPT[Phospho]IDEM[Oxidation]K/2"
TOL_PPM = 5.0


@pytest.fixture(scope="module")
def fragments():
    return list(pt.parse(PEPTIDE).fragment(ion_types=["b", "y"], charges=[1, 2], neutral_deltas=["H2O"]))


@pytest.fixture(scope="module")
def synthetic(fragments):
    """Every fragment as a peak, plus noise peaks well away from any fragment."""
    rng = np.random.default_rng(7)
    frag_mz = np.array(sorted({round(f.mz, 9) for f in fragments}))
    noise = np.linspace(150.0, 1100.0, 30) + 0.4321
    noise = noise[[np.min(np.abs(frag_mz - n)) > 0.05 for n in noise]]
    mz = np.concatenate([frag_mz, noise])
    intensity = np.concatenate([rng.uniform(1e3, 1e5, len(frag_mz)), np.full(len(noise), 25.0)])
    order = np.argsort(mz)
    return spx.Spectrum(mz=mz[order], intensity=intensity[order]), frag_mz


def test_every_fragment_matches_its_peak(synthetic, fragments):
    spectrum, _ = synthetic
    matches = spx.match_fragments(spectrum, fragments, tolerance=TOL_PPM, tolerance_unit="ppm")
    matched = {id(m.fragment) for m in matches}
    assert matched == {id(f) for f in fragments}
    for m in matches:
        assert abs(m.ppm_error) < 1e-3
        assert spectrum.mz[m.peak_index] == pytest.approx(m.fragment.mz, abs=1e-8)


def test_annotation_labels_are_valid_mzpaf_at_the_peak_mz(synthetic, fragments):
    spectrum, frag_mz = synthetic
    table = spx.build_annot_plot_table(
        spectrum,
        fragments,
        tolerance=TOL_PPM,
        tolerance_unit="ppm",
        include_sequence=True,
        max_labels=None,
    )
    # every fragment peak is annotated (series b/y); direct labels are thinned for
    # readability by design, but the hover text carries the full mzPAF for each peak
    matched = table[table["series"] != "unmatched"]
    assert len(matched) == len(frag_mz)
    assert (table.loc[table["series"] == "unmatched", "label"] == "").all()
    analyte = PEPTIDE.split("/")[0]
    for row in matched.itertuples():
        # hover: "m/z: ...<br>intensity: ...<br>ann1<br>ann2..."; one line per annotation
        annotations = str(row.hover).split("<br>")[2:]
        assert annotations
        if row.label:
            assert row.label == "<br>".join(annotations)
        for label in annotations:
            ann = pft.parse(label)
            assert ann.serialize() == label
            assert ann.mz() == pytest.approx(row.mz, abs=1e-5), label
            # the same label without the embedded sequence resolves against the analyte
            bare = pft.parse(ann.serialize(include_sequence=False)).resolve(analyte)
            assert bare.mz() == pytest.approx(row.mz, abs=1e-5), label


def test_score_on_perfect_spectrum(synthetic, fragments):
    spectrum, _ = synthetic
    result = spx.score(spectrum, fragments)
    assert result["matched_fraction"] == pytest.approx(1.0)
    assert result["mean_ppm_error"] == pytest.approx(0.0, abs=1e-3)


def _scans(synthetic, precursor_intensity):
    spectrum, _ = synthetic
    precursor_mz = pt.parse(PEPTIDE).mz()
    ms1 = SyntheticScan(
        mz=np.array([precursor_mz, precursor_mz + 0.5017, 1000.0]),
        intensity=np.array([5e5, 2e5, 1e3]),
        ms_level=1,
        rt_seconds=59.5,
    )
    ms2 = SyntheticScan(
        mz=spectrum.mz,
        intensity=spectrum.intensity,
        ms_level=2,
        rt_seconds=60.0,
        precursor_mz=precursor_mz,
        precursor_charge=2,
        precursor_intensity=precursor_intensity,
    )
    return ms1, ms2


@pytest.mark.parametrize("compress", [True, False])
def test_mzml_roundtrip_through_mzmlpy(tmp_path, synthetic, compress):
    ms1, ms2 = _scans(synthetic, precursor_intensity=5e5)
    path = write_mzml(tmp_path / "synthetic.mzML", [ms1, ms2], compress=compress)
    with mzmlpy.Mzml(str(path)) as reader:
        spectra = list(reader.spectra)
        assert [s.ms_level for s in spectra] == [1, 2]
        for written, read in zip((ms1, ms2), spectra, strict=True):
            np.testing.assert_array_equal(read.mz, written.mz)
            np.testing.assert_array_equal(read.intensity, written.intensity)
        ion = spectra[1].precursors[0].selected_ions[0]
        assert ion.mz == pytest.approx(ms2.precursor_mz, abs=1e-9)
        assert ion.charge == 2


def test_mzml_roundtrip_through_spxtacular_and_rematch(tmp_path, synthetic, fragments):
    ms1, ms2 = _scans(synthetic, precursor_intensity=5e5)
    path = write_mzml(tmp_path / "synthetic.mzML", [ms1, ms2])
    with spx.MzmlReader(path) as reader:
        read = next(iter(reader.ms2))
    np.testing.assert_array_equal(read.mz, ms2.mz)
    np.testing.assert_array_equal(read.intensity, ms2.intensity)
    assert read.rt == pytest.approx(60.0)
    assert read.precursors is not None
    assert read.precursors[0].precursor_mz == pytest.approx(ms2.precursor_mz, abs=1e-9)
    assert read.precursors[0].charge == 2
    # annotation after the round trip gives the same mzPAF labels as before it
    original, _ = synthetic
    before = [
        pft.to_mzpaf(m.fragment).serialize()
        for m in spx.match_fragments(original, fragments, tolerance=TOL_PPM, tolerance_unit="ppm")
    ]
    after = [
        pft.to_mzpaf(m.fragment).serialize()
        for m in spx.match_fragments(read, fragments, tolerance=TOL_PPM, tolerance_unit="ppm")
    ]
    assert after == before


def test_indexed_gzip_mzml_roundtrip(tmp_path, synthetic):
    ms1, ms2 = _scans(synthetic, precursor_intensity=5e5)
    path = write_mzml(tmp_path / "synthetic.mzML", [ms1, ms2])
    gz = tmp_path / "synthetic.mzML.gz"
    result = spx.write_indexed_mzml_gzip(path, gz)
    assert result.spectrum_count == 2
    assert mzmlpy.is_embedded_indexed_gzip(gz)
    with spx.MzmlReader(gz) as reader:
        assert reader.access_strategy == "embedded"
        read = next(iter(reader.ms2))
    np.testing.assert_array_equal(read.mz, ms2.mz)
    np.testing.assert_array_equal(read.intensity, ms2.intensity)


def test_precursor_without_intensity_is_kept(tmp_path, synthetic):
    ms1, ms2 = _scans(synthetic, precursor_intensity=None)
    path = write_mzml(tmp_path / "no_precursor_intensity.mzML", [ms1, ms2])
    with mzmlpy.Mzml(str(path)) as reader:  # mzmlpy itself exposes the precursor fine
        assert list(reader.spectra)[1].precursors[0].selected_ions[0].mz is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with spx.MzmlReader(path) as reader:
            read = next(iter(reader.ms2))
    assert read.precursors is not None
    assert read.precursors[0].precursor_mz == pytest.approx(ms2.precursor_mz, abs=1e-9)


def test_mzmlpy_validates_spec_conformant_file(tmp_path, synthetic):
    ms1, ms2 = _scans(synthetic, precursor_intensity=5e5)
    path = write_mzml(tmp_path / "synthetic.mzML", [ms1, ms2])
    report = mzmlpy.validate(str(path), decode_binary=True)
    assert report.valid, report.issues


def test_mzmlpy_validates_its_own_example(tmp_path):
    source = PACKAGES / "mzmlpy" / "tests" / "data" / "example.mzML"
    if not source.exists():
        pytest.skip(f"{source} not present")
    copy = tmp_path / "example.mzML"  # never read in place: other sessions work in packages/
    shutil.copyfile(source, copy)
    report = mzmlpy.validate(str(copy), decode_binary=True)
    assert report.valid, report.issues


# --------------------------------------------------------------------------------------
# An indexedmzML whose offset index names fewer spectra than the document holds. The
# document is the authority and the index is a cache of it, so every lookup must answer
# from the document; and the shortness must be reported rather than silently repaired.
#
# Two shapes, and only one of them is hard. `counted` leaves spectrumList@count at the true
# number, which the count cross-check has always caught: the index is thrown away and every
# answer is right for a reason that has nothing to do with the fix. `short` is what a writer
# that stopped early produces - the index, indexList@count and spectrumList@count all agree
# on the short number, and only the document disagrees - so the count cross-check passes and
# the index is believed. Measured on the two shapes with mzmlpy 0.10.2 (no fix) and the
# fixed checkout, which both report version 0.10.2: `counted` gives identical answers on
# both, while `short` gives len 3, ids 3, iteration 5, `in` False and a raise from both
# lookups without the fix, against all five and no raise with it. Keeping both shapes is
# what stops declared_count being quietly dropped again.
# --------------------------------------------------------------------------------------

INDEXED_SCANS = 5
INDEX_ENTRIES = 3


def _codes(path):
    """``mzmlpy.validate`` issue codes, sorted: the order is not part of the contract."""
    return sorted(issue.code for issue in mzmlpy.validate(str(path), decode_binary=True).issues)


@pytest.fixture
def indexed_pair(tmp_path):
    scans = [
        SyntheticScan(
            mz=np.array([100.0 + i, 200.0 + i, 300.0 + i]),
            intensity=np.array([10.0 + i, 20.0 + i, 30.0 + i]),
            ms_level=1 if i == 0 else 2,
            rt_seconds=60.0 + i,
            precursor_mz=None if i == 0 else 500.0 + i,
            precursor_charge=None if i == 0 else 2,
        )
        for i in range(INDEXED_SCANS)
    ]
    full = write_indexed_mzml(tmp_path / "full.mzML", scans)
    short = write_indexed_mzml(
        tmp_path / "short.mzML", scans, index_entries=INDEX_ENTRIES, declared_count=INDEX_ENTRIES
    )
    counted = write_indexed_mzml(tmp_path / "counted.mzML", scans, index_entries=INDEX_ENTRIES)
    return scans, full, short, counted


def _answers(path):
    """Every way of asking the reader how many spectra there are, and for the last one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the reader warns that it is rebuilding
        with mzmlpy.Mzml(str(path)) as reader:
            spectra = reader.spectra
            last_id = f"scan={INDEXED_SCANS}"
            return {
                "len": len(spectra),
                "ids": len(spectra.ids),
                "iterated": sum(1 for _ in spectra),
                "contains_last": last_id in spectra,
                "by_id_mz": np.asarray(spectra[last_id].mz).tolist(),
                "by_position_mz": np.asarray(spectra[INDEXED_SCANS - 1].mz).tolist(),
            }


def test_a_short_offset_index_changes_no_answer(indexed_pair):
    """``len``, ``ids``, ``in``, iteration, id lookup and position lookup must all agree.

    An index that names three of five spectra was believed by the lookups while iteration
    streamed the document, so the two halves of the reader answered the same file
    differently and the spectra past the index were reachable by no key. Asking the same
    six questions of both files is the check: the only difference between them is the
    index and the count that agrees with it, so any disagreement is the index being
    treated as authoritative. Without the fix the last spectrum is reachable by neither
    id nor position - both raise - which is the silent data loss stated as an exception.
    """
    scans, full, short, _ = indexed_pair
    assert _answers(short) == _answers(full)

    answers = _answers(short)
    assert answers["len"] == INDEXED_SCANS
    assert answers["len"] > INDEX_ENTRIES  # the file is genuinely short-indexed
    assert answers["by_id_mz"] == scans[-1].mz.tolist()
    assert answers["by_position_mz"] == scans[-1].mz.tolist()


def test_a_short_offset_index_is_reported_not_silently_repaired(indexed_pair):
    """Rebuilding from the document is the right answer; doing it quietly is not.

    The repair is what makes the lookups correct, which means no count and no peak
    comparison can detect that the file was malformed - this is the only assertion in the
    pair that can.
    """
    _, full, short, counted = indexed_pair
    assert _codes(full) == []
    assert _codes(short) == ["count_mismatch", "index_ids_mismatch"]
    assert _codes(counted) == ["index_ids_mismatch"]


def test_spxtacular_reads_past_a_short_offset_index(indexed_pair):
    scans, _, short, _ = indexed_pair
    expected = [scan for scan in scans if scan.ms_level == 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with spx.MzmlReader(short) as reader:
            read = list(reader.ms2)
    assert len(read) == len(expected)
    np.testing.assert_array_equal(read[-1].mz, expected[-1].mz)
