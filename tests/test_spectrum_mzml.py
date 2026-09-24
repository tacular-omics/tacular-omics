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
from conftest import PACKAGES, SyntheticScan, write_mzml

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
