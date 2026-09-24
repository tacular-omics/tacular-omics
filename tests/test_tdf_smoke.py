"""Pipeline 6: tdfpy smoke test. A timsTOF DDA frame read with tdfpy, centroided, and
turned into a spxtacular spectrum must agree with spxtacular's own DReader.

Skips cleanly when tdfpy (or its native reader) is unavailable or the example data set
is not present in the local checkout. The data set is only read, never written.
"""

from __future__ import annotations

import numpy as np
import pytest
import spxtacular as spx
from conftest import PACKAGES

tdfpy = pytest.importorskip("tdfpy")

DDA_PATH = PACKAGES / "tdfpy" / "tests" / "data" / "example_dda.d"
if not (DDA_PATH / "analysis.tdf").exists():
    pytest.skip(f"tdfpy example data not found at {DDA_PATH}", allow_module_level=True)


@pytest.fixture(scope="module")
def tdf_first():
    try:
        with tdfpy.DDA(str(DDA_PATH)) as dda:
            frame = next(iter(dda.ms1))
            centroid = np.asarray(frame.centroid())
            precursor = next(iter(dda.precursors))
            return centroid, np.asarray(precursor.merged_peaks()), precursor
    except OSError as exc:  # native timsdata library missing on this platform
        pytest.skip(f"tdfpy cannot open the example data: {exc}")


@pytest.fixture(scope="module")
def dreader_first():
    with spx.DReader(DDA_PATH) as reader:
        return next(iter(reader.ms1)), next(iter(reader.ms2))


def test_tdfpy_centroid_to_spxtacular_spectrum(tdf_first):
    centroid, _, _ = tdf_first
    assert centroid.ndim == 2 and centroid.shape[1] == 3 and len(centroid) > 0
    spectrum = spx.Spectrum(mz=centroid[:, 0], intensity=centroid[:, 1], im=centroid[:, 2])
    assert len(spectrum.mz) == len(centroid)
    np.testing.assert_array_equal(np.sort(spectrum.mz), np.sort(centroid[:, 0]))
    assert np.all(spectrum.intensity > 0)


def test_dreader_ms1_matches_tdfpy_centroid(tdf_first, dreader_first):
    centroid, _, _ = tdf_first
    ms1, _ = dreader_first
    assert len(ms1.mz) == len(centroid)
    np.testing.assert_allclose(np.sort(ms1.mz), np.sort(centroid[:, 0]))
    np.testing.assert_allclose(np.sort(ms1.intensity), np.sort(centroid[:, 1]))


def test_dreader_ms2_matches_tdfpy_precursor(tdf_first, dreader_first):
    _, peaks, precursor = tdf_first
    _, ms2 = dreader_first
    assert len(ms2.mz) == len(peaks)
    np.testing.assert_allclose(np.sort(ms2.mz), np.sort(peaks[:, 0]))
    assert ms2.precursors
    assert ms2.precursors[0].charge == precursor.charge
