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
def dda_open():
    """The reader kept open: ``ook0_range`` and friends are read lazily through it.

    ``tdf_first`` closes its reader, so the elements it returns raise
    ``ReaderClosedError`` for anything not already materialised.
    """
    try:
        with tdfpy.DDA(str(DDA_PATH)) as dda:
            yield dda
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


def test_mobility_range_endpoints_are_scans_the_window_contains(dda_open):
    """Every ``ook0_range`` endpoint must be a value the window's own scans produce.

    A scan range is ``[scan_num_begin, scan_num_end)``, so converting the mobility at
    ``scan_num_end`` reports the first scan of the *next* window and widens the range by
    one scan. Containment cannot catch that - a range one scan too wide still contains
    every peak - so this asserts the endpoints equal the extremes over the scans in the
    window, and that the scan past the end lies strictly outside. The second half is the
    one that fails when the conversion runs at the exclusive end.
    """
    precursor = next(iter(dda_open.precursors))
    windows = precursor.pasef_frame_msms_infos
    assert windows

    for window in windows:
        begin, end = window.scan_num_begin, window.scan_num_end
        assert begin < end
        scans = np.asarray(window.timsdata.scan_num_to_ook0(window.frame_id, list(range(begin, end))), dtype=float)
        low, high = window.ook0_range
        assert low == pytest.approx(float(scans.min()))
        assert high == pytest.approx(float(scans.max()))

        past_end = float(window.timsdata.scan_num_to_ook0(window.frame_id, [end])[0])
        assert not low <= past_end <= high, (
            f"frame {window.frame_id} window [{begin}, {end}): the mobility at the exclusive end "
            f"scan {end} is {past_end}, inside the reported range {window.ook0_range}"
        )

    lows = [w.ook0_range[0] for w in windows]
    highs = [w.ook0_range[1] for w in windows]
    assert precursor.ook0_range == (pytest.approx(min(lows)), pytest.approx(max(highs)))


def test_spxtacular_precursor_mobility_agrees_with_tdfpy(dda_open, dreader_first):
    """spxtacular's precursor mobility is tdfpy's, in tdfpy's units, inside tdfpy's range."""
    precursor = next(iter(dda_open.precursors))
    _, ms2 = dreader_first
    spx_precursor = ms2.precursors[0]

    assert spx_precursor.im_type == "ook0"
    assert spx_precursor.im == pytest.approx(precursor.ook0)
    low, high = precursor.ook0_range
    assert low <= spx_precursor.im <= high
