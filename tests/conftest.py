"""Shared fixtures for the end-to-end tests across the member packages.

In the tacular-omics workspace these tests run against the local sibling checkouts in
``packages/`` through the shared ``.venv``; standalone (CI) they run against the
installed releases. A few tests read example data from sibling checkouts
(``PACKAGES``) and skip when it is absent. No test writes outside pytest's
``tmp_path``.
"""

from __future__ import annotations

import base64
import os
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psimodpy
import pytest
import unimodpy

# Sibling checkouts: set TACULAR_OMICS_PACKAGES, else the workspace layout
# (<workspace>/packages/tacular-omics/tests/conftest.py).
PACKAGES = Path(os.environ.get("TACULAR_OMICS_PACKAGES") or Path(__file__).resolve().parents[2])

# A heavy base peptide so that large negative mod deltas never push the total mass
# below zero (peptacular rejects negative masses).
HEAVY_BASE = "W" * 20 + "K"


@pytest.fixture(scope="session")
def unimod_db() -> unimodpy.UnimodDatabase:
    return unimodpy.load()


@pytest.fixture(scope="session")
def psimod_db() -> psimodpy.PsiModDatabase:
    return psimodpy.load()


# --------------------------------------------------------------------------------------
# Minimal mzML writer. mzmlpy reads mzML but does not write spectra from arrays, so the
# suite writes a small spec-conformant mzML 1.1 file itself and reads it back through
# mzmlpy and spxtacular.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticScan:
    mz: np.ndarray
    intensity: np.ndarray
    ms_level: int = 2
    rt_seconds: float = 60.0
    precursor_mz: float | None = None
    precursor_charge: int | None = None
    precursor_intensity: float | None = None


def _binary_array(values: np.ndarray, accession: str, name: str, compress: bool) -> str:
    raw = np.asarray(values, dtype="<f8").tobytes()
    if compress:
        raw = zlib.compress(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    compression = (
        '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>'
        if compress
        else '<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>'
    )
    return (
        f'<binaryDataArray encodedLength="{len(encoded)}">'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
        f"{compression}"
        f'<cvParam cvRef="MS" accession="{accession}" name="{name}"/>'
        f"<binary>{encoded}</binary></binaryDataArray>"
    )


def _precursor_xml(scan: SyntheticScan) -> str:
    if scan.ms_level < 2 or scan.precursor_mz is None:
        return ""
    params = [
        (
            f'<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="{scan.precursor_mz!r}" '
            'unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
        )
    ]
    if scan.precursor_charge is not None:
        params.append(
            f'<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="{scan.precursor_charge}"/>'
        )
    if scan.precursor_intensity is not None:
        params.append(
            f'<cvParam cvRef="MS" accession="MS:1000042" name="peak intensity" value="{scan.precursor_intensity!r}" '
            'unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>'
        )
    return (
        '<precursorList count="1"><precursor>'
        f'<selectedIonList count="1"><selectedIon>{"".join(params)}</selectedIon></selectedIonList>'
        '<activation><cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation"/></activation>'
        "</precursor></precursorList>"
    )


def write_mzml(path: Path, scans: Sequence[SyntheticScan], compress: bool = True) -> Path:
    spectra = []
    for index, scan in enumerate(scans):
        kind = ("MS:1000579", "MS1 spectrum") if scan.ms_level == 1 else ("MS:1000580", "MSn spectrum")
        spectra.append(
            f'<spectrum index="{index}" id="scan={index + 1}" defaultArrayLength="{len(scan.mz)}">'
            f'<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{scan.ms_level}"/>'
            f'<cvParam cvRef="MS" accession="{kind[0]}" name="{kind[1]}"/>'
            '<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum"/>'
            '<cvParam cvRef="MS" accession="MS:1000130" name="positive scan"/>'
            '<scanList count="1"><cvParam cvRef="MS" accession="MS:1000795" name="no combination"/><scan>'
            f'<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{scan.rt_seconds!r}" '
            'unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>'
            "</scan></scanList>"
            f"{_precursor_xml(scan)}"
            '<binaryDataArrayList count="2">'
            f"{_binary_array(scan.mz, 'MS:1000514', 'm/z array', compress)}"
            f"{_binary_array(scan.intensity, 'MS:1000515', 'intensity array', compress)}"
            "</binaryDataArrayList></spectrum>"
        )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="tacular_integration">'
        '<cvList count="2">'
        '<cv id="MS" fullName="PSI-MS" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>'
        '<cv id="UO" fullName="Unit Ontology" URI="http://ontologies.berkeleybop.org/uo.obo"/>'
        "</cvList>"
        '<fileDescription><fileContent><cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum"/>'
        "</fileContent></fileDescription>"
        '<softwareList count="1"><software id="tacular_integration" version="0"/></softwareList>'
        '<instrumentConfigurationList count="1"><instrumentConfiguration id="IC1"/></instrumentConfigurationList>'
        '<dataProcessingList count="1"><dataProcessing id="dp">'
        '<processingMethod order="0" softwareRef="tacular_integration"/></dataProcessing></dataProcessingList>'
        '<run id="run1" defaultInstrumentConfigurationRef="IC1">'
        f'<spectrumList count="{len(spectra)}" defaultDataProcessingRef="dp">{"".join(spectra)}</spectrumList>'
        "</run></mzML>\n"
    )
    path.write_text(xml, encoding="utf-8")
    return path
