"""Cross-package invariants for charge sign and charge carriers.

peptacular computes fragment and precursor m/z, paftacular recomputes them from mzPAF, and
spxtacular matches them to peaks and converts m/z to neutral mass through its ionization
models. A sign error or a carrier mismatch at any of these boundaries shows up as a
disagreement between two packages, so every check below compares two of them:

0. Anchor: precursor m/z of PEPTIDE from all three packages equals literals computed by
   hand from CODATA/AME constants typed into this file, so a shared wrong constant or a
   sign error common to all three packages still fails.
1. peptacular fragment m/z == paftacular m/z of the mzPAF annotation for that fragment
   (both serializers, through the mzPAF string), and the mzPAF charge keeps the sign.
   ProForma ``set_charge(fragment.charge_adducts)`` survives ``parse(serialize())``.
2. spxtacular ``match_fragments`` puts each fragment on its own peak and, after
   ``decharge`` with the matching ionization model, finds it again at its peptacular
   neutral mass. The ``{(IonType, signed charge): [m/z, ...]}`` input builds fragments
   that keep the sign and give peptacular's neutral mass.
3. Precursor m/z agrees between peptacular, paftacular and the spxtacular ionization
   model. ``write_ms2`` writes the signed charge and MH+ on the ``Z`` line, and
   ``Ms2Reader`` reads back the charge and the ``S``-line precursor m/z (the reader
   ignores the ``Z``-line mass, so this is not a full round trip of that line).

Grid: every ``tacular.IonType`` except ``n`` (a neutral molecule, which mzPAF cannot
write) x charges -3..+3 except 0 (protons; negative charges are deprotonation) plus the
carriers Na+, 2 Na+, NH4+, H- (hydride), Cl- and a mixed Na+/H+ adduct x four peptides.
peptacular has no electron carrier, so electron (radical) ions are checked as paftacular
m/z against this file's own arithmetic (peptacular neutral mass and the electron mass).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import paftacular as pft
import peptacular as pt
import pytest
import spxtacular as spx
from tacular import IonType
from tacular.constants import ELECTRON_MASS, PROTON_MASS

# Every m/z here is computed from the same masses on both sides, so agreement is exact
# to rounding; 1e-6 leaves room for summation order only.
MZ_TOL = 1e-6

PEPTIDES = {
    # C-terminal K, unmodified, has T/I for the threonine/isoleucine satellite ions.
    "PEPTIDEK": "PEPTIDEK",
    # C-terminal R, modified residues (d/w/v ions on and next to Oxidation/Phospho),
    # and T, I, V for every satellite-ion variant.
    "mods-R": "EM[Oxidation]S[Phospho]TIVDER",
    # Selenocysteine.
    "U": "AUCDK",
    # N-terminal modification, which a/b/c, d and immonium ions must carry or drop.
    "nterm-acetyl": "[Acetyl]-PEITVK",
}

ION_TYPES = [t for t in IonType if t is not IonType.NEUTRAL]

# Ion types paftacular.resolve() supports, for the peptide-free mzPAF round trip.
RESOLVABLE = {
    IonType.A,
    IonType.B,
    IonType.C,
    IonType.X,
    IonType.Y,
    IonType.Z,
    IonType.PRECURSOR,
    IonType.BY,
    IonType.AX,
    IonType.CZ,
    IonType.AY,
    IonType.AZ,
    IonType.BX,
    IonType.BZ,
    IonType.CX,
    IonType.CY,
}


@dataclass(frozen=True)
class Carrier:
    """A peptacular charge argument and the spxtacular ionization model for it."""

    charge: int | str | list[str]
    sign: int
    magnitude: int
    model: str | None  # spxtacular IonizationModelLike, None when spxtacular has none
    no_model_reason: str = ""


CARRIERS = {
    "+1": Carrier(1, 1, 1, "protonated"),
    "+2": Carrier(2, 1, 2, "protonated"),
    "+3": Carrier(3, 1, 3, "protonated"),
    "-1": Carrier(-1, -1, 1, "deprotonated"),
    "-2": Carrier(-2, -1, 2, "deprotonated"),
    "-3": Carrier(-3, -1, 3, "deprotonated"),
    "Na+": Carrier("Na:z+1", 1, 1, "Na:z+1"),
    "2Na+": Carrier("Na:z+1^2", 1, 2, "Na:z+1"),
    "NH4+": Carrier("NH4:z+1", 1, 1, "NH4:z+1"),
    "hydride-": Carrier("H:z-1", -1, 1, "H:z-1"),
    "Cl-": Carrier("Cl:z-1", -1, 1, "Cl:z-1"),
    "Na+H+": Carrier(
        ["Na:z+1", "H:z+1"],
        1,
        2,
        None,
        "spxtacular IonizationModel is one repeated carrier by design; mixed adducts are rejected",
    ),
}

# Fragment variants: plain ions, and the first 13C isotope peak with and without neutral
# losses (the monoisotopic peak is already in "plain").
VARIANTS = {
    "plain": {},
    "losses-13C": {"neutral_deltas": ["H2O", "H3PO4"], "isotopes": [1]},
}

PEPTIDE_IDS = list(PEPTIDES)
CARRIER_IDS = list(CARRIERS)

# mzPAF has no notation for a hydride (H-) carrier: [M+H] is H+, and paftacular rejects a
# charge that disagrees with its carriers (mzPAF 4.7). These carriers skip the mzPAF checks.
NO_MZPAF = {"hydride-": "mzPAF cannot express a hydride (H-) carrier"}

# Fragment grid: plain ions for every peptide; the loss/isotope variant on the modified
# peptide only (it has the phospho loss and keeps the default run under ~20 s).
FRAGMENT_CASES = [
    pytest.param(variant, peptide_id, carrier_id, id=f"{variant}-{peptide_id}-{carrier_id}")
    for variant, peptide_ids in (("plain", PEPTIDE_IDS), ("losses-13C", ["mods-R"]))
    for peptide_id in peptide_ids
    for carrier_id in CARRIER_IDS
]


def _fragments(peptide: str, carrier: Carrier, ion_types=ION_TYPES, **kwargs) -> list[pt.Fragment]:
    frags: list[pt.Fragment] = []
    for ion_type in ion_types:
        frags.extend(pt.fragment(peptide, ion_types=[ion_type], charges=[carrier.charge], **kwargs))
    assert frags
    return frags


def _label(f: pt.Fragment) -> str:
    return f"{f.ion_type}{f.position}^{f.charge_state} ({f.parent_sequence})"


# --------------------------------------------------------------------------------------
# 0. Anchor against hand-typed constants
# --------------------------------------------------------------------------------------

# Typed by hand from CODATA 2018 / AME2020 on purpose: importing tacular.constants here
# would let a wrong shared constant pass every comparison between packages.
_H = 1.00782503207
_C = 12.0
_N = 14.0030740048
_O = 15.99491461956
_NA = 22.9897692820
_CL = 34.968852682
_PROTON = 1.007276466621
_ELECTRON = 0.000548579909065

# PEPTIDE is C34H53N7O15.
ANCHOR_PEPTIDE = "PEPTIDE"
ANCHOR_NEUTRAL = 34 * _C + 53 * _H + 7 * _N + 15 * _O

# carrier id -> (peptacular charge, spxtacular model, signed z, ion mass from the constants,
# reviewed literal m/z). The literal is checked against the arithmetic so a typo in either fails.
ANCHORS = {
    "+1": (1, "protonated", 1, ANCHOR_NEUTRAL + _PROTON, 800.3672405),
    "+2": (2, "protonated", 2, ANCHOR_NEUTRAL + 2 * _PROTON, 400.6872585),
    "-1": (-1, "deprotonated", -1, ANCHOR_NEUTRAL - _PROTON, 798.3526876),
    "-2": (-2, "deprotonated", -2, ANCHOR_NEUTRAL - 2 * _PROTON, 398.6727055),
    "Na+": ("Na:z+1", "Na:z+1", 1, ANCHOR_NEUTRAL + _NA - _ELECTRON, 822.3491847),
    "Cl-": ("Cl:z-1", "Cl:z-1", -1, ANCHOR_NEUTRAL + _CL + _ELECTRON, 834.3293653),
    "hydride-": ("H:z-1", "H:z-1", -1, ANCHOR_NEUTRAL + _H + _ELECTRON, 800.3683376),
    "NH4+": ("NH4:z+1", "NH4:z+1", 1, ANCHOR_NEUTRAL + _N + 4 * _H - _ELECTRON, 817.3937896),
}


@pytest.mark.parametrize("carrier_id", list(ANCHORS))
def test_precursor_mz_matches_hand_computed_constants(carrier_id):
    charge, model_name, signed, ion_mass, literal = ANCHORS[carrier_id]
    magnitude = abs(signed)
    assert ANCHOR_NEUTRAL == pytest.approx(799.3599640, abs=MZ_TOL)
    assert ion_mass / magnitude == pytest.approx(literal, abs=MZ_TOL)

    assert pt.parse(ANCHOR_PEPTIDE).neutral_mass() == pytest.approx(ANCHOR_NEUTRAL, abs=MZ_TOL)
    assert pt.mz(ANCHOR_PEPTIDE, charge=charge) == pytest.approx(literal, abs=MZ_TOL), "peptacular"

    (frag,) = pt.fragment(ANCHOR_PEPTIDE, ion_types=["p"], charges=[charge])
    assert frag.charge_state == signed
    if carrier_id not in NO_MZPAF:
        ann = pft.parse(pft.to_mzpaf(frag).serialize()).resolve(ANCHOR_PEPTIDE)
        assert ann.charge == signed
        assert ann.mz() == pytest.approx(literal, abs=MZ_TOL), f"paftacular {ann.serialize()}"

    model = spx.resolve_ionization_model(model_name)
    assert model.polarity == ("positive" if signed > 0 else "negative")
    assert model.ion_mz(ANCHOR_NEUTRAL, magnitude) == pytest.approx(literal, abs=MZ_TOL), "spxtacular"


# --------------------------------------------------------------------------------------
# 1. peptacular fragment m/z == paftacular m/z of its mzPAF annotation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("variant", "peptide_id", "carrier_id"), FRAGMENT_CASES)
def test_fragment_mz_survives_mzpaf_round_trip(variant, peptide_id, carrier_id):
    if carrier_id in NO_MZPAF:
        pytest.skip(NO_MZPAF[carrier_id])
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    failures = []
    for f in _fragments(peptide, carrier, **VARIANTS[variant]):
        assert f.charge_state == carrier.sign * carrier.magnitude, _label(f)
        texts = {
            "paftacular.to_mzpaf": pft.to_mzpaf(f).serialize(),
            "Fragment.to_mzpaf": f.to_mzpaf(),
        }
        # resolve() is the slow path; the plain ions cover it.
        if variant == "plain" and f.ion_type in RESOLVABLE:
            texts["no-sequence+resolve"] = pft.to_mzpaf(f, include_sequence=False).serialize()
        for source, text in texts.items():
            ann = pft.parse(text)
            # The peptide-free label, and a precursor (whose sequence is resolved
            # context, not written), need the analyte to give a complete mass.
            if source == "no-sequence+resolve" or f.ion_type is IonType.PRECURSOR:
                ann = ann.resolve(peptide)
            got = ann.mz()
            if abs(got - f.mz) > MZ_TOL or ann.charge != f.charge_state:
                failures.append(
                    f"{_label(f)} via {source} {text!r}: mzPAF m/z {got} charge {ann.charge}, "
                    f"peptacular m/z {f.mz} charge {f.charge_state}"
                )
    assert not failures, "\n".join(failures[:20])


def _peptacular_48_open() -> bool:
    """Released peptacular serializes a negative proton charge as H:z+1^-n, which its own
    ProForma parser rejects; the fix is in peptacular #48. Probe instead of pinning a
    version so the strict xfail holds against PyPI and the fixed local checkout alike."""
    (frag,) = pt.fragment("PEPTIDE", ion_types=["p"], charges=[-1])
    try:
        pt.parse(pt.parse("PEPTIDE").set_charge(frag.charge_adducts).serialize())
    except Exception:
        return True
    return False


_SET_CHARGE_XFAIL = {
    cid: pytest.mark.xfail(_peptacular_48_open(), strict=True, reason="peptacular #48")
    for cid, c in CARRIERS.items()
    if isinstance(c.charge, int) and c.sign < 0
}


@pytest.mark.parametrize(
    "carrier_id",
    [pytest.param(cid, marks=_SET_CHARGE_XFAIL.get(cid, ())) for cid in CARRIER_IDS],
)
@pytest.mark.parametrize("peptide_id", PEPTIDE_IDS)
def test_set_charge_adducts_survives_proforma_round_trip(peptide_id, carrier_id):
    """``parse(serialize())`` of ``set_charge(fragment.charge_adducts)`` keeps charge and m/z."""
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    (frag,) = pt.fragment(peptide, ion_types=["p"], charges=[carrier.charge])
    text = pt.parse(peptide).set_charge(frag.charge_adducts).serialize()
    back = pt.parse(text)
    assert back.serialize() == text
    assert back.charge_state == carrier.sign * carrier.magnitude, text
    assert pt.mz(back) == pytest.approx(frag.mz, abs=MZ_TOL), text


# --------------------------------------------------------------------------------------
# 2. spxtacular matching keeps each fragment on its own peak with the right sign
# --------------------------------------------------------------------------------------


def _synthetic_spectrum(frags: list[pt.Fragment], magnitude: int) -> spx.Spectrum:
    mz = np.unique(np.array([f.mz for f in frags], dtype=np.float64))
    return spx.Spectrum(
        mz,
        np.full(mz.size, 100.0),
        charge=np.full(mz.size, magnitude, dtype=np.int32),
        spectrum_type="deconvoluted",
    )


@pytest.mark.parametrize(("variant", "peptide_id", "carrier_id"), FRAGMENT_CASES)
def test_match_fragments_finds_each_fragment_on_its_own_peak(variant, peptide_id, carrier_id):
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    frags = _fragments(peptide, carrier, **VARIANTS[variant])
    spectrum = _synthetic_spectrum(frags, carrier.magnitude)

    matches = spx.match_fragments(spectrum, frags, tolerance=MZ_TOL, peak_selection="all")
    by_fragment: dict[int, list[spx.MatchedFragment]] = {}
    for m in matches:
        by_fragment.setdefault(id(m.fragment), []).append(m)

    failures = []
    for f in frags:
        own = [m for m in by_fragment.get(id(f), []) if abs(m.peak_mz - f.mz) <= MZ_TOL]
        if not own:
            failures.append(f"{_label(f)} m/z {f.mz}: not matched to its own peak")
            continue
        # m.fragment is the input object here, so its charge is not re-checked; the
        # dict-input test below covers fragments that match_fragments builds itself.
        for m in own:
            if int(spectrum.charge[m.peak_index]) != abs(m.fragment.charge_state):
                failures.append(f"{_label(f)}: peak charge {spectrum.charge[m.peak_index]}")
    assert not failures, "\n".join(failures[:20])


@pytest.mark.parametrize(("variant", "peptide_id", "carrier_id"), FRAGMENT_CASES)
def test_decharged_spectrum_matches_peptacular_neutral_masses(variant, peptide_id, carrier_id):
    """decharge() with the carrier's ionization model gives back Fragment.neutral_mass."""
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    if carrier.model is None:
        pytest.skip(carrier.no_model_reason)
    model = spx.resolve_ionization_model(carrier.model)
    assert model.polarity == ("positive" if carrier.sign > 0 else "negative")
    frags = _fragments(peptide, carrier, **VARIANTS[variant])
    decharged = _synthetic_spectrum(frags, carrier.magnitude).decharge(ionization_model=model)

    matched = {id(m.fragment) for m in spx.match_fragments(decharged, frags, tolerance=MZ_TOL, peak_selection="all")}
    missing = [f"{_label(f)} neutral {f.neutral_mass}" for f in frags if id(f) not in matched]
    assert not missing, "\n".join(missing[:20])

    # Negative control for protons: decharging with the opposite polarity is off by
    # two proton masses per charge, so the sign really is what the match depends on.
    if isinstance(carrier.charge, int):
        wrong = spx.resolve_ionization_model("deprotonated" if carrier.sign > 0 else "protonated")
        for f in frags:
            offset = wrong.neutral_mass(f.mz, carrier.magnitude) - f.neutral_mass
            assert abs(offset) == pytest.approx(2 * PROTON_MASS * carrier.magnitude, abs=MZ_TOL), _label(f)


DICT_ION_TYPES = [IonType.B, IonType.Y]


@pytest.mark.parametrize("decharged", [False, True], ids=["deconvoluted", "decharged"])
@pytest.mark.parametrize("carrier_id", ["+2", "-2"])
@pytest.mark.parametrize("peptide_id", PEPTIDE_IDS)
def test_match_fragments_dict_input_keeps_sign(peptide_id, carrier_id, decharged):
    """``{(IonType, signed charge): [m/z, ...]}`` input: match_fragments builds each Fragment.

    Each returned fragment sits on its peak (m/z, or neutral mass once decharged), keeps the
    signed charge of its key, and has peptacular's neutral mass.
    """
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    signed = carrier.sign * carrier.magnitude
    by_key: dict[tuple[IonType, int], list[pt.Fragment]] = {}
    for ion_type in DICT_ION_TYPES:
        frags = sorted(pt.fragment(peptide, ion_types=[ion_type], charges=[signed]), key=lambda f: f.position)
        by_key[(ion_type, signed)] = frags
    all_frags = [f for frags in by_key.values() for f in frags]
    spectrum = _synthetic_spectrum(all_frags, carrier.magnitude)
    if decharged:
        spectrum = spectrum.decharge(ionization_model=spx.resolve_ionization_model(carrier.model))

    query = {key: [f.mz for f in frags] for key, frags in by_key.items()}
    matches = spx.match_fragments(spectrum, query, tolerance=MZ_TOL, peak_selection="all")

    found: set[tuple[IonType, int]] = set()
    failures = []
    for m in matches:
        got = m.fragment
        # The dict list index is the position, starting at 1.
        ref = by_key[(IonType(got.ion_type), signed)][got.position - 1]
        label = _label(ref)
        target = got.neutral_mass if decharged else got.mz
        if abs(target - m.peak_mz) > MZ_TOL:
            continue  # a neighbour's peak within tolerance; only own-peak matches count
        found.add((IonType(got.ion_type), got.position))
        if got.charge_state != signed:
            failures.append(f"{label}: built with charge {got.charge_state}, key had {signed}")
        if abs(got.mz - ref.mz) > MZ_TOL:
            failures.append(f"{label}: m/z {got.mz} vs peptacular {ref.mz}")
        if abs(got.neutral_mass - ref.neutral_mass) > MZ_TOL:
            failures.append(f"{label}: neutral mass {got.neutral_mass} vs peptacular {ref.neutral_mass}")
    missing = [_label(f) for f in all_frags if (IonType(f.ion_type), f.position) not in found]
    assert not missing, "not matched to own peak: " + ", ".join(missing[:20])
    assert not failures, "\n".join(failures[:20])


# --------------------------------------------------------------------------------------
# 3. Precursor m/z, ionization models and the .ms2 Z line
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("carrier_id", CARRIER_IDS)
@pytest.mark.parametrize("peptide_id", PEPTIDE_IDS)
def test_precursor_mz_peptacular_vs_paftacular(peptide_id, carrier_id):
    if carrier_id in NO_MZPAF:
        pytest.skip(NO_MZPAF[carrier_id])
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    expected = pt.mz(peptide, charge=carrier.charge)
    (frag,) = pt.fragment(peptide, ion_types=["p"], charges=[carrier.charge])
    assert frag.mz == pytest.approx(expected, abs=MZ_TOL)
    ann = pft.parse(pft.to_mzpaf(frag).serialize()).resolve(peptide)
    assert ann.charge == carrier.sign * carrier.magnitude
    assert ann.mz() == pytest.approx(expected, abs=MZ_TOL), ann.serialize()


@pytest.mark.parametrize("carrier_id", CARRIER_IDS)
@pytest.mark.parametrize("peptide_id", PEPTIDE_IDS)
def test_precursor_mz_spxtacular_and_ms2_charge_line(peptide_id, carrier_id, tmp_path: Path):
    """Ionization model m/z <-> neutral mass, then the .ms2 ``Z`` line and what is read back.

    Checks the written ``Z`` line (signed charge, MH+) as text, and the charge plus the
    ``S``-line precursor m/z from ``Ms2Reader``. The reader ignores the ``Z``-line mass, so
    this is not a full round trip of that line.
    """
    peptide, carrier = PEPTIDES[peptide_id], CARRIERS[carrier_id]
    if carrier.model is None:
        pytest.skip(carrier.no_model_reason)
    model = spx.resolve_ionization_model(carrier.model)
    neutral = pt.parse(peptide).neutral_mass()
    precursor_mz = pt.mz(peptide, charge=carrier.charge)
    assert model.ion_mz(neutral, carrier.magnitude) == pytest.approx(precursor_mz, abs=MZ_TOL)
    assert model.neutral_mass(precursor_mz, carrier.magnitude) == pytest.approx(neutral, abs=MZ_TOL)

    signed = carrier.sign * carrier.magnitude
    spectrum = spx.MsnSpectrum(
        np.array([100.0, 200.0]),
        np.array([10.0, 20.0]),
        spectrum_type="centroid",
        ms_level=2,
        polarity="positive" if carrier.sign > 0 else "negative",
        precursors=[spx.Precursor(precursor_mz=precursor_mz, charge=signed)],
    )
    # Protons: also check the default model, which write_ms2 picks from the charge sign.
    explicit = [model] + ([None] if isinstance(carrier.charge, int) else [])
    for i, ionization_model in enumerate(explicit):
        path = spx.write_ms2(spectrum, tmp_path / f"run{i}.ms2", ionization_model=ionization_model)
        (z_line,) = [line.split() for line in path.read_text().splitlines() if line.startswith("Z")]
        assert int(z_line[1]) == signed
        assert float(z_line[2]) - PROTON_MASS == pytest.approx(neutral, abs=MZ_TOL), z_line

        read = next(iter(spx.Ms2Reader(path).ms2))
        (prec,) = read.precursors
        assert prec.charge == signed
        assert prec.precursor_mz == pytest.approx(precursor_mz, abs=MZ_TOL)
        assert model.neutral_mass(prec.precursor_mz, abs(prec.charge)) == pytest.approx(neutral, abs=MZ_TOL)


# --------------------------------------------------------------------------------------
# Electron carriers: paftacular [M-e] / [M+e] vs this test's own arithmetic.
# peptacular has no electron charge carrier and spxtacular has no built-in electron model,
# so the expected m/z is (peptacular neutral mass -/+ n electrons) / n, computed here.
# --------------------------------------------------------------------------------------

ELECTRON_IONS = [IonType.PRECURSOR, IonType.A, IonType.B, IonType.C, IonType.X, IonType.Y, IonType.Z]


@pytest.mark.parametrize("magnitude", [1, 2, 3])
@pytest.mark.parametrize("sign", [1, -1], ids=["e-loss", "e-gain"])
@pytest.mark.parametrize("peptide_id", PEPTIDE_IDS)
def test_electron_carrier_paftacular_vs_own_arithmetic(peptide_id, sign, magnitude):
    """paftacular m/z of an electron-carrier ion vs (neutral mass -/+ n e) / n from this test."""
    peptide = PEPTIDES[peptide_id]
    count = "" if magnitude == 1 else str(magnitude)
    adduct = f"{'-' if sign > 0 else '+'}{count}e"
    failures = []
    for f in _fragments(peptide, CARRIERS["+1"], ion_types=ELECTRON_IONS):
        # Replace the default proton of the +1 ion with the electron carrier.
        base = pft.to_mzpaf(f)
        assert not base.adducts, base.serialize()
        text = replace(base, adducts=(pft.Adduct.parse(adduct),), charge=sign * magnitude).serialize()
        ann = pft.parse(text)
        if f.ion_type is IonType.PRECURSOR:
            ann = ann.resolve(peptide)
        expected = (f.neutral_mass - sign * magnitude * ELECTRON_MASS) / magnitude
        if abs(ann.mz() - expected) > MZ_TOL or ann.charge != sign * magnitude:
            failures.append(f"{ann.serialize()}: paftacular {ann.mz()} vs expected {expected}")
    assert not failures, "\n".join(failures[:20])
