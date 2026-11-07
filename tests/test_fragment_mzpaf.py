"""Pipeline 2: peptacular fragments -> paftacular mzPAF annotation -> serialize -> parse
-> the same m/z. Also checks peptacular's own mzPAF serializer against paftacular."""

from __future__ import annotations

import paftacular as pft
import peptacular as pt
import pytest

# Da tolerance for m/z agreement. peptacular's fragment masses use the CV's listed mod
# mass while paftacular recomputes from composition, which differ by up to ~5e-7 Da.
MZ_TOL = 1e-5

PEPTIDES = [
    "PEPTIDEK",
    "PEPT[Phospho]IDEK",
    "[Acetyl]-PEPC[Carbamidomethyl]IDEK",
    "EM[UNIMOD:35]EVTS[MOD:00046]K",
    "PEPTIDE[+15.9949]K",
    pytest.param(
        "PEPTIDEK[Formula:[13C6]C-6]",
        id="13C6-label",
    ),
    "SAMPLER/2",
]

FRAGMENT_KW = {
    "ion_types": ["a", "b", "c", "x", "y", "z"],
    "charges": [1, 2],
    "neutral_deltas": ["H2O"],
    "isotopes": [0, 1],
}


def fragments_of(proforma: str):
    frags = list(pt.parse(proforma).fragment(**FRAGMENT_KW))
    assert frags
    return frags


@pytest.mark.parametrize("proforma", PEPTIDES)
@pytest.mark.parametrize("include_sequence", [True, False])
def test_paftacular_to_mzpaf_roundtrip_mz(proforma, include_sequence):
    for frag in fragments_of(proforma):
        text = pft.to_mzpaf(frag, include_sequence=include_sequence).serialize()
        parsed = pft.parse(text)
        # serialize is idempotent on the parsed form
        assert parsed.serialize() == text
        # the peptide-free label needs the analyte to recover a mass
        if not include_sequence:
            parsed = parsed.resolve(proforma.split("/")[0])
        assert parsed.mz() == pytest.approx(frag.mz, abs=MZ_TOL), text


@pytest.mark.parametrize("proforma", PEPTIDES)
def test_peptacular_mzpaf_string_parses_to_same_mz(proforma):
    for frag in fragments_of(proforma):
        text = frag.to_mzpaf()
        parsed = pft.parse(text)
        assert parsed.mz() == pytest.approx(frag.mz, abs=MZ_TOL), text


@pytest.mark.parametrize("proforma", PEPTIDES)
def test_both_serializers_describe_the_same_ion(proforma):
    """peptacular.Fragment.to_mzpaf and paftacular.to_mzpaf agree on ion, charge, losses.

    They differ only in isotope spelling ('+i' vs '+i13C'), which mzPAF treats as the
    same 13C isotope peak, so compare the parsed fields instead of the strings.
    """
    for frag in fragments_of(proforma):
        a = pft.parse(frag.to_mzpaf())
        b = pft.to_mzpaf(frag)
        assert a.ion_type == b.ion_type
        assert a.charge == b.charge
        assert a.neutral_losses == b.neutral_losses
        assert a.mz() == pytest.approx(b.mz(), abs=MZ_TOL)


def test_mzpaf_multi_annotation_roundtrip():
    frags = fragments_of("PEPT[Phospho]IDEK")[:25]
    joined = ",".join(pft.to_mzpaf(f).serialize() for f in frags)
    parsed = pft.parse_multi(joined)
    assert len(parsed) == len(frags)
    for ann, frag in zip(parsed, frags, strict=True):
        assert ann.mz() == pytest.approx(frag.mz, abs=MZ_TOL)


# --------------------------------------------------------------------------------------
# An unlocalized modification. peptacular refuses to fragment such an analyte at all, so
# it is deliberately absent from PEPTIDES above and the two packages cannot be compared
# fragment for fragment here. What can be compared: the mass paftacular resolves must be
# the one peptacular gives for the same slice with the modification removed, because an
# unlocalized modification is assigned to no residue and no fragment can claim it.
# --------------------------------------------------------------------------------------

UNLOCALIZED_ANALYTE = "[Phospho]?PEPTIDE"
BARE_SEQUENCE = "PEPTIDE"


def _peptacular_mz(label: str) -> float:
    """The m/z peptacular gives for the ion `label` names, on the bare sequence.

    Matched on the parsed mzPAF fields rather than the label text, because peptacular
    writes the embedded sequence into its own labels (``b2{PE}``) and there is no one
    spelling to compare against.
    """
    wanted = pft.parse(label)
    hits = []
    for frag in pt.parse(BARE_SEQUENCE).fragment(ion_types=["b", "y"], charges=[1, 2]):
        got = pft.parse(frag.to_mzpaf())
        # ion_type is a PeptideIon; peptacular embeds the slice sequence in it and the
        # bare label does not, so compare the series and position rather than the whole.
        if (got.ion_type.series, got.ion_type.position, got.charge) == (
            wanted.ion_type.series,
            wanted.ion_type.position,
            wanted.charge,
        ):
            hits.append(frag.mz)
    assert len(hits) == 1, f"{label} matched {len(hits)} peptacular fragments"
    return hits[0]


def test_peptacular_will_not_fragment_an_unlocalized_modification():
    """Why this analyte is not in PEPTIDES: peptacular declines, rather than guessing."""
    with pytest.raises(pt.PeptacularError, match="unknown modifications"):
        list(pt.parse(UNLOCALIZED_ANALYTE).fragment(**FRAGMENT_KW))


@pytest.mark.parametrize("label", ["b2", "y2", "b2^2", "y3"])
def test_a_fragment_does_not_claim_an_unlocalized_modification(label):
    """The resolved m/z is the bare slice's, and the warning says why.

    Carrying the analyte's unlocalized modification into every slice is right for a
    sequence slice and wrong for a fragment: it adds the whole modification mass to every
    ion and publishes one theoretical m/z as if the position were known.
    """
    with pytest.warns(UserWarning, match="unlocalized modification"):
        resolved = pft.parse(label).resolve(UNLOCALIZED_ANALYTE)

    expected = _peptacular_mz(label)
    assert resolved.mz() == pytest.approx(expected, abs=MZ_TOL)

    # and it is not the value that claiming the modification would give
    charge = pft.parse(label).charge
    phospho = pt.parse("PEPT[Phospho]IDE").mass() - pt.parse(BARE_SEQUENCE).mass()
    assert phospho > 79.0
    claimed = expected + phospho / charge
    assert abs(resolved.mz() - claimed) > phospho / charge / 2
