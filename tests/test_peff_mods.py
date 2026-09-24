"""Pipeline 4: PEFF ModRes annotations (pefftacular) resolved through psimodpy, unimodpy,
uniprotptmpy and tacular, turned into ProForma, then masses and digests from peptacular."""

from __future__ import annotations

import io

import pefftacular as pf
import peptacular as pt
import pytest
import tacular as tc
import uniprotptmpy

PEFF = """# PEFF 1.0
# //
# DbName=integration
# Prefix=tr
# DbVersion=1
# DbSource=local
# NumberOfEntries=1
# SequenceType=AA
# //
>tr:INT1 \\PName=Integration protein \\NcbiTaxId=9606 \\Length=16 \
\\ModResUnimod=(2|UNIMOD:35|Oxidation)(4|UNIMOD:4|Carbamidomethyl) \
\\ModResPsi=(5|MOD:00046|O-phospho-L-serine)(12|MOD:00047|O-phospho-L-threonine) \
\\ModRes=(9||N6-acetyllysine)
AMCCSPEPKIDTTEKR
"""

SEQUENCE = "AMCCSPEPKIDTTEKR"


@pytest.fixture(scope="module")
def entry():
    header, entries = pf.read_peff(io.StringIO(PEFF))
    assert len(entries) == 1
    return header, entries[0]


def _accession_number(accession: str) -> int:
    return int(accession.split(":")[1])


@pytest.fixture(scope="module")
def resolved(entry, unimod_db, psimod_db):
    """{1-based position: (proforma tag, mono delta)} from every ModRes flavour."""
    _, e = entry
    out: dict[int, tuple[str, float]] = {}
    for mod in e.mod_res_unimod:
        uid = _accession_number(mod.accession)
        u = unimod_db.get_by_id(uid)
        t = tc.UNIMOD_LOOKUP[uid]
        assert u.name == mod.name == t.name
        assert u.delta_mono_mass == pytest.approx(t.monoisotopic_mass, abs=1e-6)
        for pos in mod.positions:
            out[pos] = (mod.accession, u.delta_mono_mass)
    for mod in e.mod_res_psi:
        mid = _accession_number(mod.accession)
        m = psimod_db[mid]
        t = tc.PSIMOD_LOOKUP[mid]
        assert m.name == mod.name == t.name
        assert m.diff_mono == pytest.approx(t.monoisotopic_mass, abs=1e-6)
        for pos in mod.positions:
            out[pos] = (mod.accession, m.diff_mono)
    # generic ModRes carries only a name; UniProt PTM names are not a ProForma CV, so
    # map via the UniProt PTM cross-reference to PSI-MOD.
    uniprot = uniprotptmpy.load()
    for mod in e.mod_res:
        ptm = uniprot.get_by_name(mod.name)
        t = tc.UNIPROT_PTM_LOOKUP.query_name(mod.name)
        assert t is not None
        assert ptm.monoisotopic_mass == pytest.approx(t.monoisotopic_mass, abs=1e-6)
        psi = [x.accession for x in ptm.cross_references if x.database == "PSI-MOD"]
        assert psi == ["MOD:00064"]
        assert "PSI-MOD; MOD:00064." in t.cross_references
        psimod = t.get_psimod()
        assert psimod is not None and f"MOD:{psimod.id}" == psi[0]
        assert psimod.monoisotopic_mass == pytest.approx(ptm.monoisotopic_mass, abs=1e-6)
        for pos in mod.positions:
            out[pos] = (psi[0], ptm.monoisotopic_mass)
    return out


def _proforma(sequence: str, mods: dict[int, tuple[str, float]], offset: int = 0) -> str:
    parts = []
    for i, aa in enumerate(sequence, start=1 + offset):
        parts.append(f"{aa}[{mods[i][0]}]" if i in mods else aa)
    return "".join(parts)


def test_peff_parses_all_modres_flavours(entry):
    _, e = entry
    assert e.sequence == SEQUENCE
    assert [(m.positions, m.accession) for m in e.mod_res_unimod] == [
        ((2,), "UNIMOD:35"),
        ((4,), "UNIMOD:4"),
    ]
    assert [(m.positions, m.accession) for m in e.mod_res_psi] == [
        ((5,), "MOD:00046"),
        ((12,), "MOD:00047"),
    ]
    assert [(m.positions, m.name) for m in e.mod_res] == [((9,), "N6-acetyllysine")]


def test_peff_write_read_roundtrip(entry):
    header, e = entry
    buf = io.StringIO()
    pf.write_peff(header, [e], buf)
    _, again = pf.read_peff(io.StringIO(buf.getvalue()))
    assert again == [e]


def test_modified_residues_match_cv_origin(resolved, psimod_db):
    expected_residue = {2: "M", 4: "C", 5: "S", 9: "K", 12: "T"}
    assert set(resolved) == set(expected_residue)
    for pos, aa in expected_residue.items():
        assert SEQUENCE[pos - 1] == aa
    for pos in (5, 12, 9):
        mid = _accession_number(resolved[pos][0])
        assert psimod_db[mid].origin == SEQUENCE[pos - 1]


def test_proforma_mass_is_unmodified_plus_cv_deltas(resolved):
    proforma = _proforma(SEQUENCE, resolved)
    expected = pt.parse(SEQUENCE).mass() + sum(delta for _, delta in resolved.values())
    assert pt.parse(proforma).mass() == pytest.approx(expected, abs=1e-5)


def test_proforma_by_name_equals_by_accession(resolved, unimod_db, psimod_db):
    by_name = {}
    for pos, (acc, delta) in resolved.items():
        db, n = acc.split(":")
        name = unimod_db.get_by_id(int(n)).name if db == "UNIMOD" else psimod_db[int(n)].name
        prefix = "U" if db == "UNIMOD" else "M"
        by_name[pos] = (f"{prefix}:{name}", delta)
    assert pt.parse(_proforma(SEQUENCE, by_name)).mass() == pytest.approx(
        pt.parse(_proforma(SEQUENCE, resolved)).mass(), abs=1e-6
    )


def test_digested_peptides_keep_their_mods(resolved):
    protein = pt.parse(SEQUENCE)
    seen = set()
    for span in protein.digest_spans("trypsin", missed_cleavages=1, min_len=1, max_len=None):
        peptide = SEQUENCE[span.start : span.end]
        mods = {p: v for p, v in resolved.items() if span.start < p <= span.end}
        seen |= set(mods)
        proforma = _proforma(peptide, mods, offset=span.start)
        expected = pt.parse(peptide).mass() + sum(d for _, d in mods.values())
        assert pt.parse(proforma).mass() == pytest.approx(expected, abs=1e-5), proforma
    assert seen == set(resolved)
