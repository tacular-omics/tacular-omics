"""Pipeline 1: FASTA (fastatacular) -> protein -> peptacular trypsin digest -> ProForma
peptides carrying UNIMOD / PSI-MOD modifications resolved through tacular, with
unimodpy / psimodpy agreeing on every mass."""

from __future__ import annotations

import re

import fastatacular as ft
import peptacular as pt
import pytest
import tacular as tc
from conftest import HEAVY_BASE

# Human serum albumin fragment (UniProt P02768), long enough for many tryptic sites
# including K/R-P pairs.
ALBUMIN = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
    "LHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPE"
    "LLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKV"
)
PROTEINS = {
    "sp|P02768|ALBU_HUMAN": ("Albumin", ALBUMIN),
    "sp|P99999|TEST_HUMAN": ("Test protein", "MSTKPEPTIDERSAMPLEKCMSTRPQK"),
}

# (UNIMOD id, PSI-MOD id, PSI-MOD name, residue) for the same chemical modification.
MOD_PAIRS = [
    (35, 425, "monohydroxylated residue", "M"),
    (4, 1060, "S-carboxamidomethyl-L-cysteine", "C"),
    (21, 46, "O-phospho-L-serine", "S"),
    (21, 47, "O-phospho-L-threonine", "T"),
    (1, 64, "N6-acetyl-L-lysine", "K"),
    (7, 684, None, "N"),
]


@pytest.fixture(scope="module")
def fasta_entries(tmp_path_factory):
    path = tmp_path_factory.mktemp("fasta") / "proteins.fasta"
    ft.write_fasta(
        [
            ft.SequenceEntry(
                identifier=ident,
                sequence=seq,
                pname=name,
                os_name="Homo sapiens",
                ncbi_tax_id=9606,
            )
            for ident, (name, seq) in PROTEINS.items()
        ],
        path,
    )
    return path, ft.read_fasta(path)


def reference_tryptic_spans(sequence: str, missed: int) -> set[tuple[int, int, int]]:
    """Independent digest from tacular's trypsin regex, no peptacular involved."""
    regex = tc.PROTEASE_LOOKUP["trypsin"].regex
    sites = sorted({0, len(sequence), *(m.start() for m in re.finditer(regex, sequence))})
    spans = set()
    for i in range(len(sites) - 1):
        for mc in range(missed + 1):
            j = i + 1 + mc
            if j < len(sites):
                spans.add((sites[i], sites[j], mc))
    return spans


def test_fasta_roundtrip(fasta_entries):
    _, entries = fasta_entries
    assert [e.identifier for e in entries] == list(PROTEINS)
    assert [e.accession for e in entries] == ["P02768", "P99999"]
    assert [e.sequence for e in entries] == [seq for _, seq in PROTEINS.values()]
    # fastatacular is the one FASTA parser; its sequences go straight into peptacular.
    assert [pt.parse(e.sequence).sequence for e in entries] == [e.sequence for e in entries]


@pytest.mark.parametrize("missed", [0, 1, 2])
def test_trypsin_digest_matches_tacular_protease_regex(fasta_entries, missed):
    _, entries = fasta_entries
    for entry in entries:
        protein = pt.parse(entry.sequence)
        got = {(s.start, s.end, s.missed_cleavages) for s in protein.digest_spans("trypsin", missed_cleavages=missed)}
        assert got == reference_tryptic_spans(entry.sequence, missed), entry.identifier
        # Enum and string protease spellings give the same result.
        by_enum = {
            (s.start, s.end, s.missed_cleavages)
            for s in protein.digest_spans(pt.Protease.TRYPSIN, missed_cleavages=missed)
        }
        assert by_enum == got


def test_fully_cleaved_peptide_masses_sum_to_protein(fasta_entries):
    _, entries = fasta_entries
    water = pt.chem_mass("H2O")
    for entry in entries:
        protein = pt.parse(entry.sequence)
        peptides = [protein[s] for s in protein.digest_spans("trypsin", missed_cleavages=0)]
        assert "".join(p.sequence for p in peptides) == entry.sequence
        total = sum(p.mass() for p in peptides) - (len(peptides) - 1) * water
        assert total == pytest.approx(protein.mass(), abs=1e-6)


@pytest.mark.parametrize(("unimod_id", "psimod_id", "psimod_name", "residue"), MOD_PAIRS)
def test_mod_notations_resolve_to_one_mass(unimod_db, psimod_db, unimod_id, psimod_id, psimod_name, residue):
    uni = unimod_db.get_by_id(unimod_id)
    psi = psimod_db[psimod_id]
    expected = uni.delta_mono_mass
    assert tc.UNIMOD_LOOKUP[unimod_id].monoisotopic_mass == pytest.approx(expected, abs=1e-6)
    assert psi.diff_mono == pytest.approx(expected, abs=1e-6)
    assert tc.PSIMOD_LOOKUP[psimod_id].monoisotopic_mass == pytest.approx(expected, abs=1e-6)
    if psimod_name is not None:
        assert psi.name == psimod_name

    base = f"PEP{residue}TIDEK"
    unmodified = pt.parse(base).mass()
    notations = [
        uni.name,
        f"UNIMOD:{unimod_id}",
        f"U:{uni.name}",
        f"MOD:{psimod_id:05d}",
        f"M:{psi.name}",
        f"{uni.name}|MOD:{psimod_id:05d}",
        f"Formula:{uni.proforma_formula}",
    ]
    for tag in notations:
        peptide = pt.parse(f"PEP{residue}[{tag}]TIDEK")
        assert peptide.mass() - unmodified == pytest.approx(expected, abs=1e-5), tag
        # serialize -> parse is stable and mass-preserving
        assert pt.parse(peptide.serialize()).mass() == pytest.approx(peptide.mass(), abs=1e-9), tag


def test_digested_peptides_with_static_and_variable_mods(fasta_entries, unimod_db):
    _, entries = fasta_entries
    carbamidomethyl = unimod_db.get_by_id(4).delta_mono_mass
    oxidation = unimod_db.get_by_id(35).delta_mono_mass
    checked = 0
    for entry in entries:
        protein = pt.parse(entry.sequence)
        for span in protein.digest_spans("trypsin", missed_cleavages=2, min_len=6, max_len=30):
            seq = protein[span].sequence
            modded = "".join(f"{aa}[UNIMOD:35]" if aa == "M" else aa for aa in seq)
            proforma = f"<[UNIMOD:4]@C>{modded}"
            peptide = pt.parse(proforma)
            expected = pt.parse(seq).mass() + seq.count("C") * carbamidomethyl + seq.count("M") * oxidation
            assert peptide.mass() == pytest.approx(expected, abs=1e-5), proforma
            assert pt.parse(peptide.serialize()).mass() == pytest.approx(peptide.mass(), abs=1e-9)
            checked += 1
    assert checked > 50


def test_every_unimod_accession_resolves_through_peptacular(unimod_db):
    base = pt.parse(HEAVY_BASE).mass()
    bad = []
    for entry in unimod_db:
        if entry.id == 0 or entry.delta_mono_mass is None:
            continue
        mass = pt.parse(f"{HEAVY_BASE}[UNIMOD:{entry.id}]").mass() - base
        if abs(mass - entry.delta_mono_mass) > 1e-6:
            bad.append((entry.id, mass, entry.delta_mono_mass))
    assert bad == []


def test_every_psimod_accession_resolves_through_peptacular(psimod_db):
    base = pt.parse(HEAVY_BASE).mass()
    bad = []
    for entry in psimod_db:
        if entry.is_obsolete or entry.diff_mono is None:
            continue
        mass = pt.parse(f"{HEAVY_BASE}[MOD:{entry.id:05d}]").mass() - base
        if abs(mass - entry.diff_mono) > 1e-6:
            bad.append((entry.id, mass, entry.diff_mono))
    assert bad == []


def test_unimodpy_and_tacular_unimod_agree(unimod_db):
    bad = []
    for entry in unimod_db:
        if entry.id == 0:
            continue
        info = tc.UNIMOD_LOOKUP.get(entry.id)
        if info is None:
            bad.append((entry.id, "missing from tacular"))
            continue
        if entry.name != info.name:
            bad.append((entry.id, entry.name, info.name))
        if entry.delta_mono_mass is not None and abs(entry.delta_mono_mass - (info.monoisotopic_mass or 0)) > 1e-6:
            bad.append((entry.id, entry.delta_mono_mass, info.monoisotopic_mass))
        if entry.dict_composition and dict(entry.dict_composition) != dict(info.dict_composition or {}):
            bad.append((entry.id, entry.dict_composition, info.dict_composition))
    assert bad == []


def test_psimodpy_and_tacular_psimod_agree(psimod_db):
    bad = []
    for entry in psimod_db:
        if entry.is_obsolete or entry.diff_mono is None:
            continue
        info = tc.PSIMOD_LOOKUP.get(entry.id)
        if info is None:
            bad.append((entry.id, "missing from tacular"))
            continue
        if entry.name != info.name or abs(entry.diff_mono - (info.monoisotopic_mass or 0)) > 1e-6:
            bad.append(
                (
                    entry.id,
                    entry.name,
                    entry.diff_mono,
                    info.name,
                    info.monoisotopic_mass,
                )
            )
    assert bad == []


# --------------------------------------------------------------------------------------
# Regression (once a cross-package bug): names from unimodpy / psimodpy that end in "(...)".
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Acetyl:2H(3)", "Label:13C(6)", "Propionyl:13C(3)"])
def test_prefixed_unimod_name_with_parentheses(unimod_db, name):
    expected = unimod_db.get_by_name(name).delta_mono_mass
    assert tc.UNIMOD_LOOKUP.query_name(name) is not None  # tacular knows the name
    assert pt.parse(f"K[U:{name}]").mass() - pt.parse("K").mass() == pytest.approx(expected, abs=1e-6)


def test_unprefixed_unimod_name_with_count_suffix(unimod_db):
    expected = unimod_db.get_by_name("HexNAc(2)").delta_mono_mass  # 406.158745
    assert pt.parse("N[HexNAc(2)]").mass() - pt.parse("N").mass() == pytest.approx(expected, abs=1e-6)


def test_prefixed_psimod_name_with_parentheses(psimod_db):
    entry = psimod_db.get_by_name("L-cystine (cross-link)")
    assert tc.PSIMOD_LOOKUP.query_name(entry.name) is not None
    got = pt.parse(f"{HEAVY_BASE}[M:{entry.name}]").mass() - pt.parse(HEAVY_BASE).mass()
    assert got == pytest.approx(entry.diff_mono, abs=1e-6)


def test_every_cv_name_resolves_through_peptacular(unimod_db, psimod_db):
    base = pt.parse(HEAVY_BASE).mass()
    bad = []
    items = [("U", e.name, e.delta_mono_mass) for e in unimod_db if e.id and e.delta_mono_mass is not None]
    items += [("M", e.name, e.diff_mono) for e in psimod_db if not e.is_obsolete and e.diff_mono is not None]
    for prefix, name, expected in items:
        try:
            got = pt.parse(f"{HEAVY_BASE}[{prefix}:{name}]").mass() - base
        except Exception as exc:  # noqa: BLE001 - collect every failure for the report
            bad.append((prefix, name, type(exc).__name__))
            continue
        if abs(got - expected) > 1e-6:
            bad.append((prefix, name, got, expected))
    assert bad == []
