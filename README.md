# tacular-omics

[![PyPI](https://img.shields.io/pypi/v/tacular-omics)](https://pypi.org/project/tacular-omics/)
[![CI](https://github.com/tacular-omics/tacular-omics/actions/workflows/ci.yml/badge.svg)](https://github.com/tacular-omics/tacular-omics/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/tacular-omics/tacular-omics)](https://github.com/tacular-omics/tacular-omics/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/tacular-omics)](https://pypi.org/project/tacular-omics/)

One install for the [tacular-omics](https://github.com/tacular-omics) proteomics packages.
`tacular-omics` has no code of its own: it depends on every core package at a set of
released versions that were tested together. Import the member packages directly.

## Install

```bash
pip install tacular-omics
```

Optional extras:

```bash
pip install "tacular-omics[mcp]"   # the MCP servers of every package that ships one
pip install "tacular-omics[all]"   # every optional feature of every package
```

Check what you got:

```bash
python -m tacular_omics            # or: tacular-omics
```

```python
import tacular_omics
tacular_omics.versions()   # {"tacular": "1.2.0", "psimodpy": "1.0.0", ...}
```

## Member packages

| Package | What it does |
|---|---|
| [tacular](https://github.com/tacular-omics/tacular) | Proteomics ontology and reference-data lookups (UNIMOD, PSI-MOD, elements, amino acids, proteases, ...). |
| [psimodpy](https://github.com/tacular-omics/psimodpy) | The PSI-MOD protein modification ontology. |
| [unimodpy](https://github.com/tacular-omics/unimodpy) | Parse and query the UNIMOD modifications database. |
| [uniprotptmpy](https://github.com/tacular-omics/uniprotptmpy) | Parse and query the UniProt PTM controlled vocabulary. |
| [fastatacular](https://github.com/tacular-omics/fastatacular) | Read and write FASTA sequence files. |
| [pefftacular](https://github.com/tacular-omics/pefftacular) | Read and write PEFF (PSI Extended FASTA Format) files. |
| [mzmlpy](https://github.com/tacular-omics/mzmlpy) | Lightweight mzML mass spectrometry file parser. |
| [tdfpy](https://github.com/tacular-omics/tdfpy) | Bruker timsTOF (`.d` / TDF) data with centroiding and noise filtering. |
| [peptacular](https://github.com/tacular-omics/peptacular) | Parse, annotate and analyze ProForma 2.1 peptide and protein sequences. |
| [paftacular](https://github.com/tacular-omics/paftacular) | Parse, serialize and analyze HUPO-PSI mzPAF peak annotations. |
| [spxtacular](https://github.com/tacular-omics/spxtacular) | Mass spectrometry spectrum processing. |

## How versions are chosen

Each `tacular-omics` release pins the member packages to the set that was released
and tested together: every requirement has a floor (the tested release) and a cap (the
next major version, or the next minor for 0.x packages such as `mzmlpy>=0.9.3,<0.10`).
You get bug-fix and feature releases of each member automatically, but never a breaking
release that has not been tested with the others. When the members release a new
batch, a new `tacular-omics` version raises the pins. The exact pins are in
[`pyproject.toml`](pyproject.toml).

To use a newer member than the cap allows, install that package without
`tacular-omics`.

## Citation

Please cite the packages you use, not this installer. Each member repository has a
`CITATION.cff` (GitHub's "Cite this repository" button) and a Zenodo DOI:

| Package | DOI |
|---|---|
| tacular | [10.5281/zenodo.18475556](https://doi.org/10.5281/zenodo.18475556) |
| psimodpy | [10.5281/zenodo.22926360](https://doi.org/10.5281/zenodo.22926360) |
| unimodpy | [10.5281/zenodo.22926362](https://doi.org/10.5281/zenodo.22926362) |
| uniprotptmpy | [10.5281/zenodo.22926364](https://doi.org/10.5281/zenodo.22926364) |
| fastatacular | [10.5281/zenodo.22926358](https://doi.org/10.5281/zenodo.22926358) |
| pefftacular | [10.5281/zenodo.22925639](https://doi.org/10.5281/zenodo.22925639) |
| mzmlpy | [10.5281/zenodo.21960079](https://doi.org/10.5281/zenodo.21960079) |
| tdfpy | [10.5281/zenodo.19100532](https://doi.org/10.5281/zenodo.19100532) |
| peptacular | [10.5281/zenodo.15054278](https://doi.org/10.5281/zenodo.15054278) |
| paftacular | [10.5281/zenodo.19076277](https://doi.org/10.5281/zenodo.19076277) |
| spxtacular | [10.5281/zenodo.19342437](https://doi.org/10.5281/zenodo.19342437) |

## License

MIT. Each member package has its own license; see its repository.
