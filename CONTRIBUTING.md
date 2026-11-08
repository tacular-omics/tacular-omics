# Contributing to tacular-omics

tacular-omics is a meta-package. It pins and re-exports twelve packages and holds the
end-to-end tests that run across them. It contains very little code of its own.

## Where your issue probably belongs

Almost every bug in tacular-omics is a bug in one of its member packages, and it is fixed
there. If you can name the package, please file the issue on that repository instead, because
that is where the maintainers and the tests for that behaviour live.

| what you are working with | repository |
|---|---|
| peptide sequences, ProForma, fragments, digestion | [peptacular](https://github.com/tacular-omics/peptacular) |
| spectra, spectral libraries, mzSpecLib, deconvolution | [spxtacular](https://github.com/tacular-omics/spxtacular) |
| mzML reading and indexing | [mzmlpy](https://github.com/tacular-omics/mzmlpy) |
| Bruker timsTOF .d and TDF | [tdfpy](https://github.com/tacular-omics/tdfpy) |
| FASTA parsing and decoys | [fastatacular](https://github.com/tacular-omics/fastatacular) |
| PEFF | [pefftacular](https://github.com/tacular-omics/pefftacular) |
| PSM and alignment files, PAF | [paftacular](https://github.com/tacular-omics/paftacular) |
| modification ontologies: PSI-MOD, Unimod, UniProt PTM | [psimodpy](https://github.com/tacular-omics/psimodpy), [unimodpy](https://github.com/tacular-omics/unimodpy), [uniprotptmpy](https://github.com/tacular-omics/uniprotptmpy) |
| chemical formulas, masses, isotopes | [tacular](https://github.com/tacular-omics/tacular) |

File here instead when the problem is the combination rather than any one package: an
installation that cannot resolve, a version pin that is too tight or too loose, or two
packages that disagree with each other.

## Development setup

```
git clone https://github.com/tacular-omics/tacular-omics
cd tacular-omics
uv sync --locked
uv run pytest
```

`uv sync --locked` installs everything the test suite needs, the optional readers included.

## What a change here looks like

The version floors in `pyproject.toml` are the main thing this repository owns. Each member
package is pinned with a floor and a major cap, for example `peptacular>=5.1.0,<6`. A pull
request that raises a floor should say which release of that package it needs and why.

Do not change the version of tacular-omics itself, and do not add a version heading to
`CHANGELOG.md`. Put your entry under `## [Unreleased]`.
