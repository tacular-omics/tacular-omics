# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] (2026-09-24)

### Changed

- **Breaking:** raises every member to its new release: tacular 2.0, peptacular 5.0,
  paftacular 2.0, tdfpy 5.0, mzmlpy 0.10 and spxtacular 0.9 (all with breaking API
  changes; see each package's changelog), and psimodpy, unimodpy, uniprotptmpy,
  fastatacular and pefftacular 1.1.

### Added

- End-to-end tests across the member packages: a FASTA protein through digestion,
  fragmentation (checked against `peptacular.fragment_arrays`), a synthetic spectrum
  annotated with spxtacular, and an mzSpecLib written and streamed back with the peaks,
  mzPAF annotations and ProForma unchanged; paftacular recomputes every fragment m/z.
  A test that the `ToleranceUnit` and `Polarity` aliases of mzmlpy and tdfpy equal
  `tacular.types`. The cross-package tests from the development workspace (mods,
  mzPAF, isotope formulas, PEFF, mzML, tdfpy) moved here too.

## [0.1.0] (2026-09-24)

### Added

- First release. Installs the tacular-omics core packages at their current released
  versions: tacular 1.2, psimodpy 1.0, unimodpy 1.0, uniprotptmpy 1.0, fastatacular
  1.0, pefftacular 1.0, mzmlpy 0.9.3, tdfpy 4.1.1, peptacular 4.2, paftacular 1.4,
  spxtacular 0.8. Extras `mcp` and `all`.
- `tacular_omics.versions()` and `python -m tacular_omics` report the installed
  version of each member package.
