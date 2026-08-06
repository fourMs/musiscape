# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are summaries of what the commits say rather than notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely while pre-1.0.

## [0.3.2] — 2026-08-06

### Changed
- Documentation overhaul: beginner-oriented README, verified command
  reference, neutral voice throughout docs, wiki, and docstrings; the
  seventeenth thumbnail style (`trajectory`) is now documented; missing
  `sonic` and `extract` commands added to the guides; new demo figures
  from genuine tool output.

## [0.3.1] — 2026-08-03

### Fixed
- **`__version__` said 0.1.0 while `pyproject.toml` said 0.3.0.** The package on PyPI had been
  announcing a version two releases behind what it actually was, and nothing checked. Both now agree.
- **CI had been red since 2026-08-01.** `test_stereo_card` imported `tests.conftest`, and there is
  no `tests/__init__.py`, so `tests` is not an importable package. It passed locally, where the
  working directory happened to make it resolve, and failed on every CI run. `from conftest import`
  is the form that works in both places.

### Changed
- Documentation prose follows the project's dash convention; two catalogues that had been built out
  of dashes are recast.
- Shared "Related toolboxes" section; live CI badge and supported Python versions in the README.

## [0.3.0] — 2026-07-25
### Added
- Sonic thumbnails.
- Stereo thumbnail style: pan-by-frequency field, goniometer, width.
- mkdocs documentation site with a style gallery and a deploy workflow.
### Changed
- TARSOM card uses a centre-periphery rose rather than gauge rows.

## [0.2.0] — 2026-07-25
### Added
- `tarsom` thumbnail style, after Schaeffer's seven morphological criteria.
- `schaeffer` thumbnail style: a typo-morphology object timeline.
- Circular styles: vinyl, spiral, tonnetz, arcs, with a beat-wheel inset.
- Freesound-style waveform thumbnail, coloured by spectral centroid.

## [0.1.0] — 2026-07-25
First release: a corpus analysis toolbox for music collections, with CI and PyPI
trusted-publishing workflows.
