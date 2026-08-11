# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are summaries of what the commits say rather than notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely while pre-1.0.

## [Unreleased]

### Changed
- **`features` documents two descriptors that answer when there is nothing to answer.** Neither is a
  bug and both mislead on any input that is not a music collection --- field recordings, broadcast
  audio, anything where "music" was decided by a detector rather than by a track listing.
  `tempo_bpm` is librosa's prior-based estimator and has no failure value: with no periodicity in the
  onset envelope it returns the tempogram bin nearest its 120 BPM prior, and white noise reproducibly
  gives 123.05 BPM. Measured on 704 five-minute spans of domestic television audio it returned five
  distinct values in all, 93 % of them exactly 123.0 and the rest adjacent grid points --- a result
  indistinguishable from noise. `pulse_R` is the gate: below about 0.1 there is no pulse for a tempo
  to describe. `key` and `key_conf` fail the same way through a near-uniform chroma, where the
  Krumhansl–Schmuckler correlation falls to whatever tiny bias survives and does so *consistently*,
  so a confidence split does not reveal it; `chroma_entropy` near its log2(12) ceiling is the gate.
  The spectral and temporal descriptors were unaffected on the same material and stay usable.

## [0.4.0] — 2026-08-08

### Fixed
- A worker killed by the operating system no longer loses the whole
  collection. Extracting features from a very long track costs several
  gigabytes in one process --- a three-quarter-hour span needed over six,
  and the kernel took the worker --- and that is not an exception: the
  process is gone, so the executor breaks and every future still pending
  dies with it. `extract_collection` used `ProcessPoolExecutor.map`, which
  re-raises on the first broken future, so one unlucky track cost all the
  others. Found on a domestic recording whose longest music span ran to
  forty-six minutes: it took the worker and seventy-one other tracks went
  with it, while the stage reported success.

### Added
- Tracks whose worker died are retried one at a time in their own process,
  with the analysis window capped to `retry_cap_s` (default 600 s) so the
  retry fits in memory. A capped result records `analysis_capped_s`, so a
  shortened window is visible in the output rather than implied by a
  duration that looks like a short track. `retry_cap_s=None` retries at
  full length, which will usually be killed again.
- Nothing is capped on the first attempt, so ordinary collections are
  extracted exactly as before. Only a track that has already proved fatal
  is analysed on a shorter window.

## [0.3.3] — 2026-08-08

### Fixed
- The package version has one source. `__version__` in
  `src/musiscape/__init__.py` is now the only place the number is written,
  and setuptools reads it from there through a dynamic version; the static
  `version` in `pyproject.toml` is gone. The two had drifted: 0.3.2 was
  released reporting itself as 0.3.1, because that bump edited
  `pyproject.toml` alone. Anything citing this toolbox by version is
  otherwise citing a number the installed package does not confirm.
  `tests/test_version.py` fails if a static version reappears in
  `pyproject.toml`, if the build stops reading the module attribute, or if
  the number setuptools would package differs from the one the module
  reports.

### Changed
- Documentation deploys through the Pages artifact flow, and the docs
  carry git revision dates (the house pattern across the toolboxes).
- `site/`, the local mkdocs build output, is ignored rather than sitting
  untracked in the working tree.

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
