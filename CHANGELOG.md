# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are summaries of what the commits say rather than notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely while pre-1.0.

> **Release order, from 2026-08-12.** Moving music analysis out of ambiscape
> coupled these three releases, and shipping them out of order breaks things:
>
> 1. **micromotion** must go first. musiscape now needs `circular_sd` and
>    `rayleigh_from_R`, which exist only in the unreleased tree; PyPI has
>    1.2.1.
> 2. **musiscape** second. Until it ships, the published 0.4.0 still does
>    `from ambiscape import music`.
> 3. **ambiscape** last. Releasing it before musiscape breaks the published
>    musiscape 0.4.0, whose feature extraction imports the functions that
>    moved.
>
> **Resolved 2026-08-16.** micromotion 1.12.2 is on PyPI, so step 1 is done
> and musiscape is free to ship. ambiscape still goes last.


## 0.8.0 — 2026-08-19

### Added
- `musiscape.examples`, so the documentation runs with no music of your own. Every guide
  pointed at `~/Music/my-collection`, a folder the reader may not have and which certainly does
  not hold what the text assumes. `examples.demo_collection(path)` writes two albums of three
  synthetic tracks, plucked against drone, chosen so the contrast the analysis is meant to find
  is genuinely there — about 1.3 MB, and every command in the guides works against it. The
  generators are the ones `tests/conftest.py` already used; the fixture imports them now rather
  than keeping its own copy, since a fixture that drifts from the shipped example tests
  something no user can reproduce.
- Every module named in the API reference is now reachable from `import musiscape`:
  `categorize`, `concert`, `corpus`, `features`, `figures`, `io`, `music`, `report`, `sonic`,
  `stability` and `thumbnails`, plus `open_collection`, `extract_collection`, `extract_track`
  and `load_features`. The package exported two names before this, so the documentation listed
  eleven modules a reader following it could not see, and anyone scripting the toolbox rather
  than driving its command line had to guess the import paths.
- `__dir__`, so tab completion in a notebook lists the same names.

### Changed
- The submodules resolve LAZILY, on first attribute access, through PEP 562 `__getattr__`.
  Importing them eagerly is the obvious way to write this and it is wrong: `figures` and
  `thumbnails` pull in matplotlib, so a bare `import musiscape` would drag the plotting stack
  into `musiscape sonic`, which needs none of it. That was not caught by review — the existing
  `test_sonic_does_not_import_the_plotting_stack` failed the moment the eager version was
  written, which is a test earning its keep.

## 0.7.3 — 2026-08-16

### Changed
- **`--min-gap` defaults to 12 s rather than 8.** Chosen from the stability of the answer rather
  than from taste. Near the flatness threshold the per-second decision is unstable: on the reference
  concert 17 % of frames sat within 0.2 of the Otsu cut, and a passage of continuous music at a
  steady −20 dB produced runs of up to eight non-music seconds. That concert yields nine songs at a
  gap of 8 s and eight at anything from 10 to 15, so the old default sat on a cliff and the new one
  is in the middle of the plateau. Real breaks between songs on that material ran 24 s and longer.


## 0.7.2 — 2026-08-16

### Fixed
- **The concert clock no longer drifts.** Frames were pooled by a rounded number of STFT hops, so at
  22.05 kHz with a 512 hop a "one second" frame was really 43 hops, or 0.99846 s. Every reported
  time therefore ran about 0.15 % late: two seconds by the end of a 23-minute camera file, four and
  a half across a 53-minute concert. It was enough for a span to end after the file it named, which
  made `songs.json` claim a length the clip beside it did not have, and it put every exported
  filename's timestamp progressively late. Pooling boundaries now come from time rather than from an
  accumulated frame count, and the frame count from the audio's real duration.


## 0.7.1 — 2026-08-16

### Changed
- The concert timeline carries its class colours on the waveform itself rather than in a separate
  ribbon above it. One lane shows both what happened and how loud it was, so an applause swell dying
  away looks different from a block saying only that applause occurred.


## 0.7.0 — 2026-08-16

### Added
- **A labelled timeline of a whole recording.** `segment` now classifies every second as `music`,
  `applause`, `voices`, `quiet` or `other`, writes `regions.json`, and draws `timeline.png`: one
  ribbon for a whole evening, with a level curve under it. Music is taken from the setlist so the
  ribbon and the song listing cannot disagree; the rest is decided from level, spectral flatness and
  how much that flatness moves. A syllable-rate modulation measure was tried for speech first and
  separates nothing on this material, where what does separate speech from a clapping room is that
  speech swings between voiced and unvoiced while a room is stationary.
- **Non-music spans exported for a soundscape toolbox.** `segment` writes everything that is not a
  song into `other/`, which `ambiscape analyze` reads as one session. The files are FLAC and each
  name leads with a `YYYYMMDD_HHMMSS` stamp from the recording's own start time, which is what lets
  the other toolbox put the evening on a real clock. The two packages meet at the file boundary and
  neither imports the other.
- `io.recording_start_time`, reading a container's `creation_time` or a filename stamp.

### Fixed
- **`segment` no longer reads its own output.** The default output folder sits inside the input
  folder and what it writes there is audio, so a recursive scan fed the second run the first run's
  songs: a three-file concert came back as eleven recordings and fifteen songs. `list_recordings`
  takes an `exclude` argument and the command passes its output folder.


## 0.6.2 — 2026-08-16

### Added
- Cross-toolbox agreement tests. Any measure appearing in more than one fourMs toolbox has a single
  owner, so musiscape's circular statistics are pinned against `micromotion`, which owns them, and
  against ambiscape's standalone copy when it happens to be installed. The ambiscape checks skip
  when it is not, since musiscape must not require it.

### Changed
- The related-toolboxes sections say plainly that each package runs on its own and that shared
  measures have one implementation, replacing a claim that musiscape reuses ambiscape's circular
  statistics, which stopped being true when they moved to micromotion.


## 0.6.1 — 2026-08-16

### Fixed
- **The `rhythm` card's beat-wheel inset is drawn again.** It was computed with a function imported
  from `ambiscape`, which is not a dependency, inside a `try` that swallowed the ImportError. The
  inset is documented behaviour, so for anyone installing musiscape from PyPI a documented feature
  was silently missing. It now uses `micromotion.circular`, which the package already requires.
- Stale cross-references and an install hint pointing at `ambiscape[music]` removed from
  `musiscape.music`, which has not lived in ambiscape for some time.

### Changed
- Description reworded to "A Python toolbox for analysing large music collections and long music
  recordings", which covers what `segment` added.
- Documentation and wiki rewritten for readers meeting the toolbox for the first time, with
  development history left to this file and to git.


## 0.6.0 — 2026-08-16

### Added
- **`musiscape.concert`: the songs inside a continuous recording.** The toolbox assumed one file is
  one track, so a live set averaged a whole evening into a single key and a single tempo. `segment`
  finds the songs first and writes one audio file per song plus a `songs.json` manifest; the folder
  it writes is an ordinary collection that every other verb accepts. Songs are separated from
  applause by **spectral flatness** rather than by level — an enthusiastic room is as loud as the
  band — with the threshold taken from each recording's own bimodal distribution, guarded so that a
  single-mode recording is not cut in half. A concert arriving as several camera files is one
  concert: a span that runs to the end of one file and resumes at the start of the next is joined,
  and the minimum-song test is applied after that join rather than before.
- **Video containers** (`.mp4`, `.mov`, `.mkv`, `.m4v`, `.avi`, `.mts`, `.m2ts`) are accepted by
  `segment`, decoded through ffmpeg, which is required only when such a file is actually handed
  over. Collections stay audio-only.
- `--min-song` and `--min-gap` on the command line.
- **`musiscape.stability`: whether an estimate held, not just what it averaged to.** `features`'
  two gates are whole-track averages, which catches a descriptor measuring noise but cannot tell
  that from one measuring four minutes of real music at too long a timescale. On live material the
  second case is the common one. `key_agreement`, `key_windowed`, `tempo_agreement` and
  `tempo_windowed_bpm` now travel beside the gated numbers in `features.json`, computed from the
  chromagram and onset envelope extraction already has in hand.
- **`figures`: labelled chromagram and tempogram per track**, at any `--width` (default 1920 px).
  The thumbnail cards stay unlabelled — they are for browsing; these are for reading numbers off.
- **`pdf`: a summary table of every track's estimates, then a page of figures per track.** Written
  with matplotlib's `PdfPages`, so no new dependency.

### Not added, deliberately
- **No single-number "is there a pulse" descriptor.** Two were implemented and measured against a
  real concert. Beat salience scored 1.58–1.93 on the songs and 1.50–1.67 on the applause between
  them; tempogram peak prominence scored 1.09–1.62 against 1.07–1.24. Both separate synthetic
  extremes cleanly and neither separates the real thing, because applause is itself rhythmic. An
  inter-beat regularity number failed earlier and more simply: `beat_track` fits one global grid, so
  its intervals cannot move when the tempo does. The reasoning is kept in `musiscape.stability` so
  the next person does not spend the afternoon rediscovering it.

### Fixed
- **`thumbnails` no longer dies on a collection whose audio sits in the root folder.** That album is
  named `"."`, and `"." + ".png"` is `..png` — a name Path reads as a dotfile with no suffix, so PIL
  could not infer a format and refused to save the contact sheet, losing the run after every card
  had been rendered. The README's own quickstart produces this album.

### Changed
- `cli` imports matplotlib lazily, at the verbs that plot. `segment` has no figures to draw and
  should not pay for the import.
- `feats_2hz` moves from `thumbnails` to `features`, and the album-name-to-filename rule from
  `thumbnails` to `io`. `sonic` reached into the plotting module for both, so an audio-only verb
  pulled in matplotlib — and pulled it in late, inside a process that had already loaded the audio
  stack, which on at least one machine segfaulted.

### Fixed
- **`sonic`'s medley for the root album is no longer a hidden file.** It was named `. medley.wav`,
  invisible on every Unix desktop.


## 0.5.0 — 2026-08-12

### Added
- **`musiscape.music`: the analysis this package was already built on.** Seven functions move here
  from `ambiscape.music` — `tempogram`, `chromagram`, `dominant_period`, `pulse_clarity`,
  `fifths_center`, `tonal_center_spread`, `tartyp_profile` — where they had lived in the soundscape
  toolbox while this one imported six of their symbols across `features.py`, `corpus.py` and
  `thumbnails.py`. Nothing was duplicated, so this was a relocation rather than a merge.

### Changed
- **musiscape no longer depends on ambiscape.** The dependency is now `micromotion>=1.8.0`, which
  owns circular statistics across the four toolboxes and gained `circular_sd` and `rayleigh_from_R`
  for this move. That also removes a packaging hazard: installing musiscape used to resolve
  ambiscape from PyPI, which could silently replace an editable checkout with a wheel of the same
  version number.

  The two session-aware functions did not travel. `load_w` and `run_session` know what an ambiscape
  `Session` is, and stayed there as a bridge — in the same sense as `musicalgestures._soundscape` is
  one the other way. MGT owns pixels, ambiscape owns samples, micromotion owns bodies, musiscape
  owns music, and each crossing is one small module that says so.


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
