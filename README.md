# musiscape

[![CI](https://github.com/fourMs/musiscape/actions/workflows/ci.yml/badge.svg)](https://github.com/fourMs/musiscape/actions/workflows/ci.yml)
[![docs](https://github.com/fourMs/musiscape/actions/workflows/docs.yml/badge.svg)](https://fourms.github.io/musiscape/)
[![PyPI version](https://img.shields.io/pypi/v/musiscape)](https://pypi.org/project/musiscape/)
[![Python](https://img.shields.io/pypi/pyversions/musiscape.svg)](https://pypi.org/project/musiscape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21964192.svg)](https://doi.org/10.5281/zenodo.21964192)

A Python toolbox for analysing large music collections and long music
recordings.

Point it at a folder of audio files, where each subfolder counts as an
album, and it renders figures, tables and thumbnails that let you compare
many tracks at a glance: albums, similarity, outliers, categories. Point it
at a concert recording instead and it finds the songs inside it first, so
the same tools apply to a live set.

## Install

```bash
pip install musiscape
```

## Quickstart

```bash
musiscape report ~/Music/my-collection
```

One command runs the whole pipeline and writes `analysis/README.md` next to
your music: an album table, a similarity landscape, self-explaining
categories, and the corpus extremes.

![Per-album fingerprint bars from a synthetic demo collection](docs/img/fingerprints.png)

## Commands

| Command | What it does |
|---|---|
| `probe` | list albums and tracks, no analysis |
| `extract` | per-track features, cached in `features.json` |
| `fingerprint` | per-album profile bars |
| `landscape` | PCA similarity map and album-affinity matrix |
| `categorize` | k-means categories with named signatures |
| `report` | everything above, gathered into `analysis/README.md` |
| `thumbnails` | one visual card per track (`--style`, seventeen styles) |
| `poster` | the whole collection as one image (`--style vinyl` for discs) |
| `sonic` | a ~12-second audio summary per track, plus album medleys |
| `segment` | finds the songs in a concert recording, labels every second of it, and writes one file per song |
| `figures` | labelled chromagram and tempogram per track, at any pixel width |
| `pdf` | one PDF: a summary table, then a page of figures per track |

A live recording is not a collection until it is cut into songs, which is what `segment` is for
(video files included, since concerts usually arrive as video):

```bash
musiscape segment ~/video/concert -o ~/video/concert/analysis
musiscape report  ~/video/concert/analysis/songs
```

Common options: `-o` sets the output folder (default `<root>/analysis`),
`--workers N` parallelises, `--duration S` analyses only the first S seconds
per track, and `-k` fixes the number of categories.

## Learn more

- [Documentation](https://fourms.github.io/musiscape/)—guides, an
  illustrated gallery of the visualisation styles, and the API reference
- [Wiki](https://github.com/fourMs/musiscape/wiki)—methodology notes and a
  case study

The toolbox is meant for quick visualisations and overviews. Combine it
with listening.

## Related toolboxes

These toolboxes come out of the [fourMs lab](https://github.com/fourMs) at
the University of Oslo.

- [Musical Gestures Toolbox](https://github.com/fourMs/MGT-python)
  (`musicalgestures`): video and audio, with motiongrams, videograms and
  motion analysis from ordinary video files
- [ambiscape](https://github.com/fourMs/ambiscape): soundscapes, the sonic
  ambience of a place, including multichannel and ambisonic material
- [micromotion](https://github.com/fourMs/micromotion): human micromotion,
  with quantity of motion from optical markers, accelerometers, respiration
  belts and force plates

Each is a separate package with its own release cycle, and each runs on its
own. musiscape needs none of the others installed to do anything it
documents.

They are still meant to be combined, so any measure that appears in more
than one of them has a single owner and a single implementation. Circular
statistics belong to micromotion, and musiscape imports pulse clarity,
fifths-circle centres and the Rayleigh test from there rather than keeping
its own copy. Switching a measure between toolboxes, or computing it in two
of them, gives the same number. A test suite in each package checks that
against the owning implementation, and skips when the sibling is not
installed.

## Licence and credit

MIT licence. musiscape is developed as part of the
[AMBIENT](https://www.uio.no/ritmo/english/projects/ambient/) project at
[RITMO Centre for Interdisciplinary Studies in Rhythm, Time and
Motion](https://www.uio.no/ritmo/english/), University of Oslo.

## Citing

Cite the concept DOI, which always resolves to the newest version:

> Jensenius, A. R. (2026). *musiscape: A Python toolbox for analysing large music collections and
> long music recordings* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21964192

Where the exact behaviour matters, add the version you ran. Every release has its own DOI, listed
on the [Zenodo record](https://doi.org/10.5281/zenodo.21964192).

`CITATION.cff` in this repository carries the same information in machine-readable form.
