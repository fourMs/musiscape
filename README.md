# Musiscape

[![CI](https://github.com/fourMs/musiscape/actions/workflows/ci.yml/badge.svg)](https://github.com/fourMs/musiscape/actions/workflows/ci.yml)
[![docs](https://github.com/fourMs/musiscape/actions/workflows/docs.yml/badge.svg)](https://fourms.github.io/musiscape/)
[![PyPI version](https://img.shields.io/pypi/v/musiscape)](https://pypi.org/project/musiscape/)
[![Python](https://img.shields.io/pypi/pyversions/musiscape.svg)](https://pypi.org/project/musiscape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A toolbox for analysing music collections in the form of audio files in folders. The aim is to provide tools to visualise many tracks so you can compare them. It can help answer questions about how albums differ, which tracks resemble which, how internally consistent each album is, where the outliers live, and what categories the corpus falls into.

Musiscape is a sibling of [ambiscape](https://github.com/fourMs/ambiscape), which is focused on analysing *soundscapes* (including those with some music being played). The toolboxes share some analysis types, including circular statistics and Pierre Schaeffer-inspired typology machinery (`ambiscape.music`), tonal centres on the circle of fifths, and object-level TARSOM (*Tableau rècapitulatif du solfège des objets musicaux*) and TARTYP (*Tableau rècapitulatif de la typologie*).

See the [documentation](https://fourms.github.io/musiscape/) for guides and an illustrated gallery of the visualisation styles, and the [wiki](https://github.com/fourMs/musiscape/wiki) for methodology notes and a case study.

## Install

```bash
pip install musiscape
```

## Quickstart

A *collection* is simply a folder tree — each subfolder holding audio files
(mp3/wav/flac/ogg/m4a) is an album:

```bash
musiscape probe       ~/Music/my-collection    # what's here?
musiscape report      ~/Music/my-collection    # everything → analysis/README.md
musiscape fingerprint ~/Music/my-collection    # per-album profile bars
musiscape landscape   ~/Music/my-collection    # PCA similarity map + affinity
musiscape categorize  ~/Music/my-collection    # k-means with named signatures
musiscape thumbnails  ~/Music/my-collection --style combo   # per-track cards
musiscape poster      ~/Music/my-collection    # whole collection as one image
musiscape sonic       ~/Music/my-collection    # ~12 s audio summary per track
```

The seventeen thumbnail styles range from spectrograms and Freesound-style
coloured waveforms through tonality vinyl discs, keyscapes and repetition
arcs to Schaeffer-inspired TARTYP timelines, TARSOM roses and a
stereo-field view. Posters stack
every track as a harmony barcode, or lay the collection out as a grid of
tonality discs (`--style vinyl`).

`report` produces a per-collection `README.md` with an album table
(note density, brightness, dynamics, minor-key share, internal consistency,
key-space clustering R), overview figures, self-explaining categories
("sparse, dark, drone-like" — signed z-scores of the distinguishing
features, never just "cluster 3"), and the corpus extremes.

## In Python

```python
import musiscape
from musiscape import features, corpus, categorize

coll = musiscape.open_collection("~/Music/my-collection")
f = features.load_features(features.extract_collection(coll, "analysis"))
corpus.album_stats(f)      # per-album fingerprints
corpus.similarity(f)       # track matrix + album affinity/consistency
corpus.landscape(f)        # PCA coords, variance, loadings
corpus.tonal_spread(f)     # key-space clustering per album (circular)
categorize.cluster(f)      # interpretable categories
```

## Combine with listening

This toolbox is meant for quick visualisations and overviews. Combine it with listening!

## License

MIT.

## Related toolboxes

These four toolboxes come out of the [fourMs lab](https://github.com/fourMs) at the University of
Oslo. They are separate packages with separate release cycles, but they are built to be used
together and share several implementations, so a measure computed in one agrees with the same
measure computed in another.

- [Musical Gestures Toolbox](https://github.com/fourMs/MGT-python) (`musicalgestures`) — video and audio: motiongrams, videograms, and motion analysis from ordinary video files
- [ambiscape](https://github.com/fourMs/ambiscape) — soundscapes: the sonic ambience of a place, across level, spectral, spatial, temporal, ecological and source descriptors
- [micromotion](https://github.com/fourMs/micromotion) — human micromotion: quantity of motion from optical markers, accelerometers, respiration belts and force plates
