# Musiscape

A toolbox for analysing music collections in the form of audio files in folders. The aim is to provide tools to visualise many tracks so you can compare them. It can help answer questions about how albums differ, which tracks resemble which, how internally consistent each album is, where the outliers live, and what categories the corpus falls into.

Musiscape is a sibling of [ambiscape](https://github.com/fourMs/ambiscape), which is focused on analysing *soundscapes* (including those with some music being played). The toolboxes share some analysis types, including circular statistics and Pierre Schaeffer-inspired typology machinery (`ambiscape.music`), tonal centres on the circle of fifths, and object-level TARSOM (*Tableau rècapitulatif du solfège des objets musicaux*) and TARTYP (*Tableau rècapitulatif de la typologie*).

## Install

```bash
pip install -e .          # from a checkout; PyPI release to follow
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
```

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
