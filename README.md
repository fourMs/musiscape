# musiscape

A toolbox for analysing **music collections**. Existing tools answer *"what
is this track?"* — musiscape answers **"what is this collection?"**: how its
albums differ, which tracks resemble which, how internally consistent each
album is, where the outliers live, and what categories the corpus falls
into — with every number carrying a musical name.

Sibling of [ambiscape](https://github.com/fourMs/ambiscape) (soundscapes),
reusing its circular-statistics and Schaeffer typology machinery
(`ambiscape.music`): pulse clarity for rubato-heavy material where BPM
fails, tonal centers on the circle of fifths, and object-level TARTYP
profiles.

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

## Honesty

Features are interpretable signal proxies — a deliberate trade against
embedding models: weaker raw similarity, but every axis can be argued
about. Key estimates are Krumhansl–Schmuckler correlations (indicative for
drones); pulse clarity conflates slow tempo drift with rubato; TARTYP
classes are corpus-calibrated proxies for aural categories. Treat every
category as a draft for listening, not a verdict.

## License

MIT.
