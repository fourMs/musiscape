# Features, corpus statistics & categories

## Per-track features

`features.extract_collection` computes an interpretable descriptor set per
track (cached in `features.json`, extracted in parallel):

| Feature | Musical reading |
|---|---|
| `onset_rate` | note density—plucked events per second |
| `centroid_hz` | brightness (spectral centroid) |
| `flatness` | inharmonic texture—buzz, bowls, breath |
| `zcr`, `flux` | surface noisiness, spectral change |
| `perc_ratio` | percussive share (harmonic/percussive separation) |
| `dyn_range_db` | loud-to-quiet span within the track |
| `chroma_entropy` | pitch-class spread |
| `key`, `key_conf` | Krumhansl–Schmuckler estimate + correlation |
| `pulse_R`, `pulse_bpm` | circular pulse clarity and its period |
| `tempo_bpm` | perceptually-weighted tempo (what cards display) |
| `fifths_center`, `fifths_R` | tonal centre and focus on the circle of fifths |
| `tartyp` | duration-shares of Schaeffer object types |

The set is deliberately small enough to explain, and it is the interpretable
counterpart to embedding models. The trade is weaker raw similarity, but
every axis has a musical name.

!!! note "Tempo, honestly"
    Beat trackers fail on rubato material, so two numbers are kept apart:
    `pulse_R` measures *metric lock* (circular concentration of onset
    phases; 0 = free, 1 = metronomic), while `tempo_bpm` is librosa's
    perceptually-weighted estimate targeting the felt beat. Cards display
    the latter, `~`-prefixed when `pulse_R < 0.1`.

## Corpus statistics

```python
corpus.album_stats(f)    # mean/std/min/max per feature, keys, minor share
corpus.similarity(f)     # cosine matrix + album affinity & consistency
corpus.landscape(f)      # PCA coords, explained variance, loadings
corpus.tonal_spread(f)   # circular concentration of tonal centres per album
```

The affinity diagonal is each album's internal consistency. An album with
one instrument and one mood scores high, while an eclectic album scores near
zero. `tonal_spread` answers a question with no linear equivalent, since key
centres have no meaningful mean: a repertoire in neighbouring keys scores R
near 1, and one that wanders the whole circle near 0.

## Categories

`categorize.cluster(f, k=None)` runs k-means in the standardised feature
space (k chosen by silhouette unless given) and describes every cluster by
its three most distinguishing features as signed z-scores:

```
category 2 (15): centroid_hz +1.4, zcr +1.2, flatness +1.0
```

That reads as "bright, noisy, inharmonic": the textural tracks, wherever
their album membership put them. A category is never just "cluster 3".
