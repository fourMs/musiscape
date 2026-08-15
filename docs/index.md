# musiscape

A toolbox for analysing music collections in the form of audio files in
folders. Existing tools answer the question *"what is this track?"*, while
musiscape answers "what is this collection?". It shows how albums differ,
which tracks resemble which, how internally consistent each album is, where
the outliers live, and what categories the corpus falls into. Every number
it reports carries a musical name.

![album fingerprints](img/fingerprints.png)

musiscape is a sibling of [ambiscape](https://github.com/fourMs/ambiscape)
(soundscapes), reusing its circular-statistics and Schaeffer typology
machinery: pulse clarity for rubato-heavy material where BPM fails, tonal
centres on the circle of fifths, and object-level TARTYP / TARSOM proxies.

## What it does

- Fingerprints—per-album profiles of note density, brightness,
  inharmonic texture, dynamics, pulse clarity and pitch-class entropy.
- Landscape—every track as a point in a PCA of the standardised
  features, plus an album-affinity matrix and internal-consistency scores.
- Categories—k-means clusters that describe themselves through signed
  feature z-scores ("sparse, dark, drone-like", rather than just "cluster 3").
- Thumbnails—seventeen per-track visual card styles, from spectrograms
  through Freesound-style waveforms, tonality vinyl discs, Sapp keyscapes
  and Shape-of-Song arcs, to Schaefferian TARTYP timelines and TARSOM
  morphology roses. See the [gallery](guide/thumbnails.md).
- Posters—the whole collection as stacked harmony barcodes or a grid
  of tonality discs.
- Sonic thumbnails—a ~12-second audio summary per track, plus one
  medley file per album, for browsing a collection by ear.
- Report—one command renders everything into a per-collection
  `README.md`.

## Honesty

Features are interpretable signal proxies, a deliberate trade against
embedding models: weaker raw similarity, but every axis can be argued about.
Key estimates are Krumhansl–Schmuckler correlations (indicative for drones);
pulse clarity conflates slow tempo drift with rubato; Schaeffer classes are
corpus-calibrated proxies for aural categories. Treat every category and
card as a draft for listening, not a verdict.


## Citing

Jensenius, A. R. (2026). *musiscape: analysis of music collections* (Version 0.5.0) [Computer software]. Zenodo.
<https://doi.org/10.5281/zenodo.21948999>

That is the CONCEPT DOI and it always resolves to the newest version. Where the exact behaviour
matters, name the version you ran as well: version 0.5.0 is
<https://doi.org/10.5281/zenodo.21949000>.
