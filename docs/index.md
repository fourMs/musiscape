# musiscape

A Python toolbox for analysing large music collections and long music
recordings.

Existing tools answer the question *"what is this track?"*. musiscape
answers two others. For a collection, it shows how albums differ, which
tracks resemble which, how internally consistent each album is, where the
outliers live, and what categories the corpus falls into. For a long
recording such as a concert, it finds the songs inside it first, so the same
tools apply to a live set. Every number it reports carries a musical name.

![album fingerprints](img/fingerprints.png)

musiscape is a sibling of [ambiscape](https://github.com/fourMs/ambiscape)
(soundscapes) and [micromotion](https://github.com/fourMs/micromotion)
(human micromotion). Each runs on its own, and musiscape needs neither
installed.

Where a measure appears in more than one toolbox it has one owner and one
implementation, so combining them or moving between them gives the same
number. Circular statistics belong to micromotion: pulse clarity for
rubato-heavy material where BPM fails, tonal centres on the circle of
fifths, and the Rayleigh test all come from there. The object-level
TARTYP and TARSOM proxies live here, in `musiscape.music`.

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

Jensenius, A. R. (2026). *musiscape: A Python toolbox for analysing large music collections and
long music recordings* [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.21964192>

That is the concept DOI and it always resolves to the newest version. Where the exact behaviour
matters, add the version you ran; every release has its own DOI, listed on the Zenodo record.
