# Command line

Everything the package does is reachable from one command:

```
musiscape <verb> <collection-folder> [options]
```

A collection folder holds one subfolder per album, and `open_collection` reads it; see
[Collections & tracks](guide/collections.md) for what counts as a track. Output goes to
`<collection>/analysis/` unless `-o` says otherwise, and the folder is created if it does not
exist.

## The verbs

| verb | what it does | what it writes |
|---|---|---|
| `probe` | lists albums and their tracks, reading no audio | nothing, prints to the terminal |
| `extract` | computes per-track features | `features.json` |
| `fingerprint` | per-album profile bars from those features | `fingerprints.png`, `album_stats.json` |
| `landscape` | PCA map of the collection plus an affinity matrix | `landscape.png`, `affinity.png` |
| `categorize` | k-means over the features, with a named signature per category | `categories.json`, and the signatures to the terminal |
| `report` | runs the analysis end to end and writes the summary page | `README.md` in the output folder, with the figures beside it |
| `thumbnails` | one visual thumbnail per track | an image per track |
| `poster` | a single sheet of the whole collection | one image |
| `sonic` | a sonic thumbnail per track: about twelve seconds of audio summarising it | an audio file per track |
| `segment` | finds the songs in a continuous concert recording, video included | `songs.json` and one audio file per song under `songs/` |
| `figures` | labelled chromagram and tempogram per track --- the readable counterpart to the thumbnail cards | two PNGs per track under `figures/` |
| `pdf` | summary table of every track's estimates, then a page of figures per track | `report.pdf` |

`extract` runs first inside every verb that needs features, so the verbs above it in the table can
be run on their own; `probe`, `thumbnails`, `poster` and `sonic` do not need `features.json` to
exist, though `thumbnails` will use it if it is there.

`segment` is the exception to the collection rule: it reads a folder of *recordings* rather than a
collection, because a folder of camera files holds no audio files at all. What it writes is a
collection, which every other verb then accepts. See [Concerts & long recordings](guide/concerts.md).

## Options

| option | applies to | meaning |
|---|---|---|
| `-o`, `--out` | all | output folder; default `<collection>/analysis` |
| `--workers N` | everything that reads audio | parallel workers, default 4 |
| `--duration N` | `extract`, `report` and anything that extracts | analyse only the first N seconds of each track, which is how to try a large collection quickly |
| `-k N` | `categorize`, `report` | number of categories; chosen automatically when omitted |
| `--style NAME` | `thumbnails`, `poster` | which thumbnail to draw, default `mel` |
| `--min-song N` | `segment` | shortest span counted as a song, default 60 s |
| `--min-gap N` | `segment` | shortest break that ends a song, default 8 s |
| `--width N` | `figures` | export width in pixels, default 1920 |

`--style` accepts `mel`, `chroma`, `tempo`, `combo`, `barcode`, `ssm`, `trajectory`, `keyscape`,
`rhythm`, `wave`, `vinyl`, `spiral`, `tonnetz`, `arcs`, `schaeffer`, `tarsom` and `stereo`.
`poster` accepts only `barcode` and `vinyl`, and falls back to `barcode` for anything else. What
each one shows is in [Visual thumbnails & posters](guide/thumbnails.md).

## Figures vs. thumbnails

Two different jobs, deliberately kept apart. `thumbnails` draws cards for *browsing*: no axes, since
at card size axes are noise and the point is to recognise a piece at a glance. `figures` draws the
same material to be *read*: every axis carries its unit, time runs in `m:ss`, and the tempo axis is
logarithmic so that 60→120 occupies the same space as 120→240. Use the cards to find the track and
the figures to argue about it.

`pdf` puts both kinds of information in one document: a table of every estimate with the share of
analysis windows that agreed on it, then one page of figures per track. Nothing in it claims to say
whether a track has a pulse; see [Concerts & long recordings](guide/concerts.md) for why that
question resisted every measure tried.

## The sonic thumbnail

`sonic` is the one verb with no visual output and the one least like the others. It picks up to
three segments from each track by the track's own structure --- the most representative passage,
the peak-energy climax, and the most contrasting section that still carries energy --- places them
in chronological order and joins them with equal-power crossfades, so the summary keeps the piece's
own dramaturgy. Nothing is learned: the choice is deterministic and can be explained for any track.

## Examples

```
musiscape probe       ~/Music/my-collection
musiscape report      ~/Music/my-collection --workers 8
musiscape extract     ~/Music/my-collection --duration 60     # a quick pass over a big collection
musiscape segment     ~/video/concert                         # a live set, cut into songs
musiscape figures     ~/Music/my-collection --width 1920      # readable chromagrams and tempograms
musiscape pdf         ~/Music/my-collection                   # the hand-it-over report
musiscape categorize  ~/Music/my-collection -k 6
musiscape thumbnails  ~/Music/my-collection --style combo
musiscape sonic       ~/Music/my-collection
```
