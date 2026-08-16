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

`extract` runs first inside every verb that needs features, so the verbs above it in the table can
be run on their own; `probe`, `thumbnails`, `poster` and `sonic` do not need `features.json` to
exist, though `thumbnails` will use it if it is there.

`segment` is the exception to the collection rule: it reads a folder of *recordings* rather than a
collection, because a folder of camera files holds no audio files at all. What it writes is a
collection, which every other verb then accepts — see [Concerts & long recordings](guide/concerts.md).

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

`--style` accepts `mel`, `chroma`, `tempo`, `combo`, `barcode`, `ssm`, `trajectory`, `keyscape`,
`rhythm`, `wave`, `vinyl`, `spiral`, `tonnetz`, `arcs`, `schaeffer`, `tarsom` and `stereo`.
`poster` accepts only `barcode` and `vinyl`, and falls back to `barcode` for anything else. What
each one shows is in [Visual thumbnails & posters](guide/thumbnails.md).

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
musiscape categorize  ~/Music/my-collection -k 6
musiscape thumbnails  ~/Music/my-collection --style combo
musiscape sonic       ~/Music/my-collection
```
