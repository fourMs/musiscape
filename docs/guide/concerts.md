# Concerts & long recordings

Everything else in musiscape assumes one file is one track. A concert is the other shape: one long
recording holding a sequence of songs, and often several files rather than one, because the camera
split at a size limit partway through the evening.

Analysed as it arrives, such a recording averages a whole evening into a single key and a single
tempo — which is to say, into nothing. `segment` finds the songs inside it first.

```
musiscape segment ~/video/concert -o ~/video/concert/analysis
musiscape report  ~/video/concert/analysis/songs
```

The first command writes one audio file per song plus a manifest; the second treats that folder as
an ordinary collection. Every other verb works on it too — `thumbnails --style keyscape` for
tonality, `--style tempo` for the tempogram, `--style combo` for both beside the spectrogram.

## What separates a song from the space around it

Not level. An enthusiastic room is as loud as the band, and a hall between songs is not silent.

**Spectral flatness** is what separates them. Applause is broadband noise and measures flat; played
music is tonal and measures roughly an order of magnitude lower. The two form distinct modes in a
concert recording, and the threshold between them is taken from each recording's own distribution
(Otsu's method on log-flatness) rather than from a fixed number, because the *ratio* survives a
change of room and microphone while the absolute values do not.

Two guards keep that split honest:

- Otsu returns a threshold for any distribution, including one with a single mode. The split is
  used only when the two classes are both **separated** (`BIMODAL_MIN_DECADES`) and **populated**
  (`BIMODAL_MIN_SHARE`). A rehearsal tape that is music throughout has one mode, and falls back to
  a level test rather than being cut in half.
- The threshold is the winning histogram bin's *upper edge*, not its centre. A sharp spike puts
  many frames at one value, and a centre-valued cut halves the spike itself.

A level floor, set relative to the recording's own loud frames, drops silence and distant room
tone underneath all of this.

## Turning a mask into songs

Runs of music separated by less than `--min-gap` seconds are one song: a quiet bar or a held breath
does not end a piece. Spans shorter than `--min-song` are not songs at all, which is what keeps
tuning, a false start, and a spoken introduction over a held chord out of the listing.

| option | default | meaning |
|---|---|---|
| `--min-song` | 60 s | shortest span counted as a song |
| `--min-gap` | 8 s | shortest break that ends a song |

## Several files, one concert

Files are read in name order, which is playing order — cameras number sequentially. A span that
runs to the end of one file and resumes at the start of the next is **one song**: the camera split
at a size limit, not at a musical boundary. That join happens *before* the minimum-length test, so
a song cut ten seconds before its end survives as a song instead of being discarded as a fragment.
Its clip holds both halves, back to back.

Times are reported on a **concert clock** that runs from the start of the first file and treats the
files as butted together. A camera that stops and restarts loses a few seconds at each join, and
that loss is not recoverable from the audio, so the clock drifts behind wall time by however long
the changeovers took. Within a song, the manifest's `parts` carry the true offsets into each source
file, which is what the clips are cut from.

## What it writes

```
analysis/
  songs.json              the manifest
  songs/
    01-MAH08537-0254.flac
    02-MAH08537-0729.flac
    ...
```

Clip names carry the running order, the source file, and the concert timecode of the song's start.
Each manifest entry looks like:

```json
{
 "start_s": 2712.0, "end_s": 2947.0, "duration_s": 235.0,
 "parts": [{"file": "MAH08538.MP4", "start_s": 1309.0, "end_s": 1403.0},
           {"file": "MAH08539.MP4", "start_s": 0.0,    "end_s": 141.0}],
 "index": 8, "file": "08-MAH08538-4512.flac"
}
```

Detection runs at 22.05 kHz; the clips are written at 44.1 kHz so they stay worth listening to when
you check a boundary by ear. **Check them.** The segmenter reports where the music was, not what it
was called: matching songs to a setlist is yours to do, and it is the moment a wrong boundary
becomes obvious.

## Video files

`segment` accepts video containers — `.mp4`, `.mov`, `.mkv`, `.m4v`, `.avi`, `.mts`, `.m2ts` —
because concerts usually arrive as video. They are decoded by **ffmpeg**, which is not a package
dependency: it is asked for only when a video file is actually handed over, and its absence is
reported as a missing program rather than as a decode failure.

Collections stay audio-only. A folder of films is not an album, and letting `open_collection` pick
video up would change what every other verb sees.

## Reading the result

The two gates in [Features](analysis.md) matter more here than anywhere else, and they need care
rather than obedience.

`pulse_R` folds a whole song at *one* global period. A live band drifts, so a four-minute concert
take can read below the 0.1 threshold while having a perfectly steady beat — the drift, not the
absence of pulse, is what lowers the resultant.

`chroma_entropy` is computed on a whole-song mean chroma. A full band in a reverberant room flattens
that average far more than the solo instrumental material the threshold was calibrated on, so a
concert can sit near the ceiling song after song.

[`musiscape.stability`](../api.md) measures both quantities per 20-second window instead, and
`tempo_agreement` / `key_agreement` travel beside the gated numbers in `features.json`. On the
reference concert seven of eight windowed key estimates agreed with the whole-song one that the gate
had called unreliable. The **keyscape** card says the same thing visually: a genuine tonal centre
paints one colour across the whole triangle, an artefact paints a patchwork.

### There is no pulse gate, and that is a finding

Nothing here reports whether a track *has* a beat, because two candidates were measured against this
concert and both failed:

| measure | songs | applause & room tone |
|---|---|---|
| beat salience (onset strength at tracked beats / track mean) | 1.58 – 1.93 | 1.50 – 1.67 |
| tempogram peak prominence | 1.09 – 1.62 | 1.07 – 1.24 |

Both look decisive on synthetic material — a click train scores above 13 on the first, white noise
about 1.2 — and both overlap on the real thing. The reason is not a defective measure. **Applause is
rhythmic**: a room clapping in near-unison has a periodic onset envelope, and no statistic of
periodicity alone will separate it from a band. That is exactly why `segment` sorts music from
applause by spectral flatness instead.

So for a single track, read the tempogram. `musiscape figures --width 1920` draws it with a labelled
BPM axis: a bright band holding level across the width is a steady tempo, and one that bends is a
band speeding up.
