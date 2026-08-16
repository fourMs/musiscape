# Concerts & long recordings

Everything else in musiscape assumes one file is one track. A concert is the other shape: one long
recording holding a sequence of songs, often split across several files because the camera stopped
at a size limit partway through the evening.

Analysed as it arrives, such a recording averages a whole evening into a single key and a single
tempo, which is to say into nothing. `segment` finds the songs inside it first.

```
musiscape segment ~/video/concert -o ~/video/concert/analysis
musiscape report  ~/video/concert/analysis/songs
```

The first command writes one audio file per song plus a manifest. The second treats that folder as
an ordinary collection, and so does every other verb: `thumbnails --style keyscape` for tonality,
`figures` for a labelled tempogram, `pdf` for the whole set in one document.

## What separates a song from the space around it

Not level. An enthusiastic room is as loud as the band, and a hall between songs is not silent.

Spectral flatness is what separates them. Applause is broadband noise and measures flat; played
music is tonal and measures roughly an order of magnitude lower. On a concert recorded to a camera's
built-in microphone:

| | level | spectral flatness |
|---|---|---|
| songs | ≈ −20 dB | 0.001 to 0.006 |
| applause, talk, room | −27 to −40 dB | 0.02 to 0.07 |

The two form distinct modes, so the threshold between them is taken from each recording's own
distribution by Otsu's method on log-flatness rather than from a fixed number. The ratio survives a
change of room and microphone; the absolute values do not.

Two conditions keep that split honest. Otsu returns a threshold for any distribution, including one
with a single mode, so the flatness test is used only when the two classes are both separated
(`BIMODAL_MIN_DECADES`) and populated (`BIMODAL_MIN_SHARE`). A rehearsal tape that is music
throughout has one mode and falls back to a level test instead of being cut in half. The threshold
itself is the winning histogram bin's upper edge rather than its centre, because a sharp spike puts
many frames at one value and a centre-valued cut would divide the spike.

A level floor, set relative to the recording's own loud frames, drops silence and distant room tone
underneath all of this.

## Turning a mask into songs

Runs of music separated by less than `--min-gap` seconds are one song: a quiet bar or a held breath
does not end a piece. Spans shorter than `--min-song` are not songs at all, which keeps tuning, a
false start, and a spoken introduction over a held chord out of the listing.

| option | default | meaning |
|---|---|---|
| `--min-song` | 60 s | shortest span counted as a song |
| `--min-gap` | 8 s | shortest break that ends a song |

## Several files, one concert

Files are read in name order, which is playing order, since cameras number sequentially. A span that
runs to the end of one file and resumes at the start of the next is one song, because the camera
split at a size limit rather than at a musical boundary. That join is applied before the
minimum-length test, so a song cut ten seconds before its end survives as a song instead of being
discarded as a fragment. Its clip holds both halves, back to back.

Times are reported on a concert clock that runs from the start of the first file and treats the
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
you check a boundary by ear. Check them. The segmenter reports where the music was, not what it was
called, so matching songs to a setlist is yours to do, and it is the moment a wrong boundary becomes
obvious.

## The timeline

`segment` also labels every second of the recording and draws it as one waveform, coloured by what
was happening, so a whole evening fits on one strip:

```
analysis/
  regions.json            every span, labelled
  timeline.png            the coloured waveform, at --width pixels
```

Five labels are used. `music` comes from the setlist, so the ribbon and the song listing always
agree: `find_songs` bridges a quiet bar mid-song and a frame classifier does not, and without this
one song is drawn as three. The rest is decided frame by frame:

| label | what it is | how it is told apart |
|---|---|---|
| `music` | a piece being played | taken from the setlist |
| `applause` | the room clapping | the flattest and brightest thing in the hall |
| `voices` | someone talking | flatness that swings, as speech alternates voiced and unvoiced |
| `quiet` | room tone | below the level floor |
| `other` | audible, but nothing above fits | the classifier declining to guess |

`other` is worth reading as a real answer rather than a leftover. It marks the places where the
material is genuinely ambiguous, which on a concert recording is usually a song ending, or a stage
being reset.

The colour sits on the waveform rather than in a separate band above it, so one lane carries both
what happened and how loud it was. An applause swell dying away looks different from a block saying
only that applause occurred.

These thresholds were calibrated on one 53-minute concert recorded to a camera's built-in
microphone; the measurements behind them are in the source, and they are module constants so they
can be moved.

## Handing the rest to a soundscape toolbox

What happens between the songs is a soundscape question rather than a music one, and
[ambiscape](https://github.com/fourMs/ambiscape) is the toolbox for those. The two meet at the file
boundary rather than by importing one another, so `segment` writes every non-music span into a
folder that `ambiscape analyze` reads as one session:

```
musiscape segment  ~/video/concert -o ~/video/concert/analysis
ambiscape analyze  ~/video/concert/analysis/other
```

Two things make that handover work. The files are FLAC, which is lossless, about half the size of
WAV, and read natively on both sides. And each name leads with a `YYYYMMDD_HHMMSS` stamp taken from
the recording's own start time, which is the convention recorders use and the one ambiscape reads;
without it every span lands at the same second and the session has no clock.

The start time comes from the container's `creation_time` where there is one, else from a stamp in
the source filename. Video containers store that tag in UTC and it is converted to local time, since
what makes a session clock readable is the wall time of the room. Where neither exists the names
fall back to concert timecodes, which at least keep the folder in playing order.

## Video files

`segment` accepts video containers (`.mp4`, `.mov`, `.mkv`, `.m4v`, `.avi`, `.mts`, `.m2ts`),
because concerts usually arrive as video. They are decoded by ffmpeg, which is not a package
dependency: it is asked for only when a video file is actually handed over, and its absence is
reported as a missing program rather than as a decode failure.

Collections stay audio-only. A folder of films is not an album, and letting `open_collection` pick
video up would change what every other verb sees.

## Reading a concert's key and tempo

The two gates described in [Features](analysis.md) need care here rather than obedience. Both are
computed on whole-song statistics, and a four-minute live take is the case they were not calibrated
on.

`pulse_R` folds a whole song at one global period. A live band drifts, so the resultant collapses
even where the beat is steady, and a concert can read below the 0.1 threshold song after song.

`chroma_entropy` is computed on a whole-song mean chroma. A full band in a reverberant room flattens
that average far more than the solo instrumental material the threshold was calibrated on, so a
concert can sit near the log2(12) ceiling throughout.

`musiscape.stability` measures both quantities per 20-second window instead, and `key_agreement` and
`tempo_agreement` travel beside the gated numbers in `features.json`. They answer the narrower
question that can be answered: not whether the music is tonal or pulsed, but whether the estimate
holds still across the track. The keyscape card shows the same thing visually, since a genuine tonal
centre paints one colour across the whole triangle where an artefact paints a patchwork.

### There is no pulse gate

No descriptor here reports whether a track has a beat, because on live material no statistic of
periodicity can tell one. Applause is rhythmic: a room clapping in near-unison has a periodic onset
envelope, and measures of beat strength or tempogram peak prominence put it in the same range as the
songs it sits between. That is why `segment` separates music from applause by spectral flatness
instead.

For a single track, read the tempogram. `musiscape figures --width 1920` draws it with a labelled
BPM axis, where a bright band holding level across the width is a steady tempo and one that bends is
a band speeding up or slowing down.
