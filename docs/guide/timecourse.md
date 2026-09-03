# The time-course of a recording, and what is in front

[Concerts & long recordings](concerts.md) finds the songs inside a live set. This page is for the
recording that has no songs to find: a long improvisation, a concert of one continuous piece, a
session in which the interesting question is *when the music moves, and in what*. A whole-track key
or tempo says nothing useful about such material. A per-second view does.

## `music_timecourse`

```python
import musiscape as ms
from musiscape.io import load_recording

y, sr = load_recording("concert.wav", sr=22050)
tc = ms.music_timecourse(y, sr)        # or ms.timecourse.music_timecourse
tc["t"]                                # one row per second
tc["keys"]                             # [(t, "C minor", 0.71), ...] per 30 s window
tc["register_midi"], tc["pulse_clarity"], tc["timbre_novelty"]
```

Everything is at one frame per second, on one clock, so the arrays sit beside a motion or gaze
track from another toolbox and can be correlated, segmented and drawn on the same width. What is
computed:

| key | what it is |
|---|---|
| `chroma` (12 × n) | chroma of the harmonic component (HPSS first), normalised per second |
| `chroma_entropy`, `tonal_clarity` | harmonic complexity; how far one pitch class stands out |
| `keys` | windowed key by correlation with the Krumhansl–Kessler profiles: `(t, name, r)` |
| `hcdf` | harmonic change, the tonnetz distance between consecutive seconds |
| `tempogram`, `tempi`, `local_tempo`, `pulse_clarity` | onset-strength autocorrelation, its argmax, and its peak-to-mean ratio |
| `tempo_windows` | beat-tracked tempo per 30 s window |
| `mfcc`, `centroid`, `flatness`, `harmonic_ratio` | timbre |
| `register_midi`, `register_spread` | energy-weighted pitch of the harmonic constant-Q spectrum, and its spread |
| `timbre_novelty` | Foote novelty on the MFCC self-similarity |

Read `pulse_clarity` before reading `local_tempo`. On non-metric music the tempogram is diffuse
and the argmax is a report of the absence of a pulse, not of one; the function does not pretend
otherwise. A key that changes every window is a statement about the music (modal, wandering), not
a failure of the estimate.

## Sections and figures

```python
rows = ms.timecourse.section_summary(tc, boundaries=[102, 294, 478])   # seconds
paths = ms.timecourse.timecourse_figures(tc, "analysis", prefix="concert_", title="Concert",
                                         boundaries=[102, 294, 478])
```

`section_summary` folds the time-course into one row per section: key and its correlation,
median tempo, pulse clarity, harmonic complexity and change, brightness, flatness, harmonic share,
register and spread, level. `timecourse_figures` writes the chromagram with the windowed keys, the
tempogram with the windowed tempo, a five-panel time-course, and two *raw strips* (chromagram and
tempogram without axes) for a scrollable page that lays music beside movement.

From the command line, `musiscape timecourse <folder-of-recordings>` does all of this for every
file.

## What is in front: `instrument_foreground`

A single microphone gives one mixture. When the question is "what happened when the violin led,
and when the electronics led", the honest answer without stems is a set of *proxies*:

```python
fg = ms.instrument_foreground(y, sr)     # or ms.foreground.instrument_foreground
fg["pitched"], fg["low"], fg["noise"]     # 0–1 per second
labels = ms.foreground.foreground_labels(fg)   # "pitched" | "low" | "noise" | "mixed" | "quiet"
```

- **pitched** — harmonic energy between 180 Hz and 4 kHz, weighted by the voicing probability of
  a pitch tracker (pYIN) on the harmonic component. A bowed string, a voice or a wind instrument
  playing notes scores high; so does a synthesiser with a clear pitch.
- **low** — energy below 180 Hz.
- **noise** — the percussive/residual component's energy, times spectral flatness.

`foreground_labels` gives one label per second only where one proxy clearly leads (a margin of
0.15 by default); the rest is `"mixed"`, and seconds below −55 dBFS are `"quiet"`. Expect most of
a dense mixture to be `"mixed"`; the continuous proxies are the useful output, the labels are a
summary of their clearest moments.

These are proxies, not a separation. On the live-painting concert they were written for, the
pitched proxy followed the violin and the low and noise proxies the electronics, and that reading
was checked against photographs and an AudioSet tagger before it was used. Check yours the same
way. If stems exist, use them.

## Notes, not onsets: `transcribe_piano`

For a piano recording, an onset detector is a loss: it fires on attacks without saying how many
notes, at what pitch, how loud. `musiscape.transcribe` transcribes the piano part to note events
with the high-resolution model of Kong et al. (2021), as an optional extra:

```bash
pip install "musiscape[transcribe]"      # brings piano_transcription_inference and torch
musiscape transcribe recordings/         # <name>_notes.csv, _notes.mid, _notes_1hz.csv
```

```python
notes = ms.transcribe_piano(y, sr)        # onset_s, offset_s, midi, velocity per note
per = ms.transcribe.notes_per_second(notes, duration_s=len(y) / sr)
per["density"], per["pitch_mean"], per["velocity_mean"], per["pitch_spread"], per["sustain"]
```

The checkpoint (about 170 MB) downloads on first use; on a CPU the model runs at a few times real
time. It is for piano, and for recordings the piano dominates: on a mixture it returns the notes
it believes it hears. On a piano improvisation the signal's onset detector had found 32 to 83
events per minute where the transcription hears 58 to 300, because chords and runs merge into
single attacks; the note onsets are what strokes and gestures should be aligned to, and the
per-second density, pitch and velocity are the piano's own time-course to lay beside the one
above.
