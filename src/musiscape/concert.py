"""Songs out of a continuous recording: the concert, not the collection.

The rest of musiscape assumes one file is one track. A concert is the other
shape: one long recording, often several when the camera split at a file
size limit, holding a sequence of songs separated by applause, tuning and
talk. This module finds those songs so the collection tools can be pointed
at them.

What separates a song from the space around it is not level. An
enthusiastic room is as loud as the band. It is spectral flatness: applause
is broadband noise and measures flat, while played music is tonal and
measures peaked, typically an order of magnitude lower. The split between
the two is taken from each recording's own distribution rather than from a
fixed number, because the ratio survives across rooms and microphones while
the absolute values do not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import load_recording

#: Minimum separation (log10 units) between the two flatness modes before
#: the split is believed. A recording that is all music has one mode, and
#: Otsu would still cut it in half if allowed to.
BIMODAL_MIN_DECADES = 0.7

#: Minimum share of frames on the smaller side of the cut. Separation alone
#: does not make a mode: one stray frame far from an otherwise single spike
#: separates the class means while the threshold itself lands inside the
#: spike, splitting a continuous recording down the middle.
BIMODAL_MIN_SHARE = 0.05

#: Level floor, in dB below the recording's 95th percentile frame. Silence
#: and distant room tone fall out here; the flatness test does the rest.
LEVEL_FLOOR_DB = 15.0


def _otsu(x: np.ndarray, bins: int = 128) -> tuple[float, float, float]:
    """Otsu's threshold on ``x``, with the two class means.

    The threshold is the *upper edge* of the winning bin, not its centre:
    a distribution with a sharp spike puts many samples at one value, and
    a centre-valued threshold cuts that single spike in half.
    """
    hist, edges = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    csum = np.cumsum(p * centers)
    m0 = csum / np.maximum(w0, 1e-12)
    m1 = (csum[-1] - csum) / np.maximum(w1, 1e-12)
    i = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    return float(edges[i + 1]), float(m0[i]), float(m1[i])


def _pool_slices(n_samples: int, sr: int, hop: int, hop_s: float,
                 n_frames: int):
    """One slice of STFT frames per ``hop_s`` of audio, on a true clock.

    Pooling a *rounded* number of frames is the obvious way to do this and
    it makes the clock drift. At 22.05 kHz with a 512 hop, one second is
    43.07 hops; rounding to 43 makes each pooled frame 0.99846 s while the
    caller is told it is 1.0 s, so every reported time runs 0.15 % late.
    That is two seconds by the end of a 23-minute camera file, which is
    enough for a span to end after the file it names.

    Boundaries are therefore computed from time rather than accumulated,
    and the count comes from the audio's real duration.
    """
    m = int(n_samples / sr / hop_s)
    if m <= 0 or n_frames <= 0:
        return 0, []
    out = []
    for i in range(m):
        a = min(int(round(i * hop_s * sr / hop)), n_frames - 1)
        b = min(max(int(round((i + 1) * hop_s * sr / hop)), a + 1), n_frames)
        out.append(slice(a, b))
    return m, out


def music_mask(y: np.ndarray, sr: int, hop_s: float = 1.0) -> np.ndarray:
    """Per-``hop_s`` boolean: is this frame played music?

    Frames are kept when they are both tonal (spectral flatness below a
    threshold taken from the recording's own bimodal distribution) and
    audible (within :data:`LEVEL_FLOOR_DB` of the recording's loud frames).
    When the flatness distribution has only one mode, as in a recording
    that is music throughout or applause throughout, the flatness test is
    dropped rather than invented and level alone decides.
    """
    import librosa

    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = librosa.feature.spectral_flatness(S=S)[0]
    rms = librosa.feature.rms(S=S)[0]

    m, sl = _pool_slices(len(y), sr, hop, hop_s, len(flat))
    if m == 0:
        return np.zeros(0, dtype=bool)
    logflat = np.array([np.log10(flat[s].mean() + 1e-12) for s in sl])
    db = librosa.amplitude_to_db(
        np.array([rms[s].mean() for s in sl]) + 1e-12)

    loud = db > np.percentile(db, 95) - LEVEL_FLOOR_DB
    cut, lo, hi = _otsu(logflat)
    tonal = logflat < cut
    share = min(tonal.mean(), 1.0 - tonal.mean())
    if hi - lo < BIMODAL_MIN_DECADES or share < BIMODAL_MIN_SHARE:
        return loud
    return loud & tonal


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index spans of the True runs in ``mask``."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def segments(mask: np.ndarray, hop_s: float = 1.0,
             min_song_s: float = 60.0,
             min_gap_s: float = 8.0) -> list[dict]:
    """Turn a music mask into song spans.

    Runs of music separated by less than ``min_gap_s`` are one song, since
    a quiet bar or a held breath is not the end of a piece. Spans shorter
    than ``min_song_s`` are not songs at all, which keeps tuning, a spoken
    introduction over a held chord, and a false start out of the listing.

    Returns one dict per song with ``start_s``, ``end_s`` and
    ``duration_s``, in time order.
    """
    spans = _runs(np.asarray(mask, dtype=bool))
    if not spans:
        return []

    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if (a - merged[-1][1]) * hop_s < min_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    out = []
    for a, b in merged:
        dur = (b - a) * hop_s
        if dur < min_song_s:
            continue
        out.append({"start_s": round(a * hop_s, 2),
                    "end_s": round(b * hop_s, 2),
                    "duration_s": round(dur, 2)})
    return out


#: How close to a file boundary music must run before a span is treated as
#: continuing into (or from) the neighbouring file.
JOIN_TOL_S = 5.0


def find_songs(paths, sr: int = 22050, hop_s: float = 1.0,
               min_song_s: float = 60.0, min_gap_s: float = 8.0,
               join_tol_s: float = JOIN_TOL_S) -> list[dict]:
    """Locate the songs across an ordered sequence of recording files.

    ``paths`` must be in playing order. Times are reported on a concert
    clock that runs from the start of the first file and treats the files
    as butted together: a camera that stops and restarts loses a few
    seconds at each join, and that loss is not recoverable from the audio,
    so the clock drifts behind wall time by however long the changeovers
    took. Within a song, ``parts`` carries the true offsets into each
    source file, which is what the clips are cut from.

    A span reaching the end of one file and resuming at the start of the
    next is one song, because the camera splits at a size limit rather than
    at a musical boundary. The minimum-length test is applied after that
    join, so a song cut ten seconds before its end is not discarded as a
    fragment.
    """
    paths = [Path(p) for p in paths]
    songs: list[dict] = []
    offset, prev_open = 0.0, False

    for p in paths:
        y, _sr = load_recording(p, sr=sr)
        dur = len(y) / _sr
        spans = segments(music_mask(y, _sr, hop_s), hop_s,
                         min_song_s=0.0, min_gap_s=min_gap_s)
        for j, s in enumerate(spans):
            part = {"file": p.name, "start_s": s["start_s"],
                    "end_s": s["end_s"]}
            if j == 0 and prev_open and s["start_s"] <= join_tol_s:
                songs[-1]["parts"].append(part)
                songs[-1]["end_s"] = round(offset + s["end_s"], 2)
                songs[-1]["duration_s"] = round(
                    songs[-1]["duration_s"] + s["duration_s"], 2)
            else:
                songs.append({"start_s": round(offset + s["start_s"], 2),
                              "end_s": round(offset + s["end_s"], 2),
                              "duration_s": s["duration_s"],
                              "parts": [part]})
        prev_open = bool(spans) and (dur - spans[-1]["end_s"]) <= join_tol_s
        offset += dur

    songs = [s for s in songs if s["duration_s"] >= min_song_s]
    for i, s in enumerate(songs, start=1):
        s["index"] = i
    return songs


def _clip_name(song: dict) -> str:
    """``03-MAH08538-1204.flac``: order, source file, concert timecode."""
    stem = Path(song["parts"][0]["file"]).stem
    t = int(song["start_s"])
    return f"{song['index']:02d}-{stem}-{t // 60:02d}{t % 60:02d}.flac"


def split_recording(paths, out_dir: str | Path, sr: int = 22050,
                    write_sr: int = 44100, hop_s: float = 1.0,
                    min_song_s: float = 60.0, min_gap_s: float = 8.0,
                    join_tol_s: float = JOIN_TOL_S) -> Path:
    """Cut a concert into one audio file per song, plus a manifest.

    Writes ``<out_dir>/songs/NN-<source>-<mmss>.flac`` and returns the path
    to ``<out_dir>/songs.json``. The songs folder is an ordinary musiscape
    collection: point ``report``, ``thumbnails`` or any other verb at it.

    Detection runs at ``sr``; the clips are written at ``write_sr`` so they
    stay worth listening to when you check a boundary by ear.
    """
    out_dir = Path(out_dir)
    songs_dir = out_dir / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)

    songs = find_songs(paths, sr=sr, hop_s=hop_s, min_song_s=min_song_s,
                       min_gap_s=min_gap_s, join_tol_s=join_tol_s)
    by_name = {Path(p).name: Path(p) for p in paths}

    import soundfile as sf
    for song in songs:
        pieces = []
        for part in song["parts"]:
            y, _ = load_recording(by_name[part["file"]], sr=write_sr,
                                  offset=part["start_s"],
                                  duration=part["end_s"] - part["start_s"])
            pieces.append(y)
        song["file"] = _clip_name(song)
        sf.write(songs_dir / song["file"], np.concatenate(pieces), write_sr)

    manifest = out_dir / "songs.json"
    manifest.write_text(json.dumps(songs, indent=1))
    return manifest


# --------------------------------------------------------------------------
# What the recording is doing, second by second.
#
# Four states account for nearly all of a concert recording, and they
# separate on three cheap spectral measures. Measured over one 53-minute
# concert, per frame:
#
#                    level      log flatness   flatness      centroid
#                                   (mean)    variability
#   music           -21 dB         -3.1          0.57         1100 Hz
#   applause        -27 to -34     -1.4          0.36         2400 Hz
#   voices          -28            -1.9          0.46         2100 Hz
#   room tone       -34 to -39     -1.6 to -1.7  0.24-0.29    2000 Hz
#
# Music is tonal and dark; applause is the flattest and brightest thing in
# the room; room tone is the least variable. Voices sit between applause and
# music in flatness and above both in variability, because speech swings
# between voiced and unvoiced several times a second where a clapping room
# is stationary noise. A syllable-rate (4-8 Hz) modulation measure was tried
# first and separates none of them on this material.

#: Labels returned by :func:`classify_regions`, in reporting order.
REGION_CLASSES = ("music", "applause", "voices", "quiet", "other")

#: Flatness (log10) a frame must be below to count as music here, on top of
#: the adaptive test in :func:`music_mask`. That test compares a recording
#: against itself, so a recording holding nothing but applause has one mode
#: and falls back to level, which would call the whole room music. The
#: ceiling sits between the music and voices measurements above.
MUSIC_FLATNESS = -2.2

#: Flatness above this (log10) is broadband enough to be applause rather
#: than voices, given neither is tonal enough to be music.
APPLAUSE_FLATNESS = -1.62

#: Flatness variability above this marks the voiced/unvoiced alternation of
#: speech against the stationary noise of a room or a crowd.
VOICE_VARIABILITY = 0.40

#: Level floor for "something is happening", in dB below the recording's
#: 95th-percentile frame. Below it the frame is room tone.
QUIET_FLOOR_DB = 12.0


def region_features(y: np.ndarray, sr: int, hop_s: float = 1.0) -> dict:
    """Per-``hop_s`` level, flatness, flatness variability and centroid."""
    import librosa

    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = np.log10(librosa.feature.spectral_flatness(S=S)[0] + 1e-12)
    rms = librosa.feature.rms(S=S)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]

    m, sl = _pool_slices(len(y), sr, hop, hop_s, len(flat))
    if m == 0:
        return {k: np.zeros(0) for k in
                ("db", "flatness", "flat_var", "centroid")}
    return {
        "db": librosa.amplitude_to_db(
            np.array([rms[s].mean() for s in sl]) + 1e-12),
        "flatness": np.array([flat[s].mean() for s in sl]),
        "flat_var": np.array([flat[s].std() for s in sl]),
        "centroid": np.array([cent[s].mean() for s in sl]),
    }


def classify_regions(y: np.ndarray, sr: int, hop_s: float = 1.0) -> np.ndarray:
    """Label every ``hop_s`` frame with one of :data:`REGION_CLASSES`.

    Music must pass :func:`music_mask` and be tonal in absolute terms, since
    that function compares a recording against itself and has no way to tell
    an all-applause recording from an all-music one. The rest is sorted by
    flatness and by how much that flatness moves.

    ``other`` is not a dustbin for what is left over; it is what the frame
    gets when it is audible but matches no class cleanly, and it should be
    read as the classifier declining to guess.
    """
    f = region_features(y, sr, hop_s)
    n = len(f["db"])
    if n == 0:
        return np.array([], dtype=object)

    labels = np.full(n, "other", dtype=object)
    music = music_mask(y, sr, hop_s)
    music = np.resize(music, n) if len(music) != n else music

    audible = f["db"] > np.percentile(f["db"], 95) - QUIET_FLOOR_DB
    labels[~audible] = "quiet"
    labels[audible & (f["flat_var"] >= VOICE_VARIABILITY)] = "voices"
    labels[audible & (f["flatness"] >= APPLAUSE_FLATNESS)
           & (f["flat_var"] < VOICE_VARIABILITY)] = "applause"
    labels[music & (f["flatness"] < MUSIC_FLATNESS)] = "music"
    return labels


def regions(labels, hop_s: float = 1.0, min_s: float = 5.0) -> list[dict]:
    """Merge a label sequence into spans, dropping flickers.

    A span shorter than ``min_s`` is absorbed into whichever neighbour it
    interrupts, because a second or two of a different label mid-song is the
    classifier wobbling rather than an event in the hall.
    """
    labels = list(labels)
    if not labels:
        return []

    spans = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.append([labels[start], start, i])
            start = i

    changed = True
    while changed and len(spans) > 1:
        changed = False
        for i, (_lab, a, b) in enumerate(spans):
            if (b - a) * hop_s >= min_s:
                continue
            # absorb into the longer neighbour, then re-merge equal labels
            prev = spans[i - 1] if i > 0 else None
            nxt = spans[i + 1] if i + 1 < len(spans) else None
            take = prev if (nxt is None or
                            (prev is not None and
                             (prev[2] - prev[1]) >= (nxt[2] - nxt[1]))) else nxt
            if take is None:
                continue
            take[1], take[2] = min(take[1], a), max(take[2], b)
            spans.pop(i)
            merged = [spans[0]]
            for s in spans[1:]:
                if s[0] == merged[-1][0]:
                    merged[-1][2] = s[2]
                else:
                    merged.append(s)
            spans = merged
            changed = True
            break

    return [{"label": lab, "start_s": round(a * hop_s, 2),
             "end_s": round(b * hop_s, 2),
             "duration_s": round((b - a) * hop_s, 2)}
            for lab, a, b in spans]


def _span_name(span: dict, start_time=None, suffix: str = ".flac") -> str:
    """Filename for one exported span.

    With a known recording start this leads with ``YYYYMMDD_HHMMSS``, which
    is the stamp recorders write and the one a soundscape tool reads to put
    the spans on a real clock. Without one it falls back to concert
    timecodes, which at least keep the folder in playing order.
    """
    a, b = int(span["start_s"]), int(span["end_s"])
    if start_time is not None:
        import datetime as _dt
        when = start_time + _dt.timedelta(seconds=a)
        return f"{when:%Y%m%d_%H%M%S} {span['label']}{suffix}"
    return (f"{a // 60:02d}{a % 60:02d}-{b // 60:02d}{b % 60:02d} "
            f"{span['label']}{suffix}")


def map_regions(paths, sr: int = 22050, hop_s: float = 1.0,
                min_s: float = 5.0, songs=None) -> dict:
    """Label a whole concert, on the concert clock.

    Returns ``{"spans": [...], "total_s": float, "level_db": [...]}``, where
    the spans run continuously from the first file to the last and the level
    is one value per ``hop_s``. Files are butted together exactly as
    :func:`find_songs` butts them, so the two share a clock.

    Pass ``songs`` (what :func:`find_songs` returned) to let the setlist
    decide where the music is. It bridges a gap of a few seconds mid-song
    and the frame classifier does not, so without this a single song with a
    quiet bar in it is drawn as two or three. Everything outside the songs
    is still classified frame by frame.
    """
    paths = [Path(p) for p in paths]
    labels, level = [], []
    for p in paths:
        y, _sr = load_recording(p, sr=sr)
        labels.extend(classify_regions(y, _sr, hop_s))
        level.extend(region_features(y, _sr, hop_s)["db"])

    labels = np.array(labels, dtype=object)
    if songs is not None:
        labels[labels == "music"] = "other"
        for song in songs:
            a = int(round(song["start_s"] / hop_s))
            b = min(int(round(song["end_s"] / hop_s)), len(labels))
            labels[max(a, 0):b] = "music"

    return {"spans": regions(labels, hop_s, min_s),
            "total_s": round(len(labels) * hop_s, 2),
            "level_db": [round(float(v), 2) for v in level]}


def export_regions(paths, out_dir: str | Path, spans, sr: int = 44100,
                   exclude=("music",), hop_s: float = 1.0,
                   start_time=None, suffix: str = ".flac") -> Path:
    """Write every non-music span as its own file, for a soundscape tool.

    musiscape describes music; what happens between the songs is a
    soundscape question, and ambiscape is the toolbox for those. The two
    meet at the file boundary rather than by importing one another, so this
    writes an ordinary folder of WAVs that ``ambiscape analyze`` reads as
    one session.

    Spans are cut from the concert clock, so one crossing a file boundary is
    assembled from both files. FLAC by default: lossless, about half the
    size of WAV, and read natively by the tools on both sides.

    ``start_time`` puts the recording's wall clock into the filenames, which
    is what lets the other tool lay the evening out on a timeline instead of
    stacking every span at the same second.
    """
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in paths]

    bounds, offset = [], 0.0
    for p in paths:
        y, _sr = load_recording(p, sr=22050)
        dur = len(y) / _sr
        bounds.append((p, offset, offset + dur))
        offset += dur

    for span in spans:
        if span["label"] in exclude:
            continue
        pieces = []
        for p, a, b in bounds:
            lo, hi = max(span["start_s"], a), min(span["end_s"], b)
            if hi - lo <= 0.05:
                continue
            y, _ = load_recording(p, sr=sr, offset=lo - a, duration=hi - lo)
            pieces.append(y)
        if pieces:
            sf.write(out_dir / _span_name(span, start_time, suffix),
                     np.concatenate(pieces), sr)
    return out_dir
