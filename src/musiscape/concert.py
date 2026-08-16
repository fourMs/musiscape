"""Songs out of a continuous recording---the concert, not the collection.

The rest of musiscape assumes one file is one track. A concert is the other
shape: one long recording (often several, when the camera split at a file
size limit) holding a sequence of songs separated by applause, tuning and
talk. This module finds those songs so the collection tools can be pointed
at them.

What separates a song from the space around it here is not level alone---an
enthusiastic room is as loud as the band---but **spectral flatness**.
Applause is broadband noise and measures flat; played music is tonal and
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
#: Otsu will still cut it in half if allowed to.
BIMODAL_MIN_DECADES = 0.7

#: Minimum share of frames on the smaller side of the cut. Separation alone
#: does not make a mode: one stray frame far from an otherwise single spike
#: separates the class means beautifully while the threshold itself lands
#: inside the spike, splitting a continuous recording down the middle.
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


def music_mask(y: np.ndarray, sr: int, hop_s: float = 1.0) -> np.ndarray:
    """Per-``hop_s`` boolean: is this frame played music?

    Frames are kept when they are both tonal (spectral flatness below a
    threshold taken from the recording's own bimodal distribution) and
    audible (within :data:`LEVEL_FLOOR_DB` of the recording's loud frames).
    When the flatness distribution has only one mode---a recording that is
    music throughout, or applause throughout---the flatness test is dropped
    rather than invented, and level alone decides.
    """
    import librosa

    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = librosa.feature.spectral_flatness(S=S)[0]
    rms = librosa.feature.rms(S=S)[0]

    n = max(1, int(round(hop_s * sr / hop)))
    m = len(flat) // n
    if m == 0:
        return np.zeros(0, dtype=bool)
    logflat = np.array([np.log10(flat[i * n:(i + 1) * n].mean() + 1e-12)
                        for i in range(m)])
    db = librosa.amplitude_to_db(
        np.array([rms[i * n:(i + 1) * n].mean() for i in range(m)]) + 1e-12)

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

    Runs of music separated by less than ``min_gap_s`` are one song---a
    quiet bar or a held breath is not the end of a piece---and spans
    shorter than ``min_song_s`` are not songs at all, which is what keeps
    tuning, a spoken introduction over a held chord, or a false start out
    of the listing.

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

    ``paths`` must be in playing order. Times are reported on a **concert
    clock** that runs from the start of the first file and treats the files
    as butted together: a camera that stops and restarts loses a few
    seconds at each join, and that loss is not recoverable from the audio,
    so the clock drifts behind wall time by however long the changeovers
    took. Within a song, ``parts`` carries the true offsets into each
    source file, which is what the clips are cut from.

    A span reaching the end of one file and resuming at the start of the
    next is one song: the camera splits at a size limit, not at a musical
    boundary. The minimum-length test is applied *after* that join, so a
    song cut ten seconds before its end is not discarded as a fragment.
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
    """``03-MAH08538-1204.flac``---order, source file, concert timecode."""
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
