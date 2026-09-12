"""Concert segmentation from AudioSet posteriors (PANNs through ambiscape).

:mod:`musiscape.concert` labels a concert from spectral flatness and level,
which costs nothing per frame and was calibrated on one camera-mic concert. It
mistakes loud rock for applause and hears noise music as "other". This module
does the same job from a sound-event tagger: `ambiscape.ml.tag_frames` gives
AudioSet posteriors every couple of seconds, and the chain below turns them
into the same ``spans`` vocabulary (``music``, ``applause``, ``voices``,
``quiet``, ``other``) so :func:`musiscape.figures.concert_timeline` and
:func:`musiscape.concert.export_regions` work unchanged.

Needs ``ambiscape[ml]`` (PANNs, torch). Roughly a minute for a 90-minute
concert on a laptop GPU (``device="auto"``), half an hour on CPU.
"""
from __future__ import annotations

import numpy as np

from .concert import REGION_CLASSES

#: AudioSet labels pooled (by max) into each region class. Choir, singing and
#: the common instruments are in ``music`` because "Music" itself sits low on
#: a capella singing and on solo instruments heard from a room microphone.
GROUPS: dict[str, list[str]] = {
    "music": ["Music", "Musical instrument", "Singing", "Choir", "A capella", "Piano", "Guitar",
              "Synthesizer", "Electronic music", "Drum kit", "Bass guitar", "Violin, fiddle"],
    "voices": ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
               "Narration, monologue", "Conversation"],
    "applause": ["Applause", "Clapping", "Cheering", "Crowd"],
    "quiet": ["Silence"],
}

#: Applause is short and far from the microphone, so its posterior rarely
#: reaches the music or speech level of the frames around it.
WEIGHTS = {"music": 1.0, "voices": 1.0, "applause": 1.6, "quiet": 1.0}
QUIET_DBFS = -60.0     #: frame RMS below this is ``quiet`` whatever the tagger says
OTHER_FLOOR = 0.25     #: nothing above this after weighting -> ``other``
SMOOTH_FRAMES = 5      #: majority filter width (frames)
MIN_DURATION_S = {"music": 20.0, "voices": 6.0, "applause": 4.0, "quiet": 10.0, "other": 6.0}
SNAP_S = 6.0           #: snap music edges to :func:`concert.find_songs` boundaries this close


def tag_frames(y: np.ndarray, sr: int, win_s: float = 4.0, hop_s: float = 2.0, device=None):
    """``(times, probs, names)`` from ``ambiscape.ml.tag_frames``; a clear error without the extra."""
    try:
        from ambiscape import ml
    except ImportError as e:  # pragma: no cover
        raise ImportError("musiscape.tagging needs ambiscape with the [ml] extra "
                          "(pip install 'ambiscape[ml]')") from e
    return ml.tag_frames(np.asarray(y, dtype=np.float32), sr, win_s=win_s, hop_s=hop_s, device=device)


def frame_level_db(y: np.ndarray, sr: int, times: np.ndarray, win_s: float) -> np.ndarray:
    """RMS in dBFS of the window centred on each time."""
    out = np.full(len(times), -99.0)
    half = int(win_s * sr / 2)
    for k, t in enumerate(times):
        a, b = max(0, int(t * sr) - half), min(len(y), int(t * sr) + half)
        if b > a:
            out[k] = 20 * np.log10(np.sqrt(np.mean(np.asarray(y[a:b], dtype=np.float64) ** 2)) + 1e-9)
    return out


def group_scores(P: np.ndarray, names: list[str], groups: dict[str, list[str]] = GROUPS) -> dict[str, np.ndarray]:
    """Max posterior over each group's labels, per frame. Labels missing from ``names`` are ignored."""
    ix = {n: i for i, n in enumerate(names)}
    out = {}
    for g, labs in groups.items():
        cols = [ix[l] for l in labs if l in ix]
        out[g] = P[:, cols].max(axis=1) if cols else np.zeros(len(P))
    return out


def decide_frames(scores: dict[str, np.ndarray], level_db: np.ndarray, weights: dict = WEIGHTS,
                  quiet_db: float = QUIET_DBFS, other_floor: float = OTHER_FLOOR) -> np.ndarray:
    """One region label per frame: weighted argmax, level gate for ``quiet``, floor for ``other``."""
    kinds = [k for k in ("music", "voices", "applause", "quiet") if k in scores]
    W = np.stack([scores[k] * weights.get(k, 1.0) for k in kinds], axis=1)
    lab = np.array([kinds[i] for i in W.argmax(axis=1)], dtype=object)
    lab[W.max(axis=1) < other_floor] = "other"
    lab[np.asarray(level_db) < quiet_db] = "quiet"
    return lab


def mode_filter(lab: np.ndarray, width: int = SMOOTH_FRAMES) -> np.ndarray:
    """Sliding majority vote; a tie keeps the centre label."""
    if width <= 1:
        return lab.copy()
    h, n, out = width // 2, len(lab), lab.copy()
    for i in range(n):
        w = lab[max(0, i - h): min(n, i + h + 1)]
        vals, counts = np.unique(w, return_counts=True)
        winners = set(vals[counts == counts.max()])
        out[i] = lab[i] if lab[i] in winners else vals[counts.argmax()]
    return out


def runs_to_spans(lab: np.ndarray, hop_s: float, win_s: float, total_s: float,
                  scores: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Run-length encode frame labels into contiguous spans covering ``[0, total_s]``."""
    spans, n, i = [], len(lab), 0
    off = (win_s - hop_s) / 2.0        # frame i is centred at i*hop + win/2 and owns +-hop/2
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        start = 0.0 if i == 0 else i * hop_s + off
        end = total_s if j == n - 1 else (j + 1) * hop_s + off
        conf = float(np.mean(scores[lab[i]][i:j + 1])) if scores and lab[i] in scores else 0.0
        spans.append(_span(lab[i], start, end, conf))
        i = j + 1
    return spans


def _span(label, start, end, conf=0.0) -> dict:
    return {"label": str(label), "start_s": float(start), "end_s": float(end),
            "duration_s": float(end - start), "confidence": float(conf)}


def _merge_adjacent(spans: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in spans:
        if out and out[-1]["label"] == s["label"]:
            a = out[-1]
            tot = a["duration_s"] + s["duration_s"]
            a["confidence"] = (a["confidence"] * a["duration_s"] + s["confidence"] * s["duration_s"]) / tot if tot else 0.0
            a["end_s"] = s["end_s"]; a["duration_s"] = a["end_s"] - a["start_s"]
        else:
            out.append(dict(s))
    return out


def enforce_min_duration(spans: list[dict], min_s: dict[str, float] = MIN_DURATION_S) -> list[dict]:
    """Absorb spans shorter than their class minimum into the longer neighbour, shortest first, until stable."""
    spans = _merge_adjacent(spans)
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for k in sorted(range(len(spans)), key=lambda k: spans[k]["duration_s"]):
            s = spans[k]
            if s["duration_s"] >= min_s.get(s["label"], 0.0):
                continue
            prev = spans[k - 1] if k > 0 else None
            nxt = spans[k + 1] if k + 1 < len(spans) else None
            if prev is None and nxt is None:
                break
            target = prev if (nxt is None or (prev is not None and prev["duration_s"] >= nxt["duration_s"])) else nxt
            if target is prev:
                prev["end_s"] = s["end_s"]; prev["duration_s"] = prev["end_s"] - prev["start_s"]
            else:
                nxt["start_s"] = s["start_s"]; nxt["duration_s"] = nxt["end_s"] - nxt["start_s"]
            del spans[k]
            spans = _merge_adjacent(spans)
            changed = True
            break
    return spans


def absorb_other(spans: list[dict]) -> list[dict]:
    """``other`` next to music and not next to voices is part of the performance.

    Noise music, laptop pieces and extended techniques score low on "Music" and come out as
    loud sound that is neither speech, applause nor silence; between or beside music, that is
    the piece."""
    out = [dict(s) for s in spans]
    for k, s in enumerate(out):
        if s["label"] != "other":
            continue
        prev = out[k - 1]["label"] if k > 0 else None
        nxt = out[k + 1]["label"] if k + 1 < len(out) else None
        if "music" in (prev, nxt) and "voices" not in (prev, nxt):
            s["label"] = "music"
    return _merge_adjacent(out)


def snap_to_songs(spans: list[dict], songs: list[dict], snap_s: float = SNAP_S) -> list[dict]:
    """Move music edges onto :func:`concert.find_songs` boundaries when the two agree within ``snap_s``."""
    out = [dict(s) for s in spans]
    for k, s in enumerate(out):
        if s["label"] != "music":
            continue
        for song in songs:
            if abs(song["start_s"] - s["start_s"]) <= snap_s:
                s["start_s"] = float(song["start_s"])
                if k > 0:
                    out[k - 1]["end_s"] = s["start_s"]
            if abs(song["end_s"] - s["end_s"]) <= snap_s:
                s["end_s"] = float(song["end_s"])
                if k + 1 < len(out):
                    out[k + 1]["start_s"] = s["end_s"]
    for s in out:
        s["duration_s"] = s["end_s"] - s["start_s"]
    return [s for s in out if s["duration_s"] > 0]


def refine_music_onsets(spans: list[dict], y: np.ndarray, sr: int, look_back_s: float = 10.0,
                        rise_db: float = 8.0, frame_s: float = 0.1) -> list[dict]:
    """Move each music span's start back to where the sound actually begins.

    Tag windows are seconds long and the majority filter is longer, so a detected start
    trails the first note by a few seconds. Within ``look_back_s`` before the detected start,
    the onset is the earliest frame from which the level stays ``rise_db`` above the floor
    of that window until the detected start."""
    out = [dict(s) for s in spans]
    hop = max(1, int(frame_s * sr))
    for k, s in enumerate(out):
        if s["label"] != "music" or k == 0:
            continue
        a = max(0, int((s["start_s"] - look_back_s) * sr)); b = min(len(y), int((s["start_s"] + 2.0) * sr))
        if b - a < 5 * hop:
            continue
        seg = np.asarray(y[a:b], dtype=np.float64)
        n = len(seg) // hop
        lv = 20 * np.log10(np.sqrt((seg[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
        floor = np.percentile(lv, 10)
        above = lv > floor + rise_db
        i0 = int(round((s["start_s"] * sr - a) / hop))
        i0 = min(max(i0, 1), n - 1)
        j = i0
        while j > 0 and above[j - 1]:
            j -= 1
        onset = a / sr + j * frame_s
        earliest = out[k - 1]["start_s"] + 1.0
        if onset < s["start_s"] and onset > earliest:
            s["start_s"] = float(onset); out[k - 1]["end_s"] = float(onset)
    for s in out:
        s["duration_s"] = s["end_s"] - s["start_s"]
    return [s for s in out if s["duration_s"] > 0]


def segment_concert(y: np.ndarray, sr: int, songs: list[dict] | None = None, win_s: float = 4.0,
                    hop_s: float = 2.0, device=None, min_s: dict[str, float] = MIN_DURATION_S,
                    refine_onsets: bool = True) -> dict:
    """The whole chain on one mono array.

    Returns ``{"spans": [...], "total_s": float, "frames": {"t", "music", "voices", "applause", "quiet", "level_db"}}``
    with spans in the :data:`musiscape.concert.REGION_CLASSES` vocabulary, contiguous from 0 to the end.
    """
    total_s = len(y) / sr
    t, P, names = tag_frames(y, sr, win_s=win_s, hop_s=hop_s, device=device)
    level = frame_level_db(y, sr, t, win_s)
    scores = group_scores(P, names)
    lab = mode_filter(decide_frames(scores, level))
    spans = runs_to_spans(lab, hop_s, win_s, total_s, scores)
    spans = enforce_min_duration(spans, min_s)
    spans = absorb_other(spans)
    if songs:
        spans = snap_to_songs(spans, songs)
    if refine_onsets:
        spans = refine_music_onsets(spans, y, sr)
    for s in spans:
        sel = (t >= s["start_s"]) & (t < s["end_s"])
        if sel.any() and s["label"] in scores:
            s["confidence"] = float(scores[s["label"]][sel].mean())
        s["confidence"] = round(s["confidence"], 3)
        assert s["label"] in REGION_CLASSES
    frames = {"t": t, "level_db": level, **{k: v for k, v in scores.items()}}
    return {"spans": spans, "total_s": round(total_s, 2), "frames": frames}
