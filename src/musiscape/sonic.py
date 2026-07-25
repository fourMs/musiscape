"""Sonic thumbnails: a short audio summary of each track.

Where the visual thumbnails answer "what does this piece look like", the
sonic thumbnail answers "what does it sound like" in ~12 seconds: a
montage of up to three segments chosen deterministically from the track's
own structure — the most *representative* passage (the window whose
features are closest to the whole track, in the audio-thumbnailing
tradition of Bartsch & Wakefield), the *climax* (peak energy), and the
most *contrasting* section that still carries energy. Segments are placed
in chronological order and joined with equal-power crossfades, so the
summary preserves the piece's own dramaturgy.

Everything is explainable: no learned model decides what matters.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .io import Collection, Track, load

SEG_S = 4.0        #: seconds per selected segment
FADE_S = 0.3       #: crossfade length
STEP_S = 1.0       #: candidate-window hop


def _select_segments(y, sr, n_segments=3, seg_s=SEG_S):
    """Segment start times (s), chosen for representativeness, climax
    and contrast, returned in chronological order."""
    from .thumbnails import _feats_2hz
    C, M, rms = _feats_2hz(y, sr)
    F = np.vstack([C, M / (np.abs(M).max() + 1e-9),
                   rms[None, :] / (rms.max() + 1e-9)])
    Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
    n = F.shape[1]
    w = int(seg_s * 2)                       # frames per window (2 Hz)
    step = max(1, int(STEP_S * 2))
    if n <= w:
        return [0.0]
    starts = np.arange(0, n - w, step)
    wins = np.stack([Fn[:, s:s + w].mean(1) for s in starts])
    wins = wins / (np.linalg.norm(wins, axis=1, keepdims=True) + 1e-9)
    rep_score = (wins @ Fn).mean(1)
    energy = np.array([rms[s:s + w].mean() for s in starts])

    chosen = [int(starts[np.argmax(rep_score)])]
    if n_segments >= 2:
        clim = int(starts[np.argmax(energy)])
        if all(abs(clim - c) >= w for c in chosen):
            chosen.append(clim)
    if n_segments >= 3:
        rep_vec = wins[np.argmax(rep_score)]
        contrast = (1 - wins @ rep_vec) * (energy > np.median(energy))
        order = np.argsort(-contrast)
        for k in order:
            cand = int(starts[k])
            if all(abs(cand - c) >= w for c in chosen):
                chosen.append(cand)
                break
    return sorted(c / 2.0 for c in chosen)


def sonic_thumbnail(y, sr, n_segments=3, seg_s=SEG_S,
                    fade_s=FADE_S) -> np.ndarray:
    """A ~12 s audio summary montage of ``y`` (mono float array)."""
    starts = _select_segments(y, sr, n_segments=n_segments, seg_s=seg_s)
    nf = int(fade_s * sr)
    fade_in = np.sin(np.linspace(0, np.pi / 2, nf)) ** 2
    segs = []
    for t0 in starts:
        a = int(t0 * sr)
        seg = y[a:a + int(seg_s * sr)].copy()
        if len(seg) < nf * 2:
            continue
        seg[:nf] *= fade_in
        seg[-nf:] *= fade_in[::-1]
        segs.append(seg)
    if not segs:
        return y[: int(seg_s * sr)]
    out = segs[0]
    for seg in segs[1:]:
        head, tail = out[:-nf], out[-nf:]
        out = np.concatenate([head, tail + seg[:nf], seg[nf:]])
    peak = np.abs(out).max() + 1e-9
    return (out / peak * 0.89).astype(np.float32)


def _work(job):
    path, album, out, sr = job
    import soundfile as sf
    try:
        t = Track(path=Path(path), album=album)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load(t, sr=sr)
            thumb = sonic_thumbnail(y, sr)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, thumb, sr)
        print(f"[{album}] {t.title}", flush=True)
        return str(out)
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR [{album}] {path}: {e}", flush=True)
        return None


def export_collection(coll: Collection, out_dir: str | Path,
                      workers: int = 4, sr: int = 22050) -> Path:
    """Sonic thumbnails for every track → ``<out_dir>/sonic/<album>/``,
    plus one concatenated medley file per album."""
    import soundfile as sf
    out_dir = Path(out_dir) / "sonic"
    jobs = [(str(t.path), t.album,
             out_dir / t.album / f"{t.title}.wav", sr) for t in coll.tracks]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_work, jobs))
    else:
        list(map(_work, jobs))
    gap = np.zeros(int(0.5 * sr), dtype=np.float32)
    for a in coll.albums:
        parts = []
        for t in a.tracks:
            f = out_dir / a.name / f"{t.title}.wav"
            if f.exists():
                parts += [sf.read(f, dtype="float32")[0], gap]
        if parts:
            sf.write(out_dir / f"{a.name.replace('/', '_')} medley.wav",
                     np.concatenate(parts[:-1]), sr)
    return out_dir
