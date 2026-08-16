"""Does an estimate hold up across the track, or only on average?

:mod:`features` gates two descriptors on whole-track statistics: near-uniform
chroma makes ``key`` an artefact, and an onset envelope with no periodicity
makes ``tempo_bpm`` a report of librosa's prior. Both gates catch real
failures. Both are also averages over the whole track, which is a second way
to be wrong: not by measuring noise, but by measuring something real over too
long a window.

Live music is where the difference shows. ``pulse_R`` folds an entire take at
one global period, so a band that drifts a few BPM across four minutes
collapses the resultant while playing a steady beat. ``chroma_entropy``
averages chroma over the whole take, and a full band in a reverberant room
flattens that average far past a threshold calibrated on solo instrumental
material.

This module measures the same two quantities per window and reports how far
the windows agree. High agreement on a gated track means the gate was too
coarse for the material. Low agreement means the track really does wander,
which is worth knowing and is a different statement from "unmeasurable".

The question answered is therefore narrow and answerable: not "is there a
pulse" but "does the estimate hold still". No descriptor here reports whether
a track has a beat at all, because on real material no statistic of
periodicity can tell one. Applause is rhythmic, so a room clapping in
near-unison scores in the same range as the band it is applauding, whether
measured by beat strength or by tempogram peak prominence. Separating the two
is what :mod:`musiscape.concert` uses spectral flatness for, and on a single
track the tempogram figure read by eye is the honest answer.

Both functions take features already computed elsewhere (a chromagram, an
onset envelope) rather than audio, so adding them to an extraction costs
almost nothing: :func:`features.extract_track` has both in hand.
"""
from __future__ import annotations

import numpy as np

from .features import estimate_key

#: Window length. Long enough to hold a phrase and settle a key estimate,
#: short enough that a four-minute take yields a dozen independent votes.
WIN_S = 20.0

#: Tempo tolerance for counting two windows as agreeing.
TEMPO_TOL = 0.02


def _n_windows(n_frames: int, sr: int, hop: int, win_s: float) -> tuple[int, int]:
    """Frames per window, and how many whole windows fit."""
    n = max(1, int(round(win_s * sr / hop)))
    return n, max(1, n_frames // n)


def key_stability(chroma: np.ndarray, sr: int, hop: int = 512,
                  win_s: float = WIN_S) -> dict:
    """Krumhansl--Schmuckler key per window, and how often they agree.

    ``chroma`` is a (12, frames) chromagram, ``chroma_cqt`` on the harmonic
    component as :mod:`features` computes it. Returns the modal key across
    windows, the share of windows holding it, and the window count.

    ``agreement`` is ``None`` when only one window fits: a single window
    agrees with itself trivially, and reporting 1.0 for a short track would
    make the least evidence look like the most.
    """
    chroma = np.asarray(chroma, float)
    n, m = _n_windows(chroma.shape[1], sr, hop, win_s)
    keys = [estimate_key(chroma[:, i * n:(i + 1) * n].mean(axis=1))[0]
            for i in range(m)]
    modal = max(set(keys), key=keys.count)
    return {"key": modal,
            "agreement": round(keys.count(modal) / len(keys), 3)
            if len(keys) > 1 else None,
            "n_windows": len(keys), "keys": keys}


def tempo_stability(onset_env: np.ndarray, sr: int, hop: int = 512,
                    win_s: float = WIN_S, tol: float = TEMPO_TOL) -> dict:
    """Tempo per window, and how often the windows agree.

    ``onset_env`` is an onset-strength envelope. Returns the median
    windowed tempo, the share of windows within ``tol`` of it, and the
    number of windows.

    Nothing is returned for "is there a beat at all"; see the module
    docstring for why that question has no reliable answer here.

    Windowed tempos are folded by metrical octave before being compared. A
    window heard at double or half time agrees about where the beat is, and
    counting it as disagreement would make every syncopated track look
    unstable.
    """
    import librosa
    try:
        from librosa.feature.rhythm import tempo as _tempo
    except ImportError:                                  # librosa < 0.10
        _tempo = librosa.beat.tempo

    env = np.asarray(onset_env, float)
    n, m = _n_windows(len(env), sr, hop, win_s)
    t = np.array([float(_tempo(onset_envelope=env[i * n:(i + 1) * n], sr=sr,
                               hop_length=hop)[0]) for i in range(m)])

    med = np.median(t)
    folded = t.copy()
    folded[folded < med / 1.5] *= 2
    folded[folded > med * 1.5] /= 2
    tmed = float(np.median(folded))
    agree = float(np.mean(np.abs(folded - tmed) / max(tmed, 1e-9) < tol))

    return {"tempo_bpm": round(tmed, 1),
            "agreement": round(agree, 3) if m > 1 else None,
            "n_windows": m}
