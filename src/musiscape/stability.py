"""Does an estimate hold up across the track, or only on average?

:mod:`features` gates two descriptors on whole-track statistics, and says
so plainly: near-uniform chroma makes ``key`` an artefact, and an onset
envelope with no periodicity makes ``tempo_bpm`` a report of librosa's
prior. Both gates catch real failures. Both are also *averages over the
whole track*, and that is a second way to fail --- not by measuring noise,
but by measuring something real over too long a window.

Live music is where the difference shows. ``pulse_R`` folds an entire take
at one global period, so a band that drifts a few BPM across four minutes
collapses the resultant while playing a perfectly steady beat.
``chroma_entropy`` averages chroma over the whole take, and a full band in
a reverberant room flattens that average far past the threshold calibrated
on solo instrumental material. On a concert measured with these functions,
seven of eight songs failed both gates, and seven of eight windowed key
estimates nonetheless agreed with the whole-song one.

So this module measures the same two quantities *per window* and reports
how much the windows agree. High agreement on a gated track means the gate
was too coarse, not that the track is tonal or pulsed by luck. Low
agreement means the track really does wander --- which is itself worth
knowing, and is a different statement from "unmeasurable".

**A single-number pulse gate was attempted twice and abandoned, and that is
a result worth recording rather than a gap.**

Inter-beat regularity fails first. ``beat_track`` fits one global grid, so
its intervals stay even through a tempo change: a 120 BPM click train and a
120-then-80 train both come back with beats 0.5 s apart, differing only by
hop quantisation. The number cannot move.

Beat salience --- onset strength at the tracked beats over the track mean
--- fails second, and more instructively. It looks decisive on synthetic
material (about 1.2 on white noise, above 13 on a click train) and does not
survive contact with a real room: across one concert the songs measured
1.58--1.93 and the applause between them 1.67, with room tone at 1.50.
Tempogram peak prominence does better and still overlaps: songs 1.09--1.62
against 1.07--1.24 for applause and room tone.

The reason is not a defective measure. **Applause is rhythmic.** A room
clapping in near-unison has a periodic onset envelope, and no statistic of
periodicity alone will separate it from a band. Sorting the two is what
:mod:`musiscape.concert` uses spectral flatness for, and on a single track
the honest answer is the tempogram figure read by eye.

What is reported here is therefore the narrower question that can be
answered: not "is there a pulse" but "does the tempo estimate hold still".

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

    ``chroma`` is a (12, frames) chromagram---``chroma_cqt`` on the
    harmonic component, as :mod:`features` computes it. Returns the modal
    key across windows, the share of windows holding it, and the window
    count.

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

    What is *not* returned is any single number answering "is there a beat
    at all", because two candidates were tried against real material and
    both failed. See the module docstring: the honest answer on a concert
    recording is the tempogram figure, read by eye.

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
