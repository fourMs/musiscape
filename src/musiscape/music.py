"""Tempogram, chromagram, and the circular views built on them.

The MIR-standard time-resolved views: the tempogram (onset autocorrelation
over time, in BPM) and the chromagram (12-bin pitch-class energy over time).
:mod:`musiscape.figures` draws both with labelled axes.

Three views use circular statistics, which come from
:mod:`micromotion.circular`:

- :func:`pulse_clarity`, the metric lock of the onsets, useful where a plain
  BPM estimate fails on rubato material;
- :func:`fifths_center`, the tonal centre and focus of a chroma vector on the
  circle of fifths;
- :func:`tonal_center_spread`, how tightly a set of recordings clusters in
  key space, a between-recording statistic with no linear equivalent.

:func:`tartyp_profile` classifies onset-bounded sound objects on a simplified
Schaeffer typology grid; see the Schaeffer guide for what the classes mean.

Audio is resampled to 22.05 kHz. Long recordings are fine: a 25 minute file
takes on the order of a minute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_librosa():
    try:
        import librosa
        return librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required: pip install musiscape") from e


def tempogram(y, sr, hop=512, win_s=8.0):
    """Autocorrelation tempogram of the onset-strength envelope.

    Returns (times, bpm_axis, T, tempo_bpm): the tempogram plus librosa's
    global tempo estimate (which resolves the octave ambiguity a raw
    tempogram argmax suffers from).
    """
    librosa = _require_librosa()
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = int(win_s * sr / hop)
    T = librosa.feature.tempogram(onset_envelope=onset, sr=sr,
                                  hop_length=hop, win_length=win)
    times = librosa.frames_to_time(np.arange(T.shape[1]), sr=sr,
                                   hop_length=hop)
    bpm = librosa.tempo_frequencies(T.shape[0], sr=sr, hop_length=hop)
    try:
        from librosa.feature.rhythm import tempo as _tempo
    except ImportError:                      # librosa < 0.10
        _tempo = librosa.beat.tempo
    t_est = float(_tempo(onset_envelope=onset, sr=sr, hop_length=hop)[0])
    return times, bpm, T, t_est


def chromagram(y, sr, hop=512):
    """STFT chromagram (12 pitch classes over time)."""
    librosa = _require_librosa()
    C = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr,
                                   hop_length=hop)
    return times, C


# --------------------------------------------------------------------------
# Circular-statistics views and the object-level Schaeffer profile.
# The TARTYP thresholds below are calibrated on a tonal instrumental corpus
# (a five-album solo-harp catalogue, 57 tracks), and should be treated as
# indicative on other material.

FIFTHS_ANGLE = 2 * np.pi * (7 * np.arange(12) % 12) / 12

# Onset-strength floor (dB-difference units): real attacks measure O(1-10),
# peak-picked numerical noise on steady signals O(0.01).
ONSET_FLOOR = 0.5

#: Note names, indexed by pitch class.
NOTE = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def dominant_period(onset_env, sr, hop=512, bpm_range=(40.0, 200.0)):
    """Dominant beat period (s) from the onset-envelope autocorrelation."""
    librosa = _require_librosa()
    ac = librosa.autocorrelate(onset_env, max_size=int(4 * sr / hop))
    lo = max(1, int(60.0 / bpm_range[1] * sr / hop))
    hi = min(len(ac), int(60.0 / bpm_range[0] * sr / hop))
    lag = lo + int(np.argmax(ac[lo:hi]))
    return lag * hop / sr


def pulse_clarity(y, sr, hop=512, bpm_range=(40.0, 200.0)) -> dict:
    """Metric lock of the onsets: circular concentration of onset phases.

    Onsets are folded at the dominant period and their strength-weighted
    resultant length R taken as the score — 0 is free rubato, 1 metronomic.
    Works where a beat tracker's BPM is meaningless (rubato, drones); the
    single global period means slow tempo drift also reads as low R, so
    treat R as "lock to one steady grid", not "has any pulse at all".
    """
    from micromotion.circular import circ_mean, rayleigh_from_R
    librosa = _require_librosa()
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    p0 = dominant_period(env, sr, hop, bpm_range)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                    hop_length=hop, units="frames")
    fr = fr[env[fr] >= ONSET_FLOOR]
    t = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
    if len(t) < 8:
        return {"R": 0.0, "period_s": round(p0, 4),
                "period_bpm": round(60.0 / p0, 1), "n_onsets": int(len(t))}
    w = np.interp(t, librosa.times_like(env, sr=sr, hop_length=hop), env)
    # The ACF peak may sit an octave off the felt pulse (folding 120 BPM
    # onsets at a 1 s period cancels the resultant), so let R itself pick
    # among the peak and its metrical neighbours.
    period, R = p0, -1.0
    for p in (p0 / 2, p0, p0 * 2):
        if not (60.0 / bpm_range[1] <= p <= 60.0 / bpm_range[0]):
            continue
        r = circ_mean(2 * np.pi * (t / p % 1.0), weights=w)["R"]
        if r > R:
            period, R = p, r
    return {"R": round(R, 4), "period_s": round(period, 4),
            "period_bpm": round(60.0 / period, 1),
            "rayleigh_p": rayleigh_from_R(R, len(t)), "n_onsets": int(len(t))}


def fifths_center(chroma) -> dict:
    """Tonal centre and focus of a chroma vector on the circle of fifths.

    The 12 pitch classes are placed a fifth apart around the circle and the
    chroma-weighted resultant taken: the mean angle is the tonal centre
    (returned as the nearest note name and as an angle in fifths steps),
    R the tonal focus — diatonic material concentrates on one arc, chromatic
    or inharmonic material smears. Where the full 12-bin shape is wanted,
    use the chroma vector itself.
    """
    from micromotion.circular import circ_mean
    w = np.asarray(chroma, float)
    _c = circ_mean(FIFTHS_ANGLE, weights=w)
    mu, R = _c["mean"], _c["R"]
    k = (mu / (2 * np.pi) * 12) % 12                 # steps along the fifths circle
    note = NOTE[int(7 * int(round(k)) % 12)]
    return {"center_note": note, "center_fifths": round(float(k), 3),
            "R": round(R, 4)}


def tonal_center_spread(chromas) -> dict:
    """Concentration of many recordings' tonal centres on the fifths circle.

    Feed one mean-chroma vector per recording; returns the resultant length
    R of their centres — near 1 when a corpus stays in neighbouring keys
    (a suite), near 0 when it wanders the whole circle. Key centres have no
    meaningful linear mean, so this is inherently a circular statistic.
    """
    from micromotion.circular import circ_mean, circular_sd
    angles = []
    for c in chromas:
        w = np.asarray(c, float)
        mu = circ_mean(FIFTHS_ANGLE, weights=w)["mean"]
        angles.append(mu)
    R = circ_mean(np.asarray(angles))["R"]
    return {"R": round(R, 4),
            "circ_sd_fifths": round(circular_sd(R) / (2 * np.pi) * 12, 3),
            "n": len(angles)}


# TARTYP proxy thresholds (median object spectral flatness; std of log2
# centroid in octaves; share of 4–20 Hz envelope-modulation energy)
TARTYP_TONIC = 0.004
TARTYP_COMPLEX = 0.02
TARTYP_DRIFT = 0.35
TARTYP_ITER = 0.45
TARTYP_IMPULSE_S = 0.3


def tartyp_profile(y, sr, hop=512) -> dict:
    """Duration-weighted Schaeffer typology profile of onset-bounded objects.

    Each inter-onset segment is one sound object, classified on a simplified
    TARTYP grid: mass N (tonic) / Y (variable) / X (complex) from spectral
    flatness and centroid drift, facture held / impulse (``'``) / iteration
    (``''``) from duration and 4–20 Hz envelope modulation. Returns the
    share of sounding time per type plus the object count.

    These are signal proxies for aural categories, a complement to reduced
    listening rather than a substitute. Mass reads roughly as N for tonic,
    Y for tonic-complex and X for complex or noisy objects.
    """
    librosa = _require_librosa()
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = librosa.feature.spectral_flatness(S=S)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    fine_hop = hop // 8
    fine = librosa.feature.rms(y=y, frame_length=fine_hop * 4,
                               hop_length=fine_hop)[0]
    fine_rate = sr / fine_hop
    frame_t = hop / sr

    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                       hop_length=hop, units="frames")
    peaks = peaks[onset_env[peaks] >= ONSET_FLOOR]
    onsets = (librosa.onset.onset_backtrack(peaks, onset_env)
              if len(peaks) else peaks)
    bounds = np.unique(np.concatenate([[0], onsets, [S.shape[1]]]))

    def _iter_ratio(a, b):
        seg = fine[int(a * hop / fine_hop):int(b * hop / fine_hop)]
        seg = seg - seg.mean()
        if len(seg) < 16 or not np.any(seg):
            return 0.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fq = np.fft.rfftfreq(len(seg), 1 / fine_rate)
        tot = spec[fq > 0.2].sum()
        return float(spec[(fq >= 4) & (fq < 20)].sum() / tot) if tot else 0.0

    type_time, n_objects = {}, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        dur = (b - a) * frame_t
        if dur < 0.05:
            continue
        n_objects += 1
        fmed = float(np.median(flat[a:b]))
        drift = float(np.std(np.log2(cent[a:b] + 1e-6)))
        if dur < TARTYP_IMPULSE_S:
            suffix = "'"
        elif _iter_ratio(a, b) > TARTYP_ITER:
            suffix = "''"
        else:
            suffix = ""
        if fmed >= TARTYP_COMPLEX:
            mass = "X"
        elif fmed >= TARTYP_TONIC or drift > TARTYP_DRIFT:
            mass = "Y"
        else:
            mass = "N"
        key = mass + suffix
        type_time[key] = type_time.get(key, 0.0) + dur

    tot = sum(type_time.values())
    dist = {k: round(v / tot, 4) for k, v in sorted(type_time.items())} if tot else {}
    return {"dist": dist, "n_objects": n_objects}


