"""Which sound is in front, second by second, when there is no multitrack.

A concert recorded from one microphone gives one mixture. When the question is "what did the
painter do when the violin led, and when the electronics led", the honest answer without stems
is a set of *proxies*: envelopes that track the kind of sound each instrument makes rather than
the instrument itself. This module computes three such envelopes per second and says plainly
what each one is.

- **pitched** --- energy of the harmonic component (HPSS) between `pitched_band`, weighted by
  the voicing probability of a pitch tracker (pYIN) run on that component. A bowed string,
  a voice or a wind instrument playing notes scores high; a synthesiser drone with a clear
  pitch scores high too, which is why this is a proxy and not a separation.
- **low** --- energy below `low_hz`. Bass, sub-bass and the weight of a PA.
- **noise** --- the percussive/residual component's energy plus the spectral flatness of the
  mixture: attacks, noise textures, applause, brushes on canvas.

Each envelope is scaled to its own 99th percentile, and :func:`foreground_labels` turns the three
into one label per second (``"pitched"``, ``"low"``, ``"noise"``, ``"mixed"`` or ``"quiet"``)
with a margin, so a second is labelled only when one proxy clearly leads. On the live-painting
concert this was written for, the pitched proxy followed the violin and the low+noise proxies
the electronics; that reading was checked against the photographs and the AudioSet tagger, not
assumed. Check yours the same way before calling a proxy an instrument.
"""
from __future__ import annotations

import numpy as np


def instrument_foreground(y, sr: int, hop: int = 512, pitched_band=(180.0, 4000.0),
                          low_hz: float = 180.0, fmin: float = 150.0, fmax: float = 3000.0) -> dict:
    """Per-second proxies for pitched, low and noise-like foreground.

    Args:
        y: Mono audio.
        sr (int): Sample rate.
        hop (int): Hop for the frame analysis. Defaults to 512.
        pitched_band (tuple): Frequency band, Hz, of the harmonic energy behind ``pitched``.
        low_hz (float): Upper edge of the ``low`` band. Defaults to 180 Hz.
        fmin, fmax (float): Pitch-tracker range in Hz. Defaults to 150–3000 Hz.

    Returns:
        dict: ``t`` (bin centres in seconds), ``pitched``, ``low``, ``noise`` (each scaled 0–1 by
        its 99th percentile), ``voiced`` (pYIN voicing probability per second), ``f0_midi``
        (median voiced pitch per second, NaN where unvoiced) and ``level_db``.
    """
    import librosa

    y = np.asarray(y, dtype=float)
    n = int(np.ceil(len(y) / sr))
    yh, yp = librosa.effects.hpss(y, margin=(1.0, 3.0))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    Sh = np.abs(librosa.stft(yh, n_fft=2048, hop_length=hop))
    Sp = np.abs(librosa.stft(yp, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    fr = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    sec = np.clip(np.floor(fr).astype(int), 0, n - 1)
    cnt = np.maximum(np.bincount(sec, minlength=n), 1)

    def per_sec(v):
        return np.bincount(sec, weights=v[:len(sec)], minlength=n) / cnt

    band = (freqs >= pitched_band[0]) & (freqs <= pitched_band[1])
    e_pitched = per_sec((Sh[band] ** 2).sum(0))
    e_low = per_sec((S[freqs < low_hz] ** 2).sum(0))
    e_noise = per_sec((Sp ** 2).sum(0))
    flat = per_sec(librosa.feature.spectral_flatness(S=S)[0])
    f0, voiced_flag, voiced_prob = librosa.pyin(yh, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop)
    m = min(len(voiced_prob), len(sec))
    voiced = np.bincount(sec[:m], weights=np.nan_to_num(voiced_prob[:m]), minlength=n) / cnt
    f0m = np.full(n, np.nan)
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), np.nan, f0))
    for s in range(n):
        sel = (sec[:m] == s) & ~np.isnan(midi[:m])
        if sel.any():
            f0m[s] = float(np.median(midi[:m][sel]))
    scale = lambda v: v / (np.percentile(v, 99) + 1e-12)  # noqa: E731
    pitched = scale(e_pitched * voiced)
    low = scale(e_low)
    noise = scale(e_noise * (0.5 + flat))
    level = 20 * np.log10(per_sec(librosa.feature.rms(S=S)[0]) + 1e-9)
    return {"t": np.arange(n) + 0.5, "pitched": pitched, "low": low, "noise": noise, "voiced": voiced, "f0_midi": f0m, "level_db": level}


def foreground_labels(fg: dict, margin: float = 0.15, quiet_db: float = -55.0) -> np.ndarray:
    """One label per second from the three proxies.

    Args:
        fg (dict): From :func:`instrument_foreground`.
        margin (float): How far the leading proxy must exceed the runner-up. Defaults to 0.15.
        quiet_db (float): Seconds below this level are ``"quiet"``. Defaults to −55 dBFS.

    Returns:
        numpy.ndarray: Strings ``"pitched"``, ``"low"``, ``"noise"``, ``"mixed"`` or ``"quiet"``.
    """
    P = np.vstack([fg["pitched"], fg["low"], fg["noise"]])
    names = np.array(["pitched", "low", "noise"])
    order = np.argsort(-P, axis=0)
    lead = P[order[0], np.arange(P.shape[1])]
    second = P[order[1], np.arange(P.shape[1])]
    lab = np.where(lead - second >= margin, names[order[0]], "mixed")
    lab = np.where(fg["level_db"] < quiet_db, "quiet", lab)
    return lab
