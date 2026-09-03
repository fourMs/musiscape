"""How a recording changes over its own length: pitch, harmony, pulse and timbre, second by second.

:mod:`features` describes a track by one number per descriptor, and :mod:`stability` asks
whether that number holds still. This module gives the time-course itself, at one frame per
second, for a recording that is *meant* to change: a concert, a long improvisation, a session in
which the interesting question is not "what key is it in" but "when does it move, and in what".

Everything here is per second, on one clock, so the arrays can sit beside a motion or gaze track
from another toolbox and be correlated, segmented and drawn on the same width. The descriptors:

- **Chroma** of the harmonic component (HPSS first, so a piano's attacks and a room's noise do
  not smear the pitch classes), normalised per second; **chroma entropy** as harmonic complexity;
  **tonal clarity** as how far one pitch class stands out.
- **Key** in overlapping windows by correlation with the Krumhansl–Kessler profiles, reported as
  a list of (time, key, correlation), because a key that changes every window is a statement
  about the music (modal, wandering) and not a failure.
- **Harmonic change** as the tonnetz distance between consecutive seconds (Harte's HCDF).
- **Tempogram** (onset-strength autocorrelation) with its per-second argmax as *local tempo*
  and its peak-to-mean ratio as *pulse clarity*; a windowed beat-tracked tempo alongside. On
  non-metric music these are honest reports of the absence of a pulse, and the module does not
  pretend otherwise: read `pulse_clarity` before reading `local_tempo`.
- **Timbre**: MFCCs, spectral centroid, flatness, and the harmonic share from HPSS; **timbre
  novelty** from a Foote kernel on the MFCC self-similarity.
- **Register** as the energy-weighted MIDI pitch of the harmonic constant-Q spectrum, with its
  spread, so "the violin went up" is a number.

:func:`section_summary` folds the time-course into one row per section for a table, and
:func:`timecourse_figures` draws the three standard pictures (chromagram, tempogram,
time-course). The functions take audio, not a track, so they work on a cut of a concert as
readily as on a file. The time-course of the live-painting concert this was written for showed a
music organised around pedal points rather than progressions, which the whole-track key estimate
had hidden behind a single "C minor".
"""
from __future__ import annotations

import numpy as np

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
#: Krumhansl & Kessler (1982) major and minor key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_from_chroma(chroma_mean) -> tuple[str, float]:
    """The best-matching key for a 12-bin chroma vector, and its profile correlation.

    Args:
        chroma_mean: Twelve values, C first.

    Returns:
        tuple: ``(name, r)`` such as ``("A minor", 0.71)``; the correlation is the maximum over
        the 24 rotated profiles. A flat chroma gives a low ``r`` rather than an error.
    """
    v = np.asarray(chroma_mean, dtype=float)
    if v.std() < 1e-9:
        return "C major", 0.0
    best = (-2.0, "C major")
    for i in range(12):
        for prof, lab in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            r = float(np.corrcoef(np.roll(prof, i), v)[0, 1])
            if r > best[0]:
                best = (r, f"{KEY_NAMES[i]} {lab}")
    return best[1], best[0]


def _per_second(M: np.ndarray, sec: np.ndarray, n: int) -> np.ndarray:
    """Mean of each row of a (d, frames) matrix within each second -> (d, n)."""
    out = np.zeros((M.shape[0], n))
    cnt = np.bincount(sec, minlength=n)
    for k in range(M.shape[0]):
        out[k] = np.bincount(sec, weights=M[k], minlength=n) / np.maximum(cnt, 1)
    return out


def foote_novelty(X: np.ndarray, half_window: int) -> np.ndarray:
    """Foote (2000) novelty of a (time, features) matrix with a checkerboard kernel.

    Args:
        X: Standardised features, one row per frame.
        half_window (int): Kernel half-size in frames.

    Returns:
        numpy.ndarray: Novelty per frame, scaled to a maximum of 1; zero within
        `half_window` of the edges.
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    L = half_window
    sign = np.r_[-np.ones(L), np.ones(L)]
    k = np.outer(sign, sign) * np.outer(np.hanning(2 * L), np.hanning(2 * L))
    n = S.shape[0]
    nov = np.zeros(n)
    for i in range(L, n - L):
        nov[i] = (S[i - L:i + L, i - L:i + L] * k).sum()
    nov = np.maximum(nov, 0.0)
    return nov / (nov.max() + 1e-9)


def music_timecourse(y, sr: int, hop: int = 512, key_window_s: float = 30.0,
                     key_step_s: float = 10.0, tempo_window_s: float = 30.0,
                     tempo_range=(40.0, 200.0), novelty_half_window_s: float = 30.0) -> dict:
    """The per-second time-course of a recording's pitch, harmony, pulse and timbre.

    Args:
        y: Mono audio.
        sr (int): Sample rate.
        hop (int): STFT hop in samples for the underlying frame analysis. Defaults to 512.
        key_window_s (float): Window for the key estimates. Defaults to 30 s.
        key_step_s (float): Step between key windows. Defaults to 10 s.
        tempo_window_s (float): Window for the beat-tracked tempo. Defaults to 30 s.
        tempo_range (tuple): BPM range considered for local tempo and pulse clarity.
        novelty_half_window_s (float): Half-size of the timbre-novelty kernel in seconds.

    Returns:
        dict: ``t`` (seconds, one per row, bin centres), ``chroma`` (12, n) column-normalised,
        ``chroma_entropy``, ``tonal_clarity``, ``keys`` (list of ``(t, name, r)``), ``hcdf``,
        ``tempogram`` (bins, n) within `tempo_range`, ``tempi`` (BPM per bin), ``local_tempo``,
        ``pulse_clarity``, ``tempo_windows`` (list of ``(t, bpm)``), ``mfcc`` (13, n),
        ``centroid``, ``flatness``, ``harmonic_ratio``, ``register_midi``,
        ``register_spread``, ``rms``, ``timbre_novelty``, ``duration``.
    """
    import librosa

    y = np.asarray(y, dtype=float)
    dur = len(y) / sr
    n = int(np.ceil(dur))
    yh, yp = librosa.effects.hpss(y, margin=(1.0, 3.0))
    chroma = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=hop, bins_per_octave=36, n_chroma=12)
    fr_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    sec = np.clip(np.floor(fr_t).astype(int), 0, n - 1)
    ch = _per_second(chroma, sec, n)
    chn = ch / (ch.sum(0, keepdims=True) + 1e-9)
    entropy = -(chn * np.log2(chn + 1e-9)).sum(0) / np.log2(12)
    clarity = chn.max(0) - chn.mean(0)
    keys = []
    W = int(key_window_s)
    for s0 in range(0, max(n - W + 1, 1), max(int(key_step_s), 1)):
        name, r = key_from_chroma(ch[:, s0:s0 + W].mean(1))
        keys.append((s0 + W / 2, name, r))
    tz = _per_second(librosa.feature.tonnetz(chroma=chroma, sr=sr), sec, n)
    hcdf = np.r_[0.0, np.linalg.norm(np.diff(tz, axis=1), axis=0)]
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop, win_length=int(8 * sr / hop))
    tempi = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
    valid = (tempi >= tempo_range[0]) & (tempi <= tempo_range[1])
    sec_o = np.clip(np.floor(librosa.frames_to_time(np.arange(tg.shape[1]), sr=sr, hop_length=hop)).astype(int), 0, n - 1)
    tg1 = _per_second(tg[valid], sec_o, n)
    local_tempo = np.array([tempi[valid][np.argmax(tg1[:, i])] if tg1[:, i].max() > 0 else np.nan for i in range(n)])
    pulse_clarity = tg1.max(0) / (tg1.mean(0) + 1e-9)
    tempo_windows = []
    Wt = int(tempo_window_s)
    for s0 in range(0, max(n - Wt + 1, 1), max(int(key_step_s), 1)):
        seg = oenv[int(s0 * sr / hop):int((s0 + Wt) * sr / hop)]
        if len(seg) < 4:
            continue
        try:
            est = librosa.feature.tempo(onset_envelope=seg, sr=sr, hop_length=hop, aggregate=None)
            tempo_windows.append((s0 + Wt / 2, float(np.median(est))))
        except Exception:  # pragma: no cover - librosa version differences
            tempo_windows.append((s0 + Wt / 2, float("nan")))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    Sh = np.abs(librosa.stft(yh, n_fft=2048, hop_length=hop))
    Sp = np.abs(librosa.stft(yp, n_fft=2048, hop_length=hop))
    sec_s = np.clip(np.floor(librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)).astype(int), 0, n - 1)
    mfcc = _per_second(librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), n_mfcc=13), sec_s, n)
    centroid = _per_second(librosa.feature.spectral_centroid(S=S, sr=sr), sec_s, n)[0]
    flatness = _per_second(librosa.feature.spectral_flatness(S=S), sec_s, n)[0]
    eh = _per_second((Sh ** 2).sum(0, keepdims=True), sec_s, n)[0]
    ep = _per_second((Sp ** 2).sum(0, keepdims=True), sec_s, n)[0]
    harmonic_ratio = eh / (eh + ep + 1e-12)
    rms = _per_second(librosa.feature.rms(S=S), sec_s, n)[0]
    cq = np.abs(librosa.cqt(yh, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C2"), n_bins=72, bins_per_octave=12))
    sec_c = np.clip(np.floor(librosa.frames_to_time(np.arange(cq.shape[1]), sr=sr, hop_length=hop)).astype(int), 0, n - 1)
    cq1 = _per_second(cq, sec_c, n)
    midi = librosa.note_to_midi("C2") + np.arange(72)
    register = (cq1 * midi[:, None]).sum(0) / (cq1.sum(0) + 1e-9)
    spread = np.sqrt(((cq1 * (midi[:, None] - register) ** 2).sum(0)) / (cq1.sum(0) + 1e-9))
    X = np.column_stack([(mfcc[i] - mfcc[i].mean()) / (mfcc[i].std() + 1e-9) for i in range(1, 13)])
    L = max(int(novelty_half_window_s), 2)
    timbre_novelty = foote_novelty(X, L) if n > 2 * L + 1 else np.zeros(n)
    return {
        "t": np.arange(n) + 0.5, "duration": dur, "chroma": chn, "chroma_entropy": entropy,
        "tonal_clarity": clarity, "keys": keys, "hcdf": hcdf, "tempogram": tg1, "tempi": tempi[valid],
        "local_tempo": local_tempo, "pulse_clarity": pulse_clarity, "tempo_windows": tempo_windows,
        "mfcc": mfcc, "centroid": centroid, "flatness": flatness, "harmonic_ratio": harmonic_ratio,
        "register_midi": register, "register_spread": spread, "rms": rms, "timbre_novelty": timbre_novelty,
    }


def section_summary(tc: dict, boundaries) -> list[dict]:
    """One row per section between `boundaries` (seconds): key, tempo, harmony, timbre, register.

    Args:
        tc (dict): From :func:`music_timecourse`.
        boundaries: Section boundary times in seconds; the recording's start and end are added.

    Returns:
        list: Dicts with ``section``, ``start``, ``end``, ``key``, ``key_r``, ``tempo_bpm_median``,
        ``pulse_clarity``, ``chroma_entropy``, ``hcdf``, ``centroid_hz``, ``flatness``,
        ``harmonic_ratio``, ``register_midi``, ``register_spread``, ``rms_db``.
    """
    t = tc["t"]
    edges = np.r_[0.0, np.asarray(boundaries, dtype=float), tc["duration"]]
    tw = np.array([w[0] for w in tc["tempo_windows"]]) if tc["tempo_windows"] else np.array([])
    tv = np.array([w[1] for w in tc["tempo_windows"]]) if tc["tempo_windows"] else np.array([])
    rows = []
    for i in range(len(edges) - 1):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if not m.any():
            continue
        name, r = key_from_chroma(tc["chroma"][:, m].mean(1))
        mt = (tw >= edges[i]) & (tw < edges[i + 1]) if len(tw) else np.array([], dtype=bool)
        rows.append({
            "section": i + 1, "start": float(edges[i]), "end": float(edges[i + 1]), "key": name, "key_r": r,
            "tempo_bpm_median": float(np.nanmedian(tv[mt])) if mt.any() else float("nan"),
            "pulse_clarity": float(np.nanmean(tc["pulse_clarity"][m])), "chroma_entropy": float(tc["chroma_entropy"][m].mean()),
            "hcdf": float(tc["hcdf"][m].mean()), "centroid_hz": float(tc["centroid"][m].mean()),
            "flatness": float(tc["flatness"][m].mean()), "harmonic_ratio": float(tc["harmonic_ratio"][m].mean()),
            "register_midi": float(tc["register_midi"][m].mean()), "register_spread": float(tc["register_spread"][m].mean()),
            "rms_db": float(20 * np.log10(tc["rms"][m].mean() + 1e-9)),
        })
    return rows


def timecourse_figures(tc: dict, out_dir, prefix: str = "", title: str = "", boundaries=()) -> dict:
    """Write the chromagram, tempogram and time-course figures, and the raw strips.

    Args:
        tc (dict): From :func:`music_timecourse`.
        out_dir: Folder for the PNGs.
        prefix (str): Filename prefix.
        title (str): Figure title prefix.
        boundaries: Section boundaries to draw as vertical lines.

    Returns:
        dict: Paths of the files written (``chromagram``, ``tempogram``, ``timecourse``,
        ``chromagram_raw``, ``tempogram_raw``).
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t, dur = tc["t"], tc["duration"]
    sm = lambda v, w=15: np.convolve(np.nan_to_num(v), np.ones(w) / w, "same")  # noqa: E731
    paths = {}
    fig, axs = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
    axs[0].imshow(tc["chroma"], aspect="auto", origin="lower", cmap="magma", extent=[0, dur, -0.5, 11.5], interpolation="nearest")
    axs[0].set_yticks(range(12)); axs[0].set_yticklabels(KEY_NAMES); axs[0].set_ylim(-0.5, 15.5)
    for k, (tk, name, r) in enumerate(tc["keys"]):
        if k % 6 == 0:
            axs[0].text(tk, 11.7, name, fontsize=6, rotation=90, va="bottom", ha="center")
    axs[0].set_title(f"{title}: chromagram (harmonic component, 1 s) with key estimates and sections", pad=4)
    for b in boundaries:
        axs[0].axvline(b, color="cyan", lw=.8); axs[1].axvline(b, color="tab:blue", lw=.8)
    axs[1].plot(t, tc["chroma_entropy"], color="grey", lw=.8, label="chroma entropy")
    axs[1].plot(t, sm(tc["hcdf"], 10) / (np.nanmax(sm(tc["hcdf"], 10)) + 1e-9), color="tab:purple", lw=.8, label="harmonic change (normalised)")
    axs[1].plot(t, tc["tonal_clarity"] * 3, color="tab:orange", lw=.8, label="tonal clarity (x3)")
    axs[1].legend(ncol=3, fontsize=8); axs[1].set_xlabel("time (s)"); axs[1].set_xlim(0, dur); axs[1].set_ylim(0, 1.05)
    fig.tight_layout(); paths["chromagram"] = os.path.join(out_dir, f"{prefix}chromagram.png"); fig.savefig(paths["chromagram"]); plt.close(fig)
    tgn = tc["tempogram"] / (tc["tempogram"].max(0, keepdims=True) + 1e-9)
    fig, axs = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
    axs[0].imshow(tgn, aspect="auto", origin="lower", cmap="magma", extent=[0, dur, tc["tempi"].min(), tc["tempi"].max()], interpolation="nearest")
    if tc["tempo_windows"]:
        axs[0].plot([w[0] for w in tc["tempo_windows"]], [w[1] for w in tc["tempo_windows"]], color="white", lw=1)
    axs[0].set_ylabel("tempo (BPM)"); axs[0].set_title(f"{title}: tempogram with windowed tempo estimate (white)")
    axs[1].plot(t, sm(tc["pulse_clarity"], 10), color="tab:blue", lw=1, label="pulse clarity (peak/mean, 10 s smooth)")
    axs[1].legend(fontsize=8); axs[1].set_xlabel("time (s)"); axs[1].set_xlim(0, dur)
    for b in boundaries:
        axs[0].axvline(b, color="cyan", lw=.8); axs[1].axvline(b, color="tab:blue", lw=.8)
    fig.tight_layout(); paths["tempogram"] = os.path.join(out_dir, f"{prefix}tempogram.png"); fig.savefig(paths["tempogram"]); plt.close(fig)
    lvl = 20 * np.log10(tc["rms"] + 1e-9); lvl = np.maximum(lvl, np.nanpercentile(lvl, 1) - 5)
    fig, axs = plt.subplots(5, 1, figsize=(20, 11), sharex=True)
    axs[0].plot(t, lvl, color="tab:blue", lw=.5, alpha=.4); axs[0].plot(t, sm(lvl), color="tab:blue", lw=1.2); axs[0].set_ylabel("level (dB)")
    axs[1].plot(t, sm(tc["register_midi"]), color="tab:purple", lw=1.2, label="register (energy-weighted MIDI)")
    axs[1].fill_between(t, sm(tc["register_midi"] - tc["register_spread"]), sm(tc["register_midi"] + tc["register_spread"]), color="tab:purple", alpha=.2, label="± spread")
    axs[1].set_ylabel("pitch (MIDI)"); axs[1].legend(fontsize=8)
    axs[2].plot(t, sm(tc["centroid"]), color="tab:orange", lw=1.2, label="spectral centroid (Hz)"); axs[2].legend(fontsize=8); axs[2].set_ylabel("brightness (Hz)")
    axs[3].plot(t, sm(tc["harmonic_ratio"]), color="tab:green", lw=1.2, label="harmonic share (HPSS)")
    axs[3].plot(t, sm(tc["flatness"] * 10), color="grey", lw=1.2, label="spectral flatness (x10)"); axs[3].set_ylim(0, 1); axs[3].legend(fontsize=8); axs[3].set_ylabel("timbre")
    axs[4].plot(t, tc["timbre_novelty"], color="tab:red", lw=1, label="timbre novelty (MFCC)")
    axs[4].plot(t, sm(tc["hcdf"]) / (np.nanmax(sm(tc["hcdf"])) + 1e-9), color="tab:purple", lw=1, label="harmonic change (normalised)")
    axs[4].legend(fontsize=8); axs[4].set_ylabel("change"); axs[4].set_xlabel("time (s)"); axs[4].set_xlim(0, dur)
    for ax in axs:
        for b in boundaries:
            ax.axvline(b, color="tab:blue", lw=.6, alpha=.7)
    axs[0].set_title(f"{title}: musical time-course")
    fig.tight_layout(); paths["timecourse"] = os.path.join(out_dir, f"{prefix}timecourse.png"); fig.savefig(paths["timecourse"]); plt.close(fig)
    paths["chromagram_raw"] = os.path.join(out_dir, f"{prefix}chromagram_raw.png")
    plt.imsave(paths["chromagram_raw"], tc["chroma"][::-1] ** 0.7, cmap="magma")
    paths["tempogram_raw"] = os.path.join(out_dir, f"{prefix}tempogram_raw.png")
    plt.imsave(paths["tempogram_raw"], tgn[::-1], cmap="magma")
    return paths
