"""Per-track visual thumbnails: a piece at a glance.

Each track becomes one card, in a choice of representations:

- ``mel`` / ``chroma`` / ``tempo`` / ``combo`` — the spectrogram family
  (timbre & texture, harmony over time, rhythmic periodicity, all three);
- ``barcode`` — harmony as color: each moment's hue is its position on the
  circle of fifths, saturation its tonal focus, brightness its loudness;
- ``ssm`` — self-similarity matrix: musical *form* as texture (repetition
  blocks, sections, drone slabs);
- ``trajectory`` — the piece as a smoothed path through its own timbre
  space (MFCC PCA), colored start → end;
- ``keyscape`` — Sapp-style triangle: every analysis window at every time
  scale colored by its Krumhansl–Schmuckler key (hue = tonic on the circle
  of fifths, light = major, dark = minor);
- ``rhythm`` — Poincaré portrait of successive inter-onset intervals:
  metric playing collapses to points, rubato spreads into clouds;
- ``wave`` — Freesound-style waveform: the amplitude envelope with each
  moment colored by its spectral centroid (dark blue = dark timbre,
  red = bright), so timbre rides on the waveform itself;
- ``vinyl`` — the track as a tonality disc (12 o'clock = start,
  clockwise; hue = harmony on the circle of fifths, radius = loudness),
  with the Freesound-style centroid-colored waveform as the strip
  underneath;
- ``spiral`` — time-integrated energy on the Shepard helix (angle = pitch
  class, radius = octave): the only view that shows register;
- ``tonnetz`` — the harmony's path on the circle-of-fifths plane of the
  tonal centroid (Harte's tonnetz), colored start → end;
- ``arcs`` — Shape-of-Song-style arc diagram: repeated sections found in
  the self-similarity structure joined by arcs over the timeline.

- ``tarsom`` — the track's position on Schaeffer's seven morphological
  criteria (TARSOM: masse, timbre harmonique, grain, allure, dynamique,
  profil mélodique, profil de masse) as a centre–periphery rose: each
  criterion a sector radiating from the centre pole (tonic, dark, smooth,
  slow, percussive, static, fixed) toward its periphery pole (complex,
  bright, granular, fast, soft, mobile, evolving);
- ``schaeffer`` — the track's sound objects on a typo-morphology (TARTYP)
  timeline: three mass lanes (N tonic / Y variable / X complex), facture
  as mark style (impulse ticks, hatched iterations, solid held blocks),
  with a TARTYP-grid fingerprint inset. Uses the same signal proxies and
  thresholds as ``ambiscape.music.tartyp_profile``.

The ``rhythm`` card carries a beat-wheel inset: onset phases on the
dominant-period circle with the pulse-clarity resultant arrow.

Albums additionally get a contact sheet, and :func:`poster` stacks every
track's barcode into a single collection image where albums read as color
families. Thumbnails are meant for browsing a collection visually.
"""
from __future__ import annotations

import colorsys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import GRID, INK, MUT, album_colors
from .io import Collection, Track, load

#: available card styles
STYLES = ("mel", "chroma", "tempo", "combo", "barcode", "ssm",
          "trajectory", "keyscape", "rhythm", "wave",
          "vinyl", "spiral", "tonnetz", "arcs", "schaeffer", "tarsom")
#: taller cards for the square-ish representations
_TALL = {"combo": 5.4, "ssm": 5.2, "trajectory": 5.2, "keyscape": 5.2,
         "rhythm": 5.2, "vinyl": 5.6, "spiral": 5.6, "tonnetz": 5.2,
         "arcs": 4.4, "schaeffer": 4.4, "tarsom": 5.6}

_FIFTHS = 2 * np.pi * (7 * np.arange(12) % 12) / 12


# --------------------------------------------------------------------------
# shared feature helpers

def _feats_2hz(y, sr):
    """Chroma, MFCC and RMS aggregated to 2 Hz frames."""
    import librosa
    hop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)[1:]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    step = max(1, int(0.5 * sr / hop))
    n = max(1, C.shape[1] // step)
    agg = lambda X: np.stack([X[:, i * step:(i + 1) * step].mean(1)
                              for i in range(n)], 1)
    return agg(C), agg(M), np.array([rms[i * step:(i + 1) * step].mean()
                                     for i in range(n)])


def barcode_rgb(C, rms):
    """RGB strip (n×3) from 2 Hz chroma + RMS — the barcode's colors."""
    z = (C * np.exp(1j * _FIFTHS)[:, None]).sum(0) / (C.sum(0) + 1e-9)
    hue = (np.angle(z) / (2 * np.pi)) % 1.0
    sat = np.clip(np.abs(z) * 1.6, 0, 1)
    val = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0.12, 1) ** 0.6
    return np.array([colorsys.hsv_to_rgb(h, s, v)
                     for h, s, v in zip(hue, sat, val)])


def wave_colors(y, sr, cols=1200):
    """Amplitude envelope + turbo-mapped spectral-centroid colors."""
    import librosa
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    step = max(1, len(y) // cols)
    env = np.abs(y[: len(y) // step * step]).reshape(-1, step).max(axis=1)
    ci = np.interp(np.linspace(0, len(cent) - 1, len(env)),
                   np.arange(len(cent)), cent)
    norm = np.clip((np.log2(ci + 1e-9) - np.log2(300))
                   / (np.log2(4000) - np.log2(300)), 0, 1)
    return env / (env.max() + 1e-9), matplotlib.colormaps["turbo"](norm)


def _key_profiles():
    """24 z-normalised Krumhansl–Schmuckler profiles (12 major, 12 minor)."""
    from .features import _MAJ, _MIN
    P = np.array([np.roll(_MAJ, i) for i in range(12)]
                 + [np.roll(_MIN, i) for i in range(12)])
    return (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)


def keyscape_rgb(C, levels=48):
    """Sapp-style keyscape image (levels×n×3) from 2 Hz chroma."""
    n = C.shape[1]
    P = _key_profiles()
    cum = np.concatenate([np.zeros((12, 1)), np.cumsum(C, axis=1)], axis=1)
    img = np.ones((levels, n, 3))
    widths = np.unique(np.linspace(max(2, n // 80), n, levels).astype(int))[::-1]
    for li, w in enumerate(np.resize(widths, levels)):
        x0 = np.arange(0, n - w + 1)
        W = ((cum[:, x0 + w] - cum[:, x0]) / w).T          # windows × 12
        Wn = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-9)
        idx = np.argmax(Wn @ P.T, axis=1)
        tonic, minor = idx % 12, idx >= 12
        hue = ((7 * tonic) % 12) / 12.0
        hsv = np.stack([hue,
                        np.where(minor, 0.75, 0.55),
                        np.where(minor, 0.55, 0.95)], axis=1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        img[li, x0 + w // 2] = rgb
        # fill edges of the row so the triangle reads solid
        img[li, :w // 2] = np.nan
        img[li, w // 2 + len(x0):] = np.nan
    return img


# --------------------------------------------------------------------------
# card rendering

def _panels(y, sr, style):
    """Image panels (label, matrix, vmin) for the spectrogram styles."""
    import librosa
    out = []
    if style in ("mel", "combo"):
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=8000)
        db = librosa.power_to_db(M, ref=np.max)
        out.append(("mel", db, db.max() - 80))
    if style in ("chroma", "combo"):
        out.append(("chroma", librosa.feature.chroma_stft(y=y, sr=sr), 0.0))
    if style in ("tempo", "combo"):
        env = librosa.onset.onset_strength(y=y, sr=sr)
        T = librosa.feature.tempogram(onset_envelope=env, sr=sr,
                                      win_length=int(8 * sr / 512))
        bpm = librosa.tempo_frequencies(T.shape[0], sr=sr)
        out.append(("tempo", T[(bpm >= 30) & (bpm <= 300)], 0.0))
    return out


def _mod_share(env, rate, lo, hi):
    """Share of envelope-modulation energy in [lo, hi] Hz + peak freq."""
    env = env - env.mean()
    if len(env) < 16 or not np.any(env):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fq = np.fft.rfftfreq(len(env), 1 / rate)
    tot = spec[fq > 0.2].sum() + 1e-12
    band, bf = spec[(fq >= lo) & (fq < hi)], fq[(fq >= lo) & (fq < hi)]
    peak = float(bf[np.argmax(band)]) if len(band) else 0.0
    return float(band.sum() / tot), peak


def _tarsom_criteria(y, sr, hop=512):
    """Signal proxies for Schaeffer's seven morphological criteria."""
    import librosa
    from scipy.ndimage import median_filter
    from ambiscape.music import ONSET_FLOOR
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = librosa.feature.spectral_flatness(S=S)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rms = librosa.feature.rms(S=S)[0]
    fine = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                    hop_length=hop, units="frames")
    fr = fr[env[fr] >= ONSET_FLOOR]
    frame_t = hop / sr
    atks = []
    for o in fr:
        w = rms[o:o + int(0.5 / frame_t)]
        if len(w) > 2:
            atks.append(np.argmax(w) * frame_t)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr)
    pc = median_filter(np.argmax(chroma, axis=0), size=9, mode="nearest")
    dur = len(y) / sr
    grain, _ = _mod_share(fine, sr / 64, 20, 100)
    _, allure_hz = _mod_share(rms, 1 / frame_t, 0.5, 8)
    return {
        "masse": float(np.median(flat)),
        "timbre": float(cent.mean()),
        "grain": grain,
        "allure": allure_hz,
        "dynamique": float(np.median(atks)) if atks else 0.3,
        "profil_mel": float(np.sum(np.diff(pc) != 0) / max(dur, 1e-9)),
        "profil_masse": float(np.std(np.log10(flat + 1e-6))),
    }


#: (key, name, gloss, lo, hi, log, left anchor, right anchor)
TARSOM_ROWS = [
    ("masse", "masse", "mass", 1e-4, 2e-2, True, "tonic", "complex"),
    ("timbre", "timbre harmonique", "harmonic timbre", 500, 2000, True,
     "dark", "bright"),
    ("grain", "grain", "surface texture", 0.0, 0.2, False,
     "smooth", "granular"),
    ("allure", "allure", "characteristic pulsation", 0.3, 3.0, True,
     "slow", "fast"),
    ("dynamique", "dynamique", "attack facture", 0.02, 0.3, True,
     "percussive", "soft onset"),
    ("profil_mel", "profil mélodique", "pitch mobility", 1.0, 4.0, False,
     "static", "mobile"),
    ("profil_masse", "profil de masse", "spectral evolution", 0.2, 1.2,
     False, "fixed", "evolving"),
]


def _schaeffer_objects(y, sr, hop=512):
    """Onset-bounded sound objects with (t0, t1, mass, facture).

    The object-level detail behind ``ambiscape.music.tartyp_profile`` —
    same segmentation, same corpus-calibrated thresholds (imported from
    ambiscape), but keeping every object rather than the aggregate."""
    import librosa
    from ambiscape.music import (ONSET_FLOOR, TARTYP_COMPLEX, TARTYP_DRIFT,
                                 TARTYP_IMPULSE_S, TARTYP_ITER, TARTYP_TONIC)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = librosa.feature.spectral_flatness(S=S)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    fine_hop = hop // 8
    fine = librosa.feature.rms(y=y, frame_length=fine_hop * 4,
                               hop_length=fine_hop)[0]
    fine_rate = sr / fine_hop
    frame_t = hop / sr
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                       hop_length=hop, units="frames")
    peaks = peaks[env[peaks] >= ONSET_FLOOR]
    onsets = (librosa.onset.onset_backtrack(peaks, env)
              if len(peaks) else peaks)
    bounds = np.unique(np.concatenate([[0], onsets, [S.shape[1]]]))

    def iter_ratio(a, b):
        seg = fine[int(a * hop / fine_hop):int(b * hop / fine_hop)]
        seg = seg - seg.mean()
        if len(seg) < 16 or not np.any(seg):
            return 0.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fq = np.fft.rfftfreq(len(seg), 1 / fine_rate)
        tot = spec[fq > 0.2].sum()
        return float(spec[(fq >= 4) & (fq < 20)].sum() / tot) if tot else 0.0

    objects = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        dur = (b - a) * frame_t
        if dur < 0.05:
            continue
        fmed = float(np.median(flat[a:b]))
        drift = float(np.std(np.log2(cent[a:b] + 1e-6)))
        if dur < TARTYP_IMPULSE_S:
            fact = "imp"
        elif iter_ratio(a, b) > TARTYP_ITER:
            fact = "iter"
        else:
            fact = "held"
        if fmed >= TARTYP_COMPLEX:
            mass = "X"
        elif fmed >= TARTYP_TONIC or drift > TARTYP_DRIFT:
            mass = "Y"
        else:
            mass = "N"
        objects.append((a * frame_t, b * frame_t, mass, fact))
    return objects


def _repeat_runs(F, min_len=8, min_gap=8, thresh=0.85, top=12):
    """Repeated-section pairs from a feature sequence (diagonal runs of
    the cosine self-similarity matrix). Returns (i, j, length) in frames."""
    Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
    S = Fn.T @ Fn
    n = S.shape[0]
    runs = []
    for d in range(min_gap, n - min_len):
        m = np.diagonal(S, offset=d) > thresh
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]:
                    j += 1
                if j - i >= min_len:
                    runs.append((i, i + d, j - i))
                i = j
            else:
                i += 1
    runs.sort(key=lambda r: -r[2])
    picked = []
    for i, j, L in runs:
        if all(abs(i - a) > L / 2 or abs(j - b) > L / 2
               for a, b, _ in picked):
            picked.append((i, j, L))
        if len(picked) >= top:
            break
    return picked, n


def _draw_main(ax, y, sr, style, color="#2a78d6"):
    """Draw a non-spectrogram main panel onto ``ax``."""
    import librosa
    from scipy.ndimage import uniform_filter, uniform_filter1d

    if style == "barcode":
        C, _, rms = _feats_2hz(y, sr)
        ax.imshow(barcode_rgb(C, rms)[None, :, :], aspect="auto")

    elif style == "ssm":
        C, M, _ = _feats_2hz(y, sr)
        F = np.vstack([C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9),
                       M / (np.abs(M).max() + 1e-9)])
        Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
        S = uniform_filter(Fn.T @ Fn, size=3)
        ax.imshow(S, cmap="magma", origin="lower", aspect="auto")

    elif style == "trajectory":
        _, M, _ = _feats_2hz(y, sr)
        Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
        U, S_, Vt = np.linalg.svd(Z.T, full_matrices=False)
        P = uniform_filter1d(U[:, :2] * S_[:2], size=5, axis=0)
        ax.plot(P[:, 0], P[:, 1], color=MUT, alpha=0.35, linewidth=0.8)
        ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="magma",
                   s=9, alpha=0.9)
        ax.margins(0.06)

    elif style == "keyscape":
        C, _, _ = _feats_2hz(y, sr)
        ax.imshow(keyscape_rgb(C), aspect="auto", origin="upper",
                  interpolation="nearest")

    elif style == "vinyl":
        C, _, rms = _feats_2hz(y, sr)
        rgb = barcode_rgb(C, rms)
        n = len(rgb)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        loud = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0.15, 1)
        ax.bar(theta, 0.45 + 0.55 * loud, width=2 * np.pi / n * 1.5,
               bottom=0.55, color=rgb, linewidth=0)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.6)
        ax.spines["polar"].set_visible(False)

    elif style == "spiral":
        CQ = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz("C2"),
                                n_bins=5 * 36, bins_per_octave=36))
        e = CQ.mean(axis=1)
        e = e / (e.max() + 1e-9)
        k = np.arange(len(e))
        ax.scatter(2 * np.pi * (k % 36) / 36, 0.3 + (k / 36) / 5,
                   s=4 + 220 * e ** 1.4, c=e, cmap="magma", alpha=0.85,
                   linewidths=0)
        pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        for name, pc in pcs.items():
            ax.text(2 * np.pi * pc / 12, 1.44, name, ha="center",
                    va="center", fontsize=7, color=MUT)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.5)
        ax.spines["polar"].set_visible(False)

    elif style == "tonnetz":
        yh = librosa.effects.harmonic(y)
        T6 = librosa.feature.tonnetz(y=yh, sr=sr)
        step = max(1, T6.shape[1] // 400)
        P = uniform_filter1d(T6[:2, ::step], size=7, axis=1).T
        circ = plt.Circle((0, 0), 1.0, fill=False, color=GRID, linewidth=1)
        ax.add_patch(circ)
        for pc in range(12):
            a = 2 * np.pi * pc / 12
            ax.text(1.12 * np.sin(a), 1.12 * np.cos(a),
                    ["C", "G", "D", "A", "E", "B", "F#", "C#",
                     "G#", "D#", "A#", "F"][pc],
                    ha="center", va="center", fontsize=6.5, color=MUT)
        ax.plot(P[:, 0], P[:, 1], color=MUT, alpha=0.3, linewidth=0.8)
        ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="magma",
                   s=8, alpha=0.9)
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")

    elif style == "arcs":
        C, M, _ = _feats_2hz(y, sr)
        runs, n = _repeat_runs(np.vstack([C, M / (np.abs(M).max() + 1e-9)]))
        hop = max(1, len(y) // 1200)
        env = np.abs(y[: len(y) // hop * hop]).reshape(-1, hop).max(axis=1)
        env = env / (env.max() + 1e-9)
        sc = len(env) / max(n, 1)
        ax.fill_between(np.arange(len(env)), 0, 0.16 * env, color=color,
                        linewidth=0)
        if runs:
            Lmax = runs[0][2]
            for i, j, L in runs:
                a, b = (i + L / 2) * sc, (j + L / 2) * sc
                th = np.linspace(0, np.pi, 60)
                ax.plot((a + b) / 2 + (b - a) / 2 * np.cos(th),
                        0.18 + 0.8 * ((b - a) / len(env)) * np.sin(th),
                        color=color, alpha=0.55,
                        linewidth=0.8 + 2.6 * L / Lmax,
                        solid_capstyle="round")
        else:
            ax.text(0.5, 0.6, "no strong repetitions found",
                    transform=ax.transAxes, ha="center", color=MUT,
                    fontsize=9)
        ax.set_xlim(0, len(env)); ax.set_ylim(0, 1.02)

    elif style == "schaeffer":
        objects = _schaeffer_objects(y, sr)
        MASS_Y = {"N": 2, "Y": 1, "X": 0}
        MASS_C = {"N": "#2a78d6", "Y": "#eda100", "X": "#e34948"}
        total = max((b for _, b, _, _ in objects), default=1.0)
        for t0, t1, mass, fact in objects:
            yy = MASS_Y[mass]
            if fact == "imp":
                ax.plot([t0, t0], [yy - 0.36, yy + 0.36],
                        color=MASS_C[mass], linewidth=1.1, alpha=0.75,
                        solid_capstyle="butt")
            else:
                from matplotlib.patches import FancyBboxPatch, Rectangle
                ax.add_patch(Rectangle(
                    (t0, yy - 0.31), t1 - t0, 0.62,
                    facecolor=MASS_C[mass], edgecolor="none", alpha=0.8,
                    hatch="///" if fact == "iter" else None))
        for mass, lab in (("N", "N — tonic"), ("Y", "Y — variable"),
                          ("X", "X — complex")):
            ax.text(0.005 * total, MASS_Y[mass] + 0.40, lab, ha="left",
                    va="bottom", fontsize=7, color=MUT, zorder=5,
                    bbox=dict(facecolor="white", alpha=0.75,
                              edgecolor="none", pad=1))
            ax.axhline(MASS_Y[mass], color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlim(0, total)
        ax.set_ylim(-0.6, 2.6)
        # TARTYP-grid fingerprint inset (mass rows x facture cols)
        shares = np.zeros((3, 3))
        FI = {"held": 0, "imp": 1, "iter": 2}
        for t0, t1, mass, fact in objects:
            shares[2 - MASS_Y[mass], FI[fact]] += t1 - t0
        shares = shares / (shares.sum() + 1e-9)
        ia = ax.inset_axes([0.858, 0.03, 0.132, 0.30])
        ia.imshow(shares ** 0.5, cmap="Blues", vmin=0, vmax=1,
                  aspect="auto")
        for r in range(3):
            for c in range(3):
                if shares[r, c] >= 0.0005:
                    ia.text(c, r, f"{shares[r, c] * 100:.0f}",
                            ha="center", va="center", fontsize=5.5,
                            color=INK)
        ia.set_xticks(range(3), ["h", "'", "''"], fontsize=5.5)
        ia.set_yticks(range(3), ["N", "Y", "X"], fontsize=5.5)
        ia.tick_params(length=0, colors=MUT)
        for s in ia.spines.values():
            s.set_color(GRID)

    elif style == "tarsom":
        crit = _tarsom_criteria(y, sr)
        n = len(TARSOM_ROWS)
        wedge = 2 * np.pi / n
        for r, (key, name, gloss, lo, hi, logsc, la, ra) in \
                enumerate(TARSOM_ROWS):
            v = crit[key]
            if logsc:
                x = (np.log10(max(v, 1e-12)) - np.log10(lo)) \
                    / (np.log10(hi) - np.log10(lo))
            else:
                x = (v - lo) / (hi - lo)
            x = float(np.clip(x, 0.06, 1.0))
            th = r * wedge
            ax.bar(th, 1.0, width=wedge * 0.86, bottom=0,
                   color=GRID, alpha=0.35, linewidth=0)
            ax.bar(th, x, width=wedge * 0.86, bottom=0,
                   color=color, alpha=0.9, linewidth=0)
            sx = np.sin(th)
            ha = "left" if sx > 0.25 else "right" if sx < -0.25 else "center"
            for txt, dy, fs, col, wt in ((name, 7, 7.5, INK, "semibold"),
                                         (f"{la} → {ra}", -6, 5.5, MUT,
                                          "normal")):
                ax.annotate(txt, xy=(th, 1.08), xytext=(0, dy),
                            textcoords="offset points", ha=ha, va="center",
                            fontsize=fs, color=col, fontweight=wt,
                            annotation_clip=False)
        ring = np.linspace(0, 2 * np.pi, 120)
        for rr in (0.5, 1.0):
            ax.plot(ring, np.full_like(ring, rr), color=GRID,
                    linewidth=0.7, zorder=0)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.5)
        ax.spines["polar"].set_visible(False)

    elif style == "wave":
        env, colors = wave_colors(y, sr)
        ax.bar(np.arange(len(env)), 2 * env, bottom=-env, width=1.0,
               color=colors, linewidth=0)
        ax.set_xlim(0, len(env))
        ax.set_ylim(-1.05, 1.05)

    elif style == "rhythm":
        from ambiscape.music import ONSET_FLOOR
        env = librosa.onset.onset_strength(y=y, sr=sr)
        fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                        units="frames")
        fr = fr[env[fr] >= ONSET_FLOOR]
        t = librosa.frames_to_time(fr, sr=sr)
        ioi = np.diff(t)
        ioi = ioi[(ioi > 0.05) & (ioi < 2.5)]
        if len(ioi) > 3:
            ax.scatter(ioi[:-1], ioi[1:], c=np.arange(len(ioi) - 1),
                       cmap="magma", s=12, alpha=0.75)
            ax.plot([0.05, 2.5], [0.05, 2.5], color=GRID, linewidth=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.05, 2.5); ax.set_ylim(0.05, 2.5)
            # beat-wheel inset: onset phases + pulse-clarity arrow
            try:
                from ambiscape.circstats import mean_resultant
                from ambiscape.music import dominant_period
                w = env[fr]
                p0 = dominant_period(env, sr)
                best = (p0, -1.0, 0.0)
                for p in (p0 / 2, p0, p0 * 2):
                    if not 60 / 200 <= p <= 60 / 40:
                        continue
                    mu, R = mean_resultant(2 * np.pi * (t / p % 1.0),
                                           weights=w)
                    if R > best[1]:
                        best = (p, R, mu)
                p, R, mu = best
                ph = 2 * np.pi * (t / p % 1.0)
                ia = ax.inset_axes([0.70, 0.70, 0.29, 0.29],
                                   projection="polar")
                bins = np.linspace(0, 2 * np.pi, 25)
                hist, _ = np.histogram(ph, bins=bins, weights=w)
                hist = hist / (hist.max() + 1e-9)
                ia.bar((bins[:-1] + bins[1:]) / 2, hist,
                       width=np.diff(bins), color=color, alpha=0.7,
                       linewidth=0)
                ia.annotate("", xy=(mu, min(R * 3, 1.0)), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=INK,
                                            lw=1.4))
                ia.set_theta_zero_location("N")
                ia.set_theta_direction(-1)
                ia.set_ylim(0, 1.05)
                ia.set_xticks([]); ia.set_yticks([])
                ia.spines["polar"].set_visible(False)
                ia.set_title(f"R={R:.2f}", fontsize=6, color=MUT, pad=1)
            except Exception:                             # noqa: BLE001
                pass
        else:
            ax.text(0.5, 0.5, "too few onsets", transform=ax.transAxes,
                    ha="center", color=MUT, fontsize=9)


def render_track(track: Track, color: str, out_path: str | Path,
                 sr: int = 22050, note: str = "", style: str = "mel") -> Path:
    """One card: the chosen representation over a waveform strip."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = load(track, sr=sr)
        spectro = style in ("mel", "chroma", "tempo", "combo")
        panels = _panels(y, sr, style) if spectro else [(style, None, None)]
        has_strip = style != "wave"
        hop = max(1, len(y) // 1200)
        env = np.abs(y[: len(y) // hop * hop]).reshape(-1, hop).max(axis=1)

        heights = {"mel": 4.2, "chroma": 2.4, "tempo": 2.4}
        ratios = ([heights[n] for n, _, _ in panels] if spectro
                  else [4.2 if style in ("barcode", "wave") else 6.0])
        if has_strip:
            ratios = ratios + [1.0]
        fig_h = _TALL.get(style, 3.6)
        fig = plt.figure(figsize=(6.4, fig_h), dpi=100)
        top = 0.905 if fig_h > 4 else 0.86
        gs = fig.add_gridspec(len(panels) + int(has_strip), 1,
                              height_ratios=ratios,
                              left=0.015, right=0.985, top=top, bottom=0.05,
                              hspace=0.10)
        for i, (name, M, vmin) in enumerate(panels):
            polar = style in ("vinyl", "spiral", "tarsom")
            ax = (fig.add_subplot(gs[i], projection="polar") if polar
                  else fig.add_subplot(gs[i]))
            if spectro:
                ax.imshow(M, origin="lower", aspect="auto", cmap="magma",
                          vmin=vmin if vmin else None)
                if style == "combo":
                    ax.text(0.006, 0.93, name, transform=ax.transAxes,
                            fontsize=7, color="white", va="top", alpha=0.8)
            else:
                _draw_main(ax, y, sr, style, color=color)
            if style != "rhythm":
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.tick_params(labelsize=6, colors=MUT)
            if not polar:
                for s in ax.spines.values():
                    s.set_visible(False)
        if has_strip:
            ax1 = fig.add_subplot(gs[-1])
            if style == "vinyl":
                wenv, wcolors = wave_colors(y, sr)
                ax1.bar(np.arange(len(wenv)), 2 * wenv, bottom=-wenv,
                        width=1.0, color=wcolors, linewidth=0)
                ax1.set_xlim(0, len(wenv))
                ax1.set_ylim(-1.05, 1.05)
            else:
                x = np.arange(len(env))
                ax1.fill_between(x, -env, env, color=color, linewidth=0)
                ax1.set_xlim(0, len(env))
            ax1.set_xticks([]); ax1.set_yticks([])
            for s in ax1.spines.values():
                s.set_visible(False)

        dur = len(y) / sr
        ty = 0.97 if fig_h > 4 else 0.955
        fig.text(0.015, ty, track.title, fontsize=11, color=INK,
                 fontweight="semibold", va="top")
        right = (f"{int(dur // 60)}:{int(dur % 60):02d}"
                 + (f" · {note}" if note else ""))
        fig.text(0.985, ty, f"{track.album} · {right}", fontsize=8.5,
                 color=MUT, va="top", ha="right")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# collection-level rendering

def _work(job):
    path, album, color, out, note, style = job
    try:
        p = render_track(Track(path=Path(path), album=album), color, out,
                         note=note, style=style)
        print(f"[{album}] {Path(path).stem}", flush=True)
        return str(p)
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR [{album}] {path}: {e}", flush=True)
        return None


def contact_sheet(paths: list[Path], out_path: str | Path, cols: int = 3):
    """Tile thumbnails into one album overview image."""
    from PIL import Image
    imgs = [Image.open(p) for p in paths]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)
    return Path(out_path)


def render_collection(coll: Collection, out_dir: str | Path,
                      notes: dict | None = None, workers: int = 4,
                      style: str = "mel") -> Path:
    """All thumbnails → ``<out_dir>/thumbnails/<album>/<track>.png``
    plus a contact sheet per album. ``notes`` maps (album, title) to a
    short annotation (e.g. the estimated key from ``features.json``)."""
    out_dir = Path(out_dir) / "thumbnails"
    colors = album_colors(coll.album_names)
    notes = notes or {}
    jobs = [(str(t.path), t.album, colors[t.album],
             out_dir / t.album / f"{t.title}.png",
             notes.get((t.album, t.title), ""), style) for t in coll.tracks]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_work, jobs))
    else:
        list(map(_work, jobs))
    for a in coll.albums:
        paths = [out_dir / a.name / f"{t.title}.png" for t in a.tracks]
        paths = [p for p in paths if p.exists()]
        if paths:
            contact_sheet(paths, out_dir / f"{a.name.replace('/', '_')}.png")
    return out_dir


def _strip_work(job):
    path, album, width = job
    try:
        t = Track(path=Path(path), album=album)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load(t)
            C, _, rms = _feats_2hz(y, sr)
        rgb = barcode_rgb(C, rms)
        xi = np.linspace(0, len(rgb) - 1, width)
        res = np.stack([np.interp(xi, np.arange(len(rgb)), rgb[:, c])
                        for c in range(3)], axis=1)
        print(f"[{album}] {t.title}", flush=True)
        return album, t.title, res
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR {path}: {e}", flush=True)
        return None


def _disc_work(job):
    path, album = job
    try:
        t = Track(path=Path(path), album=album)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load(t)
            C, _, rms = _feats_2hz(y, sr)
        print(f"[{album}] {t.title}", flush=True)
        return album, t.title, barcode_rgb(C, rms), np.clip(
            rms / (np.percentile(rms, 95) + 1e-9), 0.15, 1)
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR {path}: {e}", flush=True)
        return None


def _vinyl_poster(coll: Collection, out_dir: Path, workers: int) -> Path:
    """Disc-grid poster: one vinyl glyph per track, grouped by album."""
    jobs = [(str(t.path), t.album) for t in coll.tracks]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = [r for r in ex.map(_disc_work, jobs) if r]
    else:
        res = [r for r in map(_disc_work, jobs) if r]
    discs = {(a, t): (rgb, loud) for a, t, rgb, loud in res}

    cols = 6
    rows_per_album = [(a, (len(a.tracks) + cols - 1) // cols)
                      for a in coll.albums]
    total_rows = sum(r for _, r in rows_per_album)
    fig = plt.figure(figsize=(2.3 * cols, 2.6 * total_rows
                              + 0.5 * len(coll.albums)), dpi=110)
    gs = fig.add_gridspec(total_rows, cols, hspace=0.55, wspace=0.15)
    colors = album_colors(coll.album_names)
    row = 0
    for a, nrows in rows_per_album:
        for k, t in enumerate(a.tracks):
            d = discs.get((a.name, t.title))
            if d is None:
                continue
            rgb, loud = d
            ax = fig.add_subplot(gs[row + k // cols, k % cols],
                                 projection="polar")
            n = len(rgb)
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
            ax.bar(theta, 0.45 + 0.55 * loud, width=2 * np.pi / n * 1.5,
                   bottom=0.55, color=rgb, linewidth=0)
            ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
            ax.set_ylim(0, 1.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines["polar"].set_visible(False)
            ax.set_title(t.title[:26], fontsize=6.5, color=INK, pad=2)
            if k == 0:
                ax.text(-0.35, 0.5, a.name, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=8,
                        color=colors[a.name], fontweight="semibold")
        row += nrows
    out = Path(out_dir) / "poster.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def poster(coll: Collection, out_dir: str | Path, workers: int = 4,
           strip_w: int = 1200, strip_h: int = 16,
           style: str = "barcode") -> Path:
    """One image for the whole collection. ``style="barcode"`` stacks every
    track as a horizontal color strip; ``style="vinyl"`` lays the tracks
    out as a grid of disc glyphs. Albums read as color families either
    way. → ``<out_dir>/poster.png``"""
    if style == "vinyl":
        return _vinyl_poster(coll, Path(out_dir), workers)
    from PIL import Image, ImageDraw
    jobs = [(str(t.path), t.album, strip_w) for t in coll.tracks]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = [r for r in ex.map(_strip_work, jobs) if r]
    else:
        res = [r for r in map(_strip_work, jobs) if r]
    strips = {(a, t): s for a, t, s in res}

    margin, gap, header = 210, 3, 26
    n_rows = len(strips)
    H = 12 + len(coll.albums) * (header + gap) + n_rows * (strip_h + gap) + 12
    W = margin + strip_w + 14
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    colors = album_colors(coll.album_names)
    yy = 12
    for a in coll.albums:
        d.rectangle([margin, yy + 4, margin + 12, yy + 16],
                    fill=colors[a.name])
        d.text((margin + 20, yy + 4), a.name, fill=INK)
        yy += header + gap
        for t in a.tracks:
            s = strips.get((a.name, t.title))
            if s is None:
                continue
            arr = (np.repeat(s[None, :, :], strip_h, axis=0)
                   * 255).astype(np.uint8)
            img.paste(Image.fromarray(arr), (margin, yy))
            d.text((margin - 6, yy + 2), t.title[:30], fill=MUT, anchor="ra")
            yy += strip_h + gap
    out = Path(out_dir) / "poster.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def notes_from_features(feats: list[dict]) -> dict:
    """(album, track) → "key · bpm" annotation for thumbnail title bars.

    The BPM shown is the perceptually-weighted tempo estimate
    (``tempo_bpm``, librosa's prior-based estimator, which targets the
    felt beat rather than the subdivision the phase-lock peaks on);
    when pulse clarity is low (R < 0.1, i.e. rubato or drifting
    material) it is prefixed with ``~`` — a nominal tempo, not a felt
    one. Falls back to ``pulse_bpm`` for older feature files.
    """
    out = {}
    for f in feats:
        parts = []
        if f.get("key"):
            parts.append(f["key"])
        bpm = f.get("tempo_bpm") or f.get("pulse_bpm")
        if bpm:
            approx = "~" if f.get("pulse_R", 0.0) < 0.1 else ""
            parts.append(f"{approx}{bpm:.0f} bpm")
        out[(f["album"], f["track"])] = " · ".join(parts)
    return out
