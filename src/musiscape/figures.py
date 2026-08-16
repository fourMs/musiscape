"""Overview figures: fingerprints, landscape, affinity.

Categorical album colours use a fixed, colour-blind-validated order (never
cycled); the affinity matrix is a blue/red diverging scale around zero.
Past eight albums the palette folds—facet or filter rather than invent
a ninth hue.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .corpus import albums_of

# fixed categorical order (CVD-validated); do not re-order per plot
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUT, GRID = "#171614", "#898781", "#e1e0d9"
DIV_POS, DIV_NEG = "#2a78d6", "#e34948"

FINGERPRINT_MEASURES = [
    ("onset_rate", "note density (events/s)"),
    ("centroid_hz", "brightness (Hz)"),
    ("flatness", "inharmonic texture"),
    ("dyn_range_db", "dynamic range (dB)"),
    ("pulse_R", "pulse clarity"),
    ("chroma_entropy", "pitch-class entropy"),
]


def album_colors(names: list[str]) -> dict[str, str]:
    """Stable album→colour map in first-appearance order."""
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(names)}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def fingerprints(stats: dict, out_path: str | Path, title: str = ""):
    """Small-multiple bars: one panel per measure, one bar per album."""
    names = list(stats)
    colors = album_colors(names)
    n = len(FINGERPRINT_MEASURES)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(12.8, 2.2 * ((n + 2) // 3) + 1),
                             dpi=130)
    for ax, (key, label) in zip(np.ravel(axes), FINGERPRINT_MEASURES):
        vals = [stats[a][key]["mean"] for a in names]
        ax.barh(range(len(names))[::-1], vals,
                color=[colors[a] for a in names], height=0.62)
        ax.set_yticks(range(len(names))[::-1], names, fontsize=8)
        ax.set_title(label, fontsize=9, loc="left", color=INK)
        _style(ax)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def landscape_plot(feats: list[dict], land: dict, out_path: str | Path,
                   title: str = ""):
    """PCA scatter, one colour per album, direct legend."""
    names = albums_of(feats)
    colors = album_colors(names)
    xy = np.array(land["coords"])
    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=130)
    for a in names:
        idx = [i for i, f in enumerate(feats) if f["album"] == a]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=42, color=colors[a], label=a,
                   edgecolors="white", linewidths=1.2)
    e = land["explained"]
    ax.set_xlabel(f"PC1 ({e[0]:.0%})", color=MUT, fontsize=9)
    ax.set_ylabel(f"PC2 ({e[1]:.0%})", color=MUT, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def affinity_plot(affinity: dict, out_path: str | Path, title: str = ""):
    """Album-affinity matrix, diverging around zero, values in cells."""
    names = list(affinity)
    M = np.array([[affinity[a][b] for b in names] for a in names])
    lim = max(abs(M).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2.4,
                                    1.0 * len(names) + 1.6), dpi=130)
    ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "aff", [DIV_NEG, "#f0efec", DIV_POS]), vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right",
                  fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8, color=INK)
    ax.tick_params(colors=MUT)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-track analysis figures.
#
# The thumbnail cards in :mod:`thumbnails` are deliberately unlabelled: they
# are for browsing a collection, where axes would be noise at card size.
# These are the other thing --- figures you read numbers off --- so every
# axis carries its unit and time runs in mm:ss rather than in seconds,
# because a four-minute song on a seconds axis cannot be read.

#: Pitch-class names, bottom to top on a chromagram's y axis.
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B")

#: Tempo range drawn, in BPM. Outside this the tempogram is metrical
#: aliasing rather than anything anyone taps to.
BPM_RANGE = (30.0, 300.0)


def _mmss(x, _pos=None):
    """Tick formatter: seconds → ``m:ss``."""
    x = max(0.0, float(x))
    return f"{int(x // 60)}:{int(x % 60):02d}"


def _time_axis(ax, label="time (m:ss)"):
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.set_xlabel(label, color=MUT, fontsize=9)


def draw_tempogram(ax, y, sr, hop: int = 512, mark_bpm: float | None = None):
    """Autocorrelation tempogram onto ``ax``, labelled in BPM.

    Bright horizontal bands are the periodicities the onsets actually hold:
    a band that stays level across the whole width is a steady tempo, and
    one that bends is a band speeding up or slowing down. A tempo is drawn
    over it as a dashed line so the two can be compared.

    ``mark_bpm`` sets which tempo that line shows; the default is this
    figure's own estimate. Callers that quote a tempo elsewhere on the page
    should pass theirs, because the two are computed from different onset
    envelopes and a page that shows one number in its header and another on
    its plot contradicts itself.
    """
    from . import music as amusic
    times, bpm, T, t_est = amusic.tempogram(y, sr, hop=hop)
    keep = (bpm >= BPM_RANGE[0]) & (bpm <= BPM_RANGE[1])
    b = bpm[keep]
    im = ax.pcolormesh(times, b, T[keep], shading="auto", cmap="magma",
                       rasterized=True)
    t_est = float(mark_bpm) if mark_bpm else t_est
    ax.axhline(t_est, color="#ffffff", ls="--", lw=1.0, alpha=0.8)
    ax.text(times[-1], t_est, f" {t_est:.0f} BPM ", color="#ffffff",
            fontsize=8, va="center", ha="right",
            bbox=dict(fc="#00000066", ec="none", pad=1.5))
    # A log tempo axis spaces the musically-equal steps equally --- 60 to
    # 120 is the same distance as 120 to 240 --- but its automatic ticks
    # label the axis "3 x 10^2", which is the one thing a tempo axis must
    # not say. Fixed BPM ticks, minor ticks off.
    ax.set_yscale("log")
    ticks = [40, 60, 80, 100, 120, 160, 200, 240]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_ylim(*BPM_RANGE)
    ax.set_ylabel("tempo (BPM)", color=MUT, fontsize=9)
    _time_axis(ax)
    _style(ax)
    return im


def draw_chromagram(ax, y, sr, hop: int = 512):
    """Chromagram onto ``ax``, labelled with the twelve pitch classes.

    A tonal centre reads as one or two rows staying lit across the width;
    a modulation moves that pattern bodily up or down the axis.
    """
    from . import music as amusic
    times, C = amusic.chromagram(y, sr, hop=hop)
    im = ax.pcolormesh(times, np.arange(13) - 0.5,
                       np.vstack([C, C[-1:]]), shading="auto", cmap="magma",
                       rasterized=True)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(PITCH_CLASSES)
    ax.set_ylim(-0.5, 11.5)
    ax.set_ylabel("pitch class", color=MUT, fontsize=9)
    _time_axis(ax)
    _style(ax)
    return im


def _export(fig, out_path, width_px, height_px):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(width_px / 100.0, height_px / 100.0)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


def tempogram_plot(y, sr, out_path, width_px: int = 1920,
                   height_px: int = 640, title: str = ""):
    """Labelled tempogram → ``out_path``, exactly ``width_px`` wide."""
    fig, ax = plt.subplots()
    im = draw_tempogram(ax, y, sr)
    fig.colorbar(im, ax=ax, pad=0.01).set_label("onset autocorrelation",
                                                color=MUT, fontsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return _export(fig, out_path, width_px, height_px)


def chromagram_plot(y, sr, out_path, width_px: int = 1920,
                    height_px: int = 640, title: str = ""):
    """Labelled chromagram → ``out_path``, exactly ``width_px`` wide."""
    fig, ax = plt.subplots()
    im = draw_chromagram(ax, y, sr)
    fig.colorbar(im, ax=ax, pad=0.01).set_label("pitch-class energy",
                                                color=MUT, fontsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return _export(fig, out_path, width_px, height_px)
