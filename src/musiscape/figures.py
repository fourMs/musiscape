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
# These are the counterpart, meant to be read rather than glanced at, so
# every axis carries its unit and time runs in m:ss rather than in seconds.

#: Pitch-class names, bottom to top on a chromagram's y axis.
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B")

#: Tempo range drawn, in BPM. Outside this the tempogram shows metrical
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
    figure's own estimate. Callers quoting a tempo elsewhere on the page
    should pass theirs, since the two come from different onset envelopes
    and would otherwise disagree in print.
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
    # A log tempo axis spaces musically equal steps equally, so 60 to 120
    # covers the same distance as 120 to 240. Its automatic ticks would
    # label the axis "3 x 10^2", so the BPM ticks are fixed and the minor
    # ticks are off.
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


# --------------------------------------------------------------------------
# The concert timeline: what the recording was doing, end to end.

#: Colour per region class. Music takes the first categorical hue so it
#: reads as the subject; the two things the room does take the next two;
#: room tone and unclassified frames stay grey so they recede.
REGION_COLORS = {
    "music": PALETTE[0],
    "applause": PALETTE[1],
    "voices": PALETTE[2],
    "quiet": "#dcdbd4",
    "other": "#a8a69f",
}


def draw_concert_timeline(ax, spans, total_s: float, level=None):
    """Labelled timeline of a concert's regions onto ``ax``.

    ``spans`` is what :func:`musiscape.concert.regions` returns. With
    ``level``, a per-second dB array, the class colour is carried by the
    waveform itself rather than by a separate ribbon: one lane reads faster,
    and it shows an applause swell dying away where a block only shows that
    applause happened. Without a level, spans are drawn as plain blocks.

    The legend names only the classes that occur, since an entry for a class
    that never happens invites the reader to hunt for it.
    """
    seen = [s["label"] for s in spans]
    seen = [c for i, c in enumerate(seen) if c not in seen[:i]]

    if level is not None and len(level):
        lv = np.asarray(level, float)
        t = np.linspace(0, total_s, len(lv))
        lo, hi = np.percentile(lv, 2), np.percentile(lv, 98)
        amp = np.clip((lv - lo) / max(hi - lo, 1e-9), 0, 1)
        for span in spans:
            # one sample of overlap each side, so neighbouring bands meet
            m = (t >= span["start_s"]) & (t <= span["end_s"])
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            a = max(idx[0] - 1, 0)
            b = min(idx[-1] + 2, len(t))
            sl = slice(a, b)
            ax.fill_between(
                t[sl], -amp[sl], amp[sl], linewidth=0,
                color=REGION_COLORS.get(span["label"], REGION_COLORS["other"]))
        ax.set_ylim(-1.08, 1.08)
    else:
        for span in spans:
            ax.axvspan(
                span["start_s"], span["end_s"], 0.0, 1.0, linewidth=0,
                color=REGION_COLORS.get(span["label"], REGION_COLORS["other"]))
        ax.set_ylim(0, 1)

    from matplotlib.patches import Patch
    order = [c for c in REGION_COLORS if c in seen]
    ax.legend(handles=[Patch(facecolor=REGION_COLORS[c], label=c)
                       for c in order],
              loc="upper left", bbox_to_anchor=(0, -0.18), ncol=len(order),
              frameon=False, fontsize=9)

    ax.set_xlim(0, total_s)
    ax.set_yticks([])
    _time_axis(ax, "concert time (m:ss)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    return ax


def concert_timeline(spans, total_s: float, out_path, width_px: int = 1920,
                     height_px: int = 300, title: str = "", level=None):
    """Concert timeline → ``out_path``, exactly ``width_px`` wide."""
    fig, ax = plt.subplots()
    draw_concert_timeline(ax, spans, total_s, level=level)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    return _export(fig, out_path, width_px, height_px)
