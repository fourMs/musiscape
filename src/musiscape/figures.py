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
