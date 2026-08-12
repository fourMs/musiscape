"""Corpus-level statistics—the questions per-track tools do not answer.

The questions answered here are about the *collection*: how do its albums
differ (fingerprints), which tracks resemble which (similarity, landscape),
how internally consistent is each album, and how tightly does each cluster
in key space (a circular statistic—key centres have no linear mean).
"""
from __future__ import annotations

import numpy as np

from .features import FEATURES, LOG_FEATURES


def albums_of(feats: list[dict]) -> list[str]:
    """Album names in first-appearance order."""
    seen: dict[str, None] = {}
    for f in feats:
        seen.setdefault(f["album"], None)
    return list(seen)


def feature_matrix(feats: list[dict]) -> np.ndarray:
    """Standardised (z-scored, log-compressed where skewed) feature matrix."""
    X = np.array([[f[k] for k in FEATURES] for f in feats], dtype=float)
    for i, k in enumerate(FEATURES):
        if k in LOG_FEATURES:
            X[:, i] = np.log1p(X[:, i])
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)


def album_stats(feats: list[dict]) -> dict:
    """Per-album mean/std/min/max of every feature, plus keys and counts."""
    out = {}
    for a in albums_of(feats):
        sel = [f for f in feats if f["album"] == a]
        st = {k: {"mean": round(float(np.mean([f[k] for f in sel])), 4),
                  "std": round(float(np.std([f[k] for f in sel])), 4),
                  "min": round(float(np.min([f[k] for f in sel])), 4),
                  "max": round(float(np.max([f[k] for f in sel])), 4)}
              for k in FEATURES}
        st["n_tracks"] = len(sel)
        st["total_min"] = round(sum(f["duration_s"] for f in sel) / 60, 1)
        st["keys"] = [f["key"] for f in sel]
        st["minor_share"] = round(
            sum("minor" in f["key"] for f in sel) / len(sel), 3)
        out[a] = st
    return out


def landscape(feats: list[dict]) -> dict:
    """PCA of the standardised features: 2-D coords, variance, loadings."""
    Z = feature_matrix(feats)
    U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    expl = (S ** 2 / np.sum(S ** 2))[:2]
    return {
        "coords": [[round(float(x), 3), round(float(y), 3)] for x, y in pcs],
        "explained": [round(float(e), 3) for e in expl],
        "loadings": {FEATURES[i]: [round(float(Vt[0, i]), 3),
                                   round(float(Vt[1, i]), 3)]
                     for i in range(len(FEATURES))},
    }


def similarity(feats: list[dict]) -> dict:
    """Track cosine-similarity matrix + album affinity and consistency.

    ``affinity[a][b]`` is the mean similarity between the tracks of albums
    a and b; the diagonal (mean *pairwise* similarity within an album) is
    its internal consistency—one instrument and one mood score high,
    an eclectic album scores near zero.
    """
    Z = feature_matrix(feats)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    sim = Zn @ Zn.T
    albums = [f["album"] for f in feats]
    names = albums_of(feats)
    aff: dict[str, dict[str, float]] = {a: {} for a in names}
    for a in names:
        ia = [i for i, x in enumerate(albums) if x == a]
        for b in names:
            ib = [i for i, x in enumerate(albums) if x == b]
            block = sim[np.ix_(ia, ib)]
            if a == b:
                v = (block[np.triu_indices(len(ia), k=1)]
                     if len(ia) > 1 else np.array([1.0]))
            else:
                v = block.flatten()
            aff[a][b] = round(float(v.mean()), 3)
    return {"matrix": [[round(float(v), 3) for v in row] for row in sim],
            "affinity": aff}


def tonal_spread(feats: list[dict]) -> dict:
    """Per-album concentration of tonal centres on the circle of fifths."""
    from .music import tonal_center_spread
    out = {}
    for a in albums_of(feats):
        chromas = [f["chroma"] for f in feats if f["album"] == a]
        out[a] = tonal_center_spread(chromas)
    return out
