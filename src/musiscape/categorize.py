"""Categorisation that can explain itself.

Tracks are clustered in the standardised feature space (k-means, k chosen
by silhouette unless given), and every cluster is described by the features
that most distinguish it from the rest of the corpus—so a category is
never just "cluster 3", it is "sparse, dark, drone-like". This is the
interpretable counterpart to embedding-space clustering: fewer dimensions,
weaker similarity, but every axis has a musical name.
"""
from __future__ import annotations

import numpy as np

from .corpus import feature_matrix
from .features import FEATURES


def _silhouette(Z: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (euclidean), tiny-corpus friendly."""
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    vals = []
    for i in range(len(Z)):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            continue
        a = D[i, own].mean()
        b = min(D[i, labels == c].mean()
                for c in set(labels) if c != labels[i])
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def cluster(feats: list[dict], k: int | None = None, seed: int = 0) -> dict:
    """K-means clustering with named-feature descriptions per cluster.

    Returns labels aligned with ``feats``, the silhouette score, and for
    each cluster its size, member tracks, and the three most distinguishing
    features as signed z-scores (e.g. ``onset_rate -1.2`` = far sparser
    than the corpus norm).
    """
    from scipy.cluster.vq import kmeans2
    Z = feature_matrix(feats)
    ks = [k] if k else list(range(2, min(9, len(feats))))
    best = None
    for kk in ks:
        cents, labels = kmeans2(Z, kk, minit="++", seed=seed)
        if len(set(labels)) < kk:                    # empty cluster: skip
            continue
        s = _silhouette(Z, labels)
        if best is None or s > best[0]:
            best = (s, kk, labels)
    if best is None:
        raise ValueError("clustering failed for every k tried")
    sil, kk, labels = best

    clusters = []
    for c in range(kk):
        idx = np.where(labels == c)[0]
        zmean = Z[idx].mean(axis=0)
        top = np.argsort(-np.abs(zmean))[:3]
        clusters.append({
            "size": int(len(idx)),
            "tracks": [f"{feats[i]['album']}/{feats[i]['track']}"
                       for i in idx],
            "signature": {FEATURES[j]: round(float(zmean[j]), 2)
                          for j in top},
        })
    return {"k": kk, "silhouette": round(sil, 3),
            "labels": [int(x) for x in labels], "clusters": clusters}
