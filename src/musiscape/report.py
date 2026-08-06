"""Per-collection report: one README.md that answers "what is this collection?"

Runs the full pipeline (extract → stats → landscape → similarity → tonal
spread → clusters), writes the figures, and renders a markdown report with
an overview table, album fingerprints, affinity, categories, and notable
extremes—the file to open first when handed a folder of music.
"""
from __future__ import annotations

from pathlib import Path

from . import categorize, corpus, features, figures
from .io import Collection


def _extreme(feats: list[dict], key: str, largest: bool = True) -> str:
    f = max(feats, key=lambda x: x[key]) if largest else \
        min(feats, key=lambda x: x[key])
    return f"*{f['track']}* ({f['album']}, {f[key]})"


def run(coll: Collection, out_dir: str | Path, workers: int = 4,
        duration: float | None = None, k: int | None = None) -> Path:
    """Full pipeline → ``<out_dir>/README.md`` (+ features.json, figures)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = features.extract_collection(coll, out_dir, workers=workers,
                                        duration=duration)
    feats = features.load_features(fpath)
    stats = corpus.album_stats(feats)
    land = corpus.landscape(feats)
    sim = corpus.similarity(feats)
    spread = corpus.tonal_spread(feats)
    cats = categorize.cluster(feats, k=k) if len(feats) >= 4 else None

    figures.fingerprints(stats, out_dir / "fingerprints.png",
                         title=coll.root.name)
    figures.landscape_plot(feats, land, out_dir / "landscape.png")
    figures.affinity_plot(sim["affinity"], out_dir / "affinity.png")

    L: list[str] = []
    L.append(f"# {coll.root.name} — collection analysis\n")
    total = sum(s["total_min"] for s in stats.values())
    L.append(f"{len(feats)} tracks in {len(stats)} albums, "
             f"{total / 60:.1f} h total.\n")

    L.append("## Albums\n")
    L.append("| album | tracks | min | notes/s | centroid Hz | dyn dB | "
             "minor keys | consistency | key-cluster R |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a, s in stats.items():
        L.append(
            f"| {a} | {s['n_tracks']} | {s['total_min']} "
            f"| {s['onset_rate']['mean']:.2f} "
            f"| {s['centroid_hz']['mean']:.0f} "
            f"| {s['dyn_range_db']['mean']:.1f} "
            f"| {s['minor_share']:.0%} "
            f"| {sim['affinity'][a][a]:+.2f} "
            f"| {spread[a]['R']:.2f} |")
    L.append("\n![album fingerprints](fingerprints.png)\n")

    L.append("## Similarity landscape\n")
    e = land["explained"]
    L.append(f"PCA of the standardised features "
             f"({e[0]:.0%} + {e[1]:.0%} of variance).\n")
    L.append("![landscape](landscape.png)\n")
    L.append("![album affinity](affinity.png)\n")

    if cats:
        L.append("## Categories\n")
        L.append(f"k-means, k={cats['k']} "
                 f"(silhouette {cats['silhouette']:.2f}). Signatures are "
                 f"signed z-scores of the most distinguishing features.\n")
        for i, c in enumerate(cats["clusters"]):
            sig = ", ".join(f"{k2} {v:+.1f}" for k2, v in c["signature"].items())
            L.append(f"- **Category {i + 1}** ({c['size']} tracks): {sig}")
            L.append(f"  - " + " · ".join(c["tracks"][:8])
                     + (" · …" if c["size"] > 8 else ""))
        L.append("")

    L.append("## Extremes\n")
    L.append(f"- Densest playing: {_extreme(feats, 'onset_rate')}; "
             f"sparsest: {_extreme(feats, 'onset_rate', largest=False)}")
    L.append(f"- Brightest: {_extreme(feats, 'centroid_hz')}; "
             f"darkest: {_extreme(feats, 'centroid_hz', largest=False)}")
    L.append(f"- Steadiest pulse: {_extreme(feats, 'pulse_R')}; "
             f"freest: {_extreme(feats, 'pulse_R', largest=False)}")
    L.append(f"- Widest dynamics: {_extreme(feats, 'dyn_range_db')}")
    L.append("\n*Features are interpretable signal proxies "
             "(see musiscape docs); treat categories as drafts for "
             "listening, not verdicts.*\n")

    readme = out_dir / "README.md"
    readme.write_text("\n".join(L))
    return readme
