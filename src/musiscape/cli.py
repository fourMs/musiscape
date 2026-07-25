"""Command line: ``musiscape <verb> <collection-folder>``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import categorize, corpus, features, figures, report
from .io import open_collection


def _out(args, coll) -> Path:
    return Path(args.out) if args.out else coll.root / "analysis"


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="musiscape",
        description="Analyse a music collection: corpus fingerprints, "
                    "similarity landscape, honest categories.")
    p.add_argument("verb", choices=["probe", "extract", "fingerprint",
                                    "landscape", "categorize", "report",
                                    "thumbnails"])
    p.add_argument("folder", help="collection root (albums = subfolders)")
    p.add_argument("-o", "--out", help="output folder (default <root>/analysis)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--duration", type=float,
                   help="analyse only the first N seconds per track")
    p.add_argument("-k", type=int, help="number of categories (default: auto)")
    args = p.parse_args(argv)

    coll = open_collection(args.folder)
    out = _out(args, coll)

    if args.verb == "thumbnails":
        from . import thumbnails
        notes = {}
        fpath = out / "features.json"
        if fpath.exists():
            notes = thumbnails.notes_from_features(
                features.load_features(fpath))
        print(thumbnails.render_collection(coll, out, notes=notes,
                                           workers=args.workers))
        return

    if args.verb == "probe":
        for a in coll.albums:
            mins = "?"
            print(f"{a.name}: {len(a.tracks)} tracks")
            for t in a.tracks[:50]:
                print(f"  {t.title}")
        print(f"total: {len(coll.tracks)} tracks in {len(coll.albums)} albums")
        return

    fpath = features.extract_collection(coll, out, workers=args.workers,
                                        duration=args.duration)
    feats = features.load_features(fpath)

    if args.verb == "extract":
        print(f"{len(feats)} tracks → {fpath}")
    elif args.verb == "fingerprint":
        stats = corpus.album_stats(feats)
        figures.fingerprints(stats, out / "fingerprints.png",
                             title=coll.root.name)
        (out / "album_stats.json").write_text(json.dumps(stats, indent=1))
        print(out / "fingerprints.png")
    elif args.verb == "landscape":
        land = corpus.landscape(feats)
        figures.landscape_plot(feats, land, out / "landscape.png")
        sim = corpus.similarity(feats)
        figures.affinity_plot(sim["affinity"], out / "affinity.png")
        print(out / "landscape.png")
    elif args.verb == "categorize":
        cats = categorize.cluster(feats, k=args.k)
        (out / "categories.json").write_text(json.dumps(cats, indent=1))
        for i, c in enumerate(cats["clusters"]):
            sig = ", ".join(f"{k} {v:+.1f}" for k, v in c["signature"].items())
            print(f"category {i + 1} ({c['size']}): {sig}")
    elif args.verb == "report":
        print(report.run(coll, out, workers=args.workers,
                         duration=args.duration, k=args.k))


if __name__ == "__main__":
    main()
