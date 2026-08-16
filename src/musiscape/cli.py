"""Command line: ``musiscape <verb> <collection-folder>``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import categorize, corpus, features, report
from .io import open_collection


def _out(args, coll) -> Path:
    return Path(args.out) if args.out else coll.root / "analysis"


def main(argv=None):
    """Entry point for the ``musiscape`` command."""
    p = argparse.ArgumentParser(
        prog="musiscape",
        description="Analyse a music collection: corpus fingerprints, "
                    "similarity landscape, honest categories.")
    p.add_argument("verb", choices=["probe", "extract", "fingerprint",
                                    "landscape", "categorize", "report",
                                    "thumbnails", "poster", "sonic",
                                    "segment", "figures", "pdf"])
    p.add_argument("folder", help="collection root (albums = subfolders); "
                                  "for segment, a folder of recordings")
    p.add_argument("-o", "--out", help="output folder (default <root>/analysis)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--duration", type=float,
                   help="analyse only the first N seconds per track")
    p.add_argument("-k", type=int, help="number of categories (default: auto)")
    p.add_argument("--style", default="mel",
                   help="thumbnail style: mel|chroma|tempo|combo|barcode|"
                        "ssm|trajectory|keyscape|rhythm|wave|vinyl|spiral|tonnetz|arcs|schaeffer|tarsom|stereo (default mel); poster accepts barcode|vinyl")
    p.add_argument("--min-song", type=float, default=60.0,
                   help="segment: shortest span counted as a song (s)")
    p.add_argument("--min-gap", type=float, default=8.0,
                   help="segment: shortest silence that ends a song (s)")
    p.add_argument("--width", type=int, default=1920,
                   help="figures: export width in pixels (default 1920)")
    args = p.parse_args(argv)

    # segment runs before the collection is opened: a folder of camera
    # files holds no audio files at all, and open_collection would refuse
    # it. Its output folder is where a collection then comes from.
    if args.verb == "segment":
        from . import concert
        from .io import list_recordings
        root = Path(args.folder).expanduser().resolve()
        paths = list_recordings(root)
        out = Path(args.out) if args.out else root / "analysis"
        manifest = concert.split_recording(paths, out,
                                           min_song_s=args.min_song,
                                           min_gap_s=args.min_gap)
        songs = json.loads(manifest.read_text())
        for s in songs:
            t = int(s["start_s"])
            across = (f" across {len(s['parts'])} files"
                      if len(s["parts"]) > 1 else "")
            print(f"{s['index']:2d}. {t // 60:3d}:{t % 60:02d} "
                  f"{s['duration_s'] / 60:5.1f} min  {s['file']}{across}")
        print(f"{len(songs)} songs from {len(paths)} recording(s) → "
              f"{manifest.parent / 'songs'}")
        return

    coll = open_collection(args.folder)
    out = _out(args, coll)

    if args.verb == "sonic":
        from . import sonic
        print(sonic.export_collection(coll, out, workers=args.workers))
        return

    if args.verb == "poster":
        from . import thumbnails
        pstyle = "vinyl" if args.style == "vinyl" else "barcode"
        print(thumbnails.poster(coll, out, workers=args.workers,
                                style=pstyle))
        return

    if args.verb == "thumbnails":
        from . import thumbnails
        notes = {}
        fpath = out / "features.json"
        if fpath.exists():
            notes = thumbnails.notes_from_features(
                features.load_features(fpath))
        print(thumbnails.render_collection(coll, out, notes=notes,
                                           workers=args.workers,
                                           style=args.style))
        return

    if args.verb == "figures":
        from . import figures as afig
        from .io import load
        fdir = out / "figures"
        for t in coll.tracks:
            y, sr = load(t, duration=args.duration)
            stem = f"{t.album.replace('/', '_')}_{t.title}".lstrip("._")
            afig.chromagram_plot(y, sr, fdir / f"{stem} chromagram.png",
                                 width_px=args.width, title=t.title)
            afig.tempogram_plot(y, sr, fdir / f"{stem} tempogram.png",
                                width_px=args.width, title=t.title)
            print(f"[{t.album}] {t.title}", flush=True)
        print(fdir)
        return

    if args.verb == "pdf":
        from . import pdfreport
        print(pdfreport.build(coll, out, workers=args.workers,
                              duration=args.duration))
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
        from . import figures
        stats = corpus.album_stats(feats)
        figures.fingerprints(stats, out / "fingerprints.png",
                             title=coll.root.name)
        (out / "album_stats.json").write_text(json.dumps(stats, indent=1))
        print(out / "fingerprints.png")
    elif args.verb == "landscape":
        from . import figures
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
