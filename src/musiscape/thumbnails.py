"""Per-track visual thumbnails: a piece at a glance.

Each track becomes one compact card — log-mel spectrogram over a waveform
strip in the album's color, titled with track, duration, and (when a
``features.json`` is available) the estimated key. Albums additionally get
a contact sheet. Thumbnails are meant for browsing a collection visually:
drones, plucked flows, and ensemble pieces read differently at a glance.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import INK, MUT, album_colors
from .io import Collection, Track, load


def render_track(track: Track, color: str, out_path: str | Path,
                 sr: int = 22050, note: str = "") -> Path:
    """One 640×360 card: mel spectrogram, waveform strip, title bar."""
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = load(track, sr=sr)
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=8000)
        db = librosa.power_to_db(M, ref=np.max)
        # coarse peak envelope for the waveform strip
        hop = max(1, len(y) // 1200)
        env = np.abs(y[: len(y) // hop * hop]).reshape(-1, hop).max(axis=1)

    fig = plt.figure(figsize=(6.4, 3.6), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0],
                          left=0.015, right=0.985, top=0.86, bottom=0.05,
                          hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(db, origin="lower", aspect="auto", cmap="magma",
               vmin=db.max() - 80)
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(len(env))
    ax1.fill_between(x, -env, env, color=color, linewidth=0)
    ax1.set_xlim(0, len(env))
    for ax in (ax0, ax1):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    dur = len(y) / sr
    fig.text(0.015, 0.955, track.title, fontsize=11, color=INK,
             fontweight="semibold", va="top")
    right = f"{int(dur // 60)}:{int(dur % 60):02d}" + (f" · {note}" if note else "")
    fig.text(0.985, 0.955, f"{track.album} · {right}", fontsize=8.5,
             color=MUT, va="top", ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


def _work(job):
    path, album, color, out, note = job
    try:
        p = render_track(Track(path=Path(path), album=album), color, out,
                         note=note)
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
                      notes: dict | None = None, workers: int = 4) -> Path:
    """All thumbnails → ``<out_dir>/thumbnails/<album>/<track>.png``
    plus a contact sheet per album. ``notes`` maps (album, title) to a
    short annotation (e.g. the estimated key from ``features.json``)."""
    out_dir = Path(out_dir) / "thumbnails"
    colors = album_colors(coll.album_names)
    notes = notes or {}
    jobs = [(str(t.path), t.album, colors[t.album],
             out_dir / t.album / f"{t.title}.png",
             notes.get((t.album, t.title), "")) for t in coll.tracks]
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


def notes_from_features(feats: list[dict]) -> dict:
    """(album, track) → estimated key, for thumbnail annotations."""
    return {(f["album"], f["track"]): f.get("key", "") for f in feats}
