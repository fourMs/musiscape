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


#: available card styles; "combo" stacks mel + chroma + tempogram
STYLES = ("mel", "chroma", "tempo", "combo")


def _panels(y, sr, style):
    """Compute the image panels (label, matrix) for a card style."""
    import librosa
    out = []
    if style in ("mel", "combo"):
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=8000)
        db = librosa.power_to_db(M, ref=np.max)
        out.append(("mel", db, db.max() - 80))
    if style in ("chroma", "combo"):
        C = librosa.feature.chroma_stft(y=y, sr=sr)
        out.append(("chroma", C, 0.0))
    if style in ("tempo", "combo"):
        env = librosa.onset.onset_strength(y=y, sr=sr)
        T = librosa.feature.tempogram(onset_envelope=env, sr=sr,
                                      win_length=int(8 * sr / 512))
        bpm = librosa.tempo_frequencies(T.shape[0], sr=sr)
        band = (bpm >= 30) & (bpm <= 300)
        out.append(("tempo", T[band], 0.0))
    return out


def render_track(track: Track, color: str, out_path: str | Path,
                 sr: int = 22050, note: str = "", style: str = "mel") -> Path:
    """One card: analysis panel(s) over a waveform strip, title bar.

    Styles — ``mel`` (timbre & texture, the default), ``chroma`` (harmony:
    12 pitch classes over time), ``tempo`` (rhythmic periodicity; diffuse
    for rubato material), ``combo`` (all three stacked, taller card).
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = load(track, sr=sr)
        panels = _panels(y, sr, style)
        hop = max(1, len(y) // 1200)
        env = np.abs(y[: len(y) // hop * hop]).reshape(-1, hop).max(axis=1)

    heights = {"mel": 4.2, "chroma": 2.4, "tempo": 2.4}
    ratios = [heights[name] for name, _, _ in panels] + [1.0]
    fig_h = 3.6 if style != "combo" else 5.4
    fig = plt.figure(figsize=(6.4, fig_h), dpi=100)
    top = 0.86 if style != "combo" else 0.905
    gs = fig.add_gridspec(len(panels) + 1, 1, height_ratios=ratios,
                          left=0.015, right=0.985, top=top, bottom=0.05,
                          hspace=0.10)
    for i, (name, M, vmin) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(M, origin="lower", aspect="auto", cmap="magma",
                  vmin=vmin if vmin else None)
        if style == "combo":
            ax.text(0.006, 0.93, name, transform=ax.transAxes, fontsize=7,
                    color="white", va="top", alpha=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax1 = fig.add_subplot(gs[-1])
    x = np.arange(len(env))
    ax1.fill_between(x, -env, env, color=color, linewidth=0)
    ax1.set_xlim(0, len(env))
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_visible(False)

    dur = len(y) / sr
    ty = 0.955 if style != "combo" else 0.97
    fig.text(0.015, ty, track.title, fontsize=11, color=INK,
             fontweight="semibold", va="top")
    right = f"{int(dur // 60)}:{int(dur % 60):02d}" + (f" · {note}" if note else "")
    fig.text(0.985, ty, f"{track.album} · {right}", fontsize=8.5,
             color=MUT, va="top", ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


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


def notes_from_features(feats: list[dict]) -> dict:
    """(album, track) → "key · bpm" annotation for thumbnail title bars.

    The BPM is the octave-corrected dominant period from the circular
    pulse analysis; when pulse clarity is low (R < 0.1, i.e. rubato or
    drifting material) it is prefixed with ``~`` — a nominal period, not
    a felt tempo.
    """
    out = {}
    for f in feats:
        parts = []
        if f.get("key"):
            parts.append(f["key"])
        bpm = f.get("pulse_bpm")
        if bpm:
            approx = "~" if f.get("pulse_R", 0.0) < 0.1 else ""
            parts.append(f"{approx}{bpm:.0f} bpm")
        out[(f["album"], f["track"])] = " · ".join(parts)
    return out
