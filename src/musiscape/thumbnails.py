"""Per-track visual thumbnails: a piece at a glance.

Each track becomes one card, in a choice of representations:

- ``mel`` / ``chroma`` / ``tempo`` / ``combo`` — the spectrogram family
  (timbre & texture, harmony over time, rhythmic periodicity, all three);
- ``barcode`` — harmony as color: each moment's hue is its position on the
  circle of fifths, saturation its tonal focus, brightness its loudness;
- ``ssm`` — self-similarity matrix: musical *form* as texture (repetition
  blocks, sections, drone slabs);
- ``trajectory`` — the piece as a smoothed path through its own timbre
  space (MFCC PCA), colored start → end;
- ``keyscape`` — Sapp-style triangle: every analysis window at every time
  scale colored by its Krumhansl–Schmuckler key (hue = tonic on the circle
  of fifths, light = major, dark = minor);
- ``rhythm`` — Poincaré portrait of successive inter-onset intervals:
  metric playing collapses to points, rubato spreads into clouds;
- ``wave`` — Freesound-style waveform: the amplitude envelope with each
  moment colored by its spectral centroid (dark blue = dark timbre,
  red = bright), so timbre rides on the waveform itself.

Albums additionally get a contact sheet, and :func:`poster` stacks every
track's barcode into a single collection image where albums read as color
families. Thumbnails are meant for browsing a collection visually.
"""
from __future__ import annotations

import colorsys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import GRID, INK, MUT, album_colors
from .io import Collection, Track, load

#: available card styles
STYLES = ("mel", "chroma", "tempo", "combo",
          "barcode", "ssm", "trajectory", "keyscape", "rhythm", "wave")
#: taller cards for the square-ish representations
_TALL = {"combo": 5.4, "ssm": 5.2, "trajectory": 5.2, "keyscape": 5.2,
         "rhythm": 5.2}

_FIFTHS = 2 * np.pi * (7 * np.arange(12) % 12) / 12


# --------------------------------------------------------------------------
# shared feature helpers

def _feats_2hz(y, sr):
    """Chroma, MFCC and RMS aggregated to 2 Hz frames."""
    import librosa
    hop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)[1:]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    step = max(1, int(0.5 * sr / hop))
    n = max(1, C.shape[1] // step)
    agg = lambda X: np.stack([X[:, i * step:(i + 1) * step].mean(1)
                              for i in range(n)], 1)
    return agg(C), agg(M), np.array([rms[i * step:(i + 1) * step].mean()
                                     for i in range(n)])


def barcode_rgb(C, rms):
    """RGB strip (n×3) from 2 Hz chroma + RMS — the barcode's colors."""
    z = (C * np.exp(1j * _FIFTHS)[:, None]).sum(0) / (C.sum(0) + 1e-9)
    hue = (np.angle(z) / (2 * np.pi)) % 1.0
    sat = np.clip(np.abs(z) * 1.6, 0, 1)
    val = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0.12, 1) ** 0.6
    return np.array([colorsys.hsv_to_rgb(h, s, v)
                     for h, s, v in zip(hue, sat, val)])


def _key_profiles():
    """24 z-normalised Krumhansl–Schmuckler profiles (12 major, 12 minor)."""
    from .features import _MAJ, _MIN
    P = np.array([np.roll(_MAJ, i) for i in range(12)]
                 + [np.roll(_MIN, i) for i in range(12)])
    return (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)


def keyscape_rgb(C, levels=48):
    """Sapp-style keyscape image (levels×n×3) from 2 Hz chroma."""
    n = C.shape[1]
    P = _key_profiles()
    cum = np.concatenate([np.zeros((12, 1)), np.cumsum(C, axis=1)], axis=1)
    img = np.ones((levels, n, 3))
    widths = np.unique(np.linspace(max(2, n // 80), n, levels).astype(int))[::-1]
    for li, w in enumerate(np.resize(widths, levels)):
        x0 = np.arange(0, n - w + 1)
        W = ((cum[:, x0 + w] - cum[:, x0]) / w).T          # windows × 12
        Wn = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-9)
        idx = np.argmax(Wn @ P.T, axis=1)
        tonic, minor = idx % 12, idx >= 12
        hue = ((7 * tonic) % 12) / 12.0
        hsv = np.stack([hue,
                        np.where(minor, 0.75, 0.55),
                        np.where(minor, 0.55, 0.95)], axis=1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        img[li, x0 + w // 2] = rgb
        # fill edges of the row so the triangle reads solid
        img[li, :w // 2] = np.nan
        img[li, w // 2 + len(x0):] = np.nan
    return img


# --------------------------------------------------------------------------
# card rendering

def _panels(y, sr, style):
    """Image panels (label, matrix, vmin) for the spectrogram styles."""
    import librosa
    out = []
    if style in ("mel", "combo"):
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=8000)
        db = librosa.power_to_db(M, ref=np.max)
        out.append(("mel", db, db.max() - 80))
    if style in ("chroma", "combo"):
        out.append(("chroma", librosa.feature.chroma_stft(y=y, sr=sr), 0.0))
    if style in ("tempo", "combo"):
        env = librosa.onset.onset_strength(y=y, sr=sr)
        T = librosa.feature.tempogram(onset_envelope=env, sr=sr,
                                      win_length=int(8 * sr / 512))
        bpm = librosa.tempo_frequencies(T.shape[0], sr=sr)
        out.append(("tempo", T[(bpm >= 30) & (bpm <= 300)], 0.0))
    return out


def _draw_main(ax, y, sr, style):
    """Draw a non-spectrogram main panel onto ``ax``."""
    import librosa
    from scipy.ndimage import uniform_filter, uniform_filter1d

    if style == "barcode":
        C, _, rms = _feats_2hz(y, sr)
        ax.imshow(barcode_rgb(C, rms)[None, :, :], aspect="auto")

    elif style == "ssm":
        C, M, _ = _feats_2hz(y, sr)
        F = np.vstack([C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9),
                       M / (np.abs(M).max() + 1e-9)])
        Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
        S = uniform_filter(Fn.T @ Fn, size=3)
        ax.imshow(S, cmap="magma", origin="lower", aspect="auto")

    elif style == "trajectory":
        _, M, _ = _feats_2hz(y, sr)
        Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
        U, S_, Vt = np.linalg.svd(Z.T, full_matrices=False)
        P = uniform_filter1d(U[:, :2] * S_[:2], size=5, axis=0)
        ax.plot(P[:, 0], P[:, 1], color=MUT, alpha=0.35, linewidth=0.8)
        ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="magma",
                   s=9, alpha=0.9)
        ax.margins(0.06)

    elif style == "keyscape":
        C, _, _ = _feats_2hz(y, sr)
        ax.imshow(keyscape_rgb(C), aspect="auto", origin="upper",
                  interpolation="nearest")

    elif style == "wave":
        hop = 512
        cent = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                 hop_length=hop)[0]
        cols = 1200
        step = max(1, len(y) // cols)
        env = np.abs(y[: len(y) // step * step]).reshape(-1, step).max(axis=1)
        ci = np.interp(np.linspace(0, len(cent) - 1, len(env)),
                       np.arange(len(cent)), cent)
        norm = np.clip((np.log2(ci + 1e-9) - np.log2(300))
                       / (np.log2(4000) - np.log2(300)), 0, 1)
        colors = matplotlib.colormaps["turbo"](norm)
        ax.bar(np.arange(len(env)), 2 * env, bottom=-env, width=1.0,
               color=colors, linewidth=0)
        ax.set_xlim(0, len(env))
        m = env.max() * 1.05 + 1e-9
        ax.set_ylim(-m, m)

    elif style == "rhythm":
        from ambiscape.music import ONSET_FLOOR
        env = librosa.onset.onset_strength(y=y, sr=sr)
        fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                        units="frames")
        fr = fr[env[fr] >= ONSET_FLOOR]
        t = librosa.frames_to_time(fr, sr=sr)
        ioi = np.diff(t)
        ioi = ioi[(ioi > 0.05) & (ioi < 2.5)]
        if len(ioi) > 3:
            ax.scatter(ioi[:-1], ioi[1:], c=np.arange(len(ioi) - 1),
                       cmap="magma", s=12, alpha=0.75)
            ax.plot([0.05, 2.5], [0.05, 2.5], color=GRID, linewidth=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.05, 2.5); ax.set_ylim(0.05, 2.5)
        else:
            ax.text(0.5, 0.5, "too few onsets", transform=ax.transAxes,
                    ha="center", color=MUT, fontsize=9)


def render_track(track: Track, color: str, out_path: str | Path,
                 sr: int = 22050, note: str = "", style: str = "mel") -> Path:
    """One card: the chosen representation over a waveform strip."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = load(track, sr=sr)
        spectro = style in ("mel", "chroma", "tempo", "combo")
        panels = _panels(y, sr, style) if spectro else [(style, None, None)]
        has_strip = style != "wave"
        hop = max(1, len(y) // 1200)
        env = np.abs(y[: len(y) // hop * hop]).reshape(-1, hop).max(axis=1)

        heights = {"mel": 4.2, "chroma": 2.4, "tempo": 2.4}
        ratios = ([heights[n] for n, _, _ in panels] if spectro
                  else [4.2 if style in ("barcode", "wave") else 6.0])
        if has_strip:
            ratios = ratios + [1.0]
        fig_h = _TALL.get(style, 3.6)
        fig = plt.figure(figsize=(6.4, fig_h), dpi=100)
        top = 0.905 if fig_h > 4 else 0.86
        gs = fig.add_gridspec(len(panels) + int(has_strip), 1,
                              height_ratios=ratios,
                              left=0.015, right=0.985, top=top, bottom=0.05,
                              hspace=0.10)
        for i, (name, M, vmin) in enumerate(panels):
            ax = fig.add_subplot(gs[i])
            if spectro:
                ax.imshow(M, origin="lower", aspect="auto", cmap="magma",
                          vmin=vmin if vmin else None)
                if style == "combo":
                    ax.text(0.006, 0.93, name, transform=ax.transAxes,
                            fontsize=7, color="white", va="top", alpha=0.8)
            else:
                _draw_main(ax, y, sr, style)
            if style != "rhythm":
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.tick_params(labelsize=6, colors=MUT)
            for s in ax.spines.values():
                s.set_visible(False)
        if has_strip:
            ax1 = fig.add_subplot(gs[-1])
            x = np.arange(len(env))
            ax1.fill_between(x, -env, env, color=color, linewidth=0)
            ax1.set_xlim(0, len(env))
            ax1.set_xticks([]); ax1.set_yticks([])
            for s in ax1.spines.values():
                s.set_visible(False)

        dur = len(y) / sr
        ty = 0.97 if fig_h > 4 else 0.955
        fig.text(0.015, ty, track.title, fontsize=11, color=INK,
                 fontweight="semibold", va="top")
        right = (f"{int(dur // 60)}:{int(dur % 60):02d}"
                 + (f" · {note}" if note else ""))
        fig.text(0.985, ty, f"{track.album} · {right}", fontsize=8.5,
                 color=MUT, va="top", ha="right")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# collection-level rendering

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


def _strip_work(job):
    path, album, width = job
    try:
        t = Track(path=Path(path), album=album)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = load(t)
            C, _, rms = _feats_2hz(y, sr)
        rgb = barcode_rgb(C, rms)
        xi = np.linspace(0, len(rgb) - 1, width)
        res = np.stack([np.interp(xi, np.arange(len(rgb)), rgb[:, c])
                        for c in range(3)], axis=1)
        print(f"[{album}] {t.title}", flush=True)
        return album, t.title, res
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR {path}: {e}", flush=True)
        return None


def poster(coll: Collection, out_dir: str | Path, workers: int = 4,
           strip_w: int = 1200, strip_h: int = 16) -> Path:
    """One image for the whole collection: every track a barcode strip,
    grouped by album — albums read as color families, drones as
    desaturated bands. → ``<out_dir>/poster.png``"""
    from PIL import Image, ImageDraw
    jobs = [(str(t.path), t.album, strip_w) for t in coll.tracks]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = [r for r in ex.map(_strip_work, jobs) if r]
    else:
        res = [r for r in map(_strip_work, jobs) if r]
    strips = {(a, t): s for a, t, s in res}

    margin, gap, header = 210, 3, 26
    n_rows = len(strips)
    H = 12 + len(coll.albums) * (header + gap) + n_rows * (strip_h + gap) + 12
    W = margin + strip_w + 14
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    colors = album_colors(coll.album_names)
    yy = 12
    for a in coll.albums:
        d.rectangle([margin, yy + 4, margin + 12, yy + 16],
                    fill=colors[a.name])
        d.text((margin + 20, yy + 4), a.name, fill=INK)
        yy += header + gap
        for t in a.tracks:
            s = strips.get((a.name, t.title))
            if s is None:
                continue
            arr = (np.repeat(s[None, :, :], strip_h, axis=0)
                   * 255).astype(np.uint8)
            img.paste(Image.fromarray(arr), (margin, yy))
            d.text((margin - 6, yy + 2), t.title[:30], fill=MUT, anchor="ra")
            yy += strip_h + gap
    out = Path(out_dir) / "poster.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def notes_from_features(feats: list[dict]) -> dict:
    """(album, track) → "key · bpm" annotation for thumbnail title bars.

    The BPM shown is the perceptually-weighted tempo estimate
    (``tempo_bpm``, librosa's prior-based estimator, which targets the
    felt beat rather than the subdivision the phase-lock peaks on);
    when pulse clarity is low (R < 0.1, i.e. rubato or drifting
    material) it is prefixed with ``~`` — a nominal tempo, not a felt
    one. Falls back to ``pulse_bpm`` for older feature files.
    """
    out = {}
    for f in feats:
        parts = []
        if f.get("key"):
            parts.append(f["key"])
        bpm = f.get("tempo_bpm") or f.get("pulse_bpm")
        if bpm:
            approx = "~" if f.get("pulse_R", 0.0) < 0.1 else ""
            parts.append(f"{approx}{bpm:.0f} bpm")
        out[(f["album"], f["track"])] = " · ".join(parts)
    return out
