"""One PDF: a summary table, then a page of figures per track.

The Markdown report in :mod:`report` is for reading on screen next to the
audio. This is the thing you hand someone --- a front table that fits an
entire concert on one page, and behind it one page per track carrying the
figures the table's numbers came from.

Every estimate is printed with its cross-check beside it rather than alone.
``key`` and ``tempo_bpm`` each travel with the share of 20-second windows
that agreed on them. That is the whole point of the layout: a number in
this report is never presented as more certain than it is, and the reader
can see which tracks the analysis is confident about without knowing
anything about how it works.

No column claims to say whether a track has a pulse. Nothing measured here
could tell a band from an audience clapping along, and a column that
implied otherwise would be worse than none --- see :mod:`stability`.

Written with matplotlib's ``PdfPages``, so no PDF library is needed beyond
what the package already depends on.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .figures import GRID, INK, MUT, draw_chromagram, draw_tempogram

#: Page size, inches. A4 landscape: wide enough for a spectrogram to be
#: worth looking at, which portrait is not.
PAGE = (11.69, 8.27)

#: Agreement at or above this is reported as a confident estimate.
STRONG = 0.75
#: Below this the estimate is reported as not holding up.
WEAK = 0.5


def confidence(agreement: float | None) -> str:
    """Word for a window-agreement share, or ``"—"`` when unmeasured."""
    if agreement is None:
        return "—"
    if agreement >= STRONG:
        return "strong"
    if agreement >= WEAK:
        return "fair"
    return "weak"


def _fmt(v, spec="{:.0f}"):
    return "—" if v is None else spec.format(v)


def _summary_rows(feats: list[dict]) -> tuple[list[str], list[list[str]]]:
    head = ["#", "track", "min", "key", "key conf.", "tempo", "tempo conf.",
            "windows"]
    rows = []
    for i, f in enumerate(feats, start=1):
        rows.append([
            str(i),
            f["track"],
            f"{f['duration_s'] / 60:.1f}",
            f.get("key_windowed") or f["key"],
            f"{confidence(f.get('key_agreement'))}"
            f" ({_fmt(f.get('key_agreement'), '{:.0%}')})",
            f"{_fmt(f.get('tempo_windowed_bpm') or f['tempo_bpm'])} BPM",
            f"{confidence(f.get('tempo_agreement'))}"
            f" ({_fmt(f.get('tempo_agreement'), '{:.0%}')})",
            str(f.get("key_windows", 0)),
        ])
    return head, rows


def _summary_page(pdf, feats: list[dict], title: str):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.995, title, color=INK, fontsize=17, va="top")
    total = sum(f["duration_s"] for f in feats) / 60.0
    fig.text(0.06, 0.945,
             f"{len(feats)} tracks · {total:.0f} min of music",
             color=MUT, fontsize=10, va="top")

    head, rows = _summary_rows(feats)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.80])
    ax.axis("off")
    # the track name needs roughly three times a number column
    widths = [0.03, 0.28, 0.06, 0.11, 0.15, 0.10, 0.15, 0.09]
    tbl = ax.table(cellText=rows, colLabels=head, loc="upper center",
                   cellLoc="left", colLoc="left", colWidths=widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="semibold")
            cell.set_facecolor("#f4f3ee")
        else:
            cell.set_text_props(color=INK)

    fig.text(0.06, 0.045,
             "Confidence is the share of 20-second windows agreeing with the "
             "reported estimate; the last column is how many windows voted. "
             "No single number is offered for\n\"is there a pulse\": on this "
             "material every candidate overlapped with applause, which is "
             "itself rhythmic. Read the tempogram on each track's page.",
             color=MUT, fontsize=8, va="top")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _track_page(pdf, y, sr, feat: dict, index: int):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.985, f"{index}. {feat['track']}", color=INK,
             fontsize=14, va="top")

    ka, ta = feat.get("key_agreement"), feat.get("tempo_agreement")
    line = (
        f"{feat.get('key_windowed') or feat['key']}"
        f"  ({confidence(ka)}, {_fmt(ka, '{:.0%}')} of "
        f"{feat.get('key_windows', 0)} windows)"
        f"     ·     "
        f"{_fmt(feat.get('tempo_windowed_bpm') or feat['tempo_bpm'])} BPM"
        f"  ({confidence(ta)}, {_fmt(ta, '{:.0%}')})"
        f"     ·     {feat['duration_s'] / 60:.1f} min"
    )
    fig.text(0.06, 0.94, line, color=MUT, fontsize=10, va="top")

    ax1 = fig.add_axes([0.075, 0.535, 0.845, 0.35])
    draw_chromagram(ax1, y, sr)
    ax2 = fig.add_axes([0.075, 0.085, 0.845, 0.35])
    draw_tempogram(ax2, y, sr,
                   mark_bpm=feat.get("tempo_windowed_bpm")
                   or feat["tempo_bpm"])

    whole = (f"whole-track estimates: key {feat['key']} "
             f"(conf {feat['key_conf']:.2f}), tempo {feat['tempo_bpm']:.0f} "
             f"BPM, pulse_R {feat['pulse_R']:.3f}, "
             f"chroma entropy {feat['chroma_entropy']:.3f} of 3.585 max")
    fig.text(0.06, 0.035, whole, color=MUT, fontsize=8, va="top")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def build(coll, out_dir: str | Path, workers: int = 4,
          duration: float | None = None, title: str | None = None) -> Path:
    """Summary table + one figure page per track → ``<out_dir>/report.pdf``.

    Features are extracted (and cached) exactly as every other verb does,
    so running this after ``report`` costs only the drawing.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    from . import features as afeat
    from .io import load

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = afeat.extract_collection(coll, out_dir, workers=workers,
                                     duration=duration)
    feats = afeat.load_features(fpath)
    by_title = {f["track"]: f for f in feats}
    tracks = [t for t in coll.tracks if t.title in by_title]
    ordered = [by_title[t.title] for t in tracks]

    pdf_path = out_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        _summary_page(pdf, ordered, title or coll.root.name)
        for i, track in enumerate(tracks, start=1):
            y, sr = load(track, duration=duration)
            _track_page(pdf, y, sr, by_title[track.title], i)
    return pdf_path
