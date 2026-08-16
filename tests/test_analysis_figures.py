"""Labelled tempogram and chromagram figures, and the PDF report.

The thumbnail cards are deliberately unlabelled --- they are for browsing a
collection at a glance. These are the other thing: figures you read numbers
off, so every axis carries a unit.
"""
import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import musiscape
from musiscape import figures

SR = 22050


def _music(dur, root=220.0, bpm=120.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.0, dur - 0.5, 60.0 / bpm):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.4 * SR))) / SR
        env = np.exp(-seg / 0.12)
        for f in (root, root * 2 ** (4 / 12), root * 2 ** (7 / 12)):
            y[i:i + len(seg)] += env * np.sin(2 * np.pi * f * seg)
    return 0.25 * y + 1e-4 * rng.standard_normal(len(t))


# --------------------------------------------------------------------------
# axis labelling

def test_tempogram_axes_carry_time_and_bpm_units():
    fig, ax = plt.subplots()
    figures.draw_tempogram(ax, _music(20.0), SR)

    assert "BPM" in ax.get_ylabel()
    assert "time" in ax.get_xlabel().lower()
    plt.close(fig)


def test_chromagram_y_axis_is_labelled_with_the_twelve_pitch_classes():
    fig, ax = plt.subplots()
    figures.draw_chromagram(ax, _music(20.0), SR)

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
    assert "time" in ax.get_xlabel().lower()
    plt.close(fig)


def test_time_axis_is_formatted_as_minutes_and_seconds():
    """A four-minute song on a seconds axis is unreadable."""
    fig, ax = plt.subplots()
    figures.draw_chromagram(ax, _music(90.0), SR)
    fig.canvas.draw()

    ticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert any(":" in t for t in ticks), f"no mm:ss ticks: {ticks}"
    plt.close(fig)


# --------------------------------------------------------------------------
# export width

@pytest.mark.parametrize("plot", ["tempogram_plot", "chromagram_plot"])
def test_plots_export_at_the_requested_pixel_width(tmp_path, plot):
    out = tmp_path / f"{plot}.png"

    getattr(figures, plot)(_music(20.0), SR, out, width_px=1920)

    assert Image.open(out).size[0] == 1920


# --------------------------------------------------------------------------
# the PDF report

def test_pdf_report_has_a_summary_page_then_one_page_per_track(tmp_path):
    from musiscape import pdfreport

    for i, (root, bpm) in enumerate([(220.0, 120.0), (330.0, 90.0)]):
        sf.write(tmp_path / f"0{i + 1} song.wav",
                 _music(20.0, root, bpm, seed=i), SR)
    coll = musiscape.open_collection(tmp_path)

    pdf = pdfreport.build(coll, tmp_path / "analysis", workers=1)

    assert pdf.suffix == ".pdf"
    assert pdf.stat().st_size > 0
    PdfReader = pytest.importorskip("pypdf").PdfReader
    assert len(PdfReader(str(pdf)).pages) == 1 + 2


@pytest.mark.parametrize("draw", ["draw_tempogram", "draw_chromagram"])
def test_meshes_are_rasterised_so_the_pdf_stays_openable(draw):
    """A tempogram is hundreds of thousands of quads.

    Left as vectors, each one is written into the PDF as its own path: an
    eight-song concert came to 156 MB and no viewer would scroll it.
    """
    fig, ax = plt.subplots()
    getattr(figures, draw)(ax, _music(20.0), SR)

    meshes = [c for c in ax.collections]
    assert meshes, "nothing drawn"
    assert all(m.get_rasterized() for m in meshes)
    plt.close(fig)


def test_tempogram_shows_only_bpm_tick_labels():
    """The log scale adds its own minor ticks.

    Left on, they label the axis "3 x 10^2" beside the BPM values that were
    asked for, which is the one thing a tempo axis must not say.
    """
    fig, ax = plt.subplots()
    figures.draw_tempogram(ax, _music(20.0), SR)
    fig.canvas.draw()

    labels = [t.get_text() for t in ax.get_yticklabels(which="both")
              if t.get_text()]
    assert labels, "no y tick labels"
    assert all(t.replace(".", "").isdigit() for t in labels), labels
    plt.close(fig)


def test_tempogram_can_mark_the_tempo_the_report_cites():
    """The figure's own estimate and the report's need not agree.

    They come from different onset envelopes, so a page showing one number
    in the header and another on the plot contradicts itself.
    """
    fig, ax = plt.subplots()
    figures.draw_tempogram(ax, _music(20.0), SR, mark_bpm=97.0)

    marks = [t.get_text() for t in ax.texts]
    assert any("97" in m for m in marks), marks
    plt.close(fig)
