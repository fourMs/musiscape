"""Thumbnails for a collection whose audio sits in the root folder.

``open_collection`` names that album ``"."``---the README's own quickstart
("point it at a folder of audio files") produces it, and so does every
folder ``segment`` writes.
"""
import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

import musiscape
from musiscape import thumbnails

SR = 22050


def test_contact_sheet_for_the_root_album_keeps_its_png_extension(tmp_path):
    """``"." + ".png"`` is ``..png``, which Path reads as a dotfile with no
    suffix---PIL then refuses to save it and the whole run dies."""
    rng = np.random.default_rng(0)
    t = np.arange(int(6.0 * SR)) / SR
    for i, f in enumerate((220.0, 330.0)):
        sf.write(tmp_path / f"0{i + 1} song.wav",
                 0.3 * np.sin(2 * np.pi * f * t)
                 + 1e-3 * rng.standard_normal(len(t)), SR)

    coll = musiscape.open_collection(tmp_path)
    assert coll.album_names == ["."], "fixture no longer builds a root album"

    out = thumbnails.render_collection(coll, tmp_path / "analysis",
                                       workers=1, style="chroma")

    # the root album's own cards land in this same folder, so name the
    # sheet rather than globbing for it
    sheet = out / "collection.png"
    assert sheet.is_file(), \
        f"no contact sheet; got {sorted(p.name for p in out.iterdir())}"
