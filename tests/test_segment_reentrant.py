"""Running ``segment`` twice must not feed it its own output.

The default output folder sits inside the input folder, and what it writes
there is audio. A recursive scan then treats the songs from the first run as
recordings for the second.
"""
import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from musiscape.cli import main
from musiscape.io import list_recordings

SR = 22050


def _clip(path, dur, tonal):
    rng = np.random.default_rng(0)
    t = np.arange(int(dur * SR)) / SR
    if tonal:
        y = 0.3 * sum(np.sin(2 * np.pi * f * t) for f in (220, 275, 330)) / 3
    else:
        y = 0.05 * rng.standard_normal(len(t))
    sf.write(path, y, SR)


def test_list_recordings_can_skip_a_folder(tmp_path):
    _clip(tmp_path / "a.wav", 2.0, True)
    out = tmp_path / "analysis"
    out.mkdir()
    _clip(out / "already-written.wav", 2.0, True)

    found = list_recordings(tmp_path, exclude=[out])

    assert [p.name for p in found] == ["a.wav"]


def test_running_segment_twice_gives_the_same_answer(tmp_path):
    """The second run must see three recordings, not three plus its own."""
    rng = np.random.default_rng(1)
    t = np.arange(int(100.0 * SR)) / SR
    music = 0.3 * sum(np.sin(2 * np.pi * f * t) for f in (220, 275, 330)) / 3
    gap = 0.05 * rng.standard_normal(int(20.0 * SR))
    sf.write(tmp_path / "cam.wav", np.concatenate([gap, music, gap]), SR)
    out = tmp_path / "analysis"

    main(["segment", str(tmp_path), "-o", str(out), "--min-song", "30"])
    first = json.loads((out / "songs.json").read_text())

    main(["segment", str(tmp_path), "-o", str(out), "--min-song", "30"])
    second = json.loads((out / "songs.json").read_text())

    assert len(second) == len(first)
    assert [s["parts"][0]["file"] for s in second] == ["cam.wav"] * len(first)
