"""Exporting the non-music spans for a soundscape toolbox.

musiscape describes music; what happens between the songs is a soundscape
question, and ambiscape is the toolbox for those. They meet at the file
boundary, so what matters here is that the folder written carries enough for
the other tool to place the material on a clock.
"""
import datetime as dt

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from musiscape import concert
from musiscape.io import recording_start_time

SR = 22050

SPANS = [
    {"label": "music", "start_s": 0.0, "end_s": 60.0, "duration_s": 60.0},
    {"label": "applause", "start_s": 60.0, "end_s": 80.0, "duration_s": 20.0},
    {"label": "quiet", "start_s": 80.0, "end_s": 100.0, "duration_s": 20.0},
]


@pytest.fixture
def recording(tmp_path):
    rng = np.random.default_rng(0)
    sf.write(tmp_path / "cam.wav", 0.1 * rng.standard_normal(int(100 * SR)), SR)
    return tmp_path / "cam.wav"


def test_export_skips_the_music_and_writes_the_rest(recording, tmp_path):
    out = concert.export_regions([recording], tmp_path / "other", SPANS)

    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 2
    assert all("music" not in n for n in names)
    assert all(n.endswith(".flac") for n in names), names


def test_export_names_carry_the_wall_clock_when_it_is_known(recording, tmp_path):
    """ambiscape reads a leading YYYYMMDD_HHMMSS stamp and places the files
    on a real timeline; without one every span lands at the same second."""
    start = dt.datetime(2024, 4, 17, 14, 14, 10)

    out = concert.export_regions([recording], tmp_path / "other", SPANS,
                                 start_time=start)

    names = sorted(p.name for p in out.iterdir())
    assert names[0].startswith("20240417_141510"), names   # 60 s in
    assert names[1].startswith("20240417_141530"), names   # 80 s in


def test_export_falls_back_to_concert_timecodes(recording, tmp_path):
    out = concert.export_regions([recording], tmp_path / "other", SPANS)

    names = sorted(p.name for p in out.iterdir())
    assert names[0].startswith("0100-0120"), names


def test_recording_start_time_reads_a_filename_stamp(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "20240417_141410 cam.wav"
    sf.write(p, 0.1 * rng.standard_normal(SR), SR)

    assert recording_start_time(p) == dt.datetime(2024, 4, 17, 14, 14, 10)


def test_recording_start_time_is_none_when_nothing_says(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "cam.wav"
    sf.write(p, 0.1 * rng.standard_normal(SR), SR)

    assert recording_start_time(p) is None
