"""Song segmentation of a continuous concert recording.

The synthetic concert here is the shape the real material has: loud tonal
spans (songs) separated by quieter broadband spans (applause and talk).
"""
import json

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa")

from musiscape import concert
from musiscape.cli import main

SR = 22050


def _music(dur, root=220.0, seed=0):
    """A loud, tonal span: a decaying-pluck triad. Low spectral flatness."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.0, dur - 0.5, 0.5):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.5 * SR))) / SR
        env = np.exp(-seg / 0.15)
        for f in (root, root * 1.25, root * 1.5):
            y[i:i + len(seg)] += env * np.sin(2 * np.pi * f * seg)
    return 0.25 * y + 1e-4 * rng.standard_normal(len(t))


def _applause(dur, seed=0):
    """A quieter, broadband span. High spectral flatness."""
    rng = np.random.default_rng(seed)
    return 0.03 * rng.standard_normal(int(dur * SR))


def test_music_mask_separates_tonal_music_from_applause():
    """The mask is a per-second indicator, reliable in aggregate.

    Isolated frames inside a song may drop out where the music itself
    thins; bridging those is ``segments``' job, not the mask's. What the
    mask must get right is the distinction.
    """
    y = np.concatenate([_applause(20.0, seed=1), _music(40.0, seed=2)])
    mask = concert.music_mask(y, SR, hop_s=1.0)

    assert not mask[2:18].any(), "applause span marked as music"
    assert mask[22:58].mean() > 0.9, "music span not marked as music"


def test_music_mask_keeps_an_all_music_recording_whole():
    """One mode is not two: a rehearsal tape is not half applause.

    Otsu always returns a cut, so an unguarded flatness split would slice a
    single-mode recording down the middle and invent a boundary.
    """
    y = _music(60.0, seed=3)
    mask = concert.music_mask(y, SR, hop_s=1.0)

    assert mask.mean() > 0.9, "single-mode recording was split by flatness"


def test_segments_recovers_two_songs_separated_by_applause():
    y = np.concatenate([
        _applause(15.0, seed=1),
        _music(90.0, root=220.0, seed=2),
        _applause(20.0, seed=3),
        _music(80.0, root=330.0, seed=4),
        _applause(15.0, seed=5),
    ])
    songs = concert.segments(concert.music_mask(y, SR), hop_s=1.0)

    assert len(songs) == 2
    assert songs[0]["start_s"] == pytest.approx(15.0, abs=4.0)
    assert songs[0]["end_s"] == pytest.approx(105.0, abs=4.0)
    assert songs[1]["start_s"] == pytest.approx(125.0, abs=4.0)
    assert songs[1]["end_s"] == pytest.approx(205.0, abs=4.0)


def test_segments_ignores_a_span_shorter_than_the_minimum_song():
    """A few bars of tuning between songs is not a song."""
    y = np.concatenate([
        _applause(15.0, seed=1),
        _music(20.0, seed=2),            # too short to be a song
        _applause(20.0, seed=3),
        _music(90.0, seed=4),
        _applause(15.0, seed=5),
    ])
    songs = concert.segments(concert.music_mask(y, SR), hop_s=1.0,
                             min_song_s=60.0)

    assert len(songs) == 1
    assert songs[0]["start_s"] == pytest.approx(55.0, abs=4.0)


def test_segments_bridges_a_brief_dropout_inside_one_song():
    """A quiet bar mid-song must not split it into two."""
    mask = np.ones(200, dtype=bool)
    mask[100:104] = False                # 4 s dropout, shorter than min_gap

    songs = concert.segments(mask, hop_s=1.0, min_gap_s=8.0)

    assert len(songs) == 1
    assert songs[0]["duration_s"] == pytest.approx(200.0, abs=1.0)


# --------------------------------------------------------------------------
# Concerts arriving as several files: a camera splits at a size limit, and
# it splits wherever it happens to be --- routinely mid-song.

def _write(path, *parts):
    sf.write(path, np.concatenate(parts), SR)
    return path


def test_find_songs_reports_times_on_the_concert_clock(tmp_path):
    """A song in the second file is timed from the start of the concert."""
    _write(tmp_path / "a.wav", _applause(10.0, seed=1), _music(40.0, seed=2),
           _applause(10.0, seed=3))
    _write(tmp_path / "b.wav", _applause(10.0, seed=4), _music(40.0, seed=5),
           _applause(10.0, seed=6))

    songs = concert.find_songs([tmp_path / "a.wav", tmp_path / "b.wav"],
                               min_song_s=20.0)

    assert len(songs) == 2
    assert songs[1]["start_s"] == pytest.approx(70.0, abs=5.0)
    assert songs[1]["parts"][0]["start_s"] == pytest.approx(10.0, abs=5.0)


def test_find_songs_joins_a_song_split_across_two_files(tmp_path):
    """Music running to the end of one file and on from the start of the
    next is one song, not two."""
    _write(tmp_path / "a.wav", _applause(10.0, seed=1), _music(40.0, seed=2))
    _write(tmp_path / "b.wav", _music(40.0, seed=3), _applause(10.0, seed=4))

    songs = concert.find_songs([tmp_path / "a.wav", tmp_path / "b.wav"],
                               min_song_s=20.0)

    assert len(songs) == 1
    assert len(songs[0]["parts"]) == 2
    assert songs[0]["duration_s"] == pytest.approx(80.0, abs=6.0)


def test_find_songs_keeps_songs_apart_when_the_file_ends_between_them(tmp_path):
    """Applause at the end of a file is a real boundary, not a join."""
    _write(tmp_path / "a.wav", _music(40.0, seed=1), _applause(15.0, seed=2))
    _write(tmp_path / "b.wav", _applause(15.0, seed=3), _music(40.0, seed=4))

    songs = concert.find_songs([tmp_path / "a.wav", tmp_path / "b.wav"],
                               min_song_s=20.0)

    assert len(songs) == 2


def test_split_recording_writes_one_clip_per_song_and_a_manifest(tmp_path):
    _write(tmp_path / "a.wav", _applause(10.0, seed=1), _music(40.0, seed=2),
           _applause(15.0, seed=3), _music(40.0, seed=4))
    out = tmp_path / "analysis"

    manifest = concert.split_recording([tmp_path / "a.wav"], out,
                                       min_song_s=20.0)

    songs = json.loads(manifest.read_text())
    assert len(songs) == 2
    clips = sorted((out / "songs").glob("*.flac"))
    assert len(clips) == 2
    y, sr = librosa.load(str(clips[0]), sr=None)
    assert len(y) / sr == pytest.approx(40.0, abs=6.0)
    assert songs[0]["file"] == clips[0].name


def test_split_recording_writes_a_split_song_as_one_continuous_clip(tmp_path):
    """The clip for a camera-split song holds both halves, back to back."""
    _write(tmp_path / "a.wav", _applause(10.0, seed=1), _music(40.0, seed=2))
    _write(tmp_path / "b.wav", _music(40.0, seed=3), _applause(10.0, seed=4))
    out = tmp_path / "analysis"

    manifest = concert.split_recording(
        [tmp_path / "a.wav", tmp_path / "b.wav"], out, min_song_s=20.0)

    songs = json.loads(manifest.read_text())
    assert len(songs) == 1
    y, sr = librosa.load(str(out / "songs" / songs[0]["file"]), sr=None)
    assert len(y) / sr == pytest.approx(80.0, abs=6.0)


def test_cli_segment_makes_a_collection_out_of_a_recording(tmp_path):
    """``segment`` must work on a folder holding no *audio* files at all,
    which is what a folder of camera files is."""
    room = tmp_path / "concert"
    room.mkdir()
    _write(room / "a.wav", _applause(10.0, seed=1), _music(40.0, seed=2),
           _applause(15.0, seed=3), _music(40.0, seed=4))
    out = tmp_path / "analysis"

    main(["segment", str(room), "-o", str(out), "--min-song", "20"])

    songs = json.loads((out / "songs.json").read_text())
    assert len(songs) == 2
    assert sorted(p.name for p in (out / "songs").glob("*.flac")) == \
        [s["file"] for s in songs]
