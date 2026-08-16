"""A span must not claim time the recording does not have.

Frames are pooled by a whole number of STFT hops, and that number is
rounded: at 22.05 kHz with a 512 hop, a one-second frame is really 43 hops,
or 0.99846 s, while the code calls it 1.0 s. Every reported time is then
about 0.15 % late, which is two seconds by the end of a 23-minute camera
file and enough to make a span end after the file it names. The error is
proportional to the rounding, so a shorter hop shows it plainly: at
``hop_s=0.1`` the frame is 0.0929 s and the clock runs 7 % fast.
"""
import json

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa")

from musiscape import concert

SR = 22050
#: deliberately not a whole number of seconds, as a camera's file is not
DUR = 40.4


def _music(dur, root=220.0, seed=0):
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


@pytest.mark.parametrize("hop_s", [0.1, 0.25, 1.0])
def test_a_frame_covers_exactly_its_hop(hop_s):
    """The clock must not drift: n frames of ``hop_s`` cover n * hop_s."""
    mask = concert.music_mask(_music(DUR), SR, hop_s=hop_s)

    assert len(mask) == int(DUR / hop_s), \
        f"{len(mask)} frames of {hop_s}s claims {len(mask) * hop_s:.2f}s of {DUR}s"


@pytest.mark.parametrize("hop_s", [0.1, 0.25, 1.0])
def test_region_features_keep_the_same_clock(hop_s):
    f = concert.region_features(_music(DUR), SR, hop_s=hop_s)

    assert len(f["db"]) == int(DUR / hop_s)


def test_song_parts_stay_inside_the_files_they_name(tmp_path):
    sf.write(tmp_path / "a.wav", _music(DUR), SR)
    sf.write(tmp_path / "b.wav", _music(DUR, 330.0, seed=1), SR)

    songs = concert.find_songs([tmp_path / "a.wav", tmp_path / "b.wav"],
                               min_song_s=20.0, hop_s=0.1)

    for song in songs:
        for part in song["parts"]:
            assert part["end_s"] <= DUR + 0.01, \
                f"{part['file']} span ends at {part['end_s']} of {DUR} s"


def test_written_clip_is_as_long_as_the_manifest_says(tmp_path):
    """The contract a reader relies on: the number in songs.json is the
    length of the file beside it."""
    sf.write(tmp_path / "a.wav", _music(DUR), SR)
    sf.write(tmp_path / "b.wav", _music(DUR, 330.0, seed=1), SR)
    out = tmp_path / "analysis"

    manifest = concert.split_recording([tmp_path / "a.wav", tmp_path / "b.wav"],
                                       out, min_song_s=20.0, hop_s=0.1)

    for song in json.loads(manifest.read_text()):
        actual = sf.info(str(out / "songs" / song["file"])).duration
        assert actual == pytest.approx(song["duration_s"], abs=0.2), \
            f"{song['file']}: file {actual:.1f}s, manifest {song['duration_s']:.1f}s"
