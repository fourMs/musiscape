"""Classifying what a concert recording is doing, second by second.

Four things happen in a hall: the band plays, the room applauds, someone
speaks, and nothing much happens. The classifier tells them apart from cheap
spectral features, because a fifty-minute recording cannot afford anything
expensive per frame.

The fixtures here are built *at* the values one real concert measured, rather
than from whatever a convenient synthetic signal happens to produce. White
noise measures a spectral flatness of about -0.25 where a clapping room
measures -1.38, so a test built on white noise would be testing a signal that
does not occur. :func:`_noise_at` searches for the spectral tilt that lands
on a stated flatness, which lets each test say plainly which real material it
stands for.
"""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from musiscape import concert

SR = 22050

#: Measured over one 53-minute concert: (log10 flatness, centroid Hz).
CONCERT = {
    "music": -3.06,
    "applause": -1.38,
    "voices": -1.86,
    "quiet": -1.73,
}


def _flatness(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    return float(np.log10(
        librosa.feature.spectral_flatness(S=S)[0] + 1e-12).mean())


def _tilted(n, exponent, rng):
    """Noise with a 1/f**exponent spectrum, peak-normalised."""
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    y = np.fft.irfft(w / f ** exponent, n)
    return y / (np.abs(y).max() + 1e-9)


def _noise_at(dur, target, seed=0):
    """Noise measuring ``target`` log-flatness, found by bisecting the tilt.

    Flatness falls monotonically as the spectrum tilts, so a plain bisection
    lands on any value a real room produces.
    """
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    lo, hi = 0.0, 3.0
    y = _tilted(n, 1.0, rng)
    for _ in range(18):
        mid = (lo + hi) / 2
        y = _tilted(n, mid, np.random.default_rng(seed))
        if _flatness(y) > target:
            lo = mid
        else:
            hi = mid
    return y


def _blend_to(tonal, noise, target):
    """Mix a tonal signal into noise until the pair measures ``target``."""
    lo, hi = 0.0, 1.0
    out = noise
    for _ in range(18):
        mid = (lo + hi) / 2
        out = (1 - mid) * noise + mid * tonal
        if _flatness(out) > target:
            lo = mid
        else:
            hi = mid
    return out / (np.abs(out).max() + 1e-9)


def _triad(dur, root=220.0):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.0, dur - 0.5, 0.5):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.5 * SR))) / SR
        env = np.exp(-seg / 0.15)
        for f in (root, root * 1.25, root * 1.5):
            y[i:i + len(seg)] += env * np.sin(2 * np.pi * f * seg)
    return y / (np.abs(y).max() + 1e-9)


def _music(dur, root=220.0, seed=0):
    """Tonal and loud, at the concert's measured music flatness."""
    return 0.30 * _blend_to(_triad(dur, root),
                            _noise_at(dur, -0.9, seed + 1), CONCERT["music"])


def _applause(dur, seed=0):
    """A clapping room: broadband, bright, and stationary."""
    return 0.12 * _noise_at(dur, CONCERT["applause"], seed + 2)


def _quiet(dur, seed=0):
    """Room tone, far below everything else and unchanging."""
    return 0.008 * _noise_at(dur, CONCERT["quiet"], seed + 3)


def _voices(dur, seed=0):
    """Speech in a live room.

    What separates it from applause is not its rate but its *variability*:
    it swings between voiced and unvoiced several times a second, where a
    clapping room is stationary. Built by alternating a tonal and a noisy
    segment, then set to the concert's measured voices flatness.
    """
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    tonal = np.zeros(n)
    voiced = True
    for s in np.arange(0.0, dur - 0.2, 0.18):
        i = int(s * SR)
        k = min(n - i, int(0.16 * SR))
        seg = np.arange(k) / SR
        if voiced:
            for h in range(1, 9):
                tonal[i:i + k] += np.sin(2 * np.pi * 190 * h * seg) / h
        voiced = not voiced
    tonal = tonal / (np.abs(tonal).max() + 1e-9)
    return 0.10 * _blend_to(tonal, _noise_at(dur, -0.9, seed + 4),
                            CONCERT["voices"])


def _labels(y):
    return concert.classify_regions(y, SR, hop_s=1.0)


def _dominant(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return str(vals[np.argmax(counts)])


# --------------------------------------------------------------------------
# the fixtures must stand for the material they claim to

@pytest.mark.parametrize("name,fn", [("music", _music), ("applause", _applause),
                                     ("voices", _voices), ("quiet", _quiet)])
def test_fixture_measures_like_the_concert_material_it_stands_for(name, fn):
    assert _flatness(fn(20.0)) == pytest.approx(CONCERT[name], abs=0.25)


# --------------------------------------------------------------------------
# each class

def test_music_is_labelled_music():
    assert _dominant(_labels(_music(40.0))) == "music"


def test_applause_is_labelled_applause():
    assert _dominant(_labels(_applause(40.0))) == "applause"


def test_room_tone_is_labelled_quiet():
    labels = _labels(np.concatenate([_applause(20.0), _quiet(40.0)]))

    assert _dominant(labels[25:58]) == "quiet"


def test_voices_are_told_apart_from_applause():
    """The one that matters. Neither is tonal, so a flatness threshold on its
    own puts them in the same bin."""
    labels = _labels(np.concatenate([_applause(30.0), _voices(30.0)]))

    assert _dominant(labels[2:28]) == "applause"
    assert _dominant(labels[32:58]) == "voices"


# --------------------------------------------------------------------------
# spans

def test_regions_merges_frames_into_labelled_spans():
    y = np.concatenate([_quiet(20.0), _music(60.0), _applause(20.0)])

    spans = concert.regions(_labels(y), hop_s=1.0, min_s=5.0)

    assert [s["label"] for s in spans] == ["quiet", "music", "applause"]
    assert spans[1]["start_s"] == pytest.approx(20.0, abs=4.0)
    assert spans[1]["end_s"] == pytest.approx(80.0, abs=4.0)
    assert spans[-1]["end_s"] == pytest.approx(100.0, abs=4.0)


def test_regions_absorbs_a_span_too_short_to_report():
    """A one-second flicker is a classifier wobble, not an event."""
    labels = np.array(["music"] * 60, dtype=object)
    labels[30] = "applause"

    spans = concert.regions(labels, hop_s=1.0, min_s=5.0)

    assert [s["label"] for s in spans] == ["music"]
    assert spans[0]["duration_s"] == pytest.approx(60.0, abs=1.0)


def test_map_regions_takes_the_songs_as_given(tmp_path):
    """The timeline and the setlist must agree about where the songs are.

    ``find_songs`` bridges a gap of a few seconds mid-song; the frame
    classifier does not, so left to itself it reports one song as three.
    Where a setlist exists it decides, and the classifier describes only
    what is between.
    """
    import soundfile as sf

    y = np.concatenate([_quiet(20.0), _music(30.0), _quiet(6.0),
                        _music(30.0), _applause(20.0)])
    sf.write(tmp_path / "a.wav", y, SR)
    songs = [{"start_s": 20.0, "end_s": 86.0, "duration_s": 66.0,
              "parts": [{"file": "a.wav", "start_s": 20.0, "end_s": 86.0}]}]

    rmap = concert.map_regions([tmp_path / "a.wav"], songs=songs)

    music = [s for s in rmap["spans"] if s["label"] == "music"]
    assert len(music) == 1, [s["label"] for s in rmap["spans"]]
    assert music[0]["start_s"] == pytest.approx(20.0, abs=1.5)
    assert music[0]["end_s"] == pytest.approx(86.0, abs=1.5)
