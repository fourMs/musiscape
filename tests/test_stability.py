"""Does an estimate hold up across the track, or only on average?

The gated descriptors in ``features`` are whole-track averages. These tests
pin the shorter-timescale cross-check: an estimate that survives window by
window is real, one that scatters is not, and the two cases must come back
with visibly different numbers.
"""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from musiscape import stability

SR = 22050
HOP = 512


def _triad(dur, root_hz, seed=0):
    """A sustained major triad---one unambiguous key, no rhythm."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * t)
            for f in (root_hz, root_hz * 2 ** (4 / 12), root_hz * 2 ** (7 / 12)))
    return 0.3 * y / 3 + 1e-4 * rng.standard_normal(len(t))


def _clicks(dur, bpm, seed=0):
    """A click train at a fixed tempo."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.0, dur - 0.2, 60.0 / bpm):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.1 * SR))) / SR
        y[i:i + len(seg)] += np.exp(-seg / 0.02) * np.sin(2 * np.pi * 880 * seg)
    return 0.4 * y + 1e-4 * rng.standard_normal(len(t))


def _chroma(y):
    return librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)


def _env(y):
    return librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)


# --------------------------------------------------------------------------
# key

def test_key_stability_is_total_for_a_track_that_never_leaves_its_key():
    res = stability.key_stability(_chroma(_triad(60.0, 261.63)), SR,
                                  win_s=20.0)

    assert res["n_windows"] == 3
    assert res["agreement"] == 1.0


def test_key_stability_falls_when_the_track_changes_key_halfway():
    y = np.concatenate([_triad(40.0, 261.63), _triad(40.0, 369.99, seed=1)])

    res = stability.key_stability(_chroma(y), SR, win_s=20.0)

    assert res["n_windows"] == 4
    assert res["agreement"] == pytest.approx(0.5, abs=0.26)
    assert res["agreement"] < 1.0, "a key change must cost agreement"


def test_key_stability_reports_no_agreement_from_a_single_window():
    """One window agrees with itself trivially, which is not evidence."""
    res = stability.key_stability(_chroma(_triad(10.0, 261.63)), SR,
                                  win_s=20.0)

    assert res["n_windows"] == 1
    assert res["agreement"] is None


# --------------------------------------------------------------------------
# tempo

def test_tempo_stability_recovers_a_steady_click_train():
    res = stability.tempo_stability(_env(_clicks(60.0, 120.0)), SR,
                                    win_s=20.0)

    assert res["tempo_bpm"] == pytest.approx(120.0, rel=0.05)
    assert res["agreement"] == 1.0
    assert res["n_windows"] == 3


def test_tempo_stability_falls_when_the_tempo_steps_mid_track():
    """Window agreement is the measure that moves here.

    ``beat_track`` imposes one global grid on the whole track, so the
    spacing of its beats barely notices a tempo change --- a steady 120 BPM
    train and a 120-then-80 train come back with beats 0.5 s apart either
    way. Only the per-window estimate sees the step.
    """
    steady = stability.tempo_stability(_env(_clicks(60.0, 120.0)), SR,
                                       win_s=20.0)
    stepped = stability.tempo_stability(
        _env(np.concatenate([_clicks(30.0, 120.0), _clicks(30.0, 80.0)])),
        SR, win_s=20.0)

    assert stepped["agreement"] < steady["agreement"]


def test_tempo_stability_still_reports_a_tempo_under_drift():
    """Drift is not absence of tempo.

    ``pulse_R`` folds the whole take at one period and collapses here. The
    windowed estimate still lands on the right number; what it loses is
    agreement, which is the honest thing to lose.
    """
    drifting = _env(np.concatenate([_clicks(20.0, 116.0), _clicks(20.0, 120.0),
                                    _clicks(20.0, 124.0)]))

    res = stability.tempo_stability(drifting, SR, win_s=20.0)

    assert res["tempo_bpm"] == pytest.approx(120.0, rel=0.08)
    assert res["n_windows"] == 3


# --------------------------------------------------------------------------
# integration: the descriptors must reach features.json

def test_extract_track_reports_the_stability_of_its_own_estimates():
    """``key`` and ``tempo_bpm`` ship with a note on whether they held."""
    from musiscape.features import extract_track

    y = np.concatenate([_triad(25.0, 261.63) + _clicks(25.0, 120.0),
                        _triad(25.0, 261.63, seed=2) + _clicks(25.0, 120.0)])

    feat = extract_track(y, SR)

    assert feat["key_agreement"] == pytest.approx(1.0)
    assert feat["key_windows"] >= 2
    assert feat["tempo_agreement"] is not None
