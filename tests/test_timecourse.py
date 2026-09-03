"""The time-course must move when the music moves, and hold still when it does not.

A triad that changes key half-way must show the change in the windowed keys and in the harmonic
change function; a click train must give its tempo and a clear pulse; a rising tone must raise
the register. The foreground proxies must put a pitched tone under ``pitched`` and a bass hum
under ``low``.
"""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from musiscape import foreground, timecourse

SR = 22050


def _triad(dur, root_hz, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * t) for f in (root_hz, root_hz * 2 ** (4 / 12), root_hz * 2 ** (7 / 12)))
    return 0.3 * y / 3 + 1e-4 * rng.standard_normal(len(t))


def _clicks(dur, bpm, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.0, dur - 0.2, 60.0 / bpm):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.1 * SR))) / SR
        y[i:i + len(seg)] += np.exp(-seg / 0.02) * np.sin(2 * np.pi * 880 * seg)
    return 0.4 * y + 1e-4 * rng.standard_normal(len(t))


def test_key_from_chroma_names_the_triad():
    v = np.zeros(12); v[[0, 4, 7]] = 1.0
    name, r = timecourse.key_from_chroma(v)
    assert name == "C major" and r > 0.6
    assert timecourse.key_from_chroma(np.ones(12))[1] == 0.0


def test_key_change_appears_in_windows_and_hcdf():
    y = np.r_[_triad(40, 261.63), _triad(40, 392.0)]  # C major then G major
    tc = timecourse.music_timecourse(y, SR, key_window_s=20, key_step_s=10)
    early = [k[1] for k in tc["keys"] if k[0] < 30]
    late = [k[1] for k in tc["keys"] if k[0] > 50]
    assert all(k == "C major" for k in early) and all(k == "G major" for k in late)
    # harmonic change peaks at the boundary
    assert 36 <= int(np.argmax(tc["hcdf"])) <= 43
    assert len(tc["t"]) == 80 and tc["chroma"].shape == (12, 80)


def test_click_train_tempo_and_pulse_clarity():
    tc = timecourse.music_timecourse(_clicks(40, 120), SR, tempo_window_s=20, key_step_s=10)
    bpm = np.array([w[1] for w in tc["tempo_windows"]])
    assert np.all(np.abs(bpm - 120) < 3) or np.all(np.abs(bpm - 60) < 3)
    assert np.nanmedian(tc["pulse_clarity"][5:-5]) > 1.5


def test_register_rises_with_pitch():
    y = np.r_[_triad(20, 130.81), _triad(20, 523.25)]  # C3 then C5 triads
    tc = timecourse.music_timecourse(y, SR)
    assert tc["register_midi"][5:15].mean() + 12 < tc["register_midi"][25:35].mean()


def test_section_summary_rows():
    y = np.r_[_triad(30, 261.63), _triad(30, 392.0)]
    tc = timecourse.music_timecourse(y, SR)
    rows = timecourse.section_summary(tc, [30])
    assert [r["section"] for r in rows] == [1, 2]
    assert rows[0]["key"] == "C major" and rows[1]["key"] == "G major"
    assert rows[0]["harmonic_ratio"] > 0.9


def test_timecourse_figures_written(tmp_path):
    tc = timecourse.music_timecourse(_triad(20, 261.63), SR)
    paths = timecourse.timecourse_figures(tc, tmp_path, prefix="x_", title="test", boundaries=[10])
    for k in ("chromagram", "tempogram", "timecourse", "chromagram_raw", "tempogram_raw"):
        assert paths[k].exists() if hasattr(paths[k], "exists") else __import__("os").path.exists(paths[k])


def test_foreground_separates_tone_from_bass():
    rng = np.random.default_rng(1)
    t = np.arange(20 * SR) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) * (t < 10)         # pitched for the first 10 s
    bass = 0.3 * np.sin(2 * np.pi * 60 * t) * (t >= 10)          # bass for the last 10 s
    fg = foreground.instrument_foreground(tone + bass + 1e-4 * rng.standard_normal(len(t)), SR)
    assert fg["pitched"][:10].mean() > fg["pitched"][10:].mean() * 3
    assert fg["low"][10:].mean() > fg["low"][:10].mean() * 3
    lab = foreground.foreground_labels(fg)
    assert (lab[1:9] == "pitched").mean() > 0.7 and (lab[11:19] == "low").mean() > 0.7
    assert np.nanmedian(fg["f0_midi"][1:9]) == pytest.approx(69, abs=1)
