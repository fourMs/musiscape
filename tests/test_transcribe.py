"""Notes onto the second clock, and a transcription when the model is there.

`notes_per_second` is pure and tested exactly. `transcribe_piano` needs an optional package and
a 170 MB checkpoint, so it runs only where both are present, on a synthetic piano-like tone
whose pitch it must recover.
"""
import os

import numpy as np
import pandas as pd
import pytest

from musiscape import transcribe


def test_notes_per_second_folds_exactly():
    notes = pd.DataFrame({"onset_s": [0.2, 0.7, 1.5, 3.1], "offset_s": [0.9, 2.4, 1.8, 3.4],
                          "midi": [60, 64, 67, 72], "velocity": [40, 80, 60, 100]})
    out = transcribe.notes_per_second(notes, 4.0)
    assert out["density"].tolist() == [2, 1, 0, 1]
    assert out["pitch_mean"][0] == 62 and np.isnan(out["pitch_mean"][2]) and out["velocity_mean"][3] == 100
    assert out["pitch_spread"][0] == 2 and np.isnan(out["pitch_spread"][1])
    assert out["sustain"].tolist() == [2, 2, 1, 1]      # the 0.7 s note sounds into second 2


def test_notes_per_second_handles_empty():
    out = transcribe.notes_per_second(pd.DataFrame(columns=["onset_s", "offset_s", "midi", "velocity"]), 3.0)
    assert out["density"].tolist() == [0, 0, 0] and np.isnan(out["pitch_mean"]).all()


def test_availability_is_a_bool():
    assert transcribe.transcription_available() in (True, False)


@pytest.mark.skipif(not transcribe.transcription_available() or
                    not os.path.isdir(os.path.expanduser("~/piano_transcription_inference_data")),
                    reason="optional transcription package or its checkpoint not present")
def test_transcribe_recovers_a_piano_like_tone():
    sr = 16000
    t = np.arange(int(6 * sr)) / sr
    y = np.zeros_like(t)
    for k, (start, midi) in enumerate(((0.5, 60), (2.0, 67), (3.5, 72))):
        f = 440.0 * 2 ** ((midi - 69) / 12)
        seg = (t >= start) & (t < start + 1.2)
        env = np.exp(-(t[seg] - start) / 0.5)
        y[seg] += env * sum(np.sin(2 * np.pi * f * h * (t[seg] - start)) / h for h in (1, 2, 3, 4))
    y = 0.3 * y / np.abs(y).max()
    notes = transcribe.transcribe_piano(y, sr)
    assert len(notes) >= 3
    # the three fundamentals must be among the notes heard; harmonics may be heard as extra notes
    assert {60, 67, 72} <= set(notes.midi.tolist())
    assert notes.onset_s.min() < 1.0
