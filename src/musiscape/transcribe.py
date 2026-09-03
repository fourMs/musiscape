"""Notes, not onsets: piano transcription for the analyses that need to know what was played.

Everything else in this toolbox works from the signal. For a piano recording that is a loss:
an onset detector fires on attacks without saying how many notes, at what pitch, how loud, and
it fires on the pianist's chair as readily as on a chord. This module transcribes the piano
part to note events --- onset, offset, MIDI pitch, velocity --- with the high-resolution
transcription model of Kong et al. (2021), which runs on a CPU at a few times real time and
was trained on solo piano. It is only for piano, and only for recordings in which the piano
dominates; on a mixture it returns the notes it believes it hears, which may be many.

Two products follow from the notes. :func:`notes_per_second` folds them onto the one-second
clock the rest of the toolbox uses (density, mean pitch, mean velocity, pitch spread), so
they sit beside a motion or gaze track. The note onsets themselves are the events that
MGT-python's `event_alignment` tests strokes and gestures against; on the painter--pianist
session this was written for, the transcription found 58, 260 and 300 notes per minute in the
three takes where the onset detector had found 32, 78 and 83, because chords and fast
passages had been merged into single onsets.

The model is an optional dependency: ``pip install "musiscape[transcribe]"``. Its checkpoint
(about 170 MB) is downloaded on first use to ``~/piano_transcription_inference_data``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["transcribe_piano", "notes_per_second", "transcription_available"]

#: Sample rate the model expects.
MODEL_SR = 16000


def transcription_available() -> bool:
    """Whether the optional transcription package is installed."""
    try:
        import piano_transcription_inference  # noqa: F401
    except ImportError:
        return False
    return True


def _require():
    try:
        from piano_transcription_inference import PianoTranscription, sample_rate
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "piano transcription needs the optional package: pip install \"musiscape[transcribe]\" "
            "(piano_transcription_inference, which brings torch); the model checkpoint is downloaded "
            "on first use.") from e
    return PianoTranscription, sample_rate


def transcribe_piano(y, sr: int, midi_path=None, device: str = "cpu", threads: int | None = None):
    """Transcribe a piano recording to note events.

    Args:
        y: Mono audio.
        sr (int): Its sample rate; resampled to the model's 16 kHz.
        midi_path (optional): Where to write a MIDI file of the result. Defaults to none.
        device (str): ``"cpu"`` or ``"cuda"``. Defaults to CPU, which runs at a few times real time.
        threads (int, optional): Torch threads to use. Defaults to torch's choice.

    Returns:
        pandas.DataFrame: One row per note with ``onset_s``, ``offset_s``, ``midi`` and
        ``velocity`` (0–127), sorted by onset. Empty when nothing was heard.
    """
    import pandas as pd
    import librosa
    PianoTranscription, model_sr = _require()
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    y = np.asarray(y, dtype=np.float32)
    if sr != model_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=model_sr)
    tr = PianoTranscription(device=device, checkpoint_path=None)
    out = tr.transcribe(y, midi_path if midi_path is not None else None)
    events = out.get("est_note_events", [])
    if not events:
        return pd.DataFrame(columns=["onset_s", "offset_s", "midi", "velocity"])
    notes = pd.DataFrame([{"onset_s": float(n["onset_time"]), "offset_s": float(n["offset_time"]),
                           "midi": int(n["midi_note"]), "velocity": int(n["velocity"])} for n in events])
    return notes.sort_values("onset_s").reset_index(drop=True)


def notes_per_second(notes, duration_s: float, bin_s: float = 1.0) -> dict:
    """Fold note events onto a per-bin clock.

    Args:
        notes: The table from :func:`transcribe_piano` (or any with ``onset_s``, ``midi``,
            ``velocity``).
        duration_s (float): Length of the recording.
        bin_s (float): Bin width in seconds. Defaults to 1.0.

    Returns:
        dict: ``t`` (bin centres), ``density`` (notes per bin), ``pitch_mean`` and
        ``velocity_mean`` (NaN in empty bins), ``pitch_spread`` (standard deviation of pitch
        within the bin, NaN with fewer than two notes), ``sustain`` (notes sounding during the
        bin, counting offsets).
    """
    n = int(np.ceil(duration_s / bin_s))
    on = np.asarray(notes["onset_s"], dtype=float)
    off = np.asarray(notes["offset_s"], dtype=float) if "offset_s" in notes else on + 0.1
    midi = np.asarray(notes["midi"], dtype=float)
    vel = np.asarray(notes["velocity"], dtype=float)
    b = np.clip((on / bin_s).astype(int), 0, n - 1)
    density = np.bincount(b, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pitch_mean = np.bincount(b, weights=midi, minlength=n) / density
        vel_mean = np.bincount(b, weights=vel, minlength=n) / density
        sq = np.bincount(b, weights=midi ** 2, minlength=n) / density
        spread = np.sqrt(np.maximum(sq - pitch_mean ** 2, 0))
    pitch_mean[density == 0] = np.nan
    vel_mean[density == 0] = np.nan
    spread[density < 2] = np.nan
    sustain = np.zeros(n)
    for a, z in zip(on, off):
        lo, hi = int(a / bin_s), min(int(z / bin_s), n - 1)
        sustain[lo:hi + 1] += 1
    return {"t": (np.arange(n) + 0.5) * bin_s, "density": density, "pitch_mean": pitch_mean,
            "velocity_mean": vel_mean, "pitch_spread": spread, "sustain": sustain}
