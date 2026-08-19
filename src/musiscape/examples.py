"""A synthetic music collection, so the documentation runs with no music of your own.

Every example in the guides points at `~/Music/my-collection`, which is a folder the
reader may not have and which certainly does not contain what the text assumes. This
module writes a small collection that does: two albums of three tracks, one plucked and
one drone, chosen so that the contrast the analysis is supposed to find is actually
there.

    >>> import musiscape as ms
    >>> root = ms.examples.demo_collection("/tmp/demo")
    >>> coll = ms.open_collection(root)
    >>> coll.album_names
    ['drone album', 'plucked album']

Then anything in the guides works against `root`, including the command line::

    musiscape report /tmp/demo

It is SYNTHETIC. The tracks are sine tones with envelopes, not music, so they are useful
for learning what each command produces and for checking an installation, and useless as
material for a claim about music. The test suite builds its collection with these same
functions, which is why they are worth trusting to run and not worth citing.
"""

from __future__ import annotations

import pathlib

import numpy as np

__all__ = ["SR", "plucks", "drone", "demo_collection"]

SR = 22050

# Two albums, three tracks each, at pitches far enough apart to be separable and close
# enough to sit in one key region. Frozen: the doctest above and the test suite's
# assertions about album separation both depend on these exact values.
_PLUCK_FREQS = (220.0, 294.0, 392.0)
_DRONE_FREQS = (110.0, 147.0, 196.0)


def plucks(dur: float, spacing: float, freq: float, seed: int = 0) -> np.ndarray:
    """A repeating plucked note: sharp onsets, fast decay, high onset rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.2, dur - 0.3, spacing):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.25 * SR))) / SR
        y[i:i + len(seg)] += np.exp(-seg / 0.05) * np.sin(2 * np.pi * freq * seg)
    return 0.5 * y + 1e-4 * rng.standard_normal(len(t))


def drone(dur: float, freq: float, am: float = 0.3, seed: int = 0) -> np.ndarray:
    """A sustained tone with slow amplitude modulation: almost no onsets."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.sin(2 * np.pi * freq * t) * (1 + am * np.sin(2 * np.pi * 0.3 * t))
    return 0.4 * y + 1e-4 * rng.standard_normal(len(t))


def demo_collection(path, dur_s: float = 10.0) -> pathlib.Path:
    """Write the two-album demo collection under ``path`` and return that folder.

    Six WAV files of ``dur_s`` seconds each, about 1.3 MB at the default. Existing files
    are overwritten, so calling it twice is safe.

    Args:
        path: Folder to write into. Created if it does not exist.
        dur_s (float): Seconds per track. Shorten it to make a smoke test quick; the
            defaults in the guides assume 10.

    Returns:
        pathlib.Path: The collection root, ready for :func:`musiscape.open_collection`.
    """
    import soundfile as sf

    root = pathlib.Path(path).expanduser()
    plucked, droned = root / "plucked album", root / "drone album"
    plucked.mkdir(parents=True, exist_ok=True)
    droned.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(_PLUCK_FREQS):
        sf.write(plucked / f"0{i + 1} pluck {int(f)}.wav",
                 plucks(dur_s, 0.25, f, seed=i), SR)
    for i, f in enumerate(_DRONE_FREQS):
        sf.write(droned / f"0{i + 1} drone {int(f)}.wav",
                 drone(dur_s, f, seed=10 + i), SR)
    return root
