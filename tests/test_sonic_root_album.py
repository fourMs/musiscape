"""``sonic``: the audio-only verb, and what it should not need.

Two things it got wrong. Its medley for the root album ``"."`` was named
``. medley.wav`` --- hidden on every Unix desktop, invisible in the file
manager the user goes looking with. And it reached into ``thumbnails`` for
one feature helper, dragging the whole plotting stack into a verb with no
visual output.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

import musiscape
from musiscape import sonic

SR = 22050


def test_root_album_medley_is_not_a_hidden_file(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(int(20.0 * SR)) / SR
    for i, f in enumerate((220.0, 330.0)):
        sf.write(tmp_path / f"0{i + 1} song.wav",
                 0.3 * np.sin(2 * np.pi * f * t)
                 + 1e-3 * rng.standard_normal(len(t)), SR)

    coll = musiscape.open_collection(tmp_path)
    assert coll.album_names == ["."], "fixture no longer builds a root album"

    out = sonic.export_collection(coll, tmp_path / "analysis", workers=1)

    medleys = [p for p in out.iterdir() if p.name.endswith("medley.wav")]
    assert medleys, f"no medley written; got {sorted(p.name for p in out.iterdir())}"
    assert not medleys[0].name.startswith("."), \
        f"medley is a hidden file: {medleys[0].name}"


def test_sonic_does_not_import_the_plotting_stack(tmp_path):
    """An audio-only verb should not need matplotlib.

    It borrowed one feature helper from ``thumbnails``, which imports
    matplotlib at module scope, so every sonic run paid for the plotting
    stack --- and imported it late, deep inside a process that had already
    loaded the audio stack.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(20.0 * SR)) / SR
    sf.write(tmp_path / "01 song.wav",
             0.3 * np.sin(2 * np.pi * 220.0 * t)
             + 1e-3 * rng.standard_normal(len(t)), SR)

    code = textwrap.dedent(f"""
        import sys
        import musiscape
        from musiscape import sonic
        coll = musiscape.open_collection({str(tmp_path)!r})
        sonic.export_collection(coll, {str(tmp_path / "out")!r}, workers=1)
        assert "matplotlib" not in sys.modules, "sonic imported matplotlib"
    """)
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
