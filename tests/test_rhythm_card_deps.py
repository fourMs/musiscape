"""The rhythm card's beat-wheel inset must not need an undeclared package.

The inset is documented behaviour, and it was drawn from ``ambiscape``
inside a ``try``. ambiscape is not a dependency, so for anyone installing
musiscape from PyPI the inset silently did not appear.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

SR = 22050


def test_rhythm_card_draws_its_beat_wheel_without_ambiscape(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(int(20.0 * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.0, 19.5, 0.5):                    # 120 BPM
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.2 * SR))) / SR
        y[i:i + len(seg)] += np.exp(-seg / 0.03) * np.sin(2 * np.pi * 440 * seg)
    sf.write(tmp_path / "01 beat.wav", 0.4 * y + 1e-4 * rng.standard_normal(len(t)), SR)

    code = textwrap.dedent(f"""
        import sys
        sys.modules["ambiscape"] = None          # simulate a plain install
        import musiscape
        from musiscape import thumbnails
        coll = musiscape.open_collection({str(tmp_path)!r})
        thumbnails.render_collection(coll, {str(tmp_path / "out")!r},
                                     workers=1, style="rhythm")
        from musiscape.io import load
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y, sr = load(coll.tracks[0])
        fig, ax = plt.subplots()
        thumbnails._draw_main(ax, y, sr, "rhythm")
        # the inset is a child of the main axes, not a figure-level one
        assert any(getattr(a, "get_theta_direction", None)
                   for a in ax.child_axes), "no beat-wheel inset drawn"
    """)
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2500:]
