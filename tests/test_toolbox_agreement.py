"""Shared measures must give the same number in every fourMs toolbox.

musiscape, ambiscape and micromotion are separate packages on separate
release cycles, and each has to work with none of the others installed. That
independence is only safe if a measure appearing in more than one of them is
computed the same way, so a user switching between toolboxes or combining
them is not handed two different answers to one question.

micromotion owns circular statistics. These tests pin musiscape's use of
them to that owner, and, when ambiscape happens to be installed alongside,
check its standalone copy against the same reference. The ambiscape checks
skip cleanly when it is absent, since musiscape must not require it.
"""
import numpy as np
import pytest

from micromotion.circular import circ_mean, circular_sd, rayleigh_from_R

CASES = 200


def _random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 200))
    return rng.uniform(0, 2 * np.pi, n), rng.random(n)


def test_musiscape_fifths_center_matches_micromotion():
    """``fifths_center`` must be micromotion's circular mean, not a copy."""
    from musiscape.music import FIFTHS_ANGLE, fifths_center

    rng = np.random.default_rng(7)
    for _ in range(CASES):
        chroma = rng.random(12)
        got = fifths_center(chroma)
        want = circ_mean(FIFTHS_ANGLE, weights=chroma)

        assert got["R"] == pytest.approx(want["R"], abs=1e-4)


def test_musiscape_pulse_clarity_R_is_a_micromotion_resultant():
    """The pulse-clarity score is a resultant length, on micromotion's
    definition, so it is comparable with the same number elsewhere."""
    librosa = pytest.importorskip("librosa")
    from musiscape.music import pulse_clarity

    sr = 22050
    t = np.arange(int(20.0 * sr)) / sr
    y = np.zeros(len(t))
    for s in np.arange(0.0, 19.5, 0.5):
        i = int(s * sr)
        seg = np.arange(min(len(t) - i, int(0.2 * sr))) / sr
        y[i:i + len(seg)] += np.exp(-seg / 0.03) * np.sin(2 * np.pi * 440 * seg)

    res = pulse_clarity(0.4 * y, sr)

    assert 0.0 <= res["R"] <= 1.0
    assert res["rayleigh_p"] == pytest.approx(
        rayleigh_from_R(res["R"], res["n_onsets"]), rel=1e-9)


# --------------------------------------------------------------------------
# Cross-toolbox: only when the sibling is installed. The import is inside
# each test rather than at module scope, where skipping would take the
# musiscape checks above with it.

def test_ambiscape_circular_mean_agrees_with_micromotion():
    ambi = pytest.importorskip("ambiscape.circstats",
                               reason="ambiscape not installed")
    for seed in range(CASES):
        angles, w = _random_case(seed)
        mu_a, r_a = ambi.mean_resultant(angles, weights=w)
        want = circ_mean(angles, weights=w)

        assert r_a == pytest.approx(want["R"], abs=1e-12)
        assert np.abs((mu_a - want["mean"] + np.pi) % (2 * np.pi) - np.pi) < 1e-12


def test_ambiscape_rayleigh_and_sd_agree_with_micromotion():
    """These two disagreed once, on about a fifth of random cases, because
    the packages used different published approximations of the same test."""
    ambi = pytest.importorskip("ambiscape.circstats",
                               reason="ambiscape not installed")
    for seed in range(CASES):
        angles, w = _random_case(seed)
        _, R = ambi.mean_resultant(angles, weights=w)
        n = len(angles)

        assert ambi.circular_sd(R) == pytest.approx(circular_sd(R), rel=1e-12)
        assert ambi.rayleigh_p(R, n) == pytest.approx(
            rayleigh_from_R(R, n), rel=1e-12)
