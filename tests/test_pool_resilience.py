"""A worker killed by the operating system must not lose the collection.

Extracting features from a very long track costs several gigabytes in one
worker, and the kernel kills the process. That is not an exception: the
worker is gone, so ``ProcessPoolExecutor`` breaks and every future still
pending dies with it. ``map`` then re-raises on the first broken future and
the whole run is lost, which is what happened on a domestic recording whose
longest music span ran to forty-six minutes: one span took the worker and
the other seventy-one tracks went with it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _killworker import cap_aware, kill_on          # noqa: E402

from musiscape.features import _run_isolated, _run_pool   # noqa: E402


def _jobs(names):
    return [(i, (n, "album", 22050, None)) for i, n in enumerate(names)]


def test_pool_returns_survivors_when_a_worker_is_killed():
    jobs = _jobs(["a.wav", "poison.wav", "b.wav", "c.wav"])
    done = _run_pool(jobs, workers=2, fn=kill_on)

    assert 1 not in done, "the killed job cannot have produced a result"
    survivors = {i for i in done}
    assert survivors, "a killed worker must not lose every other track"
    # every survivor carries its own path, so results stay aligned to jobs
    for i, r in done.items():
        assert r["track"] == jobs[i][1][0]


def test_missing_jobs_are_identifiable_for_retry():
    """What the pool did not return is exactly what the caller must retry."""
    names = ["a.wav", "poison.wav", "b.wav"]
    jobs = _jobs(names)
    done = _run_pool(jobs, workers=2, fn=kill_on)
    missing = [i for i, _ in jobs if i not in done]
    assert 1 in missing


def test_isolated_retry_survives_a_second_kill():
    """A job that kills its worker again costs only itself."""
    assert _run_isolated(("poison.wav", "album", 22050, None),
                         fn=kill_on) is None
    ok = _run_isolated(("fine.wav", "album", 22050, None), fn=kill_on)
    assert ok is not None and ok["track"] == "fine.wav"


def test_capped_retry_succeeds_and_records_the_cap():
    """The retry caps the analysis window, and says so in the result."""
    job = ("poison.wav", "album", 22050, 600.0, True)
    r = _run_isolated(job, fn=cap_aware)
    assert r is not None, "a capped retry should fit in memory and complete"
    assert r["analysis_capped_s"] == 600.0, \
        "a shortened analysis window must be visible in the output"
