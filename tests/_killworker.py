"""Worker functions for the pool-resilience tests.

These live in their own importable module because a process pool pickles
the callable by reference: a function defined inside a test body, or
monkeypatched in the parent, is not what the child process ends up
calling.
"""
import os


def kill_on(job: tuple) -> dict | None:
    """Return a result, except for the job that kills its own process.

    ``os._exit`` bypasses interpreter shutdown, which is the point: it
    reproduces a worker dying without raising, the way an out-of-memory
    kill from the operating system does. A worker that raised would be
    reported through its future and would never break the pool.
    """
    path, album, sr, duration = job[:4]
    if "poison" in str(path):
        os._exit(1)
    return {"track": str(path), "album": album, "duration": duration}


def cap_aware(job: tuple) -> dict | None:
    """Kill the poison job only when it is run uncapped."""
    path, album, sr, duration = job[:4]
    capped = job[4] if len(job) > 4 else False
    if "poison" in str(path) and not capped:
        os._exit(1)
    out = {"track": str(path), "album": album}
    if capped:
        out["analysis_capped_s"] = duration
    return out
