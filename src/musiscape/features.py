"""Per-track feature extraction—the interpretable descriptor set.

Every number here has a musicological reading: note density (plucked events
per second), brightness (spectral centroid), inharmonic texture (spectral
flatness), dynamic range, harmonic/percussive balance, estimated key
(Krumhansl–Schmuckler), pitch-class entropy, pulse clarity and tonal focus
(circular statistics via :mod:`micromotion.circular`), and a Schaeffer TARTYP
object profile. The set is deliberately small enough to explain; it does
not compete with embedding models on raw similarity.

``extract_collection`` caches to ``features.json`` in the output folder and
runs tracks in parallel; delete the file to force re-extraction.

Two descriptors answer even when there is nothing to answer, and both must be
gated before use. This matters whenever the input is not a music
collection: field recordings, broadcast audio, anything where "music" was
decided by a detector rather than by a track listing.

``tempo_bpm`` is librosa's prior-based estimator and it has no failure value:
given an onset envelope with no periodicity it returns the tempogram bin
nearest its 120 BPM prior. White noise returns 123.05 BPM, reproducibly. On
704 five-minute spans of domestic television audio it returned five distinct
values in all, 93 % of them exactly 123.0, and the other four were the
adjacent grid points, a result indistinguishable from noise. Read ``pulse_R``
first: below about 0.1 there is no pulse for a tempo to describe, and
``tempo_bpm`` is reporting the prior rather than the track.

``key`` and ``key_conf`` degrade the same way. The Krumhansl--Schmuckler
correlation is taken against whatever chroma vector arrives, including a
near-uniform one. On the same material ``chroma_entropy`` sat at a median
3.541 against a maximum of log2(12) = 3.585, with 80 % of spans within 2 % of
that ceiling: no tonal centre exists, so the estimate falls to whichever tiny
bias survives, and it does so consistently: 78 % of spans came back minor and
one key took a quarter of them. Consistency is not confidence here. Read
``chroma_entropy`` first; near the ceiling the key is an artefact, and
splitting by ``key_conf`` will not reveal it, because the artefact is
confident.

The spectral and temporal descriptors are unaffected and stay usable on such
material: onset rate, centroid, flatness, zero-crossing rate, percussive ratio
and dynamic range all varied normally on the same spans.

Both gates are whole-track averages, which is a second way to be wrong. They
catch a descriptor answering about noise, but they do not distinguish that
from a descriptor answering about four minutes of real music at too long a
timescale. On live material the second case is the common one: a band
drifting a few BPM collapses ``pulse_R`` while playing a steady beat, and a
full band in a reverberant room flattens mean chroma far past a threshold
calibrated on solo instrumental recordings.

:mod:`musiscape.stability` measures the same two quantities per window and
reports how far the windows agree, which separates the cases. Its results
travel beside the gated numbers here: ``key_agreement`` and ``key_windowed``
beside ``key``, ``tempo_agreement`` and ``tempo_windowed_bpm`` beside
``tempo_bpm``. They answer whether an estimate holds still across the track.
Nothing answers whether a track has a pulse at all; see that module.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .io import Collection, Track, load

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

#: features used for landscapes/similarity/clustering, in this order
FEATURES = ["onset_rate", "centroid_hz", "flatness", "zcr", "flux",
            "perc_ratio", "dyn_range_db", "chroma_entropy", "duration_s",
            "pulse_R"]
#: log1p-compress these before standardising (right-skewed)
LOG_FEATURES = {"onset_rate", "centroid_hz", "flatness", "zcr", "duration_s"}


def feats_2hz(y, sr):
    """Chroma, MFCC and RMS aggregated to 2 Hz frames.

    Shared by the visual cards and the sonic thumbnails. It lives here
    rather than in :mod:`thumbnails` so that :mod:`sonic`, which has no
    visual output, need not import the plotting stack to use it.
    """
    import librosa
    hop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)[1:]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    step = max(1, int(0.5 * sr / hop))
    n = max(1, C.shape[1] // step)
    agg = lambda X: np.stack([X[:, i * step:(i + 1) * step].mean(1)
                              for i in range(n)], 1)
    return agg(C), agg(M), np.array([rms[i * step:(i + 1) * step].mean()
                                     for i in range(n)])


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, float]:
    """Krumhansl–Schmuckler key estimate (name, correlation)."""
    best, best_r = "C major", -2.0
    for i in range(12):
        for prof, mode in ((_MAJ, "major"), (_MIN, "minor")):
            r = float(np.corrcoef(np.roll(prof, i), chroma_mean)[0, 1])
            if r > best_r:
                best_r, best = r, f"{KEYS[i]} {mode}"
    return best, best_r


def extract_track(y: np.ndarray, sr: int) -> dict:
    """All per-track descriptors from decoded audio."""
    import librosa
    from . import music as amusic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dur = len(y) / sr
        yt, _ = librosa.effects.trim(y, top_db=40)

        S = np.abs(librosa.stft(yt, n_fft=2048, hop_length=512))
        rms = librosa.feature.rms(S=S)[0]
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        zcr = librosa.feature.zero_crossing_rate(yt, hop_length=512)[0]
        flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
        peaks = librosa.onset.onset_detect(onset_envelope=flux, sr=sr,
                                           units="frames")
        onsets = peaks[flux[peaks] >= amusic.ONSET_FLOOR]

        yh, yp = librosa.effects.hpss(yt)
        he, pe = float(np.sum(yh ** 2)), float(np.sum(yp ** 2))

        chromagram = librosa.feature.chroma_cqt(y=yh, sr=sr)
        chroma = chromagram.mean(axis=1)
        cm = chroma / (chroma.sum() + 1e-12)
        key, key_conf = estimate_key(chroma)

        db = librosa.amplitude_to_db(rms + 1e-12)
        try:
            from librosa.feature.rhythm import tempo as _tempo
        except ImportError:                              # librosa < 0.10
            _tempo = librosa.beat.tempo
        tempo_bpm = float(_tempo(onset_envelope=flux, sr=sr)[0])
        pulse = amusic.pulse_clarity(yt, sr)
        fifths = amusic.fifths_center(chroma)
        tartyp = amusic.tartyp_profile(yt, sr)

        # The shorter-timescale cross-check for the two gated descriptors.
        # Both reuse arrays computed above, so this costs almost nothing.
        from . import stability as astab
        ks = astab.key_stability(chromagram, sr)
        ts = astab.tempo_stability(flux, sr)

    return {
        "duration_s": round(dur, 1),
        "onset_rate": round(len(onsets) / max(len(yt) / sr, 1e-9), 3),
        "centroid_hz": round(float(centroid.mean()), 1),
        "flatness": round(float(flatness.mean()), 5),
        "zcr": round(float(zcr.mean()), 5),
        "flux": round(float(flux.mean()), 3),
        "perc_ratio": round(pe / (he + pe + 1e-12), 4),
        "dyn_range_db": round(float(np.percentile(db, 95)
                                    - np.percentile(db, 10)), 2),
        "chroma_entropy": round(float(-np.sum(cm * np.log2(cm + 1e-12))), 3),
        "key": key, "key_conf": round(key_conf, 3),
        "chroma": [round(float(c), 4) for c in chroma],
        "pulse_R": pulse.get("R", 0.0),
        "pulse_bpm": pulse.get("period_bpm"),
        "tempo_bpm": round(tempo_bpm, 1),
        "key_windowed": ks["key"],
        "key_agreement": ks["agreement"],
        "key_windows": ks["n_windows"],
        "tempo_windowed_bpm": ts["tempo_bpm"],
        "tempo_agreement": ts["agreement"],
        "fifths_center": fifths["center_note"],
        "fifths_R": fifths["R"],
        "tartyp": tartyp["dist"],
    }


def _work(job: tuple) -> dict | None:
    path, album, sr, duration = job[:4]
    capped = job[4] if len(job) > 4 else False
    try:
        t = Track(path=Path(path), album=album)
        y, sr = load(t, sr=sr, duration=duration)
        feat = extract_track(y, sr)
        feat.update({"album": album, "track": t.title})
        if capped:
            feat["analysis_capped_s"] = duration
        print(f"[{album}] {t.title}"
              + (f" (capped to {duration:.0f}s)" if capped else ""), flush=True)
        return feat
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR [{album}] {path}: {e}", flush=True)
        return None


def _run_pool(indexed_jobs: list, workers: int, fn=_work) -> dict:
    """Run jobs in a process pool; return ``{index: result}`` for those done.

    A worker killed by the operating system does not raise an exception in
    the worker, because the process is gone, so the executor breaks and
    every future still pending dies with it. Returning what completed lets the
    caller retry the rest instead of losing the whole collection, which is
    what ``ProcessPoolExecutor.map`` does when it re-raises on the first
    broken future.

    The case seen in practice is an out-of-memory kill on a very long
    track: analysing three quarters of an hour of audio costs several
    gigabytes in one worker, and the kernel takes the process.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    done: dict = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, j): i for i, j in indexed_jobs}
            for f in as_completed(futs):
                try:
                    r = f.result()
                except BrokenProcessPool:
                    continue          # its worker died; the caller retries
                except Exception as e:                    # noqa: BLE001
                    print(f"ERROR job {futs[f]}: {e}", flush=True)
                    continue
                if r:
                    done[futs[f]] = r
    except BrokenProcessPool:
        pass                          # shutdown of a broken pool
    return done


def _run_isolated(job: tuple, fn=_work) -> dict | None:
    """Run one job in its own pool, so a second kill costs only this job."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    try:
        with ProcessPoolExecutor(max_workers=1) as ex:
            return ex.submit(fn, job).result()
    except BrokenProcessPool:
        print(f"ERROR [{job[1]}] {job[0]}: worker killed again, skipping "
              f"(most likely out of memory)", flush=True)
        return None
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR [{job[1]}] {job[0]}: {e}", flush=True)
        return None


def extract_collection(coll: Collection, out_dir: str | Path,
                       sr: int = 22050, duration: float | None = None,
                       workers: int = 4, force: bool = False,
                       retry_cap_s: float | None = 600.0) -> Path:
    """Extract every track (parallel, cached) → ``<out_dir>/features.json``.

    Tracks whose worker dies without raising, an out-of-memory kill on a
    very long track being the case seen in practice, are retried one at a
    time in their own process, with the analysis window capped to
    ``retry_cap_s`` seconds so the retry fits in memory. A capped result
    records ``analysis_capped_s`` so the shortened window is visible in the
    output rather than implied by a duration. Pass ``retry_cap_s=None`` to
    retry at full length, which will usually be killed again.

    Nothing is capped on the first attempt, so ordinary collections are
    extracted exactly as before.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "features.json"
    if out.exists() and not force:
        return out
    jobs = [(str(t.path), t.album, sr, duration) for t in coll.tracks]

    if workers > 1:
        results = _run_pool(list(enumerate(jobs)), workers)
    else:
        results = {i: r for i, j in enumerate(jobs) if (r := _work(j))}

    missing = [(i, j) for i, j in enumerate(jobs) if i not in results]
    if missing and workers > 1:
        print(f"{len(missing)} track(s) did not complete; retrying "
              f"individually" + (f", capped to {retry_cap_s:.0f}s"
                                 if retry_cap_s else ""), flush=True)
        for i, j in missing:
            cap = retry_cap_s
            if cap is not None and j[3] is not None:
                cap = min(j[3], cap)
            job = (j[0], j[1], j[2], cap, cap is not None)
            if r := _run_isolated(job):
                results[i] = r

    res = [results[i] for i in sorted(results)]
    out.write_text(json.dumps(res, indent=1))
    return out


def load_features(path: str | Path) -> list[dict]:
    """Read a ``features.json`` produced by :func:`extract_collection`."""
    return json.loads(Path(path).read_text())
