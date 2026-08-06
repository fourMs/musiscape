"""Per-track feature extraction—the interpretable descriptor set.

Every number here has a musicological reading: note density (plucked events
per second), brightness (spectral centroid), inharmonic texture (spectral
flatness), dynamic range, harmonic/percussive balance, estimated key
(Krumhansl–Schmuckler), pitch-class entropy, pulse clarity and tonal focus
(circular statistics via :mod:`ambiscape.music`), and a Schaeffer TARTYP
object profile. The set is deliberately small enough to explain; it does
not compete with embedding models on raw similarity.

``extract_collection`` caches to ``features.json`` in the output folder and
runs tracks in parallel; delete the file to force re-extraction.
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
    from ambiscape import music as amusic

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

        chroma = librosa.feature.chroma_cqt(y=yh, sr=sr).mean(axis=1)
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
        "fifths_center": fifths["center_note"],
        "fifths_R": fifths["R"],
        "tartyp": tartyp["dist"],
    }


def _work(job: tuple) -> dict | None:
    path, album, sr, duration = job
    try:
        t = Track(path=Path(path), album=album)
        y, sr = load(t, sr=sr, duration=duration)
        feat = extract_track(y, sr)
        feat.update({"album": album, "track": t.title})
        print(f"[{album}] {t.title}", flush=True)
        return feat
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR [{album}] {path}: {e}", flush=True)
        return None


def extract_collection(coll: Collection, out_dir: str | Path,
                       sr: int = 22050, duration: float | None = None,
                       workers: int = 4, force: bool = False) -> Path:
    """Extract every track (parallel, cached) → ``<out_dir>/features.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "features.json"
    if out.exists() and not force:
        return out
    jobs = [(str(t.path), t.album, sr, duration) for t in coll.tracks]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = [r for r in ex.map(_work, jobs) if r]
    else:
        res = [r for r in map(_work, jobs) if r]
    out.write_text(json.dumps(res, indent=1))
    return out


def load_features(path: str | Path) -> list[dict]:
    """Read a ``features.json`` produced by :func:`extract_collection`."""
    return json.loads(Path(path).read_text())
