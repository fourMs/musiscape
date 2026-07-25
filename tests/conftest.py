"""Synthetic two-album collection: plucked melodies vs. drones."""
import numpy as np
import pytest
import soundfile as sf

SR = 22050


def plucks(dur, spacing, freq, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros(len(t))
    for s in np.arange(0.2, dur - 0.3, spacing):
        i = int(s * SR)
        seg = np.arange(min(len(t) - i, int(0.25 * SR))) / SR
        y[i:i + len(seg)] += np.exp(-seg / 0.05) * np.sin(2 * np.pi * freq * seg)
    return 0.5 * y + 1e-4 * rng.standard_normal(len(t))


def drone(dur, freq, am=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    y = np.sin(2 * np.pi * freq * t) * (1 + am * np.sin(2 * np.pi * 0.3 * t))
    return 0.4 * y + 1e-4 * rng.standard_normal(len(t))


@pytest.fixture(scope="session")
def collection_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("collection")
    a = root / "plucked album"
    b = root / "drone album"
    a.mkdir(); b.mkdir()
    for i, f in enumerate([220.0, 294.0, 392.0]):
        sf.write(a / f"0{i + 1} pluck {int(f)}.wav",
                 plucks(10.0, 0.25, f, seed=i), SR)
    for i, f in enumerate([110.0, 147.0, 196.0]):
        sf.write(b / f"0{i + 1} drone {int(f)}.wav",
                 drone(10.0, f, seed=10 + i), SR)
    return root


@pytest.fixture(scope="session")
def feats(collection_dir, tmp_path_factory):
    import musiscape
    from musiscape import features
    coll = musiscape.open_collection(collection_dir)
    out = tmp_path_factory.mktemp("analysis")
    path = features.extract_collection(coll, out, workers=1)
    return features.load_features(path)
