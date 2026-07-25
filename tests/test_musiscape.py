"""End-to-end tests on the synthetic two-album collection."""
import numpy as np
import pytest

import musiscape
from musiscape import categorize, corpus, report


def test_open_collection(collection_dir):
    coll = musiscape.open_collection(collection_dir)
    assert coll.album_names == ["drone album", "plucked album"]
    assert len(coll.tracks) == 6


def test_features_separate_albums(feats):
    assert len(feats) == 6
    pluck = [f for f in feats if f["album"] == "plucked album"]
    dron = [f for f in feats if f["album"] == "drone album"]
    assert np.mean([f["onset_rate"] for f in pluck]) > \
        2 * np.mean([f["onset_rate"] for f in dron])
    assert all("tartyp" in f and "pulse_R" in f for f in feats)


def test_corpus_affinity_and_landscape(feats):
    sim = corpus.similarity(feats)
    aff = sim["affinity"]
    # within-album affinity beats cross-album affinity
    assert aff["plucked album"]["plucked album"] > \
        aff["plucked album"]["drone album"]
    land = corpus.landscape(feats)
    assert len(land["coords"]) == 6
    assert land["explained"][0] > 0.3


def test_categorize_recovers_albums(feats):
    cats = categorize.cluster(feats, k=2)
    labels = cats["labels"]
    albums = [f["album"] for f in feats]
    # the two clusters should align with the two albums (either mapping)
    per_album = {a: {labels[i] for i, x in enumerate(albums) if x == a}
                 for a in set(albums)}
    assert all(len(v) == 1 for v in per_album.values())
    assert per_album["plucked album"] != per_album["drone album"]


def test_thumbnails(collection_dir, tmp_path):
    from musiscape import thumbnails
    coll = musiscape.open_collection(collection_dir)
    out = thumbnails.render_collection(coll, tmp_path, workers=1)
    pngs = list(out.rglob("*.png"))
    # 6 track cards + 2 album contact sheets
    assert len(pngs) == 8


def test_report(collection_dir, tmp_path):
    coll = musiscape.open_collection(collection_dir)
    readme = report.run(coll, tmp_path, workers=1)
    text = readme.read_text()
    assert "collection analysis" in text
    assert (tmp_path / "fingerprints.png").exists()
    assert (tmp_path / "landscape.png").exists()
    assert (tmp_path / "affinity.png").exists()
    assert "Categories" in text
