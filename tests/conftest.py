"""Synthetic two-album collection: plucked melodies vs. drones.

The generators live in `musiscape.examples` now, so that the documentation can build
the same collection a reader can run against. Imported rather than copied: a fixture
that drifts from the shipped example would test something no user can reproduce.
"""
import pytest

from musiscape.examples import SR, demo_collection, drone, plucks  # noqa: F401


@pytest.fixture(scope="session")
def collection_dir(tmp_path_factory):
    return demo_collection(tmp_path_factory.mktemp("collection"))


@pytest.fixture(scope="session")
def feats(collection_dir, tmp_path_factory):
    import musiscape
    from musiscape import features
    coll = musiscape.open_collection(collection_dir)
    out = tmp_path_factory.mktemp("analysis")
    path = features.extract_collection(coll, out, workers=1)
    return features.load_features(path)
