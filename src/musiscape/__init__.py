"""musiscape: a Python toolbox for analysing large music collections and
long music recordings.

Existing tools answer "what is this track?". musiscape answers two questions
they do not: what is this *collection*, through corpus fingerprints,
similarity landscapes, internal consistency, key-space clustering and
categorisation that can explain itself; and what is inside this *recording*,
through song segmentation of continuous live material.

Sibling of `ambiscape <https://github.com/fourMs/ambiscape>`_ (soundscapes).
The circular statistics it builds on come from
`micromotion <https://github.com/fourMs/micromotion>`_, which owns them.

Two ways in. On the command line, ``musiscape report <folder>`` runs the whole
pipeline; from Python, start at :func:`open_collection` and reach for the module
that matches the step you want::

    import musiscape as ms

    coll = ms.open_collection("~/Music/my-collection")
    feats = ms.features.extract_collection(coll)
    stats = ms.corpus.album_stats(feats)

Every module named in the API reference is reachable here, so ``import musiscape``
is enough to find all of them. Until 0.8.0 only ``open_collection`` and ``io``
were exported, which meant the documentation listed eleven modules that a reader
following it could not actually see.

They are resolved LAZILY, on first attribute access. Importing them eagerly is the
obvious way to write this and it is wrong: ``figures`` and ``thumbnails`` pull in
matplotlib, so ``import musiscape`` would drag the plotting stack into
``musiscape sonic``, which needs none of it. ``test_sonic_does_not_import_the_plotting_stack``
exists for exactly that and caught it.
"""
import importlib

_SUBMODULES = (
    "categorize", "concert", "corpus", "features", "figures", "io", "music",
    "report", "sonic", "stability", "thumbnails",
)
# Convenience verbs, mapped to the module that owns each. Same laziness, same reason.
_VERBS = {
    "open_collection": "io",
    "extract_collection": "features",
    "extract_track": "features",
    "load_features": "features",
}

__all__ = [*_SUBMODULES, *_VERBS]


def __getattr__(name):
    """PEP 562 lazy access, so a name costs nothing until it is used."""
    if name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    elif name in _VERBS:
        value = getattr(importlib.import_module(f".{_VERBS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value          # resolve once, then it is an ordinary global
    return value


def __dir__():
    return sorted(__all__)


__version__ = "0.8.0"
