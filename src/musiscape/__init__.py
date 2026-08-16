"""musiscape—a toolbox for analysing music collections.

Existing tools answer "what is this track?"; musiscape answers **"what is
this collection?"**—corpus fingerprints, similarity landscapes, internal
consistency, key-space clustering, and categorisation that can explain
itself. Sibling of `ambiscape <https://github.com/fourMs/ambiscape>`_
(soundscapes); the circular statistics it builds on come from
`micromotion <https://github.com/fourMs/micromotion>`_, which owns them.
"""
from .io import open_collection  # noqa: F401

__version__ = "0.6.0"
