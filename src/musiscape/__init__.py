"""musiscape — a toolbox for analysing music collections.

Existing tools answer "what is this track?"; musiscape answers **"what is
this collection?"** — corpus fingerprints, similarity landscapes, internal
consistency, key-space clustering, and categorisation that can explain
itself. Sibling of `ambiscape <https://github.com/fourMs/ambiscape>`_
(soundscapes), reusing its circular-statistics and Schaeffer machinery.
"""
from .io import open_collection  # noqa: F401

__version__ = "0.1.0"
