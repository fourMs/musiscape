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
"""
from .io import open_collection  # noqa: F401

__version__ = "0.7.1"
