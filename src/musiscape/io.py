"""Collections, albums, tracks—the folder-shaped data model.

A *collection* is a folder tree of audio files. Every directory that
directly contains audio becomes an *album* (named by its path relative to
the root; files sitting in the root itself form the album ``"."``), and
every audio file a *track* named by its stem. Metadata tags are deliberately
not required—the folder structure people already keep their music in is
the ground truth here; the optional ``[tags]`` extra is reserved for
metadata enrichment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


@dataclass
class Track:
    """One audio file: its path and the album it belongs to."""
    path: Path
    album: str

    @property
    def title(self) -> str:
        """Track title, taken from the filename stem."""
        return self.path.stem


@dataclass
class Album:
    """One folder of tracks, named by its path relative to the root."""
    name: str
    tracks: list[Track] = field(default_factory=list)


@dataclass
class Collection:
    """A scanned folder tree: the root path and its albums."""
    root: Path
    albums: list[Album]

    @property
    def tracks(self) -> list[Track]:
        """Flat list of every track across all albums."""
        return [t for a in self.albums for t in a.tracks]

    @property
    def album_names(self) -> list[str]:
        """Album names in scan (sorted-path) order."""
        return [a.name for a in self.albums]


def open_collection(root: str | Path) -> Collection:
    """Scan a folder tree into a Collection (albums sorted, tracks sorted)."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    by_dir: dict[str, list[Track]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            album = str(p.parent.relative_to(root)) or "."
            by_dir.setdefault(album, []).append(Track(path=p, album=album))
    if not by_dir:
        raise FileNotFoundError(f"no audio files under {root}")
    albums = [Album(name=k, tracks=v) for k, v in sorted(by_dir.items())]
    return Collection(root=root, albums=albums)


def load(track: Track, sr: int = 22050, duration: float | None = None):
    """Load a track as mono float audio at ``sr`` (librosa's decoders)."""
    import librosa
    y, sr = librosa.load(str(track.path), sr=sr, mono=True, duration=duration)
    return y, sr


def load_stereo(track: Track, sr: int = 22050,
                duration: float | None = None):
    """Load a track as a (2, n) stereo pair; mono files are duplicated."""
    import librosa
    import numpy as np
    y, sr = librosa.load(str(track.path), sr=sr, mono=False,
                         duration=duration)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y[:2], sr
