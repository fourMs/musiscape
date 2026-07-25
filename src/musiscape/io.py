"""Collections, albums, tracks — the folder-shaped data model.

A *collection* is a folder tree of audio files. Every directory that
directly contains audio becomes an *album* (named by its path relative to
the root; files sitting in the root itself form the album ``"."``), and
every audio file a *track* named by its stem. Metadata tags are deliberately
not required — the folder structure people already keep their music in is
the ground truth here; the optional ``[tags]`` extra can enrich titles later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


@dataclass
class Track:
    path: Path
    album: str

    @property
    def title(self) -> str:
        return self.path.stem


@dataclass
class Album:
    name: str
    tracks: list[Track] = field(default_factory=list)


@dataclass
class Collection:
    root: Path
    albums: list[Album]

    @property
    def tracks(self) -> list[Track]:
        return [t for a in self.albums for t in a.tracks]

    @property
    def album_names(self) -> list[str]:
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
