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

#: Video containers, accepted only where a *recording* is asked for (the
#: concert tools). Collections stay audio-only: a folder of films is not an
#: album, and letting ``open_collection`` pick them up would change what
#: every existing verb sees.
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".mts", ".m2ts"}

#: Anything the concert tools will decode.
RECORDING_EXTS = AUDIO_EXTS | VIDEO_EXTS


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


def album_stem(album: str) -> str:
    """Filename stem for a per-album output file.

    Audio sitting in the collection root forms the album ``"."``. Naming a
    file after it directly gives ``..png``, which Path reads as a dotfile
    with no suffix and PIL refuses to save, or ``". medley.wav"``, which is
    hidden on every Unix desktop. Leading dots are therefore dropped.
    """
    stem = album.replace("/", "_").lstrip(".")
    return stem or "collection"


def list_recordings(root: str | Path) -> list[Path]:
    """Recordings under ``root``, in name order, which is playing order.

    Cameras number their files sequentially, so sorting by name puts a
    split concert back in the order it was played. A folder whose files
    are named otherwise needs the order fixed by renaming.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    found = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in RECORDING_EXTS)
    if not found:
        raise FileNotFoundError(f"no recordings under {root}")
    return found


def load(track: Track, sr: int = 22050, duration: float | None = None):
    """Load a track as mono float audio at ``sr`` (librosa's decoders)."""
    import librosa
    y, sr = librosa.load(str(track.path), sr=sr, mono=True, duration=duration)
    return y, sr


def load_recording(path: str | Path, sr: int = 22050, offset: float = 0.0,
                   duration: float | None = None):
    """Load any recording, audio file or video container, as mono float.

    Audio goes through librosa as everywhere else. Video is decoded by
    ffmpeg, which is not a package dependency: it is asked for only when a
    video file is actually handed over, and its absence is reported as a
    missing program rather than a decode failure.
    """
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTS:
        import librosa
        return librosa.load(str(path), sr=sr, mono=True, offset=offset,
                            duration=duration)

    import shutil
    import subprocess

    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"reading {path.suffix} needs ffmpeg on PATH (install ffmpeg)")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{offset:.6f}", "-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}: "
                           f"{proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr


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
