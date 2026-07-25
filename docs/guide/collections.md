# Collections & tracks

`open_collection(root)` scans a folder tree: every directory that directly
contains audio becomes an `Album` (named by its path relative to the root;
files sitting in the root itself form the album `"."`), and every audio file
a `Track` named by its stem.

```python
coll = musiscape.open_collection("~/Music/my-collection")
coll.album_names        # ["ebb tide", "inta", ...]
coll.tracks             # flat list across albums
```

This is deliberately tag-free: album membership comes from folders, titles
from filenames. Two consequences worth knowing:

- Duplicates count twice. If a collection holds both `mp3/` and lossless
  `wav/` copies, both are scanned. Point musiscape at the subset you want
  analysed (or build a symlink/copy tree).
- Nested folders are albums too. `wav cd quality/Album_wav/` becomes its
  own album named by the relative path.

Supported extensions: `.mp3 .wav .flac .ogg .m4a .aiff .aif`. Decoding goes
through librosa; every analysis runs on a mono 22.05 kHz reference.
