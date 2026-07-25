# Install

```bash
pip install musiscape
```

This pulls in the scientific stack (numpy, scipy, librosa, soundfile,
matplotlib) and [ambiscape](https://github.com/fourMs/ambiscape), whose
`music` module provides the circular statistics and Schaeffer proxies.

MP3 decoding uses librosa's backends; on Linux you may want `ffmpeg`
installed system-wide for the widest codec coverage. WAV, FLAC and OGG work
out of the box via libsndfile.

Optional extras:

```bash
pip install "musiscape[tags]"    # mutagen, for metadata-tag support (planned)
```

## Development install

```bash
git clone https://github.com/fourMs/musiscape
cd musiscape
pip install -e .
pytest tests/
```
