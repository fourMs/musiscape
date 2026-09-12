"""The PANNs segmentation chain, on synthetic posteriors (the tagger itself is an optional extra)."""
import numpy as np

from musiscape import tagging as tg


def test_group_scores_pool_by_max():
    names = ["Music", "Choir", "Speech", "Applause", "Silence", "Dog"]
    P = np.array([[0.1, 0.8, 0.0, 0.0, 0.0, 0.9]])
    g = tg.group_scores(P, names)
    assert g["music"][0] == 0.8 and g["voices"][0] == 0.0 and g["applause"][0] == 0.0


def test_decide_frames_gate_weights_and_floor():
    scores = {"music": np.array([0.7, 0.1, 0.2, 0.05, 0.05]), "voices": np.array([0.1, 0.8, 0.3, 0.05, 0.05]),
              "applause": np.array([0.0, 0.0, 0.35, 0.0, 0.0]), "quiet": np.array([0.0, 0.0, 0.0, 0.1, 0.0])}
    lab = tg.decide_frames(scores, np.array([-20.0, -30.0, -25.0, -70.0, -30.0]))
    assert list(lab) == ["music", "voices", "applause", "quiet", "other"]


def test_mode_filter_removes_a_single_flip():
    lab = np.array(["music"] * 6 + ["voices"] + ["music"] * 6, dtype=object)
    assert all(tg.mode_filter(lab, 5) == "music")


def test_runs_min_duration_and_contiguity():
    hop, win = 2.0, 4.0
    lab = np.array(["voices"] * 5 + ["music"] * 20 + ["applause"] + ["music"] * 20 + ["applause"] * 4 + ["voices"] * 5, dtype=object)
    spans = tg.runs_to_spans(lab, hop, win, len(lab) * hop)
    assert spans[0]["start_s"] == 0.0 and spans[-1]["end_s"] == len(lab) * hop
    spans = tg.enforce_min_duration(spans)
    assert [s["label"] for s in spans] == ["voices", "music", "applause", "voices"]
    for a, b in zip(spans, spans[1:]):
        assert a["end_s"] == b["start_s"]


def test_absorb_other_beside_music_but_not_beside_voices():
    sp = [tg._span("voices", 0, 10), tg._span("other", 10, 16), tg._span("music", 16, 60), tg._span("other", 60, 70),
          tg._span("music", 70, 120), tg._span("other", 120, 200), tg._span("quiet", 200, 220)]
    out = tg.absorb_other(sp)
    assert [s["label"] for s in out] == ["voices", "other", "music", "quiet"]
    assert out[2]["start_s"] == 16 and out[2]["end_s"] == 200


def test_snap_to_songs_moves_both_neighbours():
    sp = [tg._span("voices", 0, 100), tg._span("music", 100, 300), tg._span("applause", 300, 320)]
    out = tg.snap_to_songs(sp, [{"start_s": 103.0, "end_s": 297.0}])
    assert out[1]["start_s"] == 103.0 and out[1]["end_s"] == 297.0
    assert out[0]["end_s"] == 103.0 and out[2]["start_s"] == 297.0


def test_refine_music_onsets_finds_the_first_note():
    sr = 8000
    quiet = np.random.default_rng(0).normal(0, 1e-4, sr * 20)
    loud = np.sin(2 * np.pi * 220 * np.arange(sr * 20) / sr) * 0.3
    y = np.concatenate([quiet, loud])                       # the sound starts at 20.0 s
    sp = [tg._span("voices", 0, 24.0), tg._span("music", 24.0, 40.0)]   # detected 4 s late
    out = tg.refine_music_onsets(sp, y, sr)
    assert abs(out[1]["start_s"] - 20.0) < 0.3 and out[0]["end_s"] == out[1]["start_s"]


def test_segment_concert_end_to_end_with_a_stub_tagger(monkeypatch):
    sr = 8000
    names = ["Music", "Speech", "Applause", "Silence"]
    def fake_tag_frames(y, sr_, win_s=4.0, hop_s=2.0, device=None):
        n = int((len(y) / sr_ - win_s) // hop_s) + 1
        t = np.arange(n) * hop_s + win_s / 2
        P = np.zeros((n, 4), np.float32)
        P[:, 1] = np.where(t < 30, 0.9, 0.05)                 # talk, then music, then applause
        P[:, 0] = np.where((t >= 30) & (t < 90), 0.9, 0.05)
        P[:, 2] = np.where(t >= 90, 0.6, 0.0)
        return t, P, names
    monkeypatch.setattr(tg, "tag_frames", fake_tag_frames)
    y = np.sin(2 * np.pi * 220 * np.arange(sr * 100) / sr).astype(np.float32) * 0.2
    res = tg.segment_concert(y, sr, refine_onsets=False)
    assert [s["label"] for s in res["spans"]] == ["voices", "music", "applause"]
    assert res["spans"][1]["confidence"] > 0.8 and res["total_s"] == 100.0
