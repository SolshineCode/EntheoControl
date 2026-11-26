"""SOUND TAGS (soundtags.py): features, matching, the trigger logic and the CPU budget. Synthetic audio only
(tests/synth_audio.py: sine chords, drums, noise bursts, two different 'songs'); no device, no GL."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import soundtags as S  # noqa: E402
from synth_audio import drums, feed, noise_bursts, silence, song_a, song_b  # noqa: E402


def frames_of(y):
    e = S.SoundTags()
    feed(e, y)
    return e.frames_since(0)


def example_of(y):
    return S.make_example(*frames_of(y))


@pytest.fixture(scope="module")
def ex_a():
    return example_of(song_a(12))


def sims_over(y, tpl, skip_s=1.5):
    """Similarity rounds while `y` plays (the engine's own 4 Hz cadence); the first `skip_s` are ignored."""
    e = S.SoundTags()
    e.set_templates({"x": tpl})
    e.active = True
    seen, out = None, []
    y = np.asarray(y, np.float32)
    for i in range(0, len(y) - 511, 512):
        b = y[i:i + 512]
        e.feed(np.stack([b, b], axis=1))
        if e.sims_t != seen and "x" in e.sims:
            seen = e.sims_t
            if e.sims_t / S.FRAME_HZ >= S.WINDOW_S + skip_s:
                out.append(e.sims["x"])
    assert out, "no similarity rounds happened"
    return np.array(out)


# ---------------------------------------------------------------- features


def test_frame_rate_and_layout():
    fr, loud = frames_of(song_a(5))
    assert fr.shape[1] == S.D == 33
    assert abs(len(fr) - 5 * S.FRAME_HZ) <= 1
    on = loud > S.SILENCE_DB
    assert on.all()
    assert np.allclose(fr[on][:, S.CH].sum(axis=1), 1.0, atol=1e-4)          # chroma is a distribution
    b = fr[on][:, S.BASS:S.HIGH + 1].sum(axis=1)
    assert np.allclose(b, 1.0, atol=1e-4)


def test_features_are_level_invariant():
    """The same music 20 dB quieter gives the same features (only loudness differs)."""
    a, la = frames_of(song_a(8))
    b, lb = frames_of(song_a(8, gain=0.1))
    assert np.all(np.abs((la - lb) - 20.0) < 0.5)
    d = np.abs(a[8:, :S.ONS] - b[8:, :S.ONS])                                  # skip the onset warm-up
    assert d.max() < 0.02, d.max()
    assert np.abs(a[30:, S.ONS:] - b[30:, S.ONS:]).max() < 0.08


def test_chroma_finds_the_chord():
    """C major triad -> energy on C, E, G."""
    from synth_audio import note, tone
    fr, _ = frames_of(tone([note(60), note(64), note(67)], 3.0, amp=0.3))
    ch = fr[5:, S.CH].mean(axis=0)
    top = set(np.argsort(ch)[-3:])
    assert top == {0, 4, 7}, ch.round(3)


def test_tempo_from_a_drum_pattern():
    fr, loud = frames_of(drums(10, 120))
    fr = fr[-30:][loud[-30:] > S.SILENCE_DB]                                   # frames between two hits are silence
    bpm = 80 * 2 ** np.median(fr[:, S.TEMPO])
    assert abs(bpm - 120) < 3, bpm
    assert fr[:, S.BEAT].mean() > 0.3
    assert fr[:, S.ONS].mean() > 0.1                                           # several onsets / s
    fr, loud = frames_of(drums(10, 90, "break"))
    fr = fr[-30:][loud[-30:] > S.SILENCE_DB]
    bpm = 80 * 2 ** np.median(fr[:, S.TEMPO])
    assert min(abs(bpm - 90), abs(bpm - 135)) < 4, bpm                          # a broken beat may read as 3/2


def test_silence_is_flagged_and_never_matches(ex_a):
    fr, loud = frames_of(silence(6))
    assert (loud < S.SILENCE_DB).all()
    assert S.window(fr, loud) is None
    sims = sims_over(np.concatenate([silence(8)]), S.compile_tag([ex_a], "vibe"))
    assert sims.max() == 0.0


def test_feed_takes_any_block_size_mono_stereo_and_glitches():
    y = song_a(4).astype(np.float32)
    a = S.SoundTags()
    feed(a, y, block=512)
    b = S.SoundTags()
    i = 0
    rng = np.random.default_rng(0)
    while i < len(y):
        n = int(rng.integers(1, 1500))
        b.feed(y[i:i + n])                                                     # mono, odd sizes
        i += n
    assert b.count == a.count
    fa, _ = a.frames_since(0)
    fb, _ = b.frames_since(0)
    assert np.allclose(fa, fb, atol=1e-5)
    c = S.SoundTags()
    bad = np.full((512, 2), np.nan, np.float32)
    for _ in range(20):
        c.feed(bad)                                                            # a device glitch never poisons it
    feed(c, y)
    fc, _ = c.frames_since(0)
    assert np.isfinite(fc).all()


def test_frames_since_is_cut_to_the_ring():
    e = S.SoundTags(history_s=10.0)
    feed(e, song_a(14))
    fr, loud = e.frames_since(0)
    assert len(fr) == e.maxf < e.count


# ---------------------------------------------------------------- matching


@pytest.mark.parametrize("mode", ["vibe", "section"])
def test_same_section_matches_different_music_does_not(ex_a, mode):
    tpl = S.compile_tag([ex_a], mode)
    same = sims_over(song_a(14, offset=4), tpl)
    other = sims_over(song_b(14), tpl)
    noise = sims_over(noise_bursts(10), tpl)
    beat = sims_over(drums(12, 120), tpl)
    thr = S.threshold(0.5)
    assert np.median(same) > 0.9 and same.min() > thr, same
    assert other.max() < thr - S.HYST, other
    assert noise.max() < thr - S.HYST and beat.max() < thr - S.HYST


@pytest.mark.parametrize("mode", ["vibe", "section"])
def test_volume_changes_still_match(ex_a, mode):
    tpl = S.compile_tag([ex_a], mode)
    quiet = sims_over(song_a(12, gain=0.12), tpl)
    loud = sims_over(song_a(12, gain=2.0), tpl)
    assert quiet.min() > S.threshold(0.5) and loud.min() > S.threshold(0.5)


def test_key_change_section_mode_needs_key_invariance(ex_a):
    shifted = song_a(12, shift=3)
    plain = sims_over(shifted, S.compile_tag([ex_a], "section", key_inv=False))
    inv = sims_over(shifted, S.compile_tag([ex_a], "section", key_inv=True))
    assert plain.max() < S.threshold(0.5) - S.HYST                             # other key = other section
    assert inv.min() > S.threshold(0.5)                                        # ... unless the tag is key-invariant
    vibe = sims_over(shifted, S.compile_tag([ex_a], "vibe"))
    assert vibe.min() > S.threshold(0.5)                                       # vibe never looks at the key


def test_more_examples_widen_a_tag(ex_a):
    ex_b = example_of(song_b(12))
    both = S.compile_tag([ex_a, ex_b], "vibe")
    assert sims_over(song_a(10, offset=2), both).min() > 0.8
    assert sims_over(song_b(10), both).min() > 0.8


def test_example_is_compact_json_and_cleaning_round_trips(ex_a):
    blob = json.dumps(ex_a)
    assert len(blob) < 4000
    assert ex_a["secs"] > 10 and len(ex_a["mu"]) == len(ex_a["sd"]) == S.NV and len(ex_a["ch"]) <= S.MAX_EX_BINS
    again = S.clean_example(json.loads(blob))
    assert again == ex_a
    for bad in (None, {}, dict(ex_a, mu=ex_a["mu"][:5]), dict(ex_a, ch=[[1, 2]]), dict(ex_a, mu=["x"] * S.NV),
                dict(ex_a, sd=[float("nan")] * S.NV), dict(ex_a, ch="nope")):
        assert S.clean_example(bad) is None


def test_too_short_or_too_quiet_recording_is_refused():
    with pytest.raises(ValueError):
        S.make_example(*frames_of(song_a(2)))
    with pytest.raises(ValueError):
        S.make_example(*frames_of(np.concatenate([silence(8), song_a(1.5)])))


# ---------------------------------------------------------------- the trigger logic


def drive(dec, seq, tags, dt=0.25, **kw):
    """seq: list of {id: sim} per tick (dt apart). Returns [(t, fired id)]."""
    out = []
    for i, sims in enumerate(seq):
        t = i * dt
        k, _, _ = dec.step(t, sims, tags, **kw)
        if k:
            out.append((round(t, 2), k))
    return out


def test_threshold_from_sensitivity():
    assert S.threshold(0.5) == pytest.approx(0.7)
    assert S.threshold(1.0) < S.threshold(0.5) < S.threshold(0.0)
    assert S.threshold(5) == S.threshold(1.0)


def test_min_hold_then_fire_once_then_rearm_after_leaving():
    tags = {"a": dict(S.TAG_DEFAULTS, hold=2.0, cooldown=0.0)}
    seq = [{"a": 0.95}] * 40 + [{"a": 0.1}] * 20 + [{"a": 0.95}] * 40
    fires = drive(S.Decider(), seq, tags, gap=0.0)
    assert len(fires) == 2                                   # once per arrival, not again while it stays
    t1, t2 = fires[0][0], fires[1][0]
    assert 2.0 <= t1 <= 2.75                                 # the match had to hold 2 s (plus smoothing)
    assert t2 >= 15.0 + 2.0


def test_short_blips_never_fire():
    tags = {"a": dict(S.TAG_DEFAULTS, hold=3.0)}
    seq = ([{"a": 0.95}] * 8 + [{"a": 0.2}] * 8) * 10       # 2 s on / 2 s off
    assert drive(S.Decider(), seq, tags) == []


def test_cooldown_blocks_a_quick_return():
    tags = {"a": dict(S.TAG_DEFAULTS, hold=1.0, cooldown=30.0)}
    seq = ([{"a": 0.95}] * 20 + [{"a": 0.1}] * 20) * 6       # back every 10 s
    fires = drive(S.Decider(), seq, tags, gap=0.0)
    ts = [t for t, _ in fires]
    assert len(ts) >= 2 and all(b - a >= 30.0 for a, b in zip(ts, ts[1:]))


def test_priority_then_similarity_decides():
    tags = {"lo": dict(S.TAG_DEFAULTS, hold=1.0, prio=0), "hi": dict(S.TAG_DEFAULTS, hold=1.0, prio=5)}
    seq = [{"lo": 0.99, "hi": 0.8}] * 20
    fires = drive(S.Decider(), seq, tags)
    assert fires[0][1] == "hi"
    tags = {"a": dict(S.TAG_DEFAULTS, hold=1.0), "b": dict(S.TAG_DEFAULTS, hold=1.0)}
    fires = drive(S.Decider(), [{"a": 0.8, "b": 0.97}] * 20, tags)
    assert fires[0][1] == "b"


def test_global_gap_between_two_recalls():
    tags = {"a": dict(S.TAG_DEFAULTS, hold=1.0), "b": dict(S.TAG_DEFAULTS, hold=1.0)}
    seq = [{"a": 0.95, "b": 0.1}] * 8 + [{"a": 0.1, "b": 0.95}] * 60
    fires = drive(S.Decider(), seq, tags, gap=8.0)
    assert [k for _, k in fires] == ["a", "b"] and fires[1][0] - fires[0][0] >= 8.0


def test_blocked_and_disarmed_never_fire_and_performer_block_uses_up_the_trigger():
    tags = {"a": dict(S.TAG_DEFAULTS, hold=1.0, cooldown=0.0)}
    seq = [{"a": 0.95}] * 40
    assert drive(S.Decider(), seq, tags, armed=False) == []
    assert drive(S.Decider(), seq, tags, block="flashguard") == []
    d = S.Decider()
    drive(d, seq[:20], tags, block="performer", consume=True)
    k, meters, _ = d.step(10.0, {"a": 0.95}, tags)           # block gone, music still there: no stale recall
    assert k is None and meters[0][3] == 2
    d2 = S.Decider()
    drive(d2, seq[:20], tags, block="flashguard")            # not consumed: fires once the guard lets go
    assert d2.step(10.0, {"a": 0.95}, tags)[0] == "a"


def test_only_with_autopilot_and_disabled_tags():
    tags = {"a": dict(S.TAG_DEFAULTS, hold=1.0, only_auto=True)}
    seq = [{"a": 0.95}] * 20
    assert drive(S.Decider(), seq, tags, autopilot=False) == []
    assert drive(S.Decider(), seq, tags, autopilot=True)
    off = {"a": dict(S.TAG_DEFAULTS, hold=1.0, on=False)}
    d = S.Decider()
    assert drive(d, seq, off) == []
    assert d.step(9.0, {"a": 0.95}, off)[1][0][3] == 4


def test_decider_is_deterministic():
    rng = np.random.default_rng(3)
    seq = [{"a": float(x), "b": float(y)} for x, y in rng.random((400, 2))]
    tags = {"a": dict(S.TAG_DEFAULTS, hold=0.5, cooldown=2.0, sens=0.9), "b": dict(S.TAG_DEFAULTS, hold=0.5, sens=0.9)}
    assert drive(S.Decider(), seq, tags, gap=1.0) == drive(S.Decider(), seq, tags, gap=1.0)


# ---------------------------------------------------------------- suggestions + cost


def test_suggest_finds_the_sections_that_come_back():
    y = np.concatenate([song_a(16), song_b(16), song_a(16, offset=8), song_b(16)])
    fr, loud = frames_of(y)
    sug = S.suggest(fr, loud)
    assert len(sug) >= 2
    assert all(s["n"] >= 2 for s in sug[:2])
    a_tpl = S.compile_tag([sug[0]["example"]], "vibe")
    b_tpl = S.compile_tag([sug[1]["example"]], "vibe")
    sa = sims_over(song_a(10, offset=2), a_tpl).mean()
    sb = sims_over(song_a(10, offset=2), b_tpl).mean()
    assert max(sa, sb) > 0.85 and min(sa, sb) < 0.6           # one suggestion is section A, the other is B
    once = S.suggest(*frames_of(np.concatenate([song_a(16), song_b(16)])))
    assert once == []                                         # nothing came back yet


def test_cpu_budget_features_and_matching():
    """<= 0.3 ms per 60 fps frame equivalent on the audio thread, with 8 tags x 4 examples (section, key-invariant)."""
    ex = example_of(song_a(15))
    e = S.SoundTags()
    e.set_templates({f"t{i}": S.compile_tag([ex] * 4, "section", True) for i in range(8)})
    e.active = True
    secs = 20.0
    feed(e, song_a(secs))
    per_frame_ms = e.cost_s / secs / 60 * 1000
    print(f"[soundtags cost] {per_frame_ms:.3f} ms per 60 fps frame, match every {e.match_dt:.2f} s")
    assert per_frame_ms <= 0.3
    assert e.match_dt <= 0.5                                  # still >= 2 Hz with that many tags
    f = S.SoundTags()
    feed(f, song_a(secs))
    assert f.cost_s / secs / 60 * 1000 <= 0.2                  # features alone (no tags, matching off)


def test_batched_matcher_equals_the_reference_similarity(ex_a):
    """The engine's batched round (all examples of all tags at once) gives the per-template reference numbers."""
    ex_b = example_of(song_b(9))
    tpls = {f"t{i}": S.compile_tag([ex_a, ex_b][: 1 + i % 2] * 2, "section" if i % 2 else "vibe", i % 3 == 0)
            for i in range(8)}
    for y in (song_a(6, shift=2), song_a(8), song_b(7), silence(6)):
        e = S.SoundTags()
        feed(e, y)
        e.set_templates(tpls)
        fr, loud = e.recent(S.WINDOW_S)
        w = S.window(fr, loud)
        ref = {k: S.similarity(w, t) for k, t in tpls.items()}
        got = e.match()
        assert set(got) == set(ref)
        assert max(abs(ref[k] - got[k]) for k in ref) < 1e-5
