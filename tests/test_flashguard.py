"""Engine-wide photosensitive limiter (flashguard.py): synthetic sequences through the real GL passes, counted with
the standards' own numbers (3 flashes / s, luminance change 0.10, darker state < 0.80, more than 25 % of the frame)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flashguard as FG  # noqa: E402

W, H = 320, 200
STROBE_16 = 175.0 / 60.0 * 4.0          # 1/16 notes at 175 BPM = 11.67 Hz
STROBE_8 = 175.0 / 60.0 * 2.0


# ------------------------------------------------------------------------------------------------ helpers
def ARM(op):
    """An arming command the way remote_control hands it over when it came from the laptop itself."""
    return FG.mark_local({"cmd": "flashguard", "op": op})


def base(level=0.10, checker=0.10):
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 4), dtype="f4")
    img[..., :3] = level + checker * (((x // 16) + (y // 16)) % 2)[..., None]
    img[..., 3] = 1.0
    return img


def strobe(t, hz, rgb=(1, 1, 1), amp=0.8, mask=None):
    img = base()
    if (t * hz) % 1.0 < 0.5:
        add = amp * np.asarray(rgb, dtype="f4")
        if mask is None:
            img[..., :3] = np.clip(img[..., :3] + add, 0, 1)
        else:
            img[mask, :3] = np.clip(img[mask, :3] + add, 0, 1)
    return img


def clock(secs=3.0, fps=60.0):
    return [i / fps for i in range(int(secs * fps))]


def jittered(secs=3.0, seed=3):
    rng, t, out = np.random.default_rng(seed), 0.0, []
    while t < secs:
        out.append(t)
        t += float(rng.choice([1 / 90, 1 / 60, 1 / 60, 1 / 45, 1 / 30, 1 / 24, 0.05]))
    return out


def flashes(times, frames):
    m = np.array([FG.tile_means(np.asarray(f)[..., :3]) for f in frames])
    return (FG.count_flashes(times, m[..., 0], FG.DELTA, FG.DARK_MAX), FG.count_flashes(times, m[..., 1], FG.RED_DELTA))


def never_brighter(frames, outs, tol=3e-3):
    for a, b in zip(frames, outs):
        a3 = np.nan_to_num(np.clip(np.asarray(a)[..., :3], 0, 1))
        assert np.all(np.nan_to_num(b) <= a3 + tol)


@pytest.fixture(scope="module")
def gl():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception:
        import fx_stack
        ctx, _win = fx_stack.hidden_context()
    return ctx


def run(gl, fn, times, before=None):
    frames = [fn(t) for t in times]
    outs, fg = FG.run_sequence(gl, (W, H), frames, times) if before is None else before(frames)
    return frames, outs, fg


def same(frames, outs):
    """Bit-identical to what the engine's own texture holds (half float)."""
    return all(np.array_equal(np.asarray(a)[..., :3].astype("f2").astype("f4"), b) for a, b in zip(frames, outs))


# ------------------------------------------------------------------------------------------------ pure logic
def test_tracker_counts_opposing_transitions_only():
    tr = FG.FlashTracker(1, 0.10, 0.80)
    n = sum(int(tr.push(i / 60, [0.1 + 0.5 * (i // 3 % 2)])[0]) for i in range(60))
    assert n == 19                                             # 10 Hz square: 20 edges in 1 s, the first sets the level
    up = FG.FlashTracker(1, 0.10, 0.80)
    assert sum(int(up.push(i / 60, [i / 600])[0]) for i in range(600)) == 1          # a slow rise is ONE transition
    bright = FG.FlashTracker(1, 0.10, 0.80)
    assert sum(int(bright.push(i / 60, [0.85 + 0.12 * (i // 3 % 2)])[0]) for i in range(60)) == 0   # darker >= 0.80


def feed(lg, fn, times):
    for t in times:
        lg.push(t, fn(t))


def tiles(l, s=0.0):
    a = np.zeros((FG.GRID[1], FG.GRID[0], 4))
    a[..., 0], a[..., 1] = l, s
    return a


def test_logic_engages_on_fast_flashing_and_not_on_legal_or_small_or_slow():
    fast = FG.GuardLogic()
    seen = None
    for t in clock(1.0):
        fast.push(t, tiles(0.5 if (t * STROBE_16) % 1 < 0.5 else 0.05))
        if fast.engaged() and seen is None:
            seen = int(fast.tile_counts(fast.IN_L).max())
    assert seen is not None and seen <= 4                     # by the fourth transition: two flashes are out, no more
    assert fast.slew(1.0) == (FG.SLEW_L, FG.FREE) and fast.events == 1

    legal = FG.GuardLogic()                                    # 175 BPM on the quarter note = 2.92 flashes / s
    feed(legal, lambda t: tiles(0.5 if (t * 175 / 60) % 1 < 0.5 else 0.05), clock(6.0))
    assert not legal.engaged() and legal.events == 0 and legal.slew(6.0) is None
    assert 2.5 <= legal.meter["rate_in"] <= 3.0

    small = FG.GuardLogic()                                    # 4 of 40 tiles = 10 % of the frame

    def spot(t):
        a = tiles(0.1)
        a[0, :4, 0] = 0.9 if (t * 10) % 1 < 0.5 else 0.1
        return a
    feed(small, spot, clock(3.0))
    assert not small.engaged() and small.slew(3.0) is None

    fade = FG.GuardLogic()
    feed(fade, lambda t: tiles(0.5 + 0.45 * np.sin(t * 2 * np.pi * 0.5)), clock(6.0))
    assert not fade.engaged() and fade.events == 0


def test_logic_lets_go_after_the_flashing_stops():
    lg = FG.GuardLogic()
    feed(lg, lambda t: tiles(0.5 if (t * STROBE_16) % 1 < 0.5 and t < 1.0 else 0.05), clock(4.0))
    assert not lg.engaged() and lg.events == 1 and lg.slew(4.0) is None and lg.mode(4.0) == "idle"
    assert lg.release_until < 2.7                             # within about a second and a half of the last flash


def test_logic_red_is_caught_below_the_luminance_threshold_and_is_stricter():
    red = FG.GuardLogic()                                      # dim red: luminance swing 0.045, red energy swing 0.21
    feed(red, lambda t: tiles(0.045 if (t * 4) % 1 < 0.5 else 0.0, 0.21 if (t * 4) % 1 < 0.5 else 0.0), clock(2.0))
    assert red.red and not red.general and red.slew(2.0) == (FG.SLEW_L, FG.SLEW_S)
    edge = FG.GuardLogic()                                     # 2.8 Hz: legal for white, too fast for red here
    feed(edge, lambda t: tiles(0.045 if (t * 2.8) % 1 < 0.5 else 0.0, 0.21 if (t * 2.8) % 1 < 0.5 else 0.0), clock(3.0))
    assert edge.red


def test_logic_variable_frame_rate_uses_the_timestamps():
    lg = FG.GuardLogic()
    feed(lg, lambda t: tiles(0.6 if (t * 6.0) % 1 < 0.5 else 0.05), jittered(3.0))
    assert lg.engaged()
    ok = FG.GuardLogic()
    feed(ok, lambda t: tiles(0.6 if (t * 2.5) % 1 < 0.5 else 0.05), jittered(6.0))
    assert not ok.engaged() and ok.events == 0


def test_logic_survives_nan_and_its_state_is_plain_json():
    lg = FG.GuardLogic()
    bad = tiles(0.3)
    bad[0, 0] = np.nan
    bad[1, 1] = np.inf
    for i, t in enumerate(clock(1.0)):
        lg.push(t, bad if i % 2 else tiles(0.3))
    lg.push(float("nan"), tiles(0.3))
    lg.push(1.1, np.full((FG.GRID[1], FG.GRID[0], 4), np.nan))
    s = lg.state(1.1)
    json.dumps(s, allow_nan=False)
    assert s["mode"] in ("idle", "attenuating", "releasing") and len(json.dumps(s)) < 320


def test_override_needs_a_two_second_hold_and_never_survives_a_restart():
    lg = FG.GuardLogic()
    feed(lg, lambda t: tiles(0.5 if (t * STROBE_16) % 1 < 0.5 else 0.05), clock(1.0))
    assert lg.slew(1.0) is not None and lg.state(1.0)["mode"] == "attenuating"
    assert lg.command(ARM("arm_confirm"), wall=100.0) == "refused"                 # no hold at all
    lg.command(ARM("arm_begin"), wall=100.0)
    assert lg.state(1.0, wall=100.5)["pending"]
    assert lg.command(ARM("arm_confirm"), wall=101.0) == "refused" and not lg.armed  # 1 s is not a hold
    lg.command(ARM("arm_begin"), wall=200.0)
    assert lg.command(ARM("arm_confirm"), wall=230.0) == "refused"                 # went stale
    lg.command(ARM("arm_begin"), wall=300.0)
    lg.command({"op": "arm_cancel"}, wall=301.0)
    assert lg.command(ARM("arm_confirm"), wall=303.0) == "refused"
    lg.command(ARM("arm_begin"), wall=400.0)
    assert lg.command(ARM("arm_confirm"), wall=402.1) == "armed" and lg.armed
    s = lg.state(1.0)
    assert lg.slew(1.0) is None and s["mode"] == "override" and s["armed"] and s["would"]
    assert lg.command({"op": "disarm"}) == "disarmed" and lg.slew(1.0) is not None
    assert FG.GuardLogic().armed is False                      # a new process starts guarded


def test_validate_command():
    assert FG.validate_command({"cmd": "flashguard", "op": "disarm", "junk": 1}) == {"cmd": "flashguard", "op": "disarm"}
    for bad in ({"cmd": "flashguard"}, {"cmd": "flashguard", "op": "arm"}, {"cmd": "flashguard", "op": 3}, "x"):
        with pytest.raises(ValueError):
            FG.validate_command(bad)
    import remote_control
    assert remote_control.validate_command({"cmd": "flashguard", "op": "arm_begin"}) == {"cmd": "flashguard", "op": "arm_begin"}
    with pytest.raises(ValueError):
        remote_control.validate_command({"cmd": "flashguard", "op": "off"})


# ------------------------------------------------------------------------------------------------ GL: must attenuate
@pytest.mark.parametrize("hz", [STROBE_16, STROBE_8])
def test_fast_strobe_comes_out_at_three_flashes_or_fewer(gl, hz):
    times = clock(4.0)
    frames, outs, fg = run(gl, lambda t: strobe(t, hz), times)
    assert flashes(times, frames)[0] > 3 and flashes(times, outs)[0] <= 3
    never_brighter(frames, outs)
    s = fg.state()
    assert s["mode"] == "attenuating" and s["events"] == 1 and s["rate_in"] > 3 and s["rate_out"] <= 3
    late = [o for t, o in zip(times, outs) if t > 1.0]         # settled: the picture is still there (the dark phase)
    assert min(float(o.mean()) for o in late) > 0.09
    assert flashes([t for t in times if t > 1.0], late)[0] == 0
    fg.release()


@pytest.mark.parametrize("level", [1.0, 0.5])
def test_red_flash_at_4hz_comes_out_at_three_or_fewer(gl, level):
    times = clock(4.0)

    def red(t):
        img = np.zeros((H, W, 4), dtype="f4")
        img[..., 3] = 1.0
        img[..., 0] = level if (t * 4.0) % 1 < 0.5 else 0.0
        return img
    frames, outs, fg = run(gl, red, times)
    assert flashes(times, frames)[1] > 3
    g, r = flashes(times, outs)
    assert r <= 3 and g <= 3
    never_brighter(frames, outs)
    assert fg.state()["red"] and fg.state()["red_out"] <= 3
    fg.release()


def test_rapid_blackout_toggling_and_rapid_cuts_are_smoothed(gl):
    times = clock(4.0)
    scene_a, scene_b = base(0.55, 0.3), base(0.05, 0.05)
    for fn in (lambda t: base(0.5, 0.2) if (t * 6.0) % 1 < 0.5 else np.dstack([np.zeros((H, W, 3), "f4"), np.ones((H, W, 1), "f4")]),
               lambda t: scene_a if (t * 8.0) % 1 < 0.5 else scene_b):
        frames, outs, fg = run(gl, fn, times)
        assert flashes(times, frames)[0] > 3 and flashes(times, outs)[0] <= 3
        never_brighter(frames, outs)
        fg.release()


def test_invert_style_antiphase_flashing_is_caught(gl):
    times = clock(3.0)
    left = np.zeros((H, W), bool)
    left[:, : W // 2] = True

    def fn(t):
        on = (t * 7.0) % 1 < 0.5
        img = base()
        img[left if on else ~left, :3] += 0.7
        return img
    frames, outs, fg = run(gl, fn, times)
    assert flashes(times, frames)[0] > 3 and flashes(times, outs)[0] <= 3
    never_brighter(frames, outs)
    fg.release()


def test_steady_parts_of_the_picture_are_left_alone_while_the_rest_is_held(gl):
    times = clock(3.0)
    right = np.zeros((H, W), bool)
    right[:, W // 2:] = True

    def fn(t):
        img = strobe(t, STROBE_16, mask=right)
        img[~right, :3] = 0.2 + 0.6 * base(0.0, 1.0)[~right, :3]         # a steady, bright, high-contrast left half
        return img
    frames, outs, fg = run(gl, fn, times)
    assert fg.state()["mode"] == "attenuating" and flashes(times, outs)[0] <= 3
    for a, b in list(zip(frames, outs))[90:]:
        assert np.array_equal(a[:, : W // 2, :3].astype("f2").astype("f4"), b[:, : W // 2])
    fg.release()


def test_variable_frame_rate(gl):
    times = jittered(4.0)
    frames, outs, fg = run(gl, lambda t: strobe(t, 9.0), times)
    assert flashes(times, frames)[0] > 3 and flashes(times, outs)[0] <= 3
    never_brighter(frames, outs)
    fg.release()


def test_nan_frames_do_not_poison_the_guard(gl):
    times = clock(3.0)

    def fn(t):
        img = strobe(t, STROBE_16)
        img[10:40, 10:60, :3] = np.nan
        img[60:70, 10:60, 0] = np.inf
        return img
    frames, outs, fg = run(gl, fn, times)
    clean = [np.nan_to_num(np.clip(f, 0, 1), nan=0.0) for f in outs]
    assert flashes(times, clean)[0] <= 3
    assert np.isfinite(outs[-1][100:, 100:]).all() and float(outs[-1][100:, 100:].mean()) > 0.05
    json.dumps(fg.state(), allow_nan=False)
    fg.release()


# ------------------------------------------------------------------------------------------------ GL: must NOT touch
def test_calm_content_slow_fades_and_legal_pulses_are_bit_identical(gl):
    y, x = np.mgrid[0:H, 0:W]

    def moving(t):                                             # drifting colour field, a slow fade, a moving bar
        img = np.zeros((H, W, 4), dtype="f4")
        img[..., 0] = 0.5 + 0.4 * np.sin(x / 23.0 + t * 1.3)
        img[..., 1] = 0.5 + 0.4 * np.sin(y / 17.0 - t * 0.9)
        img[..., 2] = 0.3
        img[..., :3] *= 0.55 + 0.45 * np.sin(t * 2 * np.pi * 0.25)
        img[:, int(t * 40) % W: int(t * 40) % W + 12, :3] = 1.0
        img[..., 3] = 1.0
        return img
    def drift(t):                                              # the same field without the fade and the bar
        img = moving(0.0)
        img[..., 0] = 0.5 + 0.4 * np.sin(x / 23.0 + t * 0.02)
        return img
    for fn, secs in ((moving, 6.0), (drift, 3.0), (lambda t: strobe(t, 175 / 60), 4.0), (lambda t: base(0.5 * min(1.0, t / 3.0), 0.2), 4.0)):
        times = clock(secs)
        frames, outs, fg = run(gl, fn, times)
        assert same(frames, outs)
        if True:
            assert fg.passes == 0                              # nothing but the measuring pass ever ran
        assert fg.state()["events"] == 0
        fg.release()


def test_one_hard_cut_and_one_blackout_tap_pass_untouched(gl):
    times = clock(3.0)
    black = np.dstack([np.zeros((H, W, 3), "f4"), np.ones((H, W, 1), "f4")])
    cut = lambda t: base(0.05, 0.05) if t < 1.0 else base(0.6, 0.3)                       # noqa: E731
    tap = lambda t: black if 1.0 <= t < 1.2 else base(0.6, 0.3)                           # noqa: E731
    for fn in (cut, tap):
        frames, outs, fg = run(gl, fn, times)
        assert same(frames, outs) and fg.state()["events"] == 0
        fg.release()


def test_flicker_on_less_than_a_quarter_of_the_frame_passes(gl):
    times = clock(3.0)
    spot = np.zeros((H, W), bool)
    spot[: H // 5 * 2, : W // 8 * 2] = True                    # 2 x 2 tiles = 10 % of the frame
    frames, outs, fg = run(gl, lambda t: strobe(t, 10.0, mask=spot), times)
    assert same(frames, outs) and fg.state()["events"] == 0
    fg.release()


def test_override_passes_the_strobe_and_says_so(gl):
    times = clock(2.0)
    frames = [strobe(t, STROBE_16) for t in times]
    tex = gl.texture((W, H), 4, dtype="f2")
    fbo = gl.framebuffer([tex])
    fg = FG.FlashGuard(gl, (W, H))
    fg.logic.command(ARM("arm_begin"), wall=0.0)
    assert fg.logic.command(ARM("arm_confirm"), wall=2.5) == "armed"
    outs = []
    for img, t in zip(frames, times):
        tex.write(img.astype("f2").tobytes())
        assert fg.process(tex, t) is False
        outs.append(np.frombuffer(fbo.read(components=3, dtype="f4"), dtype="f4").reshape(H, W, 3).copy())
    assert same(frames, outs)
    s = fg.state()
    assert s["mode"] == "override" and s["would"] and s["rate_in"] > 3
    fg.command({"cmd": "flashguard", "op": "disarm"})
    for i in range(30):
        t = 2.0 + i / 60
        tex.write(strobe(t, STROBE_16).astype("f2").tobytes())
        fg.process(tex, t)
    assert fg.attenuating and fg.state()["mode"] == "attenuating"
    fg.release()


def test_guard_restores_the_callers_framebuffer_and_reads_back_late(gl):
    tex = gl.texture((W, H), 4, dtype="f2")
    mine = gl.simple_framebuffer((64, 32))
    fg = FG.FlashGuard(gl, (W, H))
    pushed = []
    real = fg.logic.push
    fg.logic.push = lambda t, a, b=None: (pushed.append(t), real(t, a, b))[1]
    for i in range(6):
        tex.write(strobe(i / 60, STROBE_16).astype("f2").tobytes())
        mine.use()
        gl.viewport = (0, 0, 64, 32)
        fg.process(tex, i / 60)
        assert gl.fbo is mine and tuple(gl.viewport) == (0, 0, 64, 32)
        assert pushed == [k / 60 for k in range(max(0, i - FG.LAG + 1))]      # a frame is read `LAG` frames later
    wrong = gl.texture((64, 64), 4, dtype="f2")
    assert fg.process(wrong, 1.0) is False                    # not its size: left alone, no crash
    fg.release()


# ------------------------------------------------------------------------------------------------ the hook
def test_engine_hook_sits_after_the_filters_and_before_every_tap():
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    hook = src.index("fguard.process, disp")
    assert src.index("filters_run(filt, disp") < hook
    assert src.rindex("scene_vao.render(moderngl.TRIANGLE_STRIP)   # <filters]") < hook
    for later in ("spout.sendTexture(disp.glo", "preview.service, remote, disp", "fbos[dst].read(components=3, attachment=0",
                  "surf_draw.render(mapper.set, disp", "out_vao.render(moderngl.TRIANGLE_STRIP)", "recorder.frame,"):
        assert hook < src.index(later), later
    # B3: the mapped frame (surfaces, grid, AUTO MAP patterns) is drawn into the guard's stage and guarded LAST
    begin, end = src.index("fguard.output_begin"), src.index("fguard.output_end")
    assert begin < src.index("surf_draw.render(mapper.set, disp") < src.index('guarded("automap", amap.draw, target') < end
    assert src.index("out_vao.render(moderngl.TRIANGLE_STRIP)") < end < src.index("recorder.frame,")
    assert "fguard.note_panics(" in src
    assert re.search(r'state\["flashguard"\]\s*=\s*fguard\.state\(\)', src)


def test_panel_indicator_card_and_the_two_second_hold(tmp_path):
    pw_api = pytest.importorskip("playwright.sync_api")
    import socket
    import threading
    import time
    import control
    from remote_control import RemoteControl
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    lg = FG.GuardLogic()
    stop, lock, box = threading.Event(), threading.Lock(), dict(strobe=False, t=0.0)

    def loop():
        while not stop.is_set():
            with lock:
                for c in ctl.apply(rc.drain()):
                    if c.get("cmd") == "flashguard":
                        lg.command(c)
                ctl.update(1 / 60, [0.5, 0.3, 0.1])
                box["t"] += 1 / 60
                t = box["t"]
                lg.push(t, tiles(0.5 if box["strobe"] and (t * STROBE_16) % 1 < 0.5 else 0.05))
                rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60,
                                      flashguard=lg.state(t)))
            time.sleep(1 / 60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        with pw_api.sync_playwright() as pw:
            br = None
            for ch in ("msedge", "chrome", None):
                try:
                    br = pw.chromium.launch(channel=ch, headless=True) if ch else pw.chromium.launch(headless=True)
                    break
                except Exception:
                    pass
            if br is None:
                pytest.skip("no browser")
            page = br.new_page(viewport=dict(width=1400, height=900))
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("document.getElementById('fgm').style.display!=='none'", timeout=8000)
            assert page.inner_text("#fgm").strip() == "FLASH GUARD" and not page.eval_on_selector("#fgm", "e=>e.classList.contains('att')")
            with lock:
                box["strobe"] = True
            page.wait_for_selector("#fgm.att", timeout=5000)
            assert "HOLDING 1" in page.inner_text("#fgm")
            page.click("#fgm")
            page.wait_for_selector("#fgcard #fgstat", timeout=5000)
            page.wait_for_function("document.getElementById('fgstat').textContent.indexOf('HOLDING')>=0", timeout=5000)
            (ROOT / "test_frames").mkdir(exist_ok=True)
            page.screenshot(path=str(ROOT / "test_frames" / "flashguard_panel.png"))
            # a short press does nothing
            box_ = page.locator("#fgarm").bounding_box()
            x, y = box_["x"] + box_["width"] / 2, box_["y"] + box_["height"] / 2
            page.mouse.move(x, y); page.mouse.down(); page.wait_for_timeout(600); page.mouse.up()
            page.wait_for_timeout(500)
            assert not lg.armed and not page.is_visible("#fgconf")
            # hold 2 s, then confirm
            page.mouse.down(); page.wait_for_selector("#fgconf.on", timeout=5000); page.mouse.up()
            assert not lg.armed
            page.click("#fgyes")
            page.wait_for_selector("#fgbadge.on", timeout=5000)
            page.wait_for_selector("#fgm.ovr", timeout=5000)
            assert lg.armed and lg.slew(box["t"]) is None and "OFF" in page.inner_text("#fgm")
            page.locator("#fgcard").screenshot(path=str(ROOT / "test_frames" / "flashguard_card_override.png"))
            page.click("#fgbadge")
            page.wait_for_function("!document.getElementById('fgbadge').classList.contains('on')", timeout=5000)
            assert not lg.armed
            assert not errs, errs
            br.close()
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()


# ------------------------------------------------------------------------------------------------ the real engine
def _engine(tmp_path, name, *args, driver=None):
    import subprocess
    out = tmp_path / name
    cmd = [sys.executable, str(ROOT / "tests" / "flashguard_engine_driver.py"), *map(str, driver)] if driver else \
          [sys.executable, str(ROOT / "live_visuals.py")]
    r = subprocess.run([*cmd, "--test", "300", "--out", str(out), *map(str, args)], capture_output=True, text=True,
                       timeout=300, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("[flashguard] events=")]
    stats = {k: float(v) for k, v in (kv.split("=") for kv in line[-1].split()[1:])} if line else None
    return out, stats


def test_engine_default_look_is_pixel_identical_with_the_guard_on(tmp_path):
    pytest.importorskip("moderngl")
    plain, none = _engine(tmp_path, "plain")
    guard, stats = _engine(tmp_path, "guard", "--flashguard")
    assert none is None and stats["events"] == 0 and stats["passes"] == 0
    files = sorted(f.name for f in plain.glob("frame_*.png"))
    assert files and all((plain / f).read_bytes() == (guard / f).read_bytes() for f in files)


def test_engine_strobe_and_blackout_hammering_come_out_under_the_limit(tmp_path):
    pytest.importorskip("moderngl")
    _, strobe = _engine(tmp_path, "strobe", "--flashguard", driver=("strobe", STROBE_16))        # 1/16 at 175 BPM
    assert strobe["rate_in"] > 3 and strobe["rate_out"] <= 3 and strobe["events"] == 1 and strobe["passes"] > 200
    bright = ("--set", "symmetry=0", "--set", "l_ai=1", "--set", "density=1", "--set", "l_beads=1", "--set", "l_lines=1")
    _, black = _engine(tmp_path, "blackout", "--flashguard", *bright, driver=("blackout", 6.0))
    assert black["rate_in"] > 3 and black["rate_out"] <= 3 and black["events"] >= 1


# ------------------------------------------------------------------------------------------------ review round (Neon B1-B4)
def test_b1_one_steady_bright_pixel_per_block_does_not_unlock_the_strobe(gl):
    times = clock(3.0)
    dots = np.zeros((H, W), bool)
    dots[::4, ::4] = True                                      # 1 pixel in 16, lit all the time

    def fn(t):
        img = strobe(t, STROBE_16)
        img[dots, :3] = 1.0
        return img
    frames, outs, fg = run(gl, fn, times)
    assert flashes(times, frames)[0] > 3 and flashes(times, outs)[0] <= 3
    late = [o for t, o in zip(times, outs) if t > 1.0]
    assert flashes([t for t in times if t > 1.0], late)[0] == 0
    assert max(float(o[~dots].mean()) for o in late) < 0.2       # the strobe itself is held at its dark phase
    assert all(np.array_equal(o[dots], np.ones_like(o[dots])) for o in late)      # the steady dots are untouched
    never_brighter(frames, outs)
    fg.release()


def test_b2_red_to_orange_flashing_is_held_by_the_red_gate(gl):
    times = clock(3.0)

    def fn(t):
        img = np.zeros((H, W, 4), dtype="f4")
        img[..., 3] = 1.0
        img[..., 0] = 1.0
        img[..., 1] = 0.0 if (t * 6.0) % 1 < 0.5 else 0.55     # saturated red <-> orange (R share 0.79: not "red")
        return img
    frames, outs, fg = run(gl, fn, times)
    assert flashes(times, frames)[1] > 3
    g, r = flashes(times, outs)
    assert r <= 3 and g <= 3
    never_brighter(frames, outs)
    fg.release()


def test_b3_mapped_output_stage_guards_what_the_output_pass_adds(gl):
    """What the surfaces / overlays draw AFTER the picture guard goes through a second guard, then to the projector."""
    size = (W, H)
    target = gl.simple_framebuffer(size)
    fg = FG.FlashGuard(gl, size)
    assert not fg.wants_output_stage(False, False, None) and fg.wants_output_stage(True) and fg.wants_output_stage(False, True)

    class Ov:
        _active, _pending = False, None
    amap = type("A", (), {"overlay": Ov()})()
    assert not fg.wants_output_stage(False, False, amap)
    amap.overlay._pending = (None,)
    assert fg.wants_output_stage(False, False, amap)
    times = clock(3.0)
    for name, level in (("flash", lambda t: 0.9 if (t * 8.0) % 1 < 0.5 else 0.1), ("calm", lambda t: 0.4)):
        fg = FG.FlashGuard(gl, size)
        ins, outs = [], []
        for t in times:
            v = level(t)
            fbo = fg.output_begin(size)                        # the engine draws surfaces / grid / patterns here
            fbo.clear(v, v, v, 1.0)
            ins.append(np.full((H, W, 3), round(v * 255) / 255, "f4"))
            fg.output_end(target, size, t)
            assert gl.fbo is target
            outs.append(np.frombuffer(target.read(components=3, dtype="f4"), dtype="f4").reshape(H, W, 3).copy())
        if name == "flash":
            assert flashes(times, ins)[0] > 3 and flashes(times, outs)[0] <= 3
            assert fg.state()["mode"] == "attenuating" and fg.state()["mapped"] and fg.summary()["map_events"] == 1
        else:
            assert all(np.array_equal(a, b) for a, b in zip(ins, outs)) and fg.summary()["map_passes"] == 0
        never_brighter(ins, outs)
        fg.release()


def test_b4_override_only_from_the_laptop_and_panic_brings_the_guard_back(gl):
    lg = FG.GuardLogic()
    assert lg.command({"cmd": "flashguard", "op": "arm_begin"}, wall=0.0) == "refused_remote"      # phone / LAN client
    assert lg.command({"cmd": "flashguard", "op": "arm_confirm"}, wall=2.5) == "refused_remote" and not lg.armed
    assert lg.state(0.0, wall=3.0)["refused"] and not lg.state(0.0, wall=20.0)["refused"]
    lg.command(ARM("arm_begin"), wall=10.0)                    # a LAN confirm cannot finish a local hold either
    assert lg.command({"op": "arm_confirm"}, wall=12.5) == "refused_remote" and not lg.armed
    assert lg.command(ARM("arm_confirm"), wall=13.0) == "refused"
    forged = FG.validate_command({"cmd": "flashguard", "op": "arm_begin", "_local": True, "local": 1})
    assert forged == {"cmd": "flashguard", "op": "arm_begin"}  # nothing a client sends can carry the marker
    assert FG.mark_local({"cmd": "rec", "op": "start"}) == {"cmd": "rec", "op": "start"}
    src = (ROOT / "remote_control.py").read_text(encoding="utf-8")
    i = src.index("if flashguard.is_local_request(self.client_address[0]")
    assert src.index("cmds = [validate_command(c) for c in cmds]") < i < src.index("self.server.rc._q.put_nowait(c)")
    assert "0.0.0.0" not in FG.LOOPBACK and "127.0.0.1" in FG.LOOPBACK

    fg = FG.FlashGuard(gl, (W, H))
    fg.note_panics(0)
    fg.command(ARM("arm_begin"))
    fg.logic.arm_t0 -= 3.0
    assert fg.command(ARM("arm_confirm")) == "armed" and fg.logic.armed
    fg.note_panics(0)
    assert fg.logic.armed
    fg.note_panics(1)                                          # PANIC: guarded again, at once
    assert not fg.logic.armed and fg.state()["mode"] != "override"
    assert fg.command({"cmd": "flashguard", "op": "disarm"}) == "disarmed"     # switching it back ON works from anywhere
    fg.release()


def test_b3_engine_multi_surface_output_goes_through_the_second_guard(tmp_path):
    pytest.importorskip("moderngl")
    surf = tmp_path / "surfaces.json"
    quad = lambda x0, x1: [[x0, 80], [x1, 80], [x1, 720], [x0, 720]]                       # noqa: E731
    mk = lambda i, x0, x1, b: dict(id=i, name=f"S{i}", order=i - 1, enabled=True, shape="quad", corners=quad(x0, x1),   # noqa: E731
                                   source=dict(mode="full", rect=[0, 0, 1, 1]), content="main", color=[1, 1, 1],
                                   opacity=1.0, brightness=b, glow=0.0, feather=0)
    surf.write_text(json.dumps(dict(version=1, output=[1280, 800], sel=1, surfaces=[mk(1, 40, 620, 2.0), mk(2, 660, 1240, 1.0)])))
    plain, none = _engine(tmp_path, "m_plain", "--surfaces", surf)
    guard, stats = _engine(tmp_path, "m_guard", "--surfaces", surf, "--flashguard")
    assert none is None and stats["map_frames"] == 300 and stats["map_passes"] == 0 and stats["passes"] == 0
    files = sorted(f.name for f in plain.glob("frame_*.png"))
    assert files and all((plain / f).read_bytes() == (guard / f).read_bytes() for f in files)     # calm: bit-identical
    _, st = _engine(tmp_path, "m_strobe", "--surfaces", surf, "--flashguard", driver=("strobe", STROBE_16))
    assert st["rate_in"] > 3 and st["rate_out"] <= 3 and st["map_rate_out"] <= 3


def test_b4_arming_over_http_needs_loopback_a_loopback_host_and_a_same_site_post(tmp_path):
    assert FG.is_local_request("127.0.0.1", "127.0.0.1:8765") and FG.is_local_request("::1", "[::1]:8765")
    assert FG.is_local_request("127.0.0.1", "localhost:8765")
    for addr, host in (("192.168.1.20", "192.168.1.5:8765"), ("192.168.1.20", "127.0.0.1:8765"), ("127.0.0.1", "192.168.1.5:8765"),
                       ("127.0.0.1", "evil.example:8765"), ("127.0.0.1", "127.0.0.1.evil.example"), ("127.0.0.1", None), (None, "localhost")):
        assert not FG.is_local_request(addr, host), (addr, host)

    import http.client
    import socket
    from remote_control import RemoteControl
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()

    def post(op, **headers):
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        con.putrequest("POST", "/cmd", skip_host=True)
        body = json.dumps({"cmd": "flashguard", "op": op}).encode()
        base = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "X-Token": "tok", "Content-Length": str(len(body))}
        for k, v in {**base, **headers}.items():
            con.putheader(k, v)
        con.endheaders(body)
        code = con.getresponse().status
        con.close()
        return code, [c for c in rc.drain() if c.get("cmd") == "flashguard"]
    try:
        lg = FG.GuardLogic()
        code, cmds = post("arm_begin", Origin=f"http://127.0.0.1:{port}")           # the panel on the laptop itself
        assert code == 200 and len(cmds) == 1 and lg.command(cmds[0], wall=0.0) == "pending"
        code, cmds = post("arm_begin", Host=f"192.168.1.5:{port}")                  # asked for the LAN address
        assert code == 200 and lg.command(cmds[0], wall=1.0) == "refused_remote"
        code, cmds = post("arm_confirm", Host=f"evil.example:{port}", Origin=f"http://evil.example:{port}")   # DNS rebinding
        assert code == 403 and cmds == []
        code, cmds = post("arm_confirm", Origin="http://evil.example")              # another site's page, on this laptop
        assert code == 403 and cmds == []
        code, cmds = post("arm_confirm", **{"Sec-Fetch-Site": "cross-site"})
        assert code == 403 and cmds == []
        code, cmds = post("arm_begin", **{"X-Token": "wrong"})
        assert code in (401, 403) and cmds == [] and not lg.armed
    finally:
        rc.stop()
