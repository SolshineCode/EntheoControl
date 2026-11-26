"""Show recorder (recorder.py): fake ffmpeg, fake frames, fake clock; no GL, no sound card, no real encoder."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recorder as rec  # noqa: E402

W, H = 64, 40
FB = W * H * 4


# ----------------------------------------------------------------------------- fakes
class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeStdin:
    def __init__(self, proc):
        self.proc, self.closed = proc, False

    def write(self, b):
        p = self.proc
        if p.dead or self.closed:
            raise BrokenPipeError("ffmpeg gone")
        if p.slow:
            time.sleep(p.slow)
        p.bytes += len(b)
        p.frames = p.bytes // p.frame_bytes if p.frame_bytes else 0
        if p.die_after is not None and p.frames >= p.die_after:
            p.dead, p.returncode = True, 1
        return len(b)

    def flush(self):
        pass

    def close(self):
        self.closed = True
        p = self.proc
        if not p.dead:
            Path(p.out).write_bytes(b"\0" * 64)
            p.returncode = 0


class FakePopen:
    """Encoder (reads '-i -' rawvideo) or mux (writes the output file at once)."""
    procs = []
    die_after = None
    slow = 0.0
    mux_fail = False

    def __init__(self, cmd, stdin=None, stdout=None, stderr=None, **kw):
        self.cmd, self.out = list(cmd), cmd[-1]
        self.bytes = self.frames = 0
        self.dead, self.returncode = False, None
        self.die_after, self.slow = FakePopen.die_after, FakePopen.slow
        self.encoder = "-" in cmd and cmd[cmd.index("-") - 1] == "-i"
        if self.encoder:
            s = cmd[cmd.index("-s") + 1]
            w, h = map(int, s.split("x"))
            self.frame_bytes = w * h * 4
            self.stdin = FakeStdin(self)
        else:
            self.frame_bytes = 0
            self.stdin = None
            if FakePopen.mux_fail:
                self.returncode = 1
            else:
                Path(self.out).write_bytes(b"\0" * 128)
                self.returncode = 0
        FakePopen.procs.append(self)

    def poll(self):
        return self.returncode if (self.dead or self.returncode is not None) else None

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.dead, self.returncode = True, -9


@pytest.fixture(autouse=True)
def _reset():
    FakePopen.procs, FakePopen.die_after, FakePopen.slow, FakePopen.mux_fail = [], None, 0.0, False
    yield


class FakeCapture:
    """Stands in for the PBO ring: grab() remembers the frame, collect() hands it back one 'frame' later."""

    def __init__(self):
        self.pending, self.grabs = [], 0

    def grab(self, reader, t, tag=None):
        data, size = reader
        self.pending.append((t, data, size, tag))
        self.grabs += 1

    def collect(self, force=False):
        out, self.pending = self.pending, []
        return out

    def release(self):
        pass


class FakeAudio:
    """An audio source the recorder subscribes to; the test pushes blocks by hand."""

    def __init__(self, mode):
        self.mode, self.listeners, self.closed = mode, [], False

    def add(self, fn):
        self.listeners.append(fn)

    def remove(self, fn):
        if fn in self.listeners:
            self.listeners.remove(fn)

    def push(self, block, t):
        for fn in list(self.listeners):
            fn(block, t)

    def close(self):
        self.closed = True


def make(tmp_path, clock=None, **kw):
    clock = clock or Clock()
    audios = []

    def audio_factory(mode, engine_audio=None):
        a = FakeAudio(mode)
        audios.append(a)
        return a

    r = rec.Recorder(tmp_path / "recordings", ffmpeg="ffmpeg", popen=FakePopen, clock=clock, capture=FakeCapture(),
                     encoders=["libx264"], audio_factory=audio_factory, persist=False, start_async=False,
                     program_size=(W, H), **kw)
    return r, clock, audios


def frame_bytes(v=0):
    return bytes([v % 256]) * FB


def run_frames(r, clock, seconds, render_fps=60.0, jitter=0.0, rng=None):
    n = int(seconds * render_fps)
    for i in range(n):
        clock.t += 1.0 / render_fps + (rng.uniform(-jitter, jitter) if rng is not None else 0.0)
        r.frame({"program": (frame_bytes(i), (W, H)), "projector": (frame_bytes(i), (W, H))})


def wait_idle(r, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if not r.busy():
            return
        time.sleep(0.01)
    raise AssertionError("recorder never went idle")


# ----------------------------------------------------------------------------- CFR scheduling
def test_cfr_drops_extra_render_frames_at_30fps():
    c = rec.CFR(30)
    c.start(10.0)
    got = [s for i in range(1, 121) if (s := c.take(10.0 + i / 60.0 + 0.001)) is not None]   # 2 s of 60 fps render
    assert got == list(range(1, 61))                                                      # one slot per 1/30 s


def test_cfr_slots_follow_wall_clock_through_stalls():
    c = rec.CFR(30)
    c.start(0.0)
    assert c.take(0.0) == 0
    assert c.take(1.0) == 30                      # a 1 s stall: the writer fills 1..29 with duplicates
    assert c.take(1.01) is None                   # same slot again: dropped
    assert c.take(1.034) == 31


def test_cfr_60fps_stable_under_vsync_jitter():
    rng = np.random.default_rng(1)
    c = rec.CFR(60)
    c.start(0.0)
    slots = [c.take(i / 60.0 + rng.uniform(-0.003, 0.003)) for i in range(1, 600)]
    assert None not in slots and slots == sorted(slots) and len(set(slots)) == len(slots)


# ----------------------------------------------------------------------------- video writer
class SinkProc:
    def __init__(self, slow=0.0, die_after=None):
        self.frames, self.slow, self.die_after, self.closed = [], slow, die_after, False
        self.stdin = self

    def write(self, b):
        if self.die_after is not None and len(self.frames) >= self.die_after:
            raise BrokenPipeError("dead")
        if self.slow:
            time.sleep(self.slow)
        self.frames.append(b[0])
        return len(b)

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_writer_drops_instead_of_blocking_when_encoder_is_slow():
    p = SinkProc(slow=0.05)
    w = rec.VideoWriter(p, FB, maxq=3)
    w.start()
    t = time.perf_counter()
    accepted = sum(w.offer(i, frame_bytes(i)) for i in range(40))
    assert time.perf_counter() - t < 0.2          # offer() never waits on the encoder
    assert accepted < 40 and w.dropped == 40 - accepted
    w.finish(40)
    w.join(10)
    assert len(p.frames) == 40                    # CFR held: the gaps were filled with duplicates
    assert w.dups >= w.dropped - 1


def test_writer_fills_gaps_with_previous_frame_and_pads_to_end():
    p = SinkProc()
    w = rec.VideoWriter(p, FB, maxq=8)
    w.start()
    w.offer(0, frame_bytes(1)); w.offer(3, frame_bytes(2)); w.offer(2, frame_bytes(9))   # 2 arrives late: skipped
    w.finish(6)
    w.join(5)
    assert p.frames == [1, 1, 1, 2, 2, 2] and p.closed


def test_writer_reports_encoder_death():
    p = SinkProc(die_after=5)
    w = rec.VideoWriter(p, FB, maxq=8)
    w.start()
    for i in range(10):
        w.offer(i, frame_bytes(i))
        time.sleep(0.005)
    w.join(5)
    assert w.error and "dead" in w.error


# ----------------------------------------------------------------------------- audio sync
def test_audio_sync_holds_av_sync_over_10_minutes_with_drift_jitter_and_a_gap():
    sr, fps = 48000, 30
    sink = rec.CountSink()
    t0 = 0.0
    a = rec.AudioSync(t0, sink, sr=sr)
    rng = np.random.default_rng(7)
    dev_rate = sr * (1 + 150e-6)                  # the sound card clock runs 150 ppm fast vs perf_counter
    blk = 512
    produced, worst, t = 0, 0.0, 0.0
    while t < 600.0:
        produced += blk
        t = produced / dev_rate + 0.004 + rng.uniform(-0.003, 0.003)    # capture latency + scheduling jitter
        if 300.0 < t < 302.5:                     # HDMI audio device disappears for 2.5 s: no blocks at all
            a.tick(t)
            continue
        a.push(np.full((blk, 2), 0.1, dtype="f4"), t)
        a.tick(t)
        if t > 5 and int(t * 10) % 50 == 0:
            worst = max(worst, abs(a.pos / sr - t))
    assert worst < 1.0 / fps                      # never more than one video frame out
    n_total = int(round(600.0 * fps)) * sr // fps
    a.finish(n_total)
    assert sink.n == n_total
    assert sink.silence >= int(2.0 * sr)          # the gap became silence, not a time shift


def test_audio_sync_trims_audio_from_before_the_start():
    sink = rec.CountSink()
    a = rec.AudioSync(10.0, sink, sr=48000)
    a.push(np.ones((4800, 2), dtype="f4"), 10.05)        # covers 9.95 .. 10.05: only the last 50 ms counts
    assert abs(sink.n - 2400) <= 1


def test_audio_sync_fills_silence_when_no_device_at_all():
    sink = rec.CountSink()
    a = rec.AudioSync(0.0, sink, sr=48000)
    for i in range(1, 101):
        a.tick(i * 0.1)                           # 10 s, nothing ever arrives
    assert abs(a.pos / 48000 - 10.0) < 0.35      # the filler trails real time by 0.05..0.3 s; finish() pads
    a.finish(480000)
    assert sink.n == 480000 and sink.silence == 480000


def test_ring_sink_returns_window_and_zero_fills_the_missing():
    r = rec.RingSink(seconds=1.0, sr=100, ch=2)
    r.write(np.ones((250, 2), dtype="f4"))
    out = r.read(100, 260)
    assert out.shape == (160, 2)
    assert (out[:50] == 0).all() and (out[50:150] == 1).all() and (out[150:] == 0).all()   # ring holds 150..250
    assert (r.read(0, 50) == 0).all()             # older than the ring: zeros


# ----------------------------------------------------------------------------- recorder end to end (fake ffmpeg)
def test_record_start_stop_finalizes_mp4(tmp_path):
    r, clock, audios = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "fps": 30})
    r.command({"cmd": "rec", "op": "start"})
    assert r.active
    run_frames(r, clock, 2.0)
    r.command({"cmd": "rec", "op": "stop"})
    assert not r.active
    wait_idle(r)
    mp4s = list((tmp_path / "recordings").glob("show_*.mp4"))
    assert len(mp4s) == 1
    enc = [p for p in FakePopen.procs if p.encoder][0]
    assert enc.frames == pytest.approx(60, abs=2)
    assert "rawvideo" in enc.cmd and "-" in enc.cmd
    st = r.state()
    assert st["recent"] and st["recent"][0]["name"] == mp4s[0].name
    assert st["recent"][0]["secs"] == pytest.approx(2.0, abs=0.1)
    assert not (tmp_path / "recordings" / ".parts").exists() or not any((tmp_path / "recordings" / ".parts").iterdir())


def test_audio_toggle_selects_source_and_muxes_audio(tmp_path):
    r, clock, audios = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "laptop"})
    r.command({"cmd": "rec", "op": "toggle"})
    assert audios and audios[-1].mode == "laptop" and audios[-1].listeners
    for i in range(100):                          # ~1 s of 60 fps render with loopback blocks
        clock.t += 1 / 60
        audios[-1].push(np.full((800, 2), 0.2, dtype="f4"), clock.t)
        r.frame({"program": (frame_bytes(i), (W, H))})
    r.command({"cmd": "rec", "op": "toggle"})
    wait_idle(r)
    assert audios[-1].closed and not audios[-1].listeners
    mux = [p for p in FakePopen.procs if not p.encoder]
    assert mux and "s16le" in mux[-1].cmd and "aac" in mux[-1].cmd
    # off -> no audio input in the mux
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 0.5)
    r.command({"cmd": "rec", "op": "stop"})
    wait_idle(r)
    assert "s16le" not in [p for p in FakePopen.procs if not p.encoder][-1].cmd
    # mic mode asks for the mic
    r.command({"cmd": "rec", "op": "set", "audio": "mic"})
    r.command({"cmd": "rec", "op": "start"})
    assert audios[-1].mode == "mic"
    r.close(wait=True)


def test_ffmpeg_death_stops_recording_without_raising(tmp_path):
    FakePopen.die_after = 10
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    for _ in range(5):                            # the render loop keeps going no matter what
        run_frames(r, clock, 0.5)
        time.sleep(0.02)
    wait_idle(r)
    assert not r.active
    st = r.state()
    assert st["notice"] and st["notice"]["level"] == "error"
    r.close(wait=True)


def test_close_finalizes_running_recording(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 1.0)
    r.close(wait=True)                            # show exit / Shift+Esc / crash unwinding
    assert list((tmp_path / "recordings").glob("show_*.mp4"))


def test_mux_failure_keeps_the_video(tmp_path):
    FakePopen.mux_fail = True
    r, clock, audios = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "laptop"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 0.5)
    r.command({"cmd": "rec", "op": "stop"})
    wait_idle(r)
    assert list((tmp_path / "recordings").glob("show_*.mp4"))     # the bare video part survives as the output
    assert r.state()["notice"]["level"] in ("warn", "error")


def test_source_and_resolution_choice(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "source": "projector", "res": "1080p", "fps": 60})
    r.command({"cmd": "rec", "op": "start"})
    r.frame({"program": (frame_bytes(1), (W, H)), "projector": (frame_bytes(2), (W, H))})
    clock.t += 1 / 60
    r.frame({"program": (frame_bytes(1), (W, H)), "projector": (frame_bytes(2), (W, H))})
    r.close(wait=True)
    enc = [p for p in FakePopen.procs if p.encoder][0]
    vf = enc.cmd[enc.cmd.index("-vf") + 1]
    assert "vflip" in vf and "1920" in vf and "pad" in vf
    assert enc.cmd[enc.cmd.index("-framerate") + 1] == "60"


def test_projector_size_change_mid_recording_duplicates_instead_of_crashing(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "source": "projector"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 0.5)
    for _ in range(20):
        clock.t += 1 / 60
        r.frame({"projector": (b"\0" * (32 * 20 * 4), (32, 20))})
    r.command({"cmd": "rec", "op": "stop"})
    wait_idle(r)
    assert list((tmp_path / "recordings").glob("show_*.mp4"))


def test_disk_space_warn_and_stop(tmp_path):
    free = {"v": 10 * 2 ** 30}
    r, clock, _ = make(tmp_path, disk_free=lambda p: free["v"])
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    free["v"] = int(1.5 * 2 ** 30)
    run_frames(r, clock, 3.0)
    assert r.active and r.state()["notice"]["level"] == "warn"
    free["v"] = 400 * 2 ** 20
    run_frames(r, clock, 3.0)
    assert not r.active and r.state()["notice"]["level"] == "error"
    wait_idle(r)
    r.command({"cmd": "rec", "op": "start"})      # refused while the disk is this full
    assert not r.active
    r.close(wait=True)


def test_governor_hooks(tmp_path):
    r, clock, _ = make(tmp_path)
    assert r.active is False and r.cost()["active"] is False
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 0.5)
    c = r.cost()
    assert c["active"] is True and c["encoders"] == 1 and "readback_ms" in c
    r.close(wait=True)


def test_idle_recorder_never_grabs_frames(tmp_path):
    r, clock, _ = make(tmp_path)
    run_frames(r, clock, 1.0)
    assert r.capture.grabs == 0


def test_validate_cmd():
    assert rec.validate_cmd({"cmd": "rec", "op": "toggle"}) == {"cmd": "rec", "type": "rec", "op": "toggle"}
    c = rec.validate_cmd({"cmd": "rec", "op": "set", "fps": 60, "source": "projector", "audio": "laptop",
                          "res": "1080p", "quality": "high", "replay": True, "replay_s": 9999, "junk": 1})
    assert c["fps"] == 60 and c["replay_s"] == rec.REPLAY_MAX and "junk" not in c
    for bad in ({"cmd": "rec", "op": "explode"}, {"cmd": "rec", "op": "set", "fps": 25},
                {"cmd": "rec", "op": "set", "audio": "everything"}, {"cmd": "rec"}):
        with pytest.raises(ValueError):
            rec.validate_cmd(bad)


def test_settings_persist(tmp_path):
    clock = Clock()
    kw = dict(ffmpeg="ffmpeg", popen=FakePopen, clock=clock, capture=FakeCapture(), encoders=["libx264"],
              audio_factory=lambda m, engine_audio=None: FakeAudio(m), start_async=False, program_size=(W, H))
    r = rec.Recorder(tmp_path / "recordings", persist=True, **kw)
    r.command({"cmd": "rec", "op": "set", "fps": 60, "audio": "mic"})
    r.close(wait=True)
    r2 = rec.Recorder(tmp_path / "recordings", persist=True, **kw)
    assert r2.settings["fps"] == 60 and r2.settings["audio"] == "mic"
    r2.close(wait=True)


def test_recover_orphaned_parts_after_crash(tmp_path):
    parts = tmp_path / "recordings" / ".parts" / "show_20260101_000000"
    parts.mkdir(parents=True)
    (parts / "video.mp4").write_bytes(b"\0" * 100)
    (parts / "audio.pcm").write_bytes(b"\0" * 4000)
    (parts / "meta.json").write_text(json.dumps(dict(name="show_20260101_000000.mp4", audio=True, fps=30)))
    r, clock, _ = make(tmp_path)
    r.recover(wait=True)
    assert (tmp_path / "recordings" / "show_20260101_000000.mp4").exists()
    assert not parts.exists()
    r.close(wait=True)


# ----------------------------------------------------------------------------- replay buffer
def test_replay_buffer_saves_last_seconds(tmp_path):
    r, clock, audios = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "replay": True, "replay_s": 10})
    assert r.replay_active
    enc = [p for p in FakePopen.procs if p.encoder][-1]
    assert "segment" in enc.cmd
    seg_dir = Path(enc.cmd[-1]).parent
    run_frames(r, clock, 1.0)
    # fake ffmpeg: pretend six 2-second segments were completed
    lines = []
    for i in range(6):
        (seg_dir / f"seg{i:03d}.ts").write_bytes(b"\0" * 10)
        lines.append(f"seg{i:03d}.ts,{i * 2:.6f},{i * 2 + 2:.6f}")
    (seg_dir / "list.csv").write_text("\n".join(lines) + "\n")
    r.command({"cmd": "rec", "op": "save_replay"})
    wait_idle(r)
    outs = list((tmp_path / "recordings").glob("replay_*.mp4"))
    assert len(outs) == 1
    mux = [p for p in FakePopen.procs if not p.encoder][-1]
    assert "concat" in mux.cmd and "copy" in mux.cmd
    r.command({"cmd": "rec", "op": "set", "replay": False})
    assert not r.replay_active
    r.close(wait=True)


def test_encoder_args_fallbacks():
    nv = rec.encoder_args("h264_nvenc", "standard", 30)
    x = rec.encoder_args("libx264", "standard", 30)
    mf = rec.encoder_args("h264_mf", "high", 60)
    assert "h264_nvenc" in nv and "-cq" in nv
    assert "libx264" in x and "veryfast" in x and "-crf" in x
    assert "h264_mf" in mf and "-b:v" in mf
    assert rec.letterbox((1280, 800), (1920, 1080)) == (1728, 1080)
    assert rec.letterbox((1920, 1080), (1920, 1080)) == (1920, 1080)


# ----------------------------------------------------------------------------- hooks in control.py / remote_control.py
def test_midi_learnable_record_toggle(tmp_path):
    from control import Controller
    c = Controller(tmp_path, midi=False, ai=False, restore=False, seed=1)
    c.midi.clear()
    c.apply([{"cmd": "midi_learn", "target": "rec"}])
    c.midi.inject("note_on", num=60, val=100)                 # learning press binds
    c.update(1 / 60, None)
    c.apply([])
    c.midi.inject("note_on", num=60, val=100)
    c.update(1 / 60, None)
    out = c.apply([])
    assert {"cmd": "rec", "type": "rec", "op": "toggle"} in out
    c.close()


def test_remote_passes_rec_commands_through_validated():
    import remote_control
    assert remote_control.validate_command({"cmd": "rec", "op": "start"})["op"] == "start"
    assert remote_control.validate_command({"type": "rec", "op": "set", "fps": "60"})["fps"] == 60
    with pytest.raises(ValueError):
        remote_control.validate_command({"cmd": "rec", "op": "rm -rf"})


def shot(page, name):
    """REC_SHOTS=1 saves the panel screenshots to test_frames/recorder/ for a look."""
    import os
    if os.environ.get("REC_SHOTS"):
        out = ROOT / "test_frames" / "recorder"
        out.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out / name))


# ----------------------------------------------------------------------------- panel: REC button + RECORD card (headless browser)
def test_panel_rec_button_and_card_drive_the_recorder(tmp_path):
    pw_api = pytest.importorskip("playwright.sync_api")
    import socket
    import control
    from remote_control import RemoteControl
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    r, clock, audios = make(tmp_path)
    stop = threading.Event()

    def loop():
        i = 0
        while not stop.is_set():
            for c in ctl.apply(rc.drain()):
                if c.get("cmd") == "rec":
                    r.command(c)
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            clock.t += 1 / 60
            i += 1
            r.frame({"program": (frame_bytes(i), (W, H))})
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60, rec=r.state()))
            time.sleep(1 / 60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    errors = []
    try:
        with pw_api.sync_playwright() as pw:
            br = None
            for channel in ("msedge", "chrome", None):
                try:
                    br = pw.chromium.launch(channel=channel, headless=True) if channel else pw.chromium.launch(headless=True)
                    break
                except Exception:
                    continue
            if br is None:
                pytest.skip("no Chromium-family browser available")
            page = br.new_page(viewport=dict(width=1600, height=1000))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
            page.click('#tabs button[data-tab="midi"]')
            page.wait_for_selector(".recc", timeout=5000)
            page.click(".recc .seg button[data-v='off']")             # audio OFF
            page.wait_for_function("document.querySelector('.recc .seg button[data-v=\"off\"]').classList.contains('on')")
            page.click("#recbtn")
            page.wait_for_function("document.querySelector('#recbtn').classList.contains('on')", timeout=5000)
            assert r.active and r.settings["audio"] == "off"
            page.wait_for_timeout(600)
            assert ":" in page.inner_text("#recinfo")                 # elapsed time shown
            shot(page, "panel_rec_on.png")
            page.click("#recbtn")
            page.wait_for_function("!document.querySelector('#recbtn').classList.contains('on')", timeout=5000)
            page.wait_for_function("document.querySelector('.recc table').innerText.indexOf('show_') >= 0", timeout=8000)
            page.wait_for_timeout(700)
            shot(page, "panel_rec_card.png")
            for w in (390,):                                         # phone width: no horizontal page scroll
                page.set_viewport_size(dict(width=w, height=844))
                page.wait_for_timeout(200)
                shot(page, "panel_rec_phone.png")
                assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
            br.close()
    finally:
        stop.set()
        th.join(2)
        r.close(wait=True)
        ctl.close()
        rc.stop()
    assert not errors, errors
    assert list((tmp_path / "recordings").glob("show_*.mp4"))
