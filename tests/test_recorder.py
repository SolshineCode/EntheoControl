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
                     encoders=["h264_nvenc"], audio_factory=audio_factory, persist=False, start_async=False,
                     spawn=lambda parts: rec.finalize_parts(parts, "ffmpeg", FakePopen),
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
    r.close()


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
    r.close()


def test_close_finalizes_running_recording(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 1.0)
    r.close()                            # show exit / Shift+Esc / crash unwinding
    assert list((tmp_path / "recordings").glob("show_*.mp4"))


def test_mux_failure_keeps_video_and_audio_and_recover_finishes_it(tmp_path):
    """Opus H3: a failed mux must not move the video out and leave audio.pcm for recover() to delete."""
    FakePopen.mux_fail = True
    r, clock, audios = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "laptop"})
    r.command({"cmd": "rec", "op": "start"})
    for i in range(30):
        clock.t += 1 / 60
        audios[-1].push(np.full((800, 2), 0.2, dtype="f4"), clock.t)
        r.frame({"program": (frame_bytes(i), (W, H))})
    r.command({"cmd": "rec", "op": "stop"})
    wait_idle(r)
    rd = tmp_path / "recordings"
    assert not list(rd.glob("show_*.mp4"))
    parts = list((rd / ".parts").iterdir())
    assert len(parts) == 1 and (parts[0] / "video.mp4").exists() and (parts[0] / "audio.pcm").stat().st_size > 0
    assert r.state()["notice"]["level"] in ("warn", "error") and "kept" in r.state()["notice"]["msg"]
    meta = json.loads((parts[0] / "meta.json").read_text())
    assert meta["sealed"] and meta["frames"] > 0 and meta["mux_failed"] == 1
    FakePopen.mux_fail = False                    # next start: the mux works again
    r.recover(wait=True)
    assert len(list(rd.glob("show_*.mp4"))) == 1 and not parts[0].exists()
    mux = [p for p in FakePopen.procs if not p.encoder][-1]
    assert "s16le" in mux.cmd                     # with the audio
    r.close()


def test_finalize_needs_room_for_the_mux_copy(tmp_path):
    parts = tmp_path / "recordings" / ".parts" / "show_x"
    parts.mkdir(parents=True)
    (parts / "video.mp4").write_bytes(b"\0" * 1000)
    (parts / "audio.pcm").write_bytes(b"\0" * 4000)
    (parts / "meta.json").write_text(json.dumps(dict(name="show_x.mp4", audio=True, fps=30, frames=30, sealed=True)))
    st, msg, _ = rec.finalize_parts(parts, "ffmpeg", FakePopen, disk_free=lambda p: 100 * 2 ** 20)
    assert st == "nospace" and parts.exists() and not FakePopen.procs
    st, msg, secs = rec.finalize_parts(parts, "ffmpeg", FakePopen, disk_free=lambda p: 10 * 2 ** 30)
    assert st == "ok" and secs == pytest.approx(1.0) and not parts.exists()
    assert (tmp_path / "recordings" / "show_x.mp4").exists()


def test_recover_never_deletes_audio_without_video(tmp_path):
    parts = tmp_path / "recordings" / ".parts" / "show_y"
    parts.mkdir(parents=True)
    (parts / "audio.pcm").write_bytes(b"\1" * 4000)
    r, clock, _ = make(tmp_path)
    r.recover(wait=True)
    assert (parts / "audio.pcm").exists()
    r.close()


def test_finalize_skips_a_dir_another_process_is_finishing(tmp_path):
    parts = tmp_path / "recordings" / ".parts" / "show_z"
    parts.mkdir(parents=True)
    (parts / "video.mp4").write_bytes(b"\0" * 100)
    (parts / rec.LOCK_NAME).write_text("1234")
    st, _, _ = rec.finalize_parts(parts, "ffmpeg", FakePopen)
    assert st == "busy" and parts.exists()


def test_disk_stop_counts_the_mux_copy(tmp_path):
    free = {"v": 10 * 2 ** 30}
    r, clock, _ = make(tmp_path, disk_free=lambda p: free["v"])
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 0.5)
    r.rec.part_bytes = lambda: 3 * 2 ** 30          # a 3 GB recording so far
    free["v"] = int(3.2 * 2 ** 30)                 # above 500 MB, but the stop-time mux would not fit
    run_frames(r, clock, 1.5)
    assert not r.active and r.state()["notice"]["level"] == "error"
    r.close()


# ----------------------------------------------------------------------------- exit / crash path is bounded (Opus H2)
class HangMux(FakePopen):
    """Mux that runs until killed (an hour-long AAC encode)."""

    def __init__(self, cmd, **kw):
        super().__init__(cmd, **kw)
        self.killed = threading.Event()
        if not self.encoder:
            self.returncode = None

    def wait(self, timeout=None):
        if self.encoder:
            return super().wait(timeout)
        self.killed.wait(timeout if timeout is not None else 60)
        self.returncode = -9 if self.killed.is_set() else 0
        return self.returncode

    def kill(self):
        self.killed.set()
        super().kill()


def test_crash_exit_is_fast_and_leaves_parts_for_recovery(tmp_path):
    FakePopen.slow = 0.4                           # an encoder that has fallen behind
    spawned = []
    r, clock, _ = make(tmp_path)
    r.spawn = spawned.append
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 1.0)
    t = time.perf_counter()
    r.close(crash=True, budget=2.0)
    assert time.perf_counter() - t < 4.0
    assert not spawned                             # crash: the next start's recover() finishes it
    parts = list((tmp_path / "recordings" / ".parts").iterdir())
    assert len(parts) == 1 and (parts[0] / "video.mp4").exists() or (parts[0] / "meta.json").exists()


def test_clean_exit_hands_off_to_a_detached_finisher_fast(tmp_path):
    spawned = []
    r, clock, _ = make(tmp_path)
    r.spawn = spawned.append
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 1.0)
    t = time.perf_counter()
    r.close()
    assert time.perf_counter() - t < 3.0
    assert len(spawned) == 1 and (spawned[0] / "meta.json").exists()
    meta = json.loads((spawned[0] / "meta.json").read_text())
    assert meta["sealed"] and meta["frames"] == pytest.approx(30, abs=2)
    st, _, _ = rec.finalize_parts(spawned[0], "ffmpeg", FakePopen)     # what the detached process does
    assert st == "ok" and list((tmp_path / "recordings").glob("show_*.mp4"))


def test_exit_during_a_long_mux_kills_it_and_hands_off(tmp_path):
    spawned = []
    clock = Clock()
    r = rec.Recorder(tmp_path / "recordings", ffmpeg="ffmpeg", popen=HangMux, clock=clock, capture=FakeCapture(),
                     encoders=["h264_nvenc"], audio_factory=lambda m, engine_audio=None: FakeAudio(m), persist=False,
                     start_async=False, program_size=(W, H), spawn=spawned.append)
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 1.0)
    r.command({"cmd": "rec", "op": "stop"})
    end = time.time() + 5
    while time.time() < end and not r.finishing[0].mux_procs:
        time.sleep(0.01)
    assert r.finishing[0].mux_procs                # the stop-time mux is running (and would run for an hour)
    t = time.perf_counter()
    r.close()
    assert time.perf_counter() - t < 3.0
    assert len(spawned) == 1 and (spawned[0] / "video.mp4").exists()


def test_shutdown_never_waits_on_a_hung_starting_probe(tmp_path):
    r, clock, _ = make(tmp_path)
    r.bg.append(threading.Thread(target=time.sleep, args=(3,), daemon=True))
    r.bg[-1].start()
    t = time.perf_counter()
    r.close(budget=1.0)
    assert time.perf_counter() - t < 2.0


# ----------------------------------------------------------------------------- replay restart race (Opus M3)
def test_replay_restart_uses_its_own_folder(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "replay": True, "replay_s": 10})
    old = r.replay
    run_frames(r, clock, 0.3)
    r.command({"cmd": "rec", "op": "set", "quality": "high"})      # restarts the ring
    new = r.replay
    assert new is not old and new.parts != old.parts
    (new.parts / "seg000.ts").write_bytes(b"x")
    wait_idle(r)                                   # the old ring's finalize has run
    assert not old.parts.exists() and (new.parts / "seg000.ts").exists()
    r.close()


# ----------------------------------------------------------------------------- long pause, 0x0 frames (Opus M8, H1, M1, M7)
def test_long_pause_is_filled_in_real_time_not_in_a_burst():
    p = SinkProc()
    now = [0.0]
    w = rec.VideoWriter(p, FB, maxq=8, due=lambda: int(now[0] * 30) - 15)
    w.start()
    w.offer(0, frame_bytes(1))
    time.sleep(0.2)
    now[0] = 300.0                                 # five minutes with the projector unplugged
    time.sleep(0.5)
    before = len(p.frames)
    assert before >= 300 * 30 - 15 - 1            # kept up while nothing arrived
    w.offer(9001, frame_bytes(2))                  # resume
    w.finish(9002)
    w.join(10)
    assert len(p.frames) == 9002 and len(p.frames) - before < 40      # no burst on resume


def test_zero_size_reader_is_ignored(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "source": "projector"})
    r.command({"cmd": "rec", "op": "start"})
    for _ in range(10):
        clock.t += 1 / 30
        r.frame({"projector": (b"", (0, 0))})      # window minimised
    assert r.capture.grabs == 0
    r.close()


class FakeBuf:
    made = released = 0

    def __init__(self, n):
        if n <= 0:
            raise ValueError("buffer cannot be empty")
        self.size = n
        FakeBuf.made += 1

    def read(self):
        return b"\0" * self.size

    def release(self):
        FakeBuf.released += 1


class FakeCtx:
    def buffer(self, reserve, dynamic=True):
        return FakeBuf(reserve)


class FakeFbo:
    def read_into(self, buf, **kw):
        pass


def test_glcapture_two_sources_no_churn_and_zero_size_safe():
    FakeBuf.made = FakeBuf.released = 0
    cap = rec.GLCapture(FakeCtx())
    fbo = (FakeFbo(), 0)
    for i in range(200):                           # program 64x40 + projector 32x20 every frame
        cap.collect()
        cap.grab((fbo, (64, 40)), i, "a")
        cap.grab((fbo, (32, 20)), i, "b")
        assert cap.grab((fbo, (0, 0)), i, "c") is False      # minimised: refused, nothing leaks
        assert cap.count <= cap.pool
    assert FakeBuf.made <= cap.pool and FakeBuf.released == 0
    assert len(cap.collect(force=True)) > 0


def test_glcapture_exhausted_pool_with_nothing_pending_does_not_raise():
    cap = rec.GLCapture(FakeCtx(), pool=1)
    cap.count = 1                                  # e.g. after a failed allocation elsewhere
    assert cap.grab(((FakeFbo(), 0), (8, 8)), 0.0) in (True, False)


# ----------------------------------------------------------------------------- MIDI CC learned to rec (Opus M2)
def test_midi_cc_fader_on_rec_toggles_once_per_crossing_and_note_off_never(tmp_path):
    from control import Controller
    c = Controller(tmp_path, midi=False, ai=False, restore=False, seed=1)
    c.midi.clear()
    c.apply([{"cmd": "midi_learn", "target": "rec"}])
    c.midi.inject("cc", ch=0, num=21, val=10)      # learning message binds
    c.update(1 / 60, None)
    c.apply([])

    def sweep(vals):
        for v in vals:
            c.midi.inject("cc", ch=0, num=21, val=v)
        c.update(1 / 60, None)
        return [x for x in c.apply([]) if x.get("cmd") == "rec"]

    assert len(sweep(range(0, 128, 3))) == 1       # a fader pushed all the way up: one toggle, not ~21
    assert len(sweep([127] * 20)) == 0             # held at the top / repeats: nothing
    assert len(sweep(range(127, -1, -5))) == 0     # pulled down: nothing
    assert len(sweep([0, 100, 0, 100])) == 2       # two real presses
    c.apply([{"cmd": "midi_learn", "target": "rec"}])
    c.midi.inject("note_on", num=60, val=100)
    c.update(1 / 60, None)
    c.apply([])
    c.midi.inject("note_off", num=60)
    c.update(1 / 60, None)
    assert not [x for x in c.apply([]) if x.get("cmd") == "rec"]
    c.close()


def test_midi_cc_on_bypass_uses_the_same_edge_helper(tmp_path):
    from control import Controller, BYPASS_COVER
    key = sorted(BYPASS_COVER)[0]
    c = Controller(tmp_path, midi=False, ai=False, restore=False, seed=1)
    c.midi.clear()
    c.apply([{"cmd": "midi_learn", "target": f"bypass:{key}"}])
    c.midi.inject("cc", ch=0, num=22, val=0)
    c.update(1 / 60, None)
    before = key in c.state().get("bypass", [])
    for v in range(0, 128, 2):
        c.midi.inject("cc", ch=0, num=22, val=v)
    c.update(1 / 60, None)
    after = key in c.state().get("bypass", [])
    assert after != before                         # toggled exactly once (an even number would put it back)
    c.close()


def test_toggle_debounce(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "toggle"})
    clock.t += 0.1
    r.command({"cmd": "rec", "op": "toggle"})      # a bounce: ignored
    assert r.active
    clock.t += 1.0
    r.command({"cmd": "rec", "op": "toggle"})
    assert not r.active
    r.close()


def test_rec_value_types():
    with pytest.raises(ValueError):
        rec.validate_cmd({"cmd": "rec", "op": "set", "replay": "false"})     # a string is not a bool (L8)
    for bad in ({"fps": 60.5}, {"fps": None}, {"quality": True}, {"source": 1}, {"replay_s": float("nan")},
                {"replay_s": "abc"}, {"audio": None}):
        with pytest.raises(ValueError):
            rec.validate_cmd(dict({"cmd": "rec", "op": "set"}, **bad))
    assert rec.validate_cmd({"cmd": "rec", "op": "set", "replay": False})["replay"] is False
    assert rec.validate_cmd({"type": "rec", "op": "stop"})["op"] == "stop"


def test_ctrl_r_is_reset_while_calibrating_and_record_otherwise():
    """Engine key handler (live_visuals.on_key): Ctrl+R while calibrating stays 'reset this surface'; only outside
    calibration does it toggle the recorder. A GL-free check of the branch order."""
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    reset = src.index('elif key == glfw.KEY_R and st["calib"]')
    record = src.index('elif key == glfw.KEY_R and recorder is not None')
    assert reset < record and "and not st[\"calib\"]" not in src[reset:record].split("\n")[0]


# ----------------------------------------------------------------------------- encoder choice (Opus M4, M5)
def test_cpu_encoder_is_announced_loudly(tmp_path):
    r, clock, _ = make(tmp_path)
    r.encoders = ["libx264"]
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    st = r.state()
    assert st["notice"]["level"] == "warn" and "CPU" in st["notice"]["msg"] and st["cpu_encode"]
    assert r.cost()["cpu_encode"] is True
    r.close()


def test_probe_does_not_cache_a_result_without_nvenc(monkeypatch):
    calls = []

    class P:
        def __init__(self, cmd, **kw):
            calls.append(cmd[cmd.index("-c:v") + 1])
            self.enc = calls[-1]

        def wait(self, timeout=None):
            return 1 if self.enc == "h264_nvenc" else 0

    monkeypatch.setattr(rec, "_probe_cache", {})
    assert rec.probe_encoders("ffx", P) == ["libx264", "h264_mf"]
    assert rec.probe_encoders("ffx", P) == ["libx264", "h264_mf"]
    assert calls.count("h264_nvenc") == 2          # probed again, not stuck on the CPU for the whole show


def test_nvenc_takes_rgb0_without_cpu_conversion():
    nv = rec.encoder_args("h264_nvenc", "standard", 30)
    assert "-pix_fmt" not in nv
    assert "-pix_fmt" in rec.encoder_args("libx264", "standard", 30)


def test_drift_steering_resamples_without_a_splice():
    x = np.sin(np.linspace(0, 20 * np.pi, 512)).astype("f4")
    a = np.stack([x, x], 1)
    for n in (496, 528):
        b = rec.resample(a, n)
        assert b.shape == (n, 2)
        assert np.abs(np.diff(b[:, 0])).max() < np.abs(np.diff(x)).max() * 1.2      # no jump


def test_source_and_resolution_choice(tmp_path):
    r, clock, _ = make(tmp_path)
    r.command({"cmd": "rec", "op": "set", "audio": "off", "source": "projector", "res": "1080p", "fps": 60})
    r.command({"cmd": "rec", "op": "start"})
    r.frame({"program": (frame_bytes(1), (W, H)), "projector": (frame_bytes(2), (W, H))})
    clock.t += 1 / 60
    r.frame({"program": (frame_bytes(1), (W, H)), "projector": (frame_bytes(2), (W, H))})
    r.close()
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
    r.close()


def test_governor_hooks(tmp_path):
    r, clock, _ = make(tmp_path)
    assert r.active is False and r.cost()["active"] is False
    r.command({"cmd": "rec", "op": "set", "audio": "off"})
    r.command({"cmd": "rec", "op": "start"})
    run_frames(r, clock, 0.5)
    c = r.cost()
    assert c["active"] is True and c["encoders"] == 1 and "readback_ms" in c
    r.close()


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
    r.close()
    r2 = rec.Recorder(tmp_path / "recordings", persist=True, **kw)
    assert r2.settings["fps"] == 60 and r2.settings["audio"] == "mic"
    r2.close()


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
    r.close()


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
    r.close()


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
            page.wait_for_timeout(1000)                              # past the 0.5 s toggle debounce
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
        r.close()
        ctl.close()
        rc.stop()
    assert not errors, errors
    assert list((tmp_path / "recordings").glob("show_*.mp4"))
