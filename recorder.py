"""Show recorder: the projection (clean program feed or the mapped projector frame) to MP4, optionally with the audio
that is playing on the laptop (WASAPI loopback) or the mic.

Pipeline (the render thread never waits on the encoder):
  render thread   Recorder.frame(readers) once per frame. Picks a constant-frame-rate slot for the frame from a
                  monotonic clock (CFR, 30 or 60 fps), starts an ASYNC readback into a pixel-buffer ring (GLCapture,
                  4 PBOs) and maps the PBO it started two frames ago (the GPU is done with it, so no stall).
  writer thread   VideoWriter: bounded queue (10 frames) -> raw RGB0 frames into ffmpeg's stdin. A full queue drops
                  the frame (counted) instead of blocking; gaps in the slot sequence are filled with the previous
                  frame, so the file stays exactly CFR and its length follows wall-clock time.
  audio thread    AudioSync places each captured block by its arrival time: soft drift correction (a few samples at
                  a time) against the same monotonic clock, silence for device gaps (projector HDMI audio appearing /
                  disappearing), so audio sample N is always within a few ms of video frame N * 48000 / fps.
  finalize        stop -> pad video + audio to the same length -> mux (ffmpeg -c:v copy, audio -> AAC 192k,
                  +faststart) -> recordings/show_YYYYmmdd_HHMMSS.mp4. Runs on its own thread.

Why a temp PCM file muxed at stop instead of a second live ffmpeg input (named pipe)? On Windows a named pipe as a
second ffmpeg input deadlocks easily (ffmpeg reads inputs in lockstep, so a stalled audio pipe stalls the video pipe
and vice versa) and needs pywin32 for the pipe server. A raw PCM file on disk cannot block anything, survives a
crash, and the stop-time mux is a stream copy (seconds even for an hour). The video part is fragmented MP4, so even
a hard kill leaves a playable file; the next start finalizes orphaned parts (Recorder.recover).

Audio: "laptop" taps the engine's own loopback capture (live_visuals.Audio.taps) when the engine listens to loopback,
so there is only one loopback stream; otherwise (engine on mic / none / headless) it opens its own soundcard
loopback recorder. "mic" likewise. Device changes reopen the stream; the gap becomes silence.

Replay buffer (default off): a second encoder writes 2 s MPEG-TS segments in a wrap-around set plus a RAM ring of
audio; "save replay" concatenates the last N seconds into recordings/replay_*.mp4 (copy, no re-encode).

Commands: {"cmd": "rec", "op": "toggle" | "start" | "stop" | "set" | "save_replay" | "open_folder", ...settings}
  set: source program|projector, fps 30|60, res native|1080p, audio off|laptop|mic, quality draft|standard|high,
       replay bool, replay_s 5..120
Governor: Recorder.active (bool) and Recorder.cost() -> {active, encoders, readback_ms, dropped, fps, res}.
"""
import collections
import json
import re
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SR, CH = 48000, 2
SOURCES = ("program", "projector")
FPS_CHOICES = (30, 60)
RES_CHOICES = ("native", "1080p")
AUDIO_MODES = ("off", "laptop", "mic")
QUALITIES = ("draft", "standard", "high")
REPLAY_MIN, REPLAY_MAX, SEG = 5, 120, 2
WARN_FREE, STOP_FREE = 2 * 2 ** 30, 500 * 2 ** 20
TOGGLE_DEBOUNCE = 0.5                                   # s: toggles closer than this are one bounce
OPS = ("toggle", "start", "stop", "set", "save_replay", "open_folder")
DEFAULTS = dict(source="program", fps=30, res="native", audio="laptop", quality="standard", replay=False, replay_s=30)
ENCODER_ORDER = ("h264_nvenc", "libx264", "h264_mf")
_WIN = sys.platform == "win32"
_FLAGS = (0x08000000 | 0x00004000) if _WIN else 0      # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS


# ----------------------------------------------------------------------------- command validation (server thread)
def validate_cmd(c):
    """Normalized copy of a {"cmd": "rec"} command; ValueError if malformed."""
    if not isinstance(c, dict) or c.get("cmd", c.get("type")) != "rec":
        raise ValueError("not a rec command")
    op = c.get("op")
    if op not in OPS:
        raise ValueError("unknown rec op")
    out = {"cmd": "rec", "type": "rec", "op": op}
    if op != "set":
        return out
    for key, allowed in (("source", SOURCES), ("res", RES_CHOICES), ("audio", AUDIO_MODES), ("quality", QUALITIES)):
        if key in c:
            if c[key] not in allowed:
                raise ValueError(f"bad {key}")
            out[key] = c[key]
    if "fps" in c:
        try:
            if isinstance(c["fps"], bool):
                raise TypeError
            f = float(c["fps"])
            fps = int(f)
            if fps != f:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError("bad fps")
        if fps not in FPS_CHOICES:
            raise ValueError("bad fps")
        out["fps"] = fps
    if "replay" in c:
        if not isinstance(c["replay"], bool):                # "false" must not read as True
            raise ValueError("bad replay")
        out["replay"] = c["replay"]
    if "replay_s" in c:
        try:
            v = float(c["replay_s"])
        except (TypeError, ValueError):
            raise ValueError("bad replay_s")
        if not math.isfinite(v):
            raise ValueError("bad replay_s")
        out["replay_s"] = int(min(max(v, REPLAY_MIN), REPLAY_MAX))
    return out


# ----------------------------------------------------------------------------- ffmpeg helpers
def find_ffmpeg():
    """System ffmpeg (full build, NVENC) first, else the one bundled with imageio-ffmpeg."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    here = Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
    found = sorted(here.glob("ffmpeg*.exe")) if here.is_dir() else []
    return str(found[-1]) if found else None


_probe_cache = {}
_probe_lock = threading.Lock()


def probe_encoders(ffmpeg, popen=subprocess.Popen, order=ENCODER_ORDER):
    """The H.264 encoders that really work here (NVENC can be listed yet fail: old driver, sessions used up).
    One probe at a time (a REC press during the start-up probe waits for it instead of probing NVENC twice), and
    only a result that includes NVENC is cached: a transient NVENC failure is probed again at the next start."""
    if not ffmpeg:
        return []
    with _probe_lock:
        return _probe_locked(ffmpeg, popen, order)


def _probe_locked(ffmpeg, popen, order):
    if ffmpeg in _probe_cache:
        return _probe_cache[ffmpeg]
    ok = []
    for enc in order:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x160:r=30:d=0.2",
               "-c:v", enc] + (["-pix_fmt", "yuv420p"] if enc == "libx264" else []) + ["-f", "null", "-"]
        try:
            p = popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      creationflags=_FLAGS)
            if p.wait(timeout=15) == 0:
                ok.append(enc)
        except Exception:
            pass
    if "h264_nvenc" in ok:
        _probe_cache[ffmpeg] = ok
    return ok


def encoder_args(enc, quality, fps, gop=None):
    q = QUALITIES.index(quality) if quality in QUALITIES else 1
    g = str(int(gop or fps * 2))
    if enc == "h264_nvenc":
        cq = ("28", "23", "19")[q]
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0",
                "-maxrate", "60M", "-bufsize", "120M", "-g", g, "-bf", "0"]    # rgb0 in: NVENC converts on the GPU
    if enc == "h264_mf":
        br = ("6M", "12M", "24M")[q]
        return ["-c:v", "h264_mf", "-b:v", br, "-g", g, "-pix_fmt", "nv12"]
    crf = ("26", "21", "18")[q]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-g", g, "-pix_fmt", "yuv420p"]


def letterbox(src, dst):
    """Largest even size of src's aspect that fits dst."""
    (w, h), (W, H) = src, dst
    s = min(W / w, H / h)
    return int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2


def video_filter(size, res):
    vf = "vflip"                                         # glReadPixels rows are bottom-up
    if res == "1080p":
        sw, sh = letterbox(size, (1920, 1080))
        vf += f",scale={sw}:{sh}:flags=bicubic,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"
    return vf + ",setsar=1"


def encode_cmd(ffmpeg, size, fps, enc, quality, res, out, segment=None):
    w, h = size
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostats", "-y", "-f", "rawvideo", "-pix_fmt", "rgb0",
           "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", "-vf", video_filter(size, res)]
    if segment:
        wrap = segment
        cmd += encoder_args(enc, quality, fps, gop=fps * SEG)
        cmd += ["-force_key_frames", f"expr:gte(t,n_forced*{SEG})", "-f", "segment", "-segment_time", str(SEG),
                "-segment_wrap", str(wrap), "-segment_list", str(Path(out).parent / "list.csv"),
                "-segment_list_type", "csv", "-segment_list_size", str(wrap), "-reset_timestamps", "1",
                "-segment_format", "mpegts", str(out)]
    else:
        cmd += encoder_args(enc, quality, fps) + ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(out)]
    return cmd


def mux_cmd(ffmpeg, video, pcm, out, pad=False, concat=False):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    cmd += (["-f", "concat", "-safe", "0", "-i", str(video)] if concat else ["-i", str(video)])
    if pcm:
        cmd += ["-f", "s16le", "-ar", str(SR), "-ac", str(CH), "-i", str(pcm), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        if pad:
            cmd += ["-af", "apad", "-shortest"]
    else:
        cmd += ["-map", "0:v:0", "-c:v", "copy"]
    return cmd + ["-movflags", "+faststart", str(out)]


# ----------------------------------------------------------------------------- CFR slot clock
class CFR:
    """Maps frame times (monotonic seconds) to constant-frame-rate slots: slot = nearest multiple of 1/fps.
    A slot already taken is a drop; skipped slots become duplicates in the writer."""

    def __init__(self, fps):
        self.fps, self.t0, self.last = int(fps), None, -1

    def start(self, t0):
        self.t0, self.last = t0, -1

    def slot(self, t):
        return int(math.floor((t - self.t0) * self.fps + 0.5))

    def take(self, t):
        if self.t0 is None:
            return None
        s = self.slot(t)
        if s <= self.last or s < 0:
            return None
        self.last = s
        return s


# ----------------------------------------------------------------------------- video writer thread
class VideoWriter(threading.Thread):
    """Frames -> encoder stdin. offer() never blocks; finish(n) pads to exactly n frames and closes stdin.

    due(): optional, the slot that should already be on disk by now (wall clock minus a lag). While no frames arrive
    (projector unplugged, window minimised, a long hitch) the writer keeps the file growing in real time with the
    last frame, so a long pause never turns into a burst of thousands of duplicates on resume. abort(): stop at once,
    no padding (crash / exit path); the fragmented MP4 written so far stays playable."""

    def __init__(self, proc, frame_bytes, maxq=10, due=None):
        super().__init__(daemon=True, name="rec-writer")
        self.proc, self.frame_bytes, self.due = proc, frame_bytes, due
        self._abort = False
        self.q = queue.Queue(maxq)
        self.next_slot, self.last = 0, None
        self.dropped = self.dups = self.written = 0
        self.error = None
        self.total = None
        self._finish = threading.Event()
        self.closed = False

    def offer(self, slot, data):
        if self.error or self._finish.is_set():
            return False
        try:
            self.q.put_nowait((slot, data))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def finish(self, total):
        self.total = total
        self._finish.set()

    def abort(self):
        self._abort = True
        self._finish.set()

    def _fill_to(self, slot):
        while self.next_slot < slot and not self._abort:
            self._write(self.last)
            self.dups += 1
            self.next_slot += 1

    def _write(self, data):
        self.proc.stdin.write(data)
        self.written += 1

    def run(self):
        try:
            while True:
                if self._abort:
                    break
                try:
                    slot, data = self.q.get(timeout=0.05)
                except queue.Empty:
                    if self._finish.is_set():
                        break
                    if self.due is not None and self.last is not None:
                        self._fill_to(self.due())       # nothing arriving: keep real time with the last frame
                    continue
                if slot < self.next_slot:
                    continue
                if self.last is None:
                    self.last = data
                self._fill_to(slot)
                if self._abort:
                    break
                self._write(data)
                self.last, self.next_slot = data, slot + 1
            if self.total is not None and self.last is not None and not self._abort:
                self._fill_to(self.total)
        except Exception as ex:                          # BrokenPipe / OSError / ValueError: the encoder died
            self.error = f"{type(ex).__name__}: {ex}"
            while True:                                  # release the queued frames
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.closed = True


# ----------------------------------------------------------------------------- audio
class CountSink:
    """Test sink: counts samples (and how many were silence)."""

    def __init__(self):
        self.n = self.silence = 0

    def write(self, a):
        self.n += len(a)

    def write_silence(self, n):
        self.n += n
        self.silence += n

    def truncate(self, n):
        self.n = min(self.n, n)


class FileSink:
    """48 kHz stereo s16le raw PCM on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "wb")
        self.n = 0

    def write(self, a):
        self.f.write(np.clip(a * 32767.0, -32768, 32767).astype("<i2").tobytes())
        self.n += len(a)

    def write_silence(self, n):
        while n > 0:
            k = min(n, SR)
            self.f.write(b"\0" * (k * CH * 2))
            self.n += k
            n -= k

    def truncate(self, n):
        if n < self.n:
            self.f.flush()
            self.f.truncate(n * CH * 2)
            self.f.seek(n * CH * 2)
            self.n = n

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class RingSink:
    """The last `seconds` of audio in RAM, addressed by absolute sample index (replay buffer)."""

    def __init__(self, seconds, sr=SR, ch=CH):
        self.cap = int(seconds * sr)
        self.buf = np.zeros((self.cap, ch), dtype="f4")
        self.n = 0
        self.lock = threading.Lock()

    def write(self, a):
        with self.lock:
            total = len(a)
            a = a[-self.cap:]
            k, i = len(a), (self.n + total - len(a)) % self.cap
            first = min(k, self.cap - i)
            self.buf[i:i + first] = a[:first]
            if k > first:
                self.buf[:k - first] = a[first:]
            self.n += total

    def write_silence(self, n):
        while n > 0:
            k = min(n, self.cap)
            self.write(np.zeros((k, self.buf.shape[1]), dtype="f4"))
            n -= k

    def truncate(self, n):
        pass

    def read(self, start, end):
        out = np.zeros((max(0, end - start), self.buf.shape[1]), dtype="f4")
        with self.lock:
            lo, hi = max(start, self.n - self.cap, 0), min(end, self.n)
            if hi > lo:
                idx = np.arange(lo, hi) % self.cap
                out[lo - start:hi - start] = self.buf[idx]
        return out


def resample(a, n):
    """Linear-interpolation resample of one block to n samples: drift steering without a splice (no click on tonal
    material; a 3 % pitch bend for 10 ms is inaudible)."""
    m = len(a)
    if n == m or m < 2 or n < 1:
        return a
    x = np.linspace(0.0, m - 1, n)
    return np.stack([np.interp(x, np.arange(m), a[:, c]) for c in range(a.shape[1])], axis=1).astype("f4")


def as_stereo(block):
    a = np.asarray(block, dtype="f4")
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    elif a.shape[1] > 2:
        a = a[:, :2]
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


class AudioSync:
    """Places captured audio blocks on the recording's timeline (sample 0 = t0 on the monotonic clock).

    d = (t_arrival - t0) * sr - (pos + n) is how far the stream would be behind real time after this block. Delays
    only ever add to d, so the minimum of d over a one-second window is the honest clock error; it is steered to zero
    a few samples per block (device clock drift, ~100 ppm, is corrected long before it adds up to a frame). The first
    block (and the first one after a fill) is aligned exactly. A device gap (d above `gap` s, or nothing at all for
    `gap` s, seen by tick) becomes silence."""

    def __init__(self, t0, sink, sr=SR, gap=0.25, soft=0.002, step=16):
        self.t0, self.sink, self.sr = t0, sink, sr
        self.gap, self.soft, self.step = gap * sr, soft * sr, step
        self.pos, self.align = 0, True
        self.m_cur, self.m_win, self.win_t = math.inf, None, None
        self.silence = 0

    def _silence(self, n):
        n = int(n)
        if n > 0:
            self.sink.write_silence(n)
            self.pos += n
            self.silence += n

    def push(self, block, t):
        a = as_stereo(block)
        n = len(a)
        if n == 0:
            return
        d = (t - self.t0) * self.sr - (self.pos + n)
        if self.align or d > self.gap:
            if d >= 0:
                self._silence(round(d))
            else:
                k = int(round(-d))
                if k >= n:
                    return                                 # all of it is from before the start / already covered
                a = a[k:]
            self.align = False
            self.m_cur, self.m_win, self.win_t = math.inf, None, t
        else:
            self.m_cur = min(self.m_cur, d)
            if self.win_t is None:
                self.win_t = t
            if t - self.win_t >= 1.0:
                self.m_win, self.m_cur, self.win_t = self.m_cur, math.inf, t
            if self.m_win is not None and abs(self.m_win) > self.soft:
                k = int(min(self.step, abs(self.m_win), max(1, len(a) // 8)))
                if self.m_win > 0:                         # behind: stretch this block by k samples
                    a = resample(a, len(a) + k)
                    self.m_win -= k
                    self.m_cur -= k
                else:                                      # ahead (device clock fast): squeeze it by k
                    a = resample(a, len(a) - k)
                    self.m_win += k
                    self.m_cur += k
        if len(a):
            self.sink.write(a)
            self.pos += len(a)

    def tick(self, now):
        """No audio at all for a while (device unplugged, nothing captured): keep the timeline moving with silence."""
        behind = (now - self.t0) * self.sr - self.pos
        if behind > self.gap * 1.2:
            self._silence(behind - 0.1 * self.sr)
            self.align = True

    def finish(self, n_total):
        if self.pos < n_total:
            self._silence(n_total - self.pos)
        elif self.pos > n_total:
            self.sink.truncate(n_total)
            self.pos = n_total


class EngineTap:
    """Subscribe to the engine's own capture thread (live_visuals.Audio.taps): one loopback stream for both."""

    def __init__(self, engine_audio):
        self.engine, self.fns = engine_audio, []

    def add(self, fn):
        self.fns.append(fn)
        self.engine.taps.append(fn)

    def remove(self, fn):
        for lst in (self.fns, self.engine.taps):
            try:
                lst.remove(fn)
            except ValueError:
                pass

    def close(self):
        for fn in list(self.fns):
            self.remove(fn)


class SoundcardSource:
    """Own capture thread (loopback of the default output, or the default mic). Reopens on errors and when the
    Windows default device changes (projector HDMI audio appearing / disappearing); gaps become silence downstream."""

    def __init__(self, kind, log=print):
        self.kind, self.log, self.fns, self.running = kind, log, [], True
        self.device = None
        self.th = threading.Thread(target=self._run, daemon=True, name=f"rec-audio-{kind}")
        self.th.start()

    def add(self, fn):
        self.fns.append(fn)

    def remove(self, fn):
        if fn in self.fns:
            self.fns.remove(fn)

    def close(self):
        self.running = False
        self.fns = []

    def _run(self):
        try:
            import warnings
            import soundcard as sc
            warnings.filterwarnings("ignore", module="soundcard")
        except Exception as ex:
            self.log(f"[rec] audio capture unavailable ({ex}); recording silence")
            return
        while self.running:
            try:
                if self.kind == "loopback":
                    spk = sc.default_speaker()
                    dev_id, dev, ch = spk.id, sc.get_microphone(id=str(spk.name), include_loopback=True), 2
                else:
                    mic = sc.default_microphone()
                    dev_id, dev, ch = mic.id, mic, 1
                self.device = dev.name
                self.log(f"[rec] audio {self.kind}: {dev.name}")
                last = time.perf_counter()
                with dev.recorder(samplerate=SR, channels=ch, blocksize=480) as r:
                    while self.running:
                        blk = r.record(numframes=480)
                        t = time.perf_counter()
                        for fn in list(self.fns):
                            fn(blk, t)
                        if t - last > 1.0:
                            last = t
                            try:
                                cur = (sc.default_speaker() if self.kind == "loopback" else sc.default_microphone()).id
                            except Exception:
                                cur = None
                            if cur is not None and cur != dev_id:
                                self.log("[rec] audio device changed -> reopening (gap = silence)")
                                break
            except Exception as ex:
                if self.running:
                    self.log(f"[rec] audio device lost ({ex}); silence until it is back")
                self.device = None
                time.sleep(0.5)


def default_audio_factory(mode, engine_audio=None):
    want = "loopback" if mode == "laptop" else "mic"
    if engine_audio is not None and getattr(engine_audio, "source", None) == want \
            and isinstance(getattr(engine_audio, "taps", None), list) and getattr(engine_audio, "running", False):
        return EngineTap(engine_audio)
    return SoundcardSource(want)


# ----------------------------------------------------------------------------- GL readback ring
class GLCapture:
    """Async readback: fbo.read_into(PBO) returns at once; the PBO is mapped `latency` frames later, when the GPU
    has long finished, so reading it costs a memcpy and no pipeline stall. Free PBOs are kept per size, so two
    sources (or a projector of another size) never release and re-create buffers every frame; the pool covers two
    sources x (latency + 1) grabs in flight."""

    def __init__(self, ctx, pool=6, latency=2):
        self.ctx, self.pool, self.latency = ctx, pool, latency
        self.free, self.pending, self.n = {}, collections.deque(), 0
        self.count = 0
        self._ready = []

    def _pbo(self, nbytes):
        lst = self.free.get(nbytes)
        if lst:
            return lst.pop()
        if self.count >= self.pool:                         # make room: drop a free buffer of another size
            for k, other in self.free.items():
                if other:
                    other.pop().release()
                    self.count -= 1
                    break
        if self.count < self.pool:
            b = self.ctx.buffer(reserve=nbytes, dynamic=True)   # count only what really exists
            self.count += 1
            return b
        return None

    def grab(self, reader, t, tag=None):
        (fbo, att), (w, h) = reader
        nbytes = int(w) * int(h) * 4
        if nbytes <= 0:                                     # minimised window / 0x0 framebuffer: nothing to read
            return False
        b = self._pbo(nbytes)
        if b is None and self.pending:                      # every PBO still in flight: map the oldest early
            self._ready.append(self._map(self.pending.popleft()))
            b = self._pbo(nbytes)
        if b is None:
            return False
        fbo.read_into(b, viewport=(0, 0, w, h), components=4, attachment=att, alignment=1)
        self.pending.append((b, t, (w, h), tag, self.n))
        return True

    def _map(self, item):
        b, t, size, tag, _ = item
        data = b.read()
        self.free.setdefault(b.size, []).append(b)
        return (t, data, size, tag)

    def collect(self, force=False):
        self.n += 1
        out, self._ready = list(self._ready), []
        while self.pending and (force or self.n - self.pending[0][4] >= self.latency):
            out.append(self._map(self.pending.popleft()))
        return out

    def release(self):
        for b in [x for lst in self.free.values() for x in lst] + [p[0] for p in self.pending]:
            try:
                b.release()
            except Exception:
                pass
        self.free, self.pending, self.count = {}, collections.deque(), 0


# ----------------------------------------------------------------------------- parts dir -> finished MP4
LOCK_NAME, LOCK_STALE = "finalizing.lock", 30.0


def index_add(folder, name, secs, s):
    p = Path(folder) / ".index.json"
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        idx = {}
    idx[name] = dict(secs=round(float(secs or 0), 2), fps=s.get("fps"), source=s.get("source"), audio=s.get("audio"))
    try:
        p.write_text(json.dumps(idx), encoding="utf-8")
    except Exception:
        pass


def run_mux(ffmpeg, popen, video, pcm, out, logpath, pad=False, concat=False, procs=None, log=print):
    """ffmpeg mux into <out>.muxing.mp4, then rename. procs: a list the running process is put in (so an exit can
    kill it). False on any failure; never touches the inputs."""
    if not ffmpeg:
        return False
    out = Path(out)
    tmp = out.with_name(out.stem + ".muxing.mp4")
    try:
        with open(logpath, "ab") as lg:
            p = popen(mux_cmd(ffmpeg, video, pcm, tmp, pad=pad, concat=concat), stdin=subprocess.DEVNULL,
                      stdout=subprocess.DEVNULL, stderr=lg, creationflags=_FLAGS)
            if procs is not None:
                procs.append(p)
            try:
                rc = p.wait(timeout=3600)
            finally:
                if procs is not None and p in procs:
                    procs.remove(p)
        if rc == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, out)
            return True
    except Exception as ex:
        log(f"[rec] mux error: {ex}")
    try:
        tmp.unlink()
    except OSError:
        pass
    return False


def _lock_fresh(d):
    try:
        return time.time() - (Path(d) / LOCK_NAME).stat().st_mtime < LOCK_STALE
    except OSError:
        return False


_SAFE_MP4 = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()-]{0,150}[.]mp4$", re.I)


def safe_output_name(name, fallback, folder):
    """meta.json is a file on disk and may be damaged or planted: the output name taken from it must be a plain
    .mp4 file name that lands directly inside `folder` (no path, no drive, no "..", no odd characters). Anything
    else falls back to the parts folder's own name (made safe the same way)."""
    folder = Path(folder)
    for cand in (name, fallback, "recovered.mp4"):
        if not isinstance(cand, str) or not _SAFE_MP4.match(cand) or ".." in cand or Path(cand).name != cand:
            continue
        try:
            if (folder / cand).resolve().parent == folder.resolve():
                return cand
        except OSError:
            continue
    return "recovered.mp4"


def finalize_parts(parts, ffmpeg, popen=subprocess.Popen, disk_free=None, log=print, procs=None, wait_lock=0.0):
    """Turn one sealed (or crash-left) parts dir into recordings/<name>.mp4. Returns (status, message, secs).

    status: "ok" (the MP4 is written, the parts dir removed), "busy" (another process is finalizing it), "nospace"
    (the mux needs a full extra copy: parts kept), "failed" (parts kept, retried at the next start), "empty" (nothing
    worth keeping, dir removed). Audio is never deleted unless it made it into an MP4."""
    parts = Path(parts)
    folder = parts.parent.parent
    end = time.time() + wait_lock
    while _lock_fresh(parts):
        if time.time() >= end:
            return "busy", f"{parts.name} is being finalized by another process", 0.0
        time.sleep(0.5)
    try:
        meta = json.loads((parts / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    for k in ("fps", "frames", "mux_failed"):               # numbers only: a planted string must not crash recovery
        if not isinstance(meta.get(k), (int, float)) or isinstance(meta.get(k), bool) or not meta[k] == meta[k] or meta[k] < 0:
            meta.pop(k, None)
    name = safe_output_name(meta.get("name"), f"{parts.name}.mp4", folder)
    video, pcm = parts / "video.mp4", parts / "audio.pcm"
    vsize = video.stat().st_size if video.exists() else 0
    asize = pcm.stat().st_size if pcm.exists() else 0
    if vsize == 0:
        if asize > 0:
            return "failed", f"{name}: no video, audio kept in {parts}", 0.0
        shutil.rmtree(parts, ignore_errors=True)
        return "empty", f"{name}: nothing recorded", 0.0
    need = vsize + asize // 8 + 200 * 2 ** 20                # the mux writes a full copy of the video (+ AAC)
    try:
        free = (disk_free or (lambda p: shutil.disk_usage(p).free))(folder)
    except Exception:
        free = None
    if free is not None and free < need:
        return "nospace", (f"{name}: not enough disk to finish ({free / 2 ** 30:.1f} GB free, needs "
                           f"{need / 2 ** 30:.1f} GB); parts kept in {parts}, finished at the next start"), 0.0
    lock = parts / LOCK_NAME
    stop = threading.Event()

    def beat():                                             # heartbeat: recover() in another process waits on it
        while not stop.wait(5.0):
            try:
                os.utime(lock)
            except OSError:
                pass
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        fps = meta.get("fps") or 30
        secs = (meta["frames"] / fps) if meta.get("frames") else 0.0
        use_pcm = pcm if meta.get("audio", asize > 0) and asize > 0 else None
        out = folder / name
        ok = run_mux(ffmpeg, popen, video, use_pcm, out, parts / "ffmpeg.log", pad=not meta.get("sealed"),
                     procs=procs, log=log)
        if ok:
            stop.set()
            shutil.rmtree(parts, ignore_errors=True)
            index_add(folder, name, secs, meta)
            return "ok", f"saved {name} ({secs:.0f} s, {out.stat().st_size / 2 ** 20:.0f} MB)", secs
        meta["mux_failed"] = int(meta.get("mux_failed", 0)) + 1
        try:
            (parts / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        except OSError:
            pass
        return "failed", f"{name}: could not write the MP4; parts (video + audio) kept in {parts}, retried at the next start", secs
    finally:
        stop.set()
        try:
            lock.unlink()
        except OSError:
            pass


def spawn_finalizer(parts):
    """Finish a sealed recording in a detached process, so the show window can close at once."""
    flags = (0x00000008 | 0x00000200 | 0x08000000 | 0x00004000) if _WIN else 0  # DETACHED | NEW_GROUP | NO_WINDOW | BELOW_NORMAL
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--finalize", str(parts)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags, close_fds=True, cwd=str(Path(parts).parent))


# ----------------------------------------------------------------------------- one output (file or replay ring)
class Session:
    def __init__(self, rec, settings, kind="file"):
        self.rec, self.kind = rec, kind
        self.s = dict(settings)
        self.fps, self.source = int(self.s["fps"]), self.s["source"]
        self.size = tuple(rec._sizes.get(self.source) or rec.program_size)
        self.cfr = CFR(self.fps)
        self.ready = self.failed = self.stopping = self.done = self.sealed = False
        self.handoff = False                                 # exit: a detached process / the next start finishes it
        self.error = None
        self.writer = self.proc = self.sync = self.audio = self.sink = self.log = None
        self.t0 = self.t_end = None
        self.size_skips = 0
        self.blocks = collections.deque()
        self.threads = []
        self.mux_procs = []
        self.frames = 0
        self._lock = threading.Lock()
        self._late_stop = False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if kind == "file":
            self.name = f"show_{stamp}.mp4"
            n, k = self.name, 1
            while (rec.dir / n).exists() or (rec.dir / ".parts" / Path(n).stem).exists():
                k += 1
                n = f"show_{stamp}_{k}.mp4"
            self.name = n
            self.parts = rec.dir / ".parts" / Path(self.name).stem
            self.video = self.parts / "video.mp4"
        else:                                                # each replay ring has its own folder (M3)
            rec._replay_n += 1
            self.name = "replay"
            self.parts = rec.dir / ".replay" / f"{stamp}_{os.getpid()}_{rec._replay_n}"
            self.video = self.parts / "seg%03d.ts"
        self.pcm = self.parts / "audio.pcm"
        self.with_audio = self.s["audio"] != "off"

    # -- start (render thread, or a starter thread)
    def start(self):
        rec = self.rec
        try:
            self.parts.mkdir(parents=True, exist_ok=True)
            encs = rec.encoders if rec.encoders is not None else probe_encoders(rec.ffmpeg, rec.popen)
            if not rec.ffmpeg or not encs:
                raise RuntimeError("no working ffmpeg H.264 encoder found")
            self.encoder = encs[0]
            wrap = (int(self.s["replay_s"]) // SEG + 4) if self.kind == "replay" else None
            cmd = encode_cmd(rec.ffmpeg, self.size, self.fps, self.encoder, self.s["quality"], self.s["res"],
                             self.video, segment=wrap)
            self.log = open(self.parts / "ffmpeg.log", "ab")
            self.proc = rec.popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
                                  creationflags=_FLAGS)
            if self.kind == "file":
                self._write_meta()
            lag = max(1, int(self.fps * 0.5))
            self.writer = VideoWriter(self.proc, self.size[0] * self.size[1] * 4,
                                      due=lambda: self.cfr.slot(rec.clock()) - lag)
            self.writer.start()
            self.t0 = rec.clock()
            self.cfr.start(self.t0)
            if self.with_audio:
                self.sink = FileSink(self.pcm) if self.kind == "file" else RingSink(int(self.s["replay_s"]) + 3 * SEG + 6)
                self.sync = AudioSync(self.t0, self.sink)
                self.audio = rec.audio_factory(self.s["audio"], engine_audio=rec.engine_audio)
                self.audio.add(self._on_audio)
                th = threading.Thread(target=self._audio_loop, daemon=True, name="rec-audio-sync")
                th.start()
                self.threads.append(th)
            with self._lock:
                self.ready = True
                late = self._late_stop
            if late:
                self._begin_finalize()
        except Exception as ex:
            self.failed, self.error = True, f"could not start: {ex}"
            self.stopping = self.done = True
            self._close_audio()
            try:
                if self.proc is not None:
                    self.proc.kill()
            except Exception:
                pass

    def _write_meta(self, **extra):
        d = dict(name=self.name, audio=self.with_audio, fps=self.fps, source=self.source, res=self.s["res"],
                 encoder=getattr(self, "encoder", None), size=list(self.size), started=time.time())
        d.update(extra)
        try:
            (self.parts / "meta.json").write_text(json.dumps(d), encoding="utf-8")
        except OSError:
            pass

    def _on_audio(self, block, t):                       # capture thread: just queue it
        if not self.stopping:
            self.blocks.append((block, t))

    def _audio_loop(self):
        while True:
            fin = self.stopping
            while self.blocks:
                b, t = self.blocks.popleft()
                if self.t_end is not None and t > self.t_end + 0.5:
                    continue
                try:
                    self.sync.push(b, t)
                except Exception as ex:
                    self.rec._log(f"[rec] audio block dropped: {ex}")
            if fin:
                break
            try:
                self.sync.tick(self.rec.clock())
            except Exception:
                pass
            time.sleep(0.02)

    def _close_audio(self):
        if self.audio is not None:
            try:
                self.audio.remove(self._on_audio)
                self.audio.close()
            except Exception:
                pass

    # -- per frame (render thread)
    def offer(self, slot, data, size):
        if not self.ready or self.stopping:
            return
        if tuple(size) != self.size or len(data) != self.size[0] * self.size[1] * 4:
            self.size_skips += 1                         # projector resized mid-recording: hold the last frame
            if self.size_skips == 1:
                self.rec.notice("frame size changed mid-recording: holding the last frame", "warn")
            return
        self.writer.offer(slot, data)

    def dead(self):
        if not self.ready or self.stopping:
            return None
        if self.writer.error:
            return self.writer.error
        try:
            rc = self.proc.poll()
        except Exception:
            rc = None
        if rc is not None:
            return f"ffmpeg exited ({rc})"
        return None

    def elapsed(self):
        if self.t0 is None:
            return 0.0
        return max(0.0, (self.t_end if self.t_end is not None else self.rec.clock()) - self.t0)

    def part_bytes(self):
        try:
            if self.kind == "file":
                return self.video.stat().st_size + (self.pcm.stat().st_size if self.pcm.exists() else 0)
        except OSError:
            pass
        return 0

    def mb(self):
        try:
            if self.kind == "file":
                return self.video.stat().st_size / 2 ** 20
        except OSError:
            pass
        return 0.0

    # -- stop (render thread) -> seal + publish (own thread)
    def stop(self, error=None):
        with self._lock:
            if self.stopping:
                return
            self.stopping = True
            if error:
                self.error = error
            self.t_end = self.rec.clock()
            if self.failed:
                self.done = True
                return
            if not self.ready:
                self._late_stop = True                     # still starting: start() finalizes when it is up
                return
        self._begin_finalize()

    def _begin_finalize(self):
        t_end = self.t_end if self.t_end is not None else self.rec.clock()
        self.total = max(self.cfr.last + 1, int(round((t_end - self.t0) * self.fps)), 1)
        self.writer.finish(self.total)
        th = threading.Thread(target=self._finalize, daemon=True, name="rec-finalize")
        th.start()
        self.threads.append(th)

    def seal(self, timeout=60.0, abort=False):
        """Close the encoder and the audio file, pad audio to the video, note the frame count in meta.json.
        Bounded: a hung encoder is killed after `timeout`; abort = no padding (exit / crash path)."""
        if self.sealed or self.writer is None:
            return
        if abort:
            self.writer.abort()
        self.writer.join(timeout)
        if self.writer.is_alive():
            self.writer.abort()
            try:
                self.proc.kill()                           # unblocks a write stuck in the pipe
            except Exception:
                pass
            self.writer.join(2.0)
        try:
            self.proc.wait(timeout=max(1.0, timeout / 4))
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.frames = max(self.writer.next_slot, 1)
        for th in self.threads:
            if th is not threading.current_thread() and th.name == "rec-audio-sync":
                th.join(min(10.0, timeout))
        self._close_audio()
        if self.sync is not None:
            try:
                self.sync.finish(self.frames * SR // self.fps)
            except Exception:
                pass
            if isinstance(self.sink, FileSink):
                self.sink.close()
        try:
            self.log.close()
        except Exception:
            pass
        if self.kind == "file":
            self._write_meta(frames=self.frames, sealed=True, dropped=self.writer.dropped)
        self.sealed = True

    def _finalize(self):
        rec = self.rec
        try:
            self.seal(60.0)
            if self.kind == "replay":
                shutil.rmtree(self.parts, ignore_errors=True)    # only this ring's own folder
                return
            if self.handoff:
                return
            status, msg, secs = finalize_parts(self.parts, rec.ffmpeg, rec.popen, rec.disk_free, rec._log,
                                               procs=self.mux_procs)
            if self.handoff and status != "ok":
                return                                         # exit killed the mux: a detached finalizer retries
            rec._recent_t = -1e9
            if status == "ok":
                if self.writer.dropped:
                    msg += f", {self.writer.dropped} frames dropped (encoder behind)"
                if self.error:
                    rec.notice(f"recording stopped: {self.error}; {msg}", "error")
                else:
                    rec.notice(msg, "info")
            else:
                pre = f"recording stopped: {self.error}; " if self.error else ""
                rec.notice(pre + msg, "error" if self.error or status == "nospace" else "warn")
        except Exception as ex:
            rec.notice(f"finalize failed: {ex} (parts kept in {self.parts})", "error")
        finally:
            self.done = True


# ----------------------------------------------------------------------------- the recorder
class Recorder:
    def __init__(self, out_dir, ffmpeg=None, popen=subprocess.Popen, clock=time.perf_counter, capture=None,
                 encoders=None, audio_factory=default_audio_factory, engine_audio=None, persist=True,
                 start_async=True, program_size=(1280, 800), disk_free=None, log=print, spawn=None):
        self.dir = Path(out_dir)
        self.ffmpeg = ffmpeg if ffmpeg is not None else find_ffmpeg()
        self.popen, self.clock, self.capture = popen, clock, capture
        self.encoders = encoders
        self.audio_factory, self.engine_audio = audio_factory, engine_audio
        self.persist, self.start_async = persist, start_async
        self.program_size = tuple(program_size)
        self.disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
        self.spawn = spawn or spawn_finalizer
        self._log = log
        self.settings = dict(DEFAULTS)
        self.rec = self.replay = None
        self.finishing = []
        self.bg = []
        self._notice = None
        self._sizes = {}
        self._disk_t = -1e9
        self._warned = False
        self._recent, self._recent_t = [], -1e9
        self._toggle_t = -1e9
        self._replay_n = 0
        self.readback_ms = 0.0
        self.auto_stop_at = None
        self.dir.mkdir(parents=True, exist_ok=True)
        try:                                                          # recordings never go into version control
            gi = self.dir / ".gitignore"
            if not gi.exists():
                gi.write_text("*\n", encoding="utf-8")
        except OSError:
            pass
        if persist:
            try:
                saved = json.loads((self.dir / ".settings.json").read_text(encoding="utf-8"))
                v = validate_cmd(dict(saved, cmd="rec", op="set"))
                self.settings.update({k: v[k] for k in DEFAULTS if k in v})
            except Exception:
                pass
        if self.encoders is None and self.ffmpeg and start_async:     # probe in the background, ready by first REC
            self._bg(probe_encoders, self.ffmpeg, self.popen)
        if persist and self.settings["replay"]:                       # the replay buffer was left armed
            self._start_replay()

    # -- governor hooks
    @property
    def active(self):
        return self.rec is not None and not self.rec.stopping and not self.rec.failed

    @property
    def replay_active(self):
        return self.replay is not None and not self.replay.stopping and not self.replay.failed

    def cost(self):
        live = [s for s in (self.rec, self.replay) if s is not None and not s.stopping]
        encs = [getattr(s, "encoder", None) for s in live]
        return dict(active=bool(live), encoders=len(live), readback_ms=round(self.readback_ms, 3),
                    dropped=sum(s.writer.dropped for s in live if s.writer is not None),
                    fps=max((s.fps for s in live), default=0), res=self.settings["res"],
                    encoder=encs[0] if encs else None,
                    cpu_encode=any(e is not None and e != "h264_nvenc" for e in encs))

    # -- helpers
    def notice(self, msg, level="info"):
        self._notice = dict(msg=str(msg), level=level, t=time.time())
        self._log(f"[rec] {msg}")

    def _bg(self, fn, *a):
        th = threading.Thread(target=fn, args=a, daemon=True, name="rec-bg")
        th.start()
        self.bg.append(th)
        return th

    def busy(self):
        self.bg = [t for t in self.bg if t.is_alive()]
        self.finishing = [s for s in self.finishing if not s.done]
        starting = any(s is not None and not s.ready and not s.failed and not s.stopping for s in (self.rec, self.replay))
        return bool(self.bg or self.finishing or starting)

    def _save_settings(self):
        if self.persist:
            try:
                (self.dir / ".settings.json").write_text(json.dumps(self.settings), encoding="utf-8")
            except Exception:
                pass

    def _mux(self, video, pcm, out, logpath, pad=False, concat=False):
        return run_mux(self.ffmpeg, self.popen, video, pcm, out, logpath, pad=pad, concat=concat, log=self._log)

    def _index_add(self, name, secs, s):
        index_add(self.dir, name, secs, s)
        self._recent_t = -1e9

    def recent(self, n=8):
        now = time.monotonic()
        if now - self._recent_t < 2.0:
            return self._recent
        try:
            idx = json.loads((self.dir / ".index.json").read_text(encoding="utf-8"))
        except Exception:
            idx = {}
        files = []
        try:
            files = sorted((p for p in self.dir.glob("*.mp4") if not p.name.endswith(".muxing.mp4")),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:n]
        except OSError:
            pass
        out = []
        for p in files:
            try:
                out.append(dict(name=p.name, mb=round(p.stat().st_size / 2 ** 20, 1), secs=(idx.get(p.name) or {}).get("secs")))
            except OSError:
                pass
        self._recent, self._recent_t = out, now
        return out

    # -- control
    def command(self, c):
        try:
            c = validate_cmd(c)
        except ValueError as ex:
            self._log(f"[rec] bad command: {ex}")
            return
        op = c["op"]
        if op == "toggle":
            now = self.clock()
            if now - self._toggle_t < TOGGLE_DEBOUNCE:      # a CC / pad / key bounce must not start + stop + start
                return
            self._toggle_t = now
            self.stop() if self.active else self.start()
        elif op == "start":
            self.start()
        elif op == "stop":
            self.stop()
        elif op == "save_replay":
            self.save_replay()
        elif op == "open_folder":
            if _WIN:
                try:
                    os.startfile(str(self.dir))                # noqa: S606 (explorer on the laptop)
                except Exception as ex:
                    self._log(f"[rec] open folder: {ex}")
        elif op == "set":
            new = {k: v for k, v in c.items() if k in DEFAULTS}
            replay_changed = any(k in new and new[k] != self.settings[k] for k in DEFAULTS)
            self.settings.update(new)
            self._save_settings()
            if self.settings["replay"] and (replay_changed or self.replay is None):
                if self.replay is not None:
                    self._stop_session(self.replay)
                    self.replay = None
                self._start_replay()
            elif not self.settings["replay"] and self.replay is not None:
                self._stop_session(self.replay)
                self.replay = None

    def _free(self):
        try:
            return self.disk_free(self.dir)
        except Exception:
            return None

    def _disk_ok(self):
        free = self._free()
        return free is None or free >= STOP_FREE

    def start(self, seconds=None):
        if self.active:
            return
        if not self._disk_ok():
            self.notice("not recording: less than 500 MB free on the recordings drive", "error")
            return
        self._warned = False
        s = Session(self, self.settings, "file")
        self.rec = s
        self.auto_stop_at = None if not seconds else ("pending", float(seconds))
        if self.start_async:
            self._bg(self._start_session, s)
        else:
            self._start_session(s)

    def _start_session(self, s):
        s.start()
        if s.failed:
            self.notice(f"recording failed to start: {s.error}", "error")
            self.auto_stop_at = None
        elif s.kind == "file":
            extra = f", audio {s.s['audio']}" if s.with_audio else ", no audio"
            if s.encoder != "h264_nvenc":                   # M4: say it loudly, the CPU path heats the laptop
                self.notice(f"recording {s.name} on the CPU ({s.encoder}): NVENC is not available. Expect more heat "
                            f"and dropped frames; DRAFT quality / 30 fps help", "warn")
            else:
                self.notice(f"recording {s.name} ({s.source}, {s.fps} fps, {s.s['res']}, {s.encoder}{extra})", "info")

    def _start_replay(self):
        if not self._disk_ok():
            self.notice("replay buffer off: less than 500 MB free", "error")
            return
        s = Session(self, self.settings, "replay")
        self.replay = s
        if self.start_async:
            self._bg(s.start)
        else:
            s.start()

    def _stop_session(self, s, error=None):
        if s is None:
            return
        s.stop(error=error)
        self.finishing.append(s)

    def stop(self, error=None):
        if self.rec is None or self.rec.stopping:
            return
        self._stop_session(self.rec, error=error)
        self.auto_stop_at = None

    def save_replay(self):
        s = self.replay
        if s is None or not s.ready:
            self.notice("replay buffer is off", "warn")
            return
        self._bg(self._save_replay, s, int(self.settings["replay_s"]))

    def _save_replay(self, s, want):
        try:
            lines = (s.parts / "list.csv").read_text(encoding="utf-8").strip().splitlines()
            segs = []
            for ln in lines:
                parts = ln.split(",")
                if len(parts) >= 3:
                    segs.append((parts[0], float(parts[1]), float(parts[2])))
            if not segs:
                self.notice("replay buffer is still filling", "warn")
                return
            pick, total = [], 0.0
            for seg in reversed(segs):
                pick.insert(0, seg)
                total += seg[2] - seg[1]
                if total >= want:
                    break
            stamp = time.strftime("%Y%m%d_%H%M%S")
            work = self.dir / ".parts" / f"replay_{stamp}"
            work.mkdir(parents=True, exist_ok=True)
            listing = []
            for i, (fn, a, b) in enumerate(pick):
                dst = work / f"part{i:03d}.ts"
                shutil.copyfile(s.parts / fn, dst)
                listing.append(f"file '{dst.as_posix()}'")
            (work / "list.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
            pcm = None
            if s.with_audio and s.sync is not None:
                a0, a1 = int(round(pick[0][1] * SR)), int(round(pick[-1][2] * SR))
                data = s.sink.read(a0, a1)
                pcm = work / "audio.pcm"
                pcm.write_bytes(np.clip(data * 32767.0, -32768, 32767).astype("<i2").tobytes())
            out = self.dir / f"replay_{stamp}.mp4"
            ok = self._mux(work / "list.txt", pcm, out, work / "ffmpeg.log", pad=pcm is not None, concat=True)
            if ok:
                shutil.rmtree(work, ignore_errors=True)
                secs = pick[-1][2] - pick[0][1]
                self._index_add(out.name, secs, s.s)
                self.notice(f"saved replay {out.name} ({secs:.0f} s)", "info")
            else:
                self.notice(f"replay save failed (parts in {work})", "error")
        except Exception as ex:
            self.notice(f"replay save failed: {ex}", "error")

    # -- crash recovery: parts left by a hard kill / an exit -> finished MP4s
    def recover(self, wait=False):
        def work():
            rb = self.dir / ".replay"                         # replay rings left by a hard kill: just space
            if rb.is_dir():
                mine = self.replay.parts if self.replay is not None else None
                for d in rb.iterdir():
                    if d != mine:
                        shutil.rmtree(d, ignore_errors=True)
            base = self.dir / ".parts"
            if not base.is_dir():
                return
            live = {s.parts for s in [self.rec, self.replay] + self.finishing if s is not None and not s.done}
            for d in sorted(base.iterdir()):
                if not d.is_dir() or d in live or d.name.startswith("replay_"):
                    continue
                try:
                    status, msg, _ = finalize_parts(d, self.ffmpeg, self.popen, self.disk_free, self._log)
                    if status == "ok":
                        self.notice(f"recovered an interrupted recording: {msg}", "warn")
                    elif status in ("failed", "nospace"):
                        self.notice(msg, "warn")
                except Exception as ex:
                    self._log(f"[rec] could not recover {d.name}: {ex}")
            self._recent_t = -1e9
        if wait:
            work()
        else:
            self._bg(work)

    # -- the render-thread hook
    def frame(self, readers, t=None):
        """Call once per rendered frame, after the output pass (before swap). readers: {"program": reader,
        "projector": reader}; a reader is whatever the capture object understands plus its (w, h). A reader with a
        0 x 0 size (minimised window) counts as absent: the writer keeps time with the last frame."""
        readers = {k: r for k, r in readers.items()
                   if r is not None and int(r[1][0]) > 0 and int(r[1][1]) > 0}
        for src, r in readers.items():
            self._sizes[src] = tuple(r[1])
        live = [s for s in (self.rec, self.replay) if s is not None and s.ready and not s.stopping]
        cap = self.capture
        if not live and not (cap is not None and getattr(cap, "pending", None)):
            return
        t = self.clock() if t is None else t
        t_a = time.perf_counter()
        if cap is not None:
            for (tc, data, size, tag) in cap.collect():
                for s, slot in (tag or ()):
                    s.offer(slot, data, size)
        wants = {}
        for s in live:
            if readers.get(s.source) is None:
                continue
            slot = s.cfr.take(t)
            if slot is not None:
                wants.setdefault(s.source, []).append((s, slot))
        for src, tag in wants.items():
            if cap is not None:
                cap.grab(readers[src], t, tag)
        self.readback_ms += ((time.perf_counter() - t_a) * 1000.0 - self.readback_ms) * 0.05
        # housekeeping
        for s in live:
            err = s.dead()
            if err:
                self.notice(f"encoder stopped ({err}); recording stopped and saved what was captured", "error")
                if s is self.rec:
                    self.stop(error=err)
                else:
                    self._stop_session(s, error=err)
                    self.replay = None
        if self.auto_stop_at and self.rec is not None and self.rec.ready:
            if self.auto_stop_at[0] == "pending":
                self.auto_stop_at = ("at", self.rec.t0 + self.auto_stop_at[1])
            if t >= self.auto_stop_at[1]:
                self.stop()
        if t - self._disk_t >= 1.0 and (self.active or self.replay_active):
            self._disk_t = t
            free = self._free()
            part = self.rec.part_bytes() if self.active else 0      # H3: the stop-time mux needs a full copy
            if free is not None and free < STOP_FREE + part:
                self.notice(f"disk almost full ({free / 2 ** 20:.0f} MB free, the MP4 needs "
                            f"{part / 2 ** 20:.0f} MB more): recording stopped", "error")
                self.stop(error="disk full")
                if self.replay is not None:
                    self._stop_session(self.replay)
                    self.replay = None
            elif free is not None and free < WARN_FREE + part and not self._warned:
                self._warned = True
                self.notice(f"low disk space: {free / 2 ** 30:.1f} GB free (recording stops at 0.5 GB + the "
                            f"size of the recording)", "warn")

    def auto_pending(self):
        """Headless --record-seconds: True until the timed recording has stopped and been written."""
        return self.auto_stop_at is not None or self.busy()

    def state(self):
        s = self.rec if self.active else None
        st = dict(active=s is not None, starting=bool(s is not None and not s.ready),
                  elapsed=round(s.elapsed(), 1) if s else 0.0, mb=round(s.mb(), 1) if s else 0.0,
                  file=s.name if s else None, encoder=getattr(s, "encoder", None) if s else None,
                  cpu_encode=bool(s and getattr(s, "encoder", None) not in (None, "h264_nvenc")),
                  dropped=s.writer.dropped if s and s.writer else 0,
                  settings=dict(self.settings), folder=str(self.dir.resolve()),
                  replay=dict(on=self.replay_active, ready=bool(self.replay and self.replay.ready),
                              secs=round(min(self.replay.elapsed(), self.settings["replay_s"]), 1) if self.replay_active else 0),
                  saving=bool(self.finishing and any(not x.done and x.kind == "file" for x in self.finishing)),
                  notice=self._notice, recent=self.recent())
        return st

    def close(self, crash=False, budget=4.0):
        """Show exit / crash unwinding. Bounded (about `budget` s): the encoder is closed so the fragmented MP4 on
        disk is complete, the audio file is closed, and nothing is muxed here. Clean exit: a detached process muxes
        the parts into the final MP4 (the window closes at once). Crash: nothing more; the next start's recover()
        finishes it. Never blocks on a long mux (H2)."""
        t_end = time.monotonic() + budget
        self.auto_stop_at = None
        for th in list(self.bg):                          # a session that is still starting
            th.join(max(0.0, min(1.0, t_end - time.monotonic())))
        if self.capture is not None and getattr(self.capture, "pending", None) and not crash:
            try:                                          # frames still in the readback ring go in first
                for (tc, data, size, tag) in self.capture.collect(force=True):
                    for s, slot in (tag or ()):
                        s.offer(slot, data, size)
            except Exception:
                pass
        sessions = [s for s in [self.rec, self.replay] + list(self.finishing) if s is not None]
        seen, handed = set(), []
        for s in sessions:
            if id(s) in seen:
                continue
            seen.add(id(s))
            s.handoff = True
            if s.done or s.failed:
                continue
            with s._lock:
                was_stopping = s.stopping
                s.stopping = True
                if s.t_end is None:
                    s.t_end = self.clock()
            if not s.ready:
                continue
            for p in list(s.mux_procs):                   # a stop-time mux still running: kill it, retry detached
                try:
                    p.kill()
                except Exception:
                    pass
            left = max(0.5, t_end - time.monotonic())
            if not was_stopping:                          # still recording: close it here, bounded
                s.total = max(s.cfr.last + 1, int(round((s.t_end - s.t0) * s.fps)), 1)
                s.writer.finish(s.total)
                try:
                    s.seal(timeout=left, abort=crash or s.kind == "replay")
                except Exception as ex:
                    self._log(f"[rec] seal failed: {ex}")
            else:                                         # its finalize thread is running: hurry it, bounded
                if crash or not s.sealed:
                    s.writer.abort()
                for th in s.threads:
                    if th.name == "rec-finalize" and th is not threading.current_thread():
                        th.join(max(0.1, min(left, 2.0)))
                        if th.is_alive() and not s.sealed:
                            try:
                                s.proc.kill()
                            except Exception:
                                pass
            if s.kind == "replay":
                shutil.rmtree(s.parts, ignore_errors=True)
            elif s.parts.exists():
                handed.append(s)
            s.done = True
        self.rec = self.replay = None
        for s in handed:
            if crash:
                self._log(f"[rec] {s.name}: left in {s.parts}; finished at the next start")
            else:
                self._log(f"[rec] saving {s.name} in the background (a separate process finishes the MP4)")
                try:
                    self.spawn(s.parts)
                except Exception as ex:
                    self._log(f"[rec] could not start the finisher ({ex}); finished at the next start")
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass


# ----------------------------------------------------------------------------- engine glue (live_visuals.py)
def parse_opts(text):
    """'source=projector,fps=60,audio=off' -> validated settings dict (ValueError on junk)."""
    c = {"cmd": "rec", "op": "set"}
    for part in (text or "").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            c[k.strip()] = v.strip().lower() in ("1", "true", "on", "yes") if k.strip() == "replay" else v.strip()
    out = validate_cmd(c)
    return {k: out[k] for k in DEFAULTS if k in out}


def for_engine(ctx, engine_audio, program_size, args=None, headless=False, folder=None):
    """The recorder as the render engine uses it: GL PBO capture, the engine's loopback tapped, settings and crash
    recovery only in show mode (headless runs never touch recordings/.settings.json)."""
    folder = Path(getattr(args, "record_dir", None) or folder or Path(__file__).parent / "recordings")
    r = Recorder(folder, capture=GLCapture(ctx), engine_audio=engine_audio, persist=not headless,
                 program_size=program_size)
    opts = getattr(args, "record_opts", None)
    if opts:
        r.settings.update(parse_opts(opts))
    if not headless:
        r.recover()
    secs = getattr(args, "record", None)
    if secs is not None:
        r.start(seconds=secs or None)
    return r


if __name__ == "__main__":                                  # detached finisher: python recorder.py --finalize <parts>
    import argparse
    _ap = argparse.ArgumentParser(description="finish an interrupted / handed-off recording")
    _ap.add_argument("--finalize", required=True, metavar="PARTS_DIR")
    _a = _ap.parse_args()
    _st, _msg, _ = finalize_parts(_a.finalize, find_ffmpeg(), wait_lock=60.0)
    print(f"[rec] {_st}: {_msg}")
    sys.exit(0 if _st in ("ok", "empty") else 1)
