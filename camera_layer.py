"""Crowd camera, engine side: silhouettes, motion field and dancer pose from camera_worker.py woven into the show.

Off by default and free when off: no process, no thread, no texture upload, and the shader skips every camera branch
(p_l_camera is forced to 0 while no camera data is live). The performer switches it on in the panel (CAMERA tab or
the top-bar indicator); that starts camera_worker.py as a hidden BELOW_NORMAL child process, OFF stops it and so
releases the webcam. ON is never restored on start (privacy: a restart comes back with the camera off); every other
setting (device, alignment, quality) is saved in live_state.json under ext.camera.

Engine contract (live_visuals.py, all guarded):
    cam = CameraLayer.for_engine(ctx, controller, extra)      once, after the offscreen passes exist
    cam.pre_update(dt, strain)                                each frame BEFORE controller.update (mod sources, ripples)
    uni.update(cam.uniforms(vals, uni))                       right before the scene uniforms are written
    cam.command(c)                                            {"cmd": "camera", ...} (validated by validate_command)
    state["camera"] = cam.state(); cam.close()
GL: mask on texture unit 12 (sampler camM, R8 160x90, temporally smoothed), motion field on unit 13 (camF, RG16F
32x18). The shader snippet SCENE_GLSL is spliced into SCENE_FS at //@CAMERA_GLSL.

Coordinates: camera uv = (x right, y DOWN, 0..1, the raw unmirrored camera image); screen uv = v_uv (y UP);
content = ((u - 0.5) * aspect, v - 0.5), the units serpents / beings / ripples use. align_matrix() maps camera uv to
screen uv from the alignment settings (mirror, crop, position, scale, keystone).

Photosensitive safety: crowd triggers never flash. Ripples from hand moves are rate-limited to RIPPLE_HZ (2 / s, below
the 3 Hz limit); mod-source signals are slewed (attack 0.10 s, release 0.35 s) so a crowd signal on any fader cannot
swing faster than about 2 Hz; the silhouette mask is temporally smoothed (time constant >= 0.07 s). Every camera
effect is composited before the brightness limiter, which therefore meters and limits it like any other light.
"""
import http.client
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UNIT_MASK, UNIT_FLOW = 12, 13
PORTS = [p for p in range(8767, 8776)]            # 8765 panel, 8766 AI worker
SIGNALS = ("crowd_motion", "crowd_left", "crowd_centre", "crowd_right", "crowd_presence",
           "hand_height", "hands_spread", "hip_sway", "hand_speed")
RIPPLE_HZ = 2.0
ATTACK_S, RELEASE_S = 0.18, 0.35
SLEW_MAX = 1.5                                    # signal units per second at most (a 6 Hz crowd flicker -> < 0.13 p-p)
MASK_TAU_MIN = 0.16                               # s: silhouettes can never step faster than this (photosensitive)
STALE_S = 1.5                                     # no new camera data for this long -> silhouettes fade, signals -> 0
RES_LABELS = ["480x270", "320x180", "240x135"]
QUALITY = [dict(fps=None, res=None, pose_every=None), dict(fps=15, res=None, pose_every=3),
           dict(fps=10, res=2, pose_every=4, max_poses=1)]   # governor steps: 0 = as set, 1 strained, 2 critical
DEFAULTS = dict(device=0, res=1, fps=30, pose=True, max_poses=2, mirror=True, crop=[0.0, 0.0, 1.0, 1.0],
                pos_x=0.0, pos_y=0.0, scale=1.0, key_x=0.0, key_y=0.0, gain=1.0, show_ir=False, auto_quality=True)
_ACTIVE = None                                    # the engine's layer (remote_control's /camera-preview reads it)


# ------------------------------------------------------------------ settings + commands (pure python)

def _f(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return min(max(v, lo), hi) if math.isfinite(v) else default


def clean_settings(raw):
    """Saved / commanded settings -> complete, clamped dict (unknown keys dropped)."""
    s = dict(DEFAULTS, crop=list(DEFAULTS["crop"]))
    raw = raw if isinstance(raw, dict) else {}
    s["device"] = int(_f(raw.get("device", 0), 0, 9, 0))
    s["res"] = int(round(_f(raw.get("res", 1), 0, 2, 1)))
    s["fps"] = int(round(_f(raw.get("fps", 30), 5, 30, 30)))
    s["max_poses"] = int(round(_f(raw.get("max_poses", 2), 1, 3, 2)))
    for k in ("pose", "mirror", "show_ir", "auto_quality"):
        if isinstance(raw.get(k), bool):
            s[k] = raw[k]
    for k, lo, hi in (("pos_x", -1.0, 1.0), ("pos_y", -1.0, 1.0), ("scale", 0.25, 4.0), ("key_x", -1.0, 1.0),
                      ("key_y", -1.0, 1.0), ("gain", 0.25, 4.0)):
        s[k] = _f(raw.get(k, DEFAULTS[k]), lo, hi, DEFAULTS[k])
    c = raw.get("crop")
    if isinstance(c, (list, tuple)) and len(c) == 4:
        x0, y0, x1, y1 = (_f(v, 0.0, 1.0, d) for v, d in zip(c, DEFAULTS["crop"]))
        if x1 - x0 < 0.1:                               # keep a 10 % window: move the far edge, never reset all four
            x1 = min(1.0, x0 + 0.1); x0 = x1 - 0.1
        if y1 - y0 < 0.1:
            y1 = min(1.0, y0 + 0.1); y0 = y1 - 0.1
        s["crop"] = [x0, y0, x1, y1]
    return s


SET_KEYS = set(DEFAULTS)


def validate_command(c):
    """{"cmd": "camera", "op": on | off | toggle | set {settings...} | align_reset | devices} -> clean copy."""
    if not isinstance(c, dict):
        raise ValueError("camera command must be an object")
    op = c.get("op")
    if op in ("on", "off", "toggle", "align_reset", "devices"):
        return {"cmd": "camera", "type": "camera", "op": op}
    if op == "set":
        vals = {k: c[k] for k in SET_KEYS if k in c}
        if not vals:
            raise ValueError("camera set: nothing to set")
        for k, v in vals.items():
            if k == "crop":
                if not (isinstance(v, (list, tuple)) and len(v) == 4 and all(
                        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)):
                    raise ValueError("'crop' must be [x0, y0, x1, y1]")
            elif isinstance(DEFAULTS[k], bool):
                if not isinstance(v, bool):
                    raise ValueError(f"'{k}' must be true or false")
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"'{k}' must be a number")
        merged = clean_settings(vals)
        return dict({k: merged[k] for k in vals}, cmd="camera", type="camera", op="set")
    raise ValueError("camera: unknown op")


# ------------------------------------------------------------------ alignment

def homography(src, dst):
    """3x3 H with H @ [x, y, 1] ~ dst for the 4 point pairs."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def screen_quad(s):
    """Where the (cropped) camera picture lands on the frame, screen uv (y up): TL, TR, BR, BL."""
    cx, cy = 0.5 + 0.5 * s["pos_x"], 0.5 + 0.5 * s["pos_y"]
    hw = hh = 0.5 * s["scale"]
    kx, ky = 0.5 * s["key_x"], 0.5 * s["key_y"]               # keystone: top narrower (+x), right taller (+y)
    return [(cx - hw * (1 - kx), cy + hh * (1 - ky)), (cx + hw * (1 - kx), cy + hh * (1 + ky)),
            (cx + hw * (1 + kx), cy - hh * (1 + ky)), (cx - hw * (1 + kx), cy - hh * (1 - ky))]


def align_matrix(s):
    """Camera uv (y down, raw) -> screen uv (y up). Mirror: the camera's right side shows on the frame's left."""
    x0, y0, x1, y1 = s["crop"]
    cam = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if s["mirror"]:
        cam = [(x1, y0), (x0, y0), (x0, y1), (x1, y1)]
    return homography(cam, screen_quad(s))


def apply_h(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return (float(p[0] / p[2]), float(p[1] / p[2])) if abs(p[2]) > 1e-9 else (float("nan"), float("nan"))


# ------------------------------------------------------------------ bodies + signals (pure python / numpy)

def body_to_screen(pose, H, min_vis=0.5):
    """{name: [x, y, vis]} in camera uv -> {name: (u, v)} in screen uv, visible points only."""
    out = {}
    for n, p in (pose or {}).items():
        if isinstance(p, (list, tuple)) and len(p) == 3 and p[2] >= min_vis:
            u, v = apply_h(H, p[0], p[1])
            if math.isfinite(u) and math.isfinite(v):
                out[n] = (u, v)
    return out


def _mid(b, a, c):
    if a in b and c in b:
        return ((b[a][0] + b[c][0]) * 0.5, (b[a][1] + b[c][1]) * 0.5)
    return b.get(a) or b.get(c)


def torso(b):
    sh, hp = _mid(b, "l_shoulder", "r_shoulder"), _mid(b, "l_hip", "r_hip")
    if sh is None or hp is None:
        return None, sh, hp
    return max(math.hypot(sh[0] - hp[0], sh[1] - hp[1]), 0.02), sh, hp


def primary(bodies):
    """The body with the longest torso (nearest / biggest dancer), or None."""
    best, bt = None, 0.0
    for b in bodies:
        t, _, _ = torso(b)
        if t is not None and t > bt:
            best, bt = b, t
    return best


class Signals:
    """Mod sources from camera data. on_data() sets targets when a new camera frame arrives, step() slews the
    outputs every engine frame (attack ATTACK_S, release RELEASE_S), lost() lets everything fall to 0."""

    def __init__(self):
        self.v = dict.fromkeys(SIGNALS, 0.0)
        self.target = dict.fromkeys(SIGNALS, 0.0)
        self._hip_c = None
        self._wrists = None                            # previous frame's wrists of the primary body (screen uv)

    def on_data(self, info, bodies, dt, gain=1.0):
        tg = dict.fromkeys(SIGNALS, 0.0)
        for k in ("motion", "left", "centre", "right"):
            tg["crowd_" + k] = min(max(float(info.get(k, 0.0)) * gain, 0.0), 1.0)
        tg["crowd_presence"] = min(max(float(info.get("presence", 0.0)), 0.0), 1.0)
        b = primary(bodies)
        t, sh, hp = torso(b) if b else (None, None, None)
        if t is not None:
            wd = {n: b[n] for n in ("l_wrist", "r_wrist") if n in b}
            wr = list(wd.values())
            if wr:
                tg["hand_height"] = max(min(max(0.5 + (w[1] - sh[1]) / (2.0 * t), 0.0), 1.0) for w in wr)
            if len(wr) == 2:
                tg["hands_spread"] = min(math.hypot(wr[0][0] - wr[1][0], wr[0][1] - wr[1][1]) / (2.2 * t), 1.0)
            a = 1.0 - math.exp(-max(dt, 0.0) / 1.5)
            self._hip_c = hp[0] if self._hip_c is None else self._hip_c + (hp[0] - self._hip_c) * a
            tg["hip_sway"] = min(max((hp[0] - self._hip_c) / (0.5 * t), -1.0), 1.0)
            if self._wrists is not None and dt > 0:            # same wrist to same wrist only (never left -> right)
                sp = [math.hypot(w[0] - self._wrists[n][0], w[1] - self._wrists[n][1]) / dt / t
                      for n, w in wd.items() if n in self._wrists]
                if sp:
                    tg["hand_speed"] = 1.0 - math.exp(-max(sp) / 4.0)
            self._wrists = wd
        else:
            self._wrists, self._hip_c = None, None
        self.target = tg

    def lost(self):
        self.target = dict.fromkeys(SIGNALS, 0.0)
        self._wrists, self._hip_c = None, None

    def step(self, dt):
        dt = min(max(float(dt), 0.0), 0.25)
        for k, tv in self.target.items():
            v = self.v[k]
            tau = ATTACK_S if abs(tv) > abs(v) else RELEASE_S
            step = (tv - v) * (1.0 - math.exp(-dt / tau))
            lim = SLEW_MAX * dt
            self.v[k] = v + min(max(step, -lim), lim)
            if abs(self.v[k]) < 1e-4 and tv == 0.0:
                self.v[k] = 0.0
        return self.v

    def idle(self):
        return all(v == 0.0 for v in self.v.values()) and all(v == 0.0 for v in self.target.values())


class RippleGate:
    """Fast hand moves -> ripple positions, at most RIPPLE_HZ per second overall and one per hand per second."""

    def __init__(self, hz=RIPPLE_HZ):
        self.min_dt, self.last, self.per = 1.0 / hz, -1e9, {}
        self.prev = {}

    def feed(self, now, hands, dt, amount):
        """hands: {key: (u, v)} screen uv; returns [(u, v)] to spawn (0 or 1 item)."""
        out = []
        if amount <= 0.0 or dt <= 0:
            self.prev = dict(hands)
            return out
        thr = 2.2 - 1.6 * min(amount, 1.0)                    # uv per second; amount 1 = 0.6 uv/s is enough
        for k, p in hands.items():
            q = self.prev.get(k)
            if q is None:
                continue
            sp = math.hypot(p[0] - q[0], p[1] - q[1]) / dt
            if sp > thr and now - self.last >= self.min_dt and now - self.per.get(k, -1e9) >= 1.0:
                if 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0:
                    out.append(p)
                    self.last = self.per[k] = now
                    break
        self.prev = dict(hands)
        return out


def steer_offsets(pos, hands, off, amount, dt, aspect):
    """Beings follow (+) or flee (-) the nearest hand. pos (n, 2) content units, hands [(x, y)] content units,
    off (n, 2) current offsets (smoothed, updated in place). Returns the offsets."""
    tau = 0.6
    a = 1.0 - math.exp(-max(dt, 0.0) / tau)
    for i in range(len(pos)):
        tgt = np.zeros(2)
        if hands and amount != 0.0:
            base = pos[i]
            d = [np.array(h) - base for h in hands]
            j = int(np.argmin([np.hypot(*x) for x in d]))
            v, dist = d[j], float(np.hypot(*d[j]))
            if amount > 0:
                tgt = v * min(amount, 1.0) * 0.85
            else:
                R = 0.5
                if dist < R:
                    tgt = -v / max(dist, 1e-3) * (R - dist) * min(-amount, 1.0) * 1.3
        off[i] += (tgt - off[i]) * a
    # stay inside the frame
    lim = np.array([aspect * 0.5 - 0.05, 0.45])
    off[:] = np.clip(pos + off, -lim, lim) - pos
    return off


def arm_track(b, aspect):
    """Primary body's most raised arm -> (heading, offset, head x) of a straight serpent track whose head sits at
    the wrist and whose body runs back along the forearm, in content units; None if no arm is visible."""
    best = None
    for side in ("l", "r"):
        w, e, s = b.get(side + "_wrist"), b.get(side + "_elbow"), b.get(side + "_shoulder")
        base = e or s
        if w is None or base is None:
            continue
        if best is None or w[1] > best[0][1]:
            best = (w, base)
    if best is None:
        return None
    (wu, wv), (bu, bv) = best
    wx, wy, bx, by = (wu - 0.5) * aspect, wv - 0.5, (bu - 0.5) * aspect, bv - 0.5
    if math.hypot(wx - bx, wy - by) < 1e-3:
        return None
    ang = math.atan2(wy - by, wx - bx)
    ca, sa = math.cos(ang), math.sin(ang)
    return ang, -sa * wx + ca * wy, ca * wx + sa * wy + 0.03


# ------------------------------------------------------------------ worker process + link (threads, never the GL thread)

def available():
    """(ok, reason, notes): can the camera run at all, without importing anything heavy."""
    if importlib.util.find_spec("cv2") is None:
        return False, "opencv-python is not installed in the venv (.\\.venv\\Scripts\\pip.exe install opencv-python)", ""
    if not (HERE / "camera_worker.py").is_file():
        return False, "camera_worker.py is missing", ""
    notes = []
    if importlib.util.find_spec("mediapipe") is None:
        notes.append("MediaPipe missing: silhouettes from motion only, no pose")
    else:
        md = HERE / "models" / "camera"
        if not (md / "selfie_segmenter_landscape.tflite").is_file():
            notes.append("segmentation model missing (models/camera/): motion-only silhouettes")
        if not (md / "pose_landmarker_lite.task").is_file():
            notes.append("pose model missing (models/camera/): no dancer tracking")
    return True, "", "; ".join(notes)


def free_port(ports=PORTS):
    for p in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    return None


class WorkerLink:
    """Reads the worker's newest result on a daemon thread (latest frame wins); also config, status, debug JPEG."""

    def __init__(self, port, token, unpack, timeout=2.0):
        self.port, self.token, self.unpack, self.timeout = port, token, unpack, timeout
        self._lock = threading.Lock()
        self._latest = None
        self.status = {}
        self.error = ""
        self.connected = False
        self.debug = None                              # JPEG bytes
        self.debug_want = 0.0
        self._cfg = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-link", daemon=True)
        self._thread.start()

    def take(self):
        with self._lock:
            r, self._latest = self._latest, None
        return r

    def configure(self, cfg):
        with self._lock:
            self._cfg = dict(self._cfg or {}, **cfg)   # merge: a quality step never drops a queued key

    def quit(self):
        self._stop.set()
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1.0)
            c.request("POST", "/quit", body=b"{}", headers={"X-Token": self.token, "Content-Type": "application/json"})
            c.getresponse().read()
            c.close()
        except Exception:
            pass
        self._thread.join(timeout=2.5)

    def _run(self):
        conn, seq, last_status, last_dbg = None, 0, 0.0, 0.0
        hdr = {"X-Token": self.token}
        while not self._stop.is_set():
            try:
                if conn is None:
                    conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
                now = time.monotonic()
                with self._lock:
                    cfg, self._cfg = self._cfg, None
                if cfg is not None:
                    conn.request("POST", "/config", body=json.dumps(cfg).encode(), headers=dict(hdr, **{"Content-Type": "application/json"}))
                    conn.getresponse().read()
                if now - last_status > 1.0:
                    conn.request("GET", "/status", headers=hdr)
                    r = conn.getresponse()
                    body = r.read()
                    if r.status == 200:
                        self.status = json.loads(body)
                    last_status = now
                if now - self.debug_want < 3.0 and now - last_dbg > 0.4:
                    conn.request("GET", "/debug.jpg", headers=hdr)
                    r = conn.getresponse()
                    body = r.read()
                    if r.status == 200:
                        self.debug = body
                    last_dbg = now
                conn.request("GET", f"/frame?after={seq}", headers=hdr)
                r = conn.getresponse()
                body = r.read()
                self.connected = True
                if r.status == 200:
                    res = self.unpack(body)
                    seq = res["seq"]
                    with self._lock:
                        self._latest = res
                    self.error = ""
                elif r.status == 403:
                    self.error = "the camera worker refused the token"
                    self._stop.wait(1.0)
            except Exception as ex:
                self.connected = False
                self.error = f"no link to the camera worker ({type(ex).__name__})"
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                conn = None
                self._stop.wait(0.3)
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def kill_tree(proc):
    """Hard stop of OUR worker: on Windows the venv launcher's real python child too (taskkill /T on our own pid)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=5,
                           creationflags=0x08000000, stdin=subprocess.DEVNULL)
        else:
            proc.kill()
        proc.wait(timeout=2.0)
    except Exception:
        pass


def last_log_line(here=HERE):
    """The newest camera log's last [camera] line (the real reason a worker died early)."""
    try:
        logs = sorted((Path(here) / "logs").glob("camera_*.log"))
        if not logs:
            return ""
        lines = [x for x in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines() if x.startswith("[camera]")]
        return lines[-1][9:].strip()[:200] if lines else ""
    except OSError:
        return ""


def spawn_worker(port, s, here=HERE, token=None, log_dir=None):
    """Start camera_worker.py hidden, BELOW_NORMAL, its output into logs/camera_<time>.log (no frames, ever)."""
    log_dir = Path(log_dir) if log_dir is not None else here / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / time.strftime("camera_%Y%m%d_%H%M%S.log"), "ab")
    args = [sys.executable, "-u", str(here / "camera_worker.py"), "--port", str(port), "--device", str(s["device"]),
            "--fps", str(s["fps"]), "--res", str(s["res"]), "--pose-every", "2",
            "--max-poses", str(s["max_poses"] if s["pose"] else 0), "--parent", str(os.getpid())]
    flags = (0x08000000 | 0x00004000) if os.name == "nt" else 0     # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
    try:
        env = dict(os.environ, CAMERA_WORKER_TOKEN=token or "")    # never on the command line
        return subprocess.Popen(args, cwd=str(here), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags, env=env)
    finally:
        log.close()


def read_token(here=HERE):
    try:
        return (Path(here) / ".remote_token").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ------------------------------------------------------------------ the layer

class CameraLayer:
    def __init__(self, ctx=None, controller=None, here=HERE, spawn=spawn_worker, link_factory=WorkerLink,
                 clock=time.monotonic, aspect=1.6, devices_fn=None, sync=False):
        self.sync = bool(sync)                         # tests: start / stop inline; the engine: on their own threads
        self._gen = 0                                  # start/stop generation (a late starter never revives an OFF)
        self._teardown = None
        self.token = None                              # per-spawn secret shared only with our own worker (env, not argv)
        self._mask_target = None
        self.ctx, self.ctl, self.here = ctx, controller, Path(here)
        self._spawn, self._link_factory, self.clock, self.aspect = spawn, link_factory, clock, aspect
        self._devices_fn = devices_fn
        ext = getattr(controller, "ext", None)
        self.s = clean_settings(ext.get("camera") if isinstance(ext, dict) else None)
        self.H = align_matrix(self.s)
        self.Hinv = np.linalg.inv(self.H)
        self.on = False
        self.proc, self.link, self.port = None, None, None
        self.status, self.reason = "off", ""
        self.ok, why, self.notes = available()
        if not self.ok:
            self.status, self.reason = "unavailable", why
        self.devices, self._dev_busy = None, False
        self.signals = Signals()
        self.ripples = RippleGate()
        self.bodies, self.hands = [], {}
        self.mask = None                               # float32 (h, w) smoothed
        self.flow = None
        self.t_data = None                             # clock() of the newest data
        self._data_t = None                            # worker timestamp of the newest data
        self.live = 0.0                                # fade of the camera effect 0..1 (data present)
        self.tex_m = self.tex_f = None
        self.quality, self._strain_t, self._ok_t = 0, 0.0, 0.0
        self.cost_ms, self.uploads = 0.0, 0
        self.restarts, self._started = 0, 0.0
        self._being_off, self._serp_w, self._serp = None, 0.0, None
        self._vals = {}
        self._dirty = False

    @classmethod
    def for_engine(cls, ctx, controller, extra=(), aspect=1.6):
        global _ACTIVE
        layer = cls(ctx, controller, aspect=aspect)
        for ex in extra:                               # hook the beings herd and the swarm (cheap no-op while off)
            obj = ex.get("obj")
            if obj is None:
                continue
            if ex.get("param") == "l_beings" and hasattr(obj, "herd"):
                layer.attach_herd(obj.herd)
            if ex.get("param") == "l_land" and hasattr(obj, "swarm"):
                layer.attach_swarm(obj.swarm)
        _ACTIVE = layer
        return layer

    # ---- settings
    def _save(self):
        ext = getattr(self.ctl, "ext", None)
        if isinstance(ext, dict):
            ext["camera"] = dict(self.s, crop=list(self.s["crop"]))

    def _set(self, vals):
        old = dict(self.s)
        self.s = clean_settings(dict(self.s, **vals))
        self.H = align_matrix(self.s)
        self.Hinv = np.linalg.inv(self.H)
        self._save()
        if self.on and any(old[k] != self.s[k] for k in ("device",)):
            self.stop(); self.start()                  # new device: reopen
        elif self.on and any(old[k] != self.s[k] for k in ("fps", "res", "pose", "max_poses")):
            self._push_quality()

    def command(self, c):
        op = c.get("op")
        if op == "on" or (op == "toggle" and not self.on):
            self.start()
        elif op == "off" or op == "toggle":
            self.stop()
        elif op == "set":
            self._set({k: v for k, v in c.items() if k in SET_KEYS})
        elif op == "align_reset":
            self._set({k: (list(DEFAULTS[k]) if k == "crop" else DEFAULTS[k])
                       for k in ("crop", "pos_x", "pos_y", "scale", "key_x", "key_y")})
        elif op == "devices":
            self.scan_devices()

    def scan_devices(self):
        if self._dev_busy:
            return
        self._dev_busy = True

        def work():
            try:
                fn = self._devices_fn
                if fn is None:
                    from camera_worker import list_devices as fn
                self.devices = fn()
            except Exception:
                self.devices = [dict(index=0, name="Camera 0", ir=False)]
            finally:
                self._dev_busy = False
        threading.Thread(target=work, name="camera-devices", daemon=True).start()

    # ---- process
    def start(self, manual=True):
        """CAMERA ON. Returns at once: the port probe + process spawn run on a thread (never stall the render loop),
        after any previous worker has released the device."""
        if self.on:
            return
        if not self.ok:
            self.status = "unavailable"
            return
        if manual:
            self.restarts = 0
        self.on, self.status, self.reason, self._started, self.quality = True, "starting", "", self.clock(), 0
        self._gen += 1
        gen, prev = self._gen, self._teardown
        if self.sync:
            self._spawn_now(gen, prev)
        else:
            threading.Thread(target=self._spawn_now, args=(gen, prev), name="camera-start", daemon=True).start()

    def _spawn_now(self, gen, prev):
        import secrets
        if prev is not None:
            prev.join(timeout=8.0)                     # the old worker must have let go of the webcam first
        port = free_port()
        if port is None:
            self.status, self.reason, self.on = "error", "no free local port in 8767-8775", False
            return
        from camera_worker import unpack
        token = secrets.token_urlsafe(24)
        try:
            proc = self._spawn(port, self.s, self.here, token)
        except TypeError:                              # an older spawn signature (tests)
            proc = self._spawn(port, self.s, self.here)
        except Exception as ex:
            self.status, self.reason, self.on = "error", f"could not start camera_worker.py: {ex}"[:200], False
            return
        link = self._link_factory(port, token, unpack)
        if gen != self._gen or not self.on:            # switched OFF meanwhile: undo
            self._stop_worker(link, proc)
            return
        self.token, self.port, self.proc, self.link = token, port, proc, link
        print(f"[camera] ON (worker pid {getattr(proc, 'pid', '?')}, port {port})")

    @staticmethod
    def _stop_worker(link, proc):
        if link is not None:
            try:
                link.quit()
            except Exception:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=2.0)
            except Exception:
                kill_tree(proc)                        # only ever our own child (and the real python behind the launcher)

    def stop(self):
        """CAMERA OFF: returns at once; the worker is asked to quit (it releases the device) and, failing that, its
        process tree is killed, on a teardown thread. A following ON waits for that teardown."""
        was = self.on
        self.on = False
        self._gen += 1
        link, proc, self.link, self.proc = self.link, self.proc, None, None
        if link is not None or proc is not None:
            if self.sync:
                self._stop_worker(link, proc)
            else:
                self._teardown = threading.Thread(target=self._stop_worker, args=(link, proc), name="camera-stop",
                                                  daemon=True)
                self._teardown.start()
        self.status = "off" if self.ok else "unavailable"
        self.reason = "" if self.ok else self.reason
        self.t_data, self.mask, self.flow, self.bodies, self.hands = None, None, None, [], {}
        self._mask_target, self.live, self._data_t, self.port, self.token = None, 0.0, None, None, None
        self.signals.lost()
        if was:
            print("[camera] OFF (device released)")

    def _push_quality(self):
        if self.link is None:
            return
        q = QUALITY[self.quality] if self.s.get("auto_quality", True) else QUALITY[0]
        cfg = dict(fps=min(self.s["fps"], q["fps"] or 30), res=max(self.s["res"], q["res"] if q["res"] is not None else 0),
                   pose_every=q["pose_every"] or 2,
                   max_poses=0 if not self.s["pose"] else min(self.s["max_poses"], q.get("max_poses", 3)))
        self.link.configure(cfg)

    def _watch(self, now, strain):
        """Process health + governor steps (strain: 0 ok, 1 strained, 2 critical, from governor.level)."""
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            err = (self.link.status.get("error") if self.link is not None else "") or last_log_line(self.here) \
                or f"worker exited ({code})"
            healthy = now - self._started > 300.0
            self.stop()
            if healthy:
                self.restarts = 0                      # a long healthy run earns its restarts back
            if self.restarts < 2 and now - self._started > 10.0 and code not in (3, 4):
                self.restarts += 1
                self.start(manual=False)
                self.reason = f"restarted after: {err}"
            else:
                self.status, self.reason = "error", err
            return
        if self.link is not None:
            if self.t_data is not None and now - self.t_data < STALE_S:
                self.status = "running"
            elif self.status == "running" or now - self._started > 8.0:
                self.status = "starting" if self.t_data is None else "stalled"
                self.reason = self.link.status.get("error") or self.link.error or (
                    "waiting for the first frame" if self.t_data is None else "no new frames")
        want = self.quality
        if strain >= 1:
            self._ok_t = now
            if now - self._strain_t > 3.0:
                want = min(2, max(self.quality, strain))
        else:
            self._strain_t = now
            if self.quality and now - self._ok_t > 20.0:
                want, self._ok_t = self.quality - 1, now
        if want != self.quality:
            self.quality = want
            self._push_quality()

    # ---- per frame (GL thread). Everything below returns at once while the camera is off and the signals are 0.
    def pre_update(self, dt, strain=0):
        if not self.on and self.signals.idle() and self.live == 0.0:
            if getattr(self.ctl, "signals", None):
                self.ctl.signals.clear()
            return
        t0 = time.perf_counter()
        now = self.clock()
        if self.on:
            self._watch(now, strain)
            res = self.link.take() if self.link is not None else None
            if res is not None:
                self._ingest(res, now)
        if self.t_data is None or now - self.t_data > STALE_S:
            self.signals.lost()
            self.live = max(0.0, self.live - dt / 0.5)
        else:
            self.live = min(1.0, self.live + dt / 0.5)
        if self._mask_target is not None:              # silhouettes glide every frame (time constant >= MASK_TAU_MIN)
            tau = MASK_TAU_MIN + 0.45 * float(self._vals.get("crowd_soft", 0.4))
            a = 1.0 - math.exp(-max(dt, 0.0) / tau)
            self.mask = self._mask_target.copy() if self.mask is None or self.mask.shape != self._mask_target.shape \
                else self.mask + (self._mask_target - self.mask) * a
            self._dirty = True
        sig = self.signals.step(dt)
        if getattr(self.ctl, "signals", None) is not None:
            self.ctl.signals.update(sig)
        self.cost_ms = self.cost_ms * 0.95 + (time.perf_counter() - t0) * 1000 * 0.05

    def _ingest(self, res, now):
        info = res.get("info") or {}
        ddt = (res["t"] - self._data_t) if self._data_t is not None else 0.0
        ddt = ddt if 0.0 < ddt < 1.0 else 0.0
        self._data_t, self.t_data = res["t"], now
        self.bodies = [body_to_screen(p, self.H) for p in info.get("poses") or [] if isinstance(p, dict)]
        hands = {}
        for i, b in enumerate(self.bodies):
            for n in ("l_wrist", "r_wrist"):
                if n in b:
                    hands[f"{i}{n}"] = b[n]
        self.hands = hands
        self.signals.on_data(info, self.bodies, ddt, self.s["gain"])
        amt = self._vals.get("crowd_ripples", 0.0) * min(self._vals.get("l_camera", 0.0) * 4.0, 1.0)
        for u, v in self.ripples.feed(now, hands, ddt, amt):
            try:
                self.ctl.ripple(u, v)
            except Exception:
                pass
        # silhouettes: temporal smoothing (time constant >= 0.07 s: nothing it draws can flicker above ~2 Hz)
        self._mask_target = res["mask"].astype(np.float32) * (1.0 / 255.0)   # pre_update glides toward it
        fl = res["flow"]
        af = 1.0 - math.exp(-max(ddt, 1.0 / 30.0) / 0.25)
        self.flow = fl if self.flow is None or self.flow.shape != fl.shape else self.flow + (fl - self.flow) * af
        self._dirty = True

    def _upload(self):
        if self.ctx is None or self.mask is None:
            return False
        h, w = self.mask.shape
        if self.tex_m is None or self.tex_m.size != (w, h):
            if self.tex_m is not None:
                self.tex_m.release()
            self.tex_m = self.ctx.texture((w, h), 1, dtype="f1")
            self.tex_m.repeat_x = self.tex_m.repeat_y = False
        fh, fw = self.flow.shape[:2]
        if self.tex_f is None or self.tex_f.size != (fw, fh):
            if self.tex_f is not None:
                self.tex_f.release()
            self.tex_f = self.ctx.texture((fw, fh), 2, dtype="f2")
            self.tex_f.repeat_x = self.tex_f.repeat_y = False
        if getattr(self, "_dirty", False):
            self.tex_m.write(np.clip(self.mask * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())
            self.tex_f.write(self.flow.astype(np.float16).tobytes())
            self._dirty = False
            self.uploads += 1
        return True

    def uniforms(self, vals, uni=None):
        """Scene uniforms. While there is no live camera data p_l_camera is 0 and nothing is bound or uploaded."""
        self._vals = vals
        amt = float(vals.get("l_camera", 0.0)) * self.live
        if not self.on or amt <= 0.001 or self.mask is None:
            if uni is not None and (self.on or self._serp_w > 0.0):
                self._serpent(vals, uni)
            return {"p_l_camera": 0.0, "p_crowd_push": 0.0}
        t0 = time.perf_counter()
        if not self._upload():
            return {"p_l_camera": 0.0, "p_crowd_push": 0.0}
        self.tex_m.use(UNIT_MASK)
        self.tex_f.use(UNIT_FLOW)
        out = {"camM": UNIT_MASK, "camF": UNIT_FLOW, "p_l_camera": amt,
               "camH": np.ascontiguousarray(self.Hinv.T, dtype="f4"),
               "camFlipX": -1.0 if self.s["mirror"] else 1.0}
        if uni is not None:
            self._serpent(vals, uni)
        self.cost_ms += (time.perf_counter() - t0) * 1000 * 0.05
        return out

    # ---- dancer -> serpent 1 (overrides uni["serpA"][0] while an arm is seen)
    def _serpent(self, vals, uni):
        k = float(vals.get("crowd_serpent", 0.0)) * min(float(vals.get("l_camera", 0.0)) * 4.0, 1.0)
        now = self.clock()
        dt = min(max(now - getattr(self, "_serp_t", now), 0.0), 0.1)
        self._serp_t = now
        if (k <= 0.0 and self._serp_w <= 0.0) or vals.get("serpent", 0.0) <= 0.0 or "serpA" not in uni:
            return
        b = primary(self.bodies)
        tr = arm_track(b, self.aspect) if b and self.t_data is not None and self.clock() - self.t_data < STALE_S else None
        if tr is not None:
            if self._serp is None:
                self._serp = list(tr)
            else:
                d = (tr[0] - self._serp[0] + math.pi) % (2 * math.pi) - math.pi
                f = 1.0 - math.exp(-dt / 0.04)
                self._serp = [self._serp[0] + d * f, self._serp[1] + (tr[1] - self._serp[1]) * f,
                              self._serp[2] + (tr[2] - self._serp[2]) * f]
        want = k if tr is not None else 0.0
        self._serp_w += (want - self._serp_w) * (1.0 - math.exp(-dt / 0.3))   # 0.3 s glide on and off the arm
        if self._serp_w < 0.01:
            if want == 0.0:
                self._serp_w = 0.0
            return
        if self._serp is None:
            return
        w = min(self._serp_w, 1.0)
        sa = np.array(uni["serpA"], dtype="f4", copy=True)
        o = sa[0]
        d = (self._serp[0] - o[0] + math.pi) % (2 * math.pi) - math.pi
        sa[0] = (o[0] + d * w, o[1] + (self._serp[1] - o[1]) * w, o[2] * (1.0 - w), o[3] + (self._serp[2] - o[3]) * w)
        uni["serpA"] = sa
        if w > 0.5 and "serpC" in uni:                 # the previous crossing's tail would cut across: park it
            sc = np.array(uni["serpC"], dtype="f4", copy=True)
            sc[0, 3] = -50.0
            uni["serpC"] = sc

    # ---- dancer -> beings / swarm (wrappers installed once; they do nothing while off)
    def _hands_content(self):
        if not self.on or self.t_data is None or self.clock() - self.t_data > STALE_S:
            return []
        return [((u - 0.5) * self.aspect, v - 0.5) for u, v in self.hands.values()]

    def attach_herd(self, herd):
        orig = herd.step
        layer = self

        def step(dt, values, bands=(0.0, 0.0, 0.0), clock=None):
            r = orig(dt, values, bands, clock)
            try:
                k = float(values.get("crowd_beings", 0.0)) * min(float(values.get("l_camera", 0.0)) * 4.0, 1.0)
                if k == 0.0 and layer._being_off is None:
                    return r
                ua = np.array(r[0], dtype="f4", copy=True)
                if layer._being_off is None or layer._being_off.shape != (len(ua), 2):
                    layer._being_off = np.zeros((len(ua), 2))
                off = steer_offsets(ua[:, :2].astype(np.float64), layer._hands_content(), layer._being_off, k, dt,
                                    layer.aspect)
                if k == 0.0 and np.abs(off).max() < 1e-3:
                    layer._being_off = None
                ua[:, :2] += off.astype("f4")
                return (ua,) + tuple(r[1:])
            except Exception:
                return r
        herd.step = step

    def attach_swarm(self, swarm):
        orig = swarm.step
        layer = self

        def step(dt, t, turb, vort, kick):
            verts = orig(dt, t, turb, vort, kick)
            try:
                k = float(layer._vals.get("crowd_swarm", 0.0)) * min(float(layer._vals.get("l_camera", 0.0)) * 4.0, 1.0)
                hands = layer._hands_content() if k > 0.0 else []
                if hands:
                    p = swarm.pos
                    d = min(max(float(dt), 0.0), 0.05)
                    for hx, hy in hands[:6]:                 # approximate: content units -> swarm space (radius ~2.6)
                        a = np.array([hx * 2.4, hy * 2.4, 0.0], dtype="f4")
                        v = a - p
                        p += v / (np.linalg.norm(v, axis=1, keepdims=True) + 0.35) * np.float32(k * d * 2.2)
                    swarm.verts[:, 1, :3] = p
            except Exception:
                pass
            return verts
        swarm.step = step

    # ---- panel
    def state(self):
        st = dict(on=self.on, status=self.status, reason=self.reason, notes=self.notes, available=self.ok,
                  settings=dict(self.s, crop=list(self.s["crop"])),
                  devices=[dict(d) for d in self.devices] if self.devices else self.devices, quality=self.quality,
                  signals={k: round(v, 3) for k, v in self.signals.v.items()},
                  bodies=len(self.bodies), cost_ms=round(self.cost_ms, 3), res_labels=RES_LABELS)
        if self.link is not None:
            ws = self.link.status or {}
            st.update(fps=ws.get("fps"), grab_fps=ws.get("grab_fps"), ms=dict(ws.get("ms") or {}), backend=ws.get("backend"))
        return st

    def cost(self):
        """For the governor / HEALTH card: engine ms per frame (EMA), worker fps and per-stage ms."""
        ws = (self.link.status if self.link is not None else {}) or {}
        return dict(active=self.on, engine_ms=round(self.cost_ms, 3), fps=ws.get("fps"), ms=ws.get("ms"),
                    quality=self.quality)

    def still(self, skip=2, timeout=6.0):
        """AUTO MAP (its own thread, never the GL thread): a fresh grey camera still, 640x360 uint8, taken `skip`
        frames after the request. Memory only. Raises RuntimeError with a reason the wizard shows."""
        if not self.on or self.port is None:
            raise RuntimeError("switch the CAMERA on (CAMERA tab) and point it at the projection")
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request("GET", f"/still?skip={int(skip)}", headers={"X-Token": self.token or ""})
            r = c.getresponse()
            body = r.read()
        finally:
            c.close()
        if r.status != 200:
            raise RuntimeError("the camera did not deliver a frame (is it still on?)")
        from camera_worker import unpack_still
        return unpack_still(body)

    def _post_config(self, cfg, timeout=3.0):
        """Synchronous /config (wizard thread only): the change is in force when this returns."""
        if not self.on or self.port is None:
            raise RuntimeError("switch the CAMERA on (CAMERA tab) and point it at the projection")
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request("POST", "/config", body=json.dumps(cfg).encode(),
                      headers={"X-Token": self.token or "", "Content-Type": "application/json"})
            r = c.getresponse()
            r.read()
        finally:
            c.close()
        return r.status == 200

    def lock_exposure(self, on):
        try:
            self._post_config({"lock_exposure": bool(on)})
        except Exception:
            pass

    def allow_stills(self, seconds):
        """AUTO MAP only: the worker serves /still for `seconds` (0 = stop now). Otherwise /still refuses."""
        try:
            self._post_config({"allow_still_s": max(0.0, min(float(seconds), 600.0))})
        except Exception:
            pass

    def debug_jpeg(self):
        if self.link is None:
            return None
        self.link.debug_want = time.monotonic()
        return self.link.debug

    def close(self):
        global _ACTIVE
        self.stop()
        for t in (self.tex_m, self.tex_f):
            if t is not None:
                try:
                    t.release()
                except Exception:
                    pass
        self.tex_m = self.tex_f = None
        if _ACTIVE is self:
            _ACTIVE = None


def debug_jpeg():
    """remote_control GET /camera-preview: the worker's mask + skeleton JPEG (asks for it; None until it exists)."""
    layer = _ACTIVE
    return layer.debug_jpeg() if layer is not None and layer.on else None


# ------------------------------------------------------------------ shader (spliced into SCENE_FS at //@CAMERA_GLSL)
SCENE_GLSL = """
uniform sampler2D camM, camF;       // crowd camera: silhouette mask (smoothed), motion field (uv / s, camera y down)
uniform mat3 camH;                  // screen uv -> camera uv (alignment: mirror, crop, position, scale, keystone)
uniform float camFlipX;
uniform float p_l_camera, p_crowd_mode, p_crowd_glow, p_crowd_hue, p_crowd_push;
vec2 camUV(vec2 ps, vec2 asp){ vec3 c = camH*vec3(ps/asp+0.5, 1.0); return c.z > 1e-4 ? c.xy/c.z : vec2(-1.0); }
float camIn(vec2 cu){ vec2 e = smoothstep(vec2(0.0), vec2(0.02), cu)*smoothstep(vec2(0.0), vec2(0.02), 1.0-cu); return e.x*e.y; }
float camMask(vec2 ps, vec2 asp){ vec2 cu = camUV(ps, asp); return texture(camM, cu).r*camIn(cu); }
vec2 camWarp(vec2 ps, vec2 asp){                   // flow push (any mode, + mode 5) and the inner swirl of mode 2
  float cm = floor(p_crowd_mode+0.5);
  float k = max(p_crowd_push, cm == 5.0 ? 1.0 : 0.0)*p_l_camera;
  vec2 cu = camUV(ps, asp), d = vec2(0.0);
  if (k > 0.0) { vec2 fl = texture(camF, cu).rg*camIn(cu); d -= vec2(fl.x*camFlipX, -fl.y)*k*0.10; }
  if (cm == 2.0) { float m = texture(camM, cu).r*camIn(cu);
    d += m*p_l_camera*0.06*vec2(sin(t*0.7+ps.y*6.0), cos(t*0.6+ps.x*6.0)); }
  return d;
}
float camEdge(vec2 ps, vec2 asp){
  vec2 cu = camUV(ps, asp), e = 1.5/vec2(textureSize(camM, 0));
  float a = texture(camM, cu+vec2(e.x,0)).r, b = texture(camM, cu-vec2(e.x,0)).r;
  float c = texture(camM, cu+vec2(0,e.y)).r, d = texture(camM, cu-vec2(0,e.y)).r;
  return clamp(length(vec2(a-b, c-d))*2.2, 0.0, 1.0)*camIn(cu);
}
vec3 camTint(vec2 ps){ return hueRot(palette(t*0.05+0.25*length(ps)), p_crowd_hue*TAU); }
void camScene(inout vec3 col, inout float cov, vec2 ps, vec2 asp){   // before the echo: what it does trails
  float cm = floor(p_crowd_mode+0.5), m = camMask(ps, asp), a = p_l_camera;
  if (cm == 2.0) {                                  // patterns inside: bodies carry a hue-turned, brighter pattern
    vec3 inner = hueRot(col, (0.33+p_crowd_hue)*TAU)*(1.0+0.6*p_crowd_glow) + camTint(ps)*0.10*p_crowd_glow;
    col = mix(col, inner, m*a) * (1.0 - 0.55*a*(1.0-m));
  } else if (cm == 4.0) {                           // motion trails: the silhouette glows into the echo buffer
    vec3 g = camTint(ps)*(0.25*m + 0.9*camEdge(ps, asp))*p_crowd_glow*a;
    col += g; cov = max(cov, max3(g));
  }
}
void camShow(inout vec3 o, inout float cov, vec2 ps, vec2 asp, float edge){   // after the echo, before the limiter
  float cm = floor(p_crowd_mode+0.5), a = p_l_camera;
  if (cm == 0.0) { float m = camMask(ps, asp); o *= mix(1.0, m, a); cov *= mix(1.0, m, a); }
  else if (cm == 3.0) { float m = camMask(ps, asp); o *= 1.0 - a*m; }
  else if (cm == 1.0) { vec3 g = camTint(ps)*camEdge(ps, asp)*(0.35+1.1*p_crowd_glow)*a*edge; o += g; cov = max(cov, max3(g));
                        o /= max(1.0, max3(o)); }
}
"""
NULL_GLSL = ("uniform float p_l_camera; vec2 camWarp(vec2 a, vec2 b){ return vec2(0.0); } "
             "void camScene(inout vec3 c, inout float v, vec2 a, vec2 b){} void camShow(inout vec3 o, inout float v, vec2 a, vec2 b, float e){}")
