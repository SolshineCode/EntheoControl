"""Crowd camera worker: webcam -> person silhouettes, motion / optical flow, dancer pose. Its own process.

  python camera_worker.py --port 8767 --device 0        # normally started (and stopped) by camera_layer.CameraLayer

Why a separate process: a webcam driver hang or a MediaPipe crash can never take down the show, and the work runs at
BELOW_NORMAL priority on the CPU (no GPU at all). The engine starts it when the performer switches CAMERA ON and
stops it on CAMERA OFF (the device is released when the process ends; it also exits on its own when nobody has asked
for a frame for IDLE_EXIT_S, or when its parent engine is gone).

Privacy: camera frames live only in this process's memory. Nothing is written to disk and nothing leaves 127.0.0.1.
What the engine receives is a 160x90 silhouette mask, a 32x18 motion field, a few numbers and up to MAX_POSES x 11
body points. The optional debug picture (/debug.jpg) shows the mask + skeleton, never the camera image.

HTTP (127.0.0.1 only, header X-Token = contents of .remote_token, the same token as the panel / AI worker):
  GET  /frame?after=N   newest result with seq > N (waits up to 0.5 s), binary, see pack(); 204 if none yet
  GET  /status          JSON: fps, per-stage ms, device, backend, error
  GET  /debug.jpg       320x180 JPEG of mask + skeleton (rendered only while it is being asked for, <= 3 fps)
  POST /config          {fps 5..30, res 0..2, pose_every 1..6, max_poses 0..3, lock_exposure bool}
  GET  /still?skip=N    AUTO MAP only: the N-th camera frame grabbed after the request, grey 640x360 (header STL1 + w, h),
                        so a calibration pattern is captured after it is on the wall. Kept in the engine's memory only.
  POST /quit
"""
import argparse
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "camera"
SEG_MODEL = "selfie_segmenter_landscape.tflite"   # 256x144 input: matches a 16:9 webcam
POSE_MODEL = "pose_landmarker_lite.task"
DEFAULT_PORT = 8767
MASK_W, MASK_H = 160, 90          # silhouette sent to the engine
FLOW_W, FLOW_H = 32, 18           # motion field sent to the engine
FLOW_CALC = (160, 90)             # optical flow is computed at this size
RES_STEPS = [(480, 270), (320, 180), (240, 135)]   # inference size (res 0..2); capture is 640x360
MAX_POSES = 3
IDLE_EXIT_S = 15.0
MOTION_GAIN = 6.0                 # energy = 1 - exp(-gain * mean |flow| in frame widths per second)
MAGIC = b"CAM1"
HDR = struct.Struct("<4sIdHHHHI")  # magic, seq, t, mask w, mask h, flow w, flow h, json length
STILL = struct.Struct("<4sHH")     # b"STL1", width, height, then w*h grey bytes
# kept body points (MediaPipe pose landmark index); x, y in camera uv (0..1, y DOWN, not mirrored), visibility 0..1
KEEP = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
        "l_hip": 23, "r_hip": 24, "l_ankle": 27, "r_ankle": 28}
BONES = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
         ("l_hip", "l_ankle"), ("r_hip", "r_ankle")]
IR_NAME = re.compile(r"(\bIR\b|infra ?red|depth)", re.I)


# ------------------------------------------------------------------ wire format (pure numpy; the engine imports this)

def pack(seq, t, mask, flow, info):
    """mask (MASK_H, MASK_W) uint8, flow (FLOW_H, FLOW_W, 2) float (uv per second, y down), info JSON-able dict."""
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    flow = np.ascontiguousarray(flow, dtype=np.float16)
    js = json.dumps(info, separators=(",", ":"), allow_nan=False).encode()
    return b"".join((HDR.pack(MAGIC, int(seq) & 0xFFFFFFFF, float(t), mask.shape[1], mask.shape[0], flow.shape[1],
                              flow.shape[0], len(js)), mask.tobytes(), flow.tobytes(), js))


def unpack(buf):
    """Inverse of pack(); ValueError on anything malformed (sizes are checked, JSON must be an object)."""
    if len(buf) < HDR.size:
        raise ValueError("short frame")
    magic, seq, t, mw, mh, fw, fh, jl = HDR.unpack_from(buf, 0)
    if magic != MAGIC or not (0 < mw <= 1024 and 0 < mh <= 1024 and 0 < fw <= 256 and 0 < fh <= 256):
        raise ValueError("bad header")
    o, nm, nf = HDR.size, mw * mh, fw * fh * 2 * 2
    if len(buf) != o + nm + nf + jl:
        raise ValueError("bad length")
    mask = np.frombuffer(buf, np.uint8, nm, o).reshape(mh, mw)
    flow = np.frombuffer(buf, np.float16, fw * fh * 2, o + nm).reshape(fh, fw, 2).astype(np.float32)
    info = json.loads(bytes(buf[o + nm + nf:]).decode())
    if not isinstance(info, dict):
        raise ValueError("bad info")
    return dict(seq=seq, t=t, mask=mask, flow=flow, info=info)


# ------------------------------------------------------------------ analysis (pure numpy; cv2 only for the flow)

def energies(flow_uv, mask=None):
    """Crowd energy from a motion field in uv per second (any size, (h, w, 2)): overall, left / centre / right
    thirds (camera image, not mirrored), and presence (fraction of the frame covered by bodies)."""
    mag = np.hypot(flow_uv[..., 0], flow_uv[..., 1])
    w = mag.shape[1]
    thirds = [mag[:, :w // 3], mag[:, w // 3: w - w // 3], mag[:, w - w // 3:]]
    e = lambda m: float(1.0 - math.exp(-MOTION_GAIN * float(m.mean()))) if m.size else 0.0
    pres = float((np.asarray(mask) > 127).mean()) if mask is not None else 0.0
    return dict(motion=e(mag), left=e(thirds[0]), centre=e(thirds[1]), right=e(thirds[2]), presence=pres)


def pool(a, shape):
    """Average-pool a (h, w[, c]) array to shape (H, W) (exact block means when sizes divide; else nearest blocks)."""
    a = np.asarray(a, dtype=np.float32)
    H, W = shape
    h, w = a.shape[:2]
    ys = (np.arange(H + 1) * h) // H
    xs = (np.arange(W + 1) * w) // W
    s = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2)), 0), 1)
    tot = s[ys[1:]][:, xs[1:]] - s[ys[:-1]][:, xs[1:]] - s[ys[1:]][:, xs[:-1]] + s[ys[:-1]][:, xs[:-1]]
    area = ((ys[1:] - ys[:-1])[:, None] * (xs[1:] - xs[:-1])[None, :]).astype(np.float32)
    return tot / (area[..., None] if a.ndim == 3 else area)


def pose_points(landmarks):
    """One MediaPipe pose (list of 33 landmarks with .x .y .visibility) -> {name: [x, y, vis]} rounded."""
    out = {}
    for name, i in KEEP.items():
        lm = landmarks[i]
        vis = getattr(lm, "visibility", None)
        vis = 1.0 if vis is None else float(vis)
        x, y = float(lm.x), float(lm.y)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(vis):
            out[name] = [round(x, 4), round(y, 4), round(vis, 3)]
    return out


def pack_still(gray):
    g = np.ascontiguousarray(gray, np.uint8)
    return STILL.pack(b"STL1", g.shape[1], g.shape[0]) + g.tobytes()


def unpack_still(buf):
    if len(buf) < STILL.size:
        raise ValueError("short still")
    m, w, h = STILL.unpack_from(buf, 0)
    if m != b"STL1" or len(buf) != STILL.size + w * h:
        raise ValueError("bad still")
    return np.frombuffer(buf, np.uint8, w * h, STILL.size).reshape(h, w)


def debug_picture(mask, poses, size=(320, 180)):
    """Mask (grey) + skeleton (colour) as a BGR uint8 image. Never contains camera pixels."""
    import cv2
    img = cv2.resize(np.asarray(mask, np.uint8), size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor((img * 0.6).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    W, H = size
    for k, p in enumerate(poses):
        col = [(80, 255, 200), (255, 160, 60), (200, 90, 255)][k % 3]
        pts = {n: (int(v[0] * W), int(v[1] * H)) for n, v in p.items() if v[2] > 0.4}
        for a, b in BONES:
            if a in pts and b in pts:
                cv2.line(img, pts[a], pts[b], col, 2)
        for n, q in pts.items():
            cv2.circle(img, q, 5 if "wrist" in n else 3, (255, 255, 255) if "wrist" in n else col, -1)
    return img


# ------------------------------------------------------------------ devices

def list_devices(timeout=6.0):
    """[{index, name, ir}] in DirectShow order (= the cv2.CAP_DSHOW index). Uses ffmpeg's device list (hidden
    window); without ffmpeg, a single "Camera 0". IR / depth cameras are flagged (the panel hides them by default)."""
    names = []
    try:
        flags = 0x08000000 if os.name == "nt" else 0              # CREATE_NO_WINDOW
        r = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                           capture_output=True, text=True, timeout=timeout, creationflags=flags, stdin=subprocess.DEVNULL)
        for line in (r.stderr or "").splitlines():
            m = re.search(r'"([^"]+)"\s*\(video\)', line)
            if m:
                names.append(m.group(1))
    except Exception:
        names = []
    if not names:
        names = ["Camera 0"]
    return [dict(index=i, name=n[:80], ir=bool(IR_NAME.search(n))) for i, n in enumerate(names)]


# ------------------------------------------------------------------ the worker

class Worker:
    def __init__(self, token, port=DEFAULT_PORT, device=0, fps=30, res=1, pose_every=2, max_poses=2, parent=None,
                 capture_factory=None, models=None, clock=time.monotonic):
        self.token, self.port, self.device, self.parent = token, int(port), int(device), parent
        self.cfg = dict(fps=int(fps), res=int(res), pose_every=int(pose_every), max_poses=int(max_poses))
        self._capture_factory = capture_factory
        self._models = models                                     # tests: (segment fn, pose fn) or None = MediaPipe
        self.clock = clock
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest = None                                       # (seq, bytes)
        self._frame_in = None                                     # (t, bgr) from the grab thread
        self._frame_seq = 0
        self._stop = threading.Event()
        self.last_req = clock()
        self.debug_want = 0.0
        self.debug_jpg = None
        self.stats = dict(fps=0.0, grab_fps=0.0, ms={}, backend="", error="", frames=0, device=self.device)
        self.server = None
        self.cap = None
        self.still_until = 0.0                                    # /still is refused unless AUTO MAP opened it
        self._exposure = None                                     # pending (lock: bool, done: Event) request

    # ---- capture (its own thread: cap.read() blocks at the camera's rate, which drops in a dark room)
    def _open(self):
        if self._capture_factory is not None:
            return self._capture_factory(self.device)
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        hd = os.environ.get("CAMERA_HD", "").strip().lower() in ("1", "true", "yes")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 if hd else 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 if hd else 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.device} could not be opened (in use by another app, or unplugged?)")
        return cap

    def _grab_loop(self, cap):
        n, t0 = 0, self.clock()
        fails = 0
        while not self._stop.is_set():
            with self._lock:
                exp_req, self._exposure = self._exposure, None
            if exp_req is not None:                              # AUTO MAP: freeze exposure / white balance if the driver can
                lock, done = exp_req
                locked = False
                try:
                    import cv2
                    if lock:
                        cur_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
                        ok_ae = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) or cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                        if math.isfinite(cur_exp) and cur_exp != 0.0:
                            ok_ae = cap.set(cv2.CAP_PROP_EXPOSURE, cur_exp) or ok_ae
                        locked = bool(ok_ae)
                    else:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) or cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0 if lock else 1)
                except Exception:
                    pass
                self.stats["exposure_locked"] = locked
                done.set()
            ok, fr = cap.read()
            if not ok or fr is None:
                fails += 1
                if fails > 50:
                    self.stats["error"] = "camera stopped delivering frames"
                    self._stop.set()
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self._frame_in = (self.clock(), fr)
                self._frame_seq += 1
            n += 1
            now = self.clock()
            if now - t0 >= 2.0:
                self.stats["grab_fps"] = round(n / (now - t0), 1)
                n, t0 = 0, now

    # ---- models
    def _load_models(self):
        if self._models is not None:
            self.stats["backend"] = "test"
            return self._models
        import cv2
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_DIR / SEG_MODEL)),
                running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))
            pose = None
            if (MODEL_DIR / POSE_MODEL).is_file():
                pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / POSE_MODEL)),
                    running_mode=vision.RunningMode.VIDEO, num_poses=MAX_POSES,
                    min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))

            def segment(rgb, ts):
                r = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                return r.confidence_masks[0].numpy_view().reshape(rgb.shape[0], rgb.shape[1])

            def poses(rgb, ts, n):
                if pose is None or n <= 0:
                    return []
                r = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                return [pose_points(lms) for lms in r.pose_landmarks[:n]]
            self.stats["backend"] = "mediapipe" + ("" if pose is not None else " (no pose model)")
            return segment, poses
        except Exception as ex:                                  # no MediaPipe: moving things are "bodies", no pose
            sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
            self.stats["backend"] = f"motion fallback (MediaPipe unavailable: {str(ex)[:80]})"

            def segment(rgb, ts):
                return sub.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).astype(np.float32) / 255.0
            return segment, (lambda rgb, ts, n: [])

    # ---- one analysis step (also what the tests drive with fake frames)
    def analyse(self, bgr, ts_ms, dt, segment, poses_fn, do_pose, last_poses, prev_gray):
        import cv2
        W, H = RES_STEPS[min(max(self.cfg["res"], 0), len(RES_STEPS) - 1)]
        tm = {}
        a = time.perf_counter()
        small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        conf = np.asarray(segment(rgb, ts_ms), dtype=np.float32)
        b = time.perf_counter(); tm["seg"] = (b - a) * 1000
        mask = cv2.resize(conf, (MASK_W, MASK_H), interpolation=cv2.INTER_AREA)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)                 # feathered edges
        mask8 = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(cv2.resize(small, FLOW_CALC, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and dt > 0:
            fl = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            fl[..., 0] /= FLOW_CALC[0] * dt                       # px / frame -> uv / second
            fl[..., 1] /= FLOW_CALC[1] * dt
            np.clip(fl, -4.0, 4.0, out=fl)
        else:
            fl = np.zeros((FLOW_CALC[1], FLOW_CALC[0], 2), np.float32)
        c = time.perf_counter(); tm["flow"] = (c - b) * 1000
        ps = poses_fn(rgb, ts_ms, self.cfg["max_poses"]) if do_pose else last_poses
        tm["pose"] = (time.perf_counter() - c) * 1000 if do_pose else 0.0
        info = energies(fl, mask8)
        info["poses"] = ps
        return mask8, pool(fl, (FLOW_H, FLOW_W)), info, gray, tm

    def _process_loop(self):
        try:
            segment, poses_fn = self._load_models()
            cap = self._open()
        except Exception as ex:
            self.stats["error"] = str(ex)[:200]
            print(f"[camera] {ex}", flush=True)
            self._stop.set()
            return
        grab = threading.Thread(target=self._grab_loop, args=(cap,), name="cam-grab", daemon=True)
        grab.start()
        seq, seen, prev_gray, last_t, poses, k = 0, 0, None, None, [], 0
        t_start = self.clock()
        n_done, rate_t0 = 0, self.clock()
        ema = {}
        try:
            while not self._stop.is_set():
                with self._lock:
                    item, fs = self._frame_in, self._frame_seq
                if item is None or fs == seen:
                    time.sleep(0.004)
                    continue
                seen = fs
                t, bgr = item
                period = 1.0 / max(1, self.cfg["fps"])
                if last_t is not None and t - last_t < period * 0.9:
                    continue
                dt = (t - last_t) if last_t is not None else 0.0
                last_t = t
                ts_ms = int((t - t_start) * 1000) + seq                # strictly increasing (MediaPipe VIDEO mode)
                pe = max(1, self.cfg["pose_every"])
                do_pose = self.cfg["max_poses"] > 0 and k % pe == 0
                k += 1
                try:
                    mask8, flow, info, prev_gray, tm = self.analyse(bgr, ts_ms, dt, segment, poses_fn, do_pose, poses,
                                                                    prev_gray)
                except Exception as ex:                          # one bad frame / model hiccup: skip it, keep running
                    self.stats["error_last"] = f"{type(ex).__name__}: {ex}"[:160]
                    continue
                if do_pose:
                    poses = info["poses"]
                if self.cfg["max_poses"] <= 0:
                    poses = info["poses"] = []
                for key, v in tm.items():
                    if key != "pose" or do_pose:
                        ema[key] = v if key not in ema else ema[key] * 0.9 + v * 0.1
                seq += 1
                info["fps"] = self.stats["fps"]
                data = pack(seq, t, mask8, flow, info)
                with self._cond:
                    self._latest = (seq, data)
                    self._cond.notify_all()
                n_done += 1
                now = self.clock()
                if now - rate_t0 >= 2.0:
                    self.stats["fps"] = round(n_done / (now - rate_t0), 1)
                    n_done, rate_t0 = 0, now
                self.stats["ms"] = {key: round(v, 2) for key, v in ema.items()}
                self.stats["frames"] = seq
                if now - self.debug_want < 3.0 and (self.debug_jpg is None or now - self.debug_jpg[0] > 0.33):
                    import cv2
                    ok, jpg = cv2.imencode(".jpg", debug_picture(mask8, poses), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        self.debug_jpg = (now, jpg.tobytes())
        finally:
            self._stop.set()
            grab.join(timeout=2.0)
            try:
                cap.release()                                    # the device is free again
            except Exception:
                pass

    # ---- HTTP
    def latest_after(self, after, wait=0.5):
        with self._cond:
            if self._latest is None or self._latest[0] <= after:
                self._cond.wait(timeout=wait)
            if self._latest is not None and self._latest[0] > after:
                return self._latest[1]
            return None

    def still(self, skip=2, timeout=4.0):
        """Grey 640x360 of the skip-th frame grabbed after now (packed), or None on timeout."""
        import cv2
        with self._lock:
            want = self._frame_seq + skip
        t0 = self.clock()
        while self.clock() - t0 < timeout and not self._stop.is_set():
            with self._lock:
                item, fs = self._frame_in, self._frame_seq
            if item is not None and fs >= want:
                g = cv2.cvtColor(item[1], cv2.COLOR_BGR2GRAY)
                if g.shape[1] > 1280:                            # preserve native 640x360 or 1280x720; only cap >720p
                    nh = max(1, int(round(g.shape[0] * 1280.0 / g.shape[1])))
                    g = cv2.resize(g, (1280, nh), interpolation=cv2.INTER_AREA)
                return pack_still(g)
            time.sleep(0.01)
        return None

    def configure(self, body):
        if not isinstance(body, dict):
            raise ValueError("config must be an object")
        lim = dict(fps=(5, 30), res=(0, len(RES_STEPS) - 1), pose_every=(1, 6), max_poses=(0, MAX_POSES))
        new = dict(self.cfg)
        exp_done = None
        if isinstance(body.get("lock_exposure"), bool):
            exp_done = threading.Event()
            with self._lock:
                self._exposure = (body["lock_exposure"], exp_done)
        a = body.get("allow_still_s")
        if isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a):
            self.still_until = self.clock() + min(max(float(a), 0.0), 600.0) if a > 0 else 0.0
        for k, (lo, hi) in lim.items():
            if k in body:
                v = body[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f"'{k}' must be a number")
                new[k] = int(min(max(round(v), lo), hi))
        self.cfg = new
        if exp_done is not None:
            exp_done.wait(timeout=1.0)
        return new

    def start_server(self):
        worker = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 10

            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="application/json"):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                if not secrets.compare_digest(self.headers.get("X-Token", "").encode(), worker.token.encode()):
                    self._send(403, {"ok": False, "error": "bad token"})
                    return False
                worker.last_req = worker.clock()
                return True

            def do_GET(self):
                if not self._auth():
                    return
                parts = urlsplit(self.path)
                if parts.path == "/frame":
                    try:
                        after = int(parse_qs(parts.query).get("after", ["0"])[0])
                    except ValueError:
                        after = 0
                    data = worker.latest_after(after)
                    self._send(200, data, "application/octet-stream") if data else self._send(204)
                elif parts.path == "/status":
                    self._send(200, dict(worker.stats, ok=not worker.stats["error"], cfg=worker.cfg))
                elif parts.path == "/still":
                    if worker.clock() > worker.still_until:          # only while AUTO MAP is capturing
                        self._send(403, {"ok": False, "error": "stills are off"})
                        return
                    try:
                        skip = min(max(int(parse_qs(parts.query).get("skip", ["2"])[0]), 1), 30)
                    except ValueError:
                        skip = 2
                    data = worker.still(skip)
                    self._send(200, data, "application/octet-stream") if data else self._send(504)
                elif parts.path == "/debug.jpg":
                    worker.debug_want = worker.clock()
                    d = worker.debug_jpg
                    self._send(200, d[1], "image/jpeg") if d else self._send(204)
                else:
                    self._send(404, {"ok": False})

            def do_POST(self):
                if not self._auth():
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = -1
                if not 0 <= n <= 4096:
                    self._send(413, {"ok": False})
                    return
                raw = self.rfile.read(n) if n else b"{}"
                if self.path == "/config":
                    try:
                        self._send(200, {"ok": True, "cfg": worker.configure(json.loads(raw or b"{}"))})
                    except ValueError as ex:
                        self._send(400, {"ok": False, "error": str(ex)})
                elif self.path == "/quit":
                    self._send(200, {"ok": True})
                    worker._stop.set()
                else:
                    self._send(404, {"ok": False})

        self.server = ExclusiveServer(("127.0.0.1", self.port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name="cam-http", daemon=True).start()

    def parent_alive(self):
        if not self.parent:
            return True
        try:
            import psutil
            return psutil.pid_exists(int(self.parent))
        except Exception:
            return True

    def run(self):
        self.start_server()
        proc = threading.Thread(target=self._process_loop, name="cam-proc", daemon=True)
        proc.start()
        print(f"[camera] worker on 127.0.0.1:{self.port}, device {self.device}", flush=True)
        try:
            while not self._stop.wait(1.0):
                if self.clock() - self.last_req > IDLE_EXIT_S:
                    print("[camera] nobody is reading frames: exiting", flush=True)
                    break
                if not self.parent_alive():
                    print("[camera] engine is gone: exiting", flush=True)
                    break
        finally:
            self._stop.set()
            proc.join(timeout=3.0)
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
        return 1 if self.stats["error"] else 0


class ExclusiveServer(ThreadingHTTPServer):
    """No port sharing: on Windows SO_REUSEADDR lets a second server bind the same port, so it is off and the port is
    claimed exclusively (a stale worker on the port makes this one fail loudly instead of splitting traffic)."""
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        import socket
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def lower_priority():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass
    else:
        try:
            os.nice(5)
        except Exception:
            pass


def read_token(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, default=1)
    ap.add_argument("--pose-every", type=int, default=2)
    ap.add_argument("--max-poses", type=int, default=2)
    ap.add_argument("--parent", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print the video devices and exit")
    a = ap.parse_args()
    if a.list:
        print(json.dumps(list_devices(), indent=1))
        return 0
    lower_priority()
    try:
        import cv2
        cv2.setNumThreads(2)
    except ImportError:
        print("[camera] opencv-python is not installed in this venv", flush=True)
        return 3
    token = os.environ.pop("CAMERA_WORKER_TOKEN", "") or read_token(HERE / ".remote_token")   # per-spawn secret first
    if not token:
        print("[camera] no token (CAMERA_WORKER_TOKEN / .remote_token)", flush=True)
        return 4
    w = Worker(token, a.port, a.device, a.fps, a.res, a.pose_every, a.max_poses, a.parent or None)
    return w.run()


if __name__ == "__main__":
    sys.exit(main())
