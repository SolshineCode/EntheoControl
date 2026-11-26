"""Control panel server for live_visuals.py: one token-protected web page served over the LAN.

Stdlib only (http.server + threading + json; Pillow only for art uploads). The server thread never
touches GL: commands are validated, pushed onto a thread-safe queue, and the render loop drains them
once per frame. State flows the other way through publish_state(), which the page polls.

  rc = RemoteControl(port=8765)
  rc.toggle()                       # start (prints URL + QR if `qrcode` is installed) / stop
  for each frame:
      rest = controller.apply(rc.drain())                    # control.Controller takes what it owns
      actions = apply_commands(rest, st, corners, ...)       # corner pin / grid / next / save
      rc.publish_state({..., "ctl": controller.state()})     # a few times per second

HTTP API (every request needs the token: `?k=<token>` or header `X-Token`):
  GET  /                   the control panel (panel.html, built from params.PARAMS)
  GET  /state              last published state as JSON
  GET  /params             control.meta(): params, groups, options, mod sources
  POST /cmd                one command object or a list of them, e.g. {"cmd":"set","name":"zoom","v":0.4}
                           mapping: {"cmd":"surf","op":"corner","id":1,"i":0,"x":64,"y":40} (ops: see surfaces.py)
  GET  /tex/<name>         one image from textures/ (AI results)
  GET  /art                art library listing, in art_index order
  GET  /art-thumb/<name>   small cached JPEG of one art piece
  POST /upload             raw image body (png / jpeg / webp, <= 25 MB), header X-Filename -> art/<name>.png
  POST /art-delete         {"name": ...}

Standalone demo without GL (real Controller, fake audio):  python remote_control.py --port 8799
"""
import hmac
import io
import json
import math
import os
import queue
import re
import secrets
import socket
import socketserver
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import control
import governor
import recorder
import surfaces

HERE = Path(__file__).resolve().parent
MAX_BODY = 16384       # bytes; a 60 Hz fader batch is a few hundred bytes, an AI prompt up to ~1.5 kB
MAX_BATCH = 64         # commands per POST
MAX_UPLOAD = 25 << 20  # bytes per uploaded art image
MAX_ART_SIDE = 4096
MAX_ART_PIXELS = 40_000_000   # decoded size guard (decompression bombs, and RAM next to a running show)
MAX_CONNS = 32                # simultaneous client connections
UPLOAD_WAIT = 20.0            # s a second upload waits for the one in progress before it gets a 503
_UPLOAD_GATE = threading.BoundedSemaphore(1)   # one image is read + decoded at a time
THUMB_SIDE = 256
# layer panel rows: every l_* layer plus serpents (l_swarm lives inside the l_land texture)
LAYER_KEYS = [n for n in control.PARAMS if (n.startswith("l_") and n != "l_swarm") or n == "serpent"]
# thumbnail schedule: layers drawn inside the scene shader refresh every round, offscreen-pass layers every third
_FAST = [n for n in LAYER_KEYS if n in ("l_kene", "l_beads", "l_lines", "l_ai", "l_video", "serpent")]
_SLOW = [n for n in LAYER_KEYS if n not in _FAST]
THUMB_ORDER = (_FAST + _SLOW + _FAST + _FAST) or list(LAYER_KEYS)
TEX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(png|jpg|jpeg|webp)$", re.I)
IMG_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
MAX_NUDGE = 0.1        # largest relative corner step per command (UV units)
MAX_FB = 0.98
MAX_MODE = 4        # layer modes: beads / ai / both / lines / all
MAX_PALETTE = 15    # upper bound for validation; apply_commands clamps to the real palette count
UNIT_SQ = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]  # BL, BR, TR, TL (origin bottom-left)


# ------------------------------------------------------------------ validation

def _int(d, key, lo, hi):
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"'{key}' must be an integer in {lo}..{hi}")
    return v


def _num(d, key, lo, hi):
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"'{key}' must be a finite number")
    return min(hi, max(lo, float(v)))


def _bool(d, key):
    v = d.get(key)
    if not isinstance(v, bool):
        raise ValueError(f"'{key}' must be true or false")
    return v


def validate_command(c):
    """Return a normalized copy of one command (values clamped, unknown keys dropped).

    Raises ValueError for anything malformed. Runs on the server thread, so nothing
    unvalidated ever reaches the render loop.
    """
    if isinstance(c, dict) and "cmd" not in c and isinstance(c.get("type"), str):
        c = dict(c, cmd=c["type"])                   # {"type": ...} is accepted as an alias of {"cmd": ...}
    if not isinstance(c, dict) or not isinstance(c.get("cmd"), str):
        raise ValueError("command must be an object with a string 'cmd'")
    cmd = c["cmd"]
    if cmd in control.CONTROL_CMDS:                  # performance commands: validated by the control system
        return control.validate(c)
    if cmd in ("next", "save", "reset"):
        return {"cmd": cmd}
    if cmd == "gov":                                 # resource governor (governor.py): validated there
        return governor.validate_command(c)
    if cmd == "rec":                                 # show recorder (recorder.py): validated there
        return recorder.validate_cmd(c)
    if cmd == "next_video":                          # passed through to the render engine
        return control.passthrough(cmd)
    if cmd == "surf":                                # multi-surface mapping: {"cmd": "surf", "op": ...} (surfaces.py)
        return surfaces.validate_cmd(c)
    if cmd == "art_pick":                            # source picker -> engine (live_visuals.art_pick): choose by name
        out = {"source": c.get("source", "art")}
        if out["source"] not in ("art", "ai"):
            raise ValueError("'source' must be \"art\" or \"ai\"")
        if c.get("name") not in (None, ""):
            name = c["name"]
            # art files are hand-dropped too (spaces, unicode): the engine only ever matches the name against its own
            # library listing, so anything that cannot be a path is acceptable here
            if (not isinstance(name, str) or len(name) > 200 or name in (".", "..") or any(ch in name for ch in '/\\:\0')
                    or any(ord(ch) < 32 for ch in name)):
                raise ValueError("'name' must be an image file name")
            if out["source"] == "ai" and not TEX_NAME.match(name):
                raise ValueError("'name' must be a texture file name")
            out["name"] = name
        if "use" in c and c["use"] is not None:
            out["use"] = _int(c, "use", 0, 15)
        return control.passthrough(cmd, **out)
    if cmd == "show_texture":                        # source picker: cross-fade to this file in textures/
        name = c.get("name")
        if not isinstance(name, str) or not TEX_NAME.match(name) or ".." in name:
            raise ValueError("'name' must be a texture file name")
        return control.passthrough(cmd, name=name)
    if cmd == "fold":
        return {"cmd": cmd, "i": _int(c, "i", 0, 63)}
    if cmd == "mode":
        return {"cmd": cmd, "i": _int(c, "i", 0, MAX_MODE)}
    if cmd == "palette":
        return {"cmd": cmd, "i": _int(c, "i", 0, MAX_PALETTE)}
    if cmd in ("fb", "grid", "calib"):
        return {"cmd": cmd, "on": _bool(c, "on")}
    if cmd == "fbAmt":
        return {"cmd": cmd, "v": _num(c, "v", 0.0, MAX_FB)}
    if cmd == "corner":
        return {"cmd": cmd, "i": _int(c, "i", 0, 3), "u": _num(c, "u", 0.0, 1.0), "v": _num(c, "v", 0.0, 1.0)}
    if cmd == "nudge":
        return {"cmd": cmd, "i": _int(c, "i", 0, 3),
                "du": _num(c, "du", -MAX_NUDGE, MAX_NUDGE), "dv": _num(c, "dv", -MAX_NUDGE, MAX_NUDGE)}
    raise ValueError(f"unknown cmd '{cmd[:32]}'")


def quad_ok(pts, eps=1e-3):
    """True if the 4 corners form a strictly convex quad (either winding).

    A collinear / bow-tie / collapsed quad gives a singular homography, which would
    crash np.linalg.inv in the render loop, so such corner moves are refused.
    """
    cross = []
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    return all(z > eps for z in cross) or all(z < -eps for z in cross)


def apply_commands(cmds, st, corners, n_folds=9, n_palettes=MAX_PALETTE + 1):
    """Apply drained commands to the app's `st` dict and `corners` list (in place).

    Call from the GL/main thread only. Returns a set of actions the caller must
    perform itself; currently only {"save"} (write mapping.json).
    """
    actions = set()
    for c in cmds:
        cmd = c["cmd"]
        if cmd == "fold":
            if 0 <= c["i"] < n_folds:
                st["fold_i"] = c["i"]
        elif cmd == "mode":
            st["mode"] = c["i"]
        elif cmd == "palette":
            st["palette"] = min(max(c["i"], 0), max(n_palettes, 1) - 1)
        elif cmd == "fb":
            st["fb"] = c["on"]
        elif cmd == "fbAmt":
            st["fbAmt"] = c["v"]
        elif cmd == "next":
            st["next"] = True
        elif cmd == "grid":
            st["grid"] = c["on"]
        elif cmd == "calib":  # same as the C key: calibrate implies the grid overlay
            st["calib"] = st["grid"] = c["on"]
        elif cmd in ("corner", "nudge"):
            i = c["i"]
            if cmd == "corner":
                new = [c["u"], c["v"]]
            else:
                new = [min(1.0, max(0.0, corners[i][0] + c["du"])), min(1.0, max(0.0, corners[i][1] + c["dv"]))]
            trial = [list(p) for p in corners]
            trial[i] = new
            if quad_ok(trial):
                corners[i] = new
        elif cmd == "reset":
            corners[:] = [list(p) for p in UNIT_SQ]
        elif cmd == "save":
            actions.add("save")
    return actions


# ------------------------------------------------------------------ art library (uploads)

def _sniff(data):
    """Image type by magic bytes (the client's filename and Content-Type are never trusted)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def save_art(data, client_name, art_dir):
    """Validate an uploaded image and store it as art/<safe>_<random>.png. Returns the new file name.

    Accepts png / jpeg / webp by magic bytes, decode-checks with PIL, caps the longest side at
    MAX_ART_SIDE, re-encodes to PNG (so nothing but pixels survives) and writes atomically.
    Raises ValueError for anything else.
    """
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError("image must be 1 byte .. 25 MB")
    kind = _sniff(data)
    if kind is None:
        raise ValueError("only PNG, JPEG or WebP images are accepted")
    try:
        from PIL import Image
    except Exception:
        raise ValueError("Pillow is not installed on the projector laptop")
    try:
        with Image.open(io.BytesIO(data), formats=[kind.upper()]) as probe:
            w, h = probe.size
            if w < 1 or h < 1 or w * h > MAX_ART_PIXELS:
                raise ValueError("image dimensions are out of range")
            probe.verify()
        with Image.open(io.BytesIO(data), formats=[kind.upper()]) as im:
            if kind == "jpeg":                       # let the decoder shrink huge JPEGs while it decodes
                im.draft("RGB", (MAX_ART_SIDE, MAX_ART_SIDE))
            im.load()
            mode = "RGBA" if (im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info) else "RGB"
            big = max(im.size) > MAX_ART_SIDE
            if big and im.mode not in ("P", "PA", "1"):   # shrink first, so the conversion works on the small copy
                im.thumbnail((MAX_ART_SIDE, MAX_ART_SIDE), Image.LANCZOS)
                big = False
            im = im.convert(mode)
            if big:                                  # palette images cannot be resampled before conversion
                im.thumbnail((MAX_ART_SIDE, MAX_ART_SIDE), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
    except ValueError:
        raise
    except Exception as ex:                          # PIL raises many types for corrupt / hostile files
        raise ValueError(f"not a readable image ({type(ex).__name__})")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(str(client_name or "")).stem)[:40].strip("_-") or "art"
    art_dir = Path(art_dir)
    try:
        art_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()                            # sortable prefix: the newest upload is always last by name too
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
        for _ in range(8):
            name = f"{stamp}_{stem}_{secrets.token_hex(2)}.png"
            if not (art_dir / name).exists():
                break
        tmp = art_dir / f".{name}.{os.getpid()}.tmp"
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, art_dir / name)
    except OSError as ex:
        raise ValueError(f"could not save ({type(ex).__name__})")
    return name


def _depth_path(art_dir, name):
    return Path(art_dir) / (Path(name).stem + ".depth.png")


def art_listing(art_dir):
    """[{name, depth}] exactly as the engine lists them (art_layer.list_art); the position is the art_index."""
    return [dict(name=n, depth=_depth_path(art_dir, n).exists()) for n in control.list_art(art_dir)]


def art_thumb(art_dir, name):
    """Small cached JPEG for the library grid, or None. `name` must already be validated + listed."""
    try:
        from PIL import Image
        src = Path(art_dir) / name
        cache = Path(art_dir) / ".thumbs" / (name + ".jpg")
        if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
            return cache.read_bytes()
        with Image.open(src) as im:
            if im.size[0] * im.size[1] > MAX_ART_PIXELS:
                return None
            im = im.convert("RGB")
            im.thumbnail((THUMB_SIDE, THUMB_SIDE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
        try:
            cache.parent.mkdir(exist_ok=True)
            tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cache)
        except OSError:
            pass
        return data
    except Exception:
        return None


def tex_listing(tex_dir, cap=64):
    """[{name, ai, t}] for the source picker: image files in textures/, newest first (the engine keeps the same 64)."""
    out = []
    try:
        for f in Path(tex_dir).iterdir():
            if TEX_NAME.match(f.name) and f.is_file():
                out.append((f.stat().st_mtime, f.name))
    except OSError:
        return []
    out.sort(key=lambda x: (-x[0], x[1]))
    return [dict(name=n, ai=n.lower().startswith("gen_"), t=round(t, 1)) for t, n in out[:cap]]


_TEX_THUMBS = {}                # name -> (mtime, jpeg); kept in memory so nothing extra lands in textures/
_TEX_THUMBS_LOCK = threading.Lock()
_TEX_DECODE = threading.BoundedSemaphore(2)      # a picker opening fires 64 requests: never 64 decodes next to a show


def tex_thumb(tex_dir, name):
    """Small JPEG of one texture, or None. `name` must already match TEX_NAME."""
    try:
        from PIL import Image
        folder = Path(tex_dir)
        src = folder / name
        if not src.is_file() or src.resolve().parent != folder.resolve():
            return None
        mt = src.stat().st_mtime
        with _TEX_THUMBS_LOCK:
            hit = _TEX_THUMBS.get(name)
        if hit and hit[0] == mt:
            return hit[1]
        with _TEX_DECODE:
            with _TEX_THUMBS_LOCK:                       # someone else may have made it while we waited
                hit = _TEX_THUMBS.get(name)
            if hit and hit[0] == mt:
                return hit[1]
            with Image.open(src) as im:
                if im.size[0] * im.size[1] > MAX_ART_PIXELS:
                    return None
                im = im.convert("RGB")
                im.thumbnail((THUMB_SIDE, THUMB_SIDE))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=78)
            data = buf.getvalue()
        with _TEX_THUMBS_LOCK:
            if len(_TEX_THUMBS) > 160:
                _TEX_THUMBS.clear()
            _TEX_THUMBS[name] = (mt, data)
        return data
    except Exception:
        return None


def delete_art(art_dir, name):
    """Remove one library image plus its depth map and thumbnail. Only names from the listing are accepted.

    Returns the piece's index in the listing before it was removed, or -1 if the image itself could not
    be deleted (unknown name, or the file is locked); its depth map and thumbnail are then left alone.
    """
    names = control.list_art(art_dir)
    if not isinstance(name, str) or name not in names:
        return -1
    try:
        (Path(art_dir) / name).unlink()
    except OSError:
        return -1
    for f in (_depth_path(art_dir, name), Path(art_dir) / ".thumbs" / (name + ".jpg")):
        try:
            f.unlink()
        except OSError:
            pass
    return names.index(name)


def request_depth(name, url, token_paths, timeout=3.0):
    """Ask ai_worker.py for a depth map of art/<name> (fire and forget; the art layer falls back to luminance)."""
    def work():
        try:
            tok = next((t for t in (_read(p) for p in token_paths) if t), None)
            if not tok:
                return
            req = urllib.request.Request(url.rstrip("/") + "/depth", data=json.dumps({"name": name}).encode(),
                                         headers={"X-Token": tok, "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=timeout).close()
        except Exception:
            pass

    threading.Thread(target=work, name="depth-request", daemon=True).start()


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# ------------------------------------------------------------------ networking

def lan_ips():
    """All plausible LAN IPv4 addresses of this machine, most likely first."""
    found = []
    try:
        found += [ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass

    def rank(ip):
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        if ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31:
            return 2  # also where WSL / Hyper-V virtual adapters live
        return 3

    ips = sorted({ip for ip in found if not ip.startswith(("127.", "169.254.", "0."))}, key=lambda ip: (rank(ip), ip))
    return ips


def lan_ip():
    """The IPv4 address other devices on the Wi-Fi should use to reach us.

    A UDP socket "connected" to a public address sends no packets but makes the OS
    pick the default-route interface, which skips virtual adapters. With no default
    route (offline hotspot) fall back to listing the interfaces.
    """
    for probe in ("8.8.8.8", "192.168.0.1", "10.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            if not ip.startswith(("127.", "0.", "169.254.")):
                return ip
        except OSError:
            pass
        finally:
            s.close()
    ips = lan_ips()
    return ips[0] if ips else "127.0.0.1"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Windows: SO_REUSEADDR lets a second process bind the same port, and a closed
    # listener can be re-bound immediately without it. POSIX needs it for restarts.
    allow_reuse_address = sys.platform != "win32"
    request_queue_size = 32

    def __init__(self, addr, rc):
        self.rc = rc
        self.active = True
        self._conns = set()
        self._conn_lock = threading.Lock()
        super().__init__(addr, _Handler)

    def verify_request(self, request, client_address):
        with self._conn_lock:                        # refuse (close) connections beyond the cap
            return len(self._conns) < MAX_CONNS

    def server_bind(self):
        # skip HTTPServer.server_bind: its socket.getfqdn() reverse lookup can stall for seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "remote", self.server_address[1]

    def track(self, conn, add):
        with self._conn_lock:
            (self._conns.add if add else self._conns.discard)(conn)

    def drop_connections(self):
        with self._conn_lock:
            conns = list(self._conns)
        for c in conns:  # kick idle keep-alive clients so "off" really is off
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def handle_error(self, request, client_address):
        pass  # phones drop connections constantly; keep the console clean


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: no TCP handshake per 30 Hz drag update
    server_version = "remote"
    sys_version = ""
    timeout = 15

    def setup(self):
        super().setup()
        self.server.track(self.connection, True)

    def finish(self):
        try:
            super().finish()
        finally:
            self.server.track(self.connection, False)

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json", close=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _gate(self):
        """Common checks; returns the request path, or None if a response was already sent."""
        parts = urlsplit(self.path)
        if not self.server.active:
            self._send(503, {"ok": False, "error": "remote is off"}, close=True)
            return None
        given = parse_qs(parts.query).get("k", [""])[0] or self.headers.get("X-Token", "")
        if not hmac.compare_digest(given.encode("utf-8", "replace"), self.server.rc.token.encode()):
            self._send(403, {"ok": False, "error": "bad or missing token"}, close=True)
            return None
        return parts.path

    def do_GET(self):
        path = self._gate()
        if path is None:
            return
        if path in ("/", "/index.html"):
            page = _load_page().encode("utf-8") if self.server.rc.reload_page else PAGE_BYTES
            self._send(200, page, "text/html; charset=utf-8")
        elif path == "/state":
            self._send(200, self.server.rc._state_bytes())
        elif path == "/params":
            self._send(200, control.meta())
        elif path == "/art":
            self._send(200, {"ok": True, "art": art_listing(self.server.rc.art_dir)})
        elif path == "/sources":                     # source picker: uploaded art + AI / seed textures
            rc = self.server.rc
            self._send(200, {"ok": True, "art": art_listing(rc.art_dir), "tex": tex_listing(rc.tex_dir)})
        elif path == "/fonts":                       # text layer: installed fonts + the project's fonts/ folder
            try:
                import text_layer
                fonts = [dict(name=f["name"], user=f["user"]) for f in text_layer.list_fonts()]
                self._send(200, {"ok": True, "fonts": fonts, "default": (text_layer.resolve_font("") or {}).get("name", "")})
            except Exception as ex:
                self._send(200, {"ok": False, "fonts": [], "error": str(ex)[:200]})
        elif path.startswith("/font-preview/"):      # the font's name set in that font (small PNG, white on clear)
            try:
                import text_layer
                data = text_layer.font_preview_png(unquote(path[14:])[:80])
            except Exception:
                data = None
            if data is None:
                self._send(404, {"ok": False, "error": "not found"})
            else:
                self._send(200, data, "image/png")
        elif path.startswith("/tex-thumb/"):
            name = unquote(path[11:])
            data = tex_thumb(self.server.rc.tex_dir, name) if TEX_NAME.match(name) and ".." not in name else None
            if data is None:
                self._send(404, {"ok": False, "error": "not found"})
            else:
                self._send(200, data, "image/jpeg")
        elif path.startswith("/layer-thumb/"):       # layer panel: small JPEG of one layer alone
            data = self.server.rc.thumb_bytes(unquote(path[13:]))
            if data:
                self._send(200, data, "image/jpeg")
            else:
                self._send(204, b"", "image/jpeg")
        elif path == "/preview":                     # panel monitor: latest small JPEG of the output, if any
            q = parse_qs(urlsplit(self.path).query)
            data = self.server.rc.preview_bytes(q["solo"][0].split(",") if "solo" in q else ())
            if data:
                self._send(200, data, "image/jpeg")
            else:
                self._send(204, b"", "image/jpeg")   # nothing captured yet (not an error: keeps the console clean)
        elif path.startswith("/tex/"):
            self._image(self.server.rc.tex_dir, unquote(path[5:]), thumb=False)
        elif path.startswith("/art-thumb/"):
            self._image(self.server.rc.art_dir, unquote(path[11:]), thumb=True)
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def _image(self, folder, name, thumb):
        """Serve one image from `folder`. The name must be a plain file name that really is in that folder."""
        data = None
        try:
            folder = Path(folder)
            if thumb:                                # must be exactly one of the library's own names
                if name in control.list_art(folder):
                    data, ctype = art_thumb(folder, name), "image/jpeg"
            elif TEX_NAME.match(name) and "/" not in name and "\\" not in name and ".." not in name:
                f = folder / name
                if f.is_file() and f.resolve().parent == folder.resolve() and f.stat().st_size <= 16 << 20:
                    data, ctype = f.read_bytes(), IMG_TYPES[f.suffix.lower()]
        except (OSError, KeyError, ValueError):
            data = None
        if data is None:
            return self._send(404, {"ok": False, "error": "not found"})
        self._send(200, data, ctype)

    def _body(self, limit):
        """Read the request body (<= limit bytes). Returns bytes, or None after sending the error reply."""
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send(400, {"ok": False, "error": "Content-Length required"}, close=True)
            return None
        if n < 0 or n > limit:
            if 0 < n <= 1 << 20:  # swallow modest oversize bodies so the client can read the 413
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 65536))
                    if not chunk:
                        break
                    left -= len(chunk)
            self._send(413, {"ok": False, "error": "body too large"}, close=True)
            return None
        parts, left = [], n
        while left > 0:
            chunk = self.rfile.read(min(left, 1 << 16))
            if not chunk:
                self._send(400, {"ok": False, "error": "short body"}, close=True)
                return None
            parts.append(chunk)
            left -= len(chunk)
        return b"".join(parts)

    def _upload(self):
        rc = self.server.rc
        # one upload at a time: the body (up to 25 MB) and the decoded image are only ever in memory once
        if not _UPLOAD_GATE.acquire(timeout=UPLOAD_WAIT):
            return self._send(503, {"ok": False, "error": "another upload is in progress, try again"}, close=True)
        try:
            body = self._body(MAX_UPLOAD)
            if body is None:
                return
            try:
                name = save_art(body, unquote(self.headers.get("X-Filename", ""))[:200], rc.art_dir)
            except ValueError as ex:
                return self._send(400, {"ok": False, "error": str(ex)[:200]})
            finally:
                del body
        finally:
            _UPLOAD_GATE.release()
        request_depth(name, rc.ai_url, rc.token_paths)
        names = [a["name"] for a in art_listing(rc.art_dir)]
        self._send(200, {"ok": True, "name": name, "index": names.index(name) if name in names else -1})

    def _art_delete(self):
        body = self._body(MAX_BODY)
        if body is None:
            return
        try:
            name = json.loads(body.decode("utf-8")).get("name")
        except (ValueError, AttributeError, RecursionError):
            return self._send(400, {"ok": False, "error": "bad json"})
        index = delete_art(self.server.rc.art_dir, name)
        if index < 0:
            return self._send(404, {"ok": False, "error": "no such art piece, or it could not be deleted"})
        try:                                         # the controller keeps the same piece on screen
            self.server.rc._q.put_nowait({"cmd": "art_deleted", "i": index})
        except queue.Full:
            pass
        self._send(200, {"ok": True, "index": index})

    def do_POST(self):
        path = self._gate()
        if path is None:
            return
        if path == "/upload":
            return self._upload()
        if path == "/art-delete":
            return self._art_delete()
        if path != "/cmd":
            return self._send(404, {"ok": False, "error": "not found"}, close=True)
        body = self._body(MAX_BODY)
        if body is None:
            return
        try:
            data = json.loads(body.decode("utf-8"))
            cmds = data if isinstance(data, list) else [data]
            if not 1 <= len(cmds) <= MAX_BATCH:
                raise ValueError(f"send 1..{MAX_BATCH} commands")
            cmds = [validate_command(c) for c in cmds]  # all-or-nothing
        except (ValueError, RecursionError) as ex:  # JSONDecodeError / UnicodeDecodeError are ValueErrors
            return self._send(400, {"ok": False, "error": str(ex)[:200]})
        try:
            for c in cmds:
                self.server.rc._q.put_nowait(c)
        except queue.Full:
            return self._send(503, {"ok": False, "error": "render loop is not draining"})
        self._send(200, {"ok": True, "n": len(cmds)})


def _jsonable(o):
    if hasattr(o, "tolist"):  # numpy arrays / scalars
        return o.tolist()
    return float(o)


class RemoteControl:
    """Toggleable LAN remote. Thread-safe: start/stop/drain/publish_state from the main thread."""

    def __init__(self, port=8765, token=None, quiet=False, host="0.0.0.0", tex_dir=None, art_dir=None,
                 ai_url="http://127.0.0.1:8766", token_paths=None):
        self.port, self.host, self.quiet = int(port), host, quiet
        self.tex_dir = Path(tex_dir) if tex_dir else HERE / "textures"      # GET /tex/<name>
        self.art_dir = Path(art_dir) if art_dir else HERE / "art"           # POST /upload, GET /art, /art-thumb/
        self.ai_url = ai_url                                                # depth maps for uploaded art
        self.token_paths = list(token_paths) if token_paths else [HERE / ".remote_token"]
        self.reload_page = False        # True: re-read panel.html on every page load (panel development)
        self.token = str(token) if token else secrets.token_urlsafe(6)  # 8 chars, 48 bits
        self._q = queue.Queue(maxsize=2000)
        self._state = b"{}"
        self._state_obj = None          # latest published state, not yet serialised (see publish_state)
        self.state_published = self.state_encoded = 0
        self._lock = threading.Lock()
        self._views = {}                # solo key (() = real output) -> [last GET time, latest JPEG]
        self._thumb_shot = -1e9
        self._thumb_i = 0
        self.thumb_interval = 0.6       # s between layer thumbnails while a layer panel is on screen
        self._preview_asked = -1e9      # monotonic time of the last GET /preview
        self._preview_shot = -1e9       # ... of the last capture the engine started
        self.preview_interval = 0.33    # s between captures while a panel is watching
        self.preview_linger = 3.0       # s after the last GET before captures stop (no watcher = no cost)
        self._thumbs = {}               # layer name -> JPEG
        self._thumbs_asked = -1e9
        self._job_i = 0
        self._server = self._thread = None

    # -- lifecycle
    @property
    def running(self):
        return self._server is not None

    def start(self):
        """Bind and serve in a daemon thread. Raises OSError if the port is unavailable."""
        if self.running:
            return
        self.drain()  # never replay commands queued before a previous stop
        srv = _Server((self.host, self.port), self)
        self._thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1},
                                        name="remote-control", daemon=True)
        self._thread.start()
        self._server = srv

    def stop(self):
        srv, self._server = self._server, None
        if srv is None:
            return
        srv.active = False
        srv.shutdown()
        srv.server_close()
        srv.drop_connections()
        self._thread.join(timeout=2)
        self._thread = None

    def toggle(self):
        """Start if stopped (printing the URL), stop if running. Never raises; returns `running`."""
        try:
            if self.running:
                self.stop()
                self._log("[remote] off")
            else:
                self.start()
                self.announce()
        except OSError as ex:
            self._log(f"[remote] could not start on port {self.port}: {ex}")
        return self.running

    # -- addresses
    @property
    def url(self):
        return f"http://{lan_ip()}:{self.port}/?k={self.token}"

    @property
    def urls(self):
        """`url` first, then the same page on every other local IPv4 (e.g. the Windows hotspot adapter)."""
        first = lan_ip()
        return [f"http://{ip}:{self.port}/?k={self.token}" for ip in [first] + [i for i in lan_ips() if i != first]]

    def announce(self):
        if self.quiet:
            return
        urls = self.urls
        print(f"[remote] ON  -> open on phone / touch laptop (same Wi-Fi):  {urls[0]}")
        for u in urls[1:]:
            print(f"[remote]        other adapter: {u}")
        qr = _qr_text(urls[0])
        if qr:
            try:
                print(qr)
            except (UnicodeError, OSError):
                pass

    def _log(self, msg):
        if not self.quiet:
            print(msg)

    # -- data flow
    def drain(self):
        """All commands received since the last call (validated dicts, in arrival order)."""
        out = []
        try:
            while True:
                out.append(self._q.get_nowait())
        except queue.Empty:
            return out

    def publish_state(self, state):
        """Snapshot `state` (a JSON-able dict; numpy scalars ok) for the page's next GET /state."""
        # Lazy: the engine publishes several times a second, show or no show. Turning the dict into JSON is the
        # expensive part, so it happens in GET /state (the HTTP thread) and not at all while no panel is connected.
        # `state` must be a dict the caller does not mutate afterwards (both callers build a fresh one each time).
        with self._lock:
            self._state_obj = state
        self.state_published += 1

    def _state_bytes(self):
        with self._lock:
            obj, self._state_obj = self._state_obj, None
        if obj is not None:
            try:
                data = json.dumps(obj, default=_jsonable, allow_nan=False).encode()
            except (TypeError, ValueError, RuntimeError):
                data = None                          # keep the previous good snapshot
            if data is not None:
                self.state_encoded += 1
                with self._lock:
                    self._state = data
        with self._lock:
            return self._state

    # -- output monitor
    def _solo_key(self, names):
        return tuple(n for n in (names or ()) if isinstance(n, str) and n in LAYER_KEYS)[:32]

    def preview_bytes(self, solo=()):
        """Called by GET /preview[?solo=a,b]: marks THAT view as watched and returns its latest JPEG (b"" if none).
        Each solo set has its own picture, so two panels with different solos never see each other's view."""
        key = self._solo_key(solo)
        t = time.monotonic()
        with self._lock:
            if key not in self._views:
                self._preview_shot = -1e9                # a new view: capture it at once
                if len(self._views) >= 6:                # a handful of panels at most; drop the stalest view
                    del self._views[min(self._views, key=lambda k: self._views[k][0])]
                self._views[key] = [t, b""]
            self._views[key][0] = t
            self._preview_asked = t
            return self._views[key][1]

    def preview_due(self, now=None):
        """True when a panel asked recently and the capture interval has passed; calling it claims the slot."""
        t = time.monotonic()
        if t - self._preview_asked > self.preview_linger or t - self._preview_shot < self.preview_interval:
            return False
        self._preview_shot = t
        return True

    # -- layer panel: preview-only mix (never touches the projector) + one thumbnail per layer
    def thumb_bytes(self, name):
        if name not in LAYER_KEYS:
            return b""
        self._thumbs_asked = time.monotonic()
        with self._lock:
            return self._thumbs.get(name, b"")

    def set_thumb(self, name, data):
        if name in LAYER_KEYS and isinstance(data, (bytes, bytearray)) and 0 < len(data) <= 1 << 20:
            with self._lock:
                self._thumbs[name] = bytes(data)

    def preview_job(self):
        """What the engine should capture now: None, ("out", None), ("mix", layer names) or ("thumb", layer).
        Monitor views (every preview_interval, round-robin over the watched solo sets) and layer thumbnails (every
        thumb_interval while a layer panel asks for them) run on separate timers; at most one job per call."""
        t = time.monotonic()
        with self._lock:
            for k in [k for k, v in self._views.items() if t - v[0] > self.preview_linger]:
                del self._views[k]
            live = list(self._views)
        if live and t - self._preview_shot >= self.preview_interval:
            self._preview_shot = t
            self._job_i += 1
            key = live[self._job_i % len(live)]
            return ("mix", key) if key else ("out", None)
        if t - self._thumbs_asked <= self.preview_linger and t - self._thumb_shot >= self.thumb_interval:
            self._thumb_shot = t
            self._thumb_i += 1
            return ("thumb", THUMB_ORDER[self._thumb_i % len(THUMB_ORDER)])
        return None

    def set_preview(self, data, solo=()):
        if isinstance(data, (bytes, bytearray)) and 0 < len(data) <= 2 << 20:
            key = self._solo_key(solo)
            with self._lock:
                if key in self._views:
                    self._views[key][1] = bytes(data)
                elif not key:                            # the plain output is always kept (first GET gets a picture)
                    self._views[key] = [time.monotonic() - self.preview_linger * 0.5, bytes(data)]


def _qr_text(url):
    """Terminal QR code for `url` if the optional `qrcode` package is present, else ''."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=2)
        qr.add_data(url)
        qr.make(fit=True)
        buf = io.StringIO()
        qr.print_ascii(out=buf, invert=True)  # inverted = scannable on a dark terminal
        text = buf.getvalue()
        text.encode(sys.stdout.encoding or "utf-8")  # raises if the console can't show block chars
        return text
    except Exception:
        return ""


# ------------------------------------------------------------------------ page
# panel.html is one self-contained document: no CDN, no external fonts, works with no internet.
# The control surface is generated in the browser from control.meta() (i.e. from params.PARAMS).

def _load_page():
    try:
        page = (HERE / "panel.html").read_text(encoding="utf-8")
    except OSError as ex:
        return f"<!doctype html><html><body>panel.html is missing: {type(ex).__name__}</body></html>"
    meta = json.dumps(control.meta(), separators=(",", ":")).replace("</", "<\\/")
    return page.replace("/*META*/null", meta, 1)


PAGE = _load_page()
PAGE_BYTES = PAGE.encode("utf-8")


# ------------------------------------------------------------------------ demo

def _demo():
    """Serve the panel against a real Controller with fake audio (no GL), to try the UI or test it."""
    import argparse
    import tempfile
    import time

    ap = argparse.ArgumentParser(description="control panel demo: real control system, fake audio, no projector")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=None)
    ap.add_argument("--state-dir", default=None, help="where scenes / midi map go (default: a temp folder)")
    ap.add_argument("--art-dir", default=None, help="art library folder (default: a temp folder)")
    ap.add_argument("--midi", action="store_true", help="open MIDI ports too")
    args = ap.parse_args()
    state_dir = Path(args.state_dir or tempfile.mkdtemp(prefix="panel_demo_"))
    art_dir = Path(args.art_dir) if args.art_dir else state_dir / "art"
    rc = RemoteControl(port=args.port, token=args.token, art_dir=art_dir)
    rc.reload_page = True
    if not rc.toggle():
        raise SystemExit(1)
    ctl = control.Controller(state_dir, restore=True, midi=args.midi, ai=False, art_dir=art_dir)
    st = dict(fold_i=3, mode=2, fb=True, fbAmt=0.88, grid=False, calib=False, next=False, palette=0)
    mapper = surfaces.Mapper(state_dir / "surfaces.json")      # multi-surface mapping, saved in the demo folder
    corners = mapper.corners()
    tex_i, t0, last, pub = 0, time.perf_counter(), time.perf_counter(), 0.0
    print("[demo] Ctrl+C to quit")
    try:
        while True:
            now = time.perf_counter()
            dt, last, t = now - last, now, now - t0
            rest = ctl.apply(rc.drain())
            for c in rest:
                print("[demo] to engine:", c)
            for c in rest:
                if c.get("cmd") == "surf":
                    mapper.command(c)
            corners, tid = mapper.corners(), mapper.target_id()
            before = [list(p) for p in corners]
            acts = apply_commands([c for c in rest if c.get("cmd") in _LEGACY], st, corners)
            if corners != before:
                mapper.push_corners(corners, tid)
            if "save" in acts:
                mapper.save()
            mapper.tick()
            if st["next"]:
                st["next"], tex_i = False, tex_i + 1
            beat = ctl.beat % 1.0
            bands = [math.exp(-beat * 6), 0.5 + 0.3 * math.sin(t * 0.7), 0.5 + 0.5 * math.sin(t * 5.3) ** 8]
            ctl.update(dt, bands)
            if now - pub > 0.12:
                pub = now
                rc.publish_state(dict(grid=st["grid"], calib=st["calib"], corners=mapper.corners(), res=[1280, 800], fps=60.0,
                                      surfaces=mapper.set.state(), bass=bands[0], mid=bands[1], high=bands[2], tex=f"demo_{tex_i % 8:02d}.png",
                                      ctl=ctl.state()))
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.close()
        rc.stop()


_LEGACY = ("fold", "mode", "palette", "fb", "fbAmt", "next", "grid", "calib", "corner", "nudge", "reset", "save")


if __name__ == "__main__":
    _demo()
