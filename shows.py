"""Shows, presets and their files: save / load later (pure python; the I/O runs on ONE worker thread).

A SHOW is the whole project, a PRESET is one piece of it. Both are pure data: JSON with a format tag and a version,
cleaned field by field with the same validators the panel's commands go through (control.py, stack.py, surfaces.py,
recorder.py, timeline.py). Nothing in a file is ever executed. Paths are built from validated slugs inside
<state dir>/shows/ and <state dir>/presets/ only.

    shows/<slug>/show.json           the manifest (format "entheo-show", version SHOW_VERSION)
    shows/<slug>/autosave.json       written every AUTOSAVE_S while the show has unsaved changes (crash recovery)
    shows/<slug>/backups/show.<stamp>.json   the previous show.json, kept on every Save (last KEEP_BACKUPS)
    shows/<slug>/thumbs/scene_<i>.jpg        scene pad thumbnails
    shows/<slug>/media/{art,tex,audio}/      "collect all": copies of the media the show uses (optional)
    shows/_current/thumbs/scene_<i>.jpg      thumbnails of the scenes in memory now (any show, or none)
    shows/_media/<file>              audio files for the timeline (uploaded from the panel)
    shows/.recent.json, shows/.session.json  recent list; which show is open (a restart re-opens its timeline)
    presets/<kind>/<slug>.json + <slug>.jpg  one preset + its thumbnail (one low-res frame of the output)
    <slug>.entheo                    a show as one zip file (export / import / share)

Thread model: the controller (GL thread) takes cheap copies of what it owns and submits a job; the worker reads / writes
files, hashes media, builds listings and cleaned "plans"; the controller picks results up in update() with poll().
"""
import copy
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

SHOW_FORMAT, PRESET_FORMAT = "entheo-show", "entheo-preset"
SHOW_VERSION, PRESET_VERSION = 2, 1      # show v2 (2026-11-26): settings carry clips, decks, comp (clips.py)
APP = "EntheoControl"
KINDS = ("look", "layer", "fx", "mod", "mapping", "midi", "theme")      # theme: a starred THEME (themes.py)
KIND_NAMES = {"look": "Look", "layer": "Layer", "fx": "Effects", "mod": "Mod routing", "mapping": "Mapping", "midi": "MIDI map",
              "theme": "Theme"}
PARTS = ("look", "scenes", "timeline", "settings", "midi", "mapping", "recorder")
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
MEDIA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,119}$")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_EXT = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")
MAX_SHOW_JSON = 16 << 20
MAX_PRESET_JSON = 512 << 10
MAX_IMPORT = 300 << 20                      # an .entheo zip, compressed (streamed to disk, never held in RAM)
MAX_UNPACKED = 1200 << 20                   # its contents
MAX_AUDIO = 120 << 20
MAX_MEDIA_FILE = 64 << 20
MAX_TAGS, MAX_TAG = 12, 24
KEEP_BACKUPS = 10
AUTOSAVE_S = 30.0
MAX_RECENT = 10
THUMB = (160, 100)
SNAP_WAIT = 3.0                              # s the worker waits for the engine's snapshot when a thumbnail is wanted


# ------------------------------------------------------------------ names + files


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:48].strip("-")
    return s or "untitled"


def check_slug(s):
    if not isinstance(s, str) or not SLUG.match(s):
        raise ValueError("bad name")
    return s


def clean_name(v, default="Untitled"):
    if not isinstance(v, str):
        return default
    v = "".join(ch for ch in v if ord(ch) >= 32).strip()[:60]
    return v or default


def clean_tags(v):
    if not isinstance(v, list):
        return []
    out = []
    for t in v[:MAX_TAGS]:
        if isinstance(t, str):
            t = "".join(ch for ch in t.lower() if ch.isalnum() or ch in " -_").strip()[:MAX_TAG]
            if t and t not in out:
                out.append(t)
    return out


def inside(root, path):
    """True if path resolves to a location inside root (no .., no symlink escape)."""
    try:
        root, path = Path(root).resolve(), Path(path).resolve()
        return path == root or root in path.parents
    except OSError:
        return False


_GEN = {}                                    # resolved path -> writes by this process (cache keys include it)
_GEN_LOCK = threading.Lock()


def bump(path):
    """Mark a path as changed by us (write, delete, rename): every cache keyed on it misses next time."""
    try:
        k = str(Path(path).resolve())
    except OSError:
        k = str(path)
    with _GEN_LOCK:
        _GEN[k] = _GEN.get(k, 0) + 1


def stamp(path):
    """Cache key for a file's identity: (mtime_ns, ctime_ns, size, inode, our write generation). mtime alone repeats on
    NTFS for rewrites inside one ~15 ms clock tick; ino changes on every os.replace; the generation covers our own
    writes even when the file system reports nothing new. None if the file is gone."""
    path = Path(path)
    try:
        st = path.stat()
        k = str(path.resolve())
    except OSError:
        return None
    with _GEN_LOCK:
        g = _GEN.get(k, 0)
    return (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino, g)


def write_atomic(path, data):
    """bytes -> path via temp file + os.replace (a crash never leaves half a file). Returns True on success."""
    path = Path(path)
    bump(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for i in range(5):
            try:
                os.replace(tmp, path)
                bump(path)
                return True
            except PermissionError:
                if i == 4:
                    raise
                time.sleep(0.05 * (i + 1))
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def dumps(obj):
    return json.dumps(obj, allow_nan=False, separators=(",", ":")).encode("utf-8")


def quarantine(path):
    from midi import quarantine as q
    return q(path)


def parse_json(data):
    """(obj, error) from bytes; bad UTF-8 / JSON = 'corrupt'."""
    try:
        return json.loads(data.decode("utf-8")), None
    except (ValueError, RecursionError):
        return None, "corrupt"


def read_json(path, limit):
    """(obj, error). error: None | 'missing' | 'unreadable' | 'corrupt' | 'too big'. A corrupt file is NOT moved here."""
    path = Path(path)
    try:
        if not path.exists():
            return None, "missing"
        if path.stat().st_size > limit:
            return None, "too big"
        data = path.read_bytes()
    except OSError:
        return None, "unreadable"
    try:
        return json.loads(data.decode("utf-8")), None           # UnicodeDecodeError is a ValueError: corrupt
    except (ValueError, RecursionError):
        return None, "corrupt"


_HASHES = {}                                 # (path, size, mtime_ns) -> sha256
_HASH_LOCK = threading.Lock()


HASH_CACHE_MIN = 4 << 20                     # files under 4 MB are simply hashed again (cheap, never stale)


def file_hash(path):
    """sha256 of the file's CONTENT. Big media files are cached on stamp() (mtime, ctime, size, inode, our writes)."""
    path = Path(path)
    st = stamp(path)
    if st is None:
        return None
    key = (str(path),) + st if st[2] >= HASH_CACHE_MIN else None
    with _HASH_LOCK:
        if key is not None and key in _HASHES:
            return _HASHES[key]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    d = h.hexdigest()
    if key is not None and stamp(path) == st:      # unchanged while we read it: safe to remember
        with _HASH_LOCK:
            if len(_HASHES) > 4096:
                _HASHES.clear()
            _HASHES[key] = d
    return d


# ------------------------------------------------------------------ migration


def saved_at(raw):
    v = raw.get("saved") if isinstance(raw, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0


def migrate_show(raw):
    """Any older show (or a bare live_state.json) -> the current manifest layout. ValueError if it cannot be read."""
    if not isinstance(raw, dict):
        raise ValueError("not a show file")
    if raw.get("format") != SHOW_FORMAT:
        if isinstance(raw.get("base"), dict) and isinstance(raw.get("settings"), dict):      # v0: a live_state.json
            raw = {"format": SHOW_FORMAT, "version": 1, "name": "Imported state",
                   "controller": {"base": raw["base"], "mods": raw.get("mods"), "cur": raw.get("cur"),
                                  "settings": raw["settings"]}}
        else:
            raise ValueError("not a show file")
    v = raw.get("version")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError("show file has no version")
    if v > SHOW_VERSION:
        raise ValueError(f"this show was saved by a newer version (format {v}); update the software to open it")
    while v < SHOW_VERSION:
        raw = _STEPS[v](raw)
        v += 1
    return raw


def _v1_to_v2(raw):
    """v1 -> v2: the clip model. The 16 snapshot slots, the cue channel, timelines (scene cues by slot number) are
    untouched; the show gets an empty user clip library, no decks and nothing on air as a clip."""
    raw = copy.deepcopy(raw)
    ctl = raw.get("controller") if isinstance(raw.get("controller"), dict) else {}
    st = ctl.get("settings") if isinstance(ctl.get("settings"), dict) else {}
    st.setdefault("clips", [])
    st.setdefault("decks", [])
    st.setdefault("comp", {})
    ctl["settings"] = st
    raw["controller"] = ctl
    raw["version"] = 2
    return raw


_STEPS = {1: _v1_to_v2}


def migrate_preset(raw):
    if not isinstance(raw, dict) or raw.get("format") != PRESET_FORMAT:
        raise ValueError("not a preset file")
    v = raw.get("version")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError("preset file has no version")
    if v > PRESET_VERSION:
        raise ValueError(f"this preset was saved by a newer version (format {v})")
    return raw


# ------------------------------------------------------------------ cleaning (pure data, existing validators)


def _ctl():
    import control                           # lazy: control imports this module
    return control


def clean_values(raw):
    """{param: number} -> clamped, known, non-trigger params only."""
    c = _ctl()
    out = {}
    if isinstance(raw, dict):
        for n, v in raw.items():
            if n in c.CONT and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out[n] = c.clamp(n, v)
    return out


def clean_bindings(raw):
    from midi import _clean_binding
    c = _ctl()
    return [b for b in (_clean_binding(d, c.valid_target) for d in (raw if isinstance(raw, list) else [])[:2048]) if b]


def clean_surfaces(raw):
    """A surfaces list -> the validated 'surf replace' command for the engine, or None."""
    import surfaces
    if isinstance(raw, dict):
        raw = raw.get("surfaces")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        return surfaces.validate_cmd({"cmd": "surf", "op": "replace", "surfaces": raw[:surfaces.MAX_SURFACES]})
    except (ValueError, TypeError):
        return None


def clean_recorder(raw):
    try:
        import recorder
        return recorder.validate_cmd(dict(raw, cmd="rec", op="set")) if isinstance(raw, dict) else None
    except Exception:
        return None


def clean_ext(raw):
    """Other modules' settings a show may carry: governor mode and camera settings. ONLINE is never taken from a
    file (a shared show must not switch the internet on), and camera settings never include ON."""
    out = {}
    if not isinstance(raw, dict):
        return out
    g = raw.get("governor")
    if isinstance(g, dict) and g.get("mode") in ("off", "suggest", "auto"):
        out["governor"] = {"mode": g["mode"]}
    cam = raw.get("camera")
    if isinstance(cam, dict):
        try:
            import camera_layer
            out["camera"] = camera_layer.clean_settings(cam)
        except Exception:
            pass
    return out


def clean_settings(raw):
    """live_state-style settings. Field values are validated again by Controller._apply_state (same as a restart)."""
    if not isinstance(raw, dict):
        return {}
    keep = ("bpm", "clock_src", "strobe_rate", "latch", "morph", "quantize", "glide", "safe", "cue_ch", "autopilot",
            "seq", "text", "bypass", "bypass_fade", "stack", "canvas", "skins", "clips", "decks", "comp",
            "themes")                                                 # themes: the THEMES tray (themes.clean_tray)
    out = {k: copy.deepcopy(raw[k]) for k in keep if k in raw}
    out["ext"] = clean_ext(raw.get("ext"))
    return out


def clean_show(raw):
    """Manifest (already migrated) -> plan: only cleaned, bounded data; unknown keys dropped."""
    c = _ctl()
    ctl = raw.get("controller") if isinstance(raw.get("controller"), dict) else {}
    scenes = ctl.get("scenes")
    plan = {"name": clean_name(raw.get("name")), "notes": clean_name(raw.get("notes"), "")[:60]}
    plan["look"] = {"base": c._clean_base(ctl.get("base")), "mods": c._clean_mods(ctl.get("mods"))}
    cur = ctl.get("cur")
    plan["look"]["cur"] = cur if isinstance(cur, int) and not isinstance(cur, bool) and 0 <= cur < c.N_SCENES else -1
    if isinstance(scenes, list):
        sc = [c._clean_scene(x) for x in scenes[:c.N_SCENES]]
        plan["scenes"] = sc + [None] * (c.N_SCENES - len(sc))
    plan["settings"] = clean_settings(ctl.get("settings"))
    if isinstance(raw.get("midi"), dict):
        plan["midi"] = clean_bindings(raw["midi"].get("bindings"))
    m = clean_surfaces(raw.get("surfaces"))
    if m is not None:
        plan["mapping"] = m
    r = clean_recorder(raw.get("recorder"))
    if r is not None:
        plan["recorder"] = r
    if c._tl is not None:
        tls = raw.get("timelines")
        if not isinstance(tls, list) or not tls:
            tls = [raw.get("timeline")]
        plan["timelines"] = [c._tl.clean(x, c.TL_CTX) for x in tls[:c._tl.MAX_TIMELINES]]
        a = raw.get("timeline_active")
        plan["timeline_active"] = a if isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(
            plan["timelines"]) else 0
    refs = raw.get("refs") if isinstance(raw.get("refs"), dict) else {}
    plan["refs"] = {k: [dict(name=x["name"], sha256=x.get("sha256") if isinstance(x.get("sha256"), str) and re.match(
        r"^[0-9a-f]{64}$", x.get("sha256", "")) else None) for x in (refs.get(k) if isinstance(refs.get(k), list) else [])
        if isinstance(x, dict) and isinstance(x.get("name"), str) and MEDIA_NAME.match(x["name"])][:512]
        for k in ("art", "tex", "audio")}
    return plan


def layer_params(layer):
    """The params a LAYER preset of engine layer `layer` covers: its fader plus the group its tab switch covers."""
    c = _ctl()
    import params as P
    if layer not in c.CONT:
        return []
    names = [layer]
    for group, layers in getattr(P, "BYPASS_LINKS", {}).items():
        if layer in layers:
            names += [n for n, v in P.PARAMS.items() if v[3] == group and n in c.CONT]
    if layer == "serpent":
        names += [n for n in c.CONT if n.startswith("serpent_")]
    return list(dict.fromkeys(names))


def fx_params():
    c = _ctl()
    import params as P
    flt = [n for n, v in P.PARAMS.items() if v[3] == "filters" and n in c.CONT]
    return flt or [n for n in c.BYPASS_COVER.get("fx", ()) if n in c.CONT]


def clean_preset(raw):
    """A preset file / import -> clean preset dict. ValueError if it is not one."""
    raw = migrate_preset(raw)
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError("unknown preset kind")
    c = _ctl()
    d = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    data = {}
    if kind in ("look", "layer", "fx"):
        data["values"] = clean_values(d.get("values"))
        data["mods"] = {n: p for n, p in c._clean_mods(d.get("mods")).items() if n in data["values"]}
        if kind == "layer" and isinstance(d.get("layer"), str):
            data["layer"] = d["layer"][:32]
        if c._stack is not None and kind == "look" and isinstance(d.get("stack"), dict):
            data["stack"] = c._stack.clean_stack(d["stack"])
        if getattr(c, "_skins", None) is not None and kind == "look" and isinstance(d.get("skins"), dict):
            data["skins"] = c._skins.clean_skins(d["skins"])       # environment skins saved with a look ("skin preset")
        if c._stack is not None and kind == "layer" and isinstance(d.get("stack_layer"), dict):
            data["stack_layer"] = c._stack.clean_layer(d["stack_layer"])
        if not data["values"] and "stack_layer" not in data and "stack" not in data:
            raise ValueError("preset has no values")
    elif kind == "mod":
        data["mods"] = c._clean_mods(d.get("mods"))
    elif kind == "mapping":
        m = clean_surfaces(d.get("surfaces"))
        if m is None:
            raise ValueError("mapping preset has no surfaces")
        data["surfaces"] = m["surfaces"]
    elif kind == "midi":
        data["bindings"] = clean_bindings(d.get("bindings"))
    elif kind == "theme":                                    # [themes] a theme with its scope (+ sound tag)
        import themes
        th = themes.clean_theme(d.get("theme"))
        data["theme"] = {k: th[k] for k in ("name", "scope", "pal", "data", "tag")}
    groups = sorted({c.PARAMS[n][3] for n in data.get("values", {})} | {c.PARAMS[n][3] for n in data.get("mods", {})})
    created = raw.get("created")
    return {"format": PRESET_FORMAT, "version": PRESET_VERSION, "kind": kind, "name": clean_name(raw.get("name")),
            "tags": clean_tags(raw.get("tags")), "fav": raw.get("fav") is True,
            "created": float(created) if isinstance(created, (int, float)) and not isinstance(created, bool)
            and math.isfinite(created) else time.time(), "groups": groups, "data": data}


def preset_summary(p, slug, thumb):
    vals = p["data"].get("values", {})
    return {"kind": p["kind"], "slug": slug, "ref": f"{p['kind']}/{slug}", "name": p["name"], "tags": p["tags"],
            "fav": p["fav"], "created": round(p["created"], 1), "groups": p["groups"], "thumb": thumb,
            "palette": vals.get("palette"), "n": len(vals) + len(p["data"].get("mods", {})),
            "layer": p["data"].get("layer")}


# ------------------------------------------------------------------ .entheo zip


ZIP_ALLOWED = re.compile(r"^(show\.json|thumbs/scene_(?:[0-9]|1[0-5])\.jpg|media/(art|tex|audio)/[^/\\:]{1,120})$")
MAX_THUMB = 512 << 10


def zip_entry_ok(name):
    """An .entheo entry name -> its byte limit, or ValueError. Media must have the extension of its kind."""
    if not ZIP_ALLOWED.match(name) or ".." in name:
        raise ValueError(f"unexpected file inside: {name[:60]}")
    if name == "show.json":
        return MAX_SHOW_JSON
    if name.startswith("thumbs/"):
        return MAX_THUMB
    kind, fname = name.split("/")[1], name.rsplit("/", 1)[1]
    ext = Path(fname).suffix.lower()
    if not MEDIA_NAME.match(fname) or ext not in (AUDIO_EXT if kind == "audio" else IMG_EXT):
        raise ValueError("bad media file inside: " + fname[:60])
    return MAX_AUDIO if kind == "audio" else MAX_MEDIA_FILE


def _read_capped(z, info, limit):
    """Bytes of one entry, counting what really comes out (never the header's word for it)."""
    parts, n = [], 0
    with z.open(info) as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            n += len(chunk)
            if n > limit:
                raise ValueError("a file inside is bigger than its header says")
            parts.append(chunk)
    return b"".join(parts)


def export_zip(show_dir, media=True):
    """The show folder as .entheo bytes: show.json, thumbs, and (media=True) the collected media."""
    show_dir = Path(show_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(show_dir / "show.json", "show.json")
        for sub in (["thumbs"] + (["media/art", "media/tex", "media/audio"] if media else [])):
            d = show_dir / sub
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    arc = f"{sub}/{f.name}"
                    if f.is_file() and ZIP_ALLOWED.match(arc) and inside(show_dir, f):
                        z.write(f, arc, compress_type=zipfile.ZIP_STORED if f.suffix.lower() in IMG_EXT + AUDIO_EXT
                                else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def import_zip(data, shows_root):
    """.entheo bytes (or a path to the uploaded file) -> a new show folder (never overwrites one). Returns the slug. ValueError if unsafe / invalid.
    Only allowlisted names are extracted (no zip-slip), sizes are checked before anything is unpacked."""
    try:
        size = Path(data).stat().st_size if isinstance(data, (str, Path)) else len(data)
    except OSError:
        raise ValueError("could not read the upload")
    if size > MAX_IMPORT:
        raise ValueError("file too big")
    try:
        z = zipfile.ZipFile(data if isinstance(data, (str, Path)) else io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("not an .entheo file")
    with z:
        infos = z.infolist()
        if len(infos) > 2000:
            raise ValueError("too many files inside")
        total = 0
        for i in infos:
            if i.is_dir():
                continue
            if i.file_size > zip_entry_ok(i.filename):
                raise ValueError("a file inside is too big")
            total += i.file_size
        if total > MAX_UNPACKED:
            raise ValueError("contents too big")
        try:
            raw = json.loads(_read_capped(z, z.getinfo("show.json"), MAX_SHOW_JSON).decode("utf-8"))
        except KeyError:
            raise ValueError("no show.json inside")
        except (ValueError, RecursionError):
            raise ValueError("show.json is corrupt")
        raw = migrate_show(raw)
        clean_show(raw)                                      # must clean without error before anything is written
        root = Path(shows_root)
        slug = unique_slug(root, slugify(clean_name(raw.get("name"))))
        dest = root / slug
        dest.mkdir(parents=True)
        try:
            unpacked = 0
            for i in infos:
                if i.is_dir():
                    continue
                target = dest / i.filename
                if not inside(dest, target):
                    raise ValueError("unsafe path")
                data = _read_capped(z, i, zip_entry_ok(i.filename))
                unpacked += len(data)
                if unpacked > MAX_UNPACKED:
                    raise ValueError("contents too big")
                if i.filename.startswith("thumbs/") and not data.startswith(b"\xff\xd8"):
                    continue                                 # not a JPEG: dropped
                if not write_atomic(target, data):
                    raise ValueError("could not write")
                del data
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
    return slug


def unique_slug(root, slug):
    root = Path(root)
    base, n = slug[:40], 1
    while (root / slug).exists() or slug.startswith("_"):
        n += 1
        slug = f"{base}-{n}"
    return slug


# ------------------------------------------------------------------ the store (worker side)


class ShowManager:
    """Owns shows/ + presets/ and the worker thread. All public methods are cheap and safe on the GL thread."""

    def __init__(self, state_dir, enabled=True, art_dir=None, tex_dir=None, start=True):
        self.dir = Path(state_dir)
        self.shows_dir, self.presets_dir = self.dir / "shows", self.dir / "presets"
        self.media_dir, self.work_dir = self.shows_dir / "_media", self.shows_dir / "_current"
        self.art_dir = Path(art_dir) if art_dir else self.dir / "art"
        self.tex_dir = Path(tex_dir) if tex_dir else self.dir / "textures"
        self.enabled = bool(enabled)
        self.current = None                                  # slug of the open show (None = untitled)
        self.current_name = "Untitled"
        self.dirty = False
        self.missing = []                                    # media the open show refers to but this laptop lacks
        self.notice = None                                   # (time, level, text) for the panel
        self.busy = 0
        self._busy_lock = threading.Lock()                  # busy is counted on two threads
        self.rev = {"shows": 0, "presets": 0, "thumbs": 0}
        self.blobs = {"shows": (0, b'{"shows":[],"recent":[]}'), "presets": (0, b'{"presets":[]}')}
        self.thumbs = {}                                     # scene slot -> rev (pads with a thumbnail)
        self.cache = {}                                      # preset ref -> clean preset (timeline cues, recalls)
        self._jobs, self._results = queue.Queue(), queue.Queue()
        self._autosave_t = 0.0
        self._pcache = {}                                    # path -> (mtime_ns, summary)
        self._thread = None
        if self.enabled and start:
            self._thread = threading.Thread(target=self._run, name="shows", daemon=True)
            self._thread.start()
            self.submit("rescan")

    # ---------------------------------------------------------------- GL-thread API
    def submit(self, kind, **kw):
        if not self.enabled:
            self.say("warn", "saving is off in this run (headless / read-only)")
            return False
        with self._busy_lock:
            self.busy += 1
        self._jobs.put((kind, kw))
        return True

    def poll(self):
        """Finished jobs: list of (kind, payload) for the controller to apply. Never blocks."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def say(self, level, text):
        self.notice = (round(time.time(), 2), level, str(text)[:200])

    def run_sync(self):
        """Tests / no-thread mode: run queued jobs on this thread."""
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            self._job(*job)

    def close(self):
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout=5)
            self._thread = None

    def state(self):
        n = self.notice
        thumbs = dict(self.thumbs)                           # GL thread owns it; a copy before iterating anyway
        out = {"current": self.current, "name": self.current_name, "dirty": self.dirty, "rev": dict(self.rev)}
        opt = {"off": not self.enabled, "busy": self.busy > 0, "missing": self.missing[:20], "notice": list(n) if n else None,
               "thumbs": {str(k): v for k, v in thumbs.items()}}
        out.update({k: v for k, v in opt.items() if v})          # compact: a phone polls this once a second
        return out

    # ---------------------------------------------------------------- worker
    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._job(*job)

    def _job(self, kind, kw):
        try:
            getattr(self, "_do_" + kind)(**kw)
        except Exception as ex:                              # a job never kills the worker
            self.say("error", f"{kind} failed: {type(ex).__name__}: {ex}")
        finally:
            with self._busy_lock:
                self.busy = max(0, self.busy - 1)

    def _post(self, kind, payload=None):
        self._results.put((kind, payload))

    # ---- listings
    def _do_rescan(self, what=("shows", "presets")):
        if "shows" in what:
            self._list_shows()
        if "presets" in what:
            self._list_presets()

    def _list_shows(self):
        out = []
        if self.shows_dir.is_dir():
            for d in sorted(self.shows_dir.iterdir()):
                if not d.is_dir() or not SLUG.match(d.name):
                    continue
                f, a = d / "show.json", d / "autosave.json"
                try:
                    st = f.stat()
                except OSError:
                    if a.exists():
                        out.append({"slug": d.name, "name": d.name, "saved": 0, "autosave": True, "broken": True})
                    continue
                raw, err = read_json(f, MAX_SHOW_JSON)
                name = clean_name(raw.get("name")) if isinstance(raw, dict) else d.name
                newer = a.exists() and saved_at(read_json(a, MAX_SHOW_JSON)[0]) > saved_at(raw)
                backups = sorted((d / "backups").glob("show.*.json")) if (d / "backups").is_dir() else []
                out.append({"slug": d.name, "name": name, "saved": round(st.st_mtime, 1), "size": st.st_size,
                            "autosave": newer, "broken": err is not None, "backups": len(backups),
                            "thumb": (d / "thumbs" / "scene_0.jpg").exists(),
                            "template": bool(isinstance(raw, dict) and raw.get("template") is True)})
        out.sort(key=lambda s: -s.get("saved", 0))
        recent, _ = read_json(self.shows_dir / ".recent.json", 65536)
        recent = [s for s in (recent if isinstance(recent, list) else []) if isinstance(s, str) and SLUG.match(s)]
        audio = sorted(f.name for f in self.media_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXT
                       and MEDIA_NAME.match(f.name)) if self.media_dir.is_dir() else []
        rev = self.rev["shows"] + 1
        self.blobs["shows"] = (rev, dumps({"shows": out, "recent": recent[:MAX_RECENT], "audio": audio}))
        self.rev["shows"] = rev                              # advertised only once the bytes are there

    def _list_presets(self):
        out, seen = [], set()
        for kind in KINDS:
            d = self.presets_dir / kind
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.json")):
                if not SLUG.match(f.stem):
                    continue
                seen.add(str(f))
                try:
                    if f.stat().st_size > MAX_PRESET_JSON:
                        raise OSError("too big")
                    body = f.read_bytes()
                except OSError:
                    continue
                mt = hashlib.sha1(body).digest()
                hit = self._pcache.get(str(f))
                if hit is None or hit[0] != mt:
                    raw, err = parse_json(body)
                    try:
                        if err:
                            raise ValueError(err)
                        p = clean_preset(raw)
                        if p["kind"] != kind:
                            raise ValueError("kind mismatch")
                    except ValueError:
                        if err != "unreadable":
                            quarantine(f)                    # corrupt preset: moved aside, the rest still list
                        continue
                    hit = (mt, preset_summary(p, f.stem, False))
                    self._pcache[str(f)] = hit
                s = dict(hit[1], thumb=(d / (f.stem + ".jpg")).exists())
                out.append(s)
        self._pcache = {k: v for k, v in self._pcache.items() if k in seen}
        rev = self.rev["presets"] + 1
        self.blobs["presets"] = (rev, dumps({"presets": out, "kinds": KIND_NAMES}))
        self.rev["presets"] = rev

    # ---- shows
    def show_dir(self, slug):
        d = self.shows_dir / check_slug(slug)
        if not inside(self.shows_dir, d):
            raise ValueError("bad name")
        return d

    def _media_refs(self, snap):
        """Which media the show uses: the art piece(s) its look / scenes select, texture and art names in timeline
        events, the timeline's audio. Hashed (cached by size + mtime)."""
        c = _ctl()
        art = c.list_art(self.art_dir)
        idx = set()
        for b in [snap["controller"]["base"]] + [s["base"] for s in snap["controller"]["scenes"] if s]:
            if "art_index" in b:
                idx.add(int(round(b["art_index"])))
        names = {"art": {art[i] for i in idx if 0 <= i < len(art)}, "tex": set(), "audio": set()}
        for tl in (snap.get("timelines") or [snap.get("timeline") or {}]):
            for tr in tl.get("tracks", []):
                for it in tr.get("items", []):
                    cmd = it.get("cmd") or {}
                    if cmd.get("cmd") == "show_texture" or (cmd.get("cmd") == "art_pick" and cmd.get("source") == "ai"):
                        names["tex"].add(cmd.get("name"))
                    elif cmd.get("cmd") == "art_pick" and cmd.get("name"):
                        names["art"].add(cmd["name"])
            if isinstance(tl.get("audio"), dict):
                names["audio"].add(tl["audio"]["name"])
        folders = {"art": self.art_dir, "tex": self.tex_dir, "audio": self.media_dir}
        refs = {}
        for k, ns in names.items():
            refs[k] = []
            for n in sorted(x for x in ns if isinstance(x, str) and MEDIA_NAME.match(x)):
                f = folders[k] / n
                refs[k].append({"name": n, "sha256": file_hash(f) if f.is_file() else None,
                                "size": f.stat().st_size if f.is_file() else None})
        return refs, folders

    def _manifest(self, snap, name):
        refs, folders = self._media_refs(snap)
        surf, _ = read_json(self.dir / "surfaces.json", 1 << 20)
        rec, _ = read_json(self.dir / "recordings" / ".settings.json", 65536)
        man = {"format": SHOW_FORMAT, "version": SHOW_VERSION, "app": APP, "name": name, "saved": time.time(),
               "controller": snap["controller"], "midi": {"bindings": snap["midi"]},
               "surfaces": {"surfaces": surf.get("surfaces")} if isinstance(surf, dict) else None,
               "recorder": {k: v for k, v in rec.items() if k in ("source", "fps", "res", "audio", "quality")}
               if isinstance(rec, dict) else None,
               "timeline": snap.get("timeline"), "timelines": snap.get("timelines"),
               "timeline_active": snap.get("timeline_active", 0), "refs": refs}
        man["controller"]["settings"]["ext"] = clean_ext(man["controller"]["settings"].get("ext"))
        return man, refs, folders

    def _do_save(self, snap, name, slug=None, media=False, autosave=False, template=False):
        """Save (slug = the open show) or Save As (new slug from name). autosave=True writes autosave.json only."""
        if slug is None:
            slug = unique_slug(self.shows_dir, slugify(name))
        d = self.show_dir(slug)
        man, refs, folders = self._manifest(snap, name)
        if template:
            man["template"] = True
        data = dumps(man)
        if autosave:
            if write_atomic(d / "autosave.json", data):
                self._post("autosaved", slug)
                self._list_shows()                           # the Open list now offers RECOVER for this show
            return
        f = d / "show.json"
        if f.exists():                                       # keep the previous version
            stamp = time.strftime("%Y%m%d-%H%M%S")
            try:
                (d / "backups").mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / "backups" / f"show.{stamp}.json")
                for old in sorted((d / "backups").glob("show.*.json"))[:-KEEP_BACKUPS]:
                    old.unlink()
            except OSError:
                pass
        if not write_atomic(f, data):
            self.say("error", f"could not write {slug}/show.json (disk full or locked?)")
            return
        try:
            (d / "autosave.json").unlink()
        except OSError:
            pass
        self._copy_thumbs(self.work_dir / "thumbs", d / "thumbs")
        if media:                                            # collect all: copies of the media into the show folder
            n = 0
            for k, items in refs.items():
                for r in items:
                    src = folders[k] / r["name"]
                    dst = d / "media" / k / r["name"]
                    cap = MAX_AUDIO if k == "audio" else MAX_MEDIA_FILE
                    if src.is_file() and src.stat().st_size <= cap and not dst.exists():
                        try:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
                            n += 1
                        except OSError:
                            pass
            self.say("ok", f"saved '{name}' with {n} media file(s) collected")
        else:
            self.say("ok", f"saved '{name}'")
        self._touch_recent(slug)
        write_atomic(self.shows_dir / ".session.json", dumps({"show": slug}))
        self._post("saved", {"slug": slug, "name": name})
        self._list_shows()

    def _copy_thumbs(self, src, dst):
        try:
            if dst.is_dir():
                for f in dst.glob("scene_*.jpg"):
                    f.unlink()
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.glob("scene_*.jpg"):
                    if re.match(r"^scene_(?:[0-9]|1[0-5])\.jpg$", f.name):
                        shutil.copy2(f, dst / f.name)
        except OSError:
            pass

    def _touch_recent(self, slug):
        raw, _ = read_json(self.shows_dir / ".recent.json", 65536)
        lst = [s for s in (raw if isinstance(raw, list) else []) if isinstance(s, str) and SLUG.match(s) and s != slug]
        write_atomic(self.shows_dir / ".recent.json", dumps([slug] + lst[:MAX_RECENT - 1]))

    def _do_open(self, slug, recover=False, backup=None, parts=None, template=False):
        d = self.show_dir(slug)
        if backup is not None:
            f = d / "backups" / backup
            if not re.match(r"^show\.\d{8}-\d{6}\.json$", backup) or not inside(d, f):
                raise ValueError("bad backup name")
        else:
            f = d / ("autosave.json" if recover else "show.json")
        raw, err = read_json(f, MAX_SHOW_JSON)
        if err == "corrupt":
            q = quarantine(f)
            self.say("error", f"{slug}: {f.name} is corrupt; moved aside as {q.name if q else '?'}. "
                              "Open a backup or the autosave instead.")
            self._list_shows()
            return
        if err:
            self.say("error", f"{slug}: {f.name} is {err}")
            return
        try:
            plan = clean_show(migrate_show(raw))
        except ValueError as ex:
            self.say("error", f"{slug}: {ex}")
            return
        plan["missing"] = self._restore_media(d, plan["refs"])
        plan["slug"], plan["template"] = (None if template else slug), template
        plan["parts"] = [p for p in (parts or PARTS) if p in PARTS]
        if "scenes" in plan["parts"]:
            self._copy_thumbs(d / "thumbs", self.work_dir / "thumbs")
            plan["thumbs"] = sorted(int(f.stem.split("_")[1]) for f in (self.work_dir / "thumbs").glob("scene_*.jpg")
                                    if re.match(r"^scene_(?:[0-9]|1[0-5])$", f.stem))
        for cue in self._preset_cues(plan.get("timelines")):
            self._load_preset_ref(cue)
        full = {"look", "scenes", "timeline"} <= set(plan["parts"])
        plan["switch"] = full and not template
        if plan["switch"]:
            self._touch_recent(slug)
            write_atomic(self.shows_dir / ".session.json", dumps({"show": slug}))
        self._post("opened", plan)
        self._list_shows()

    def _restore_media(self, d, refs):
        """Media the show refers to: present (same hash) = fine; missing here but collected in the show = copied back
        (never over a different file of the same name); else reported missing."""
        folders = {"art": self.art_dir, "tex": self.tex_dir, "audio": self.media_dir}
        missing = []
        for k, items in refs.items():
            for r in items:
                dst = folders[k] / r["name"]
                if Path(r["name"]).suffix.lower() not in (AUDIO_EXT if k == "audio" else IMG_EXT) or not inside(folders[k], dst):
                    missing.append(f"{k}/{r['name']} (refused)")
                    continue
                if dst.is_file():
                    if r["sha256"] and file_hash(dst) != r["sha256"]:
                        missing.append(f"{k}/{r['name']} (different file with the same name)")
                    continue
                src = d / "media" / k / r["name"]
                if src.is_file() and inside(d, src):
                    try:
                        folders[k].mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        continue
                    except OSError:
                        pass
                missing.append(f"{k}/{r['name']}")
        return missing

    def _do_delete_show(self, slug):
        d = self.show_dir(slug)
        if d.is_dir():                                       # moved aside, not destroyed (undo by renaming back)
            trash = self.shows_dir / ".trash"
            trash.mkdir(parents=True, exist_ok=True)
            os.replace(d, trash / f"{slug}.{time.strftime('%Y%m%d-%H%M%S')}")
            self.say("ok", f"show '{slug}' moved to shows/.trash")
        self._list_shows()

    def _do_session(self):
        """Startup: which show was open, and its newest timeline (autosave if newer)."""
        raw, _ = read_json(self.shows_dir / ".session.json", 65536)
        slug = raw.get("show") if isinstance(raw, dict) else None
        if not isinstance(slug, str) or not SLUG.match(slug):
            raw, _ = read_json(self.work_dir / "autosave.json", MAX_SHOW_JSON)
            if isinstance(raw, dict):
                self._session_plan(raw, None)
            return
        d = self.show_dir(slug)
        f, a = d / "show.json", d / "autosave.json"
        raw, _ = read_json(f, MAX_SHOW_JSON)
        auto, _ = read_json(a, MAX_SHOW_JSON) if a.exists() else (None, None)
        if auto is not None and saved_at(auto) > saved_at(raw):
            raw = auto                                       # crashed with unsaved changes: their timeline comes back
        if raw is not None:
            self._session_plan(raw, slug)

    def _session_plan(self, raw, slug):
        try:
            plan = clean_show(migrate_show(raw))
        except ValueError:
            return
        for cue in self._preset_cues(plan.get("timelines")):
            self._load_preset_ref(cue)
        self._post("session", {"slug": slug, "name": plan["name"], "timelines": plan.get("timelines"),
                               "timeline_active": plan.get("timeline_active", 0),
                               "thumbs": sorted(int(f.stem.split("_")[1]) for f in (self.work_dir / "thumbs").glob(
                                   "scene_*.jpg") if re.match(r"^scene_(?:[0-9]|1[0-5])$", f.stem))})

    def _do_autosave(self, snap, slug, name):
        if slug is None:                                     # untitled: the working folder keeps the crash copy
            self.work_dir.mkdir(parents=True, exist_ok=True)
            man, _, _ = self._manifest(snap, name)
            write_atomic(self.work_dir / "autosave.json", dumps(man))
            self._post("autosaved", None)
            return
        self._do_save(snap, name, slug=slug, autosave=True)

    def _do_import(self, data):
        slug = import_zip(data, self.shows_dir)
        self.say("ok", f"imported show '{slug}'")
        self._list_shows()

    def _do_session_clear(self):
        write_atomic(self.shows_dir / ".session.json", dumps({"show": None}))

    # ---- presets
    def preset_path(self, kind, slug, ext=".json"):
        if kind not in KINDS:
            raise ValueError("bad kind")
        f = self.presets_dir / kind / (check_slug(slug) + ext)
        if not inside(self.presets_dir, f):
            raise ValueError("bad name")
        return f

    def _do_preset_save(self, preset, snap_t=None, slug=None):
        if preset.get("kind") == "mapping" and not (preset.get("data") or {}).get("surfaces"):
            surf, err = read_json(self.dir / "surfaces.json", 1 << 20)
            if err:
                self.say("error", "no saved mapping (surfaces.json) to make a preset from")
                return
            preset = dict(preset, data={"surfaces": surf.get("surfaces") if isinstance(surf, dict) else None})
        p = clean_preset(preset)
        d = self.presets_dir / p["kind"]
        if slug is None:
            slug = free_preset_slug(d, slugify(p["name"]))
        f = self.preset_path(p["kind"], slug)
        if not write_atomic(f, dumps(p)):
            self.say("error", "could not write the preset")
            return
        self.cache[f"{p['kind']}/{slug}"] = p
        if snap_t is not None:
            self._thumb_from_snapshot(snap_t, self.preset_path(p["kind"], slug, ".jpg"))
        self.say("ok", f"preset '{p['name']}' saved")
        self._list_presets()
        self._post("preset_saved", f"{p['kind']}/{slug}")

    def _do_preset_import(self, raw):
        p = clean_preset(raw)
        self._do_preset_save(p)

    def _do_preset_set(self, ref, name=None, tags=None, fav=None):
        kind, slug = ref.split("/")
        f = self.preset_path(kind, slug)
        raw, err = read_json(f, MAX_PRESET_JSON)
        if err:
            raise ValueError(err)
        p = clean_preset(raw)
        if name is not None:
            p["name"] = clean_name(name)
        if tags is not None:
            p["tags"] = clean_tags(tags)
        if fav is not None:
            p["fav"] = bool(fav)
        write_atomic(f, dumps(p))
        self.cache[ref] = p
        self._list_presets()

    def _do_preset_delete(self, ref):
        kind, slug = ref.split("/")
        for ext in (".json", ".jpg"):
            f = self.preset_path(kind, slug, ext)
            try:
                if f.exists():
                    trash = self.presets_dir / ".trash"
                    trash.mkdir(parents=True, exist_ok=True)
                    os.replace(f, trash / f"{kind}.{slug}.{int(time.time())}{ext}")
                    bump(f)
            except OSError:
                pass
        self.cache.pop(ref, None)
        self._pcache.pop(str(self.preset_path(kind, slug)), None)
        self._list_presets()

    def _load_preset_ref(self, ref):
        try:
            kind, slug = ref.split("/")
            path = self.preset_path(kind, slug)
        except ValueError:
            return None
        raw, err = read_json(path, MAX_PRESET_JSON)
        if err:
            return None
        try:
            p = clean_preset(raw)
        except ValueError:
            return None
        self.cache[ref] = p
        return p

    def _do_preset_load(self, ref, opts):
        p = self.cache.get(ref) or self._load_preset_ref(ref)
        if p is None:
            self.say("error", f"preset {ref} not found")
            return
        self._post("preset", (ref, p, opts))

    def _do_prefetch(self, refs):
        for r in refs:
            if r not in self.cache:
                self._load_preset_ref(r)

    @staticmethod
    def _preset_cues(tl):
        out = []
        for doc in (tl if isinstance(tl, list) else [tl]):
            for tr in (doc or {}).get("tracks", []):
                if tr.get("kind") == "cue":
                    out += [it["preset"] for it in tr["items"] if it.get("kind") == "preset"]
        return sorted(set(out))

    # ---- thumbnails (one low-res frame the engine already writes for "snapshot")
    def _thumb_from_snapshot(self, t_req, dst):
        src = self.dir / "snapshots" / "current.png"
        deadline = time.time() + SNAP_WAIT
        while time.time() < deadline:
            try:
                if src.stat().st_mtime >= t_req - 0.05:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            return False
        time.sleep(0.15)                                     # the engine writes it atomically; give os.replace a beat
        try:
            from PIL import Image
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail(THUMB)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=78)
            return write_atomic(dst, buf.getvalue())
        except Exception:
            return False

    def _do_scene_thumb(self, slot, snap_t):
        if self._thumb_from_snapshot(snap_t, self.work_dir / "thumbs" / f"scene_{int(slot)}.jpg"):
            self._post("thumb", int(slot))

    def _do_scene_thumb_clear(self, slot):
        try:
            (self.work_dir / "thumbs" / f"scene_{int(slot)}.jpg").unlink()
        except OSError:
            pass
        self._post("thumb_gone", int(slot))

    # ---- [themes] session-tray thumbnails (shows/_current/themes/<id>.jpg) + starring into presets/theme/
    def theme_thumb_path(self, tid):
        if not isinstance(tid, str) or not re.match(r"^[a-z0-9]{1,16}$", tid):
            raise ValueError("bad theme id")
        return self.work_dir / "themes" / f"{tid}.jpg"

    def _do_theme_thumb(self, tid, snap_t):
        if self._thumb_from_snapshot(snap_t, self.theme_thumb_path(tid)):
            self._post("theme_thumb", tid)

    def _do_theme_thumb_clear(self, tid):
        try:
            self.theme_thumb_path(tid).unlink()
        except OSError:
            pass

    def _do_theme_star(self, preset, tid, slug=None):
        p = clean_preset(preset)
        d = self.presets_dir / "theme"
        if slug is None or not self.preset_path("theme", slug).is_file():
            slug = free_preset_slug(d, slugify(p["name"]))
        f = self.preset_path("theme", slug)
        if not write_atomic(f, dumps(p)):
            self.say("error", "could not write the theme")
            return
        ref = f"theme/{slug}"
        self.cache[ref] = p
        self._pcache.pop(str(f), None)
        try:
            src = self.theme_thumb_path(tid)
            if src.is_file():
                write_atomic(self.preset_path("theme", slug, ".jpg"), src.read_bytes())
        except (OSError, ValueError):
            pass
        self._list_presets()
        self._post("theme_starred", (tid, ref))

    def _do_media_upload(self, name, data):
        """Timeline audio into shows/_media/ (server thread calls this directly too)."""
        save_media(self.media_dir, name, data)
        self._list_shows()


def check_media_name(name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("audio file name must be a plain file name")
    if not MEDIA_NAME.match(name) or Path(name).suffix.lower() not in AUDIO_EXT:
        raise ValueError("audio file must be mp3 / wav / ogg / m4a / flac / aac / opus with a plain name")
    return name


def free_preset_slug(d, slug):
    base, n = slug[:40], 1
    while (Path(d) / f"{slug}.json").exists():
        n += 1
        slug = f"{base}-{n}"
    return slug


def save_media(folder, name, data):
    check_media_name(name)
    if len(data) > MAX_AUDIO:
        raise ValueError("audio file too big")
    f = Path(folder) / name
    if not inside(folder, f) or not write_atomic(f, data):
        raise ValueError("could not save the audio file")
    return name
