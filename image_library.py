"""One image library: uploaded art (art/), AI generations + seed textures (textures/) and dreamified images (both),
usable everywhere. Pure Python (PIL for the copy); no GL, never imports the engine.

Ids: "art:<name>" (art/<name>) and "ai:<name>" (textures/<name>). Depth maps (<stem>.depth.png) are never images.
Library meta (titles, favourites, the hand-picked set for the AI-texture layer, which AI image became which art
piece) lives in art/.library.json, written atomically; a broken or missing file is an empty meta, never a crash.

  listing(art_dir, tex_dir)                      -> [{id, kind, name, title, fav, picked, dreamed, ai, depth, t}]
  edit(art_dir, tex_dir, op, kind, name, title)  op: fav | unfav | pick | unpick | title
  import_ai(tex_dir, art_dir, name, title=None)  AI image -> a new art piece (PNG re-encode, prompt as title, deduped,
                                                 quota-checked); returns the art file name (the caller asks for depth)
  delete(art_dir, tex_dir, kind, name)           remove one image (+ its depth map) and forget its flags
  layer_files(img_set, tex_dir, art_dir, picked, extra) -> [(key, path)] the AI-texture layer cycles through:
                                                 0 AI / seed textures (as always), 1 art uploads, 2 both, 3 picked set
                                                 keys: "<name>" for textures/, "art/<name>" for art/
"""
import json
import os
import re
import secrets
import time
from pathlib import Path

try:
    from art_layer import list_art
except Exception:                                    # pragma: no cover - art_layer always ships with the engine
    def list_art(folder):
        try:
            fs = [f for f in Path(folder).iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS
                  and not f.name.lower().endswith(DEPTH_SUFFIX) and not f.name.startswith(".")]
        except OSError:
            return []
        return [f.name for f in sorted(fs, key=lambda f: (f.stat().st_ctime, f.name))]

META_NAME = ".library.json"
DEPTH_SUFFIX = ".depth.png"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+,-]{0,159}$")
KINDS = ("art", "ai")
LAYER_SETS = ["AI images", "Art uploads", "AI + art", "Picked set"]
ART_CAP_FILES = 400                                   # art library quota (uploads + imports)
ART_CAP_BYTES = 3 << 30
MAX_SIDE, MAX_PIXELS = 4096, 40_000_000
MAX_TITLE = 120
MAX_FLAGS = 2000


def _clean_name(name):
    if (not isinstance(name, str) or not NAME_RE.match(name) or ".." in name or Path(name).name != name
            or Path(name).suffix.lower() not in IMG_EXTS or name.lower().endswith(DEPTH_SUFFIX)):
        raise ValueError("bad image name")
    return name


def _folder(kind, art_dir, tex_dir):
    if kind not in KINDS:
        raise ValueError("kind must be art or ai")
    return Path(art_dir if kind == "art" else tex_dir)


def resolve(kind, name, art_dir, tex_dir):
    """-> the existing file of one library image, inside its folder; ValueError otherwise."""
    folder = _folder(kind, art_dir, tex_dir)
    name = _clean_name(name)
    p = folder / name
    try:
        if not p.is_file() or p.resolve().parent != folder.resolve():
            raise ValueError("no such image")
    except OSError:
        raise ValueError("no such image")
    return p


def is_dreamed(kind, name):
    n = name.lower()
    return n.startswith("gen_dream_") if kind == "ai" else "_dream_" in n


# ---------------------------------------------------------------- meta

def _empty():
    return dict(version=1, titles={}, favs=[], picked=[], imported={})


def load_meta(art_dir):
    m = _empty()
    try:
        raw = json.loads((Path(art_dir) / META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return m
    if not isinstance(raw, dict):
        return m
    if isinstance(raw.get("titles"), dict):
        m["titles"] = {str(k)[:200]: str(v)[:MAX_TITLE] for k, v in list(raw["titles"].items())[:MAX_FLAGS]}
    for k in ("favs", "picked"):
        if isinstance(raw.get(k), list):
            m[k] = list(dict.fromkeys(str(x)[:200] for x in raw[k][:MAX_FLAGS] if isinstance(x, str)))
    if isinstance(raw.get("imported"), dict):
        m["imported"] = {str(k)[:200]: str(v)[:200] for k, v in list(raw["imported"].items())[:MAX_FLAGS]}
    return m


def save_meta(art_dir, meta):
    folder = Path(art_dir)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{META_NAME}.{os.getpid()}.{secrets.token_hex(3)}.tmp"
    tmp.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, folder / META_NAME)


def meta_sig(art_dir):
    try:
        st = os.stat(Path(art_dir) / META_NAME)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _clean_title(t):
    return " ".join(str(t or "").split())[:MAX_TITLE]


def png_prompt(path):
    """The generation prompt stored in an AI image's PNG text chunk (ai_worker writes it), or ''."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return _clean_title((getattr(im, "text", None) or im.info or {}).get("prompt", ""))
    except Exception:
        return ""


# ---------------------------------------------------------------- listing / flags

def _tex_names(tex_dir):
    out = []
    try:
        for f in Path(tex_dir).iterdir():
            n = f.name
            if (f.suffix.lower() in IMG_EXTS and not n.lower().endswith(DEPTH_SUFFIX) and NAME_RE.match(n)
                    and not n.startswith(".") and f.is_file()):
                out.append((f.stat().st_mtime, n))
    except OSError:
        return []
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


_PROMPTS = {}                                         # (path, mtime) -> prompt (PNG text chunks are read once)


def listing(art_dir, tex_dir, cap=64):
    """Every image: art pieces in art_index order, then textures newest first (the engine holds the newest 64)."""
    meta, out = load_meta(art_dir), []
    favs, picked, imported = set(meta["favs"]), set(meta["picked"]), set(meta["imported"].values())
    for n in list_art(art_dir):
        p = Path(art_dir) / n
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        i = "art:" + n
        out.append(dict(id=i, kind="art", name=n, title=meta["titles"].get(i) or Path(n).stem, fav=i in favs,
                        picked=i in picked, dreamed=is_dreamed("art", n), ai=n in imported, depth=(Path(art_dir) / (Path(n).stem + DEPTH_SUFFIX)).exists(),
                        t=round(t, 1)))
    for t, n in _tex_names(tex_dir)[:cap]:
        i, p = "ai:" + n, Path(tex_dir) / n
        title = meta["titles"].get(i)
        if not title:
            key = (str(p), t)
            if key not in _PROMPTS:
                if len(_PROMPTS) > 512:
                    _PROMPTS.clear()
                _PROMPTS[key] = png_prompt(p) if n.lower().startswith("gen_") else ""
            title = _PROMPTS[key] or Path(n).stem
        out.append(dict(id=i, kind="ai", name=n, title=title, fav=i in favs, picked=i in picked,
                        dreamed=is_dreamed("ai", n), ai=n.lower().startswith("gen_"),
                        depth=(Path(tex_dir) / (Path(n).stem + DEPTH_SUFFIX)).exists(), t=round(t, 1)))
    return out


def edit(art_dir, tex_dir, op, kind, name, title=None):
    """fav / unfav / pick / unpick / title one image (it must exist). Returns the new meta."""
    resolve(kind, name, art_dir, tex_dir)
    i, meta = f"{kind}:{name}", load_meta(art_dir)
    if op in ("fav", "pick"):
        lst = meta["favs" if op == "fav" else "picked"]
        if i not in lst:
            if len(lst) >= MAX_FLAGS:
                raise ValueError("too many marked images")
            lst.append(i)
    elif op in ("unfav", "unpick"):
        k = "favs" if op == "unfav" else "picked"
        meta[k] = [x for x in meta[k] if x != i]
    elif op == "title":
        t = _clean_title(title)
        if t:
            meta["titles"][i] = t
        else:
            meta["titles"].pop(i, None)
    else:
        raise ValueError("unknown library op")
    save_meta(art_dir, meta)
    return meta


def forget(art_dir, kind, name):
    """Drop every flag of an image that left the library (art deleted elsewhere, texture removed)."""
    i, meta = f"{kind}:{name}", load_meta(art_dir)
    before = json.dumps(meta, sort_keys=True)
    meta["titles"].pop(i, None)
    meta["favs"] = [x for x in meta["favs"] if x != i]
    meta["picked"] = [x for x in meta["picked"] if x != i]
    if kind == "art" and not (Path(art_dir) / name).exists():
        meta["imported"] = {k: v for k, v in meta["imported"].items() if v != name}
    else:
        meta["imported"].pop(i, None)
    if json.dumps(meta, sort_keys=True) == before:
        return                                        # nothing to forget: no file written
    try:
        save_meta(art_dir, meta)
    except OSError:
        pass


def delete(art_dir, tex_dir, kind, name):
    """Remove one image and its depth map, forget its flags. True if the image is gone."""
    p = resolve(kind, name, art_dir, tex_dir)
    try:
        p.unlink()
    except OSError:
        return False
    for f in (p.with_name(p.stem + DEPTH_SUFFIX), Path(art_dir) / ".thumbs" / (name + ".jpg")):
        try:
            f.unlink()
        except OSError:
            pass
    forget(art_dir, kind, name)
    return True


# ---------------------------------------------------------------- quota + AI image -> art piece

def check_quota(art_dir, add_bytes=0):
    """ValueError when one more piece of add_bytes would pass the art library caps."""
    names = list_art(art_dir)
    if len(names) + 1 > ART_CAP_FILES:
        raise ValueError(f"the art library is full ({ART_CAP_FILES} pieces): delete some first")
    total = 0
    for n in names:
        try:
            total += (Path(art_dir) / n).stat().st_size
        except OSError:
            pass
    if total + add_bytes > ART_CAP_BYTES:
        raise ValueError(f"the art library is full ({ART_CAP_BYTES >> 30} GB): delete some first")


def _stem(title, fallback):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", title or "")[:40].strip("_-")
    return s or re.sub(r"[^A-Za-z0-9_-]+", "_", Path(fallback).stem)[:40].strip("_-") or "ai"


def import_ai(tex_dir, art_dir, name, title=None):
    """Copy textures/<name> into the art library as a PNG piece (appended: art_index values keep their piece).
    The same AI image twice is the same piece. Title = `title`, else the prompt in its PNG, else its stem."""
    src = resolve("ai", name, art_dir, tex_dir)
    i, meta = "ai:" + name, load_meta(art_dir)
    have = meta["imported"].get(i)
    if have and have in list_art(art_dir):
        if title is not None and _clean_title(title):
            meta["titles"]["art:" + have] = _clean_title(title)
            save_meta(art_dir, meta)
        return have
    t = _clean_title(title) or meta["titles"].get(i) or png_prompt(src) or Path(name).stem
    check_quota(art_dir, src.stat().st_size)
    from PIL import Image
    try:
        with Image.open(src) as im:
            if im.size[0] < 1 or im.size[1] < 1 or im.size[0] * im.size[1] > MAX_PIXELS:
                raise ValueError("image dimensions are out of range")
            im.load()
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "PA") else "RGB")
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    except ValueError:
        raise
    except Exception as ex:
        raise ValueError(f"not a readable image ({type(ex).__name__})")
    folder = Path(art_dir)
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    for _ in range(8):
        out = f"{stamp}_{_stem(t, name)}_{secrets.token_hex(2)}.png"
        if not (folder / out).exists():
            break
    tmp = folder / f".{out}.{os.getpid()}.tmp"
    try:
        im.save(tmp, format="PNG")
        os.replace(tmp, folder / out)
    except OSError as ex:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ValueError(f"could not save ({type(ex).__name__})")
    meta = load_meta(art_dir)                         # re-read: another request may have written meanwhile
    meta["imported"][i] = out
    meta["titles"]["art:" + out] = t
    save_meta(art_dir, meta)
    return out


# ---------------------------------------------------------------- the AI-texture layer's image set

def layer_key(kind, name):
    return ("art/" + name) if kind == "art" else name


def split_key(key):
    return ("art", key[4:]) if key.startswith("art/") else ("ai", key)


def _finished(p, now):
    try:
        return now - p.stat().st_mtime > 2.0          # still being written: not yet
    except OSError:
        return False


def layer_files(img_set, tex_dir, art_dir, picked=(), extra=()):
    """[(key, path)] in key order for the AI-texture layer. 0 = textures/ exactly as before; 1 = art uploads;
    2 = both; 3 = the picked ids. An empty set falls back to textures/. `extra` keys (a show request) are added."""
    now = time.time()
    ai = [(n, Path(tex_dir) / n) for _, n in _tex_names(tex_dir)]
    ai = [(k, p) for k, p in ai if _finished(p, now)]
    s = int(round(img_set)) if isinstance(img_set, (int, float)) and img_set == img_set else 0
    if s == 0 and not extra:
        return sorted(ai)
    art = [("art/" + n, Path(art_dir) / n) for n in list_art(art_dir)]
    art = [(k, p) for k, p in art if _finished(p, now)]
    if s == 1:
        out = art
    elif s == 2:
        out = ai + art
    elif s == 3:
        want = {layer_key(*x.split(":", 1)) for x in picked if isinstance(x, str) and ":" in x}
        out = [kp for kp in ai + art if kp[0] in want]
    else:
        out = ai
    if not out:
        out = ai
    keys = {k for k, _ in out}
    out += [kp for kp in ai + art if kp[0] in set(extra) and kp[0] not in keys]
    return sorted(out)
