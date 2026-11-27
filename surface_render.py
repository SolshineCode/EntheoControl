"""Multi-surface output stage: draws the rendered frame onto every mapped surface, black everywhere else.

The visual is rendered ONCE (the engine's `disp` texture). Each enabled surface in a surfaces.SurfaceSet is
then one small draw: a screen-aligned box around its outline, and a fragment shader that
  * finds the signed distance to the outline polygon (quad, or the N-point mask cut by the quad),
  * maps the pixel through the surface's homography into the frame (full frame / own sub-rect / spanned),
  * applies content mode, brightness, opacity, soft edge (feather) and the edge glow,
  * optionally the find-it overlay: outline in the surface's colour, test grid, name label.
Surfaces composite back to front (premultiplied "over"), so overlapping faces hide what is behind them.
Cost: one draw + at most two texture reads per covered pixel per surface; 12 surfaces is far below 1 ms.

    r = SurfaceRenderer(ctx)
    r.render(sset, disp_texture, target_fbo, (fb_w, fb_h), t, bass, grid=False, hub=layer_hub, blocked=set())

PER-SURFACE LAYERS (content "layers"): up to 3 materials.LayerHub textures (or the main picture) stacked with
normal / add / screen / multiply, laid out in the surface's own 0..1 square (mirrored tiling, scale, scroll, hue).
Each source is requested from the hub ONCE per frame however many surfaces use it, and the hub renders it at most
once; a surface that is not in "layers" mode never touches the hub (zero cost). `blocked` = sources whose engine
layer is bypassed: they draw nothing on surfaces either. The hub runs at the end of the frame: one frame of latency.
MASK SURFACES (shape "mask"): the outline is a label image (RGBA: label, flow u, flow v, coverage) in projector
pixels; consecutive parts of one file with the same look are ONE draw (a 256-bit label set), whatever their number.
"""
import functools
import math

import moderngl
import numpy as np

import surfaces as S

MAXP = S.MAX_POLY
SCENE_UNIT, LABEL_UNIT, MASK_UNIT = 0, 11, 12
HUB_UNIT = 23                    # materials.UNIT_L (the LayerHub array)
MAIN_IDX = -2.0
# per-surface layer source -> the engine params whose bypass switches it off (control.BYPASS_COVER covers params)
SOURCE_PARAMS = dict(kene=("l_kene_shipibo",), kene4=("l_kene_shipibo",), beadstudy=("l_kene_hunikuin",),
                     blocks=("l_kene",), beads=("l_beads",), lines=("l_lines",), beings=("l_beings",),
                     fractal=("l_fractal",), land=("l_land",), stack=("l_stack",), art=("l_art",), ai=("l_ai",),
                     text=("l_text",), video=("l_video",), camera=("l_camera",))


def blocked_sources(bypass_keys, cover):
    """Bypassed keys (controller.bypass) + control.BYPASS_COVER -> the per-surface sources that must show nothing."""
    gone = set()
    for k in bypass_keys or ():
        gone.update(cover.get(k, ()))
    return {src for src, ps in SOURCE_PARAMS.items() if any(q in gone for q in ps)}

VS = """
#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FS = """
#version 330
uniform sampler2D scene; uniform sampler2D label; uniform sampler2D mtex;
uniform sampler2DArray texL;    // LayerHub array (per-surface layers)
uniform int nl;                 // layers in this draw (content 4)
uniform vec4 lA[3];             // array layer (-2 = main picture), blend, opacity, hue (rad)
uniform vec4 lB[3];             // scale, scroll offset u, v, -
uniform int useMask;            // 1: shape "mask": label image + 256-bit label set
uniform uint lbits[8];
uniform vec2 fbSize;
uniform mat3 M;                 // framebuffer pixel -> source uv (v up)
uniform mat3 Lm;                // framebuffer pixel -> the surface's own 0..1 square
uniform vec2 poly[%(MAXP)d]; uniform int n;
uniform int content;            // 0 main, 1 solid, 2 edge glow only, 3 black, 4 own layers
uniform vec3 color;
uniform float opacity, brightness, feather, glow, lineW, perim, time, bass;
uniform float ovl;              // 0 off, 1 outline + label, 2 + test grid
uniform float sel;
uniform vec4 labelBox;          // x, y, w, h in pixels (w 0 = no label)
uniform vec2 corner0;           // the quad's TL corner (orientation mark in the overlay)
out vec4 f;
float max3(vec3 v){ return max(v.x, max(v.y, v.z)); }
float lineAA(float d, float w){ return 1.0 - smoothstep(w, w + 1.5, d); }
vec3 hueRot(vec3 c, float a){                         // luminance-preserving hue rotation (feColorMatrix hueRotate)
  float k = cos(a), q = sin(a);
  return max(vec3(dot(c, vec3(0.213 + 0.787 * k - 0.213 * q, 0.715 - 0.715 * k - 0.715 * q, 0.072 - 0.072 * k + 0.928 * q)),
                  dot(c, vec3(0.213 - 0.213 * k + 0.143 * q, 0.715 + 0.285 * k + 0.140 * q, 0.072 - 0.072 * k - 0.283 * q)),
                  dot(c, vec3(0.213 - 0.213 * k - 0.787 * q, 0.715 - 0.715 * k + 0.715 * q, 0.072 + 0.928 * k + 0.072 * q))), 0.0);
}
void main(){
  vec2 p = gl_FragCoord.xy;
  // layer uv: the surface's own square, or (mask) the part's flow uv; derivatives taken before any discard
  vec3 lq0 = Lm * vec3(p, 1.0);
  vec2 fuv = lq0.xy / lq0.z;
  float mcov = 1.0; bool mhit = true;
  if (useMask == 1) {
    vec4 m = texture(mtex, p / fbSize);
    uint l = uint(m.r * 255.0 + 0.5);
    mhit = l > 0u && ((lbits[l >> 5u] >> (l & 31u)) & 1u) == 1u;
    mcov = m.a; fuv = m.gb;
  }
  vec2 gx = dFdx(fuv), gy = dFdy(fuv);
  if (useMask == 1 && dot(gx, gx) + dot(gy, gy) > 0.01) { gx = vec2(0.004, 0.0); gy = vec2(0.0, 0.004); }  // part edges
  if (!mhit || mcov <= 0.0) discard;
  // signed distance to the outline (negative inside), arc length of the nearest edge point, nearest vertex
  float d = 1e20, s = 1.0, acc = 0.0, arc = 0.0, dv = 1e20;
  for (int i = 0; i < n; i++) {
    vec2 a = poly[i], b = poly[i + 1 < n ? i + 1 : 0];
    vec2 e = b - a, w = p - a;
    float el = length(e), t = clamp(dot(w, e) / max(dot(e, e), 1e-9), 0.0, 1.0);
    vec2 r = w - e * t;
    float dd = dot(r, r);
    if (dd < d) { d = dd; arc = acc + t * el; }
    acc += el;
    dv = min(dv, dot(w, w));
    bvec3 c = bvec3(p.y >= a.y, p.y < b.y, e.x * w.y > e.y * w.x);
    if (all(c) || all(not(c))) s = -s;
  }
  float inside = -s * sqrt(d);                      // pixels in from the edge
  float cov = clamp(inside + 0.5, 0.0, 1.0);        // 1 px anti-aliased hard edge: whole pixels inside are exactly 1
  if (cov <= 0.0) discard;                          // nothing outside a surface is ever lit
  float a = cov;
  if (feather > 0.0) { float k = clamp((inside + 0.5) / (feather + 1.0), 0.0, 1.0); a = k * k * (3.0 - 2.0 * k); }
  a *= mcov;

  vec3 q = M * vec3(p, 1.0);
  vec2 uv = q.xy / q.z;
  vec3 live = texture(scene, uv).rgb;
  vec3 col = content == 0 ? live : (content == 1 ? color : vec3(0.0));
  if (content == 4) {
    for (int k = 0; k < nl; k++) {
      vec4 A = lA[k], B = lB[k];
      vec3 c;
      if (A.x < -1.5) c = live;
      else {
        vec2 u = (fuv - 0.5) * B.x + 0.5 + B.yz;
        vec2 m2 = 1.0 - abs(mod(u, 2.0) - 1.0);                     // mirrored tiling: any layer is seamless
        c = textureGrad(texL, vec3(m2, A.x), gx * B.x, gy * B.x).rgb;
      }
      if (A.w != 0.0) c = hueRot(c, A.w);
      int b = int(A.y + 0.5);
      if (b == 0) col = mix(col, c, A.z);
      else if (b == 1) col += c * A.z;
      else if (b == 2) col = 1.0 - (1.0 - col) * (1.0 - c * A.z);
      else col = mix(col, col * c, A.z);
    }
  }
  col *= brightness;

  if (glow > 0.0) {
    // a neon frame just inside the edge: hot core line, soft bloom falling into the face, a faint second
    // line, a jewel at every corner, and two comets of light running round the perimeter. All of it stays
    // inside the outline, so nothing spills off the object.
    float u = arc / max(perim, 1.0);
    float kc = (inside - lineW) / (lineW * 0.75), k2 = (inside - lineW * 5.0) / (lineW * 0.6);
    float core = exp(-kc * kc);
    float halo = exp(-inside / (lineW * 7.0));
    float line2 = exp(-k2 * k2) * 0.22;
    float jewel = exp(-sqrt(dv) / (lineW * 3.5));
    float ph = fract(u * 2.0 - time * 0.11);
    float comet = exp(-(1.0 - ph) * 6.0) + 0.5 * exp(-ph * 40.0);
    float pulse = 0.72 + 0.28 * clamp(bass, 0.0, 1.0);
    float shimmer = 0.85 + 0.15 * sin(u * 37.699 + time * 1.7);
    vec3 hot = live / max(max3(live), 0.35);                       // the show's own colour at this spot
    vec3 tint = mix(color, hot, clamp(max3(live) * 2.5, 0.0, 0.65));
    vec3 g = tint * (core * (0.75 + 1.1 * comet) + halo * 0.34 * (0.6 + 0.9 * comet) * shimmer + line2 + jewel * 0.9) * pulse
           + vec3(1.0) * (core * core * 0.55 * (0.5 + comet) + jewel * jewel * 0.6);
    col += g * glow * brightness;
  }

  float o = opacity;
  if (ovl > 0.5) {                                   // find-it overlay: always full strength, hard edge
    a = cov; o = 1.0;
    vec3 lq = Lm * vec3(p, 1.0);
    vec2 lc = lq.xy / lq.z;
    if (ovl > 1.5) {
      vec2 gp = lc * 8.0, gd = (0.5 - abs(fract(gp) - 0.5)) / max(fwidth(gp), vec2(1e-6));
      float gl = lineAA(min(gd.x, gd.y), 0.6);
      float fw = max(fwidth(lc.x) + fwidth(lc.y), 1e-6);
      float dg = lineAA(min(abs(lc.x - lc.y), abs(lc.x + lc.y - 1.0)) / fw, 0.4);
      float ring = lineAA(abs(length(lc - 0.5) - 0.25) / fw, 0.4);
      float cross_ = lineAA(min(abs(lc.x - 0.5), abs(lc.y - 0.5)) / fw, 0.9);
      col = color * 0.10 + color * gl * 0.9 + vec3(1.0) * max(max(dg, ring) * 0.45, cross_ * 0.8);
    }
    float ants = sel > 0.5 ? step(0.5, fract(arc / 18.0 - time * 1.6)) : 0.0;     // the selected one marches
    col = mix(col, mix(color, vec3(1.0), ants), lineAA(inside, 2.5));
    col = mix(col, color, lineAA(sqrt(dv), 6.0));                              // corner dots
    float c0 = distance(p, corner0);
    col = mix(col, vec3(1.0), lineAA(c0, 9.0));                                // corner 1 (TL) = big white dot
    if (labelBox.z > 0.0) {
      vec2 lt = (p - labelBox.xy) / labelBox.zw;
      if (all(greaterThanEqual(lt, vec2(0.0))) && all(lessThanEqual(lt, vec2(1.0)))) {
        vec4 t = texture(label, lt);
        col = mix(col, t.rgb, t.a);
      }
    }
  }
  f = vec4(col * a * o, a * o);
}
""" % dict(MAXP=MAXP)


@functools.lru_cache(maxsize=4)
def _font(px):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def label_image(text, color, height=56):
    """RGBA name tag: the text in the surface's colour on a dark rounded plate."""
    from PIL import Image, ImageDraw
    font = _font(int(height * 0.62))
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    w = int(x1 - x0 + height * 0.7)
    img = Image.new("RGBA", (max(w, height), height), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.rounded_rectangle((0, 0, img.width - 1, height - 1), radius=height // 4, fill=(0, 0, 0, 215))
    rgb = tuple(int(max(0.0, min(1.0, c)) * 255) for c in color)
    dr.text(((img.width - (x1 - x0)) / 2 - x0, (height - (y1 - y0)) / 2 - y0), text, font=font, fill=rgb + (255,))
    return img


class SurfaceRenderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.vbo = ctx.buffer(reserve=S.MAX_SURFACES * 4 * 2 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self._key, self._items, self._draws = None, [], []
        self._labels = {}                                # surface id -> (key, texture, (w, h))
        self._masks = {}                                 # mask file -> (mtime, texture)
        self._blank = ctx.texture((1, 1), 4, bytes(4))
        self._blank_arr = ctx.texture_array((1, 1, 1), 4, bytes(4))
        self.hub_requests = 0                            # tests: LayerHub.request calls made (0 when nothing uses it)
        self.draw_calls = 0

    def _mask(self, sset, name):
        """The label image of a mask group (cached; reloaded when the file changes)."""
        path = getattr(sset, "mask_dir", S.MASK_DIR) / name
        try:
            mt = path.stat().st_mtime_ns
        except OSError:
            mt = None
        have = self._masks.get(name)
        if have is not None and have[0] == mt:
            return have[1]
        if have is not None and have[1] is not None:
            have[1].release()
        tex = None
        if mt is not None:
            try:
                from PIL import Image
                img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
                tex = self.ctx.texture(img.size, 4, img.tobytes())
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                tex.repeat_x = tex.repeat_y = False
            except Exception as ex:
                print(f"[map] mask {name} unreadable: {ex}")
        else:
            print(f"[map] mask file missing: {path}")
        self._masks[name] = (mt, tex)
        return tex

    def _prepare(self, sset, fb, show_fn):
        key = (id(sset), sset.rev, fb, show_fn("key"))
        if key == self._key:
            return self._items
        items = sset.render_items(*fb)[:S.MAX_SURFACES]
        for it in items:
            pts = np.zeros((MAXP, 2), dtype="f4")
            pts[:it["n"]] = it["poly"]
            it["poly_bytes"] = pts.tobytes()
            it["M_bytes"] = np.array(it["M"], dtype="f8").reshape(3, 3).T.astype("f4").tobytes()
            it["L_bytes"] = np.array(it["L"], dtype="f8").reshape(3, 3).T.astype("f4").tobytes()
            s = sset.get(it["id"])
            cx, cy = s["corners"][0]
            it["corner0"] = (cx * fb[0] / sset.output[0], fb[1] - cy * fb[1] / sset.output[1])
            it["show"] = show_fn(it)
        # draws: one per surface, except consecutive mask parts of one file with the same look (and no overlay):
        # those are ONE draw over their joint bounding box with the union of their labels
        draws = []
        for it in items:
            if it.get("maskfile"):
                it["mtex"] = self._mask(sset, it["maskfile"])
                if it["mtex"] is None:
                    continue
            look = (it.get("maskfile"), it["content"], it["color"] if it["content"] in (1, 2) else None, it["opacity"],
                    it["brightness"], it["layers"] if it["content"] == 4 else None, it["M"])
            d = draws[-1] if draws else None
            if (d is not None and it.get("maskfile") and not it["show"] and not d["it"]["show"]
                    and d["look"] == look and d.get("bits") is not None):
                d["bits"][it["label"] >> 5] |= 1 << (it["label"] & 31)
                b = d["box"]
                d["box"] = (min(b[0], it["box"][0]), min(b[1], it["box"][1]), max(b[2], it["box"][2]), max(b[3], it["box"][3]))
                d["n"] += 1
                continue
            d = dict(it=it, look=look, box=it["box"], n=1, bits=None)
            if it.get("maskfile"):
                d["bits"] = [0] * 8
                d["bits"][it["label"] >> 5] |= 1 << (it["label"] & 31)
            draws.append(d)
        verts = np.zeros((S.MAX_SURFACES, 4, 2), dtype="f4")
        for k, d in enumerate(draws):
            x0, y0, x1, y1 = d["box"]
            nx0, ny0, nx1, ny1 = x0 / fb[0] * 2 - 1, y0 / fb[1] * 2 - 1, x1 / fb[0] * 2 - 1, y1 / fb[1] * 2 - 1
            verts[k] = [(nx0, ny0), (nx1, ny0), (nx0, ny1), (nx1, ny1)]
            if d["bits"] is not None:
                if d["n"] > 1:                           # the joint box is the outline of a merged draw
                    pts = np.zeros((MAXP, 2), dtype="f4")
                    pts[:4] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    d["poly_bytes"], d["np"] = pts.tobytes(), 4
                d["bits_v"] = tuple(int(b) for b in d["bits"])
        self.vbo.write(verts.tobytes())
        self._key, self._items, self._draws = key, items, draws
        self._prune({s["id"] for s in sset.surfaces})
        used = {it["maskfile"] for it in items if it.get("maskfile")}
        for name in [n for n in self._masks if n not in used]:
            t = self._masks.pop(name)[1]
            if t is not None:
                t.release()
        return items

    def _prune(self, alive):
        """Release the name tags of surfaces that no longer exist."""
        for sid in [k for k in self._labels if k not in alive]:
            self._labels.pop(sid)[1].release()

    def _label(self, it, index):
        key = (it["name"], it["color"], index)
        have = self._labels.get(it["id"])
        if have is None or have[0] != key:
            if have is not None:
                have[1].release()
            from PIL import Image
            img = label_image(f"{index + 1}  {it['name']}", it["color"]).transpose(Image.FLIP_TOP_BOTTOM)
            tex = self.ctx.texture(img.size, 4, img.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            have = self._labels[it["id"]] = (key, tex, img.size)
        return have

    def _layer_uniforms(self, layers, t, hub, blocked, idx):
        """-> (n, lA bytes, lB bytes, uses the hub). Sources are requested once per frame (idx cache)."""
        A = np.zeros((3, 4), dtype="f4")
        B = np.zeros((3, 4), dtype="f4")
        n, uses = 0, False
        for (src, op, blend, scale, scroll, cx, cy, hue) in layers[:3]:
            if src == "main":
                i = MAIN_IDX
            else:
                if src in blocked or hub is None:
                    continue
                if src not in idx:
                    idx[src] = hub.request(src)
                    self.hub_requests += 1
                i = idx[src]
                if i < 0:
                    continue                             # not ready yet (first frame) / unknown / hub full
                uses = True
            A[n] = (i, blend, op, hue)
            B[n] = (scale, (scroll * cx * t) % 2.0, (scroll * cy * t) % 2.0, 0.0)
            n += 1
        return n, A.tobytes(), B.tobytes(), uses

    def render(self, sset, scene, target, fb, t=0.0, bass=0.0, grid=False, hub=None, blocked=frozenset()):
        """Draw every enabled surface of `sset` into `target` (cleared to black first).
        grid=True is the old GRID / CALIBRATE switch: overlay + test grid on every surface.
        hub: materials.LayerHub for content "layers"; blocked: sources that are bypassed (show nothing)."""
        ctx, prog = self.ctx, self.prog
        fb = (int(fb[0]), int(fb[1]))
        target.use()
        ctx.viewport = (0, 0, fb[0], fb[1])
        target.clear(0.0, 0.0, 0.0, 1.0)
        mode, og = sset.overlay["mode"], sset.overlay["grid"] or grid
        if grid:
            mode = "all"

        def show_fn(it):
            if it == "key":
                return (mode, og)
            return mode == "all" or (mode == "sel" and it["selected"])
        items = self._prepare(sset, fb, show_fn)
        if not items:
            return 0
        order = {s["id"]: i for i, s in enumerate(sset.surfaces)}
        idx, hub_bound = {}, False
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            scene.use(SCENE_UNIT)
            prog["scene"].value = SCENE_UNIT
            prog["label"].value = LABEL_UNIT
            prog["time"].value = float(t % 3600.0)
            prog["bass"].value = float(bass)
            prog["fbSize"].value = (float(fb[0]), float(fb[1]))
            prog["mtex"].value = MASK_UNIT
            prog["texL"].value = HUB_UNIT

            self.draw_calls = 0
            for k, d in enumerate(self._draws):
                it = d["it"]
                show = it["show"]
                nl = 0
                if it["content"] == 4:
                    nl, la, lb, uses = self._layer_uniforms(it["layers"], t, hub, blocked, idx)
                    if nl:
                        prog["lA"].write(la)
                        prog["lB"].write(lb)
                    if uses and not hub_bound:
                        hub.use()
                        hub_bound = True
                prog["nl"].value = nl
                if d["bits"] is not None:
                    prog["useMask"].value = 1
                    prog["lbits"].value = d["bits_v"]
                    it["mtex"].use(MASK_UNIT)
                else:
                    prog["useMask"].value = 0
                prog["M"].write(it["M_bytes"])
                prog["Lm"].write(it["L_bytes"])
                prog["poly"].write(d.get("poly_bytes", it["poly_bytes"]))
                prog["n"].value = d.get("np", it["n"])
                prog["content"].value = it["content"]
                prog["color"].value = it["color"]
                prog["opacity"].value = it["opacity"]
                prog["brightness"].value = it["brightness"]
                prog["feather"].value = it["feather"]
                prog["glow"].value = it["glow"]
                prog["lineW"].value = it["line"]
                prog["perim"].value = it["perimeter"]
                prog["ovl"].value = (2.0 if og else 1.0) if show else 0.0
                prog["sel"].value = 1.0 if it["selected"] else 0.0
                prog["corner0"].value = it["corner0"]
                box = (0.0, 0.0, 0.0, 0.0)
                if show:
                    _, tex, (lw, lh) = self._label(it, order.get(it["id"], k))
                    h = min(40.0, max(14.0, 0.2 * math.sqrt(it["area"])))
                    w = h * lw / lh
                    box = (it["center"][0] - w / 2, it["center"][1] - h / 2, w, h)
                    tex.use(LABEL_UNIT)
                else:
                    self._blank.use(LABEL_UNIT)
                prog["labelBox"].value = box
                self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=4 * k)
                self.draw_calls += 1
        finally:
            ctx.disable(moderngl.BLEND)
            ctx.blend_func = moderngl.DEFAULT_BLENDING
        return len(items)

    def release(self):
        for _, tex, _ in self._labels.values():
            tex.release()
        self._labels.clear()
        for _, tex in self._masks.values():
            if tex is not None:
                tex.release()
        self._masks.clear()
        for obj in (self._blank, self._blank_arr, self.vao, self.vbo, self.prog):
            obj.release()
