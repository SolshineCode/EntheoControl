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
    r.render(sset, disp_texture, target_fbo, (fb_w, fb_h), t, bass, grid=False)
"""
import math

import moderngl
import numpy as np

import surfaces as S

MAXP = S.MAX_POLY
SCENE_UNIT, LABEL_UNIT = 0, 11

VS = """
#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FS = """
#version 330
uniform sampler2D scene; uniform sampler2D label;
uniform mat3 M;                 // framebuffer pixel -> source uv (v up)
uniform mat3 Lm;                // framebuffer pixel -> the surface's own 0..1 square
uniform vec2 poly[%(MAXP)d]; uniform int n;
uniform int content;            // 0 main, 1 solid, 2 edge glow only, 3 black
uniform vec3 color;
uniform float opacity, brightness, feather, glow, lineW, perim, time, bass;
uniform float ovl;              // 0 off, 1 outline + label, 2 + test grid
uniform float sel;
uniform vec4 labelBox;          // x, y, w, h in pixels (w 0 = no label)
uniform vec2 corner0;           // the quad's TL corner (orientation mark in the overlay)
out vec4 f;
float max3(vec3 v){ return max(v.x, max(v.y, v.z)); }
float lineAA(float d, float w){ return 1.0 - smoothstep(w, w + 1.5, d); }
void main(){
  vec2 p = gl_FragCoord.xy;
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

  vec3 q = M * vec3(p, 1.0);
  vec2 uv = q.xy / q.z;
  vec3 live = texture(scene, uv).rgb;
  vec3 col = content == 0 ? live : (content == 1 ? color : vec3(0.0));
  col *= brightness;

  if (glow > 0.0) {
    // a neon frame just inside the edge: hot core line, soft bloom falling into the face, a faint second
    // line, a jewel at every corner, and two comets of light running round the perimeter. All of it stays
    // inside the outline, so nothing spills off the object.
    float u = arc / max(perim, 1.0);
    float core = exp(-pow((inside - lineW) / (lineW * 0.75), 2.0));
    float halo = exp(-inside / (lineW * 7.0));
    float line2 = exp(-pow((inside - lineW * 5.0) / (lineW * 0.6), 2.0)) * 0.22;
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
        self._key, self._items = None, []
        self._labels = {}                                # surface id -> (key, texture, (w, h))
        self._blank = ctx.texture((1, 1), 4, bytes(4))

    def _prepare(self, sset, fb):
        key = (id(sset), sset.rev, fb)
        if key == self._key:
            return self._items
        items = sset.render_items(*fb)[:S.MAX_SURFACES]
        verts = np.zeros((S.MAX_SURFACES, 4, 2), dtype="f4")
        for k, it in enumerate(items):
            x0, y0, x1, y1 = it["box"]
            nx0, ny0, nx1, ny1 = x0 / fb[0] * 2 - 1, y0 / fb[1] * 2 - 1, x1 / fb[0] * 2 - 1, y1 / fb[1] * 2 - 1
            verts[k] = [(nx0, ny0), (nx1, ny0), (nx0, ny1), (nx1, ny1)]
            pts = np.zeros((MAXP, 2), dtype="f4")
            pts[:it["n"]] = it["poly"]
            it["poly_bytes"] = pts.tobytes()
            it["M_bytes"] = np.array(it["M"], dtype="f8").reshape(3, 3).T.astype("f4").tobytes()
            it["L_bytes"] = np.array(it["L"], dtype="f8").reshape(3, 3).T.astype("f4").tobytes()
            s = sset.get(it["id"])
            cx, cy = s["corners"][0]
            it["corner0"] = (cx * fb[0] / sset.output[0], fb[1] - cy * fb[1] / sset.output[1])
        self.vbo.write(verts.tobytes())
        self._key, self._items = key, items
        return items

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

    def render(self, sset, scene, target, fb, t=0.0, bass=0.0, grid=False):
        """Draw every enabled surface of `sset` into `target` (cleared to black first).
        grid=True is the old GRID / CALIBRATE switch: overlay + test grid on every surface."""
        ctx, prog = self.ctx, self.prog
        fb = (int(fb[0]), int(fb[1]))
        target.use()
        ctx.viewport = (0, 0, fb[0], fb[1])
        target.clear(0.0, 0.0, 0.0, 1.0)
        items = self._prepare(sset, fb)
        if not items:
            return 0
        mode, og = sset.overlay["mode"], sset.overlay["grid"] or grid
        if grid:
            mode = "all"
        order = {s["id"]: i for i, s in enumerate(sset.surfaces)}
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        try:
            scene.use(SCENE_UNIT)
            prog["scene"].value = SCENE_UNIT
            prog["label"].value = LABEL_UNIT
            prog["time"].value = float(t % 3600.0)
            prog["bass"].value = float(bass)
            for k, it in enumerate(items):
                show = mode == "all" or (mode == "sel" and it["selected"])
                prog["M"].write(it["M_bytes"])
                prog["Lm"].write(it["L_bytes"])
                prog["poly"].write(it["poly_bytes"])
                prog["n"].value = it["n"]
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
        finally:
            ctx.disable(moderngl.BLEND)
            ctx.blend_func = moderngl.DEFAULT_BLENDING
        return len(items)

    def release(self):
        for _, tex, _ in self._labels.values():
            tex.release()
        self._labels.clear()
        for obj in (self._blank, self.vao, self.vbo, self.prog):
            obj.release()
