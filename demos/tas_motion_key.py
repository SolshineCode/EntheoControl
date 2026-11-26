"""TAS-style "Motion Key" demo: static rainbow-encoded geometry + a sliding hue window = chasing light.

  python demos/tas_motion_key.py --monitor 1        (borderless on projector, over the show; ESC closes)
  python demos/tas_motion_key.py --strip out.png    (headless filmstrip)

Keys: 1 source (the rainbow-encoded art, what TAS paints in Illustrator)
      2 motion key (one sliding window)      3 dual window + mirror chase      4 auto (cycles 1-3 every 8 bars)
      UP/DOWN window width   LEFT/RIGHT chase speed   F folds   P palette   ESC quit
"""
import argparse
import time

import glfw
import moderngl
import numpy as np
from PIL import Image

FS = """
#version 330
uniform vec2 res; uniform float beat, mode, width, folds, pal;
out vec4 f;
const float TAU = 6.28318530718;
vec3 rainbow(float h){ return clamp(abs(fract(h+vec3(0,2,1)/3.)*6.-3.)-1., 0., 1.); }
vec3 jewel(float h, float p){
  // jewel palettes (Aya / Cosmic / Gorge); palette 0 = raw spectrum like TAS colour shapes
  vec3 a = vec3(.5), b = vec3(.5);
  vec3 c = p < .5 ? vec3(1) : p < 1.5 ? vec3(1,.9,.8) : p < 2.5 ? vec3(.9,1,1.1) : vec3(1,1,.9);
  vec3 d = p < .5 ? vec3(0,.33,.67) : p < 1.5 ? vec3(.5,.1,.85) : p < 2.5 ? vec3(.1,.55,.95) : vec3(.6,.25,.05);
  vec3 col = a + b*cos(TAU*(c*h + d));
  return normalize(col+1e-3)*1.6*max(max(col.r,col.g),col.b);
}
// distance-like "layer coordinate": blend of circle, stepped kene diamond and petals, in folded polar space
float layerCoord(vec2 p, out float ang){
  float r = length(p), a = atan(p.y, p.x), s = TAU/folds;
  float af = abs(mod(a, s) - s*.5);                   // fold
  vec2 q = r*vec2(cos(af), sin(af));
  float diamond = (abs(q.x)+abs(q.y))*0.75;
  float stepped = floor(diamond*18.)/18. + 0.02*step(.5, fract(diamond*18.));   // kene-like stepped edges
  float petal = r*(1.0 - 0.18*cos(folds*a));
  ang = a;
  return mix(mix(r, stepped, .55), petal, .35);
}
void main(){
  vec2 p = (gl_FragCoord.xy - .5*res)/res.y * 2.2;
  float ang; float d = layerCoord(p, ang);
  const float N = 48.;                                 // number of nested layers (each gets its own hue)
  float k = floor(d*N), fr = fract(d*N);
  float px = fwidth(d*N);
  float line = smoothstep(.5+1.5*px, .5, abs(fr-.5)*2.) ;              // filled band edges
  float outline = 1. - smoothstep(0.6*px, 1.8*px, min(fr, 1.-fr));     // crisp contour lines
  float idx = k/N;                                       // hue index 0..1, inner -> outer (the "colour shape" encoding)
  idx = fract(idx + 0.03*sin(ang*folds));               // slight petal offset so waves ripple, not just rings
  float inside = step(d, 1.0) * step(.03, d);
  vec3 col;
  if (mode < .5) {
    col = rainbow(idx) * (0.1 + 0.9*outline) * inside;                  // the static source
  } else {
    float ph = beat/4.;                                                    // one sweep per bar
    float w = mode < 1.5 ? 1.0 - abs(fract(idx - ph) - .5)*2. :           // single window (wraps)
              max(1.0 - abs(fract(idx*2. - ph*2.) - .5)*2.,               // dual repeat...
                  1.0 - abs(fract(-idx*2. - ph*2. + .5) - .5)*2.);        // ...plus counter-chase
    float key = smoothstep(1.0 - width, 1.0, w);                            // soft-edged hue window
    float kick = exp(-fract(beat)*5.);                                       // beat pulse
    vec3 c = pal < .5 ? rainbow(idx - ph*.25) : jewel(idx + ph*.15, pal);
    col = c * key * (0.08 + 0.92*outline) * (1.0 + 0.6*kick) * inside;
    col += c * 0.04 * outline * inside;                                     // faint ghost of the full structure
  }
  f = vec4(col, 1.);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=None)
    ap.add_argument("--bpm", type=float, default=124.0)
    ap.add_argument("--strip", default=None, help="render a headless filmstrip PNG and exit")
    a = ap.parse_args()
    glfw.init()
    for h, v in ((glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE), (glfw.OPENGL_FORWARD_COMPAT, True)):
        glfw.window_hint(h, v)
    W, H = 1280, 800
    if a.strip:
        glfw.window_hint(glfw.VISIBLE, False)
        win = glfw.create_window(W, H, "strip", None, None)
    elif a.monitor is not None and a.monitor < len(glfw.get_monitors()):
        m = glfw.get_monitors()[a.monitor]; vm = glfw.get_video_mode(m); mx, my = glfw.get_monitor_pos(m)
        for hint, val in ((glfw.DECORATED, False), (glfw.AUTO_ICONIFY, False), (glfw.FLOATING, True)):
            glfw.window_hint(hint, val)
        W, H = vm.size.width, vm.size.height
        win = glfw.create_window(W, H, "TAS motion key demo", None, None); glfw.set_window_pos(win, mx, my)
    else:
        win = glfw.create_window(W, H, "TAS motion key demo", None, None)
    glfw.make_context_current(win); glfw.swap_interval(1)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}", fragment_shader=FS)
    vao = ctx.vertex_array(prog, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4")), "2f", "p")])
    st = dict(mode=4, width=0.22, speed=1.0, folds=8.0, pal=0.0)

    def key(w, k, s, act, mod):
        if act not in (glfw.PRESS, glfw.REPEAT):
            return
        if k == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
        elif glfw.KEY_1 <= k <= glfw.KEY_4: st["mode"] = k - glfw.KEY_1
        elif k == glfw.KEY_UP: st["width"] = min(.95, st["width"] + .05)
        elif k == glfw.KEY_DOWN: st["width"] = max(.05, st["width"] - .05)
        elif k == glfw.KEY_RIGHT: st["speed"] = min(4, st["speed"] * 1.25)
        elif k == glfw.KEY_LEFT: st["speed"] = max(.125, st["speed"] / 1.25)
        elif k == glfw.KEY_F: st["folds"] = {6.0: 8.0, 8.0: 12.0, 12.0: 16.0, 16.0: 6.0}[st["folds"]]
        elif k == glfw.KEY_P: st["pal"] = (st["pal"] + 1) % 4
        print({k2: round(v, 2) for k2, v in st.items()})
    glfw.set_key_callback(win, key)

    def draw(beat, mode):
        ctx.viewport = (0, 0, *glfw.get_framebuffer_size(win)) if not a.strip else (0, 0, W, H)
        prog["res"].value = (W, H); prog["beat"].value = beat; prog["mode"].value = float(mode)
        prog["width"].value = st["width"]; prog["folds"].value = st["folds"]; prog["pal"].value = st["pal"]
        vao.render(moderngl.TRIANGLE_STRIP)

    if a.strip:
        fbo = ctx.simple_framebuffer((W, H)); fbo.use()
        tiles = []
        for mode, pal, beats in ((0, 0, [0]), (1, 0, [0, .5, 1, 1.5]), (2, 1, [0, .7, 1.4]), (1, 2, [.3]), (2, 3, [.9])):
            st["pal"] = pal
            for b in beats:
                draw(b, mode)
                tiles.append(Image.frombytes("RGB", (W, H), fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM))
        tw, th = 426, 266
        sheet = Image.new("RGB", (tw * 5, th * 2))
        for i, im in enumerate(tiles[:10]):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), ((i % 5) * tw, (i // 5) * th))
        sheet.save(a.strip); print("saved", a.strip)
        return

    t0, beat, last = time.perf_counter(), 0.0, time.perf_counter()
    while not glfw.window_should_close(win):
        now = time.perf_counter(); dt = now - last; last = now
        beat += dt * a.bpm / 60 * st["speed"]
        mode = st["mode"] if st["mode"] < 3 else int(beat / 32) % 3       # auto: 8 bars each
        draw(beat, mode); glfw.swap_buffers(win); glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
