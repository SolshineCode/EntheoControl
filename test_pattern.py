"""Fullscreen focus / alignment test pattern for the projector.

python test_pattern.py --monitor 1      (ESC or Q to close)
Use it to: focus the lens (sharp thin lines + small text), square the projector
physically (circles should be round, border rectangular), with keystone at 0.
"""
import argparse

import glfw
import moderngl
import numpy as np

FS = """
#version 330
uniform vec2 res; uniform float t; out vec4 f;
void main(){
  vec2 px = gl_FragCoord.xy, uv = px/res, c = px - res*0.5;
  vec3 col = vec3(0);
  vec2 g = mod(px, 80.0);  if (min(g.x,g.y) < 1.0) col = vec3(0.35);          // 80 px grid
  vec2 g2 = mod(px, 400.0); if (min(g2.x,g2.y) < 2.0) col = vec3(0.8);         // major grid
  if (px.x < 3. || px.y < 3. || px.x > res.x-4. || px.y > res.y-4.) col = vec3(1,0.2,0.6); // border
  float r = length(c);
  for (int i=1;i<=3;i++){ if (abs(r - float(i)*120.0) < 1.5) col = vec3(0.2,1,0.9); } // circles
  if (abs(c.x) < 1. || abs(c.y) < 1.) col = vec3(1,0.85,0.2);                  // crosshair
  // fine focus patches: 1-px line pairs in each corner
  vec2 q = min(px, res-px);
  if (q.x < 120. && q.y < 120. && q.x > 20. && q.y > 20.) col = vec3(mod(floor(px.x),2.0));
  // slowly rotating jewel-tone sweep so you can judge colour
  if (r < 60.0) col = 0.5+0.5*cos(6.2831*(atan(c.y,c.x)/6.2831 + t*0.1 + vec3(0,.33,.67)));
  f = vec4(col,1);
}
"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--monitor", type=int, default=1); a = ap.parse_args()
    glfw.init()
    for h, v in ((glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE), (glfw.OPENGL_FORWARD_COMPAT, True)):
        glfw.window_hint(h, v)
    ms = glfw.get_monitors(); m = ms[min(a.monitor, len(ms) - 1)]; vm = glfw.get_video_mode(m)
    w = glfw.create_window(vm.size.width, vm.size.height, "pattern", m, None)
    glfw.make_context_current(w); glfw.swap_interval(1)
    ctx = moderngl.create_context(); print("[gl]", ctx.info["GL_RENDERER"], "| monitor:", glfw.get_monitor_name(m), vm.size)
    prog = ctx.program(vertex_shader="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}", fragment_shader=FS)
    vao = ctx.vertex_array(prog, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4")), "2f", "p")])
    glfw.set_key_callback(w, lambda win, k, s, act, mod: glfw.set_window_should_close(win, True)
                          if act == glfw.PRESS and k in (glfw.KEY_ESCAPE, glfw.KEY_Q) else None)
    while not glfw.window_should_close(w):
        fw, fh = glfw.get_framebuffer_size(w); ctx.viewport = (0, 0, fw, fh)
        prog["res"].value = (fw, fh); prog["t"].value = glfw.get_time()
        vao.render(moderngl.TRIANGLE_STRIP); glfw.swap_buffers(w); glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
