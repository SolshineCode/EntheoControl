# EntheoControl is AGPL-3.0 (see LICENSE). The 2D signed-distance helpers and the sine hashes in FS below (smin, sdSeg,
# sdCone, sdEll, sdBox, sdVes, sdPoly8, sdTri, h11, h21) are adapted from Inigo Quilez's articles and shaders
# (https://iquilezles.org/articles/distfunctions2d/, https://iquilezles.org/articles/smin/), which carry this notice:
#
#   The MIT License
#   Copyright (c) 2013-2024 Inigo Quilez
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#   the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
#   OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Beings layer: living and supernatural creatures drawn as luminous line-art (GLSL 330, signed-distance silhouettes
with beadwork / labyrinth ornament inside), moving like living things and reacting to the music.

    bp = BeingsPass(ctx, (1280, 800))
    tex = bp.render(values, clock, t, dt, bands, palette_rgb)     # RGBA8: rgb = light (amount applied), a = silhouette
    bp.release()

The engine (live_visuals.py) calls render() only while `l_beings` > 0 and composites the texture over the scene (the
silhouette darkens what is behind it, the strokes add light), so the layer costs nothing when it is off.

Every form is a generic archetype generated from maths (no artwork, no franchise character is copied; the
Amazonian-cosmology forms are inspired-by only). Motion is integrated on the CPU in double precision (`Herd`), the
shader only receives wrapped phases, so nothing degrades over a long show.

Standalone:  python beings.py --gallery   -> test_frames/beings/<form>.png + contact_sheet.png
             python beings.py --colours   -> colour_modes.png (beings) + colour_modes_serpents.png (engine serpents)
             python beings.py --bench     -> ms per frame for every form at 1280x800
"""
import math
import sys
import time
from pathlib import Path

import moderngl
import numpy as np

HERE = Path(__file__).parent
TAU = 2.0 * math.pi
N_MAX = 8
HEAD, SIDE, FLOAT, TUMBLE, ANCHOR, CLUSTER = range(6)     # how a form moves (see Herd)
# key, panel label, motion, size multiplier, reach (local units from the centre that should stay on screen)
FORMS = [
    ("serpent", "Anaconda spirit", HEAD, 1.00, 1.15),
    ("dragon", "Feathered dragon", HEAD, 1.10, 1.15),
    ("jaguar", "Jaguar spirit", SIDE, 1.15, 0.80),
    ("hummingbird", "Hummingbird", SIDE, 0.95, 0.80),
    ("butterflies", "Butterflies & moths", HEAD, 1.10, 0.90),
    ("jellyfish", "Jellyfish", FLOAT, 1.05, 0.95),
    ("manta", "Manta rays", HEAD, 0.95, 0.90),
    ("koi", "Koi", HEAD, 1.00, 0.80),
    ("phoenix", "Firebird", FLOAT, 1.00, 1.00),
    ("eyes", "All-seeing eyes", FLOAT, 0.90, 0.85),
    ("angel", "Light beings", FLOAT, 1.00, 1.00),
    ("jester", "Jester sprites", TUMBLE, 0.85, 1.00),
    ("octopus", "Octopus", FLOAT, 0.95, 1.00),
    ("vine", "Spirit vine", ANCHOR, 1.00, 0.30),
    ("fireflies", "Fireflies", CLUSTER, 1.15, 0.90),
    ("whale", "Whale", SIDE, 1.45, 0.85),
    ("kitten", "Kittens", SIDE, 0.72, 1.35),
]
FORM_KEYS = [f[0] for f in FORMS]
FORM_LABELS = [f[1] for f in FORMS]
BOUND = 1.7                                # local-unit radius outside which a being draws nothing
PHASE_WRAP = TAU * 64.0                    # animation phases reach the shader wrapped here (multipliers are k/8... k/1)
FADE_S = 0.4                               # a slot fades out / in over this long when its form or the count changes
AREA_BUDGET = 4.2                          # most summed cull-box area on show, content units (the frame is 1.6 x 1)
LOW_RES, LOW_RES_ON, LOW_RES_OFF = 0.75, 0.80, 0.62    # draw at 0.75x while the load is above 80 % of the budget
MIN_RES, MIN_RES_ON, MIN_RES_OFF = 0.60, 1.45, 1.25    # and at 0.6x far over it (only a wall of vines gets there)
# cull box per form: local half-extents, measure_extents() + margin (tests check they still cover the picture)
COLOR_MODES = ["Show palette", "Single hue", "Two-tone", "Rainbow", "Each its own hue", "Light chase"]
FORM_EXT = [
    (1.66, 1.09), (1.70, 1.47), (1.59, 1.36), (1.31, 1.42), (1.53, 1.59), (1.02, 1.39),
    (1.67, 1.52), (1.48, 1.08), (1.44, 1.70), (1.55, 1.21), (1.36, 1.65), (1.63, 1.62),
    (1.56, 1.54), (1.13, 1.70), (1.68, 1.64), (1.70, 1.14),
    (1.68, 1.46),
]
GALLERY_PALETTE = [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]

VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }
"""

FS = """
#version 330
in vec2 v_uv; out vec4 f_col;
uniform vec2 uRes;
uniform int uN;
uniform vec4 uA[8];        // x, y (content units: x +-aspect/2, y +-0.5), rotation, scale
uniform vec4 uB[8];        // animation phase, form, seed, facing (-1..1, side views flip through 0)
uniform vec4 uC[8];        // body bend (turning), gait phase, growth 0..1, fade 0..1
uniform vec3 uPal[5];
uniform float uCMode, uHue, uHue2, uSat, uBright, uChasePh, uChaseW;   // colour mode 0 = show palette (as ever)
uniform float uT, uBass, uMid, uHigh, uAmt, uWig, uGlow, uDetail, uZoom;
uniform float uCull;       // 1 = cull to the per-form box, large = off (measuring)
uniform vec2 uScroll;      // ember / breath drift, wrapped on the CPU where the sparkle lattice repeats
const float PI = 3.14159265, TAU = 6.2831853, TURN = 0.15915494;
const vec3 W = vec3(1.0, 0.97, 0.90);

vec3 gC; float gA, gD, gPx, gLW;       // accumulated light, silhouette, nearest outline, pixel size, line half-width
vec3 cL, cP, cQ;                       // line / pattern / accent colour of this being
float gG, gHueSeed;                    // 0 head / top .. 1 tail / bottom of this being; its own hue offset

// h11 / h21 and the 2D SDF helpers below (smin, sdSeg, sdCone, sdEll, sdBox, sdVes, sdPoly8, and sdTri further down)
// follow Inigo Quilez's articles and shaders: Copyright (c) 2013-2024 Inigo Quilez, MIT License (see the notice at
// the top of beings.py and THIRD_PARTY_NOTICES.md)
float h11(float n){ return fract(sin(n*127.1)*43758.5453); }
float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7)))*43758.5453); }
mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
vec3 palShow(float h){
  h = fract(h)*5.0; int i = int(floor(h)); float f = fract(h); f = f*f*(3.0-2.0*f);
  return mix(uPal[i%5], uPal[(i+1)%5], f);
}
vec3 hsv(float h, float sat){
  return mix(vec3(1.0), clamp(abs(fract(h+vec3(0.0, 2.0/3.0, 1.0/3.0))*6.0-3.0)-1.0, 0.0, 1.0), sat);
}
// every colour of a being comes through here: h = which of its colours (line, pattern, accent, per-cell variety)
vec3 pal(float h){
  if (uCMode < 0.5 || uCMode > 4.5) return palShow(h);                // show palette (the light chase keys it later)
  float v = 0.035*sin(TAU*h), st = 0.80+0.20*cos(TAU*2.0*h);          // a little variety keeps the ornament readable
  if (uCMode < 1.5) return hsv(uHue+v, st);
  if (uCMode < 2.5) return mix(hsv(uHue+v, st), hsv(uHue2+v, st), gG);
  if (uCMode < 3.5) return hsv(uHue+gG+v, 1.0);
  return hsv(uHue+gHueSeed+v, st);
}
float smin(float a, float b, float k){ float h = max(k-abs(a-b), 0.0)/k; return min(a, b)-h*h*k*0.25; }
float sdSeg(vec2 p, vec2 a, vec2 b){
  vec2 pa = p-a, ba = b-a; float h = clamp(dot(pa, ba)/max(dot(ba, ba), 1e-9), 0.0, 1.0); return length(pa-ba*h);
}
float sdCone(vec2 p, vec2 a, vec2 b, float ra, float rb){
  vec2 pa = p-a, ba = b-a; float h = clamp(dot(pa, ba)/max(dot(ba, ba), 1e-9), 0.0, 1.0);
  return length(pa-ba*h)-mix(ra, rb, h);
}
float sdEll(vec2 p, vec2 r){ float k0 = length(p/r), k1 = length(p/(r*r)); return k0*(k0-1.0)/max(k1, 1e-5); }
float sdBox(vec2 p, vec2 b){ vec2 d = abs(p)-b; return length(max(d, 0.0))+min(max(d.x, d.y), 0.0); }
float sdVes(vec2 p, float hl, float hw){            // pointed leaf along x: half-length hl, half-width hw
  hw = max(hw, 1e-3); float r = 0.5*(hw+hl*hl/hw), dd = r-hw;
  p = abs(p.yx);
  return ((p.y-hl)*dd > p.x*hl) ? length(p-vec2(0.0, hl)) : length(p-vec2(-dd, 0.0))-r;
}
float sdPoly8(vec2 p, vec2 v[8]){
  float d = dot(p-v[0], p-v[0]), s = 1.0;
  for (int i = 0, j = 7; i < 8; j = i, i++) {
    vec2 e = v[j]-v[i], w = p-v[i];
    vec2 b = w-e*clamp(dot(w, e)/max(dot(e, e), 1e-9), 0.0, 1.0);
    d = min(d, dot(b, b));
    bvec3 c = bvec3(p.y >= v[i].y, p.y < v[j].y, e.x*w.y > e.y*w.x);
    if (all(c) || all(not(c))) s = -s;
  }
  return s*sqrt(d);
}
// ribbon around a centre line y = yc (slope dy) of half-width w at the clamped station xc; v = across (-1..1)
float rib(vec2 p, float xc, float yc, float dy, float w, out float v){
  float off = (p.y-yc)*inversesqrt(1.0+dy*dy), e = abs(off)-w;
  v = off/max(w, 1e-4);
  return length(vec2(p.x-xc, max(e, 0.0)))+min(e, 0.0);
}

// ---- ornament (q in pattern units, ps = one pixel in pattern units)
float pline(float d, float w, float ps){ return 1.0-smoothstep(w, w+1.5*ps, abs(d)); }
float pfade(float ps){ return smoothstep(0.34, 0.10, ps); }
float kene(vec2 q, float ps){                       // right-angle labyrinth: truchet on the diagonal lattice
  vec2 r = vec2(q.x+q.y, q.x-q.y)*0.70711, g = floor(r), f = fract(r)-0.5;
  if (h21(g) > 0.5) f.x = -f.x;
  float d = abs(abs(f.x+f.y)-0.5)*0.70711;
  return (pline(d, 0.05, ps)+0.55*pline(d-0.17, 0.022, ps))*pfade(ps);
}
float diam(vec2 q, float ps){                       // nested diamonds with a bead in the middle
  vec2 f = abs(fract(q)-0.5); float s = f.x+f.y;
  return (pline(s-0.5, 0.05, ps)+0.7*pline(s-0.27, 0.04, ps)+(1.0-smoothstep(0.08, 0.08+1.5*ps, s)))*pfade(ps);
}
float beads(vec2 q, float ps, out float id){        // offset rows of beads
  q.x += 0.5*mod(floor(q.y), 2.0);
  vec2 g = floor(q), f = fract(q)-0.5; id = h21(g);
  return (1.0-smoothstep(0.30, 0.30+1.5*ps, length(f)))*pfade(ps);
}
float scales(vec2 q, float ps){                     // overlapping arcs
  q.x += 0.5*mod(floor(q.y), 2.0);
  vec2 f = fract(q)-vec2(0.5, 0.0);
  return pline(length(f)-0.52, 0.06, ps)*pfade(ps);
}
float spark(vec2 p, float dens){                    // twinkling points (highs)
  vec2 g = mod(floor(p*dens), 64.0), f = fract(p*dens)-0.5; float h = h21(g);   // lattice repeats every 64 cells
  vec2 o = vec2(h21(g+3.1), h21(g+7.7))-0.5;
  float tw = pow(max(sin(uT*floor(24.0+40.0*h)*0.125+h*50.0), 0.0), 6.0), r = length(f-o*0.6);
  return tw*exp(-r*r*70.0);
}

// ---- painting: parts are layered, a later part covers an earlier one
void part(float d, vec3 lc, vec3 fc){
  float f = 1.0-smoothstep(-gPx, gPx, d);
  float l = 1.0-smoothstep(gLW, gLW+1.5*gPx, abs(d));
  float inner = smoothstep(gLW*1.5, gLW*3.5, -d);
  gC = gC*(1.0-f)+fc*inner+lc*(l+0.20*f*exp(min(d, 0.0)*14.0));
  gA = max(gA, f); gD = min(gD, d);
}
void stroke(float d, float w, vec3 c){              // a bare line (d = unsigned distance to its centre)
  float l = 1.0-smoothstep(w, w+1.5*gPx, d);
  gC = gC*(1.0-0.6*l)+c*l; gA = max(gA, 0.7*l); gD = min(gD, d-w);
}
void bead(vec2 q, float r, vec3 c){                 // a filled dot
  float l = 1.0-smoothstep(r-gPx, r+gPx, length(q));
  gC = mix(gC, c, l); gA = max(gA, l); gD = min(gD, length(q)-r);
}
// feather fan around +x from its root: half-angle ha, length R, n feathers; pat = quills, gaps and bars
float fan(vec2 q, float ha, float R, float n, out float pat){
  float r = length(q), a = atan(q.y, q.x), t = clamp(a/ha, -1.0, 1.0);
  float ff = fract((t*0.5+0.5)*n)-0.5;
  float Rr = R*(0.70+0.30*cos(t*1.35))*(1.0-0.55*ff*ff);
  float d = max(r-Rr, (abs(a)-ha)*r);
  float wd = 2.0*ha*r/n, lw = gLW*0.55;
  pat = (1.0-smoothstep(lw, lw+1.5*gPx, abs(ff)*wd))*smoothstep(0.05*R, 0.25*R, r)
      + (1.0-smoothstep(lw, lw+1.5*gPx, (0.5-abs(ff))*wd))
      + 0.8*(1.0-smoothstep(lw, lw+1.5*gPx, abs(r-0.58*Rr)))+0.6*(1.0-smoothstep(lw, lw+1.5*gPx, abs(r-0.80*Rr)));
  return d;
}

// ================================================================ forms (local space, about +-1.4 units)

float snake(vec2 p, float ph, float sd, float x0, float x1, float w0, out float u, out float v){
  float xc = clamp(p.x, x0, x1); u = (xc-x0)/(x1-x0);
  float env = (0.03+0.15*uWig)*(1.0-0.7*u*u), k = 5.5, a = k*xc+ph*3.0+sd*9.0;
  float w = w0*pow(smoothstep(0.0, 0.5, u), 0.7)*(1.0-0.25*smoothstep(0.80, 1.0, u))+0.003;
  return rib(p, xc, env*sin(a), env*k*cos(a), w, v);
}
vec2 snakeHead(vec2 p, float ph, float sd, float x1){      // p in the head's frame (origin at the neck, nose +x)
  float env = (0.03+0.15*uWig)*0.3, a = 5.5*x1+ph*3.0+sd*9.0;
  return rot(-atan(env*5.5*cos(a)))*(p-vec2(x1, env*sin(a)));
}

void fSerpent(vec2 p, float ph, float sd){
  float u, v, x0 = -1.30, x1 = 0.88;
  float d = snake(p, ph, sd, x0, x1, 0.090, u, v);
  float segs = 12.0+16.0*uDetail, su = u*segs, fu = abs(fract(su)-0.5)*2.0, ps = gPx*segs/(x1-x0);
  float dm = fu+abs(v)*1.1;
  float pat = pline(dm-0.80, 0.10, ps*2.0)+(1.0-smoothstep(0.20, 0.20+3.0*ps, dm));
  float tick = pline(fu-1.0, 0.10, ps*2.0)*smoothstep(0.45, 0.6, abs(v));
  part(d, cL, (pal(sd+0.21+floor(su)*0.17)*pat+cQ*tick*0.7)*pfade(ps*0.5));
  vec2 q = snakeHead(p, ph, sd, x1);
  float flick = smoothstep(0.2, 0.9, sin(ph*1.5+sd*5.0));
  if (flick > 0.0) {
    vec2 t0 = vec2(0.24, 0.0), t1 = vec2(0.24+0.15*flick, 0.0);
    float dt_ = min(sdSeg(q, t0, t1), sdSeg(vec2(q.x, abs(q.y)), t1, t1+vec2(0.07, 0.045)*flick));
    stroke(dt_, gLW*0.5, cQ);
  }
  float dh = smin(sdEll(q-vec2(0.05, 0.0), vec2(0.16, 0.105)), sdEll(q-vec2(0.17, 0.0), vec2(0.09, 0.065)), 0.06);
  part(dh, cL, cP*0.8*diam(q*(9.0+6.0*uDetail), gPx*12.0)*smoothstep(0.03, 0.06, abs(q.y)));
  vec2 e = vec2(q.x-0.09, abs(q.y)-0.062);
  bead(e, 0.030, W); bead(e*vec2(0.45, 1.0), 0.009, vec3(0.0));
}

void fDragon(vec2 p, float ph, float sd){
  float u, v, pat, x0 = -1.25, x1 = 0.70;
  // tail plume
  float env0 = 0.03+0.15*uWig, a0 = 5.5*x0+ph*3.0+sd*9.0;
  vec2 tq = rot(PI-atan(env0*5.5*cos(a0))*0.6+0.25*sin(ph*2.0))*(p-vec2(x0+0.08, env0*sin(a0)));
  float dt_ = fan(tq, 0.55, 0.36, 5.0, pat);
  part(dt_, cQ, cQ*pat*0.8);
  // wings (mirrored across the spine), slow beat
  float xw = 0.22, envw = (0.03+0.15*uWig)*(1.0-0.7*0.56), aw = 5.5*xw+ph*3.0+sd*9.0;
  vec2 wq = vec2(p.x-xw, abs(p.y-envw*sin(aw))-0.03);
  float beat = sin(ph*2.0);
  wq = rot(-(1.25+0.38*beat))*wq;
  float dw = fan(wq, 0.72, 0.92*(0.9+0.1*beat), 8.0, pat);
  part(dw, cP, pal(sd+0.3+length(wq)*0.5)*pat*0.85);
  // body with a feather frill along both flanks
  float d = snake(p, ph, sd, x0, x1, 0.075, u, v);
  float fr = abs(sin(u*46.0))*0.035*smoothstep(0.05, 0.3, u)*smoothstep(1.0, 0.8, u);
  d -= fr*smoothstep(0.02, 0.0, abs(d));
  float segs = 14.0+14.0*uDetail, su = u*segs, ps = gPx*segs/(x1-x0);
  float sc = scales(vec2(su, v*1.5+0.5), ps);
  part(d, cL, pal(sd+0.2+u*0.6)*sc*0.9+cQ*pline(v, 0.10, ps*3.0)*0.6);
  // head: horns, whiskers, mane
  vec2 q = snakeHead(p, ph, sd, x1);
  vec2 qa = vec2(q.x, abs(q.y));
  stroke(sdSeg(qa, vec2(0.02, 0.07), vec2(-0.22, 0.21)), gLW*0.9, cQ);
  stroke(sdSeg(qa, vec2(-0.04, 0.06), vec2(-0.20, 0.10)), gLW*0.7, cQ);
  float wx = clamp(q.x, 0.26, 0.62), wy = 0.05+0.5*(wx-0.26)+0.05*sin(wx*14.0-ph*3.0);
  stroke(length(vec2(q.x-wx, abs(q.y)-wy)), gLW*0.5, cP);
  float dh = smin(sdEll(q-vec2(0.06, 0.0), vec2(0.17, 0.115)), sdEll(q-vec2(0.22, 0.0), vec2(0.12, 0.070)), 0.06);
  part(dh, cL, cP*0.8*diam(q*(10.0+6.0*uDetail), gPx*13.0)*smoothstep(0.02, 0.05, abs(q.y)));
  vec2 e = vec2(q.x-0.10, abs(q.y)-0.068);
  bead(e, 0.032, W); bead(e, 0.012, vec3(0.0));
  bead(vec2(q.x-0.31, abs(q.y)-0.028), 0.010, cQ);
}

float jagLeg(vec2 p, vec2 hip, float g, float fwd){       // one walking leg: hip -> knee -> paw
  float s = sin(g), lift = max(cos(g), 0.0);
  vec2 foot = hip+vec2(0.22*s, -0.56+0.10*lift);
  vec2 knee = mix(hip, foot, 0.5)+vec2(fwd*(0.05+0.05*lift), 0.02);
  float d = min(sdCone(p, hip, knee, 0.095, 0.060), sdCone(p, knee, foot, 0.060, 0.045));
  return smin(d, sdEll(p-foot-vec2(0.035, -0.01), vec2(0.075, 0.042)), 0.03);
}
void fJaguar(vec2 p, float ph, float sd, float g){
  p.y -= 0.10+0.015*sin(g*2.0);
  // far-side legs, dimmer
  float dl = min(jagLeg(p, vec2(-0.42, -0.06), g+PI, -1.0), jagLeg(p, vec2(0.30, -0.08), g+PI*0.5, 1.0));
  part(dl, cL*0.45, vec3(0.0));
  // tail
  vec2 tq = rot(-2.55)*(p-vec2(-0.60, 0.08));
  float tx = clamp(tq.x, 0.0, 0.72), v;
  float ty = (0.03+0.16*tx)*sin(5.0*tx-ph*1.5)-0.35*tx*tx, tdy = (0.03+0.16*tx)*5.0*cos(5.0*tx-ph*1.5)-0.7*tx;
  float dtl = rib(tq, tx, ty, tdy, 0.042-0.012*tx, v);
  part(dtl, cL, cQ*pline(fract(tx*9.0)-0.5, 0.16, gPx*9.0));
  // torso + neck + head + near legs: one silhouette
  float d = sdCone(p, vec2(-0.46, 0.02), vec2(0.28, 0.04), 0.205, 0.225);
  d = smin(d, sdCone(p, vec2(0.32, 0.10), vec2(0.60, 0.22), 0.17, 0.15), 0.10);
  d = smin(d, jagLeg(p, vec2(-0.46, -0.06), g, -1.0), 0.06);
  d = smin(d, jagLeg(p, vec2(0.34, -0.08), g+PI*1.5, 1.0), 0.06);
  vec2 hc = vec2(0.66, 0.24+0.02*sin(g));
  d = smin(d, length(p-hc)-0.165, 0.05);
  d = smin(d, sdEll(p-hc-vec2(0.14, -0.06), vec2(0.105, 0.080)), 0.04);
  d = min(d, length(p-hc-vec2(-0.10, 0.155))-0.058);
  d = min(d, length(p-hc-vec2(0.04, 0.175))-0.058);
  // rosettes: broken rings with a spot inside, on a jittered lattice
  float fq = 5.5+4.0*uDetail, ps = gPx*fq;
  vec2 rq = p*fq+vec2(0.0, 0.3); rq.x += 0.5*mod(floor(rq.y), 2.0);
  vec2 gi = floor(rq), f = fract(rq)-0.5; float h = h21(gi+sd*13.0);
  vec2 o = (vec2(h, h21(gi+5.3))-0.5)*0.30; float r = length(f-o), an = atan(f.y-o.y, f.x-o.x);
  float ring = pline(r-0.28, 0.075, ps)*smoothstep(-0.75, -0.45, sin(an*2.0+h*20.0)+0.55*sin(an*5.0+h*31.0));
  float spot = 1.0-smoothstep(0.07, 0.07+1.5*ps, r);
  part(d, cL, (pal(sd+0.2+h*0.25)*ring+cQ*spot)*pfade(ps));
  // face
  bead(p-hc-vec2(0.055, 0.035), 0.030, W); bead((p-hc-vec2(0.062, 0.035))*vec2(1.0, 0.45), 0.008, vec3(0.0));
  bead(p-hc-vec2(0.235, -0.035), 0.020, cQ);
  stroke(sdSeg(p, hc+vec2(0.23, -0.06), hc+vec2(0.12, -0.105)), gLW*0.5, cL);
  for (int i = 0; i < 3; i++) {
    float fi = float(i)-1.0;
    stroke(sdSeg(p, hc+vec2(0.20, -0.07), hc+vec2(0.36, -0.07+0.055*fi+0.01*sin(ph*2.0+fi))), gLW*0.3, W*0.7);
  }
}

void fHummingbird(vec2 p, float ph, float sd){
  float pat;
  p.y -= 0.03*sin(ph*1.25);
  vec2 sh = vec2(0.06, 0.13);
  float fl = sin(ph*8.0);
  // far wing + motion ghosts
  for (int i = 0; i < 3; i++) {
    float fi = float(i), a = 1.75+1.05*sin(ph*8.0-1.0-fi*0.9);
    float dg = fan(rot(-a)*(p-sh-vec2(0.03, 0.01)), 0.26, 0.86, 6.0, pat);
    float l = (1.0-smoothstep(gLW*0.5, gLW*0.5+1.5*gPx, abs(dg)))+0.4*pat*(1.0-smoothstep(-gPx, gPx, dg));
    gC += cP*l*(0.30-0.07*fi); gD = min(gD, dg+0.05);
  }
  // tail fan
  float dt_ = fan(rot(-(3.95+0.12*fl))*(p-vec2(-0.20, -0.20)), 0.36, 0.46, 5.0, pat);
  part(dt_, cQ, cQ*pat*0.8);
  // beak
  float db = sdCone(p, vec2(0.40, 0.285), vec2(0.90, 0.33), 0.024, 0.004);
  part(db, cL, vec3(0.0));
  // body + head
  float d = sdCone(p, vec2(-0.20, -0.20), vec2(0.10, 0.06), 0.075, 0.155);
  vec2 hc = vec2(0.30, 0.26);
  d = smin(d, length(p-hc)-0.118, 0.10);
  float id, fq = 13.0+10.0*uDetail, ps = gPx*fq;
  float bd = beads(rot(0.6)*p*fq, ps, id);
  float throat = smoothstep(-0.02, 0.10, p.x-0.10+0.3*(p.y-0.05));
  part(d, cL, mix(cP*0.55, pal(sd+0.45+id*0.3), throat)*bd);
  bead(p-hc-vec2(0.035, 0.025), 0.028, W); bead(p-hc-vec2(0.040, 0.025), 0.013, vec3(0.0));
  // near wing
  float a = 1.75+1.05*fl;
  float dw = fan(rot(-a)*(p-sh), 0.27, 0.90, 6.0, pat);
  part(dw, cL, pal(sd+0.25+length(p-sh)*0.4)*pat*0.85);
}

void butterfly(vec2 p, float ph, float sd, float moth){
  float flap = 0.22+0.78*abs(cos(ph*3.0+sd*20.0));
  vec2 q = vec2(p.x, abs(p.y)/flap);
  float ps = gPx*10.0;
  // hind then fore wing
  vec2 c2 = mix(vec2(-0.20, 0.27), vec2(-0.16, 0.24), moth), r2 = mix(vec2(0.26, 0.30), vec2(0.20, 0.24), moth);
  vec2 q2 = rot(0.35)*(q-c2);
  float d2 = sdEll(q2, r2);
  float e2 = length(q2-vec2(-0.03, 0.04));
  part(d2, cL, cQ*(pline(e2-0.10, 0.02, gPx)+pline(e2-0.16, 0.012, gPx))+cP*0.7*pline(fract(atan(q.y, q.x+0.05)*5.0)-0.5, 0.10, ps));
  vec2 c1 = mix(vec2(0.14, 0.40), vec2(0.02, 0.42), moth), r1 = mix(vec2(0.25, 0.40), vec2(0.24, 0.46), moth);
  vec2 q1 = rot(mix(-0.50, -1.05, moth))*(q-c1);
  float d1 = mix(sdEll(q1, r1), max(sdEll(q1, r1), -q1.x*0.6-q1.y*0.8-0.30), moth);
  float e1 = length(q1-vec2(0.02, 0.10));
  float vein = pline(fract(atan(q.y, q.x-0.02)*(6.0+4.0*uDetail)/PI)-0.5, 0.09, ps);
  float rim = pline(d1+0.055, 0.012, gPx)*step(0.5, fract(atan(q1.y, q1.x)*7.0));
  vec3 f1 = cP*0.75*vein*smoothstep(0.02, 0.10, -d1)+cQ*(1.0-smoothstep(0.045, 0.06, e1))
          + pal(sd+0.5)*pline(e1-0.10, 0.016, gPx)+W*0.8*rim;
  part(d1, cL, f1);
  // body, antennae
  vec2 qa = vec2(p.x, abs(p.y));
  float ax = clamp(qa.x, 0.20, 0.46), ay = 0.02+0.55*(ax-0.20)+0.6*(ax-0.20)*(ax-0.20);
  stroke(length(qa-vec2(ax, ay)), gLW*0.4, cL);
  bead(qa-vec2(0.46, 0.20), mix(0.016, 0.03, moth), cQ);
  part(sdCone(p, vec2(-0.24, 0.0), vec2(0.20, 0.0), 0.022, mix(0.038, 0.06, moth)), cL,
       cQ*pline(fract(p.x*14.0)-0.5, 0.2, gPx*14.0));
}
void fButterflies(vec2 p, float ph, float sd){
  for (int j = 0; j < 4; j++) {
    float fj = float(j), h = h11(sd*17.0+fj*3.7);
    float an = fj*TAU/4.0+sd*6.0+0.45*sin(ph*0.25+fj*2.0);
    vec2 o = (j == 0) ? vec2(0.15, 0.0) : (0.95+0.12*sin(ph*0.375+fj))*vec2(cos(an), sin(an));
    float s = (j == 0) ? 0.68 : 0.40+0.14*h;
    vec2 q = p-o;
    if (dot(q, q) > s*s*1.1) continue;
    float px0 = gPx, lw0 = gLW; gPx /= s; gLW = max(gLW/s*0.8, 0.9*gPx);
    vec3 l0 = cL, p0 = cP, q0 = cQ;
    cL = mix(pal(sd+fj*0.23), W, 0.25); cP = pal(sd+fj*0.23+0.22); cQ = pal(sd+fj*0.23+0.5);
    butterfly(rot(-(0.5*sin(ph*0.5+fj*1.7)+(j == 0 ? 0.0 : an+1.3)))*q/s, ph, sd+fj*0.31, float(j%2));
    gPx = px0; gLW = lw0; cL = l0; cP = p0; cQ = q0;
  }
}

void fJellyfish(vec2 p, float ph, float sd){
  p.y -= 0.22;
  float c = sin(ph*2.0), Wd = 0.52*(1.0+0.13*c), H = 0.44*(1.0-0.12*c), v;
  float yb = 0.26+0.035*cos(p.x/Wd*PI*5.0)+0.05*c;
  // trailing tentacles (thin, beaded) and frilled oral arms
  for (int i = 0; i < 7; i++) {
    float fi = float(i)-3.0, x0 = Wd*0.86*fi/3.0, L = 1.15-0.06*fi*fi+0.06*sin(sd*9.0+fi);
    float s = clamp(0.28-p.y, 0.0, L);
    float xc = x0*(1.0-0.25*s)+(0.02+0.10*uWig*s)*sin(7.0*s-ph*3.0+fi*1.7+sd*5.0)-0.05*c*s;
    float dd = length(vec2(p.x-xc, (0.28-p.y)-s));
    stroke(dd, gLW*0.45, cP*(1.0-0.5*s/L));
    float bx = fract(s*9.0-ph*TURN*1.5+fi*0.3)-0.5;
    gC += pal(sd+0.5+fi*0.1)*exp(-(dd*dd+bx*bx*0.012)*2600.0)*1.2*step(0.05, s);
  }
  for (int i = 0; i < 3; i++) {
    float fi = float(i)-1.0, L = 0.80;
    vec2 q = vec2(0.30-p.y, p.x-0.13*fi);
    float s = clamp(q.x, 0.0, L), ag = 6.0*s-ph*2.5+fi*2.0;
    float yc = (0.02+0.09*uWig*s)*sin(ag), w = 0.050*(1.0-s/L)*(1.0+0.45*sin(34.0*s-ph*4.0))+0.004;
    float dd = rib(q, s, yc, 0.09*uWig*6.0*s*cos(ag), w, v);
    part(dd, cQ, cQ*0.6*pline(v, 0.25, gPx*30.0));
  }
  // bell
  vec2 bc = vec2(0.0, 0.26);
  float d = max(sdEll(p-bc, vec2(Wd, H)), yb-p.y);
  float a = atan(p.x, p.y-0.12), ps = gPx*12.0;
  float ribs = pline(fract(a*(5.0+4.0*uDetail)/PI)-0.5, 0.08, ps*0.6);
  float din = sdEll(p-bc-vec2(0.0, -0.02), vec2(Wd*0.62, H*0.60));
  float rimb = (1.0-smoothstep(0.022, 0.03, length(vec2(fract(p.x/Wd*7.5)-0.5, (p.y-yb-0.055)*7.5/Wd))*Wd/7.5));
  vec3 fc = cP*0.8*ribs*smoothstep(0.0, 0.05, din)+cQ*pline(din, 0.012, gPx)+pal(sd+0.55)*rimb
          + pal(sd+0.35)*0.55*kene((p-bc)*(9.0+6.0*uDetail), gPx*12.0)*smoothstep(0.02, 0.06, -din);
  part(d, cL, fc);
}

void fManta(vec2 p, float ph, float sd){
  float v, beat = ph*2.0;
  // tail
  float tx = clamp(p.x, -1.35, -0.38), tu = (-0.38-tx);
  float dtl = rib(p, tx, 0.07*tu*sin(5.0*tx+beat), 0.0, 0.018*(1.0-tu/0.97)+0.003, v);
  part(dtl, cL, vec3(0.0));
  // wings: beat = foreshortening + a wave running out to the tips
  vec2 q = vec2(p.x, abs(p.y));
  q.y /= 0.86+0.14*cos(beat);
  q.x -= (0.05+0.10*uWig)*q.y*q.y*sin(beat-q.y*2.2);
  vec2 pts[8] = vec2[8](vec2(0.40, -0.3), vec2(0.40, 0.13), vec2(0.14, 0.52), vec2(-0.16, 1.02), vec2(-0.24, 0.50),
                        vec2(-0.40, 0.13), vec2(-0.40, -0.3), vec2(-0.40, -0.3));
  float d = sdPoly8(q, pts)-0.045;
  // cephalic fins
  float dc = sdCone(q, vec2(0.40, 0.125), vec2(0.60, 0.085+0.02*sin(beat)), 0.050, 0.020);
  d = smin(d, dc, 0.04);
  float fq = 9.0+8.0*uDetail, ps = gPx*fq, id;
  float cont = pline(fract(d*(9.0+6.0*uDetail))-0.5, 0.07, gPx*12.0)*smoothstep(0.30, 0.04, -d);
  float bd = beads(q*fq, ps, id)*smoothstep(0.10, 0.20, -d)*step(0.45, id);
  float spine = pline(q.y, 0.012, gPx)*smoothstep(0.42, 0.2, abs(q.x));
  float gill = pline(fract(q.x*11.0)-0.5, 0.10, gPx*11.0)*step(abs(q.y-0.12), 0.05)*step(abs(q.x-0.12), 0.14);
  part(d, cL, cP*0.85*cont+pal(sd+0.45+id*0.2)*bd+cQ*(spine+gill));
  bead(q-vec2(0.36, 0.20), 0.026, W); bead(q-vec2(0.365, 0.20), 0.011, vec3(0.0));
}

void fKoi(vec2 p, float ph, float sd){
  float v, pat, x0 = -0.55, x1 = 0.62, k = 5.0, A = 0.03+0.15*uWig;
  float xc = clamp(p.x, x0, x1), u = (xc-x0)/(x1-x0);
  float am = A*pow(max(1.0-u, 0.0), 1.5), ag = k*xc+ph*3.0+sd*7.0, yc = am*sin(ag), dy = am*k*cos(ag);
  // tail fin
  float am0 = A, ag0 = k*x0+ph*3.0+sd*7.0;
  vec2 tq = rot(PI-(atan(am0*k*cos(ag0))+0.45*sin(ag0-0.9)))*(p-vec2(x0+0.05, am0*sin(ag0)));
  float ta = atan(tq.y, tq.x)/0.62;
  float dt_ = fan(tq, 0.62, 0.62, 7.0, pat);
  dt_ = max(dt_, length(tq)-0.62*(0.55+0.45*min(abs(ta)*1.4, 1.0)));
  part(dt_, cQ, pal(sd+0.5+length(tq)*0.5)*pat*0.8);
  // pectoral + pelvic fins
  for (int i = 0; i < 2; i++) {
    float fx = (i == 0) ? 0.28 : -0.10, R = (i == 0) ? 0.34 : 0.22;
    float uu = (fx-x0)/(x1-x0), yy = A*pow(max(1.0-uu, 0.0), 1.5)*sin(k*fx+ph*3.0+sd*7.0);
    vec2 fq = vec2(p.x-fx, abs(p.y-yy)-0.10);
    float df = fan(rot(-(2.15+0.30*sin(ph*3.0+float(i))))*fq, 0.42, R, 5.0, pat);
    part(df, cQ, cQ*pat*0.7);
  }
  // body
  float w = mix(0.045, 0.200, smoothstep(0.0, 0.66, u))*sqrt(clamp(1.0-pow(max(u-0.66, 0.0)/0.34, 2.2), 0.0, 1.0))+0.004;
  float d = rib(p, xc, yc, dy, w, v);
  float fq = 16.0+12.0*uDetail, ps = gPx*fq/(x1-x0);
  float sc = scales(vec2(-u*fq, v*3.0), ps)*smoothstep(0.80, 0.70, u);
  float blot = smoothstep(0.05, 0.25, sin(u*7.0+sd*20.0)*sin(v*2.2+u*4.0+sd*9.0));
  float dorsal = pline(v, 0.07, gPx*8.0)*smoothstep(0.25, 0.35, u)*smoothstep(0.72, 0.62, u);
  part(d, cL, mix(cP*0.7, cQ, blot)*sc+W*0.7*dorsal+cQ*0.35*blot*smoothstep(0.7, 0.8, u));
  // eyes + barbels
  vec2 hq = p-vec2(x1, 0.0);
  bead(vec2(hq.x+0.13, abs(hq.y)-0.105), 0.028, W); bead(vec2(hq.x+0.125, abs(hq.y)-0.108), 0.012, vec3(0.0));
  float bx = clamp(hq.x, 0.0, 0.14);
  stroke(length(vec2(hq.x-bx, abs(hq.y)-(0.03+0.5*bx+0.02*sin(ph*3.0+bx*20.0)))), gLW*0.4, cL);
}

void fPhoenix(vec2 p, float ph, float sd){
  float pat, v;
  p.y -= 0.42;
  // tail plumes with eye-spots
  for (int i = 0; i < 5; i++) {
    float fi = float(i)-2.0, L = 1.18-0.13*abs(fi);
    vec2 q = rot(PI*0.5-fi*0.24-0.10*sin(ph*1.5+fi))*(p-vec2(0.0, -0.12));
    float s = clamp(q.x, 0.0, L), ag = 5.0*s-ph*2.5+fi*1.3;
    float yc = (0.015+0.10*uWig)*s*sin(ag), dy = (0.015+0.10*uWig)*(sin(ag)+5.0*s*cos(ag));
    float w = 0.034*pow(sin(PI*clamp(s/L, 0.0, 1.0)*0.92+0.08), 0.6)+0.004;
    float d = rib(q, s, yc, dy, w, v);
    vec3 fc = pal(sd+0.15+s*0.35)*(pline(v, 0.16, gPx*25.0)+0.8*pline(fract(s*16.0-abs(v)*0.5)-0.5, 0.10, gPx*16.0));
    part(d, mix(cL, cQ, s/L), fc*0.8);
    vec2 eq = q-vec2(L-0.02, (0.015+0.10*uWig)*L*sin(5.0*L-ph*2.5+fi*1.3));
    float de = sdEll(eq, vec2(0.085, 0.060));
    part(de, cQ, cP*pline(length(eq)-0.035, 0.008, gPx));
    bead(eq, 0.018, W);
  }
  // wings: big primaries + a shorter covert layer
  float beat = sin(ph*2.0);
  vec2 wq = vec2(abs(p.x)-0.07, p.y-0.14);
  float th = 0.30+0.50*beat;
  float dw = fan(rot(-th)*wq, 0.66, 1.0*(0.92+0.08*beat), 9.0, pat);
  part(dw, cL, pal(sd+0.10+length(wq)*0.45)*pat*0.85);
  float dv = fan(rot(-th-0.08)*wq, 0.55, 0.52, 7.0, pat);
  part(dv, cQ, cQ*pat*0.75);
  // body, neck, head, crest, beak
  float d = sdCone(p, vec2(0.0, -0.14), vec2(0.0, 0.16), 0.085, 0.115);
  d = smin(d, sdCone(p, vec2(0.0, 0.22), vec2(0.03, 0.40), 0.065, 0.050), 0.07);
  vec2 hc = vec2(0.04, 0.44);
  for (int i = 0; i < 3; i++) {
    float fi = float(i)-1.0;
    vec2 tip = hc+0.24*vec2(sin(-0.55+fi*0.42+0.08*sin(ph*2.0+fi)), cos(-0.55+fi*0.42));
    stroke(sdSeg(p, hc, tip), gLW*0.5, cQ); bead(p-tip, 0.020, cQ);
  }
  part(sdCone(p, hc+vec2(0.05, 0.0), hc+vec2(0.19, -0.035), 0.030, 0.003), W, vec3(0.0));
  d = smin(d, length(p-hc)-0.075, 0.04);
  float id; float bd = beads(p*(18.0+10.0*uDetail), gPx*22.0, id);
  part(d, cL, pal(sd+0.4+id*0.3)*bd);
  bead(p-hc-vec2(0.025, 0.012), 0.018, W); bead(p-hc-vec2(0.03, 0.012), 0.008, vec3(0.0));
  // embers
  gC += cQ*spark(p+vec2(0.0, -uScroll.x), 5.0)*(0.5+uHigh)*smoothstep(1.5, 0.3, length(p));
}

void fEye(vec2 p, float ph, float sd){
  float cyc = fract(ph*TURN*0.21875+sd*7.3);                         // 14 blink cycles per phase wrap
  float b1 = (cyc-0.50)/0.022, b2 = (cyc-0.57)/0.018, b3 = (cyc-0.12)/0.03;
  float bl = exp(-b1*b1)+exp(-b2*b2)+exp(-b3*b3);
  float o = clamp(1.0-bl, 0.05, 1.0)*(0.86+0.14*sin(ph*0.375+sd*3.0));
  float hl = 0.86, hw = 0.44*o;
  float d = sdVes(p, hl, hw);
  // rays: a crown of long and short points around the lids
  float a = atan(p.y, p.x), n = 14.0;
  float tri = abs(fract(a*n/TAU+0.5*0.0)-0.5)*2.0, odd = mod(floor(a*n/TAU), 2.0);
  float dopen = sdVes(p, hl, 0.44);
  float rl = (0.10+0.16*odd+0.05*uBass)*(1.0-tri*1.15);
  float dr = dopen-0.07-rl;
  float ray = (1.0-smoothstep(-gPx, gPx, dr))*smoothstep(0.045, 0.07, dopen);
  gC += pal(sd+0.5+odd*0.2)*((1.0-smoothstep(gLW*0.6, gLW*0.6+1.5*gPx, abs(dr)))*smoothstep(0.05, 0.07, dopen)+0.10*ray);
  gA = max(gA, ray*0.7); gD = min(gD, max(dr, 0.0));
  stroke(abs(dopen-0.045), gLW*0.45, cQ);
  // the eye itself
  vec2 gz = vec2(0.13*sin(ph*0.375+sd*9.0)+0.05*sin(ph*0.875), 0.05*sin(ph*0.5+sd*4.0))*o;
  float r = length(p-gz), ia = atan(p.y-gz.y, p.x-gz.x), R = 0.33;
  float pup = 0.125*(1.0-0.30*uBass);
  float fib = pline(fract(ia*(11.0+8.0*uDetail)/PI)-0.5, 0.16, gPx*10.0)*smoothstep(pup, pup+0.05, r);
  vec3 iris = (pal(sd+0.2+r*1.2)*(0.35+0.65*fib)+cQ*pline(r-R*0.62, 0.010, gPx))*(1.0-smoothstep(R-gPx, R+gPx, r));
  iris += W*pline(r-R, gLW*0.7, gPx);
  iris *= smoothstep(pup-gPx, pup+gPx, r);
  iris += cL*pline(r-pup, gLW*0.5, gPx);
  float white = kene(p*(11.0+6.0*uDetail), gPx*14.0)*smoothstep(R+0.03, R+0.08, r)*0.5;
  part(d, cL, iris+cP*white);
  bead(p-gz-vec2(-0.05, 0.055)*o, 0.028*o, W);
  // lid crease above, lash line below
  stroke(abs(sdVes(p-vec2(0.0, 0.06), hl*0.97, hw+0.08))+step(p.y, 0.05)*9.0, gLW*0.4, cL*0.8);
}

void fAngel(vec2 p, float ph, float sd){
  float pat, v;
  p.y -= 0.22;
  vec2 hc = vec2(0.0, 0.40);
  // three pairs of wings
  vec2 wq = vec2(abs(p.x)-0.06, p.y-0.20);
  for (int i = 0; i < 3; i++) {
    float fi = float(i);
    float th = (-0.85+0.92*fi)+(0.22+0.06*fi)*sin(ph*1.5-fi*0.7);
    float R = 0.78+0.10*fi, hang = 0.36+0.03*fi;
    float d = fan(rot(-th)*wq, hang, R, 6.0+fi, pat);
    part(d, mix(cL, W, 0.3), pal(sd+0.15*fi+length(wq)*0.4)*pat*0.85);
  }
  // halo: ring + rays
  float hr = length(p-hc), ha_ = atan(p.y-hc.y, p.x-hc.x);
  stroke(abs(hr-0.175), gLW*0.6, cQ);
  float rays = pline(fract(ha_*12.0/TAU)-0.5, 0.06, gPx*4.0/max(hr, 0.05))*smoothstep(0.20, 0.23, hr)
             * smoothstep(0.40+0.10*uBass, 0.26, hr);
  gC += cQ*rays*0.9; gD = min(gD, hr-0.30);
  // robe (flowing hem), head, heart light
  float s = clamp(0.20-p.y, 0.0, 1.10);
  float sway = (0.02+0.07*uWig)*s*sin(5.0*s-ph*2.0);
  float w = mix(0.115, 0.012, pow(s/1.10, 1.6))+0.06*smoothstep(0.0, 0.35, s)*smoothstep(0.95, 0.40, s);
  float d = rib(vec2(0.20-p.y, p.x), s, sway, 0.0, w, v);
  d = smin(d, length(p-vec2(0.0, 0.14))-0.12, 0.05);
  float kq = kene(vec2((p.x-sway)*(16.0+8.0*uDetail), s*(16.0+8.0*uDetail)), gPx*20.0);
  part(d, cL, cP*0.8*kq*smoothstep(0.08, 0.2, s)+cQ*pline(v, 0.10, gPx*8.0)*smoothstep(0.1, 0.3, s));
  part(length(p-hc)-0.090, mix(cL, W, 0.5), W*0.12);
  vec2 hq = p-vec2(0.0, 0.10); float hrr = length(hq), haa = atan(hq.y, hq.x);
  float star = (0.035+0.02*uBass)/(hrr+0.03)*(0.55+0.45*cos(haa*4.0))*smoothstep(0.22, 0.0, hrr);
  gC += W*min(star, 1.5)*0.8;
}

void fJester(vec2 p, float ph, float sd){
  // juggled gems on a ring
  for (int i = 0; i < 5; i++) {
    float fi = float(i), a = ph*1.25+fi*TAU/5.0;
    vec2 q = rot(ph*2.0+fi)*(p-vec2(cos(a), sin(a)*0.95)*(1.05+0.08*sin(ph*2.0+fi*2.0)));
    float d = (i%3 == 0) ? sdBox(q, vec2(0.065)) : (i%3 == 1) ? (abs(q.x)+abs(q.y)-0.10)*0.707 : length(q)-0.075;
    part(d, pal(sd+fi*0.2), pal(sd+fi*0.2+0.3)*pline(length(q)-0.03, 0.008, gPx));
  }
  // limbs: two-segment chains with diamond hands
  vec2 bc = vec2(0.0, -0.36);
  for (int i = 0; i < 4; i++) {
    float fi = float(i);
    float a1 = ((i < 2) ? 0.55 : -1.05)*((i%2 == 0) ? 1.0 : -1.0)+PI*0.5*((i%2 == 0) ? 0.0 : 2.0)+0.75*sin(ph*2.0+fi*1.3);
    vec2 hip = bc+vec2((i%2 == 0) ? 0.07 : -0.07, (i < 2) ? 0.10 : -0.12);
    vec2 el = hip+0.30*vec2(cos(a1), sin(a1));
    float a2 = a1+1.1*sin(ph*3.0+fi*2.1);
    vec2 hd = el+0.28*vec2(cos(a2), sin(a2));
    part(min(sdCone(p, hip, el, 0.045, 0.030), sdCone(p, el, hd, 0.030, 0.020)), cL,
         cP*pline(fract(length(p-hip)*9.0)-0.5, 0.2, gPx*9.0));
    vec2 hq = rot(a2)*(p-hd);
    part((abs(hq.x)+abs(hq.y)-0.070)*0.707, cQ, vec3(0.0));
  }
  // torso
  vec2 tq = p-bc;
  float dt_ = (abs(tq.x)*1.15+abs(tq.y)-0.26)*0.66;
  part(dt_, cL, pal(sd+0.3)*diam(tq*(6.0+4.0*uDetail), gPx*9.0));
  // hat: three curling prongs with bells
  vec2 hc = vec2(0.0, 0.12);
  for (int i = 0; i < 3; i++) {
    float fi = float(i)-1.0, a = PI*0.5-fi*0.95+0.22*sin(ph*3.0+fi*2.0);
    vec2 mid = hc+0.42*vec2(cos(a), sin(a)), tip = mid+0.26*vec2(cos(a-fi*0.9-0.5*sin(ph*3.0+fi)), sin(a-fi*0.9-0.5*sin(ph*3.0+fi)));
    float d = min(sdCone(p, hc+0.15*vec2(cos(a), sin(a)), mid, 0.135, 0.070), sdCone(p, mid, tip, 0.070, 0.022));
    part(d, cL, pal(sd+0.2+fi*0.2)*pline(fract(length(p-hc)*8.0)-0.5, 0.22, gPx*8.0));
    bead(p-tip, 0.045, cQ); bead(p-tip, 0.016, vec3(0.0));
  }
  // ruff collar + head (a shape that keeps changing between diamond and disc)
  vec2 cq = p-vec2(0.0, -0.10); float ca = atan(cq.y, cq.x);
  float dcl = length(cq*vec2(1.0, 1.9))-0.30*(0.78+0.22*abs(sin(ca*5.0+ph)));
  part(dcl, cQ, cQ*0.6*pline(fract(ca*10.0/PI)-0.5, 0.12, gPx*8.0));
  vec2 q = p-hc;
  float m = 0.5+0.5*sin(ph*0.75);
  float d = mix((abs(q.x)+abs(q.y)-0.34)*0.707, length(q)-0.27, m);
  float msk = smoothstep(0.02, 0.06, abs(q.y+0.02)-0.07);
  part(d, cL, cP*0.8*diam(rot(0.785)*q*(7.0+5.0*uDetail), gPx*10.0)*msk*step(0.03, abs(q.x)));
  vec2 e = vec2(abs(q.x)-0.105, q.y-0.045);
  part((abs(e.x)+abs(e.y)*0.8-0.055)*0.707, W, vec3(0.0));
  bead(e-vec2(0.012*sin(ph), 0.0), 0.016, W);
  float gr = abs(length(q-vec2(0.0, 0.03))-0.15);
  stroke(gr+step(-0.06, q.y)*9.0, gLW*0.6, W);
}

void fOctopus(vec2 p, float ph, float sd){
  float v;
  p.y -= 0.30;
  float c = sin(ph*1.5);
  for (int i = 0; i < 8; i++) {
    float fi = float(i), b = -PI*0.5+(fi-3.5)*0.43, L = 1.12-0.05*abs(fi-3.5);
    vec2 dir = vec2(cos(b), sin(b));
    vec2 q = rot(-b)*(p-dir*0.10-vec2(0.0, -0.02));
    float s = clamp(q.x, 0.0, L), ag = 4.5*s-ph*2.0+fi*1.9+sd*4.0;
    float am = 0.03+0.24*uWig*s*s;
    float yc = am*sin(ag)+0.10*s*c*sign(fi-3.5), dy = am*4.5*cos(ag);
    float w = 0.072*pow(max(1.0-s/L, 0.0), 0.8)+0.005;
    float d = rib(q, s, yc, dy, w, v);
    float suck = (1.0-smoothstep(0.25, 0.35, length(vec2(fract(s*13.0)-0.5, (v-0.45)*0.9))))*step(0.1, s);
    part(d, cL, pal(sd+0.45+fi*0.06)*suck+cP*0.6*pline(v+0.35, 0.12, gPx*12.0));
  }
  float d = sdEll(p-vec2(0.0, 0.40), vec2(0.30*(1.0+0.06*c), 0.40*(1.0-0.05*c)));
  d = smin(d, sdEll(p-vec2(0.0, 0.05), vec2(0.27, 0.19)), 0.12);
  float id, fq = 10.0+8.0*uDetail;
  float pt = kene(p*fq, gPx*fq)*smoothstep(0.12, 0.25, p.y)+beads(p*fq*1.3, gPx*fq*1.3, id)*smoothstep(0.30, 0.18, p.y)*0.0;
  part(d, cL, pal(sd+0.25+p.y*0.4)*pt*0.85);
  vec2 e = vec2(abs(p.x)-0.135, p.y-0.07);
  bead(e, 0.058, W); part(sdBox(e, vec2(0.036, 0.010))-0.006, vec3(0.0), vec3(0.0));
  stroke(abs(length(e)-0.075), gLW*0.4, cQ);
}

void vineLeaf(vec2 q, float len, vec3 lc, vec3 fc){
  float d = sdVes(q-vec2(len, 0.0), len, len*0.36);
  float veins = pline(q.y, 0.006, gPx)+0.8*pline(fract((q.x-abs(q.y)*1.6)*14.0)-0.5, 0.12, gPx*14.0);
  part(d, lc, fc*veins);
}
void fVine(vec2 p, float ph, float sd, float grow){
  float v, base = -1.55, top = base+3.05*grow;
  float s = clamp(p.y, base, top);
  float k1 = 2.3, k2 = 5.1, a1 = k1*s+sd*6.0+0.35*sin(ph*0.5), a2 = k2*s+ph*0.375+sd*3.0;
  float A1 = 0.20+0.10*uWig, A2 = 0.07;
  float xc = A1*sin(a1)+A2*sin(a2), dx = A1*k1*cos(a1)+A2*k2*cos(a2);
  float tip = smoothstep(0.0, 0.5, top-s);
  // leaves, tendril curls and flowers at the nodes next to this pixel
  float sp = 0.26, kf = floor((p.y-base)/sp+0.5);
  for (int j = -2; j <= 1; j++) {
    float kk = kf+float(j); if (kk < 1.0) continue;
    float sn = base+kk*sp, age = top-sn; if (age <= 0.0) continue;
    float lf = smoothstep(0.0, 0.5, age), side = (mod(kk, 2.0) < 1.0) ? 1.0 : -1.0;
    float b1 = k1*sn+sd*6.0+0.35*sin(ph*0.5), b2 = k2*sn+ph*0.375+sd*3.0;
    vec2 n0 = vec2(A1*sin(b1)+A2*sin(b2), sn);
    float h = h11(kk*7.3+sd*31.0);
    vec2 q = p-n0; q.x *= side;
    if (mod(kk, 3.0) < 1.0) {                      // flower: petals open, beads in the heart
      vec2 fo = q-vec2(0.20, 0.07)*lf;
      stroke(sdSeg(q, vec2(0.0), vec2(0.20, 0.07)*lf), gLW*0.5, cL);
      float bloom = smoothstep(0.3, 1.0, age)*(0.9+0.1*sin(ph+kk));
      float fr = length(fo), fa = atan(fo.y, fo.x)+0.3*sin(ph*0.5+kk);
      float R = 0.17*bloom*(0.45+0.55*abs(cos(fa*3.0)));
      float df = fr-R;
      part(df, cQ, pal(sd+0.5+h*0.3)*(pline(fr-R*0.55, 0.008, gPx)+0.7*pline(fract(fa*6.0/PI)-0.5, 0.10, gPx*5.0)));
      bead(fo, 0.030*bloom, W*0.95);
    } else {
      float la = 0.55+0.25*h+0.12*sin(ph*0.75+kk*1.7);
      vineLeaf(rot(-la)*q, (0.16+0.08*h)*lf, cL, pal(sd+0.2+h*0.2));
      // tendril: a little spiral on the other side
      vec2 tq = vec2(-q.x, q.y)-vec2(0.13, 0.05)*lf;
      float tr = length(tq), ta = atan(tq.y, tq.x);
      float spi = abs(fract(tr*9.0/lf-ta/TAU+0.1*sin(ph+kk))-0.5)/9.0*lf;
      float gd0 = gD; stroke(spi+step(0.105*lf, tr)*9.0, gLW*0.4, cP); gD = min(gd0, tr-0.10*lf+0.03);
      stroke(sdSeg(vec2(-q.x, q.y), vec2(0.0), vec2(0.035, 0.03)*lf), gLW*0.4, cP);
    }
  }
  // twin strand winding round the stem, then the stem
  float tw = xc+0.075*tip*sin(11.0*s+ph*1.0);
  stroke(length(vec2(p.x-tw, p.y-s)), gLW*0.45, cP);
  float d = rib(vec2(p.y, p.x), s, xc, dx, 0.036*tip+0.006, v);
  part(d, cL, cQ*pline(fract(s*13.0+v*0.5)-0.5, 0.14, gPx*13.0)*0.9);
  // growing tip: a bright bud
  if (grow < 0.999) { vec2 tp = vec2(A1*sin(k1*top+sd*6.0+0.35*sin(ph*0.5))+A2*sin(k2*top+ph*0.375+sd*3.0), top);
    gC += W*0.0035/(dot(p-tp, p-tp)+0.0025)*smoothstep(0.30, 0.05, length(p-tp)); gD = min(gD, length(p-tp)-0.03); }
}

void fFireflies(vec2 p, float ph, float sd){
  for (int j = 0; j < 12; j++) {
    float fj = float(j), h = h11(sd*13.0+fj*5.1), h2 = h11(sd*29.0+fj*2.3), h3 = h11(fj*9.7+sd*3.0);
    float fa = (4.0+floor(h*5.0))/16.0, fb = (4.0+floor(h2*5.0))/16.0;   // sixteenths: seamless across the phase wrap
    float ta = ph*fa+h3*TAU, tb = ph*fb+h*TAU;
    float R = 0.35+0.95*sqrt(h3);
    vec2 c = R*vec2(sin(ta)*0.95+0.15*sin(tb*2.0), 0.85*sin(tb)+0.12*sin(ta*3.0));
    vec2 vel = R*vec2(cos(ta)*0.95*fa, 0.85*cos(tb)*fb);
    vec2 q = p-c; float r2 = dot(q, q);
    if (r2 > 0.14) continue;
    float blink = smoothstep(0.15, 0.85, sin(ph*(8.0+floor(h2*7.0))/16.0+h*40.0))*(0.8+0.5*uBass)+0.06;
    vec2 dir = normalize(vel+vec2(1e-4, 0.0));
    vec2 l = vec2(dot(q, dir), dot(q, vec2(-dir.y, dir.x)));
    vec3 lamp = pal(sd+0.1+h*0.25);
    // light trail, halo, lantern
    float tr = exp(-l.y*l.y*900.0)*smoothstep(-0.30, 0.0, l.x)*step(l.x, 0.0)*0.55;
    gC += lamp*blink*(tr+0.0045/(r2+0.0045)*smoothstep(0.07, 0.0, r2));
    // the beetle: wing cases, wings, head, glowing tail
    float wg = 0.5+0.5*sin(ph*9.0+fj);
    vec2 wq = vec2(l.x-0.02, abs(l.y));
    float dwing = sdEll(rot(-(0.9+0.5*wg))*(wq)-vec2(0.055, 0.0), vec2(0.060, 0.022));
    stroke(abs(dwing), gLW*0.35, mix(lamp, W, 0.25)*0.8);
    float db = sdEll(l-vec2(0.02, 0.0), vec2(0.050, 0.026));
    part(db, lamp, vec3(0.0));
    bead(l-vec2(0.075, 0.0), 0.016, lamp);
    bead(l-vec2(-0.035, 0.0), 0.024, mix(lamp, W, 0.25)*blink);
    gD = min(gD, sqrt(r2)+0.04);
  }
}

void fWhale(vec2 p, float ph, float sd){
  float v, x0 = -1.12, x1 = 0.95;
  p.y -= 0.02;
  float xc = clamp(p.x, x0, x1), u = (xc-x0)/(x1-x0);
  float A = 0.06+0.20*uWig, ag = ph*1.25-2.4*(1.0-u), tb = (1.0-u)*(1.0-u);
  float yc = A*tb*sin(ag), dy = A*(-2.0*(1.0-u)*sin(ag)+tb*2.4*cos(ag))/(x1-x0);
  // flukes
  float y0 = A*sin(ph*1.25-2.4), s0 = A*(-2.0*sin(ph*1.25-2.4)+2.4*cos(ph*1.25-2.4))/(x1-x0);
  vec2 fq = rot(-atan(s0)*1.4)*(p-vec2(x0, y0));
  fq.y = abs(fq.y);
  float dfl = sdVes(rot(-2.20)*(fq-vec2(0.04, 0.0))-vec2(0.25, 0.0), 0.26, 0.095);
  // pectoral fin (long, like a humpback's) and a small dorsal fin
  float fa = 1.05+0.30*sin(ph*1.25+0.8);
  vec2 fr = vec2(0.36, A*0.09*sin(ph*1.25-0.9)-0.17);
  float dfin = sdCone(p, fr, fr+0.58*vec2(-cos(fa), -sin(fa)), 0.075, 0.020);
  part(dfin, cL, cP*0.7*pline(fract(length(p-fr)*12.0)-0.5, 0.15, gPx*12.0));
  // body
  float w = mix(0.040, 0.300, smoothstep(0.0, 0.60, u))*sqrt(clamp(1.0-pow(max(u-0.60, 0.0)/0.40, 2.4), 0.0, 1.0))+0.004;
  float d = rib(p, xc, yc, dy, w, v);
  d = smin(d, dfl, 0.05);
  vec2 dq = p-vec2(-0.30, yc+0.20);
  d = smin(d, sdCone(dq, vec2(0.08, 0.0), vec2(-0.06, 0.11), 0.05, 0.012), 0.05);
  float fq2 = 9.0+7.0*uDetail, id;
  float pleat = pline(fract(v*6.0)-0.5, 0.10, gPx*24.0)*smoothstep(-0.12, -0.25, v)*smoothstep(0.50, 0.62, u);
  float kq = kene(vec2(u*fq2*2.2, v*fq2*0.33), gPx*fq2*1.1)*smoothstep(-0.05, 0.10, v)*smoothstep(0.02, 0.2, u);
  float st = 0.7*beads(vec2(u*38.0, v*7.0), gPx*18.0, id)*step(0.78, id)*step(v, -0.02)*step(u, 0.55);
  float mouth = pline(v+0.12-0.10*smoothstep(0.75, 1.0, u), 0.03, gPx*4.0)*smoothstep(0.62, 0.68, u);
  part(d, cL, pal(sd+0.2+u*0.4)*kq*0.85+cP*0.75*pleat+W*st+cQ*mouth);
  vec2 e = p-vec2(0.50, yc+0.045);
  bead(e, 0.030, W); bead(e, 0.014, vec3(0.0));
  // breath: a soft plume of sparks over the blowhole now and then
  float blow = smoothstep(0.75, 1.0, sin(ph*0.375+sd*5.0));
  vec2 bq = p-vec2(0.42, yc+0.30);
  gC += W*spark(bq*vec2(1.0, 0.6)+vec2(0.0, -uScroll.y), 9.0)*blow*smoothstep(0.30, 0.0, abs(bq.x)-0.25*bq.y)*step(0.0, bq.y)
      * smoothstep(0.55, 0.2, bq.y)*1.5;
}

float sdTri(vec2 p, vec2 p0, vec2 p1, vec2 p2){
  vec2 e0 = p1-p0, e1 = p2-p1, e2 = p0-p2, v0 = p-p0, v1 = p-p1, v2 = p-p2;
  vec2 q0 = v0-e0*clamp(dot(v0, e0)/dot(e0, e0), 0.0, 1.0);
  vec2 q1 = v1-e1*clamp(dot(v1, e1)/dot(e1, e1), 0.0, 1.0);
  vec2 q2 = v2-e2*clamp(dot(v2, e2)/dot(e2, e2), 0.0, 1.0);
  float sg = sign(e0.x*e2.y-e0.y*e2.x);
  vec2 d = min(min(vec2(dot(q0, q0), sg*(v0.x*e0.y-v0.y*e0.x)), vec2(dot(q1, q1), sg*(v1.x*e1.y-v1.y*e1.x))),
               vec2(dot(q2, q2), sg*(v2.x*e2.y-v2.y*e2.x)));
  return -sqrt(d.x)*sign(d.y);
}
float kitLeg(vec2 p, vec2 hip, float g){                  // a short trotting leg with a round paw
  float sn = sin(g), lift = max(cos(g), 0.0);
  vec2 foot = hip+vec2(0.14*sn, -0.26+0.09*lift);
  return smin(sdCone(p, hip, foot, 0.085, 0.062), sdEll(p-foot-vec2(0.028, -0.006), vec2(0.078, 0.046)), 0.03);
}
// kitten: side-on little body, big round head turned to look at you
void fKitten(vec2 p, float ph, float sd, float g){
  float v;
  p.y += 0.06-0.022*sin(g*2.0);
  // far legs
  part(min(kitLeg(p, vec2(-0.36, -0.24), g+PI), kitLeg(p, vec2(0.08, -0.26), g+PI*0.5)), cL*0.45, vec3(0.0));
  // tail: up and curled over the back, the curl sways
  vec2 tq = rot(-2.05)*(p-vec2(-0.50, -0.06));
  float tx = clamp(tq.x, 0.0, 0.66), curl = 1.25+0.55*sin(ph*2.0+sd*4.0);
  float ty = -curl*tx*tx+0.018*sin(9.0*tx-ph*3.0), tdy = -2.0*curl*tx+0.018*9.0*cos(9.0*tx-ph*3.0);
  float dtl = rib(tq, tx, ty, tdy, 0.058-0.022*tx, v);
  part(dtl, cL, cQ*pline(fract(tx*8.0)-0.5, 0.17, gPx*8.0)*step(0.08, tx));
  // body + near legs
  float d = sdEll(p-vec2(-0.16, -0.12), vec2(0.41, 0.25));
  d = smin(d, kitLeg(p, vec2(-0.42, -0.24), g), 0.05);
  d = smin(d, kitLeg(p, vec2(0.13, -0.26), g+PI*1.5), 0.05);
  float fq = 5.5+4.0*uDetail, ps = gPx*fq;
  float tab = pline(fract(p.x*fq+0.55*sin(p.y*8.0+sd*5.0))-0.5, 0.13, ps)*smoothstep(-0.20, -0.06, p.y)*pfade(ps);
  float id; float toe = beads(p*fq*2.2, ps*2.2, id)*smoothstep(-0.16, -0.26, p.y)*step(0.5, id);
  part(d, cL, pal(sd+0.22+floor(p.x*fq)*0.07)*tab*0.9+cQ*toe*0.8);
  // head (tilts), cheeks, ears
  vec2 hc = vec2(0.36, 0.25+0.015*sin(g));
  vec2 q = rot(-0.13*sin(ph*1.0+sd*3.0))*(p-hc), qa = vec2(abs(q.x), q.y);
  float dear = sdTri(qa, vec2(0.345, 0.10), vec2(0.315, 0.52), vec2(0.06, 0.29))-0.022;
  float dh = smin(sdEll(q, vec2(0.37, 0.325)), length(qa-vec2(0.25, -0.13))-0.135, 0.08);
  dh = smin(dh, dear, 0.03);
  float inner = sdTri(qa, vec2(0.30, 0.25), vec2(0.285, 0.43), vec2(0.15, 0.31));
  float brow = pline(fract(q.x*10.0)-0.5, 0.15, gPx*10.0)*smoothstep(0.17, 0.21, q.y)*step(abs(q.x), 0.15)*step(q.y, 0.30);
  part(dh, cL, cQ*0.9*(1.0-smoothstep(-gPx, gPx, inner))+cP*brow);
  // whiskers
  for (int i = 0; i < 3; i++) {
    float fi = float(i)-1.0;
    stroke(sdSeg(qa, vec2(0.21, -0.105), vec2(0.56, -0.09+0.085*fi+0.012*sin(ph*3.0+fi))), gLW*0.32, W*0.8);
  }
  // big eyes that blink, pupils open with the bass
  float cyc = fract(ph*TURN*0.25+sd*7.3);                              // 16 blink cycles per phase wrap
  float b1 = (cyc-0.50)/0.030, b2 = (cyc-0.15)/0.025;
  float o = clamp(1.0-exp(-b1*b1)-exp(-b2*b2), 0.10, 1.0);
  vec2 e = vec2(qa.x-0.158, q.y-0.005);
  float er = length(e);
  vec3 iris = pal(sd+0.52)*(0.55+0.45*smoothstep(0.02, 0.10, er));
  part(sdEll(e, vec2(0.108, 0.108*o)), W, iris);
  if (o > 0.3) {
    vec2 gz = vec2(0.012*sin(ph*0.5+sd), 0.0);
    part(sdEll(vec2(q.x-sign(q.x)*0.158, e.y)-gz, vec2(0.030+0.034*uBass, 0.082*o)), vec3(0.0), vec3(0.0));
    bead(vec2(q.x-sign(q.x)*0.158, e.y)-vec2(-0.034, 0.040), 0.024, W);
    bead(vec2(q.x-sign(q.x)*0.158, e.y)-vec2(0.030, -0.036), 0.011, W);
  }
  // nose, mouth, bell
  part(sdTri(q, vec2(-0.034, -0.085), vec2(0.034, -0.085), vec2(0.0, -0.125))-0.008, cQ, cQ*0.9);
  stroke(abs(length(vec2(qa.x-0.046, q.y+0.130))-0.046)+step(-0.130, q.y)*9.0, gLW*0.42, cL);
  bead(p-hc-vec2(-0.10, -0.345), 0.042, cQ); bead(p-hc-vec2(-0.10, -0.355), 0.013, vec3(0.0));
}

const vec2 EXT[NFORMS] = vec2[NFORMS](/*EXT*/vec2(1.7)/*EXT*/);

vec4 being(int form, vec2 p, float ph, float sd, float gait, float grow){
  gC = vec3(0.0); gA = 0.0; gD = 1e3;
  bool alongX = form < 5 || form == 6 || form == 7 || form >= 15;      // long bodies: head .. tail, the rest: top .. bottom
  gG = clamp(0.5-(alongX ? p.x : p.y)/2.6, 0.0, 1.0);
  gHueSeed = fract(sd*0.754877);
  cL = mix(pal(sd), W, 0.30); cP = pal(sd+0.22); cQ = pal(sd+0.48);
  if      (form == 0)  fSerpent(p, ph, sd);
  else if (form == 1)  fDragon(p, ph, sd);
  else if (form == 2)  fJaguar(p, ph, sd, gait);
  else if (form == 3)  fHummingbird(p, ph, sd);
  else if (form == 4)  fButterflies(p, ph, sd);
  else if (form == 5)  fJellyfish(p, ph, sd);
  else if (form == 6)  fManta(p, ph, sd);
  else if (form == 7)  fKoi(p, ph, sd);
  else if (form == 8)  fPhoenix(p, ph, sd);
  else if (form == 9)  fEye(p, ph, sd);
  else if (form == 10) fAngel(p, ph, sd);
  else if (form == 11) fJester(p, ph, sd);
  else if (form == 12) fOctopus(p, ph, sd);
  else if (form == 13) fVine(p, ph, sd, grow);
  else if (form == 14) fFireflies(p, ph, sd);
  else if (form == 15) fWhale(p, ph, sd);
  else                 fKitten(p, ph, sd, gait);
  float edge = smoothstep(1.70, 1.45, length(p));                    // nothing may be cut by the bounding circle
  float halo = exp(-max(gD, 0.0)*(26.0-14.0*uGlow))*(0.10+0.55*uGlow)*(1.0-gA);
  vec3 c = gC*(0.70+0.45*uGlow)*(1.0+0.40*uBass)+cL*halo*(0.8+0.6*uBass);
  c += W*spark(p, 11.0)*uHigh*(1.2*gA+1.5*halo);
  if (uCMode > 4.5) {                                                // light chase: a window of light runs down the body
    float w = 1.0-abs(fract(gG-uChasePh)-0.5)*2.0;
    c *= 0.16+1.20*smoothstep(1.0-uChaseW, 1.0, w);
  }
  if (uSat != 1.0) c = max(mix(vec3(dot(c, vec3(0.299, 0.587, 0.114))), c, uSat), 0.0);
  if (uBright != 1.0) c *= uBright;
  return vec4(c*edge, gA*edge);
}

void main(){
  vec2 P = (v_uv-0.5)*vec2(uRes.x/uRes.y, 1.0)*uZoom;
  vec3 col = vec3(0.0); float al = 0.0;
  for (int i = 0; i < uN; i++) {
    vec2 q = P-uA[i].xy; float sc = uA[i].w, fade = uC[i].w;
    if (fade <= 0.0 || dot(q, q) > sc*sc*2.89) continue;
    q = rot(-uA[i].z)*q/sc;
    float fl = uB[i].w; fl = (fl < 0.0 ? -1.0 : 1.0)*max(abs(fl), 0.40);
    q.x /= fl;
    q.y -= 0.5*uC[i].x*q.x*q.x;
    int form = int(uB[i].y+0.5);
    if (any(greaterThan(abs(q), EXT[form]*uCull))) continue;         // outside this form's box: nothing to draw
    gPx = uZoom/(uRes.y*sc);
    gLW = max(0.0105, 0.85*gPx);
    vec4 r = being(form, q, uB[i].x, uB[i].z, uC[i].y, uC[i].z);
    if (any(isnan(r)) || any(isinf(r))) r = vec4(0.0);
    r = clamp(r, 0.0, 4.0)*fade;
    col = col*(1.0-min(r.a, 1.0))+r.rgb;
    al = max(al, min(r.a, 1.0));
  }
  col = 1.0-exp(-col*1.6);                                            // soft shoulder: strokes glow, never clip flat
  vec4 o = vec4(clamp(col*1.06, 0.0, 1.0)*uAmt, al*uAmt);
  if (any(isnan(o)) || any(isinf(o))) o = vec4(0.0);
  f_col = o;
}
"""


def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _val(values, name, default, lo, hi):
    return min(max(_num(values.get(name, default), default), lo), hi)


def form_of(i, base, mix):
    """Form index of being i: being 0 always shows the chosen form; `mix` is the share of the others that differ."""
    base = int(base) % len(FORMS)
    if i == 0 or HERD_H[i, 6] >= mix:
        return base
    return (base + 1 + int(HERD_H[i, 7] * (len(FORMS) - 1))) % len(FORMS)


HERD_H = np.random.default_rng(23).random((N_MAX, 8))     # fixed per-being constants
HERD_H[:, 6] = (np.arange(N_MAX) / N_MAX + 0.06) % 1.0    # mix thresholds evenly spread: mix 0.5 = half the herd differs


class Herd:
    """Where every being is and what its body is doing. All phases are doubles; step() wraps what the shader gets.

    Each of the N_MAX slots has a fade envelope: a change of form, count or variety fades the slot out (FADE_S),
    swaps it while it is invisible and fades it back in, so nothing pops or teleports and an LFO on being_count
    breathes instead of strobing. The summed box area of the visible beings is capped (AREA_BUDGET): many beings at
    a big size shrink together, which also bounds the cost of the pass."""

    def __init__(self, aspect=1.6):
        self.aspect = aspect
        self.s = 0.0                               # path phase
        self.a = 0.0                               # animation phase
        self.gait = np.zeros(N_MAX)                # distance walked, in body lengths (legs)
        self.spin = np.zeros(N_MAX)
        self.pos = None
        self.form = None                           # form on show per slot (follows the wanted form through a fade)
        self.env = np.zeros(N_MAX)                 # fade envelope per slot, 0..1
        self.snap = True                           # next step: no fades, show the wanted herd at once (start / layer was off)
        self.load = 0.0                            # summed box area on show, content units (the frame is aspect x 1)
        self.cap = 1.0                             # scale factor the area budget applied this step

    def _path(self, s, i, form, scale):
        """Centre of being i at path phase s -> (x, y) in content units. Stays inside the frame by construction."""
        _, _, motion, _, reach = FORMS[form]
        h = HERD_H[i]
        mx = min(reach * scale * 0.85, self.aspect * 0.5 - 0.06)
        my = min(reach * scale * 0.85, 0.5 - 0.05)
        X, Y = self.aspect * 0.5 - mx, 0.5 - my
        if motion == ANCHOR:
            lanes = ((i * 0.618 + h[0] * 0.2 + 0.1) % 1.0) * 2.0 - 1.0
            return lanes * X * 0.92 + 0.01 * math.sin(0.3 * s + i), 0.0
        rate = {HEAD: 1.0, SIDE: 0.8, FLOAT: 0.45, TUMBLE: 0.7, CLUSTER: 0.3}[motion]
        s = s * rate
        if FORM_KEYS[form] == "kitten":            # kittens share one track, each a little behind the next: they chase
            h, s = HERD_H[0], s * 1.3 - 1.7 * i
        x = X * (0.66 * math.sin((0.23 + 0.19 * h[0]) * s + TAU * h[4]) + 0.34 * math.sin((0.61 + 0.30 * h[1]) * s + TAU * h[5]))
        y = Y * (0.66 * math.sin((0.29 + 0.17 * h[2]) * s + TAU * h[5]) + 0.34 * math.sin((0.53 + 0.30 * h[3]) * s + TAU * h[4]))
        if FORM_KEYS[form] == "jaguar":            # it walks: a lane, not a flight path
            y = Y * (0.75 * (2.0 * h[3] - 1.0)) + 0.02 * Y * math.sin(0.5 * s + i)
        elif FORM_KEYS[form] == "hummingbird":     # hover + dart
            x += 0.012 * math.sin(3.1 * s + i)
            y += 0.016 * math.sin(4.3 * s + 2.0 * i)
        elif motion == CLUSTER:
            x, y = x * 0.5, y * 0.5
        return x, y

    @staticmethod
    def raw_scale(i, form, size):
        """Scale of slot i showing `form` before the area budget."""
        if FORMS[form][2] == ANCHOR:               # the vine spans the frame; size only makes it bolder
            return 0.30 + 0.06 * size
        return (0.105 + 0.215 * size) * FORMS[form][3] * (0.88 + 0.24 * HERD_H[i, 2])

    def layout(self, values):
        """What the values ask for: (forms, scales before the area budget), lists of length N_MAX."""
        base = int(round(_val(values, "being", 0.0, 0.0, len(FORMS) - 1)))
        mix = _val(values, "being_mix", 0.0, 0.0, 1.0)
        size = _val(values, "being_size", 0.5, 0.0, 1.0)
        forms = [form_of(i, base, mix) for i in range(N_MAX)]
        return forms, [self.raw_scale(i, f, size) for i, f in enumerate(forms)]

    def _fades(self, dt, want, n):
        if self.form is None or self.snap:
            self.form = list(want)
            self.env = np.array([1.0 if i < n else 0.0 for i in range(N_MAX)])
            self.snap = False
            return
        step = dt / FADE_S
        for i in range(N_MAX):
            if i < n and self.form[i] == want[i]:
                self.env[i] = min(1.0, self.env[i] + step)
            else:
                self.env[i] = max(0.0, self.env[i] - step)
                if self.env[i] <= 0.0:
                    self.form[i] = want[i]         # swapped while invisible; fades in from the next step if wanted

    def step(self, dt, values, bands=(0.0, 0.0, 0.0), clock=None):
        """Advance by dt seconds and return the uniform arrays (uA, uB, uC) + the count asked for."""
        beat = _num((clock or {}).get("beat", 0.0))
        dt = min(max(_num(dt), 0.0), 0.1)
        speed = _val(values, "being_speed", 0.4, 0.0, 1.0)
        size = _val(values, "being_size", 0.5, 0.0, 1.0)
        n = int(round(_val(values, "being_count", 3.0, 1.0, N_MAX)))
        bass = min(max(_num(bands[0]) if len(bands) else 0.0, 0.0), 1.0)
        ds = dt * speed * 2.6
        self.s += ds
        self.a += dt * (0.45 + 1.5 * speed) * (1.0 + 0.35 * bass)
        want, _ = self.layout(values)
        self._fades(dt, want, n)
        forms = self.form
        vis = self.env * self.env * (3.0 - 2.0 * self.env)
        scales = [self.raw_scale(i, f, size) for i, f in enumerate(forms)]
        area = [4.0 * FORM_EXT[f][0] * FORM_EXT[f][1] * sc * sc for f, sc in zip(forms, scales)]
        fixed = sum(a * v for a, v, f in zip(area, vis, forms) if FORMS[f][2] == ANCHOR)
        free = sum(a * v for a, v, f in zip(area, vis, forms) if FORMS[f][2] != ANCHOR)
        self.cap = min(1.0, math.sqrt(max(AREA_BUDGET - fixed, 0.25 * AREA_BUDGET) / free)) if free > 1e-9 else 1.0
        scales = [sc if FORMS[f][2] == ANCHOR else sc * self.cap for f, sc in zip(forms, scales)]
        self.load = fixed + free * self.cap * self.cap
        ua, ub, uc = (np.zeros((N_MAX, 4), dtype="f4") for _ in range(3))
        eps = 0.02
        new_pos = np.zeros((N_MAX, 2))
        for i in range(N_MAX):
            f, sc = forms[i], scales[i]
            motion = FORMS[f][2]
            x, y = self._path(self.s, i, f, sc)
            x1, y1 = self._path(self.s + eps, i, f, sc)
            x0, y0 = self._path(self.s - eps, i, f, sc)
            vx, vy = (x1 - x0) / (2 * eps), (y1 - y0) / (2 * eps)
            ang, face, bend, grow, fade = 0.0, 1.0, 0.0, 1.0, 1.0
            if motion == HEAD:
                ang = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-9 else 0.0
                h0, h1 = math.atan2(y - y0, x - x0), math.atan2(y1 - y, x1 - x)
                turn = (h1 - h0 + math.pi) % TAU - math.pi                  # heading change over eps
                step_len = math.hypot(x1 - x, y1 - y) + 1e-9
                bend = min(max(turn / step_len * sc, -0.55), 0.55)          # curvature in local units
            elif motion == SIDE:
                vmax = 0.10 * (self.aspect * 0.5)
                face = min(max(vx / vmax, -1.0), 1.0)
                pitch = math.atan2(vy, abs(vx) + 0.03) * (0.25 if FORM_KEYS[f] == "jaguar" else 0.5)
                ang = (1.0 if face >= 0 else -1.0) * min(max(pitch, -0.4), 0.4)
                if FORM_KEYS[f] == "kitten":
                    k = int(math.floor(beat)) + i
                    if k % 2 == 0:                                          # pounce: one hop per beat, kittens alternate
                        bp_ = beat - math.floor(beat)
                        y += 0.50 * sc * 4.0 * bp_ * (1.0 - bp_)
                        ang += (1.0 if face >= 0 else -1.0) * 0.35 * (1.0 - 2.0 * bp_)
                    cyc = (self.a * 0.05 + HERD_H[i, 4]) % 1.0              # now and then a forward roll
                    if cyc < 0.10:
                        r = cyc / 0.10
                        ang -= (1.0 if face >= 0 else -1.0) * TAU * r * r * (3.0 - 2.0 * r)
            elif motion == FLOAT:
                ang = 0.16 * math.sin(0.5 * self.a + 5.0 * HERD_H[i, 1]) - 0.9 * min(max(vx, -0.25), 0.25)
            elif motion == TUMBLE:
                self.spin[i] += dt * (0.15 + 0.5 * speed) * (0.7 + 0.6 * HERD_H[i, 3]) * (1.0 if i % 2 == 0 else -1.0)
                ang = self.spin[i] + 0.7 * math.sin(0.9 * self.a + i)
            elif motion == ANCHOR:
                cyc = (self.a * 0.045 + HERD_H[i, 4]) % 1.0                 # grow, stand, dissolve, grow again
                grow = min(cyc / 0.55, 1.0)
                grow = grow * grow * (3.0 - 2.0 * grow)
                fade = min(max((1.0 - cyc) / 0.08, 0.0), 1.0)
                ang = 0.03 * math.sin(0.4 * self.a + i)
            new_pos[i] = (x, y)
            if self.pos is not None and motion == SIDE and vis[i] > 0.0:
                self.gait[i] += min(math.hypot(*(new_pos[i] - self.pos[i])), 0.05) / max(sc, 1e-6) * (TAU / 0.9)
            pulse = 1.0 + (0.0 if motion == ANCHOR else 0.10 * bass)
            ua[i] = (x, y, ang % TAU, sc * pulse)
            ub[i] = ((self.a * (0.85 + 0.3 * HERD_H[i, 0]) + 40.0 * HERD_H[i, 1]) % PHASE_WRAP, f, HERD_H[i, 5] + 0.37 * i, face)
            uc[i] = (bend, self.gait[i] % TAU, grow, fade * vis[i])
        self.pos = new_pos
        return ua, ub, uc, n


def pick_res(res, load):
    """Resolution tier for a load (in budgets), with hysteresis so the pass never flaps between targets."""
    if load > MIN_RES_ON or (res == MIN_RES and load > MIN_RES_OFF):
        return MIN_RES
    if load > LOW_RES_ON or (res != 1.0 and load > LOW_RES_OFF):
        return LOW_RES
    return 1.0


def _blend_query():
    """() -> bool: is GL_BLEND enabled in the current context (raw GL; moderngl has no getter). None if unavailable."""
    try:
        import ctypes
        lib = ctypes.windll.opengl32 if sys.platform == "win32" else ctypes.CDLL("libGL.so.1")
        lib.glIsEnabled.restype = ctypes.c_ubyte                     # GLboolean, not int
        lib.glIsEnabled.argtypes = [ctypes.c_uint]
        return lambda: bool(lib.glIsEnabled(0x0BE2))
    except Exception:
        return None


class BeingsPass:
    """render(values, clock, t, dt, bands, palette_rgb) -> RGBA8 texture (see module docstring).

    The texture is the size of the output; while the herd covers a lot of the frame (Herd.load) the pass draws into
    a LOW_RES copy instead and returns that one (linear filtered, the compositor samples it by uv either way)."""

    def __init__(self, ctx: moderngl.Context, out_size=(1280, 800)):
        self.ctx = ctx
        self.out_size = (max(16, int(out_size[0])), max(16, int(out_size[1])))
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._prog = ctx.program(vertex_shader=VS, fragment_shader=shader_source())
        self._vao = ctx.vertex_array(self._prog, [(self._quad, "2f", "in_pos")])
        self._targets = {}
        self._target(1.0)
        self.herd = Herd(self.out_size[0] / self.out_size[1])
        self.draws = 0                                   # shader passes drawn so far (tests / cost checks)
        self.view_zoom = 1.0                             # > 1 looks at the frame from further out (tests: what is off screen)
        self.cull = 1.0                                  # per-form cull box scale (measure_extents switches it off)
        self.res = 1.0                                   # resolution scale of the last draw (1.0 or LOW_RES)
        self.blend_assumed_off = False                   # True: GL_BLEND could not be read back, it is left disabled
        self._blank = True
        self._last_t = None
        self._pal = np.asarray(GALLERY_PALETTE, dtype="f4")
        self._blend_on = _blend_query()
        if self._blend_on is None:
            self.blend_assumed_off = True

    def _target(self, res):
        if res not in self._targets:
            size = self.out_size if res == 1.0 else (max(16, int(self.out_size[0] * res)), max(16, int(self.out_size[1] * res)))
            tex = self.ctx.texture(size, 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._targets[res] = (tex, fbo)
        return self._targets[res]

    def release(self):
        objs = [o for pair in self._targets.values() for o in pair[::-1]] + [self._vao, self._prog, self._quad]
        self._targets = {}
        for o in objs:
            try:
                o.release()
            except Exception:
                pass

    close = release

    def _set(self, name, value):
        if name in self._prog:
            self._prog[name].value = value

    def render(self, values, clock=None, t=0.0, dt=None, bands=(0.0, 0.0, 0.0), palette_rgb=None) -> moderngl.Texture:
        ctx = self.ctx
        amt = _val(values, "l_beings", 0.0, 0.0, 1.0)
        t = _num(t)
        if dt is None:                                   # no dt from the caller: derive it from show time
            dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        bands = [min(max(_num(b), 0.0), 1.0) for b in (list(bands if bands is not None else ()) + [0, 0, 0])[:3]]
        if palette_rgb is not None:
            try:
                pal = np.asarray(palette_rgb, dtype="f4").reshape(-1, 3)
                if len(pal) and np.isfinite(pal).all():
                    self._pal = np.clip(pal[np.arange(5) % len(pal)], 0.0, 1.0)
            except (TypeError, ValueError):
                pass
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        blend_on = self._blend_on() if self._blend_on is not None else False   # unreadable: assumed off, left off
        try:
            if amt <= 0.001:                             # off: nothing moves, nothing is drawn
                if not self._blank:
                    for _tex, fbo in self._targets.values():
                        fbo.clear(0.0, 0.0, 0.0, 0.0)
                    self._blank = True
                self.herd.snap = True                    # back on later: the herd of that moment, no stale fades
                return self._targets[1.0][0]
            ua, ub, uc, _n = self.herd.step(dt, values, bands, clock)
            self.res = pick_res(self.res, self.herd.load / AREA_BUDGET)
            tex, fbo = self._target(self.res)
            fbo.use()
            ctx.viewport = (0, 0, *tex.size)
            ctx.disable(moderngl.BLEND)
            self._set("uRes", tuple(float(x) for x in tex.size))
            self._set("uN", N_MAX)
            for name, arr in (("uA", ua), ("uB", ub), ("uC", uc)):
                if name in self._prog:
                    self._prog[name].write(arr.tobytes())
            if "uPal" in self._prog:
                self._prog["uPal"].write(self._pal.astype("f4").tobytes())
            self._set("uT", t % PHASE_WRAP)              # wrapped in double precision, twinkle rates are k/8
            self._set("uScroll", ((t * 0.12) % (64.0 / 5.0), (t * 0.30) % (64.0 / 9.0)))
            self._set("uBass", bands[0]); self._set("uMid", bands[1]); self._set("uHigh", bands[2])
            self._set("uAmt", amt)
            self._set("uZoom", float(self.view_zoom))
            self._set("uCull", float(self.cull))
            self._set("uWig", _val(values, "being_wiggle", 0.5, 0.0, 1.0))
            self._set("uGlow", _val(values, "being_glow", 0.5, 0.0, 1.0))
            self._set("uDetail", _val(values, "being_detail", 0.5, 0.0, 1.0))
            self._set("uCMode", float(round(_val(values, "being_color", 0.0, 0.0, len(COLOR_MODES) - 1))))
            self._set("uHue", _val(values, "being_hue", 0.52, 0.0, 1.0))
            self._set("uHue2", _val(values, "being_hue2", 0.85, 0.0, 1.0))
            self._set("uSat", _val(values, "being_sat", 1.0, 0.0, 1.0))
            self._set("uBright", _val(values, "being_bright", 1.0, 0.0, 2.0))
            chase = values.get("chase_phase")            # the engine's light-chase phase; else the bar, like the engine
            if chase is None:
                chase = _num((clock or {}).get("bar_phase", 0.0)) * _val(values, "chase_rate", 1.0, -4.0, 4.0)
            self._set("uChasePh", _num(chase) % 1.0)
            self._set("uChaseW", _val(values, "chase_width", 0.25, 0.05, 0.95))
            self._vao.render(moderngl.TRIANGLE_STRIP)
            self.draws += 1
            self._blank = False
            return tex
        finally:
            if blend_on:
                ctx.enable(moderngl.BLEND)
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()


def shader_source():
    ext = ", ".join(f"vec2({x:.2f}, {y:.2f})" for x, y in FORM_EXT)
    a, b = FS.index("/*EXT*/"), FS.rindex("/*EXT*/") + len("/*EXT*/")
    return (FS[:a] + ext + FS[b:]).replace("NFORMS", str(len(FORMS)))


def measure_extents(ctx=None, size=(640, 640), steps=40):
    """Half-extents of every form's picture in its own local space, worst case over phases, wiggle, bend and glow
    (how FORM_EXT was made; tests check FORM_EXT still covers it). -> [(x, y)] per form."""
    if ctx is None:
        ctx, _win = hidden_context()
    bp = BeingsPass(ctx, size)
    bp.cull = 1e3
    sc, out = 0.25, []
    try:
        for f, (_, _, motion, _, _) in enumerate(FORMS):
            acc = np.zeros(size[::-1], dtype=bool)
            for k in range(steps):
                ua, ub, uc = (np.zeros((N_MAX, 4), dtype="f4") for _ in range(3))
                bend = (0.0, 0.55, -0.55)[k % 3] if motion == HEAD else 0.0
                ua[0] = (0.0, 0.0, 0.0, sc)
                ub[0] = (PHASE_WRAP * ((k * 0.61803) % 1.0), f, 0.37 * k, 1.0)
                uc[0] = (bend, k * 0.7, 1.0, 1.0)
                bp.herd.step = lambda dt, values, bands, clock=None, r=(ua, ub, uc, 1): r
                bp.herd.load = 0.0
                vals = default_values(being_wiggle=1.0, being_glow=1.0, being_detail=(k % 2))
                img = read_texture(bp.render(vals, None, 3.0 + k * 1.7, 0.0, (1.0, 1.0, 1.0), GALLERY_PALETTE))
                acc |= img.max(axis=2) > 6
            ys, xs = np.nonzero(acc)
            ex = np.abs((xs + 0.5) / size[0] - 0.5).max() / sc * (size[0] / size[1])
            ey = np.abs((ys + 0.5) / size[1] - 0.5).max() / sc
            out.append((float(ex), float(ey)))
    finally:
        bp.release()
    return out


# ---------------------------------------------------------------- headless helpers (tests, gallery, bench)

def hidden_context(size=(64, 64)):
    from math_layers import hidden_context as hc
    return hc(size)


def default_values(**over):
    import params
    v = params.defaults()
    v["l_beings"] = 1.0
    v.update(over)
    return v


def make_clock(t, bpm=124.0):
    b = t * bpm / 60.0
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def read_texture(tex):
    return np.frombuffer(tex.read(), dtype=np.uint8).reshape(tex.height, tex.width, 4)[::-1]


def read_frame(bp, tex):
    """The pass's picture as (h, w, 4) uint8 at out_size, whichever of its targets it drew into."""
    img = read_texture(tex).copy()
    if tex.size != bp.out_size:
        from PIL import Image
        img = np.asarray(Image.fromarray(img, "RGBA").resize(bp.out_size, Image.BILINEAR))
    return img


def render_frames(bp, values, t0=4.0, frames=1, dt=1 / 60, bands=(0.5, 0.3, 0.2), palette=None, warm=0.0):
    """Step the pass from show time t0 (after `warm` seconds of motion in big steps) and return the last frame."""
    for _ in range(int(warm / 0.1)):
        bp.herd.step(0.1, values, bands)
    tex = None
    for i in range(frames):
        t = t0 + i * dt
        tex = bp.render(values, make_clock(t), t, dt, bands, palette or GALLERY_PALETTE)
    return read_frame(bp, tex)


def gallery(out_dir=None, out_size=(1280, 800), only=None):
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir or HERE / "test_frames" / "beings")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, _win = hidden_context()
    shots = [(k, dict(being=i, being_count=3, being_size=0.75, being_mix=0.0), i % 5, 21.0 + 3.0 * i)
             for i, k in enumerate(FORM_KEYS)]
    shots += [("mixed_8", dict(being=0, being_count=8, being_size=0.4, being_mix=1.0), 0, 30.0),
              ("mixed_5_big", dict(being=8, being_count=5, being_size=0.6, being_mix=1.0), 1, 12.0)]
    if only:
        shots = [s for s in shots if s[0] in only]
    try:
        from live_visuals import PALETTES
    except Exception:
        PALETTES = [("gallery", GALLERY_PALETTE)] * 5
    tw, th, cols = 320, 200, 6
    sheet = Image.new("RGB", (tw * cols, (th + 16) * math.ceil(len(shots) / cols)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (tag, over, pal_i, warm) in enumerate(shots):
        bp = BeingsPass(ctx, out_size)
        try:
            pal = [tuple(c) for c in PALETTES[pal_i][1]]
            img = render_frames(bp, default_values(**over), frames=2, palette=pal, warm=warm)
        finally:
            bp.release()
        im = Image.fromarray(img[..., :3])
        im.save(out_dir / f"{tag}.png")
        x, y = (i % cols) * tw, (i // cols) * (th + 16)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        dr.text((x + 4, y + th + 2), tag, fill=(200, 200, 200))
    if not only:
        sheet.save(out_dir / "contact_sheet.png")
    print(f"[beings] {len(shots)} frames -> {out_dir}")
    return out_dir


def colour_sheets(out_dir=None, cell=(480, 300)):
    """colour_modes.png: four forms x the six colour modes (this pass). colour_modes_serpents.png: the engine's
    serpents in the six modes (one small headless engine run)."""
    import json
    import subprocess
    import tempfile
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir or HERE / "test_frames" / "beings")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, _win = hidden_context()
    forms = [FORM_KEYS.index(k) for k in ("serpent", "kitten", "phoenix", "koi")]
    tw, th = cell
    sheet = Image.new("RGB", (tw * len(COLOR_MODES), (th + 18) * len(forms)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for r, f in enumerate(forms):
        for m, name in enumerate(COLOR_MODES):
            bp = BeingsPass(ctx, cell)
            try:
                vals = default_values(being=f, being_size=0.8, being_color=m, being_hue=0.52, being_hue2=0.85, chase_phase=0.3)
                img = render_frames(bp, vals, frames=2, warm=14.0 + 3.0 * r)
            finally:
                bp.release()
            sheet.paste(Image.fromarray(img[..., :3]), (m * tw, r * (th + 18)))
            dr.text((m * tw + 4, r * (th + 18) + th + 3), f"{FORM_KEYS[f]}: {m} {name}", fill=(200, 200, 200))
    sheet.save(out_dir / "colour_modes.png")
    with tempfile.TemporaryDirectory() as tmp:
        base = dict(l_kene=0.0, l_beads=0.0, l_lines=0.0, l_ai=0.0, l_echo=0.0, serpent=1.0, serpent_size=0.7, chase=0.0)
        jobs = [dict(name=f"m{m}", sets=dict(base, serpent_color=m), frames=40) for m in range(len(COLOR_MODES))]
        spec = Path(tmp) / "jobs.json"
        spec.write_text(json.dumps(jobs))
        r = subprocess.run([sys.executable, str(HERE / "live_visuals.py"), "--batch", str(spec), "--out", tmp, "--render",
                            f"{tw}x{th}", "--test-audio", "silent"], capture_output=True, text=True, cwd=HERE)
        if r.returncode == 0:
            done = {d["name"]: d for d in json.loads((Path(tmp) / "batch.json").read_text())}
            sheet = Image.new("RGB", (tw * 3, (th + 18) * 2), (0, 0, 0))
            dr = ImageDraw.Draw(sheet)
            for m, name in enumerate(COLOR_MODES):
                x, y = (m % 3) * tw, (m // 3) * (th + 18)
                sheet.paste(Image.open(Path(tmp) / done[f"m{m}"]["file"]).convert("RGB"), (x, y))
                dr.text((x + 4, y + th + 3), f"serpents: {m} {name}", fill=(200, 200, 200))
            sheet.save(out_dir / "colour_modes_serpents.png")
        else:
            print("[beings] serpent sheet skipped:", r.stderr[-300:])
    print(f"[beings] colour sheets -> {out_dir}")
    return out_dir


def bench(out_size=(1280, 800), frames=120):
    ctx, _win = hidden_context()
    bp = BeingsPass(ctx, out_size)
    rows = []
    cases = [(k, dict(being=i)) for i, k in enumerate(FORM_KEYS)]
    cases += [("mixed x8", dict(being_mix=1.0, being_count=8)), ("serpent x8 big", dict(being_count=8, being_size=1.0))]
    try:
        for tag, over in cases:
            vals = default_values(**over)
            render_frames(bp, vals, frames=3, warm=5.0)
            ctx.finish()
            t0 = time.perf_counter()
            for i in range(frames):
                bp.render(vals, None, 5.0 + i / 60, 1 / 60, (0.5, 0.3, 0.2), GALLERY_PALETTE)
            ctx.finish()
            ms = (time.perf_counter() - t0) / frames * 1000.0
            rows.append((tag, ms))
            print(f"[beings] {tag:16s} {ms:6.2f} ms/frame")
    finally:
        bp.release()
    return rows


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    elif "--colours" in sys.argv:
        colour_sheets()
    else:
        gallery(only=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
