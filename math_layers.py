"""Pure-math (no AI) fractal generator layer for the projection engine.

    fp = FractalPass(ctx, out_size=(1280, 800))
    tex = fp.render(frame.values, frame.clock, t, travel_z, palette_rgb)   # moderngl.Texture, RGBA8, linear
    ...composite tex (black = no light; alpha = max(rgb)) with opacity values["l_fractal"]...
    fp.release()

Modes (values["fractal_mode"], rounded):
  0 crystal KIFS flythrough      raymarched kaleidoscopic IFS, camera flies along z with travel_z
  1 Mandelbox-family tunnel      Julia-type box fold + sphere fold on a mirror lattice, flight along a carved corridor
  2 hyperbolic tiling            Poincare-disk {p,q} tilings, travel = hyperbolic translation (Moebius)
  3 Apollonian gasket            circle-inversion packing on a Droste (log-polar) lattice: exact infinite zoom
  4 Julia orbit-trap jewels      Koenigs-linearised Julia set: exact infinite zoom, pearl + stalk orbit traps
  5 4D polytopes                 tesseract / 24-cell / 600-cell wireframes, 4D -> 3D -> 2D, max-blended glow lines

Standalone:
  python math_layers.py --gallery     # hidden window -> test_frames/math_gallery/*.png + contact_sheet.png
  python math_layers.py --bench       # ms per mode (glFinish timing) at fractal_res 0.5 and 1.0
"""
import argparse
import cmath
import ctypes
import math
import sys
import time
from pathlib import Path

import moderngl
import numpy as np

HERE = Path(__file__).parent
MODE_NAMES = ["kifs", "mandelbox", "hyperbolic", "apollonian", "julia", "polytope"]
TILINGS = [(5, 4), (6, 4), (7, 3), (8, 3), (4, 5)]          # fractal_morph glides through these; 0.5 = {7,3}
TAU = 2.0 * math.pi

VS = """
#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

COMMON = """
#version 330
uniform vec2 uRes;
uniform float uT, uTravel, uRot, uChase, uBars, uRaw;
uniform vec3 pal[5];
uniform float p_zoom, p_symmetry, p_folds, p_swim, p_twirl, p_hue, p_saturation, p_pan_x, p_pan_y;
uniform float p_chase, p_chase_width, p_chase_mode, p_depth_fog, p_tunnel_curve;
uniform float p_fractal_morph, p_fractal_detail, p_fractal_glow, p_fractal_metal;
const float TAU = 6.28318530718, PI = 3.14159265359;
float gCut = 0.0;

mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
vec2 cmul(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cdiv(vec2 a, vec2 b){ return vec2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y)/max(dot(b, b), 1e-12); }
vec3 pal5(float x){ x = fract(x)*5.0; int i = int(x); float k = fract(x); k = k*k*(3.0 - 2.0*k);
  return mix(pal[i % 5], pal[(i + 1) % 5], k); }
vec3 film(float x){ return 0.5 + 0.5*cos(TAU*(x + vec3(0.0, 0.33, 0.67))); }
// thin-film iridescence: the palette colour shifts along the palette and through a spectral film with view angle
vec3 irid(float k, float cosT){
  vec3 base = pal5(k);
  vec3 fl = pal5(k + 0.55*(1.0 - cosT))*(0.45 + 1.1*film(1.4*cosT + 2.0*k));
  return mix(base, fl, p_fractal_metal);
}
// TAS-style light chase: a window of light sweeping along the scalar coordinate k (period 1)
float chaseWin(float k){
  float w = 0.5*p_chase_width;
  float d = abs(fract(k - uChase + 0.5) - 0.5);
  float win = smoothstep(w, 0.3*w, d);
  if (p_chase_mode > 0.5 && p_chase_mode < 1.5) {
    d = abs(fract(k - uChase) - 0.5); win = max(win, smoothstep(w, 0.3*w, d));
  } else if (p_chase_mode >= 1.5) {
    d = abs(fract(k + uChase + 0.5) - 0.5); win = max(win, smoothstep(w, 0.3*w, d));
  }
  return win;
}
float chaseAmt(float k){ return mix(1.0, 0.10 + 1.5*chaseWin(k), p_chase); }
vec3 hueRot(vec3 c, float a){
  const mat3 toY = mat3(.299,.596,.211,.587,-.274,-.523,.114,-.322,.312);
  const mat3 toR = mat3(1.,1.,1.,.956,-.272,-1.106,.621,-.647,1.703);
  vec3 y = toY*c; float cs = cos(a), sn = sin(a);
  return toR*vec3(y.x, y.y*cs - y.z*sn, y.y*sn + y.z*cs);
}
// saturation, hue shift, hue-preserving clip (no white-out), knee to true black
vec4 finish(vec3 c){
  if (uRaw > 0.5) return vec4(c, 1.0);                                // debug: pre-guard, pre-clamp colour (tests)
  if (!(c.r == c.r && c.g == c.g && c.b == c.b)) c = vec3(0.0);       // NaN guard
  c = clamp(c, 0.0, 8.0);
  float l = dot(c, vec3(0.299, 0.587, 0.114));
  c = max(mix(vec3(l), c, p_saturation), 0.0);
  if (p_hue > 0.0005) c = max(hueRot(c, p_hue*TAU), 0.0);
  c /= max(1.0, max(c.r, max(c.g, c.b)));
  c = max(c - 0.012, 0.0)*1.012;
  return vec4(c, max(c.r, max(c.g, c.b)));
}
"""

# ------------------------------------------------------------------ raymarched modes (0, 1)
RM_MAIN = """
out vec4 f;
void main(){
  setup();
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  if (length(uv) > 0.70) { f = vec4(0.0); return; }                   // outside the portal: no light, no march
  vec3 ro, rd; camera(uv, ro, rd);
  float px = 1.0/uRes.y;
  float tt = 0.0, d = 1.0, acc = 0.0; vec4 tr = vec4(1e3); int i; bool hit = false;
  for (i = 0; i < uSteps; i++) {
    d = de(ro + rd*tt, tr);
    if (d < px*tt*0.8 + 2e-4) { hit = true; break; }
    acc += exp(-d*GLOWK);
    tt += d*STEPK;
    if (tt > MAXD) break;
  }
  if (!hit && tt < MAXD && d < px*tt*6.0) hit = true;                 // ran out of steps right at a surface
  float sfrac = float(i)/float(uSteps);
  float fog = exp(-tt*tt*mix(0.004, 0.11, p_depth_fog)*FOGK);
  vec3 col = vec3(0.0);
  float kb = 0.06*tt + 0.015*uT;
  if (hit) {
    float cut = gCut;
    vec3 pos = ro + rd*tt; float h = max(px*tt*1.2, 4e-4); vec4 t2; vec2 e = vec2(1.0, -1.0);
    vec3 n = normalize(e.xyy*de(pos + e.xyy*h, t2) + e.yyx*de(pos + e.yyx*h, t2) +
                       e.yxy*de(pos + e.yxy*h, t2) + e.xxx*de(pos + e.xxx*h, t2));
    float cosT = clamp(dot(n, -rd), 0.0, 1.0), fres = pow(1.0 - cosT, 2.5);
    float k = TRAPK*log(1.0 + tr.w) + kb + 0.10*n.y + 0.12*tt;
    float mtr = min(tr.x, min(tr.y, tr.z));
    float seam = smoothstep(SEAMW, 0.35*SEAMW, mtr) + (1.0 - cut)*(0.05 + 0.4*p_fractal_glow)*exp(-mtr*SEAMK*2.0);
    float ao = clamp(1.25 - 1.3*sfrac, 0.0, 1.0);
    float ch = chaseAmt(BANDK*tt + 0.35*log(1.0 + tr.w));
    float lit = (LIT0 + LIT1*pow(cosT, 5.0) + (0.12 + 0.9*p_fractal_metal)*fres)*(1.0 - cut);
    col = irid(k, cosT)*lit*ao*ch;
    col += pal5(k + 0.4)*seam*(0.5 + 0.7*p_fractal_glow)*(0.2 + 0.8*ch)*ao*mix(1.0, CUTSEAM, cut);
    col *= fog*(LB0 + 1.5*LBAND(pos));
  }
  // glow from the march: near-misses and step count light the air around the crystal
  float g = sfrac*sfrac*(hit ? 0.45 : 1.0) + 0.15*acc/float(uSteps);
  col += pal5(kb + 0.2)*g*(1.6*p_fractal_glow)*GLOWA*fog*chaseAmt(BANDK*tt);
  col *= GAIN*smoothstep(0.70, 0.40, length(uv));
  f = finish(col);
}
"""

KIFS_FS = COMMON + """
uniform int uSteps, uIters;
uniform vec3 uOff, uBox; uniform vec4 uRAB; uniform float uSc, uN, uTw, uRing, uCarve;
#define GLOWK 14.0
#define STEPK 0.85
#define MAXD 14.0
#define FOGK 1.0
#define TRAPK 0.55
#define SEAMK 26.0
#define BANDK 0.22
#define GLOWA 1.0
#define SEAMW 0.05
#define LIT0 0.03
#define LIT1 0.40
#define CUTSEAM 1.0
#define LBAND(q) pow(0.5 + 0.5*cos(TAU*(q.z/2.6 - uT*0.05)), 2.0)
#define LB0 0.16
#define GAIN 1.0
mat2 gRA, gRB;
void setup(){ gRA = mat2(uRAB.x, -uRAB.y, uRAB.y, uRAB.x); gRB = mat2(uRAB.z, -uRAB.w, uRAB.w, uRAB.z); }
float de(vec3 p, out vec4 tr){
  p.xy = rot(p.z*uTw)*p.xy;
  p += p_swim*0.05*sin(p.yzx*2.1 + vec3(0.7, 1.9, 0.0)*uT*0.6 + uT*0.35);
  float r0 = length(p.xy);
  float sA = TAU/uN, a = atan(p.y, p.x + 1e-6);
  a = abs(mod(a + 0.5*sA, sA) - 0.5*sA);
  p.xy = r0*vec2(cos(a), sin(a));
  p.z = abs(mod(p.z, 2.0) - 1.0) - 0.5;
  p.x -= uRing;
  float s = 1.0; tr = vec4(1e3);
  for (int i = 0; i < uIters; i++) {
    p = abs(p) - uOff;
    p.xy = gRA*p.xy; p.yz = gRB*p.yz;
    p *= uSc; s *= uSc;
    tr = min(tr, vec4(abs(p), dot(p, p)));
  }
  vec3 q = abs(p) - uBox;
  float d = (length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0))/s;
  gCut = step(d, uCarve - r0);
  d = max(d, uCarve - r0);                                           // the flight corridor is always clear
  return d/(1.0 + abs(uTw)*r0 + 0.25*p_swim);
}
void camera(vec2 uv, out vec3 ro, out vec3 rd){
  float z = uTravel;
  ro = vec3(0.05*p_tunnel_curve*vec2(sin(z*0.9), cos(z*0.7)), z);
  vec3 fw = normalize(vec3(0.22*p_tunnel_curve*vec2(cos(z*0.9 + 0.4), -sin(z*0.7 + 0.4)) + vec2(p_pan_x, p_pan_y)*0.5, 1.0));
  vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
  uv = rot(uRot)*uv;
  rd = normalize(fw*(0.62*exp(0.8*p_zoom)) + rt*uv.x + up*uv.y);
}
""" + RM_MAIN

MBOX_FS = COMMON + """
uniform int uSteps, uIters;
uniform vec3 uCC, uCam; uniform float uN, uKal, uThick, uScale, uMinR2, uCarve;
#define GLOWK 10.0
#define STEPK 0.8
#define MAXD 9.0
#define FOGK 2.2
#define TRAPK 0.9
#define SEAMK 16.0
#define BANDK 0.30
#define GLOWA 0.8
#define SEAMW 0.045
#define LIT0 0.06
#define LIT1 0.75
#define CUTSEAM 0.8
#define LBAND(q) pow(0.5 + 0.5*cos(TAU*(q.x/2.6 - uT*0.05)), 1.5)
#define LB0 0.60
#define GAIN 1.5
void setup(){}
// Mandelbox-family corridor: Julia-type box fold + sphere fold on an exact mirror lattice (period 4), box ends,
// with a cylinder carved along the flight axis so the camera always has clear air.
float de(vec3 p, out vec4 tr){
  float carve = uCarve - length(p.yz - uCam.yz);
  p += p_swim*0.03*sin(p.yzx*1.7 + uT*0.4);
  p = 1.0 - abs(mod(p + 1.0, 4.0) - 2.0);
  float df = 1.0; tr = vec4(1e3);
  for (int i = 0; i < uIters; i++) {
    p = 2.0*clamp(p, -1.0, 1.0) - p;                                 // box fold
    float r2 = dot(p, p), k = uScale/clamp(r2, uMinR2, 1.0);         // sphere fold
    p *= k; df *= abs(k);
    p += uCC;
    tr = min(tr, vec4(abs(p), r2));
  }
  vec3 q = abs(p) - vec3(uThick);
  float d = (length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0))/df/(1.0 + 0.2*p_swim);
  gCut = step(d, carve);
  return max(d, carve);
}
void camera(vec2 uv, out vec3 ro, out vec3 rd){
  float x = uTravel;
  ro = uCam + vec3(x, 0.4*uCarve*p_tunnel_curve*vec2(sin(x*0.45), cos(x*0.37)));
  uv = rot(uRot + p_twirl*0.6*length(uv))*uv;
  if (uKal > 0.5) {                                                   // kaleidoscope the view around the flight axis
    float sA = TAU/uN, a = atan(uv.y, uv.x), r = length(uv);
    a = abs(mod(a + 0.5*sA, sA) - 0.5*sA);
    uv = r*vec2(cos(a), sin(a));
  }
  vec3 fw = normalize(vec3(1.0, 0.18*p_tunnel_curve*cos(x*0.45) + 0.5*p_pan_x, -0.18*p_tunnel_curve*sin(x*0.37) + 0.5*p_pan_y));
  vec3 rt = normalize(cross(fw, vec3(0.0, 0.0, 1.0))), up = cross(rt, fw);
  rd = normalize(fw*(0.60*exp(0.8*p_zoom)) + rt*uv.x + up*uv.y);
}
""" + RM_MAIN

# ------------------------------------------------------------------ mode 2: hyperbolic tiling
HYP_FS = COMMON + """
uniform int uIters;
uniform vec4 uMob0, uMob1, uTil; uniform float uTilMix;
out vec4 f;
vec2 mobius(vec2 z, vec4 m){ vec2 a = m.xy, b = m.zw;
  return cdiv(cmul(a, z) + b, cmul(vec2(b.x, -b.y), z) + vec2(a.x, -a.y)); }
vec3 tiling(vec2 z, float P, float Q, float hpx, float dScr){
  float sp = sin(PI/P), cq = cos(PI/Q), den = sqrt(max(cq*cq - sp*sp, 1e-6));
  float cd = cq/den, cr = sp/den, sA = PI/P;
  float par = 0.0; bool ok = false;
  for (int i = 0; i < uIters; i++) {
    float a = atan(z.y, z.x), k = floor(a/sA); a -= k*sA;
    if (mod(k, 2.0) > 0.5) a = sA - a;
    par += k;
    z = length(z)*vec2(cos(a), sin(a));
    vec2 dz = z - vec2(cd, 0.0); float d2 = dot(dz, dz);
    if (d2 >= cr*cr) { ok = true; break; }
    z = vec2(cd, 0.0) + dz*(cr*cr/d2); par += 1.0;
  }
  if (!ok) return vec3(0.0);
  float r2 = dot(z, z), om = max(1.0 - r2, 1e-5);
  vec2 dz = z - vec2(cd, 0.0);
  float dE = asinh(max(dot(dz, dz) - cr*cr, 0.0)/(cr*om));           // hyperbolic distance to the polygon edge
  float dA = asinh(2.0*max(z.y, 0.0)/om);                            // ... to the centre-to-edge-midpoint mirror
  float dS = asinh(2.0*max(sin(sA)*z.x - cos(sA)*z.y, 0.0)/om);      // ... to the centre-to-vertex mirror
  float rr = sqrt(r2), dO = log((1.0 + rr)/max(1.0 - rr, 1e-5));
  float inr = log((1.0 + cd - cr)/(1.0 - cd + cr));
  float e = clamp(dE/inr, 0.0, 1.0);                                 // 0 on the edge, 1 at the polygon centre
  float rv = cd*cos(sA) - sqrt(max(cd*cd*cos(sA)*cos(sA) - 1.0, 0.0));
  float dV = 2.0*length(z - rv*vec2(cos(sA), sin(sA)))/om;
  float odd = mod(par, 2.0);
  float aa = 1.5*hpx, g = p_fractal_glow;
  float kc = 0.16*dScr + 0.02*uT;                                    // colour flows over the tiles (view space)
  // edges: crisp core + halo
  float lw = 0.022;
  float edge = smoothstep(lw + aa, lw, dE) + (0.15 + 0.75*g)*exp(-dE*9.0);
  // nested contour rings inside each polygon, travelling toward the centre
  float nr = 3.0 + floor(6.0*p_fractal_detail);
  float rp = fract(e*nr - 0.35*uT*0.25 - 0.15*sin(uT*0.21));
  float rw = 0.5*inr/nr;                                             // ring spacing in hyperbolic units
  float ring = smoothstep(0.07 + aa/(2.0*rw), 0.07, abs(rp - 0.5)*1.0)*smoothstep(0.02, 0.18, e)*smoothstep(1.0, 0.8, e);
  // triangle mirrors (fine lines) and jewels at the centre / vertices
  float spoke = smoothstep(0.010 + aa, 0.010, min(dA, dS))*0.5;
  float jewelO = smoothstep(0.16 + aa, 0.16, dO) + 0.6*g*exp(-dO*5.0);
  float jewelV = smoothstep(0.10 + aa, 0.10, dV) + 0.5*g*exp(-dV*7.0);
  float petal = odd*smoothstep(0.0, 0.5, e)*smoothstep(1.0, 0.35, e);       // alternate triangles softly filled
  float ch = chaseAmt(0.5*e + 0.14*dScr);
  float ang = dA/(dA + dS + 1e-4);
  vec3 c = vec3(0.0);
  c += irid(kc + 0.30*e, 1.0 - e)*petal*0.28*ch;
  c += pal5(kc + 0.20 + 0.35*e)*ring*(0.55 + 0.4*odd)*ch;
  c += pal5(kc + 0.55)*spoke*ch;
  c += pal5(kc)*edge*(0.35 + 0.65*ch);
  c += irid(kc + 0.5, 0.3)*jewelO*ch + pal5(kc + 0.75 + 0.1*ang)*jewelV;
  return c;
}
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  float R = 0.485*exp(1.3*p_zoom);
  vec2 zs = rot(uRot)*(uv/R) + vec2(p_pan_x, p_pan_y)*0.6;
  float r2 = dot(zs, zs);
  vec3 col = vec3(0.0);
  if (r2 < 0.9995) {
    float rr = sqrt(r2), dScr = log((1.0 + rr)/(1.0 - rr));
    float hpx = 2.0/(uRes.y*R)/(1.0 - r2);                           // pixel size in hyperbolic units
    zs = rot(p_twirl*0.9*dScr)*zs;
    zs += p_swim*0.035*(1.0 - r2)*sin(zs.yx*5.0 + uT*vec2(0.5, 0.37));
    float fade = smoothstep(0.30, 0.05, hpx);
    if (fade > 0.0) {
      if (uTilMix < 0.999) col += (1.0 - uTilMix)*tiling(mobius(zs, uMob0), uTil.x, uTil.y, hpx, dScr);
      if (uTilMix > 0.001) col += uTilMix*tiling(mobius(zs, uMob1), uTil.z, uTil.w, hpx, dScr);
      col *= fade*mix(1.0, exp(-0.35*dScr), p_depth_fog);
    }
  }
  f = finish(col);
}
"""

# ------------------------------------------------------------------ mode 3: Apollonian gasket, Droste zoom
APO_FS = COMMON + """
uniform int uIters;
uniform vec2 uA; uniform float uZoom;
out vec4 f;
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  vec2 w = uv*exp(-0.9*p_zoom) - vec2(p_pan_x, p_pan_y)*0.3;
  float r = max(length(w), 1e-5);
  vec2 lg = vec2(log(r) - uZoom, atan(w.y, w.x) + uRot);
  vec2 z = cmul(uA, lg);
  float pz = length(uA)/(r*uRes.y)*exp(-0.9*p_zoom);                 // pixel size in the z plane
  z += p_swim*0.06*sin(z.yx*PI + uT*vec2(0.4, 0.31));
  float sc = 1.0; int depth = -1; int lvl = 0; float dlvl = 0.0;
  vec2 zh = vec2(0.0);
  for (int i = 0; i < uIters; i++) {
    z = abs(mod(z + 1.0, 2.0) - 1.0);
    float r2 = dot(z, z);
    if (r2 < 1.0) {
      // inside a disc of the packing: big discs hold an inverted copy of the whole gasket (one nesting level)
      if (lvl < 1 && r2 > 0.02 && (1.0/sc) > 26.0*pz) {
        lvl++; dlvl = float(i) + 1.0; zh = z;
        z /= r2; sc /= r2; continue;
      }
      depth = i; break;
    }
    vec2 d = z - vec2(1.0); float d2 = max(dot(d, d), 1e-6);
    z = vec2(1.0) + d/d2; sc /= d2;
  }
  vec3 col = vec3(0.0);
  if (depth >= 0) {
    float rho = length(z), rpx = 1.0/(sc*pz);                        // disc radius in pixels (approx)
    float epx = (1.0 - rho)*rpx;                                     // distance to the rim in pixels
    float vis = smoothstep(0.6, 2.2, rpx);                           // sub-pixel discs fade out (no shimmer)
    float fd = float(depth);
    float k = 0.17*fd + 0.31*float(lvl) + 0.03*uT + 0.08*lg.x;
    vec3 n = vec3(z, sqrt(max(1.0 - rho*rho, 0.0)));
    float cosT = n.z;
    float rim = smoothstep(0.0, 1.2, epx)*exp(-epx*mix(0.9, 0.22, p_fractal_glow)/max(1.0, 0.06*rpx));
    float rimE = pow(rho, 6.0 + 60.0/max(rpx, 4.0));
    float glint = pow(max(dot(n, normalize(vec3(-0.45, 0.55, 0.7))), 0.0), 24.0);
    float ch = chaseAmt(0.125*fd + 0.5*float(lvl) + 0.35*rho*rho);
    col = mix(pal5(k + 0.25*rho), irid(k + 0.25*rho, cosT), 0.6)*(1.1*rimE + 0.05)*ch;
    col += pal5(k + 0.45)*rim*1.2*ch;
    col += mix(pal5(k + 0.2), vec3(1.0), 0.25)*glint*0.9*(0.3 + 0.7*ch)*smoothstep(3.0, 9.0, rpx);
    col *= vis*smoothstep(0.0, 1.0, epx);
  }
  col *= mix(1.0, smoothstep(1.1, 0.25, length(uv)), 0.6*p_depth_fog)*smoothstep(0.0, 0.05, r);
  f = finish(col);
}
"""

# ------------------------------------------------------------------ mode 4: Julia orbit-trap jewels, Koenigs zoom
JULIA_FS = COMMON + """
uniform int uIters, uPre;
uniform vec2 uC, uBeta, uLam, uTrapC; uniform float uLogLam, uArgLam, uZf, uS0, uTrapR;
out vec4 f;
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes)/uRes.y;
  vec2 p = rot(uRot)*uv - vec2(p_pan_x, p_pan_y)*0.3;
  float lr = length(p);
  p = rot(p_twirl*1.2*lr)*p;
  p += p_swim*0.02*sin(p.yx*7.0 + uT*vec2(0.5, 0.41));
  // u = linearised (Koenigs) coordinate at the repelling fixed point: the image is invariant under u -> lambda*u
  float sc = uS0*exp(-uZf*uLogLam - float(uPre)*uLogLam);
  vec2 dl = sc*(rot(uZf*uArgLam + float(uPre)*uArgLam)*p);           // delta_0 = u / lambda^n
  float dm = sc/uRes.y;                                              // |d delta / d pixel|
  bool outp = false;
  for (int i = 0; i < uPre + 28; i++) {
    if (dot(dl, dl) > 0.0036) { outp = true; break; }
    dm *= length(uLam + 2.0*dl);
    dl = cmul(uLam + dl, dl);                                        // f(beta + d) - beta, no cancellation
  }
  vec3 col = vec3(0.0);
  if (outp) {
    vec2 z = uBeta + dl;
    float stalk = 1e3, sIdx = 0.0, rem = 1.0; bool esc = false;
    vec3 L = normalize(vec3(-0.45, 0.55, 0.7));
    for (int j = 0; j < uIters; j++) {
      vec2 q = z - uTrapC;
      float d = length(q);
      if (d < uTrapR && j > 0) {
        // pearl trap: every orbit visit is a glassy jewel, composited front to back (screen-filling ones dissolve)
        float rho = d/uTrapR, rpx = uTrapR/dm;
        float a = smoothstep(0.0, 1.3, (1.0 - rho)*rpx)*smoothstep(0.5, 2.0, rpx)*smoothstep(0.40, 0.14, rpx/uRes.y);
        if (a > 0.004) {
          float fj = float(j);
          vec3 n = vec3(q/uTrapR, sqrt(max(1.0 - rho*rho, 0.0)));
          float k = 0.13*fj + 0.03*uT;
          float glint = pow(max(dot(n, L), 0.0), 18.0);
          float ch = chaseAmt(0.11*fj + 0.3*rho);
          float body = 0.03 + 0.97*pow(rho, 3.0 + 0.03*min(rpx, 200.0));
          vec3 c = irid(k + 0.3*rho, n.z)*body*ch;
          c += mix(pal5(k + 0.3), vec3(1.0), 0.4)*glint*(0.3 + 0.7*ch)*smoothstep(2.5, 8.0, rpx);
          col += rem*a*c; rem *= 1.0 - a;
          if (rem < 0.03) break;
        }
      }
      float sd = min(abs(q.x), abs(q.y))/dm;                         // cross trap, measured in pixels
      if (sd < stalk && j > 0) { stalk = sd; sIdx = float(j); }
      float r2 = dot(z, z);
      if (r2 > 64.0) { esc = true; break; }
      dm *= 2.0*sqrt(r2);
      z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + uC;
    }
    if (esc) {
      float g = p_fractal_glow;
      float line = smoothstep(1.3, 0.3, stalk) + (0.1 + 0.6*g)*exp(-stalk*0.22);
      col += rem*pal5(0.09*sIdx + 0.5 + 0.03*uT)*line*(0.25 + 0.75*chaseAmt(0.11*sIdx + 0.5));
    }
  }
  col *= mix(1.0, smoothstep(1.15, 0.3, lr), 0.6*p_depth_fog);
  f = finish(col);
}
"""

# ------------------------------------------------------------------ mode 5: 4D polytope wireframes (instanced quads)
POLY_VS = """
#version 330
in vec2 in_c; in vec4 in_a, in_b; in vec2 in_id;        // in_id = (polytope index, per-edge colour coordinate)
uniform mat4 uR4; uniform vec3 uW; uniform vec2 uRes;
uniform float uDW, uD3, uFocal, uRad, uSwim, uT, uTwirl;
uniform float p_pan_x, p_pan_y;
out vec2 v_l; out float v_len, v_bright, v_k, v_dep;
vec3 proj(vec4 p, out float dep){
  p = uR4*p;
  p.xyz += uSwim*0.05*sin(p.yzx*4.0 + uT*0.7);
  float s = 1.0/(uDW - p.w);
  vec3 q = p.xyz*s*uDW*0.62;
  float z = uD3 - q.z;
  dep = clamp(0.5 + 0.38*p.w + 0.22*q.z, 0.0, 1.0);
  vec2 xy = q.xy*uFocal/z;
  float a = uTwirl*1.5*length(xy); float c = cos(a), sn = sin(a);
  return vec3(mat2(c, -sn, sn, c)*xy + vec2(p_pan_x, p_pan_y)*0.3, 1.0/z);
}
void main(){
  int pi = int(in_id.x + 0.5);
  float w = pi == 0 ? uW.x : (pi == 1 ? uW.y : uW.z);
  if (w < 0.004) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float da, db; vec3 A = proj(in_a, da), B = proj(in_b, db);
  vec2 d = B.xy - A.xy; float len = length(d); vec2 dir = len > 1e-6 ? d/len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float dep = mix(da, db, in_c.x);
  float R = uRad*(0.45 + 0.9*dep)*(pi == 2 ? 0.62 : 1.0);
  vec2 pos = mix(A.xy, B.xy, in_c.x) + dir*(in_c.x*2.0 - 1.0)*R + nrm*in_c.y*R;
  v_l = vec2(in_c.x*len + (in_c.x*2.0 - 1.0)*R, in_c.y*R)/R; v_len = len/R;
  v_bright = w*(pi == 2 ? 0.8 : 1.0); v_dep = dep; v_k = in_id.y;
  gl_Position = vec4(pos.x*2.0*uRes.y/uRes.x, pos.y*2.0, 0.0, 1.0);
}
"""

POLY_FS = COMMON + """
in vec2 v_l; in float v_len, v_bright, v_k, v_dep;
uniform float uRad;
out vec4 f;
void main(){
  float dx = max(max(-v_l.x, v_l.x - v_len), 0.0);
  float d = length(vec2(dx, v_l.y));                                 // capsule distance, 0 on the edge, 1 at the quad rim
  if (d >= 1.0) discard;
  float rpx = uRad*(0.45 + 0.9*v_dep)*uRes.y;                        // glow radius in pixels
  float cw = clamp((0.55 + 1.1*v_dep)/rpx, 0.02, 0.6), aa = 1.0/rpx;
  float core = smoothstep(cw + aa, cw, d);
  float halo = pow(1.0 - d, 3.0)*(0.08 + 0.55*p_fractal_glow);
  float along = clamp(v_l.x/max(v_len, 1e-3), 0.0, 1.0);
  float node = exp(-min(v_l.x, v_len - v_l.x)*min(v_l.x, v_len - v_l.x)*0.6)*exp(-d*d*7.0); // vertex jewels
  float dc = 0.22 + 0.78*v_dep*v_dep;                                // depth cue
  float k = v_k + 0.45*v_dep + 0.02*uT;
  float ch = chaseAmt(v_k*2.0 + 0.35*along + 0.6*v_dep);
  vec3 c = irid(k, v_dep)*(core + halo)*(0.25 + 0.75*ch) + mix(pal5(k + 0.5), vec3(1.0), 0.25)*node*0.9;
  c *= dc*v_bright;
  f = finish(c);
}
"""

KIFS = dict(off=[0.5, 0.3, 0.3], a1=0.52, da1=0.30, a2=0.35, da2=-0.22, sc=1.4, ring=1.0, box=[0.8, 0.12, 0.12], carve=0.16)
MBOX = dict(cc=[-0.3, -0.3, -0.9], dcc=0.18, cam=[0.0, 1.0, 1.0], thick=0.8, scale=-1.7, dscale=-0.12, minr2=0.4,
            steps=[26, 40], iters=[3.6, 1.4], carve=0.3)

HIRES_DETAIL_CAP = 0.6                    # raymarch modes: effective fractal_detail limit above res 0.75

FULLSCREEN = {0: KIFS_FS, 1: MBOX_FS, 2: HYP_FS, 3: APO_FS, 4: JULIA_FS}

_GL_BLEND, _GL_DRAW_FBO = 0x0BE2, 0x8CA6
_GL_BLEND_ENUMS = (0x80C9, 0x80C8, 0x80CB, 0x80CA, 0x8009, 0x883D)   # src rgb, dst rgb, src a, dst a, eq rgb, eq a


def _gl_lib():
    try:
        lib = ctypes.windll.opengl32 if sys.platform == "win32" else ctypes.CDLL("libGL.so.1")
        lib.glIsEnabled.restype = ctypes.c_ubyte                     # GLboolean, not int: the upper bytes are garbage
        lib.glIsEnabled.argtypes = [ctypes.c_uint]
        lib.glGetIntegerv.restype = None
        lib.glGetIntegerv.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        return lib
    except Exception:
        return None


def snap_res(res):
    """fractal_res snapped to 0.05 steps inside the contract range, so a modulated fader cannot thrash the allocator."""
    res = float(res) if math.isfinite(float(res)) else 0.5
    return min(max(round(res / 0.05) * 0.05, 0.25), 1.0)


def gl_get_int(lib, enum):
    v = ctypes.c_int(0)
    lib.glGetIntegerv(enum, ctypes.byref(v))
    return v.value


def _polytope_edges():
    """Unit-circumradius tesseract, 24-cell and 600-cell: float32 rows (a.xyzw, b.xyzw, poly_id, colour_k)."""
    from itertools import permutations, product

    def edges(v, length):
        v = np.array(sorted(set(tuple(np.round(x, 9)) for x in v)))
        d = np.linalg.norm(v[:, None] - v[None], axis=-1)
        i, j = np.where(np.triu(np.abs(d - length) < 1e-5, 1))
        return v, i, j

    tess = [np.array(s) * 0.5 for s in product((-1.0, 1.0), repeat=4)]
    axes = [np.eye(4)[i] * s for i in range(4) for s in (-1.0, 1.0)]
    phi = (1 + 5 ** 0.5) / 2
    even = [p for p in permutations(range(4)) if sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0]
    v600 = list(tess) + list(axes)
    for sg in product((-1.0, 1.0), repeat=3):
        base = (0.5 * phi * sg[0], 0.5 * sg[1], 0.5 / phi * sg[2], 0.0)
        for p in even:
            x = np.zeros(4)
            for src, dst in enumerate(p):
                x[dst] = base[src]
            v600.append(x)
    rows = []
    for pid, (verts, length) in enumerate([(tess, 1.0), (tess + axes, 1.0), (v600, 1.0 / phi)]):
        v, i, j = edges(verts, length)
        for a, b in zip(v[i], v[j]):
            dirn = np.abs(b - a) / np.linalg.norm(b - a)
            k = float(np.dot(dirn, [0.13, 0.31, 0.57, 0.83]) % 1.0)   # colour class by edge direction
            rows.append(list(a) + list(b) + [float(pid), k])
    return np.array(rows, dtype="f4")


def _rot4(i, j, a):
    m = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    m[i, i], m[j, j], m[i, j], m[j, i] = c, c, -s, s
    return m


def hyperbolic_view(P, Q, dist):
    """Moebius (a, b) for z -> (a z + b)/(conj(b) z + conj(a)): translation by `dist` along the geodesic through two
    adjacent edge midpoints of the central {P,Q} polygon. Returns (a, b, period); the tiling is invariant under a
    translation by `period`, so the view is wrapped into [-period/2, period/2] and never loses precision."""
    sp, cq = math.sin(math.pi / P), math.cos(math.pi / Q)
    den = math.sqrt(cq * cq - sp * sp)
    m = (cq - sp) / den                                               # Euclidean radius of the edge midpoints
    m1, m2 = complex(m, 0.0), cmath.rect(m, TAU / P)

    def mul(x, y):
        return (x[0] * y[0] + x[1] * y[1].conjugate(), x[0] * y[1] + x[1] * y[0].conjugate())

    c1 = (1.0 + 0j, -m1)                                              # m1 -> 0
    w = (m2 - m1) / (1 - m1.conjugate() * m2)
    rr = (cmath.exp(-0.5j * cmath.phase(w)), 0j)                      # put m2 on the positive real axis
    C = mul(rr, c1)
    n = math.sqrt(abs(C[0]) ** 2 - abs(C[1]) ** 2)
    C = (C[0] / n, C[1] / n)
    Ci = (C[0].conjugate(), -C[1])
    period = 2.0 * 2.0 * math.atanh(abs(w))
    d = (dist + 0.5 * period) % period - 0.5 * period
    M = mul(Ci, mul((math.cosh(0.5 * d) + 0j, math.sinh(0.5 * d) + 0j), C))
    return M[0], M[1], period


def julia_constants(theta):
    """c on a path hugging the outside of the Mandelbrot cardioid/bulb, its repelling fixed point and multiplier."""
    th = math.pi + 0.80 * math.pi * math.sin(theta)                   # stay away from the cusp (multiplier -> 1)
    e = cmath.exp(1j * th)
    c = (0.5 * e - 0.25 * e * e) * 1.045
    beta = 0.5 * (1 + cmath.sqrt(1 - 4 * c))
    return c, beta, 2 * beta


class FractalPass:
    """Renders one of six pure-math generators into an internal RGBA8 texture (see module docstring)."""

    def __init__(self, ctx: moderngl.Context, out_size=(1280, 800)):
        self.ctx = ctx
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self._gl = _gl_lib()
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._progs, self._vaos, self._uni = {}, {}, {}
        for m, fs in FULLSCREEN.items():
            prog = ctx.program(vertex_shader=VS, fragment_shader=fs)
            self._progs[m] = prog
            self._vaos[m] = ctx.vertex_array(prog, [(self._quad, "2f", "in_pos")])
        edges = _polytope_edges()
        self.edge_count = len(edges)
        self._edge_buf = ctx.buffer(edges.tobytes())
        self._corner_buf = ctx.buffer(np.array([0, -1, 1, -1, 0, 1, 1, 1], dtype="f4").tobytes())
        prog = ctx.program(vertex_shader=POLY_VS, fragment_shader=POLY_FS)
        self._progs[5] = prog
        self._vaos[5] = ctx.vertex_array(prog, [(self._corner_buf, "2f", "in_c"),
                                                (self._edge_buf, "4f 4f 2f/i", "in_a", "in_b", "in_id")])
        for m, prog in self._progs.items():
            self._uni[m] = {n: prog[n] for n in prog if isinstance(prog[n], moderngl.Uniform)}
        self._tex = self._fbo = None
        self._size = None
        self.alloc_count = 0
        self.settle_s = 0.3                  # a new resolution must hold this long before the texture is reallocated
        self.debug_raw = False               # True: shaders output the colour before finish() (NaN tests)
        self._want, self._want_since = None, 0.0
        self._last_t = self._last_beat = None
        self._T = self._rot = self._chase = self._bars = 0.0
        self._pal = np.zeros((5, 3), dtype="f4")

    # ---- resources
    def _ensure(self, w, h):
        """Allocate on first use; afterwards only once the requested (snapped) size has been stable for settle_s."""
        if self._size == (w, h):
            self._want = None
            return
        if self._size is not None and self.settle_s > 0.0:
            now = time.perf_counter()
            if self._want != (w, h):
                self._want, self._want_since = (w, h), now
            if now - self._want_since < self.settle_s:
                return                                               # keep rendering at the current size
        self._want = None
        if self._fbo is not None:
            self._fbo.release()
            self._tex.release()
        self._tex = self.ctx.texture((w, h), 4)
        self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tex.repeat_x = self._tex.repeat_y = False
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex])
        self._size = (w, h)
        self.alloc_count += 1

    def release(self):
        for o in ([self._fbo, self._tex] + list(self._vaos.values()) + list(self._progs.values())
                  + [self._quad, self._edge_buf, self._corner_buf]):
            if o is not None:
                o.release()
        self._fbo = self._tex = None
        self._vaos, self._progs, self._uni, self._size = {}, {}, {}, None

    # ---- time integration (so fader moves change speeds, never positions)
    def _advance(self, v, clock, t):
        """t is the engine's show time: already scaled by p_speed and held during freeze, so it is used as is.
        The beat clock keeps running through a freeze and can snap on tap-tempo / resync, so beat deltas are gated
        by (1 - freeze) and by the engine's rule: only 0 <= db < 2 counts as real playback."""
        t = float(t)
        if self._last_t is None:
            dt = t if abs(t) < 1e5 else 0.0
        else:
            dt = t - self._last_t
        self._last_t = t
        if not math.isfinite(dt):
            dt = 0.0
        self._T += dt
        self._rot = (self._rot + dt * float(v.get("rotate", 0.0)) * 0.6) % TAU
        beat = clock.get("beat") if clock else None
        if beat is None:
            beat = t * float((clock or {}).get("bpm", 124.0)) / 60.0
        beat = float(beat)
        db = beat - self._last_beat if self._last_beat is not None else 0.0
        self._last_beat = beat
        if not (0.0 <= db < 2.0):                                    # NaN, backwards, or a resync jump
            db = 0.0
        db *= 1.0 - min(max(float(v.get("freeze", 0.0)), 0.0), 1.0)
        self._bars += db / 4.0
        self._chase = (self._chase + db / 4.0 * float(v.get("chase_rate", 1.0))) % 1.0

    def _set(self, m, name, value):
        u = self._uni[m].get(name)
        if u is not None:
            u.value = value

    # ---- per-mode uniforms
    def _mode_uniforms(self, m, v, travel):
        T, morph, det = self._T, float(v.get("fractal_morph", 0.5)), float(v.get("fractal_detail", 0.5))
        if m < 2 and self._size[1] > 0.75 * self.out_size[1]:        # frame-time guard: 4x the pixels, cap the march
            det = min(det, HIRES_DETAIL_CAP)
        folds = int(round(float(v.get("folds", 8.0))))
        sym = float(v.get("symmetry", 1.0))
        s = self._set
        if m == 0:
            n = max(2, int(round(2 + (folds - 2) * sym)))
            mo = (morph - 0.5) * 2.0
            a1 = 0.52 + 0.30 * mo + 0.07 * math.sin(T * 0.11)
            a2 = 0.35 - 0.22 * mo + 0.06 * math.sin(T * 0.083 + 1.3)
            s(m, "uSteps", int(round(26 + 44 * det)))
            s(m, "uIters", int(round(4 + 3 * det)))
            K = KIFS
            a1, a2 = K["a1"] + K["da1"] * mo + 0.07 * math.sin(T * 0.11), K["a2"] + K["da2"] * mo + 0.06 * math.sin(T * 0.083 + 1.3)
            o = K["off"]
            s(m, "uOff", (o[0] + 0.05 * math.sin(T * 0.07), o[1] + 0.06 * mo, o[2] + 0.04 * math.sin(T * 0.05 + 2.0)))
            s(m, "uBox", tuple(K["box"]))
            s(m, "uCarve", K["carve"])
            s(m, "uRAB", (math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2)))
            s(m, "uSc", K["sc"])
            s(m, "uN", float(n))
            s(m, "uTw", 0.5 * float(v.get("twirl", 0.0)))
            s(m, "uRing", K["ring"])
            s(m, "uTravel", travel * 0.8)
        elif m == 1:
            mo = (morph - 0.5) * 2.0
            B = MBOX
            s(m, "uSteps", int(round(B["steps"][0] + B["steps"][1] * det)))
            s(m, "uIters", int(round(B["iters"][0] + B["iters"][1] * det)))
            c = B["cc"]
            s(m, "uCC", (c[0] + 0.06 * math.sin(T * 0.07), c[1] + 0.06 * math.cos(T * 0.053), c[2] + B["dcc"] * mo))
            s(m, "uCam", tuple(B["cam"]))
            s(m, "uThick", B["thick"])
            s(m, "uScale", B["scale"] + B["dscale"] * mo)
            s(m, "uMinR2", B["minr2"])
            s(m, "uCarve", B["carve"])
            s(m, "uN", float(max(2, folds)))
            s(m, "uKal", 1.0 if sym > 0.5 else 0.0)
            s(m, "uTravel", travel * 0.6)
        elif m == 2:
            x = min(max(morph, 0.0), 1.0) * (len(TILINGS) - 1)
            i = min(int(x), len(TILINGS) - 2)
            fr = x - i
            mix = min(max((fr - 0.3) / 0.4, 0.0), 1.0)
            mix = mix * mix * (3 - 2 * mix)
            (p0, q0), (p1, q1) = TILINGS[i], TILINGS[i + 1]
            for name, (p, q) in (("uMob0", (p0, q0)), ("uMob1", (p1, q1))):
                a, b, _ = hyperbolic_view(p, q, travel * 0.7)
                s(m, name, (a.real, a.imag, b.real, b.imag))
            s(m, "uTil", (float(p0), float(q0), float(p1), float(q1)))
            s(m, "uTilMix", mix)
            s(m, "uIters", int(round(10 + 30 * det)))
        elif m == 3:
            mm = max(2, min(24, folds))
            n = int(round(float(v.get("twirl", 0.0)) * 3.0))
            g = math.gcd(abs(n), mm) if n else mm
            period = TAU / g
            s(m, "uA", (n / math.pi, -mm / math.pi))
            s(m, "uZoom", (travel * 0.3) % period)
            s(m, "uIters", int(round(8 + 22 * det)))
        elif m == 4:
            c, beta, lam = julia_constants(TAU * (morph - 0.5) * 0.5 + self._bars * TAU / 96.0 + T * 0.004)
            ll, al = math.log(abs(lam)), cmath.phase(lam)
            s0 = 2.6
            pre = int(min(40, max(3, math.ceil(math.log(s0 * 1.0 / 0.004) / ll))))
            s(m, "uC", (c.real, c.imag))
            s(m, "uBeta", (beta.real, beta.imag))
            s(m, "uLam", (lam.real, lam.imag))
            s(m, "uLogLam", ll)
            s(m, "uArgLam", al)
            s(m, "uZf", (travel * 0.30 + float(v.get("zoom", 0.0))) % 1.0)
            s(m, "uS0", s0)
            s(m, "uPre", pre)
            s(m, "uTrapC", (0.10 * math.cos(T * 0.13), 0.10 * math.sin(T * 0.17)))
            s(m, "uTrapR", 0.42)
            s(m, "uIters", int(round(14 + 36 * det)))
        else:
            w = [max(0.0, 1.0 - abs(morph - c) / 0.4) for c in (0.0, 0.5, 1.0)]
            w = [x * x * (3 - 2 * x) for x in w]
            w = [x / max(w) for x in w]                      # crossfades never dip in brightness
            R = (_rot4(0, 3, 0.31 * T * 0.5 + travel * 0.5) @ _rot4(1, 3, 0.23 * T * 0.5 + 0.7)
                 @ _rot4(2, 3, 0.17 * T * 0.5 + travel * 0.31) @ _rot4(0, 1, self._rot) @ _rot4(0, 2, 0.11 * T + 0.4)
                 @ _rot4(1, 2, 0.35))
            self._uni[m]["uR4"].write(R.T.astype("f4").tobytes())
            s(m, "uW", tuple(w))
            s(m, "uDW", 1.9)
            s(m, "uD3", 3.0)
            s(m, "uFocal", 2.0 * math.exp(0.7 * float(v.get("zoom", 0.0))))
            s(m, "uRad", 0.012 + 0.040 * float(v.get("fractal_glow", 0.5)))
            s(m, "uSwim", float(v.get("swim", 0.0)))
            s(m, "uTwirl", float(v.get("twirl", 0.0)))

    def render(self, values: dict, clock: dict, t: float, travel_z: float, palette_rgb: list) -> moderngl.Texture:
        ctx, v = self.ctx, values
        res = snap_res(v.get("fractal_res", 0.5))
        self._ensure(max(16, int(round(self.out_size[0] * res))), max(16, int(round(self.out_size[1] * res))))
        w, h = self._size                                            # may lag the request while it settles
        self._advance(v, clock, t)
        m = min(max(int(round(float(v.get("fractal_mode", 0.0)))), 0), 5)
        travel = float(travel_z) if math.isfinite(float(travel_z)) else 0.0
        uni = self._uni[m]
        for name, u in uni.items():
            if name.startswith("p_"):
                u.value = float(v.get(name[2:], 0.0))
        self._set(m, "uRes", (float(w), float(h)))
        self._set(m, "uT", self._T)
        self._set(m, "uRot", self._rot)
        self._set(m, "uChase", self._chase)
        self._set(m, "uBars", self._bars)
        self._set(m, "uRaw", 1.0 if self.debug_raw else 0.0)
        if "pal" in uni:
            pal = np.asarray(palette_rgb, dtype="f4").reshape(-1, 3)
            self._pal[:] = pal[np.arange(5) % len(pal)]
            uni["pal"].write(self._pal.tobytes())
        self._mode_uniforms(m, v, travel)

        # --- save state, draw, restore (always: a failed draw must not leave MAX blending on the compositor)
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        gl = self._gl
        blend_on, blend_state = False, None
        if gl is not None:
            blend_on = bool(gl.glIsEnabled(_GL_BLEND))
            if m == 5:
                blend_state = [gl_get_int(gl, e) for e in _GL_BLEND_ENUMS]
        try:
            self._fbo.use()
            ctx.viewport = (0, 0, w, h)
            if m == 5:
                self._fbo.clear(0.0, 0.0, 0.0, 0.0)
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE)
                ctx.blend_equation = moderngl.MAX
                self._vaos[m].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.edge_count)
            else:
                ctx.disable(moderngl.BLEND)
                self._vaos[m].render(moderngl.TRIANGLE_STRIP)
        finally:
            if m == 5:
                if blend_state is not None:
                    ctx.blend_func = tuple(blend_state[:4])
                    ctx.blend_equation = (blend_state[4], blend_state[5])
                else:
                    ctx.blend_func = moderngl.DEFAULT_BLENDING
                    ctx.blend_equation = moderngl.FUNC_ADD
            if blend_on:
                ctx.enable(moderngl.BLEND)
            else:
                ctx.disable(moderngl.BLEND)
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:                # the caller's framebuffer was released under us: fall back to the screen
                ctx.screen.use()
        return self._tex


# ====================================================================== standalone gallery / benchmark
GALLERY_PALETTES = [
    ("Aya Jewel", [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]),
    ("Cosmic Gold-Violet", [(1.00, 0.68, 0.08), (1.00, 0.30, 0.02), (0.55, 0.08, 1.00), (0.10, 0.75, 1.00), (0.85, 0.10, 0.75)]),
]
GALLERY_TIMES = [3.0, 21.0, 47.0]
GALLERY_TRAVEL = [0.0, 2.0, 4.0]


def hidden_context(size=(64, 64)):
    """Hidden GLFW window + moderngl context (never visible on the projector). Returns (ctx, window)."""
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    win = glfw.create_window(size[0], size[1], "math_layers (hidden)", None, None)
    if not win:
        raise RuntimeError("could not create a hidden GLFW window")
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    return moderngl.create_context(), win


def default_values(**over):
    import params
    v = params.defaults()
    v["l_fractal"] = 1.0
    v.update(over)
    return v


def make_clock(t, bpm=124.0):
    b = t * bpm / 60.0
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def read_texture(tex):
    """RGBA uint8 array (h, w, 4), top row first."""
    return np.frombuffer(tex.read(), dtype=np.uint8).reshape(tex.height, tex.width, 4)[::-1]


def frame_stats(rgb):
    """(fraction near-black, fraction blown to near-white, mean 0..255, fraction of lit pixels that are saturated)."""
    a = rgb[..., :3].astype(np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    lit = mx > 40
    sat = ((mx - mn) / np.maximum(mx, 1) > 0.5) & lit
    return float((mx < 8).mean()), float((mn > 235).mean()), float(a.mean()), float(sat.sum() / max(1, lit.sum()))


def render_once(ctx, mode, t, travel_z, palette, out_size=(1280, 800), **over):
    """Fresh pass rendered at time t (deterministic: a new pass integrates from 0 to t in one step)."""
    fp = FractalPass(ctx, out_size)
    try:
        return read_texture(fp.render(default_values(fractal_mode=float(mode), **over), make_clock(t), t, travel_z, palette)).copy()
    finally:
        fp.release()


def gallery(out_dir=None, modes=None, **over):
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir or HERE / "test_frames" / "math_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, _win = hidden_context()
    modes = list(range(6)) if modes is None else modes
    tw, th, left, top = 256, 160, 96, 20
    ncol = len(GALLERY_TIMES) * len(GALLERY_PALETTES) + len(GALLERY_TRAVEL)
    sheet = Image.new("RGB", (left + tw * ncol, top + th * len(modes)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for row, m in enumerate(modes):
        shots = [(f"p{pi}_t{int(t):02d}", t, 0.0, pi) for pi in range(len(GALLERY_PALETTES)) for t in GALLERY_TIMES]
        shots += [(f"travel{int(z)}", 8.0, z, 0) for z in GALLERY_TRAVEL]
        for col, (tag, t, z, pi) in enumerate(shots):
            img = render_once(ctx, m, t, z, GALLERY_PALETTES[pi][1], **over)
            blk, blown, mean, sat = frame_stats(img)
            print(f"{MODE_NAMES[m]:11s} {tag:10s} black={blk:5.1%} blown={blown:5.2%} mean={mean:5.1f} sat={sat:4.0%}")
            im = Image.fromarray(img[..., :3])
            im.save(out_dir / f"{m}_{MODE_NAMES[m]}_{tag}.png")
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (left + col * tw, top + row * th))
            if row == 0:
                dr.text((left + col * tw + 4, 4), tag, fill=(200, 200, 200))
        dr.text((4, top + row * th + th // 2 - 6), f"{m} {MODE_NAMES[m]}", fill=(200, 200, 200))
    sheet.save(out_dir / "contact_sheet.png")
    print("contact sheet:", out_dir / "contact_sheet.png")


def bench(frames=80, out_size=(1280, 800)):
    ctx, _win = hidden_context()
    pal = GALLERY_PALETTES[0][1]
    for res in (0.5, 1.0):
        fp = FractalPass(ctx, out_size)
        for m in range(6):
            v = default_values(fractal_mode=float(m), fractal_res=res)
            ms = []
            for i in range(frames + 10):
                t = i / 60.0
                ctx.finish()
                t0 = time.perf_counter()
                fp.render(v, make_clock(t), t, t * 0.5, pal)
                ctx.finish()
                if i >= 10:
                    ms.append((time.perf_counter() - t0) * 1000.0)
            ms = np.array(ms)
            print(f"res {res:.2f}  {m} {MODE_NAMES[m]:11s} median {np.median(ms):6.2f} ms   min {ms.min():6.2f}   p90 {np.percentile(ms, 90):6.2f}")
        fp.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", action="store_true", help="render every mode to test_frames/math_gallery/ (hidden window)")
    ap.add_argument("--bench", action="store_true", help="ms per mode at fractal_res 0.5 and 1.0 (hidden window)")
    ap.add_argument("--modes", type=str, default="", help="comma list of modes for --gallery (default all)")
    ap.add_argument("--out", type=str, default="", help="gallery output dir")
    args = ap.parse_args()
    if args.gallery:
        gallery(args.out or None, [int(x) for x in args.modes.split(",")] if args.modes else None)
    elif args.bench:
        bench()
    else:
        ap.print_help()
