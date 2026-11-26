"""TouchVDMX merge: effects reimplemented in THIS engine (one engine, one panel) after a browser prototype that the
project owner generated with their own Gemini session (not distributed; provenance: THIRD_PARTY_NOTICES.md 1.4).

    lp = LandPass(ctx, out_size=(1280, 800))
    tex = lp.render(frame.values, frame.clock, t, travel_z, dt, palette_rgb, bands)   # moderngl.Texture, RGBA8
    ...screen-blend tex (black = no light). The pass already applies values["l_land"] / values["l_swarm"]...
    lp.release()

What is in the texture:
  * 3D landscapes (raymarched, flown through): values["land_a"] / ["land_b"] pick two of
      0 sacred corridor   vaulted corridor with star tessellation reliefs and a luminous axis
      1 torus wormhole    tube with torus side chambers; land_style = electric chapel / floral glass / neon gears
      2 prismatic vortex  spiralling funnel tiled with animated Voronoi stained glass and speed streaks
      3 hex dome          periodic honeycomb domes (sphere -> cube -> octahedron with land_morph)
    mixed by land_xfade with eight blend modes (land_blend); deck A's picture can bend deck B's rays (land_warp).
    Deck B is only rendered while land_xfade > 0. Colours come from the show palette.
  * curl-noise particle swarm (values["l_swarm"]): 6000 filaments advected on the CPU (numpy), drawn as
    additive GL lines into a decaying trail buffer. punch = radial shockwave.
Camera (land_cam): 0 cruise, 1 orbit, 2 audio dive; cam_yaw / cam_pitch / cam_roll (fly pad, gamepad) add a look
offset, travel adds flight speed, punch widens the field of view and lurches forward (warp jump).

PreviewTap: a small JPEG of the displayed frame for the panel's monitor (only while a panel is asking for it).

Standalone:
  python fx_stack.py --gallery [OUT]   # hidden window -> test_frames/touchvdmx/*.png + contact_sheet.png
  python fx_stack.py --bench           # ms per landscape at land_res 0.5 and 1.0
"""
import io
import math
import sys
import threading
import time
from pathlib import Path

import moderngl
import numpy as np

HERE = Path(__file__).parent
TAU = 2.0 * math.pi
LAND_NAMES = ["corridor", "torus", "vortex", "hexdome"]
BLEND_NAMES = ["mix", "add", "screen", "difference", "multiply", "exclusion", "divide", "luma mask"]
N_PARTICLES = 6000
# Long shows: nothing may pop when a phase wraps. Every phase is integrated on the CPU in float64 and wrapped at a
# period its own use is periodic in; the shaders only ever see small numbers.
T_WRAP = TAU * 1000.0            # uT: every sin / rot2 frequency of uT is a multiple of 0.001 rad/s
Z_WRAP = TAU * 100.0             # camera z (628): sin terms use multiples of 0.01, repeats use periods TAU / n.
#                                  Twist, tile rows and the colour drift along z are camera-relative + own phases.
V_WRAP = (32.0, 32.0, 144.0)     # tile-row phase of corridor / torus / vortex: row ids are hashed modulo these

VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HEADER = """
#version 330
in vec2 v_uv; out vec4 fragColor;
uniform float uT, uFov, uFolds, uTwist, uCurve, uMorph, uMirror, uStyle;
uniform float uLow, uMid, uHigh, uWarp, uRaw;
uniform float uTw, uV, uPz, uPal, uSk, uSh;   // CPU-wrapped phases: twist, tile rows, colour along z, palette time,
                                               // streaks, hex drift
uniform vec2 uRes, uWs;                        // uWs: used part of the (max-size) warp texture
uniform vec3 uCam;
uniform mat3 uCamMat;
uniform sampler2D texWarp;
uniform vec3 pal[5];
const float TAU = 6.2831853, PI = 3.14159265;

mat2 rot2(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
// hash12: "Hash without Sine", Copyright (c) 2014 David Hoskins, MIT License (full text: THIRD_PARTY_NOTICES.md)
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y)*p3.z); }
vec3 palC(float t){ t = fract(t)*5.0; int i = int(t); float k = fract(t); k = k*k*(3.0 - 2.0*k);
  return mix(pal[i % 5], pal[(i + 1) % 5], k); }
vec2 polarFold(vec2 p, float n){
  float a = atan(p.y, p.x), r = length(p), sec = TAU/max(n, 2.0);
  a = abs(mod(a, sec) - sec*0.5);
  return vec2(cos(a), sin(a))*r; }
vec3 rayDir(vec2 uv){
  vec2 p = (uv - 0.5)*vec2(uRes.x/uRes.y, 1.0);
  if (uWarp > 0.001) { vec3 w = texture(texWarp, uv*uWs).rgb; p += (w.rg - 0.5)*uWarp*0.35; }
  return normalize(uCamMat*vec3(p, 1.0/max(tan(radians(uFov)*0.5), 0.25))); }
vec4 finish(vec3 c){
  if (uRaw > 0.5) return vec4(c, 1.0);
  if (any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
  return vec4(clamp(c, 0.0, 1.6)/1.6, 1.0); }
"""

# ---- 0: sacred corridor
CORRIDOR_FS = HEADER + """
float starTile(vec2 uv, float n){
  vec2 g = fract(uv) - 0.5;
  float r = length(g), a = atan(g.y, g.x), sec = TAU/max(n, 3.0);
  float fa = abs(mod(a, sec) - sec*0.5);
  vec2 q = vec2(cos(fa), sin(fa))*r;
  return min(min(abs(q.x*1.65 + q.y*0.95 - 0.36), abs(r - 0.16)), abs(r - 0.44)); }

void main(){
  vec3 ro = uCam, rd = rayDir(v_uv), col = vec3(0.0);
  float t = 0.0, glow = 0.0;
  for (int i = 0; i < 48; i++) {
    vec3 p = ro + rd*t;
    p.xy *= rot2((p.z - uCam.z)*(0.06 + uTwist*0.22) + uTw);
    p.xy += vec2(sin(p.z*0.22 + uT*0.4), cos(p.z*0.19 - uT*0.3))*uCurve*0.35;
    float rad = 2.45 + 0.35*sin(p.z*0.6 + uMorph*PI);
    float dWall = min(rad - length(p.xy), 1.65 - abs(p.y));
    float zr = mod(p.z, 1.7951958) - 0.8975979, a = atan(p.y, p.x);          // period TAU/3.5
    float d = dWall - 0.08*sin(a*uFolds + zr*3.5)*cos(p.z*1.4);
    glow += 0.012/(0.03 + abs(length(p.xy) - (0.14 + uLow*0.12)))*exp(-t*0.08);
    if (d < 0.008 || t > 34.0) {
      float ang = atan(p.y, p.x);
      vec2 tuv = vec2(ang*(uFolds/PI)*floor(1.5 + uMorph*2.0), (p.z - uCam.z)*0.85*(1.2 + uMorph*1.4) + uV);
      if (uMirror > 0.5) tuv = abs(fract(tuv*0.5)*2.0 - 1.0)*1.5;
      float st = starTile(tuv, uFolds);
      float lines = smoothstep(0.045, 0.008, st), facets = smoothstep(0.22, 0.0, st);
      vec3 pc = palC((p.z - uCam.z)*0.02 + uPz + mod(floor(tuv.y), 16.0)*0.125 + st*0.9 + uPal*2.0);
      col = mix(pc*facets*0.55, mix(pc*1.5, vec3(1.0, 0.86, 0.5), 0.4), lines);
      col *= (0.85 + uMid*0.65)*exp(-t*0.06);
      break;
    }
    t += max(abs(d)*0.68, 0.03);
  }
  col += palC(uPal*3.0 + 0.4)*glow*(0.20 + uLow*0.45);
  fragColor = finish(col);
}
"""

# ---- 1: torus wormhole (tube + torus side chambers)
TORUS_FS = HEADER + """
float torusPattern(vec2 uv, float mode, float folds){
  vec2 g = fract(uv) - 0.5;
  float r = length(g), a = atan(g.y, g.x);
  if (mode < 0.5) {                                   // electric chapel: wireframe lattice + arches
    vec2 f = abs(fract(uv*2.0) - 0.5);
    return min(min(min(f.x, f.y), abs(f.x - f.y)*0.8), abs(r - 0.38));
  } else if (mode < 1.5) {                            // floral stained glass: lotus petals
    return min(abs(r - (0.28 + 0.16*cos(a*max(folds, 5.0)))), abs(r - 0.10));
  }
  float spin = a + uT*(mod(floor(uv.x) + floor(uv.y), 2.0) > 0.5 ? 1.2 : -1.2);    // neon gears
  float teeth = 0.32 + 0.07*smoothstep(-0.2, 0.2, sin(spin*max(folds, 6.0)));
  return min(abs(r - teeth), abs(r - 0.14)); }

void main(){
  vec3 ro = uCam, rd = rayDir(v_uv), col = vec3(0.0);
  float t = 0.0, glow = 0.0;
  float R = 2.6 + uCurve*1.1, rTube = 1.45 + 0.25*sin(uT*0.5 + uMorph*PI);
  for (int i = 0; i < 46; i++) {
    vec3 p = ro + rd*t, q = p;
    q.xy *= rot2((p.z - uCam.z)*(0.08 + uTwist*0.25) + uTw + uT*0.12);
    q.z = mod(q.z + PI, TAU) - PI;
    float dTor = rTube*(0.55 + 0.45*uMorph) - length(vec2(length(q.xz) - R, q.y));
    vec2 ctr = vec2(sin(p.z*0.35), cos(p.z*0.35))*(0.25 + uCurve*0.45);
    float dTube = (1.95 + 0.35*sin(p.z*0.5)) - length(q.xy - ctr);
    float d = max(dTube, dTor);                        // inside = tube OR torus chamber
    glow += 0.01/(0.05 + abs(length(q.xy) - 0.35))*exp(-t*0.1);
    if (d < 0.01 || t > 30.0) {
      vec2 uv = vec2(atan(q.y, q.x)/TAU*max(uFolds, 4.0), (p.z - uCam.z)*0.75 + uV);
      if (uMirror > 0.5) uv = polarFold(fract(uv) - 0.5, uFolds*0.5) + floor(uv);
      float pat = torusPattern(uv, uStyle, uFolds);
      float lead = smoothstep(0.025, 0.06, pat), edge = smoothstep(0.05, 0.005, pat);
      vec3 glass = palC(floor(uv.x)*0.13 + mod(floor(uv.y), 16.0)*0.1875 + pat*0.8 + uPal*2.0);
      if (uStyle < 0.5) col = glass*0.22 + palC(0.1 + uPal)*edge*1.45 + palC(0.5 + uPal)*smoothstep(0.12, 0.0, pat)*0.5;
      else {                                             // backlit cells, about a third left dark (black = no light)
        float lit = step(0.34, hash12(vec2(floor(uv.x), mod(floor(uv.y), 32.0))));
        col = glass*lead*(0.12 + 0.78*lit)*smoothstep(0.30, 0.0, pat - 0.06) + vec3(1.0, 0.9, 0.7)*edge*0.5;
      }
      col *= (0.85 + uLow*0.55)*exp(-t*0.07);
      break;
    }
    t += max(abs(d)*0.65, 0.035);
  }
  col += palC(uPal*3.0 + 0.6)*glow*(0.16 + uHigh*0.5);
  fragColor = finish(col);
}
"""

# ---- 2: prismatic stained-glass vortex
VORTEX_FS = HEADER + """
// hash2 and the animated-feature-point cell search in voronoiGlass follow Inigo Quilez's Voronoi shaders,
// Copyright (c) 2013 Inigo Quilez, MIT License (full text: THIRD_PARTY_NOTICES.md)
vec2 hash2(vec2 p){ return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3))))*43758.5453); }
vec3 voronoiGlass(vec2 x, float per){               // per: cells around the funnel, so the seam closes
  vec2 n = floor(x), f = fract(x), cid = vec2(0.0);
  float md = 8.0, md2 = 8.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 g = vec2(float(i), float(j)), id = vec2(mod(n.x + g.x, per), mod(n.y + g.y, 144.0));
    vec2 o = 0.5 + 0.42*sin(uT*0.8 + TAU*hash2(id));
    vec2 r = g + o - f;
    float d = dot(r, r);
    if (d < md) { md2 = md; md = d; cid = id; } else if (d < md2) md2 = d;
  }
  return vec3(sqrt(md), sqrt(md2) - sqrt(md), hash2(cid).x); }

void main(){
  vec3 ro = uCam, rd = rayDir(v_uv), col = vec3(0.0);
  float t = 0.0, glow = 0.0, per = floor(max(uFolds, 6.0) + 0.5);
  for (int i = 0; i < 42; i++) {
    vec3 p = ro + rd*t;
    p.xy *= rot2((p.z - uCam.z)*(0.14 + uTwist*0.42) + uTw + uT*0.35);
    float d = (1.9 + 0.45*sin(p.z*0.45 + uCurve*2.0)) - length(p.xy);
    glow += 0.015/(0.04 + length(p.xy))*exp(-t*0.09);
    if (d < 0.012 || t > 28.0) {
      float ang = atan(p.y, p.x);
      vec2 vuv = vec2((ang/TAU + 0.5)*per, (p.z - uCam.z)*(0.9 + uMorph*0.9) + uV);
      if (uMirror > 0.5) vuv.x = abs(vuv.x - per*0.5);
      float disp = 0.03 + uHigh*0.06;
      vec3 vR = voronoiGlass(vuv + vec2(disp, 0.0), per), vG = voronoiGlass(vuv, per), vB = voronoiGlass(vuv - vec2(disp, 0.0), per);
      float lead = smoothstep(0.03, 0.08, vG.y), k = (p.z - uCam.z)*0.02 + uPz + uPal*3.0;
      vec3 prism = vec3(palC(vR.z + k).r, palC(vG.z + k).g, palC(vB.z + k).b);
      float streaks = pow(0.5 + 0.5*sin(ang*per*3.0 + p.z*4.0 - uSk*TAU), 6.0);
      col = prism*lead*(0.14 + 0.95*step(0.3, vG.z))*(1.0 - 0.55*smoothstep(0.1, 0.6, vG.x)) + vec3(1.0, 0.95, 0.85)*streaks*(0.05 + uLow*0.3);
      col *= exp(-t*0.06);
      break;
    }
    t += max(abs(d)*0.65, 0.04);
  }
  col += palC(uPal*4.0 + 0.2)*glow*(0.22 + uLow*0.55);
  fragColor = finish(col);
}
"""

# ---- 3: hexagonal honeycomb domes
HEX_FS = HEADER + """
// Pointy-top hexagon lattice, apothem 0.5 (neighbouring centres 1 apart; rows sqrt(3)/2 apart, odd rows shifted by 0.5).
// Written from the lattice geometry: p -> fractional axial coordinates (a, b) with p = a*(1, 0) + b*(1/2, sqrt(3)/2),
// cube-coordinate rounding (a + b + c = 0: round all three, then re-derive the one with the largest rounding error),
// back to the cell centre. Edge distance = apothem minus the largest projection of the local offset onto the three
// edge normals (0, 60, 120 degrees). Returns (distance to the cell edge, distance to the centre, centre xy).
vec4 hexGrid(vec2 p){
  const float S3 = 1.7320508;
  float fb = p.y*(2.0/S3), fa = p.x - 0.5*fb, fc = -fa - fb;
  vec3 c = vec3(floor(fa + 0.5), floor(fb + 0.5), floor(fc + 0.5));
  vec3 err = abs(c - vec3(fa, fb, fc));
  if (err.x > err.y && err.x > err.z) c.x = -c.y - c.z;
  else if (err.y > err.z) c.y = -c.x - c.z;
  vec2 ctr = vec2(c.x + 0.5*c.y, 0.5*S3*c.y);
  vec2 lo = p - ctr, al = abs(lo);
  float edge = 0.5 - max(al.x, 0.5*al.x + 0.5*S3*al.y);
  return vec4(edge, length(lo), ctr); }

void main(){
  vec3 ro = uCam, rd = rayDir(v_uv), col = vec3(0.0);
  float t = 0.0, pulse = 0.0;
  float m = clamp(uMorph, 0.0, 1.0), nx = floor(max(uFolds*2.0, 12.0) + 0.5), ny = 1.7320508*floor(max(uFolds*0.8, 5.0) + 0.5);
  for (int i = 0; i < 46; i++) {
    vec3 p = ro + rd*t;
    vec3 q = vec3(mod(p.xy + 3.0, 6.0) - 3.0, mod(p.z + PI, TAU) - PI);
    float dPort = 1.3 - length(q.xy);                   // porthole along the flight axis: the camera never clips a shell
    q.xy *= rot2(uTwist*0.5 + uT*0.1);
    q.yz *= rot2(uCurve*0.35);
    vec3 aq = abs(q);
    float dS = 2.65 - length(q), dC = 2.25 - max(aq.x, max(aq.y, aq.z)), dO = 2.85 - (aq.x + aq.y + aq.z)*0.72;
    float dDome = m < 0.5 ? mix(dS, dC, m*2.0) : mix(dC, dO, (m - 0.5)*2.0);
    float dCore = length(q) - (0.30 + 0.10*sin(uT*2.0) + uLow*0.18);
    pulse += 0.012/(0.04 + abs(dCore))*exp(-t*0.12);
    float d = min(max(dDome, dPort), dCore);
    if (d < 0.01 || t > 26.0) {
      vec3 n = normalize(q);
      vec2 suv = vec2((atan(n.y, n.x)/TAU + 0.5)*nx, acos(clamp(n.z, -1.0, 1.0))/PI*ny);
      if (uMirror > 0.5) suv = abs(suv - vec2(nx, ny)*0.5);
      vec4 hx = hexGrid(suv + vec2(uSh, 0.0));
      float rim = smoothstep(0.08, 0.015, hx.x), cell = smoothstep(0.02, 0.14, hx.x), star = smoothstep(0.18, 0.04, hx.y);
      vec2 cid = vec2(mod(hx.z - uSh, nx), hx.w);
      float ch = hash12(floor(cid*2.0 + 0.5));
      vec3 cc = palC(ch*0.85 + hx.y*0.5 + uPal*2.0);
      col = cc*cell*(0.35 + star*1.1)*step(0.45, ch + star*0.6) + mix(cc, vec3(1.0, 0.88, 0.45), 0.5)*rim*(0.55 + uHigh*0.7);
      col *= (0.85 + uMid*0.5)*exp(-t*0.07);
      break;
    }
    t += max(abs(d)*0.68, 0.03);
  }
  col += palC(uPal*3.0 + 0.8)*pulse*(0.16 + uLow*0.45);
  fragColor = finish(col);
}
"""

LAND_FS = [CORRIDOR_FS, TORUS_FS, VORTEX_FS, HEX_FS]

# ---- environment skins (materials.py). The ORIGINAL programs above are what runs while every slot is procedural
# (pixel-identical, zero extra cost). A world with a skinned slot draws with a second program: the same source with
# the materials chunk in the header and ONE block pasted next to the line that sets the surface colour.
try:
    import materials as _mat
except Exception:                                  # the module is optional: without it the decks are as they were
    _mat = None

# The skin runs AFTER the march loop: inside the loop the hook only RECORDS the hit (mapping inputs, normal, line
# mask, shade, the world's own depth scale k); the fetch, the light, the mix and the overlay layers run once, after
# the loop. Code inside the loop body costs registers for all ~46 march steps (occupancy), code after it does not.
# f = one pixel's footprint on the surface (grown with distance: far detail is filtered a little harder than
# strictly needed, which is what stops far-wall sparkle), zr = distance along the travel axis
_SK_PX = "float f = t*max(tan(radians(uFov)*0.5), 0.25)/uRes.y*(1.0 + 0.05*t), zr = p.z - uCam.z;"
SKIN_DECL_ANCHOR = "  vec3 ro = uCam, rd = rayDir(v_uv), col = vec3(0.0);\n"
SKIN_DECL = ("  int hkS = 0; float hkT = -1.0, hkI = 0.0, hkF = 0.0, hkLn = 0.0, hkSh = 1.0, hkLm = 1.0, hkK = 1.0;\n"
             "  vec3 hkN = vec3(0.0, 0.0, 1.0), hkTN = hkN, hkP = vec3(0.0); vec2 hkUV = vec2(0.0), hkGX = hkUV, hkGY = hkUV;\n")
SKIN_POST = """
  if (hkT >= 0.0) {                                    // [skins] one hit recorded in the loop: skin it once, here
    float mp = skB[hkS].x, lit = skLight(hkN, rd, hkT, hkI)*hkLm; vec3 S;
    if (mp > 1.5) S = skTap(hkS, (v_uv - 0.5)*vec2(uRes.x/uRes.y, 1.0) + hkN.xy*0.35, vec2(1.0/uRes.y, 0.0), vec2(0.0, 1.0/uRes.y));
    else if (mp > 0.5) S = skTri(hkS, hkP, hkTN, hkF);
    else S = skTap(hkS, hkUV, hkGX, hkGY);
    col = skMix(hkS, col/hkK, S, hkLn, hkSh*lit)*hkK;
  }"""
SKIN_HOOKS = [
    # corridor: slot 0 = the vault, slot 1 = floor + ceiling
    ("      col = mix(pc*facets*0.55, mix(pc*1.5, vec3(1.0, 0.86, 0.5), 0.4), lines);\n", """
      { bool fl = (1.65 - abs(p.y)) < (rad - length(p.xy)); int s = fl ? 1 : 0;
        if (skA[s].w > 0.0005) {
          """ + _SK_PX + """
          hkS = s; hkT = t; hkI = float(i); hkF = f; hkLn = lines; hkSh = 0.66 + 0.34*facets; hkK = (0.85 + uMid*0.65)*exp(-t*0.06);
          hkN = fl ? vec3(0.0, -sign(p.y), 0.0) : vec3(-normalize(p.xy), 0.0); hkTN = hkN; hkP = vec3(p.xy, zr);
          if (fl) { hkUV = vec2(p.x + step(0.0, p.y)*1.7, zr); hkGX = vec2(f, 0.0); hkGY = vec2(0.0, f*clamp(t/1.65, 1.0, 64.0)); }
          else { hkUV = vec2(ang/TAU, zr); hkGX = vec2(f/(TAU*rad), 0.0); hkGY = vec2(0.0, f*clamp(t/rad, 1.0, 64.0)); }
        } }
""", "after"),
    # torus: slot 0 = the tube, slot 1 = the torus side chambers
    ("      col *= (0.85 + uLow*0.55)*exp(-t*0.07);\n", """
      { int s = dTor > dTube ? 1 : 0;
        if (skA[s].w > 0.0005) {
          """ + _SK_PX + """
          hkS = s; hkT = t; hkI = float(i); hkF = f; hkLn = edge; hkSh = 0.60 + 0.40*lead; hkK = (0.85 + uLow*0.55)*exp(-t*0.07);
          hkN = vec3(-normalize(q.xy - (s == 0 ? ctr : vec2(0.0))), 0.0); hkTN = hkN; hkP = vec3(q.xy, zr);
          hkUV = vec2(atan(q.y, q.x)/TAU, zr); hkGX = vec2(f/(TAU*1.9), 0.0); hkGY = vec2(0.0, f*clamp(t/1.9, 1.0, 64.0));
        } }
""", "before"),
    # vortex: the stained-glass panes are the "lines" (the lead between them stays dark)
    ("      col *= exp(-t*0.06);\n", """
      if (skA[0].w > 0.0005) {
        """ + _SK_PX + """
        hkS = 0; hkT = t; hkI = float(i); hkF = f; hkLn = 1.0 - lead; hkK = exp(-t*0.06);   // lines = the lead network
        hkSh = (0.40 + 0.60*lead)*(1.0 - 0.35*smoothstep(0.1, 0.6, vG.x));
        hkN = vec3(-normalize(p.xy), 0.0); hkTN = hkN; hkP = vec3(p.xy, zr);
        hkUV = vec2(ang/TAU, zr); hkGX = vec2(f/(TAU*1.9), 0.0); hkGY = vec2(0.0, f*clamp(t/1.9, 1.0, 64.0));
      }
""", "before"),
    # hex domes: ONE image per honeycomb cell (cell-local coordinates: no polar stretch, the rims hide the joins);
    # the domes are fixed in the world, so there is no travel follow
    ("      col *= (0.85 + uMid*0.5)*exp(-t*0.07);\n", """
      if (skA[0].w > 0.0005) {
        float f = t*max(tan(radians(uFov)*0.5), 0.25)/uRes.y*(1.0 + 0.05*t);
        hkS = 0; hkT = t; hkI = float(i); hkF = f; hkLn = rim; hkSh = 0.50 + 0.50*cell; hkK = (0.85 + uMid*0.5)*exp(-t*0.07);
        hkN = -n*sign(dot(n, -rd) + 1e-4); hkTN = n; hkP = q;
        hkLm = smoothstep(0.0, 0.15, abs(dot(n, rd)));   // silhouettes fall to black
        hkUV = suv + vec2(uSh, 0.0) - hx.zw + 0.5 + vec2(ch, fract(ch*7.31))*4.0; hkGX = vec2(f*nx/(TAU*2.4), 0.0); hkGY = vec2(0.0, f*ny/(PI*2.65));
      }
""", "before"),
]
GLOW_ANCHOR = "\n  col += palC("          # each world adds its own coloured haze once, after the march


def skinned_source(mode):
    """LAND_FS[mode] with the skin chunk, the declarations, the in-loop recorder and the post-loop skin block.
    Raises if an anchor moved (a test pins this)."""
    anchor, add, where = SKIN_HOOKS[mode]
    fs = LAND_FS[mode]
    if fs.count(anchor) != 1 or fs.count(HEADER) != 1:
        raise ValueError(f"skin hook anchor not found in landscape {mode}")
    if fs.count(GLOW_ANCHOR) != 1 or fs.count(SKIN_DECL_ANCHOR) != 1:
        raise ValueError(f"glow / declaration anchor not found in landscape {mode}")
    fs = fs.replace(HEADER, HEADER + _mat.GLSL)
    fs = fs.replace(SKIN_DECL_ANCHOR, SKIN_DECL_ANCHOR + SKIN_DECL)
    fs = fs.replace(GLOW_ANCHOR, SKIN_POST + "\n  col += skDim*palC(")   # a skinned surface dims the world's coloured haze
    return fs.replace(anchor, add + anchor if where == "before" else anchor + add)

TRAIL_FS = """
#version 330
in vec2 v_uv; out vec4 fragColor;
uniform sampler2D texPrev;
uniform float uDecay;
void main(){                       // the knee takes the tail to true black (a pure multiply never gets there)
  vec3 c = texture(texPrev, (v_uv - 0.5)*0.996 + 0.5).rgb;
  fragColor = vec4(clamp(c*uDecay - 1.5/255.0, 0.0, 1.5), 1.0); }
"""

SWARM_VS = """
#version 330
in vec3 in_pos; in float in_hue;
uniform vec2 uRes;
uniform float uFov, uTint, uHigh, uHueT, uGain;
uniform mat3 uCamMat;
uniform vec3 pal[5];
out vec4 v_col;
void main(){
  vec3 p = transpose(uCamMat)*in_pos;
  float z = p.z + 3.2, f = 1.0/max(tan(radians(uFov)*0.5), 0.25);
  gl_Position = vec4(vec2(p.x/(uRes.x/uRes.y), p.y)*(f/max(z, 0.25)), 0.0, 1.0);
  float fade = smoothstep(7.5, 1.2, z)*smoothstep(0.15, 0.9, z);
  float t = fract(in_hue + uHueT)*5.0; int i = int(t); float k = fract(t);
  vec3 c = mix(pal[i % 5], pal[(i + 1) % 5], k*k*(3.0 - 2.0*k));
  float g = fract(in_hue*3.0);
  vec3 gold = mix(mix(vec3(0.98, 0.72, 0.22), vec3(1.0, 0.95, 0.78), g), vec3(0.95, 0.42, 0.18), smoothstep(0.7, 1.0, g));
  v_col = vec4(mix(c, gold, uTint)*(0.9 + uHigh*0.8), fade*0.45*uGain);
}
"""

SWARM_FS = """
#version 330
in vec4 v_col; out vec4 fragColor;
void main(){ fragColor = v_col; }
"""

COMP_FS = """
#version 330
in vec2 v_uv; out vec4 fragColor;
uniform sampler2D texA, texB, texS;
uniform float uX, uLand, uSwarm, uDisp, uHasB, uRaw;
uniform vec2 uSc, uLim;                                // used part of the deck targets, and its last texel centre
uniform int uBlend;
vec3 tap(sampler2D tx, vec2 uv){ return texture(tx, min(max(uv, 0.0)*uSc, uLim)).rgb; }
vec3 deck(sampler2D tx){                               // radial prism split
  if (uDisp < 0.004) return tap(tx, v_uv)*1.6;
  vec2 d = (v_uv - 0.5)*uDisp*0.045;
  return vec3(tap(tx, v_uv + d).r, tap(tx, v_uv).g, tap(tx, v_uv - d).b)*1.6; }
void main(){
  vec3 land = vec3(0.0);
  if (uLand > 0.0) {
    vec3 a = deck(texA);
    if (uHasB < 0.5) land = a;
    else {
      vec3 b = deck(texB), mixed;
      if (uBlend == 0) land = mix(a, b, uX);
      else {
        if (uBlend == 1) mixed = min(a + b, 1.5);
        else if (uBlend == 2) mixed = 1.0 - (1.0 - min(a, 1.0))*(1.0 - min(b, 1.0));
        else if (uBlend == 3) mixed = abs(a - b);
        else if (uBlend == 4) mixed = a*b*1.4;
        else if (uBlend == 5) mixed = a + b - 2.0*min(a, 1.0)*min(b, 1.0);
        else if (uBlend == 6) mixed = min(a/max(b + 0.15, 0.15)*0.45, 1.5);
        else mixed = mix(a, b, smoothstep(0.2, 0.8, dot(a, vec3(0.299, 0.587, 0.114))));
        land = uX < 0.5 ? mix(a, mixed, smoothstep(0.0, 0.5, uX)) : mix(mixed, b, smoothstep(0.5, 1.0, uX));
      }
    }
    land = land/(1.0 + land*0.25)*uLand;                // filmic highlight roll-off
  }
  vec3 col = land;
  if (uSwarm > 0.0) {                                    // soft shoulder + screen: dense filaments never clip to white
    vec3 sw = (1.0 - exp(-texture(texS, v_uv).rgb*1.5))*0.88*uSwarm;
    col = col + sw*(1.0 - clamp(col, 0.0, 1.0));
  }
  if (uRaw < 0.5 && (any(isnan(col)) || any(isinf(col)))) col = vec3(0.0);
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

BLIT_FS = """
#version 330
in vec2 v_uv; out vec4 fragColor;
uniform sampler2D tex;
void main(){ fragColor = vec4(clamp(texture(tex, v_uv).rgb, 0.0, 1.0), 1.0); }
"""


def _f(v, name, default=0.0):
    try:
        x = float(v.get(name, default))
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _idx(v, name, hi):
    return min(max(int(round(_f(v, name))), 0), hi)


def snap_res(res):
    """land_res snapped to 0.05 steps inside the contract range (a modulated fader must not thrash the allocator)."""
    res = float(res) if isinstance(res, (int, float)) and math.isfinite(res) else 0.5
    return min(max(round(res / 0.05) * 0.05, 0.25), 1.0)


def camera_matrix(yaw, pitch, roll):
    """Column-major 3x3 (right, up, forward), the layout GLSL mat3 expects."""
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([cy * cr + sy * sp * sr, cp * sr, -sy * cr + cy * sp * sr,
                     -cy * sr + sy * sp * cr, cp * cr, sy * sr + cy * sp * cr,
                     sy * cp, -sp, cy * cp], dtype="f4")


class Swarm:
    """CPU side of the curl-noise swarm: positions, ages, and the prev -> current line segments."""

    def __init__(self, n=N_PARTICLES, seed=7):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((n, 3), dtype="f4")
        self.hue = self.rng.random(n, dtype="f4")
        self.age = np.zeros(n, dtype="f4")
        self.verts = np.zeros((n, 2, 4), dtype="f4")
        self.burst = 0.0
        self._spawn(np.ones(n, dtype=bool))
        self.verts[:, 0, :3] = self.verts[:, 1, :3] = self.pos
        self.verts[:, :, 3] = self.hue[:, None]

    def _spawn(self, mask):
        k = int(mask.sum())
        if not k:
            return
        r = np.cbrt(self.rng.random(k, dtype="f4")) * 2.6
        th = self.rng.random(k, dtype="f4") * TAU
        ph = np.arccos(2.0 * self.rng.random(k, dtype="f4") - 1.0)
        self.pos[mask] = np.stack([r * np.sin(ph) * np.cos(th), r * np.sin(ph) * np.sin(th), r * np.cos(ph)], axis=1)
        self.age[mask] = self.rng.random(k, dtype="f4") * 2.3 + 0.7          # seconds

    def step(self, dt, t, turb, vort, kick):
        """Advance by dt seconds. turb / vort are 0..1 faders, kick is the bass level."""
        dt = min(max(float(dt), 0.0), 0.05)
        turb, vort = 0.1 + 2.4 * turb, 2.0 * vort
        speed = (0.65 + turb * 0.55 + kick * 0.9 + self.burst * 2.4) * dt
        freq, tt = 0.85 + turb * 0.45, float(t) * 0.65
        ph = [np.float32((tt * k) % TAU) for k in (1.0, 0.8, 0.9, 0.7, 0.75)]     # float64 here: no pop at any wrap of t
        p = self.pos
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        prev = p.copy()
        vel = np.empty_like(p)
        vel[:, 0] = np.sin(y * freq + ph[0]) - np.cos(z * freq - ph[1]) - y * vort * 0.55 - x * 0.14
        vel[:, 1] = np.sin(z * freq + ph[2]) - np.cos(x * freq + ph[3]) + x * vort * 0.55 - y * 0.14
        vel[:, 2] = np.sin(x * freq - ph[4]) - np.cos(y * freq + ph[0]) - z * 0.14
        if self.burst > 0.02:
            vel += p / (np.linalg.norm(p, axis=1, keepdims=True) + 0.25) * (self.burst * 2.2)
        p += vel * np.float32(speed)
        self.age -= np.float32(dt)
        dead = (self.age <= 0.0) | ((p * p).sum(axis=1) > 14.0) | ~np.isfinite(p).all(axis=1)
        self._spawn(dead)
        prev[dead] = p[dead]                                               # a respawn draws no streak
        self.verts[:, 0, :3] = prev
        self.verts[:, 1, :3] = p
        self.burst *= 0.90 ** (dt * 60.0)
        return self.verts


class LandPass:
    """Raymarched landscape A/B deck + particle swarm, rendered into one RGBA8 texture (see module docstring)."""

    def __init__(self, ctx: moderngl.Context, out_size=(1280, 800), seed=7):
        self.ctx = ctx
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._progs = [ctx.program(vertex_shader=VS, fragment_shader=fs) for fs in LAND_FS]
        self._vaos = [ctx.vertex_array(p, [(self._quad, "2f", "in_pos")]) for p in self._progs]
        # environment skins are INACTIVE by default: nothing is compiled or allocated for them here. Setting `skins`
        # to a materials.SkinEngine (the engine does that when the performer switches skins on) compiles the four
        # skinned programs, once; a failed compile is remembered and the decks stay procedural.
        self._skins = None
        self.skinned_decks = 0               # decks drawn with a skinned program by the last render()
        self._sprogs, self._svaos = [None] * len(LAND_FS), [None] * len(LAND_FS)
        self.skin_programs = None            # None = not compiled yet, True = ready, False = failed (not retried)
        self._trail = ctx.program(vertex_shader=VS, fragment_shader=TRAIL_FS)
        self._trail_vao = ctx.vertex_array(self._trail, [(self._quad, "2f", "in_pos")])
        self._comp = ctx.program(vertex_shader=VS, fragment_shader=COMP_FS)
        self._comp_vao = ctx.vertex_array(self._comp, [(self._quad, "2f", "in_pos")])
        self.swarm = Swarm(seed=seed)
        self._sprog = ctx.program(vertex_shader=SWARM_VS, fragment_shader=SWARM_FS)
        self._sbuf = ctx.buffer(self.swarm.verts.tobytes(), dynamic=True)
        self._svao = ctx.vertex_array(self._sprog, [(self._sbuf, "3f 1f", "in_pos", "in_hue")])
        self._tex, self._fbo = {}, {}
        self._size = None
        self.alloc_count = 0
        self.settle_s = 0.3
        self.debug_raw = False
        self.decks_rendered = 0              # landscape passes drawn by the last render() (tests / cost checks)
        self.swarm_rendered = False
        self._want, self._want_since = None, 0.0
        self._last_t = self._last_travel = None
        self._T = self._z = self._roll = self._spin = 0.0
        self._tu = 0.0                       # unwrapped show time (float64): palette / hue phases are taken from it
        self._ph = dict(tw=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], pz=0.0, sk=0.0, hex=0.0)   # see _advance
        self.spin_rate = TAU * 0.25          # rad/s at spin = 1 when the engine passes no accumulated phase
        self._ping = 0
        self._pal = np.zeros((5, 3), dtype="f4")
        self._swarm_live = False
        try:                                 # blend state is read back with raw GL, as math_layers does
            import math_layers as _ml
            self._gl, self._gl_enums = _ml._gl_lib(), (_ml._GL_BLEND, _ml._GL_BLEND_ENUMS, _ml.gl_get_int)
        except Exception:
            self._gl = None

    @property
    def skins(self):
        return self._skins

    @skins.setter
    def skins(self, engine):
        self._skins = engine
        if engine is None or self.skin_programs is not None or _mat is None:
            return
        try:
            self._sprogs = [self.ctx.program(vertex_shader=VS, fragment_shader=skinned_source(i)) for i in range(len(LAND_FS))]
            self._svaos = [self.ctx.vertex_array(p, [(self._quad, "2f", "in_pos")]) for p in self._sprogs]
            self.skin_programs = True
        except Exception as ex:
            print(f"[skins] landscape skins disabled (shader did not compile): {str(ex)[:300]}")
            self._sprogs, self._svaos = [None] * len(LAND_FS), [None] * len(LAND_FS)
            self.skin_programs = False

    # ---- resources
    def _ensure(self, w, h):
        """Targets are allocated ONCE at their largest size; land_res only changes the viewport the decks draw into
        (a modulated or cross-faded land_res costs no reallocation and never wipes the swarm trails).
        a / b: decks, RGBA8, full size. s0 / s1: swarm trails, half size, 16-bit float (8 bits cannot decay to
        black). out: what the engine samples, always full size."""
        self._size = (w, h)
        if self._tex:
            return
        W, H = self.out_size
        self.swarm_size = (max(16, W // 2), max(16, H // 2))
        for name in ("a", "b", "s0", "s1", "out"):
            if name in ("s0", "s1"):
                tx = self.ctx.texture(self.swarm_size, 4, dtype="f2")
            else:
                tx = self.ctx.texture((W, H), 4)
            tx.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tx.repeat_x = tx.repeat_y = False
            self._tex[name] = tx
            self._fbo[name] = self.ctx.framebuffer(color_attachments=[tx])
            self._fbo[name].clear(0.0, 0.0, 0.0, 1.0)
        self._swarm_live = False
        self.alloc_count += 1

    def _free_targets(self):
        for d in (self._fbo, self._tex):
            for o in d.values():
                o.release()
            d.clear()

    def release(self):
        self._free_targets()
        for o in ([x for x in self._svaos + self._sprogs if x is not None] + self._vaos + self._progs + [self._trail_vao, self._trail, self._comp_vao, self._comp, self._svao,
                                              self._sprog, self._sbuf, self._quad]):
            o.release()
        self._vaos, self._progs, self._size = [], [], None

    # ---- time / camera (speeds are integrated, so fader moves never jump positions)
    def _advance(self, v, t, travel_z, bands):
        t = float(t) if math.isfinite(float(t)) else 0.0
        dt = (t if abs(t) < 1e5 else 0.0) if self._last_t is None else t - self._last_t
        self._last_t = t
        if not math.isfinite(dt) or abs(dt) > 1e4:
            dt = 0.0
        tz = float(travel_z) if math.isfinite(float(travel_z)) else 0.0
        dz = 0.0 if self._last_travel is None else tz - self._last_travel
        self._last_travel = tz
        if not abs(dz) < 50.0:
            dz = 0.0
        cam = _idx(v, "land_cam", 2)
        fwd = {0: 1.15, 1: 0.35, 2: 0.6 + bands[0] * 4.8}[cam]
        dzc = dt * fwd + dz * 3.0                                     # camera travel this frame
        dsc = dt * (1.0 + 0.6 * bands[0])                             # surface scroll this frame
        self._T = (self._T + dt) % T_WRAP
        self._tu += dt
        self._z = (self._z + dzc) % Z_WRAP
        # per-use phases (speeds integrated, so a fader move never jumps a position; each wraps where ITS use repeats)
        tw, morph, ph = _f(v, "land_twist") * 1.5, _f(v, "land_morph"), self._ph
        for i, rate in enumerate((0.06 + tw * 0.22, 0.08 + tw * 0.25, 0.14 + tw * 0.42)):       # twist along z
            ph["tw"][i] = (ph["tw"][i] + rate * dzc) % TAU
        for i, (kz, ks) in enumerate(((0.85 * (1.2 + morph * 1.4), 0.5), (0.75, 0.5), (0.9 + morph * 0.9, 2.25))):
            ph["v"][i] = (ph["v"][i] + kz * dzc - ks * dsc) % V_WRAP[i]                         # tile rows
        ph["pz"] = (ph["pz"] + 0.02 * dzc) % 1.0                                                # colour along z
        ph["sk"] = (ph["sk"] + 0.75 * dsc) % 1.0                                                # vortex streaks
        ph["hex"] += 0.25 * dsc                                                                 # hex drift (cells)
        if cam == 2:
            self._roll = (self._roll + dt * (0.18 + bands[1] * 0.6)) % TAU
        self._spin = (self._spin + dt * _f(v, "spin") * self.spin_rate) % TAU       # used unless spin_phase is given
        sk = self._skins
        if sk is not None and dzc:                                    # skins stay on their wall while the camera flies
            ea = _mat.LAND_ENVS[_idx(v, "land_a", 3)]
            eb = _mat.LAND_ENVS[_idx(v, "land_b", 3)]
            if sk.active(ea):
                sk.advance(ea, dzc)
            if eb != ea and sk.active(eb):
                sk.advance(eb, dzc)
        return dt, cam

    def _camera(self, v, cam, spin_phase=None):
        t, punch = self._T, _f(v, "punch")
        if cam == 1:
            a = t * 0.45
            x, y = math.sin(a) * 0.8, math.sin(a * 0.7) * 0.45
            yaw, pitch, roll = math.sin(a) * 0.6, math.cos(a * 0.7) * 0.25, math.sin(t * 0.3) * 0.35
        elif cam == 2:
            x, y = math.sin(t * 0.5) * 0.28, math.cos(t * 0.4) * 0.24
            yaw, pitch, roll = 0.0, 0.0, self._roll
        else:
            x, y = math.sin(t * 0.31) * 0.42, math.cos(t * 0.24) * 0.32
            yaw, pitch, roll = math.sin(t * 0.19) * 0.22, math.cos(t * 0.17) * 0.14, math.sin(t * 0.13) * 0.18
        yaw += _f(v, "cam_yaw") * 1.2
        pitch += _f(v, "cam_pitch") * 0.9
        roll += _f(v, "cam_roll") * math.pi
        sp = self._spin if spin_phase is None else float(spin_phase)
        roll -= sp if math.isfinite(sp) else 0.0       # "spin": the whole fly-through (and the swarm) rolls; the sign
        #                                                matches SCENE_FS, whose rot(spinPh) turns the picture the other way
        fov = 45.0 + 80.0 * min(max(_f(v, "land_fov", 0.45), 0.0), 1.0) + 32.0 * punch
        return (x, y, self._z + 4.5 * punch), camera_matrix(yaw, pitch, roll), min(fov, 150.0)

    def _deck(self, which, mode, v, cam_pos, cam_mat, fov, bands, warp_tex, warp):
        prog, vao = self._progs[mode], self._vaos[mode]
        sk = self._skins
        if sk is not None and self._sprogs[mode] is not None and sk.active(_mat.LAND_ENVS[mode]):
            prog, vao = self._sprogs[mode], self._svaos[mode]         # only a world with a skinned slot pays for skins
            sk.bind(prog, _mat.LAND_ENVS[mode])
            self.skinned_decks += 1
        w, h = self._size
        folds, ph, k = float(min(max(round(_f(v, "folds", 8.0)), 3), 16)), self._ph, min(mode, 2)
        nx = math.floor(max(folds * 2.0, 12.0) + 0.5)                 # hex cells around the dome (as in HEX_FS)
        ph["hex"] %= nx * 64.0                                        # keep it small; any multiple of nx is invisible
        W, H = self.out_size
        uni = dict(uT=self._T, uFov=fov, uFolds=folds, uTw=ph["tw"][k], uV=ph["v"][k], uPz=ph["pz"],
                   uPal=(self._tu * 0.01) % 1.0, uSk=ph["sk"], uSh=ph["hex"] % nx, uWs=(w / W, h / H),
                   uTwist=_f(v, "land_twist") * 1.5, uCurve=_f(v, "land_curve") * 1.5, uMorph=_f(v, "land_morph"),
                   uMirror=float(_idx(v, "land_mirror", 1)), uStyle=float(_idx(v, "land_style", 2)),
                   uLow=bands[0], uMid=bands[1], uHigh=bands[2], uWarp=warp, uRaw=1.0 if self.debug_raw else 0.0,
                   uRes=(float(w), float(h)), uCam=cam_pos, texWarp=0)
        for name, val in uni.items():
            if name in prog:
                prog[name].value = val
        if "uCamMat" in prog:
            prog["uCamMat"].write(cam_mat.tobytes())
        if "pal" in prog:
            prog["pal"].write(self._pal.tobytes())
        if warp_tex is not None:
            warp_tex.use(0)
        self._fbo[which].use()
        self.ctx.viewport = (0, 0, w, h)
        vao.render(moderngl.TRIANGLE_STRIP)

    def _swarm_pass(self, v, dt, cam_mat, fov, bands):
        w, h = self.swarm_size
        ctx = self.ctx
        ctx.viewport = (0, 0, w, h)
        if not self._swarm_live:                                     # first frame after a gap: no stale trails
            self._fbo["s0"].clear(0.0, 0.0, 0.0, 1.0)
            self._fbo["s1"].clear(0.0, 0.0, 0.0, 1.0)
            self._swarm_live = True
        if dt <= 0.0:                                                # frozen: hold the picture
            return self._tex["s%d" % self._ping]
        punch = _f(v, "punch")
        if punch > self.swarm.burst:
            self.swarm.burst = punch
        verts = self.swarm.step(dt, self._tu, _f(v, "swarm_turb", 0.4), _f(v, "swarm_vortex", 0.35), bands[0])
        self._sbuf.write(verts.tobytes())
        src, dst = "s%d" % self._ping, "s%d" % (1 - self._ping)
        self._fbo[dst].use()
        ctx.disable(moderngl.BLEND)
        self._tex[src].use(0)
        self._trail["texPrev"].value = 0
        decay = 0.80 + 0.185 * min(max(_f(v, "swarm_trail", 0.6), 0.0), 1.0)
        self._trail["uDecay"].value = decay ** (min(dt, 0.05) * 60.0)
        self._trail_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.enable(moderngl.BLEND)
        ctx.blend_equation = moderngl.FUNC_ADD
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        sp = self._sprog
        sp["uRes"].value = (float(w), float(h))
        sp["uFov"].value = fov
        sp["uTint"].value = min(max(_f(v, "swarm_tint"), 0.0), 1.0)
        sp["uHigh"].value = bands[2]
        sp["uHueT"].value = (self._tu * 0.02) % 1.0
        # equal light at every persistence: long trails draw fainter lines (steady state ~ alpha / (1 - decay))
        sp["uGain"].value = (min(max((1.0 - decay) / (1.0 - 0.911), 0.12), 2.0)
                             / (1.0 + 1.5 * self.swarm.burst))       # a shockwave draws long streaks: same light, not more
        sp["uCamMat"].write(cam_mat.tobytes())
        sp["pal"].write(self._pal.tobytes())
        self._svao.render(moderngl.LINES, vertices=self.swarm.n * 2)
        self._ping = 1 - self._ping
        return self._tex[dst]

    def render(self, values: dict, clock: dict, t: float, travel_z: float, dt: float = None, palette_rgb=None,
               bands=(0.0, 0.0, 0.0), spin_phase=None) -> moderngl.Texture:
        """spin_phase: the engine's accumulated roll angle in radians. None = integrate values["spin"] here."""
        ctx, v = self.ctx, values
        res = snap_res(_f(v, "land_res", 0.5))
        self._ensure(max(16, int(round(self.out_size[0] * res))), max(16, int(round(self.out_size[1] * res))))
        w, h = self._size
        bands = [min(max(float(b), 0.0), 1.0) if math.isfinite(float(b)) else 0.0 for b in (list(bands) + [0, 0, 0])[:3]]
        step, cam = self._advance(v, t, travel_z, bands)
        if palette_rgb is not None:
            pal = np.asarray(palette_rgb, dtype="f4").reshape(-1, 3)
            self._pal[:] = pal[np.arange(5) % len(pal)]
        land, swarm = min(max(_f(v, "l_land"), 0.0), 1.0), min(max(_f(v, "l_swarm"), 0.0), 1.0)
        x = min(max(_f(v, "land_xfade"), 0.0), 1.0)
        cam_pos, cam_mat, fov = self._camera(v, cam, spin_phase)

        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        blend_on, blend_state = False, None
        if self._gl is not None:             # save the caller's blend state; it is restored whatever happens
            en, enums, get = self._gl_enums
            blend_on = bool(self._gl.glIsEnabled(en))
            blend_state = [get(self._gl, e) for e in enums]
        self.decks_rendered, self.swarm_rendered, self.skinned_decks = 0, False, 0
        try:
            ctx.viewport = (0, 0, w, h)
            ctx.disable(moderngl.BLEND)
            has_b = land > 0.001 and x > 0.002
            if land > 0.001:
                blend = _idx(v, "land_blend", 7)
                if not (has_b and x > 0.998 and blend == 0):                 # a full plain mix to B does not need A
                    self._deck("a", _idx(v, "land_a", 3), v, cam_pos, cam_mat, fov, bands, None, 0.0)
                    self.decks_rendered += 1
                elif _f(v, "land_warp") > 0.001:
                    self._deck("a", _idx(v, "land_a", 3), v, cam_pos, cam_mat, fov, bands, None, 0.0)
                    self.decks_rendered += 1
                if has_b:
                    self._deck("b", _idx(v, "land_b", 3), v, cam_pos, cam_mat, fov, bands, self._tex["a"],
                               min(max(_f(v, "land_warp"), 0.0), 1.0) * 1.5)
                    self.decks_rendered += 1
            stex = None
            if swarm > 0.001:
                stex = self._swarm_pass(v, dt if dt is not None else step, cam_mat, fov, bands)
                self.swarm_rendered = True
            else:
                self._swarm_live = False
            ctx.disable(moderngl.BLEND)
            self._fbo["out"].use()
            ctx.viewport = (0, 0) + self.out_size
            W, H = self.out_size
            self._comp["uSc"].value = (w / W, h / H)
            self._comp["uLim"].value = ((w - 0.5) / W, (h - 0.5) / H)
            self._tex["a"].use(0)
            self._tex["b"].use(1)
            (stex or self._tex["s0"]).use(2)
            c = self._comp
            c["texA"].value, c["texB"].value, c["texS"].value = 0, 1, 2
            c["uX"].value, c["uLand"].value, c["uSwarm"].value = x, land if land > 0.001 else 0.0, swarm if stex else 0.0
            # the radial prism split turns fine image detail into rainbow mush: much gentler while a deck is skinned
            c["uDisp"].value = min(max(_f(v, "land_disp"), 0.0), 1.0) * (0.35 if self.skinned_decks else 1.0)
            c["uHasB"].value = 1.0 if has_b else 0.0
            c["uBlend"].value = _idx(v, "land_blend", 7)
            c["uRaw"].value = 1.0 if self.debug_raw else 0.0
            self._comp_vao.render(moderngl.TRIANGLE_STRIP)
        finally:
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
            except Exception:
                ctx.screen.use()
        return self._tex["out"]


# ====================================================================== panel monitor
THUMB_SIZE = (128, 80)
BOOSTABLE = ("l_kene", "l_beads", "l_lines", "l_ai", "l_video", "serpent")   # drawn inside SCENE_FS: free to preview


def layer_keys(names):
    """Layer names of the Photoshop-style layer panel, from PARAMS-like names or "p_<name>" uniform names:
    every l_* param plus serpent. l_swarm shares the l_land texture, so it is not a separate key."""
    out = []
    for n in names:
        n = n[2:] if n.startswith("p_") else n
        if (n.startswith("l_") and n != "l_swarm") or n == "serpent":
            if n not in out:
                out.append(n)
    return out


class PreviewTap:
    """Downscaled JPEG of a texture for the panel's output monitor.

    capture() is called on the GL thread: one small blit + a ~190 KB readback; the JPEG encode runs on a worker
    thread and the result goes to `sink(bytes)`. Nothing happens unless the caller decides a frame is due."""

    def __init__(self, ctx: moderngl.Context, size=(320, 200), quality=70):
        self.ctx, self.size, self.quality = ctx, (int(size[0]), int(size[1])), int(quality)
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._prog = ctx.program(vertex_shader=VS, fragment_shader=BLIT_FS)
        self._vao = ctx.vertex_array(self._prog, [(self._quad, "2f", "in_pos")])
        self._fbo = ctx.simple_framebuffer(self.size, components=3)
        self._busy = threading.Lock()
        self._mix = None
        self.captures = 0
        self._pbo = ctx.buffer(reserve=self.size[0] * self.size[1] * 3)   # readback lands here; collected a frame later
        self._pending = None                 # (sink, thumb) of the readback in flight
        self._idle = ctx.texture((1, 1), 4, data=b"\x00\x00\x00\xff")     # left bound on our unit between captures

    UNIT = 15                                # private texture unit: never one the scene shader samples

    def _blit(self, tex):
        """Draw `tex` scaled into our small target. The source is unbound again afterwards, so a render target is
        never left attached to a sampler unit (no feedback loop on the next scene draw)."""
        ctx = self.ctx
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        try:
            self._fbo.use()
            ctx.viewport = (0, 0) + self.size
            tex.use(self.UNIT)
            self._prog["tex"].value = self.UNIT
            self._vao.render(moderngl.TRIANGLE_STRIP)
            self._idle.use(self.UNIT)
        finally:
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()

    def read(self, tex):
        """RGB8 bytes (bottom row first) of `tex` scaled to self.size, synchronously (tests / one-off use)."""
        self._blit(tex)
        return self._fbo.read(components=3)

    def encode(self, data, thumb=False):
        from PIL import Image
        img = Image.frombytes("RGB", self.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        if thumb:
            img = img.resize(THUMB_SIZE, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.quality)
        return buf.getvalue()

    def capture(self, tex, sink, wait=False):
        """Read `tex` now and hand the JPEG to sink(bytes) from a worker thread. Skips if an encode is running."""
        return self._grab(lambda: self.read(tex), sink, wait, False)

    def snap(self, tex, sink):
        """One picture of `tex` for a caller that is not the panel monitor (the ONLINE assistant's SEE THE OUTPUT):
        a small blit + a readback into the pixel buffer NOW (no GPU stall), the bytes collected by a later
        collect() and JPEG-encoded on a worker thread, then sink(bytes). False if one is still in flight."""
        self.collect()
        if self._pending is not None or self._busy.locked():
            return False
        self._blit(tex)
        self._fbo.read_into(self._pbo, components=3)
        self._pending = (sink, False)
        self.captures += 1
        return True

    # ---- preview-only layer mix: the scene shader drawn once more, small, with some layers muted
    def render_mix(self, scene_prog, scene_vao, uni, mask, bright=False):
        self._draw_mix(scene_prog, scene_vao, uni, mask, bright)
        return self._mix[0].read(components=3, attachment=0, dtype="f1")

    def _draw_mix(self, scene_prog, scene_vao, uni, mask, bright=False):
        """RGB8 bytes of the scene with only the layers in `mask` (names like "l_kene", "serpent").

        Must run right after the engine's own scene draw: the program still has this frame's uniforms and
        textures bound. Layer uniforms are overwritten here; the engine rewrites every p_* next frame. The projector
        frame, the echo buffer and the limiter never see this draw. A masked-in pattern layer that is down on the
        projector (< 0.05) is previewed at 0.8, so a layer can be looked at before it is brought in."""
        ctx = self.ctx
        if self._mix is None:
            texs = [ctx.texture(self.size, 4, dtype="f2") for _ in range(2)]
            self._mix = (ctx.framebuffer(texs), texs)
        mask = set(mask or ())
        for name in layer_keys(uni):
            u = "p_" + name
            if u not in scene_prog:
                continue
            real = float(uni.get(u, 0.0))
            if name not in mask:
                real = 0.0
            elif real < 0.05 and name in BOOSTABLE:
                real = 0.8
            scene_prog[u].value = real
        if "pxh" in scene_prog:
            scene_prog["pxh"].value = 1.0 / self.size[1]
        if bright and "limiter" in scene_prog:                       # thumbnails: one layer alone needs no limiting
            scene_prog["limiter"].value = 0.0
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        try:
            self._mix[0].use()
            ctx.viewport = (0, 0) + self.size
            scene_vao.render(moderngl.TRIANGLE_STRIP)
        finally:
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()

    def service(self, remote, disp, scene_prog, scene_vao, uni, wait=False):
        """One call per frame from the engine. Nothing happens unless a panel is watching and a capture is due.

        The readback is asynchronous: a capture draws into a small target and starts a transfer into a pixel
        buffer (no GPU stall); the bytes are collected on a LATER call and JPEG-encoded on a worker thread. A job is
        only taken from the remote when it can actually be done, so a busy encoder never eats one."""
        self.collect(wait)
        if self._pending is not None or self._busy.locked():
            return None
        job = remote.preview_job()
        if not job:
            return None
        kind, arg = job
        if kind == "thumb":
            self._draw_mix(scene_prog, scene_vao, uni, [arg], True)
            self._mix[0].read_into(self._pbo, components=3, attachment=0, dtype="f1")
            self._pending = (lambda d: remote.set_thumb(arg, d), True)
        elif kind == "mix":
            self._draw_mix(scene_prog, scene_vao, uni, arg)
            self._mix[0].read_into(self._pbo, components=3, attachment=0, dtype="f1")
            self._pending = (lambda d: remote.set_preview(d, arg), False)
        else:
            self._blit(disp)
            self._fbo.read_into(self._pbo, components=3)
            self._pending = (remote.set_preview, False)
        self.captures += 1
        if wait:
            self.collect(True)
        return job

    def collect(self, wait=False):
        """Hand the finished readback (if any) to the encoder thread."""
        if self._pending is None or not self._busy.acquire(blocking=False):
            return False
        (sink, thumb), self._pending = self._pending, None
        try:
            data = self._pbo.read()
        except Exception:
            self._busy.release()
            raise

        def work():
            try:
                sink(self.encode(data, thumb))
            except Exception:
                pass
            finally:
                self._busy.release()

        th = threading.Thread(target=work, daemon=True)
        th.start()
        if wait:
            th.join()
        return True

    def _grab(self, read, sink, wait, thumb):
        if not self._busy.acquire(blocking=False):
            return False
        try:
            data = read()
        except Exception:
            self._busy.release()
            raise
        self.captures += 1

        def work():
            try:
                sink(self.encode(data, thumb))
            except Exception:
                pass
            finally:
                self._busy.release()

        th = threading.Thread(target=work, daemon=True)
        th.start()
        if wait:
            th.join()
        return True

    def release(self):
        if self._mix is not None:
            self._mix[0].release()
            for t in self._mix[1]:
                t.release()
            self._mix = None
        for o in (self._fbo, self._vao, self._prog, self._quad, self._pbo, self._idle):
            o.release()


# ====================================================================== standalone gallery / benchmark
GALLERY_PALETTE = [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]


def hidden_context(size=(64, 64)):
    from math_layers import hidden_context as hc
    return hc(size)


def default_values(**over):
    import params
    v = params.defaults()
    v["l_land"] = 1.0
    v.update(over)
    return v


def make_clock(t, bpm=124.0):
    b = t * bpm / 60.0
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def read_texture(tex):
    return np.frombuffer(tex.read(), dtype=np.uint8).reshape(tex.height, tex.width, 4)[::-1]


def render_frames(lp, values, t0=4.0, frames=1, dt=1 / 60, bands=(0.5, 0.3, 0.2), palette=None):
    """Step the pass `frames` times from show time t0 and return the last frame (h, w, 4) uint8."""
    tex = None
    for i in range(frames):
        t = t0 + i * dt
        tex = lp.render(values, make_clock(t), t, 0.0, dt, palette or GALLERY_PALETTE, bands)
    return read_texture(tex).copy()


def gallery(out_dir=None, out_size=(1280, 800)):
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir or HERE / "test_frames" / "touchvdmx")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, _win = hidden_context()
    shots = [(f"land{m}_{LAND_NAMES[m]}", dict(land_a=m), 1) for m in range(4)]
    shots += [(f"torus_style{s}", dict(land_a=1, land_style=s), 1) for s in (1, 2)]
    shots += [("hexdome_cube", dict(land_a=3, land_morph=0.5), 1), ("hexdome_octa", dict(land_a=3, land_morph=1.0), 1),
              ("corridor_mirror", dict(land_a=0, land_mirror=1, land_morph=0.8), 1),
              ("vortex_prism", dict(land_a=2, land_disp=1.0, land_twist=0.8), 1)]
    shots += [(f"ab_{BLEND_NAMES[b].replace(' ', '_')}", dict(land_a=0, land_b=3, land_xfade=0.5, land_blend=b), 1) for b in range(8)]
    shots += [("ab_warp", dict(land_a=2, land_b=1, land_xfade=1.0, land_blend=0, land_warp=1.0), 1),
              ("swarm", dict(l_land=0.0, l_swarm=1.0), 90), ("swarm_gold_vortex", dict(l_land=0.0, l_swarm=1.0, swarm_tint=1.0, swarm_vortex=1.0), 90),
              ("swarm_burst", dict(l_land=0.0, l_swarm=1.0, punch=1.0), 40),
              ("land_plus_swarm", dict(land_a=3, l_swarm=0.8), 90), ("warp_jump", dict(land_a=0, punch=1.0), 1)]
    tw, th, cols = 320, 200, 6
    sheet = Image.new("RGB", (tw * cols, (th + 16) * math.ceil(len(shots) / cols)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (tag, over, frames) in enumerate(shots):
        lp = LandPass(ctx, out_size)
        try:
            img = render_frames(lp, default_values(land_res=1.0, **over), frames=frames)
        finally:
            lp.release()
        a = img[..., :3].astype(np.float32)
        print(f"{tag:22s} mean={a.mean():6.1f} black={(a.max(axis=2) < 8).mean():5.1%}")
        im = Image.fromarray(img[..., :3])
        im.save(out_dir / f"{tag}.png")
        x, y = (i % cols) * tw, (i // cols) * (th + 16)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 16))
        dr.text((x + 4, y + 2), tag, fill=(200, 200, 200))
    sheet.save(out_dir / "contact_sheet.png")
    print("contact sheet:", out_dir / "contact_sheet.png")


def bench(frames=60, out_size=(1280, 800)):
    ctx, _win = hidden_context()
    for res in (0.5, 1.0):
        for name, over in ([(LAND_NAMES[m], dict(land_a=m)) for m in range(4)]
                           + [("A+B screen", dict(land_a=0, land_b=3, land_xfade=0.5)), ("swarm only", dict(l_land=0.0, l_swarm=1.0))]):
            lp = LandPass(ctx, out_size)
            v = default_values(land_res=res, **over)
            render_frames(lp, v, frames=3)
            ctx.finish()
            t0 = time.perf_counter()
            render_frames(lp, v, frames=frames)
            ctx.finish()
            print(f"res {res:.2f} {name:12s} {(time.perf_counter() - t0) / frames * 1000:6.2f} ms")
            lp.release()


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        rest = [a for a in sys.argv[1:] if not a.startswith("--")]
        gallery(rest[0] if rest else None)
