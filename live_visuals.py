"""Live audio-reactive psychedelic projector: render engine v2 (standalone, no TouchDesigner needed).

Everything the projector shows is drawn here; every look is driven by the shared parameter contract in
params.py (faders, modulation, scenes, MIDI and the web panel live in control.py). Each frame:

    frame = controller.update(dt, bands)   ->  every frame.values entry becomes the float uniform p_<name>

Negative-space design for projection (black = no light). Layers, all sampled through one warp chain
(ripples -> serpentine warp -> wave -> pan/rotate -> punch -> bulge -> twirl -> zoom / travel):
  kene   - generative crawling labyrinth line field (inspired by Amazonian kene, not a copy of any design):
           hash-seeded pivoting walls, bold primary lines, fine parallel infill, stepped diamonds
  beads  - Huichol-style hex-offset beadwork in nested petal medallions
  lines  - flower-of-life arcs + petal contours with analytic glow
  ai     - AI textures (textures/), luma-keyed, relief-lit
  video  - pre-rendered loops (videos/), luma-keyed underlay that the live layers fly over
  serpents - segmented luminous serpents slithering over the scene
Two topologies blended by p_symmetry: 0 = flat field tiling the whole frame (crawls along p_flow /
p_flow_dir), 1 = N-fold mandala on a conformal log-polar lattice. Travel modes fly INTO the visual
(zoom tunnel, 3D tube / ground plane, depth layers, Droste spiral). The TAS "motion key" light chase
(p_chase*) sweeps a soft hue window over every layer's per-element index, synced to the bar.
-> decaying zoom/rotate/hue echo -> performance stage (invert, flash, strobe, master, blackout) -> 4-corner pin.

Run:
  python live_visuals.py                 # windowed preview
  python live_visuals.py --monitor 1     # borderless on projector (Extend displays first)
  python live_visuals.py --panel         # + control panel in a browser on this laptop
  python live_visuals.py --spout         # also send to TouchDesigner/Resolume via Spout
  python live_visuals.py --test 300      # headless self-test, saves frames to test_frames/, prints fps
  python live_visuals.py --gallery       # headless look-check -> test_frames/gallery/contact_sheet.png
  headless helpers: --set name=value (repeatable), --mode PRESET, --extreme min|max|randN, --out DIR,
                    --batch jobs.json, --test-audio silent, --phase-offset K, --no-nan-guard, --limit-mean X

Keys (projector / preview window)
  looks      M layer preset (kene / kene+beads / beads / ai / both / lines / all) | P next palette (4 s glide)
             1-9 fold count | TAB flat field <-> mandala | T twirl | F echo on/off | UP/DOWN trails
             SPACE next AI texture | V next video clip
  space      - / = zoom out / in | [ / ] density | LEFT/RIGHT crawl speed | A/D crawl angle
             O / L bulge / pinch | U wave | N travel mode | H / Y travel slower / faster (through 0 = reverse)
             E serpents on/off
  hold       X flash | Z strobe | I invert | Q freeze | K blackout | J zoom punch
  tempo      , or . tap tempo
  mouse      click = ripple at that point (when not calibrating)
  mapping    Ctrl+C corner-pin calibrate (drag corners) | R reset pin (while calibrating) | S save pin | Ctrl+G grid
  guarded    Ctrl+BACKSPACE reset space + shape + motion to defaults | Ctrl+W toggle touch remote / panel server
  quit       preview window: ESC. On the projector (--monitor): Shift+ESC, or ESC twice within 1 s (exit code 0);
             closing the window needs the same confirmation. The hint is printed on the console, never projected.

Show mode (--monitor N): if the projector is not connected the app waits for it (retry every 2 s) instead of
opening a preview window; if it is unplugged mid-show the window is parked and rendering pauses until it is
back. The cursor is hidden over the projection except while calibrating.

Touch remote / panel: --remote starts it enabled, --remote-port picks the port (default 8765). The token
is persistent (.remote_token, or --remote-token) so an open panel tab survives a watchdog restart.
See remote_control.py, control.py and HANDOFF.md.
"""
import argparse
import itertools
import json
import logging
import os
import math
import secrets
import threading
import time
import warnings
from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw

from params import PARAMS, TRIGGERS, Frame, StaticController, defaults
from remote_control import RemoteControl, apply_commands, quad_ok

try:
    from control import Controller
except ImportError:
    Controller = None
try:
    from math_layers import FractalPass
except ImportError:
    FractalPass = None
try:
    from art_layer import ArtPass
except ImportError:
    ArtPass = None

HERE = Path(__file__).parent
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)   # it warns on every (intended) decode-time rescale
FOLDS = [3, 4, 5, 6, 8, 10, 12, 16, 24]
# layer presets (key M, --mode): name -> (l_kene, l_beads, l_lines, l_ai)
LAYER_PRESETS = [
    ("kene", (1.0, 0.0, 0.0, 0.0)),
    ("kene+beads", (1.0, 0.8, 0.0, 0.0)),
    ("beads", (0.0, 1.0, 0.0, 0.0)),
    ("ai", (0.0, 0.0, 0.0, 1.0)),
    ("both", (0.0, 1.0, 0.0, 1.0)),
    ("lines", (0.35, 0.0, 1.0, 0.0)),
    ("all", (1.0, 0.7, 0.5, 0.5)),
]
MODES = [n for n, _ in LAYER_PRESETS]
MODE_W = [w for _, w in LAYER_PRESETS]
PRESET_KEYS = ("l_kene", "l_beads", "l_lines", "l_ai")
# 5 jewel stops each, cycled smoothly (lines / AI tint) or as discrete bead colours
PALETTES = [
    ("Aya Jewel", [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]),
    ("Cosmic Gold/Violet", [(1.00, 0.68, 0.08), (1.00, 0.30, 0.02), (0.55, 0.08, 1.00), (0.10, 0.75, 1.00), (0.85, 0.10, 0.75)]),
    ("Gorge Twilight", [(0.05, 0.22, 1.00), (1.00, 0.52, 0.05), (0.00, 0.90, 0.75), (0.90, 0.15, 0.42), (0.30, 0.10, 0.90)]),
    ("Emerald Shipibo", [(0.00, 0.85, 0.35), (0.55, 1.00, 0.12), (1.00, 0.78, 0.08), (0.00, 0.62, 0.60), (1.00, 0.16, 0.08)]),
    ("Spectral", [(1.00, 0.06, 0.10), (1.00, 0.80, 0.00), (0.05, 0.95, 0.25), (0.05, 0.40, 1.00), (0.85, 0.05, 1.00)]),
]
TWISTS = [0.0, 0.35, -0.35, 0.8]  # key T: twirl presets (0 = straight)
BREATH_PERIOD = 7.5               # seconds: 4 bars at 128 BPM
N_RIPPLES = 8
METER_SIZE = (64, 40)             # brightness meter resolution
LIMIT_MEAN, LIMIT_BLACK = 0.36, 0.15   # ceiling: mean peak-channel level; least near-black area (mandala looks only)
CLOCK_UNIFORMS = ("beat", "beat_phase", "bar_phase", "phrase_phase")
TEX_CAP = 64                      # most-recent AI textures kept on the GPU (older ones are evicted)
STRICT = False                    # True in headless runs: guarded() re-raises so tests see every failure
# PARAMS that never reach the shader as p_<name>: the CPU integrates / resolves them into other uniforms
CPU_ONLY = {"speed", "flow", "rotate", "palette", "travel", "serpent_speed", "chase_rate"} | {
    n for n, v in PARAMS.items() if v[3] in ("fractal", "art")}   # consumed by math_layers.FractalPass / art_layer.ArtPass
RESET_GROUPS = ("space", "shape", "motion")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")

VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }
"""

SCENE_FS = """
#version 330
uniform sampler2D texA, texB, prevFrame, texV, texF, texR;
uniform float t, mixAB, bass, mid, high, aspect, pxh, hasVideo, limiter, nanGuard;
uniform float rotPh, chasePh, chaseCol, travelAct, breath, calm;
// Every phase that grows for hours is integrated in double precision on the CPU and arrives here already
// wrapped at a period where the pattern repeats exactly, so float32 never has to resolve a large number.
uniform float swN[6];               // swim noise offsets (x 0.21 0.31 0.23 0.17 0.05 0.03), wrapped at 1600
uniform float swS[4];               // swim angles (x 0.9 0.7 0.11 0.2), wrapped at TAU
uniform float aiDrift;              // AI texture drift, wrapped at 1
uniform float flowM, flowF;         // crawl phase, wrapped at the mandala / flat lattice period (320 half-wedges)
uniform float trU, trT, trG;        // travel: log-polar shift, tube shift (mandala period), ground-plane shift (flat period)
uniform float trZ, trSway;          // travel octave / depth-layer counter (wrapped at 256), sway phase (wrapped at 100 PI)
uniform vec4 serpA[8];              // per serpent, current crossing: heading, track offset, track bend, head position along the track
uniform vec4 serpC[8];              // same for the previous crossing (its tail is still leaving the frame)
uniform vec4 serpB[8];              // per serpent: wave phase 1, wave phase 2 (wrapped at TAU), wave seed of the current / previous crossing
uniform float serpW[2];             // serpentine-warp phases (wrapped at TAU)
uniform float beat, beat_phase, bar_phase, phrase_phase;
uniform float p_master, p_flash, p_strobe, p_invert, p_freeze, p_blackout, p_punch;
uniform float p_zoom, p_pan_x, p_pan_y, p_symmetry, p_folds, p_flow_dir;
uniform float p_travel_mode, p_tunnel_curve, p_depth_fog;
uniform float p_serpent, p_serpent_mode, p_serpent_count, p_serpent_segments, p_serpent_wiggle, p_serpent_size;
uniform float p_swim, p_ripple, p_ripple_decay, p_twirl, p_bulge, p_wave, p_density;
uniform float p_hue, p_saturation, p_chase, p_chase_width, p_chase_mode, p_trails;
uniform float p_l_kene, p_l_beads, p_l_lines, p_l_ai, p_l_echo, p_l_video, p_luma_key, p_l_fractal, p_l_art;
uniform vec3 pal[5];
uniform vec3 rip[8];                // x, y (content uv), age in seconds (< 0 = inactive)
in vec2 v_uv;
layout(location=0) out vec4 f;      // what the projector shows: crisp geometry over a dimmer echo
layout(location=1) out vec4 fbE;    // echo energy, fed back next frame (never shown directly)
const float TAU = 6.28318530718, PI = 3.14159265359, LN2 = 0.69314718;
const float SFLAT = 2.2;            // flat field: lattice units per frame height at density 0.5
const float HSF = 0.3926991;        // flat field: lobe half-period (PI/8)

// lattice description for the topology being shaded (set by main before each shade() call)
float G_hs, G_F, G_cw, G_nb, G_kl, G_thr, G_vs, G_ts, G_ar, G_cofs, G_sparse;

mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
float max3(vec3 c){ return max(c.r, max(c.g, c.b)); }
vec3 hueRot(vec3 c, float a){
  const mat3 toY=mat3(.299,.596,.211,.587,-.274,-.523,.114,-.322,.312);
  const mat3 toR=mat3(1.,1.,1.,.956,-.272,-1.106,.621,-.647,1.703);
  vec3 y=toY*c; float h=atan(y.z,y.y+1e-6)+a, ch=length(y.yz);
  return toR*vec3(y.x, ch*cos(h), ch*sin(h));
}
float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
float vnoise(vec2 p){ vec2 i=mod(floor(p),64.), j=mod(i+1.,64.), u=fract(p); u=u*u*(3.-2.*u);   // period 64
  return mix(mix(hash21(i),hash21(vec2(j.x,i.y)),u.x), mix(hash21(vec2(i.x,j.y)),hash21(j),u.x), u.y); }
float fbm(vec2 p){ float a=.5, s=0.;
  for(int i=0;i<3;i++){ s+=a*vnoise(p); p=mat2(1.6,1.2,-1.2,1.6)*p+7.3; a*=.5; } return s; }   // 0..0.875
vec2 fbm2(vec2 p){ return vec2(fbm(p), fbm(p+vec2(17.3,-9.1)))-0.4375; }
// palette lookup; the phase drifts slowly over the 8-bar phrase
vec3 palette(float x){ x=fract(x+0.06*sin(TAU*phrase_phase))*5.0; int i=int(x); float k=fract(x); k=k*k*(3.-2.*k);
  return mix(pal[i%5], pal[(i+1)%5], k); }
vec3 palStop(float k){ return pal[int(mod(k,5.0))]; }
float aa(float sd, float e){ return clamp(0.5 - sd/max(e,1e-5), 0., 1.); }     // coverage of a signed distance

// ---- TAS "motion key": every element carries a hue index 0..1; a soft window sweeps over the index.
// amount 0 = everything lit, 1 = only the window (plus a faint ghost of the structure).
float chaseKey(float idx){
  float m = floor(p_chase_mode+0.5), w;
  idx += G_cofs;
  if (m < 0.5) w = 1.0 - abs(fract(idx - chasePh) - 0.5)*2.0;
  else {
    w = 1.0 - abs(fract(idx*2.0 - chasePh*2.0) - 0.5)*2.0;
    if (m > 1.5) w = max(w, 1.0 - abs(fract(-idx*2.0 - chasePh*2.0 + 0.5) - 0.5)*2.0);
  }
  float key = smoothstep(1.0 - p_chase_width, 1.0, w);
  return mix(1.0, 0.045 + (0.955 + 0.35*bass + 0.25*exp(-beat_phase*5.0))*key, p_chase);   // the window kicks on the beat
}

// composition field: lotus-petal lobes on the lattice; >0 and <0 petals interleave
float lobe(float uu, float v){ return sin(G_F*uu)*cos(PI*v); }

// ---- kene: a generative labyrinth. Lattice points (i,j) with i+j odd each hold one wall of length 2
// (horizontal or vertical) that can pivot; rules per mirrored 8x8 super-tile give stepped diamonds,
// nested squares and grids, with hash-seeded defects that open labyrinth paths.
float keneBit(vec2 ij, float slot){
  vec2 st = floor((ij+4.0)/8.0);
  vec2 fo = abs(ij - st*8.0);
  float seed = hash21(mod(st, 8.0)*7.31+vec2(3.7,1.9));
  float base;
  if (seed < 0.62) base = mod(floor((fo.x+fo.y)*0.5), 2.0);          // nested stepped diamonds
  else if (seed < 0.84) base = step(fo.x, fo.y);                     // nested squares
  else base = mod(fo.x, 2.0);                                        // open grid
  float flip = step(hash21(fo*1.37 + seed*91.7 + mod(slot, 64.0)*13.1), 0.10);
  return abs(base-flip);
}
vec3 kene(vec2 g, float eg, out float kcov){
  float l = G_kl, e = eg/l;
  vec2 q = g/l;
  vec2 cid = floor(vec2(q.x+q.y, q.x-q.y)*0.5);
  float d = 1e3; vec2 wid = vec2(0);
  for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++){
    vec2 c = cid + vec2(dx,dy);
    vec2 ij = vec2(c.x+c.y+1.0, c.x-c.y);
    float ph = t*0.11 + hash21(mod(ij, 64.0)+0.5), slot = floor(ph);
    float s = keneBit(ij, slot);
    if (ph-slot > 0.82) s = mix(s, keneBit(ij, slot+1.0), smoothstep(0.82, 1.0, ph-slot));
    vec2 cs = normalize(vec2(1.0-s, s)+1e-4);                        // wall direction (pivots while s moves)
    vec2 lp = q - ij;
    vec2 dd = vec2(max(abs(dot(lp,cs))-1.0, 0.0), abs(dot(lp, vec2(-cs.y,cs.x))));
    float di = max(max(dd.x,dd.y), (dd.x+dd.y)*0.7071);              // octagonal metric: 45 degree chamfers
    if (di < d){ d = di; wid = ij; }
  }
  float bold = aa(d-0.115, e)*smoothstep(0.45, 0.16, e);
  float lv = floor((d-0.27)/0.30+0.5);                               // which parallel infill contour
  float dt = abs(d-0.27-lv*0.30);
  float thin = aa(dt-0.036, e)*step(0.14, d)*smoothstep(1.06, 0.92, d)*smoothstep(0.13, 0.05, e)*(1.0-G_sparse);
  float along = (wid.x+wid.y)/16.0;                                  // steps 1/8 per wall along a staircase
  float i1 = fract(along), i2 = fract(along + 0.09*(lv+1.0));
  float sh = 0.82+0.5*high*vnoise(q*3.0+t*3.0);
  float pp = 0.03125*(q.x+q.y) + t*0.02 + chaseCol;
  vec3 c = palette(pp)*bold*chaseKey(i1) + palette(pp+0.37+0.06*lv)*thin*chaseKey(i2);
  kcov = mix(smoothstep(0.30, 0.20, d), smoothstep(0.26, 0.17, d), G_sparse);
  return c*sh;
}

vec3 vidSample(vec2 g){
  vec2 uv = g*G_vs;
  uv = 1.0 - abs(1.0 - mod(uv, 2.0));                                // mirrored repeat
  return texture(texV, uv).rgb;
}

// ---- all lattice layers at lattice point g (eg = lattice units per pixel); cov = where geometry is
vec3 shade(vec2 g, float eg, out float cov){
  float hs=G_hs, F=G_F, cw=G_cw, nbh=G_nb, thr=G_thr;
  float uu=g.x, v=g.y/hs;
  float L = lobe(uu, v), aL = abs(L), lobeI = floor(F*uu/PI);
  float sb = smoothstep(0.0,0.3,p_l_beads), sa = smoothstep(0.0,0.3,p_l_ai), sk = smoothstep(0.0,0.3,p_l_kene);
  vec3 col = vec3(0); cov = 0.0;

  // Huichol beads: hex-offset beads coloured by nested petal contours
  if (p_l_beads > 0.0) {
    vec2 gb = vec2(uu/cw, v*nbh);
    float ring = floor(gb.x), odd = mod(ring,2.0);
    float gy = gb.y+0.5*odd, j = floor(gy);
    vec2 cell = vec2(fract(gb.x), fract(gy))-0.5;
    float ub = (ring+0.5)*cw, vb = (j+0.5-0.5*odd)/nbh;
    float Lb = lobe(ub, vb), lv = abs(Lb), sgn = step(0.0, Lb);
    float on = step(thr, lv) * mix(1.0, 1.0-sgn, sa);                // with AI on, beads keep the negative petals
    float k = floor(clamp((lv-thr)/(1.0-thr),0.,.999)*4.0);
    float lb = floor(F*ub/PI);
    float idx = fract((lb + k/4.0)/4.0);                             // inner -> outer, nested within a petal
    float d = length(cell), e = eg/cw;
    float bsz = 0.40+0.04*high;
    float disk = smoothstep(bsz, bsz-max(1.5*e,0.04), d)*smoothstep(0.55, 0.22, e);
    float dome = 0.35+0.65*(1.0-d*d/(bsz*bsz));
    float h = hash21(mod(vec2(ring,j), 64.0));
    float spec = smoothstep(0.17, 0.03, length(cell-vec2(0.11,-0.11)));
    float tw = pow(max(0.5+0.5*sin(beat*PI+h*TAU), 0.0), 10.0)*high;   // twinkle on the beat grid
    vec3 bc = palStop(k + lb*2.0 + 2.0*sgn);
    float lit = on*chaseKey(idx);
    col += (bc*dome + vec3(1.0,0.97,0.9)*(spec*0.45 + tw*0.9)) * disk * lit * p_l_beads;
    cov = max(cov, disk*on*p_l_beads);
  }

  // AI texture: luma-keyed into petal windows, lit as a relief
  if (p_l_ai > 0.0) {
    vec2 tuv = vec2(uu, g.y)*G_ts + vec2(0.0, aiDrift);
    float gr = min(eg*G_ts, 0.25);
    vec3 a = mix(textureGrad(texA,tuv,vec2(gr,0),vec2(0,gr)).rgb, textureGrad(texB,tuv,vec2(gr,0),vec2(0,gr)).rgb, mixAB);
    float lum = dot(a, vec3(.299,.587,.114));
    float key = smoothstep(0.26-0.08*bass, 0.58, lum);
    float te = max(gr, 0.004);
    float lx = dot(mix(textureGrad(texA,tuv+vec2(te,0),vec2(gr,0),vec2(0,gr)).rgb, textureGrad(texB,tuv+vec2(te,0),vec2(gr,0),vec2(0,gr)).rgb, mixAB), vec3(.299,.587,.114));
    float ly = dot(mix(textureGrad(texA,tuv+vec2(0,te),vec2(gr,0),vec2(0,gr)).rgb, textureGrad(texB,tuv+vec2(0,te),vec2(gr,0),vec2(0,gr)).rgb, mixAB), vec3(.299,.587,.114));
    vec3 n = normalize(vec3(-vec2(lx-lum, ly-lum)*(0.02/te), 1.0));
    vec3 ld = normalize(vec3(cos(t*0.5), sin(t*0.5), 0.55));
    float dif = max(dot(n, ld), 0.0), spc = pow(max(dot(n, normalize(ld+vec3(0,0,1))), 0.0), 24.0);
    vec3 tint = palette(lum*0.8 + lobeI*0.4 + 0.35*v + t*0.02);
    vec3 c = (mix(a*a*1.8, tint*lum*1.9, 0.72)*(0.45+0.9*dif) + tint*spc*(0.5+0.8*high))*key;
    float part = max(sb, sk);                                        // someone else owns the other petals
    float sL = mix(aL, L, sb);
    float wthr = mix(thr*0.45, thr, part);
    float win = smoothstep(wthr, wthr+0.08, sL);
    float idx = fract(lum*0.8 + lobeI*0.25);
    col += c*win*(0.85+0.5*bass)*chaseKey(idx)*p_l_ai;
    cov = max(cov, win*key*p_l_ai);
  }

  // kene line field: everywhere the medallions are not
  if (p_l_kene > 0.0) {
    float occ = smoothstep(thr-0.13, thr-0.05, aL)*max(sb, sa);
    float kc; vec3 c = kene(g, eg, kc);
    col += c*(1.0-occ)*p_l_kene;
    cov = max(cov, kc*(1.0-occ)*p_l_kene);
  }

  // petal contour (part of the sacred-lines layer)
  if (p_l_lines > 0.0) {
    float gl = 0.10*(1.0+1.5*bass);
    float gradL = F*sqrt(pow(cos(F*uu)*cos(PI*v),2.0) + pow(sin(F*uu)*sin(PI*v),2.0)) + 1e-3;
    float dc = abs(aL-thr)/(gradL*eg);
    float idx = fract(lobeI/4.0);
    col += palette(lobeI*0.2+0.5+t*0.02)*(smoothstep(1.6,0.5,dc)+gl*exp(-dc*0.15))*0.8*chaseKey(idx)*p_l_lines;
    cov = max(cov, smoothstep(3.0,0.5,dc)*p_l_lines);
  }

  // video underlay ("machine 1"): luma-keyed, the live layers fly over it
  if (p_l_video > 0.0 && hasVideo > 0.5) {
    vec3 vc = vidSample(g);
    float lum = dot(vc, vec3(.299,.587,.114));
    float k0 = p_luma_key*0.7;
    float key = smoothstep(k0, k0+0.06+0.25*p_luma_key, lum);
    float over = clamp(max(cov, max3(col)*1.5), 0.0, 1.0);
    col += vc*key*chaseKey(fract(lum*0.9))*(1.0-0.85*over)*p_l_video;
    cov = max(cov, key*lum*p_l_video);
  }
  return col;
}

// mandala lattice point from log-radius u and angle th (mirror-folded, with the symmetric swim warp)
vec2 mandalaG(float u, float th, float rr){
  float hs = G_hs;
  float v = abs(fract(th/(2.0*hs))-0.5)*2.0;
  vec2 pf = rr*vec2(cos(v*hs), sin(v*hs));
  vec2 w = fbm2(pf*3.2+vec2(swN[1],1.7));
  w = fbm2(pf*2.2+1.3*w+vec2(-swN[2],4.1));
  float amp = (0.035+0.05*mid+0.035*calm)*p_swim/0.3;
  u += w.x*amp;
  v = abs(mod((v*hs+w.y*amp)/hs+1.0,2.0)-1.0);
  return vec2(u, v*hs);
}
void setMandala(float folds, float cell){
  G_hs = PI/folds; G_F = folds;
  G_nb = max(2.0, floor(G_hs/cell+0.5)); G_cw = G_hs/G_nb;
  G_kl = G_hs/max(1.0, floor(G_hs/(cell*1.6)+0.5));
  G_vs = 0.5/G_hs;
  G_ts = max(1.0, floor(2.0*G_hs+0.5))/(2.0*G_hs);                   // AI texture: whole repeats per lattice period
  G_ar = G_hs*exp2(floor(log2(3.76/G_hs)+0.5))/8.0;                  // arc ring ratio (log): 8 rings = 2^j half-wedges
}
void setFlat(){
  G_hs = HSF; G_F = 8.0; G_nb = 7.0; G_cw = HSF/7.0; G_kl = HSF/4.0; G_vs = 0.2/HSF;
  G_ts = 0.5/HSF; G_ar = 0.47;
}
// swim: domain warp of a flat lattice point
vec2 flatWarp(vec2 q){
  float amp = (0.035+0.05*mid+0.035*calm)*p_swim/0.3*1.6;
  vec2 w = fbm2(q*0.9+vec2(swN[0],1.7));
  return q + fbm2(q*0.7+1.3*w+vec2(-swN[3],4.1))*amp;
}
// flat lattice point: rotate to the crawl direction, scale by density, swim
vec2 flatQ(vec2 p, float dscale){ return flatWarp(rot(-p_flow_dir*TAU)*p*SFLAT*dscale); }

// flower-of-life arcs, mandala: self-similar circle pairs so zoom never runs out of them
vec3 arcsMandala(float rr, float vhs, float px){
  float hs = G_hs;
  vec2 q = rr*vec2(cos(vhs), sin(vhs)), ax = vec2(cos(hs), sin(hs));
  float gl = 0.10*(1.0+1.5*bass), m = floor(log(rr/0.115)/G_ar);
  vec3 c = vec3(0);
  for (int i=-1;i<=2;i++){
    float k = m+float(i);
    float cr = 0.115*exp(G_ar*k)*(1.0+0.06*sin(swS[3]+k*1.5708)+0.05*breath);
    float d = min(abs(length(q-vec2(cr,0))-cr), abs(length(q-ax*cr)-cr))/px;
    float idx = fract(k*0.25 + vhs/hs*0.25);
    c += palette(k*0.25+t*0.03+chaseCol)*(smoothstep(1.7,0.6,d)+gl*exp(-d*0.10))*chaseKey(idx);
  }
  return c;
}
// flower-of-life, flat: circles of radius a on a triangular lattice of spacing a
vec3 arcsFlat(vec2 g, float eg){
  float a = G_hs, d = 1e3, kk = 0.0;
  vec2 s = vec2(g.x - g.y*0.57735, g.y*1.1547)/a, id = floor(s);
  for (int j=-1;j<=2;j++) for (int i=-1;i<=2;i++){
    vec2 c = id + vec2(i,j);
    float di = abs(length(g - vec2(c.x + c.y*0.5, c.y*0.8660254)*a) - a);
    if (di < d){ d = di; kk = c.x + c.y*0.5; }
  }
  d /= eg;
  float gl = 0.10*(1.0+1.5*bass), idx = fract(kk/8.0);
  return palette(idx*0.6+t*0.03+chaseCol)*(smoothstep(1.7,0.6,d)+gl*exp(-d*0.30))*chaseKey(idx);
}

// ---- serpents: each crosses the frame on a gently bent, meandering track; the CPU (Phases) lays a fresh track
// for every crossing and starts the next crossing as soon as the head has left, so a head is on screen nearly
// all the time. The body is a tapered band cut into segments with stepped-diamond markings; the chase runs
// along the segment index.
vec4 serpents(vec2 p){
  float n = floor(p_serpent_count+0.5), segs = floor(p_serpent_segments+0.5);
  float w0 = mix(0.020, 0.085, p_serpent_size), len = mix(0.75, 1.9, p_serpent_size);
  float A = 0.02+0.13*p_serpent_wiggle, kx = 5.0+7.0*p_serpent_wiggle;
  float reach = 1.6*A + 1.5*w0 + 2.0*pxh;
  vec3 acc = vec3(0); float cov = 0.0;
  for (int j=0;j<16;j++){
    int i = j/2; float fi = float(i);
    if (fi >= n) break;
    bool cur = (j-2*i) == 1;                                         // the previous crossing's tail first, then the current one
    vec4 T = cur ? serpA[i] : serpC[i];
    float seed = cur ? serpB[i].z : serpB[i].w;
    vec2 ax = vec2(cos(T.x), sin(T.x));
    float x = dot(p,ax);
    float sr = (T.w - x)/len;                                        // distance behind the head, in body lengths
    if (sr < -0.02 || sr > 1.0) continue;
    float y = dot(p,vec2(-ax.y,ax.x)) - T.y - T.z*x*x;
    if (abs(y) > reach) continue;
    float a1 = kx*x - serpB[i].x + seed*9.0, a2 = 0.41*kx*x + serpB[i].y + seed*5.0;
    float yc = A*sin(a1) + 0.6*A*sin(a2);
    float sl = A*kx*cos(a1) + 0.6*A*0.41*kx*cos(a2) + 2.0*T.z*x;
    float b = (y-yc)/sqrt(1.0+sl*sl);
    float w = w0*pow(max(1.0-sr, 0.0), 0.55)*(0.55+0.45*smoothstep(0.0,0.10,sr))*(1.0+0.45*exp(-pow((sr-0.035)/0.03,2.0)));
    w *= smoothstep(-0.02, 0.012, sr);
    float body = aa(abs(b)-w, pxh);
    if (body <= 0.0) continue;
    float bn = b/max(w,1e-4), sg = sr*segs, a = fract(sg), seg = floor(sg);
    float e = pxh/max(w,1e-4);
    float ea = e*w*segs/len;                                         // pixel size in segment units
    float gap = smoothstep(0.0, 0.10+ea, min(a, 1.0-a));             // dark joint between scales
    float rim = aa(abs(abs(bn)-0.86)-0.07, e);
    float dm = abs(a-0.5)*1.7 + abs(bn);                             // diamond coordinate
    float dia = aa(abs(dm-0.72)-0.07, e) + aa(dm-0.30, e)*0.9;       // diamond outline + stepped core
    float st2 = aa(abs(fract(dm*3.0+0.5)-0.5)/3.0-0.02, e)*step(dm, 0.62)*0.35;
    float idx = fract(seg/segs*2.0 + fi*0.17);
    vec3 pc = palette(seg/segs*0.9 + fi*0.21 + t*0.02);
    vec3 c = (pc*(rim + 0.12) + palette(seg/segs*0.9 + fi*0.21 + 0.4)*(dia+st2))*gap*chaseKey(idx);
    float eye = smoothstep(0.30, 0.12, length(vec2((sr-0.03)*len/w0*1.2, abs(bn)-0.45)))*step(sr, 0.09);
    c += vec3(1.0,0.95,0.8)*eye;
    acc = acc*(1.0-body) + c*body;
    cov = max(cov, body);
  }
  return vec4(acc, cov);
}

vec2 sway(float s){ return vec2(sin(s*0.20)+0.5*sin(s*0.46+1.3), cos(s*0.18)+0.5*sin(s*0.40)); }   // period 100 PI

void main(){
  vec2 asp = vec2(aspect, 1.0);
  vec2 p0 = (v_uv-0.5)*asp, p = p0;
  float sym = clamp(p_symmetry, 0.0, 1.0);
  float tmode = floor(p_travel_mode+0.5), smode = floor(p_serpent_mode+0.5);
  float ps = pxh;                                                    // content units per pixel, tracked through the chain

  // ---- ripples: expanding, decaying rings that displace everything
  float rglow = 0.0, rrate = mix(0.35, 4.0, p_ripple_decay);
  for (int i=0;i<8;i++){
    vec3 R = rip[i];
    if (R.z < 0.0) continue;
    vec2 dv = p0 - (R.xy-0.5)*asp; float d = length(dv)+1e-5;
    float x = d - R.z*0.55, w = 0.05+0.04*R.z;
    float en = exp(-x*x/(w*w))*exp(-R.z*rrate)*smoothstep(0.0, 0.08, R.z)*p_ripple;
    p += dv/d*sin(x*38.0)*en*0.05;
    rglow += en*(0.5+0.5*cos(x*38.0));
  }
  vec2 pscr = p;                                                     // ripple-displaced screen space (serpents, fractal)

  // ---- serpentine warp: segmented bands sliding sideways like a snake's body
  if (p_serpent > 0.0 && smode > 0.5) {
    float n = floor(p_serpent_segments+0.5)*0.5;
    float band = p.y*n, bq = floor(band)+0.5;
    float bs = mix(bq, band, 0.2)/n;
    float A = p_serpent*(0.025+0.10*p_serpent_wiggle);
    p.x += A*(sin(bs*TAU*(0.8+1.6*p_serpent_wiggle) - serpW[0]) + 0.4*sin(bs*TAU*0.45 + serpW[1]));
  }
  // ---- wave: sinusoidal reshaping
  p += p_wave*0.07*vec2(sin(p.y*7.0+swS[0]), sin(p.x*5.5-swS[1]+1.3));
  // slow asymmetric drift so nothing is a perfect stamp
  p += fbm2(p*1.7+vec2(0.,swN[0]))*(0.010+0.018*mid+0.012*calm*breath);
  // ---- camera: pan, tunnel sway, rotate, punch
  p -= vec2(p_pan_x*aspect, p_pan_y)*0.5;
  if (tmode > 0.5 && tmode < 2.5) p += p_tunnel_curve*0.07*sway(trSway+3.0);
  p = rot(rotPh + 0.2*sin(swS[2]))*p;
  float punch = 1.0 + 0.07*bass + 0.05*(breath-0.5)*(0.4+calm) + 0.30*p_punch;
  p /= punch; ps /= punch;
  // ---- bulge / pinch lens, twirl
  float r0 = length(p)+1e-5;
  float bf = pow(r0/0.45+0.02, p_bulge*0.6);
  p *= bf; ps *= bf*(1.0+0.5*abs(p_bulge));
  float tk = p_twirl*1.1 + (tmode > 2.5 ? 0.8 : 0.0);               // Droste mode adds a standing log spiral (mandala)
  float soft = 0.15*(1.0-sym)+1e-4;
  vec2 pt = rot(tk*log(r0*bf+soft))*p;                               // mandala gets the spiral; flat Droste has its own
  vec2 pf_ = rot(p_twirl*1.1*log(r0*bf+soft))*p;
  ps *= 1.0+0.35*abs(tk);

  float cell = 0.12*pow(0.25, p_density);                            // lattice cell size from density
  float zl = p_zoom*3.0;                                             // static zoom in octaves
  float fog = clamp(p_depth_fog, 0.0, 1.0);
  vec3 col = vec3(0); float cov = 0.0;
  G_sparse = 0.0;

  // =========================================================== mandala topology
  if (sym > 0.0) {
    vec3 cm = vec3(0); float cvm = 0.0;
    float folds = floor(clamp(p_folds, 2.0, 24.0)+0.5);
    setMandala(folds, cell);
    G_thr = 0.20 + 0.10*calm*breath + 0.20*smoothstep(0.0,0.3,p_l_kene);
    G_cofs = 0.0;
    float rs = length(pt)+1e-5, th = atan(pt.y, pt.x);
    float inner = smoothstep(0.035, 0.10, rs);
    if (tmode > 0.5 && tmode < 1.5) {
      // 3D tube: lattice x = depth along the tube, y = angle around it
      float z0 = 0.5/rs;
      vec2 pc = pt - p_tunnel_curve*0.16*(sway(z0+trSway) - sway(trSway))*min(z0, 8.0)/8.0;
      rs = length(pc)+1e-5; th = atan(pc.y, pc.x);
      float z = 0.5/rs;
      vec2 g = mandalaG(z + trT, th, rs); g.x -= flowM;
      float eg = ps/rs*sqrt(max(z*2.0, 1.0));
      float fg = exp(-(z-0.5)*mix(0.03, 0.85, fog))*smoothstep(0.02, 0.07, rs);
      cm = shade(g, eg, cvm)*fg; cvm *= fg;
    } else if (tmode > 1.5 && tmode < 2.5) {
      // depth layers: mandala discs flying toward the viewer, far -> near, recycling
      float base = trZ*0.75, bfr = fract(base)/3.0, bid = floor(base);
      for (int k=0;k<3;k++){
        float fk = bfr + float(k)/3.0, id = mod(bid - float(k), 64.0);
        float s = exp2((fk-0.5)*4.4);
        float al = smoothstep(0.0, 0.15, fk)*smoothstep(1.0, 0.86, fk)*mix(1.0, 0.12+0.88*smoothstep(0.1, 0.8, fk), fog);
        vec2 pk = pt - p_tunnel_curve*0.10*sway(id*3.7+1.0)*(s-1.0)*0.3;
        float rk = length(pk)/s+1e-5;
        vec2 g = mandalaG(log(rk)+id*5.731, atan(pk.y, pk.x)+id*0.9, rk); g.x -= flowM;
        float Rs = 0.475+0.03*cos(PI*g.y/G_hs)+0.015*breath;
        float m = smoothstep(Rs, Rs-0.05, rk)*smoothstep(0.035, 0.10, rk)*al;
        G_cofs = fk; G_sparse = 1.0; float cv; vec3 c = shade(g, ps/(rk*s), cv);
        cm = cm*(1.0-cv*m) + c*m; cvm = max(cvm*(1.0-cv*m), cv*m);
      }
      G_cofs = 0.0; G_sparse = 0.0;
    } else {
      // log-polar: zoom and "zoom tunnel" / Droste travel are a shift along u, so detail never runs out
      float act = (tmode < 0.5 || tmode > 2.5) ? travelAct : 0.0;
      float scale = exp(zl*LN2);
      float u = log(rs) - zl*LN2 - ((tmode < 0.5 || tmode > 2.5) ? trU : 0.0);
      float rw = exp(mod(u, 8.0*G_ar));                           // self-similar radius for the arcs (period 8 rings)
      vec2 g = mandalaG(u, th, rs); g.x -= flowM;
      float v = g.y/G_hs;
      float es = clamp(scale, 0.45, 4.0)*mix(1.0, 4.0, act);        // the disc grows to fill the frame when flying in
      float Rs = (0.475+0.03*cos(PI*v)+0.015*breath+0.006*sin(TAU*bar_phase))*es;   // breathes with the bar
      float env = smoothstep(Rs, Rs-0.05*es, rs);
      float fg = mix(1.0, smoothstep(0.03, 0.50, rs), fog*act);
      cm = shade(g, ps/rs, cvm)*env*inner*fg; cvm *= env*inner*fg;
      if (p_l_lines > 0.0) {
        float mk = smoothstep(0.74*es, 0.50*es, rs)*inner*fg;
        cm += arcsMandala(rw, g.y, ps/rs*rw)*p_l_lines*mk;
      }
      // core jewel hides the log-polar pole
      float cg = exp(-rs*rs/0.0016)*(0.65+0.6*bass+0.25*calm*breath) + 0.30*pow(1.0-v,6.0)*exp(-rs*16.0)*min(1.0, 8.0/folds);
      cm += (palette(t*0.04+0.1)*cg + vec3(1.0,0.95,0.85)*exp(-rs*rs/0.00025)*0.35)*(1.0-0.6*fog*act);
    }
    col += cm*sym; cov += cvm*sym;
  }

  // =========================================================== flat topology
  if (sym < 1.0) {
    vec3 cf = vec3(0); float cvf = 0.0;
    setFlat();
    G_thr = 0.20 + 0.10*calm*breath + 0.20*smoothstep(0.0,0.3,p_l_kene);
    G_cofs = 0.0;
    float dsc = 0.06/cell;
    if (tmode > 0.5 && tmode < 1.5) {
      // ground + sky planes in perspective, flying toward the horizon
      float yh = pf_.y, ay = max(abs(yh), 1e-3);
      float z = 0.32/ay;
      float bend = p_tunnel_curve*0.5*(sway(z*0.6+trSway).x - sway(trSway).x);
      vec2 g = flatWarp(vec2(z*1.6, pf_.x*z*1.6 + bend + step(0.0, yh)*7.31)*dsc);
      g.x += trG - flowF;
      float eg = ps*z*1.6*dsc*sqrt(max(z/0.32, 1.0));
      float fg = exp(-(z-0.64)*mix(0.03, 0.75, fog))*smoothstep(0.015, 0.06, ay);
      cf = shade(g, eg, cvf)*fg; cvf *= fg;
      if (p_l_lines > 0.0) cf += arcsFlat(g, eg)*p_l_lines*fg*smoothstep(0.25, 0.08, eg/G_hs);
      // horizon: where the planes get too fine to draw, a vanishing-point bloom and a low fog bank take over
      // (no dead stripe). The bank is a little deeper with fog and its strata roll toward the viewer at flight speed.
      float hw = 0.028+0.032*fog;
      float bank = exp(-ay/hw), core = exp(-ay/0.006), vp = exp(-pf_.x*pf_.x*2.4);
      float sun = exp(-(pf_.x*pf_.x*7.0 + yh*yh*70.0));
      float strata = 0.72+0.28*sin(4.0*log(z) + trSway*2.0);
      vec3 hc = palette(0.10+t*0.04+0.22*abs(pf_.x));
      float hg = (bank*strata*0.70*(0.10+0.90*vp) + core*0.70*(0.25+0.75*vp) + 0.45*sun)*(0.80+0.45*bass+0.20*calm*breath);
      hg = max(hg-0.045, 0.0)*1.05;                                  // knee: the haze ends in true black
      cf += mix(hc, vec3(1.0,0.95,0.85), 0.40*max(core*vp, sun*sun))*hg;
      cvf = max(cvf, hg);
    } else if (tmode > 1.5 && tmode < 2.5) {
      // depth layers: parallax planes of the field flying toward the viewer
      float base = trZ*0.75, bfr = fract(base)/3.0, bid = floor(base);
      for (int k=0;k<3;k++){
        float fk = bfr + float(k)/3.0, id = mod(bid - float(k), 64.0);
        float s = exp2((fk-0.62)*3.4);
        float al = smoothstep(0.0, 0.15, fk)*smoothstep(1.0, 0.86, fk)*mix(1.0, 0.12+0.88*smoothstep(0.1, 0.8, fk), fog);
        vec2 pk = pf_ - p_tunnel_curve*0.10*sway(id*3.7+1.0)*(s-1.0)*0.3;
        vec2 g = flatQ(pk/s, dsc) + id*vec2(17.13, 9.71); g.x -= flowF;
        float eg = ps*SFLAT*dsc/s;
        G_cofs = fk; G_sparse = 1.0; float cv; vec3 c = shade(g, eg, cv);
        if (p_l_lines > 0.0) c += arcsFlat(g, eg)*p_l_lines;
        cf = cf*(1.0-cv*al) + c*al; cvf = max(cvf*(1.0-cv*al), cv*al);
      }
      G_cofs = 0.0; G_sparse = 0.0;
    } else {
      // endless zoom: two octaves of the field cross-fade (seamless). Droste mode wraps it into a log spiral.
      vec2 zq = pf_; float L, fgf = 1.0;
      if (tmode > 2.5) {
        float lr = log(r0*bf), th = atan(p.y, p.x), cc = LN2/TAU;
        vec2 w = vec2(lr - th*cc, th + lr*cc + p_twirl*1.1*lr);      // conformal: one turn = one octave
        w.x -= (zl + trZ)*LN2;
        float Lr = w.x/LN2 + 1.152;                                  // octave level (annulus 0.45 .. 0.9)
        L = Lr - floor(Lr);
        float mag = 0.45*exp2(L);
        zq = mag*vec2(cos(w.y), sin(w.y));
        float eg0 = ps*SFLAT*dsc*mag/(r0*bf);
        fgf = mix(1.0, smoothstep(0.03, 0.45, r0), fog*travelAct)*smoothstep(0.012, 0.05, r0);
        vec2 gA = flatQ(zq, dsc); gA.x -= flowF;
        float wB = smoothstep(0.80, 1.0, L);
        float cvA; cf = shade(gA, eg0, cvA);
        if (p_l_lines > 0.0) cf += arcsFlat(gA, eg0)*p_l_lines;
        cvf = cvA;
        if (wB > 0.0) {
          vec2 gB = flatQ(zq*0.5, dsc); gB.x -= flowF;
          float cvB; vec3 cB = shade(gB, eg0*0.5, cvB);
          if (p_l_lines > 0.0) cB += arcsFlat(gB, eg0*0.5)*p_l_lines;
          cf = mix(cf, cB, wB); cvf = mix(cvA, cvB, wB);
        }
        cf *= fgf; cvf *= fgf;
      } else {
        float Z = zl + (tmode < 0.5 ? trZ : 0.0);
        float kq = floor(Z), fz = Z-kq, wB = smoothstep(0.80, 1.0, fz);
        vec2 q = flatQ(pf_, dsc);
        float mag = exp2(fz), eg0 = ps*SFLAT*dsc;
        fgf = mix(1.0, smoothstep(0.03, 0.50, r0), fog*(tmode < 0.5 ? travelAct : 0.0));
        vec2 gA = q/mag + mod(kq, 64.0)*vec2(17.13, 9.71); gA.x -= flowF;
        float cvA; cf = shade(gA, eg0/mag, cvA);
        if (p_l_lines > 0.0) cf += arcsFlat(gA, eg0/mag)*p_l_lines;
        cvf = cvA;
        if (wB > 0.0) {
          vec2 gB = q/mag*2.0 + mod(kq+1.0, 64.0)*vec2(17.13, 9.71); gB.x -= flowF;
          float cvB; vec3 cB = shade(gB, eg0/mag*2.0, cvB);
          if (p_l_lines > 0.0) cB += arcsFlat(gB, eg0/mag*2.0)*p_l_lines;
          cf = mix(cf, cB, wB); cvf = mix(cvA, cvB, wB);
        }
        cf *= fgf; cvf *= fgf;
      }
    }
    // slow breathing clouds keep the flat field from reading as wallpaper
    float cl = 0.55+0.45*smoothstep(0.25, 0.60, fbm(p0*1.3+vec2(swN[4], -swN[5])));
    col += cf*cl*(1.0-sym); cov += cvf*(1.0-sym);
  }

  // ---- pure-math fractal pass (already palette-coloured), screen-blended
  if (p_l_fractal > 0.0) {
    vec3 fc = clamp(texture(texF, pscr/asp+0.5).rgb, 0.0, 1.0)*p_l_fractal;
    col = col + fc*(1.0-clamp(col, 0.0, 1.0));
    cov = max(cov, max3(fc));
  }

  // ---- painting reliefs (art_layer.ArtPass): own colours, no palette tint
  if (p_l_art > 0.0) {
    vec3 ac = clamp(texture(texR, pscr/asp+0.5).rgb, 0.0, 1.0)*p_l_art;
    col = col + ac*(1.0-clamp(col, 0.0, 1.0));
    cov = max(cov, max3(ac));
  }

  // ---- serpents slither over everything
  if (p_serpent > 0.0 && smode != 1.0) {
    vec4 s = serpents(pscr);
    col = col*(1.0-0.92*s.a*p_serpent) + s.rgb*p_serpent;
    cov = max(cov, s.a*p_serpent);
  }

  col *= 1.0 + 1.6*min(rglow, 1.5);                                  // light rides the ripple wavefront (geometry only)
  col *= 0.82+0.38*bass+0.12*calm*breath;                            // kick bloom (bounded)
  float lum = dot(col, vec3(.299,.587,.114));
  col = max(mix(vec3(lum), col, p_saturation), 0.0);
  if (p_hue > 0.0) col = max(hueRot(col, p_hue*TAU), 0.0);
  vec2 ed = min(v_uv, 1.0-v_uv); float ew = mix(0.008, 0.035, sym);
  float edge = smoothstep(0.0,ew,ed.x)*smoothstep(0.0,ew,ed.y);       // never show the projector rectangle
  col *= edge;
  col /= max(1.0, max3(col));                                        // hue-preserving clip: no white-out
  if (nanGuard > 0.5 && !(col.r >= 0.0 && col.g >= 0.0 && col.b >= 0.0)) col = vec3(0);   // a NaN would live in the echo forever

  // ---- energy-conserving echo: zoomed/rotated copy that only ever decays (max-blend, clamp, knee to black).
  // Freeze holds it exactly. The echo is shown dimmer than fresh geometry so trails read as an aura.
  float hold = clamp(p_freeze, 0.0, 1.0);
  vec2 q = (v_uv-0.5)*asp; float fr = (0.0015+0.002*mid)*(1.0-hold), z = mix(0.9955-0.004*bass, 1.0, hold);
  q = (rot(fr)*q*z)/asp+0.5;
  bool held = hold > 0.999;                                          // fully frozen: copy the texel, no resampling
  vec3 prev = held ? texelFetch(prevFrame, ivec2(gl_FragCoord.xy), 0).rgb : texture(prevFrame,q).rgb;
  if (nanGuard > 0.5 && !(prev.r >= 0.0 && prev.g >= 0.0 && prev.b >= 0.0)) prev = vec3(0);
  vec3 trail = held ? prev : clamp(hueRot(prev, (0.012+0.035*high)*(1.0-hold)), 0.0, 1.0)*mix(p_trails*0.94, 1.0, hold);
  trail *= mix(smoothstep(0.05, 0.24, max3(trail)), 1.0, hold);
  fbE = vec4(max(col, trail), 1.0);
  vec3 o = max(col, trail*(0.30+0.25*p_trails)*min(p_l_echo/0.6, 1.6));

  // ---- performance stage (display only; the echo buffer never sees it)
  float mx = max3(o);
  float cv = clamp(max(cov*edge, smoothstep(0.02, 0.30, mx)), 0.0, 1.0);
  if (p_invert > 0.0) {                                              // palette-aware: flips brightness inside the geometry only
    vec3 hc = mix(palette(t*0.03+0.3*length(p0)), o/max(mx,1e-3), smoothstep(0.02, 0.15, mx));
    o = mix(o, hc*clamp(cv*0.72-mx*0.8, 0.0, 1.0), p_invert);
  }
  if (p_flash > 0.0) {                                               // additive white/palette bloom
    vec3 tint = mix(palette(t*0.04+0.1), vec3(1.0), 0.55);
    o = o*(1.0+1.8*p_flash) + tint*p_flash*(0.06+0.40*exp(-dot(p0,p0)*5.0))*edge;
    o /= max(1.0, max3(o)*0.92);
  }
  if (p_strobe > 0.0) {                                              // value is clock-gated by the controller: 1 = pulse
    o += p_strobe*(o*1.4 + vec3(0.30)*smoothstep(0.03, 0.30, mx));   // the geometry itself flashes; black stays black
    o /= max(1.0, max3(o)*0.95);
  }
  fbE.a = max3(o);                                                   // pre-limiter level, read by the brightness meter
  if (limiter > 0.0) {                                               // CPU-metered ceiling: lift the black knee, pull the gain
    float kn = 0.34*limiter;
    o = max(o-kn, 0.0)/(1.0-kn)*(1.0-0.5*limiter);
  }
  o *= clamp(p_master, 0.0, 1.0)*(1.0-clamp(p_blackout, 0.0, 1.0));
  f = nanGuard > 0.5 ? vec4(clamp(o-0.004, 0.0, 1.0), 1.0) : vec4(o, 1.0);   // nanGuard 0 = raw value (NaN tests)
}
"""

OUT_FS = """
#version 330
uniform sampler2D scene; uniform mat3 Hinv; uniform float grid;
uniform vec2 corners[4];
in vec2 v_uv; out vec4 f;
void main(){
  vec3 q = Hinv*vec3(v_uv,1.0); vec2 c = q.xy/q.z;
  vec3 col = vec3(0);
  if (all(greaterThanEqual(c,vec2(0))) && all(lessThanEqual(c,vec2(1)))) {
    col = texture(scene,c).rgb;
    if (grid > 0.5) { vec2 g = abs(fract(c*8.0)-0.5); if (min(g.x,g.y) > 0.48) col = vec3(1); }
  }
  if (grid > 0.5) for (int i=0;i<4;i++) if (distance(v_uv,corners[i]) < 0.012) col = vec3(1,0.2,0.6);
  f = vec4(col,1.0);
}
"""

# brightness meter: each output pixel box-filters a 20x20 block of the pre-limiter level (echo buffer alpha)
# with a 4x4 grid of bilinear taps -> r = mean level, g = fraction of taps that are near-black
METER_FS = """
#version 330
uniform sampler2D src; uniform vec2 texel;
in vec2 v_uv; out vec4 f;
void main(){
  float s = 0.0, b = 0.0;
  for (int j=0;j<4;j++) for (int i=0;i<4;i++){
    float a = texture(src, v_uv + (vec2(i,j)-1.5)*texel*5.0).a;
    if (!(a >= 0.0)) a = 0.0;
    s += min(a, 1.0); b += step(a, 0.04);
  }
  f = vec4(s/16.0, b/16.0, 0.0, 1.0);
}
"""

UNIT_SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def corners_ok(corners):
    """True for 4 finite [u, v] points forming a strictly convex quad (safe to invert)."""
    try:
        if len(corners) != 4:
            return False
        pts = [(float(c[0]), float(c[1])) for c in corners if len(c) == 2]
    except (TypeError, ValueError, IndexError):
        return False
    return len(pts) == 4 and all(math.isfinite(x) and math.isfinite(y) for x, y in pts) and quad_ok(pts)


def load_corners(path):
    """Corner pin from mapping.json; the unit square if the file is missing, malformed or degenerate."""
    try:
        data = json.loads(Path(path).read_text())
        if corners_ok(data):
            return [[float(c[0]), float(c[1])] for c in data]
        print(f"[map] {path} is not a valid 4-corner quad -> using the full frame")
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as ex:
        print(f"[map] could not read {path} ({ex}) -> using the full frame")
    return [list(map(float, c)) for c in UNIT_SQ]


def inverse_homography(corners, last_good=None):
    """Inverse corner-pin matrix; `last_good` (or identity) if the quad is degenerate. Never raises."""
    try:
        if not corners_ok(corners):
            return last_good if last_good is not None else np.eye(3)
        Hinv = np.linalg.inv(homography(UNIT_SQ, corners))
        if np.isfinite(Hinv).all():
            return Hinv
    except (np.linalg.LinAlgError, ValueError, TypeError, IndexError):
        pass
    return last_good if last_good is not None else np.eye(3)



def gallery_audio(t):
    """Synthetic set for --gallery: 4 s of 128 BPM kick + off-beat hats, then a bass-less breakdown."""
    if t < 4.0:
        beat = (t * 128 / 60) % 1.0
        return np.array([math.exp(-beat * 6), 0.45 + 0.25 * math.sin(t * 1.3), 0.9 * math.exp(-((beat - 0.5) % 1.0) * 10)])
    return np.array([0.0, 0.25 + 0.15 * math.sin(t * 0.5), 0.15 + 0.10 * math.sin(t * 2.1) ** 2])


GALLERY_SHOTS = {143: "kick", 155: "offbeat", 420: "breakdown"}
KICK = 143                                                   # default gallery shot: just after a kick


def gallery_rows():
    """[(row label, [(column label, {param: value}, {frame: [action, ...]}, shot frame)])] for --gallery.
    Actions: ("trigger", name, on) | ("ripple", x, y) | ("set", name, value)."""
    preset = lambda name: dict(zip(PRESET_KEYS, MODE_W[MODES.index(name)]))
    K, flat, ALL = preset("kene"), dict(symmetry=0.0), preset("all")
    job = lambda label, sets, script=None, shot=KICK: (label, sets, script or {}, shot)
    rows = [
        ("kene flat", [job(PALETTES[i][0], {**K, **flat, "palette": i}) for i in range(5)]),
        ("kene mandala", [job(PALETTES[i][0], {**K, "palette": i}) for i in range(5)]),
        ("kene variations", [
            job("flat dense", {**K, **flat, "density": 0.85}),
            job("flat sparse", {**K, **flat, "density": 0.15, "palette": 3}),
            job("flat 45deg breakdown", {**K, **flat, "flow_dir": 0.125, "palette": 1}, shot=420),
            job("mandala 6 fold", {**K, "folds": 6, "palette": 2}),
            job("mandala 16 fold", {**K, "folds": 16, "palette": 3}),
            job("half symmetry", {**K, "symmetry": 0.5, "palette": 4})]),
        ("layers mandala", [job(m, {**preset(m), "palette": i % 5}) for i, m in enumerate(MODES[1:])]),
        ("layers flat", [job(m, {**preset(m), **flat, "palette": (i + 2) % 5}) for i, m in enumerate(MODES[1:])]),
        ("zoom mandala (all)", [job(f"zoom {z:+.1f}", {**ALL, "zoom": z}) for z in (-1.0, -0.5, 0.0, 0.5, 1.0)]),
        ("zoom flat (kene)", [job(f"zoom {z:+.2f}", {**K, **flat, "zoom": z, "palette": 3}) for z in (-1.0, -0.3, 0.0, 0.3, 1.0)]),
        ("contort", [
            job("twirl +0.7", {**ALL, "twirl": 0.7}),
            job("twirl flat", {**K, **flat, "twirl": -0.8, "palette": 1}),
            job("bulge +1", {**K, **flat, "bulge": 1.0, "palette": 2}),
            job("pinch -1", {**ALL, "bulge": -1.0, "palette": 2}),
            job("wave 1", {**K, **flat, "wave": 1.0, "palette": 3}),
            job("swim 1", {**K, "swim": 1.0, "palette": 4})]),
        ("ripples", [
            job("centre", {**K, "ripple": 1.0}, {110: [("ripple", 0.5, 0.5)]}),
            job("flat off-centre", {**K, **flat, "ripple": 1.0, "palette": 1}, {105: [("ripple", 0.3, 0.6)]}),
            job("flat x3", {**K, **flat, "ripple": 0.8, "palette": 2},
                {95: [("ripple", 0.25, 0.3)], 110: [("ripple", 0.7, 0.65)], 125: [("ripple", 0.5, 0.5)]}),
            job("all, slow decay", {**ALL, "ripple": 1.0, "ripple_decay": 0.1, "palette": 3},
                {80: [("ripple", 0.5, 0.5)], 115: [("ripple", 0.35, 0.4)]}),
            job("beads", {**preset("beads"), **flat, "ripple": 1.0, "palette": 4}, {112: [("ripple", 0.6, 0.45)]})]),
        ("triggers", [
            job("flash", {**ALL}, {139: [("trigger", "flash", True)]}),
            job("strobe (open)", {**ALL, "palette": 1}, {100: [("trigger", "strobe", True)]}, shot=147),
            job("invert", {**ALL, "palette": 2}, {120: [("trigger", "invert", True)]}),
            job("invert flat kene", {**K, **flat, "palette": 3}, {120: [("trigger", "invert", True)]}),
            job("freeze @100", {**ALL, "palette": 3}, {100: [("trigger", "freeze", True)]}),
            job("blackout +2 frames", {**ALL, "palette": 4}, {141: [("trigger", "blackout", True)]}),
            job("punch", {**K, "palette": 4}, {136: [("trigger", "punch", True)]})]),
        ("chase (amount 1, narrow)", [
            job("single", {**preset("kene+beads"), "chase": 1.0, "chase_width": 0.12, "chase_mode": 0}),
            job("dual", {**preset("kene+beads"), "chase": 1.0, "chase_width": 0.12, "chase_mode": 1, "palette": 1}),
            job("counter", {**preset("kene+beads"), "chase": 1.0, "chase_width": 0.12, "chase_mode": 2, "palette": 2}),
            job("flat single", {**K, **flat, "chase": 1.0, "chase_width": 0.12, "chase_mode": 0, "palette": 3}),
            job("flat counter", {**K, **flat, "chase": 1.0, "chase_width": 0.12, "chase_mode": 2, "palette": 4}),
            job("lines dual wide", {**preset("lines"), "chase": 1.0, "chase_width": 0.4, "chase_mode": 1}),
            job("chase 0", {**preset("kene+beads"), "chase": 0.0, "palette": 1})]),
        ("travel mandala", [
            job("0 zoom tunnel", {**ALL, "travel": 0.7, "travel_mode": 0}),
            job("1 tube", {**preset("kene+beads"), "travel": 0.7, "travel_mode": 1, "tunnel_curve": 0.5, "palette": 1}),
            job("1 tube, no fog", {**K, "travel": 0.7, "travel_mode": 1, "depth_fog": 0.0, "palette": 3}),
            job("2 depth layers", {**preset("kene+beads"), "travel": 0.7, "travel_mode": 2, "palette": 2}),
            job("3 droste", {**ALL, "travel": 0.7, "travel_mode": 3, "palette": 4}),
            job("0 backward, fog 1", {**K, "travel": -0.7, "travel_mode": 0, "depth_fog": 1.0, "palette": 3})]),
        ("travel flat", [
            job("0 endless zoom", {**K, **flat, "travel": 0.7, "travel_mode": 0}),
            job("1 ground plane", {**K, **flat, "travel": 0.7, "travel_mode": 1, "tunnel_curve": 0.5, "palette": 1}),
            job("1 plane, beads", {**preset("kene+beads"), **flat, "travel": 0.7, "travel_mode": 1, "palette": 3}),
            job("2 depth layers", {**K, **flat, "travel": 0.7, "travel_mode": 2, "palette": 2}),
            job("2 layers, beads", {**preset("beads"), **flat, "travel": 0.7, "travel_mode": 2, "palette": 0}),
            job("3 droste", {**K, **flat, "travel": 0.7, "travel_mode": 3, "palette": 4})]),
        ("serpents", [
            job("0 serpents on black", {**K, "l_kene": 0.0, "serpent": 1.0, "serpent_mode": 0}),
            job("0 x8 small", {**K, "l_kene": 0.25, **flat, "serpent": 1.0, "serpent_count": 8, "serpent_size": 0.2, "palette": 1}),
            job("0 big, chase 1", {**K, "l_kene": 0.0, "serpent": 1.0, "serpent_size": 0.9, "serpent_segments": 40,
                                   "chase": 1.0, "chase_width": 0.3, "palette": 3}),
            job("1 serpentine warp", {**K, **flat, "serpent": 1.0, "serpent_mode": 1, "palette": 2}),
            job("1 warp mandala", {**ALL, "serpent": 1.0, "serpent_mode": 1, "serpent_segments": 12, "palette": 4}),
            job("2 both over all", {**ALL, "serpent": 1.0, "serpent_mode": 2, "palette": 0})]),
        ("video loop (test clip)", [
            job("video flat", {**K, "l_kene": 0.0, **flat, "l_video": 1.0}),
            job("video mandala", {**K, "l_kene": 0.0, "l_video": 1.0, "palette": 1}),
            job("kene over video", {**K, **flat, "l_video": 1.0, "palette": 2}),
            job("all over video", {**ALL, "l_video": 0.8, "palette": 3}),
            job("luma key 0.8", {**K, "l_kene": 0.0, **flat, "l_video": 1.0, "luma_key": 0.8}),
            job("video chase 1", {**K, "l_kene": 0.0, "l_video": 1.0, "chase": 1.0, "chase_width": 0.2})]),
        ("time (defaults)", [
            job("all kick", {}, shot=143), job("all offbeat", {}, shot=155), job("all breakdown", {}, shot=420),
            job("flat all kick", {**flat, "palette": 1}, shot=143),
            job("flat kene breakdown", {**K, **flat, "palette": 2}, shot=420),
            job("kene mandala breakdown", {**K, "palette": 3}, shot=420)]),
    ]
    return rows


def frame_stats(img):
    """(fraction near-black, mean 0..255, fraction blown to near-white, fraction of lit pixels that are saturated colour)."""
    a = np.asarray(img).astype(np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    lit = mx > 40
    sat = ((mx - mn) / np.maximum(mx, 1) > 0.5) & lit
    return float((mx < 8).mean()), float(a.mean()), float((mn > 235).mean()), float(sat.sum() / max(1, lit.sum()))


def contact_sheet(rows, out_path, thumb=(320, 200)):
    """rows: [(row label, [(column label, PIL image)])] -> one labelled sheet."""
    tw, th = thumb
    left, cap = 170, 16
    ncol = max(len(r[1]) for r in rows)
    sheet = Image.new("RGB", (left + tw * ncol, (th + cap) * len(rows)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for ri, (label, cells) in enumerate(rows):
        y = ri * (th + cap)
        dr.text((6, y + cap + th // 2 - 6), label, fill=(210, 210, 210))
        dr.line([(0, y), (sheet.width, y)], fill=(45, 45, 45))
        for ci, (clabel, img) in enumerate(cells):
            sheet.paste(img.resize(thumb, Image.LANCZOS), (left + ci * tw, y + cap))
            dr.text((left + ci * tw + 5, y + 3), clabel, fill=(170, 170, 170))
    sheet.save(out_path)


def palette_stops(value):
    """5 RGB stops for a fractional palette value: 1.5 = halfway between palettes 1 and 2 (wraps 4 -> 0)."""
    n = len(PALETTES)
    v = float(value) if math.isfinite(float(value)) else 0.0
    i = int(math.floor(v)) % n
    k = v - math.floor(v)
    a, b = np.array(PALETTES[i][1], dtype="f4"), np.array(PALETTES[(i + 1) % n][1], dtype="f4")
    return a + (b - a) * k


def load_remote_token(path, given=None):
    """Persistent panel token: `given`, else the single line in `path`, else a new one saved there.
    Never raises; an unusable file just means a fresh random token for this run."""
    if given:
        return str(given)
    try:
        tok = Path(path).read_text(encoding="utf-8").strip()
        if tok and len(tok) <= 128 and all(c.isalnum() or c in "-_" for c in tok):
            return tok
    except (OSError, UnicodeError):
        pass
    tok = secrets.token_urlsafe(6)
    try:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(tok + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
    return tok


_SNAP_SEQ = itertools.count()


def save_snapshot(path, data, size, width=768):
    """Write an RGB8 bottom-up frame to `path` (atomic, downscaled to `width`). Never raises."""
    tmp = None
    try:
        img = Image.frombytes("RGB", size, data).transpose(Image.FLIP_TOP_BOTTOM)
        if size[0] > width:
            img = img.resize((width, max(1, round(size[1] * width / size[0]))), Image.BILINEAR)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{next(_SNAP_SEQ)}.tmp.png")
        img.save(tmp)
        os.replace(tmp, path)
        return True
    except Exception as ex:
        print(f"[snapshot] failed: {ex}")
        try:
            if tmp is not None:
                tmp.unlink()                     # never leave a half-written temp file behind
        except OSError:
            pass
        return False


def make_test_clip(path, size=(320, 200), frames=48, fps=24):
    """Procedural seamless test loop (drifting rings + bars on black) for the gallery / tests. True if written."""
    try:
        import imageio_ffmpeg
        w, h = size
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x, y = (x - w / 2) / h, (y - h / 2) / h
        r, a = np.hypot(x, y), np.arctan2(y, x)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wr = imageio_ffmpeg.write_frames(str(path), size, fps=fps, macro_block_size=8, ffmpeg_log_level="error")
        wr.send(None)
        for i in range(frames):
            ph = 2 * math.pi * i / frames
            rings = np.clip(np.sin(r * 26 - ph * 2) * 1.6 - 0.5, 0, 1)
            petals = np.clip(np.sin(a * 6 + ph) * np.sin(r * 9 + ph) * 1.5 - 0.3, 0, 1)
            img = np.stack([rings * (0.6 + 0.4 * np.sin(ph + r * 4)), petals * 0.9, rings * petals + 0.5 * petals * (r > 0.3)], axis=2)
            wr.send((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
        wr.close()
        return path.exists()
    except Exception as ex:
        print(f"[video] could not write test clip: {ex}")
        return False


class _Decoder(threading.Thread):
    """ffmpeg -> RGB frames into a small preallocated ring buffer, on its own thread. Loops forever."""

    RING = 6

    def __init__(self, path, size):
        super().__init__(daemon=True, name="video-decode")
        self.path, self.size = Path(path), size
        self.nbytes = size[0] * size[1] * 3
        self.ring = [np.zeros(self.nbytes, dtype=np.uint8) for _ in range(self.RING)]
        self.out = np.zeros(self.nbytes, dtype=np.uint8)
        self.head = self.count = 0
        self.cond = threading.Condition()
        self.running, self.failed, self.fps = True, False, 30.0
        self.start()

    def _frames(self, loop_flag):
        import imageio_ffmpeg
        return imageio_ffmpeg.read_frames(str(self.path), pix_fmt="rgb24",
                                          input_params=["-stream_loop", "-1"] if loop_flag else [],
                                          output_params=["-vf", f"scale={self.size[0]}:{self.size[1]},vflip", "-an"])

    def run(self):
        loop_flag, got_any = True, False
        while self.running:
            gen, n = None, 0
            try:
                gen = self._frames(loop_flag)
                meta = next(gen)
                fps = float(meta.get("fps") or 30.0)
                self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
                for fr in gen:
                    if len(fr) != self.nbytes:
                        raise ValueError("unexpected frame size")
                    with self.cond:
                        while self.running and self.count >= self.RING:
                            self.cond.wait(0.1)
                        if not self.running:
                            break
                        self.ring[(self.head + self.count) % self.RING][:] = np.frombuffer(fr, dtype=np.uint8)
                        self.count += 1
                        self.cond.notify_all()
                    n += 1
                    got_any = True
            except Exception as ex:
                if loop_flag and not got_any:
                    loop_flag = False            # some containers reject -stream_loop: reopen at each end instead
                    continue
                if not got_any or n == 0:
                    if self.running:
                        print(f"[video] {self.path.name}: {str(ex).splitlines()[0][:120] if str(ex) else type(ex).__name__}")
                    self.failed = True
                    break
            finally:
                if gen is not None:
                    try:
                        gen.close()              # terminates the ffmpeg child
                    except Exception:
                        pass
            if n == 0 and self.running:
                self.failed = True               # empty / unreadable clip
                break
        with self.cond:
            self.cond.notify_all()

    def take(self, n, wait=0.0):
        """Advance by up to n frames; return the newest one (a reused buffer) or None if nothing was ready."""
        with self.cond:
            if wait and self.count == 0 and not self.failed:
                self.cond.wait_for(lambda: self.count > 0 or self.failed or not self.running, wait)
            k = min(max(int(n), 1), self.count)
            if k == 0:
                return None, 0
            last = (self.head + k - 1) % self.RING
            self.out[:] = self.ring[last]
            self.head, self.count = (self.head + k) % self.RING, self.count - k
            self.cond.notify_all()
            return self.out, k

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()


class VideoLayer:
    """Pre-rendered loops from a folder ("machine 1"). One reused texture, at most one upload per render
    frame, decoding on a worker thread. Missing folder / corrupt clips just leave the layer empty."""

    SIZE = (640, 400)

    def __init__(self, ctx, folder):
        self.folder = Path(folder)
        self.tex = ctx.texture(self.SIZE, 3)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = True
        self.files, self.i, self.dec, self.has = [], 0, None, False
        self.acc, self.bad, self.scan_t, self.old = 0.0, {}, -1e9, []

    @property
    def name(self):
        return self.files[self.i].name if self.dec is not None and self.i < len(self.files) else ""

    def scan(self):
        try:
            self.files = sorted(fp for fp in self.folder.iterdir() if fp.suffix.lower() in VIDEO_EXTS and fp.is_file())
        except OSError:
            self.files = []
        return self.files

    def _usable(self, fp):
        try:
            return self.bad.get(fp.name) != fp.stat().st_mtime
        except OSError:
            return False

    def _play(self, i):
        self._stop()
        self.has, self.acc = False, 0.0
        if not self.files:
            return
        for k in range(len(self.files)):
            j = (i + k) % len(self.files)
            if self._usable(self.files[j]):
                self.i = j
                try:
                    self.dec = _Decoder(self.files[j], self.SIZE)
                    print(f"[video] {self.files[j].name}")
                except Exception as ex:
                    print(f"[video] could not start {self.files[j].name}: {ex}")
                    self.dec = None
                return

    def _stop(self):
        if self.dec is not None:
            self.dec.stop()
            self.old.append(self.dec)
            self.dec = None
        self.old = [d for d in self.old if d.is_alive()]

    def next(self):
        self.scan()
        if self.files:
            self._play((self.i + 1) % len(self.files))

    def update(self, dt, now, active=True, wait=0.0):
        """Call once per render frame with the (speed-scaled) time step. Returns True while a frame is showing."""
        if not active:
            if self.dec is not None:             # layer faded out: stop decoding, free the ffmpeg process
                self._stop()
                self.has = False
            return False
        if self.dec is None:
            if now - self.scan_t < 3.0:
                return self.has
            self.scan_t = now
            self.scan()
            self._play(self.i)
            if self.dec is None:
                return False
        d = self.dec
        if d.failed:
            try:
                self.bad[d.path.name] = d.path.stat().st_mtime
            except OSError:
                self.bad[d.path.name] = None
            self._stop()
            self.has, self.scan_t = False, now - 2.0
            self.i += 1
            return False
        self.acc = min(self.acc + max(dt, 0.0) * d.fps, 4.0)
        if self.acc >= 1.0 or not self.has:
            buf, k = d.take(int(self.acc) if self.has else 1, wait)
            if buf is not None:
                try:
                    self.tex.write(buf)
                    self.has = True
                except Exception as ex:
                    print(f"[video] upload failed: {ex}")
                if self.acc >= 1.0:
                    self.acc -= k
        return self.has

    def close(self):
        self._stop()
        for d in self.old:
            d.join(timeout=1.0)
        try:
            self.tex.release()
        except Exception:
            pass


def save_mapping(path, corners):
    """Atomic write; never raises (a locked or synced Documents folder must not kill the show)."""
    try:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(corners))
        os.replace(tmp, path)
        print("saved", path)
        return True
    except OSError as ex:
        print(f"[map] could not save {path}: {ex}")
        return False


class Audio:
    """Bass/mid/high energies from loopback or mic, normalized by a decaying peak."""

    BANDS = [(40, 120), (250, 2000), (4000, 14000)]

    def __init__(self, source):
        self.source, self.raw, self.ok, self.running = source, np.zeros(3), False, True
        if source != "none":
            threading.Thread(target=self._run, daemon=True).start()

    def _open(self, sc):
        if self.source == "loopback":
            spk = sc.default_speaker()
            return spk.id, sc.get_microphone(id=str(spk.name), include_loopback=True)
        mic = sc.default_microphone()
        return mic.id, mic

    def _current_id(self, sc):
        try:
            return (sc.default_speaker() if self.source == "loopback" else sc.default_microphone()).id
        except Exception:
            return None

    def _run(self):
        """Capture forever: reopen on errors and when Windows' default device changes (headphones in/out)."""
        try:
            import soundcard as sc
        except Exception as ex:
            print(f"[audio] soundcard unavailable ({ex}); using synthetic 128 BPM beat")
            return
        warnings.filterwarnings("ignore", module="soundcard")
        n, sr = 2048, 48000
        win = np.hanning(n)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        masks = [(freqs >= lo) & (freqs < hi) for lo, hi in self.BANDS]
        while self.running:
            try:
                dev_id, dev = self._open(sc)
                print(f"[audio] {self.source}: {dev.name}")
                buf, peak, last_check = np.zeros(n), np.full(3, 1e-3), time.time()
                with dev.recorder(samplerate=sr, channels=1, blocksize=512) as rec:
                    self.ok = True
                    while self.running:
                        data = rec.record(numframes=512)[:, 0]
                        buf = np.roll(buf, -len(data)); buf[-len(data):] = data
                        mag = np.abs(np.fft.rfft(buf * win))
                        e = np.nan_to_num(np.array([np.sqrt(np.mean(mag[m] ** 2)) for m in masks]), nan=0.0, posinf=0.0, neginf=0.0)
                        peak = np.nan_to_num(np.maximum(e, peak * 0.997), nan=1e-3, posinf=1e-3)
                        self.raw = np.clip(e / np.maximum(peak, 0.05), 0, 1)
                        if time.time() - last_check > 2.0:
                            last_check = time.time()
                            cur = self._current_id(sc)
                            if cur is not None and cur != dev_id:
                                print("[audio] default device changed -> reopening")
                                break
            except Exception as ex:
                if self.ok:
                    print(f"[audio] lost device ({ex}); synthetic beat until it's back")
                self.ok = False
                time.sleep(1.0)

    def get(self, t):
        if self.ok:
            return np.nan_to_num(self.raw, nan=0.0, posinf=1.0, neginf=0.0)
        beat = (t * 128 / 60) % 1.0
        return np.array([math.exp(-beat * 6), 0.5 + 0.3 * math.sin(t * 0.7), 0.5 + 0.5 * math.sin(t * 5.3) ** 8])


TEX_GLOBS = ("*.png", "*.jpg", "*.jpeg")


def texture_files(folder):
    """Finished image files (skips anything modified in the last 2 s, i.e. still being written)."""
    out, now = [], time.time()
    for g in TEX_GLOBS:
        for fp in Path(folder).glob(g):
            try:
                if now - fp.stat().st_mtime > 2.0:
                    out.append(fp)
            except OSError:
                pass
    return sorted(out)


def _make_texture(ctx, fp):
    img = Image.open(fp)
    img.load()                                   # full decode now, so truncated files raise here
    img = img.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    tex = ctx.texture(img.size, 3, img.tobytes())
    tex.repeat_x = tex.repeat_y = True
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.anisotropy = 4.0
    return tex


def select_recent(files, cap=TEX_CAP, keep=()):
    """Names to hold on the GPU from [(name, mtime)]: the `cap` most recently written plus anything in `keep`
    (the textures on screen), in name order."""
    files = list(files)
    if cap is None or len(files) <= cap:
        return sorted(n for n, _ in files)
    newest = {n for n, _ in sorted(files, key=lambda f: (f[1], f[0]), reverse=True)[:max(int(cap), 0)]}
    return sorted(n for n, _ in files if n in newest or n in keep)


def _tex_candidates(folder, bad=None, cap=TEX_CAP, keep=()):
    """[(path, mtime)] of the textures that should be loaded, in name order. A file in `bad` is skipped until
    its mtime changes (it was rewritten), then it is tried again."""
    cands = []
    for fp in texture_files(folder):
        try:
            mtime = fp.stat().st_mtime
        except OSError:
            continue
        if bad is not None and bad.get(fp.name) == mtime:
            continue
        cands.append((fp, mtime))
    chosen = set(select_recent([(fp.name, m) for fp, m in cands], cap, keep))
    return [(fp, m) for fp, m in cands if fp.name in chosen]


def wanted_textures(folder, bad=None, cap=TEX_CAP, keep=()):
    """Names the library should hold right now (cheap: directory listing only)."""
    return [fp.name for fp, _ in _tex_candidates(folder, bad, cap, keep)]


def load_textures(ctx, folder, existing=None, max_new=None, bad=None, cap=TEX_CAP, keep=()):
    """[(name, tex)]: reuses textures in `existing`, loads up to `max_new` new files, releases removed ones.
    Bad / half-written images are skipped, never raised; `bad` ({name: mtime}) remembers them so a broken
    file is not re-decoded on every scan."""
    existing = existing or []
    have = {name: tex for name, tex in existing if name != "noise"}
    cands = _tex_candidates(folder, bad, cap, keep)
    names = [fp.name for fp, _ in cands]
    for name in [n for n in have if n not in names]:
        have.pop(name).release()                 # deleted, or evicted by the cap
    loaded = 0
    for fp, mtime in cands:
        if fp.name in have:
            continue
        if max_new is not None and loaded >= max_new:
            break
        try:
            have[fp.name] = _make_texture(ctx, fp)
            loaded += 1
        except Exception as ex:
            print(f"[tex] skipped {fp.name}: {ex}")
            if bad is not None:
                bad[fp.name] = mtime
    texs = [(n, have[n]) for n in names if n in have]
    noise = [t for n, t in existing if n == "noise"]
    if texs:
        for t in noise:
            t.release()
        return texs
    if noise:                                    # keep the placeholder so the AI layer is never empty
        return [("noise", noise[0])]
    rng = np.random.default_rng(0)
    tex = ctx.texture((64, 64), 3, (rng.random((64, 64, 3)) * 255).astype("u1").tobytes())
    tex.repeat_x = tex.repeat_y = True
    return [("noise", tex)]




def make_controller(headless=False, live=False):
    """control.Controller when available (the show), params.StaticController otherwise / for headless runs."""
    if Controller is not None and (not headless or live):
        try:
            # headless runs never restore / autosave over the show's live_state.json, and open no MIDI or AI link
            return Controller(state_dir=HERE, restore=not headless, midi=not headless, ai=not headless)
        except Exception as ex:
            print(f"[control] Controller failed to start ({ex}); using the static fallback")
    return StaticController()


def parse_sets(items):
    out = {}
    for it in items or []:
        name, _, val = it.partition("=")
        name = name.strip()
        if name not in PARAMS:
            raise SystemExit(f"--set: unknown param '{name}'")
        try:
            out[name] = float(val)
        except ValueError:
            raise SystemExit(f"--set: bad value in '{it}'")
    return out


def extreme_values(kind):
    """All params at their minimum / maximum, or a seeded random mix of min, max and in-between ('rand7')."""
    if kind == "min":
        return {k: v[0] for k, v in PARAMS.items()}
    if kind == "max":
        return {k: v[1] for k, v in PARAMS.items()}
    if kind.startswith("rand"):
        rng = np.random.default_rng(int(kind[4:] or 0))
        return {k: float((v[0], v[1], v[0] + (v[1] - v[0]) * rng.random())[rng.integers(3)]) for k, v in PARAMS.items()}
    raise SystemExit("--extreme must be min, max or randN")


def clean_bands(a):
    """Audio bands as 3 finite floats in 0..1 (a NaN from a device glitch must never reach the phases)."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64).reshape(-1)[:3], nan=0.0, posinf=1.0, neginf=0.0)
    if a.size < 3:
        a = np.concatenate([a, np.zeros(3 - a.size)])
    return np.clip(a, 0.0, 1.0)


class RateLog:
    """print(), but at most once per `every` seconds per tag (a failing per-frame step must not flood the log)."""

    def __init__(self, every=5.0, clock=time.monotonic):
        self.every, self.clock, self.last, self.count = every, clock, {}, {}

    def __call__(self, tag, msg):
        now = self.clock()
        self.count[tag] = self.count.get(tag, 0) + 1
        if now - self.last.get(tag, -1e18) >= self.every:
            self.last[tag] = now
            n, self.count[tag] = self.count[tag], 0
            print(f"[{tag}] {msg}" + (f"  (x{n})" if n > 1 else ""))
            return True
        return False


LOG = RateLog()


def guarded(tag, fn, *args, default=None, **kw):
    """Run a non-GL per-frame step; a failure is logged (rate-limited) and `default` returned, never raised.
    GL / context errors are not routed through here: those must reach the watchdog."""
    try:
        return fn(*args, **kw)
    except Exception as ex:
        if STRICT:                               # headless / tests: fail loudly
            raise
        LOG(tag, f"{type(ex).__name__}: {ex}")
        return default


class QuitGuard:
    """On the projector a single ESC (or a stray window close) must not end the show: quit needs Shift+ESC,
    or two presses within `window` seconds. In the preview window (confirm=False) one press is enough."""

    def __init__(self, confirm=True, window=1.0):
        self.confirm, self.window, self.armed_at = confirm, window, None

    def press(self, now, shift=False):
        if not self.confirm or shift or (self.armed_at is not None and 0.0 <= now - self.armed_at <= self.window):
            return True
        self.armed_at = now
        print("[quit] press ESC again within 1 s (or Shift+ESC) to quit")     # console only, never on the projector
        return False


def monitor_infos():
    """[(name, width, height, x, y)] for the connected monitors."""
    out = []
    for m in glfw.get_monitors():
        try:
            vm, (x, y) = glfw.get_video_mode(m), glfw.get_monitor_pos(m)
            name = glfw.get_monitor_name(m)
            out.append((name.decode(errors="replace") if isinstance(name, bytes) else str(name), vm.size.width, vm.size.height, x, y))
        except Exception:
            pass
    return out


def pick_monitor(infos, index=None, remembered=None):
    """Index of the projector in `infos`, or None while it is not connected. Once it has been seen it is
    recognised by (name, width, height), so it is found again even if Windows reorders the displays."""
    if remembered is not None:
        for i, info in enumerate(infos):
            if tuple(info[:3]) == tuple(remembered[:3]):
                return i
        for i, info in enumerate(infos):                 # same size under another name (driver renamed it)
            if i > 0 and tuple(info[1:3]) == tuple(remembered[1:3]):
                return i
        return None
    return index if index is not None and 0 <= index < len(infos) else None


class Limiter:
    """Soft brightness ceiling. Fed with the PRE-limiter picture (mean level 0..1, near-black fraction), so the
    correction never chases its own tail: target from the measurement with a deadband, symmetric smoothing.
    The black floor only applies to the mandala (symmetry > 0.5); a flat field legitimately fills the frame."""

    def __init__(self, mean=0.36, black=0.15, tau=0.8, deadband=0.02):
        self.mean, self.black, self.tau, self.deadband = mean, black, tau, deadband
        self.target = self.value = 0.0

    def measure(self, level, black_frac, symmetry=1.0):
        want = 0.0
        if level > self.mean > 0:
            want = min(1.0, 2.0 * (1.0 - self.mean / level))
        if symmetry > 0.5 and black_frac < self.black:
            want = max(want, min(1.0, (self.black - black_frac) / self.black))
        if abs(want - self.target) > self.deadband or (want == 0.0) != (self.target == 0.0):
            self.target = want
        return self.target

    def step(self, dt):
        self.value += (self.target - self.value) * (1.0 - math.exp(-max(dt, 0.0) / self.tau))
        if abs(self.target - self.value) < 1e-4:
            self.value = self.target
        return self.value


NOISE_P, TRZ_P, SWAY_P, LATTICE_N = 1600.0, 256.0, 100.0 * math.pi, 320.0
HSF = math.pi / 8                                 # flat-field lobe half-period (same constant as the shader)
SERP_H = np.random.default_rng(7).random((8, 3))  # fixed per-serpent constants (start position, -, pace)
SERP_LAPS = 64                                    # a serpent's track sequence repeats after this many crossings
SERP_T = np.random.default_rng(11).random((8, SERP_LAPS, 4))   # per serpent, per crossing: heading, offset, bend, wave seed
SERP_XS = np.linspace(-1.9, 1.9, 191)             # along-track samples used to find where a track is on screen
SWN_K, SWS_K = (0.21, 0.31, 0.23, 0.17, 0.05, 0.03), (0.9, 0.7, 0.11, 0.2)


def serpent_tracks(idx, laps, size, wiggle, aspect=1.6):
    """Track of serpent idx[k] on crossing laps[k] -> array (n, 6): heading, offset, bend, x_in, lap length, wave seed.

    A track is the curve y = offset + bend * x^2 in a frame rotated by `heading` about the frame centre: a shallow
    arc that crosses the picture mostly sideways. x_in is where the head enters and the lap length how far it
    travels before the next crossing starts: the stretch of the track that can be seen (the frame grown by the
    body's reach across the track), or the body length if that is longer, so the previous tail is always gone
    before a track is reused. Same constants as serpents() in the shader."""
    idx, laps = np.asarray(idx, dtype=int), np.asarray(laps, dtype=int) % SERP_LAPS
    u = SERP_T[idx, laps]
    size, wiggle = min(max(size, 0.0), 1.0), min(max(wiggle, 0.0), 1.0)
    w0, ln, amp = 0.020 + 0.065 * size, 0.75 + 1.15 * size, 0.02 + 0.13 * wiggle
    side = np.floor(u[:, 0] * 2.0)
    ang = side * math.pi + (u[:, 0] * 2.0 - side - 0.5) * 1.2          # leftward or rightward, within 34 degrees of level
    off = (u[:, 1] - 0.5) * 0.56 * (1.0 - 0.4 * wiggle)                # wide swings keep nearer the middle
    bend = (u[:, 2] - 0.5) * 0.4 - off * 0.6                           # arcs lean back toward the centre
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x = SERP_XS[None, :]
    y = off[:, None] + bend[:, None] * x * x
    px, py = ca * x - sa * y, sa * x + ca * y
    reach, pad = 1.6 * amp + 1.5 * w0, 0.04 + 0.02 * ln
    g = np.maximum(np.abs(px) - (aspect / 2 + reach * np.abs(sa) + pad), np.abs(py) - (0.5 + reach * np.abs(ca) + pad))
    seen = g < 0.0                                                     # the body can be seen around this point of the track
    seen[:, 0] = seen[:, -1] = False
    first, last = seen.argmax(axis=1), seen.shape[1] - 1 - seen[:, ::-1].argmax(axis=1)
    rows, step, ok = np.arange(len(idx)), SERP_XS[1] - SERP_XS[0], seen.any(axis=1)
    first, last = np.where(ok, first, 1), np.where(ok, last, 1)
    g0, g1 = g[rows, first - 1], g[rows, first]                        # interpolate the crossings: smooth in size / wiggle
    x_in = np.where(ok, SERP_XS[first - 1] + step * g0 / np.maximum(g0 - g1, 1e-9), -1.0)
    g0, g1 = g[rows, last], g[rows, last + 1]
    x_out = np.where(ok, SERP_XS[last] + step * g0 / np.minimum(g0 - g1, -1e-9), 1.0)
    return np.stack([ang, off, bend, x_in, np.maximum(x_out - x_in, ln + 0.05), u[:, 3]], axis=1)


class Phases:
    """Every motion phase, integrated in double precision and handed to the shader wrapped at a period where
    the pattern repeats exactly (noise 1600, lattices 320 half-wedges, octaves 256, angles TAU). Nothing the
    GPU sees grows with show length, so hour 5 is as crisp as minute 1; speed changes never jump."""

    def __init__(self, aspect=1.6):
        self.aspect = aspect
        self._tr = {}
        self.s_hp, self.s_lap = None, np.arange(8) * 8            # serpents: distance along the current crossing, crossing number
        self.T = self.swim = self.serp = self.travel = 0.0        # raw doubles
        self.flowM = self.flowF = self.trU = self.trT = self.trG = self.trZ = self.trSway = self.rot = 0.0
        self.act = self.tex = 0.0

    @staticmethod
    def periods(vals):
        folds = math.floor(min(max(vals.get("folds", 8.0), 2.0), 24.0) + 0.5)
        return LATTICE_N * math.pi / folds, LATTICE_N * HSF

    def advance(self, dt, vals, bands):
        """dt = wall step; freeze and speed are applied here. bands = the (held-while-frozen) audio bands."""
        fz = min(max(vals.get("freeze", 0.0), 0.0), 1.0)
        dtf = dt * (1.0 - fz)
        dts = dtf * vals.get("speed", 1.0)
        bass, mid = float(bands[0]), float(bands[1])
        pu, pf = self.periods(vals)
        self.T += dts
        self.tex += dtf
        self.swim += dts * (0.30 + 1.3 * mid)
        dflow = dts * (vals.get("flow", 0.0) / 0.3) * (0.25 + 0.35 * mid + 0.9 * bass) * 0.10
        self.flowM, self.flowF = (self.flowM + dflow) % pu, (self.flowF + dflow) % pf
        self.rot = (self.rot + dts * vals.get("rotate", 0.0) * 0.3) % (2 * math.pi)
        trav = vals.get("travel", 0.0)
        dtr = dts * trav * 0.6 * (1.0 + 1.2 * bass)
        if abs(trav) > 0.02:
            dtr += dtf * math.copysign(1.5, trav) * vals.get("punch", 0.0)        # punch = forward lurch
        dsc = 0.06 / (0.12 * 0.25 ** vals.get("density", 0.5))                    # flat density scale (as in the shader)
        self.travel += dtr
        self.trU, self.trT = (self.trU + dtr * math.log(2.0)) % pu, (self.trT + dtr * 2.0) % pu
        self.trG = (self.trG + dtr * 2.0 * dsc) % pf
        self.trZ, self.trSway = (self.trZ + dtr) % TRZ_P, (self.trSway + dtr * 2.0) % SWAY_P
        self.act += (min(1.0, abs(trav) * 8.0) - self.act) * (1 - math.exp(-dts / 0.6))
        dserp = dts * vals.get("serpent_speed", 0.0) * 3.0
        self.serp += dserp
        if vals.get("serpent", 0.0) > 0.0:                         # faded out: they wait where they are
            self._slither(dserp, vals)
        return dts, dtf

    def _tracks(self, vals, back=0):
        key = (back, tuple(self.s_lap), vals.get("serpent_size", 0.4), vals.get("serpent_wiggle", 0.5), self.aspect)
        hit = self._tr.get(back)
        if hit is None or hit[0] != key:                           # tracks only change at a new crossing or a fader move
            hit = self._tr[back] = (key, serpent_tracks(np.arange(8), self.s_lap - back, key[2], key[3], self.aspect))
        return hit[1]

    def _slither(self, d, vals):
        """Move every head d (x its own pace) along its track; past the end of a crossing the next one begins."""
        tr = self._tracks(vals)
        if self.s_hp is None:
            self.s_hp = (0.25 + 0.5 * SERP_H[:, 0]) * tr[:, 4]     # start spread out, all on screen
        self.s_hp = self.s_hp + d * (0.45 + 0.35 * SERP_H[:, 2])
        for _ in range(4):
            fwd, rev = self.s_hp >= tr[:, 4], self.s_hp < 0.0
            if not (fwd.any() or rev.any()):
                break
            self.s_hp = np.where(fwd, self.s_hp - tr[:, 4], self.s_hp)
            self.s_lap = (self.s_lap + fwd.astype(int) - rev.astype(int)) % SERP_LAPS
            tr = self._tracks(vals)
            self.s_hp = np.where(rev, self.s_hp + tr[:, 4], self.s_hp)
        self.s_hp = np.clip(self.s_hp, 0.0, tr[:, 4])

    def _serpents(self, vals):
        """(current tracks, previous tracks, head x on the current track, head x on the previous track)."""
        cur, prev = self._tracks(vals), self._tracks(vals, 1)
        hp = self.s_hp if self.s_hp is not None else (0.25 + 0.5 * SERP_H[:, 0]) * cur[:, 4]
        return cur, prev, cur[:, 3] + hp, prev[:, 3] + prev[:, 4] + hp

    def serpent_heads(self, vals, aspect=None):
        """Head centre of each of the 8 serpents in content units (x +-aspect/2, y +-0.5), as the shader draws it."""
        if aspect is not None:
            self.aspect = aspect
        cur, _, x, _ = self._serpents(vals)
        wig = min(max(vals.get("serpent_wiggle", 0.5), 0.0), 1.0)
        amp, kx = 0.02 + 0.13 * wig, 5.0 + 7.0 * wig
        a1 = kx * x - self.serp * (1.2 + SERP_H[:, 2]) + cur[:, 5] * 9.0
        a2 = 0.41 * kx * x + self.serp * 0.5 + cur[:, 5] * 5.0
        y = cur[:, 1] + cur[:, 2] * x * x + amp * np.sin(a1) + 0.6 * amp * np.sin(a2)
        ca, sa = np.cos(cur[:, 0]), np.sin(cur[:, 0])
        return np.stack([ca * x - sa * y, sa * x + ca * y], axis=1)

    def uniforms(self, vals, k=0):
        """Uniform values. k = whole periods added on top of every wrapped phase (tests: the picture must not change)."""
        tau = 2 * math.pi
        pu, pf = self.periods(vals)
        sa, sb, sc = np.zeros((8, 4), dtype="f4"), np.zeros((8, 4), dtype="f4"), np.zeros((8, 4), dtype="f4")
        if vals.get("serpent", 0.0) > 0.0:
            cur, prev, xc, xp = self._serpents(vals)
            sa[:, :3], sa[:, 3] = cur[:, :3], xc
            sc[:, :3], sc[:, 3] = prev[:, :3], xp
            sb[:, 0] = (self.serp * (1.2 + SERP_H[:, 2])) % tau + k * tau
            sb[:, 1] = (self.serp * 0.5) % tau + k * tau
            sb[:, 2], sb[:, 3] = cur[:, 5], prev[:, 5]
        return dict(
            t=self.T, rotPh=self.rot + k * tau, travelAct=self.act,
            swN=np.array([(self.swim * c) % NOISE_P + k * NOISE_P for c in SWN_K], dtype="f4"),
            swS=np.array([(self.swim * c) % tau + k * tau for c in SWS_K], dtype="f4"),
            aiDrift=(self.swim * 0.008) % 1.0 + k,
            flowM=self.flowM + k * pu, flowF=self.flowF + k * pf,
            trU=self.trU + k * pu, trT=self.trT + k * pu, trG=self.trG + k * pf,
            trZ=self.trZ + k * TRZ_P, trSway=self.trSway + k * SWAY_P,
            serpA=sa, serpB=sb, serpC=sc,
            serpW=np.array([(self.serp * 3.0) % tau + k * tau, (self.serp * 1.3) % tau + k * tau], dtype="f4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=None, help="borderless on this monitor index (projector usually 1); waits for it if unplugged")
    ap.add_argument("--render", default="1280x800", help="internal render size (BenQ MW632ST native is 1280x800)")
    ap.add_argument("--audio", default="loopback", choices=["loopback", "mic", "none"])
    ap.add_argument("--textures", default=str(HERE / "textures"))
    ap.add_argument("--videos", default=str(HERE / "videos"), help="folder of video loops for the video layer")
    ap.add_argument("--cycle", type=float, default=12.0, help="seconds per AI texture")
    ap.add_argument("--spout", action="store_true")
    ap.add_argument("--limit-mean", type=float, default=LIMIT_MEAN, help="brightness ceiling: mean peak-channel level 0..1 the frame may reach")
    ap.add_argument("--test", type=int, default=0, help="headless: render N frames, save samples, report fps")
    ap.add_argument("--mode", default=None, choices=MODES, help="starting layer preset")
    ap.add_argument("--palette", type=float, default=None, help=f"starting palette 0..{len(PALETTES) - 1} (fractional = blend)")
    ap.add_argument("--folds", type=int, default=None, choices=FOLDS, help="starting kaleidoscope fold count")
    ap.add_argument("--fb", type=float, default=None, help="starting trails amount 0..0.98")
    ap.add_argument("--twist", type=int, default=None, help=f"starting twirl preset index 0..{len(TWISTS) - 1}")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="set any params.py value at start (repeatable)")
    ap.add_argument("--extreme", default=None, help="headless stress: all params at min | max | randN (seeded mix)")
    ap.add_argument("--out", default=None, help="headless: folder for saved frames (default test_frames/)")
    ap.add_argument("--snapshot", action="store_true", help="headless: also write <out>/snapshots/current.png at the last frame")
    ap.add_argument("--test-audio", default="beat", choices=["beat", "silent"], help="headless --test: synthetic kick or silence")
    ap.add_argument("--phase-offset", type=int, default=0, metavar="K",
                    help="headless: add K whole wrap periods to every wrapped phase (the picture must not change)")
    ap.add_argument("--no-nan-guard", action="store_true", help="headless: shader outputs raw values, NaN guards off (tests)")
    ap.add_argument("--live-controller", action="store_true", help="headless: use control.Controller instead of the static one")
    ap.add_argument("--gallery", action="store_true",
                    help="headless: every look / contortion / trigger / travel mode -> test_frames/gallery/contact_sheet.png")
    ap.add_argument("--batch", default=None, metavar="JOBS.json",
                    help="headless: render a list of jobs [{name, sets, frames, script}] in one process -> <out>/<name>.png + batch.json")
    ap.add_argument("--remote", action="store_true", help="start the touch remote web page (Ctrl+W toggles it at runtime)")
    ap.add_argument("--remote-port", type=int, default=8765)
    ap.add_argument("--remote-token", default=None, help="panel token (default: persistent, stored in .remote_token)")
    ap.add_argument("--panel", action="store_true", help="open the control panel in a browser on this laptop (implies --remote)")
    args = ap.parse_args()
    rw, rh = map(int, args.render.split("x"))
    headless = bool(args.test or args.gallery or args.batch)
    sheet = bool(args.gallery or args.batch)             # job-list runs (synthetic audio, one shot per job)
    start_sets = parse_sets(args.set)
    global STRICT
    STRICT = headless
    proj_mode = args.monitor is not None and not headless  # show mode: guarded quit, hidden cursor, hot-plug handling

    if not glfw.init():
        raise SystemExit("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    infos = monitor_infos()
    print("[display] monitors:", [f"{i}:{n} {w}x{h}" for i, (n, w, h, _, _) in enumerate(infos)])
    projector = None                                       # (name, w, h) once seen
    if headless:
        glfw.window_hint(glfw.VISIBLE, False)
        win = glfw.create_window(rw, rh, "test", None, None)
    elif proj_mode:
        # The projector may be unplugged or still waking up: wait for it rather than opening a decorated
        # preview window on the laptop (the watchdog / panel keep running; Ctrl+C in the console aborts).
        idx, told = pick_monitor(infos, args.monitor), False
        while idx is None:
            if not told:
                print(f"[display] monitor {args.monitor} not connected (projector plugged in + Extend mode?) - waiting, retry every 2 s")
                told = True
            time.sleep(2.0)
            glfw.poll_events()
            infos = monitor_infos()
            idx = pick_monitor(infos, args.monitor)
        projector = infos[idx][:3]
        _, mw, mh, mx, my = infos[idx]
        # Borderless window covering the projector (not exclusive fullscreen): it never
        # minimizes when you click on the laptop screen, never steals focus, stays on top.
        for hint, val in ((glfw.DECORATED, False), (glfw.AUTO_ICONIFY, False), (glfw.FLOATING, True),
                          (glfw.FOCUS_ON_SHOW, False), (glfw.FOCUSED, False), (glfw.VISIBLE, False)):
            glfw.window_hint(hint, val)
        win = glfw.create_window(mw, mh, "Projection", None, None)
        glfw.set_window_pos(win, mx, my)
        glfw.show_window(win)
        print(f"[display] borderless on monitor {idx} '{projector[0]}' at {mx},{my} {mw}x{mh}")
    else:
        win = glfw.create_window(rw, rh, "Projection (preview)", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    ctx = moderngl.create_context()
    print(f"[gl] {ctx.info['GL_RENDERER']} / {ctx.info['GL_VERSION']}")

    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))
    scene_prog = ctx.program(vertex_shader=VS, fragment_shader=SCENE_FS)
    out_prog = ctx.program(vertex_shader=VS, fragment_shader=OUT_FS)
    meter_prog = ctx.program(vertex_shader=VS, fragment_shader=METER_FS)
    scene_vao = ctx.vertex_array(scene_prog, [(quad, "2f", "in_pos")])
    out_vao = ctx.vertex_array(out_prog, [(quad, "2f", "in_pos")])
    meter_vao = ctx.vertex_array(meter_prog, [(quad, "2f", "in_pos")])
    unbound = sorted(n for n in PARAMS if f"p_{n}" not in scene_prog and n not in CPU_ONLY)
    print(f"[params] {len(PARAMS)} params, unbound: {', '.join(unbound) if unbound else 'none'}")
    print("[uniforms] clock:", ", ".join(n for n in CLOCK_UNIFORMS if n in scene_prog) or "none")

    pp = [ctx.texture((rw, rh), 4, dtype="f2") for _ in range(2)]
    for tx in pp:
        tx.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tx.repeat_x = tx.repeat_y = False
    disp = ctx.texture((rw, rh), 4, dtype="f2")           # displayed / Spout frame; pp = ping-pong echo energy
    disp.filter = (moderngl.LINEAR, moderngl.LINEAR)
    disp.repeat_x = disp.repeat_y = False
    fbos = [ctx.framebuffer([disp, tx]) for tx in pp]
    for fb in fbos:
        fb.clear(0.0, 0.0, 0.0, 1.0)
    test_fbo = ctx.simple_framebuffer((rw, rh)) if headless else None
    meter_fbo = ctx.simple_framebuffer(METER_SIZE)             # box-filtered pre-limiter level for the brightness limiter
    limiter = Limiter(mean=args.limit_mean, black=LIMIT_BLACK)
    lim_hist = []

    # textures: headless loads the whole (capped) library for determinism; the show starts rendering at once
    # and streams the library in a couple of files per scan
    bad_tex = {}
    texs = load_textures(ctx, args.textures, bad=bad_tex, max_new=None if headless else 2)
    print(f"[tex] {len(texs)} textures from {args.textures}" + ("" if headless else " (loading the rest in the background)"))

    unit_sq = UNIT_SQ
    map_file = HERE / "mapping.json"
    corners = load_corners(map_file)
    hinv = inverse_homography(corners)

    test_dir = Path(args.out) if args.out else HERE / "test_frames"
    if headless:
        test_dir.mkdir(parents=True, exist_ok=True)
    gal_dir = test_dir / "gallery"
    video_dir = Path(args.videos)
    if args.gallery:
        gal_dir.mkdir(exist_ok=True)
        video_dir = test_dir / "video_test"
        make_test_clip(video_dir / "test_clip.mp4")
    elif args.batch:
        gal_dir = test_dir
    video = VideoLayer(ctx, video_dir)

    # optional offscreen passes built by other modules: (param, sampler uniform, texture unit, pass | None)
    extra = []
    for pname, uname, unit, cls in (("l_fractal", "texF", 4, FractalPass), ("l_art", "texR", 5, ArtPass)):
        obj = None
        if cls is not None and pname in PARAMS:
            try:
                obj = cls(ctx, (rw, rh), folder=HERE / "art") if pname == "l_art" else cls(ctx, (rw, rh))
            except Exception as ex:
                print(f"[{pname}] disabled: {ex}")
        extra.append(dict(param=pname, uniform=uname, unit=unit, obj=obj, fails=0))

    controller = make_controller(headless, args.live_controller)
    print(f"[control] {type(controller).__name__}")
    DEF = defaults()

    def apply_start(ctl):
        if args.extreme:
            for k, v in extreme_values(args.extreme).items():
                ctl.set(k, v)
        if args.mode:
            for k, v in zip(PRESET_KEYS, MODE_W[MODES.index(args.mode)]):
                ctl.set(k, v)
        if args.palette is not None: ctl.set("palette", args.palette)
        if args.folds is not None: ctl.set("folds", args.folds)
        if args.fb is not None: ctl.set("trails", args.fb)
        if args.twist is not None: ctl.set("twirl", TWISTS[min(max(args.twist, 0), len(TWISTS) - 1)])
        for k, v in start_sets.items():
            ctl.set(k, v)

    apply_start(controller)

    # corner-pin / UI state. The legacy keys (fold_i .. fbAmt) only carry old touch-remote commands across to params.
    st = dict(grid=False, calib=False, drag=None, next=False, fold_i=None, mode=None, palette=None, fb=None, fbAmt=None,
              vals=dict(DEF), hinv=hinv, preset=len(MODES) - 1, twist=0, snap=False, show=None,
              mon_check=False, proj=True, cursor=None)
    token_file = HERE / ".remote_token"          # single-line token, shared with ai_worker.py
    remote = RemoteControl(port=args.remote_port, token=load_remote_token(token_file, args.remote_token),
                           tex_dir=Path(args.textures), art_dir=HERE / "art", ai_url="http://127.0.0.1:8766",
                           token_paths=[token_file])
    if (args.remote or args.panel) and not headless:
        remote.toggle()
        if remote.running:
            local = f"http://localhost:{args.remote_port}/?k={remote.token}"
            print(f"[panel] laptop control panel: {local}")
            if args.panel:
                import webbrowser
                webbrowser.open(local)

    def set_preset(i):
        st["preset"] = i % len(MODES)
        for k, v in zip(PRESET_KEYS, MODE_W[st["preset"]]):
            controller.set(k, v)
        print("layers", MODES[st["preset"]])

    def set_palette(i):                          # the controller glides the (fractional) palette value itself
        controller.set("palette", float(i % len(PALETTES)))
        print("palette", PALETTES[i % len(PALETTES)][0])

    quit_guard = QuitGuard(confirm=proj_mode)
    HOLD = {glfw.KEY_X: "flash", glfw.KEY_Z: "strobe", glfw.KEY_I: "invert", glfw.KEY_Q: "freeze",
            glfw.KEY_K: "blackout", glfw.KEY_J: "punch"}
    NUDGE = {glfw.KEY_UP: ("trails", 0.02), glfw.KEY_DOWN: ("trails", -0.02),
             glfw.KEY_EQUAL: ("zoom", 0.04), glfw.KEY_MINUS: ("zoom", -0.04),
             glfw.KEY_RIGHT_BRACKET: ("density", 0.04), glfw.KEY_LEFT_BRACKET: ("density", -0.04),
             glfw.KEY_RIGHT: ("flow", 0.05), glfw.KEY_LEFT: ("flow", -0.05),
             glfw.KEY_D: ("flow_dir", 0.02), glfw.KEY_A: ("flow_dir", -0.02),
             glfw.KEY_O: ("bulge", 0.08), glfw.KEY_L: ("bulge", -0.08),
             glfw.KEY_Y: ("travel", 0.1), glfw.KEY_H: ("travel", -0.1)}

    def on_key(w, key, sc_, action, mods):
        ctrl = bool(mods & glfw.MOD_CONTROL)
        if key in HOLD and not ctrl:
            if action != glfw.REPEAT:
                controller.trigger(HOLD[key], action == glfw.PRESS)
            return
        if key in NUDGE and not ctrl and action in (glfw.PRESS, glfw.REPEAT):
            if NUDGE[key][0] in PARAMS:
                controller.nudge(*NUDGE[key])
            return
        if action != glfw.PRESS:
            return
        v = st["vals"]
        if key == glfw.KEY_ESCAPE:
            if quit_guard.press(time.monotonic(), shift=bool(mods & glfw.MOD_SHIFT)):
                glfw.set_window_should_close(w, True)
        elif ctrl:
            # things that would wreck a running show if hit by accident live behind Ctrl
            if key == glfw.KEY_BACKSPACE:
                for name, spec in PARAMS.items():
                    if spec[3] in RESET_GROUPS:
                        controller.set(name, spec[2])
                print("reset space / shape / motion")
            elif key == glfw.KEY_G: st["grid"] = not st["grid"]
            elif key == glfw.KEY_C:
                st["calib"] = not st["calib"]; st["grid"] = st["calib"]
                if not st["calib"]:
                    st["drag"] = None
            elif key == glfw.KEY_W: remote.toggle()
        elif glfw.KEY_1 <= key <= glfw.KEY_9: controller.set("folds", float(FOLDS[key - glfw.KEY_1]))
        elif key == glfw.KEY_M: set_preset(st["preset"] + 1)
        elif key == glfw.KEY_P: set_palette(int(round(v["palette"])) + 1)
        elif key == glfw.KEY_T:
            st["twist"] = (st["twist"] + 1) % len(TWISTS); controller.set("twirl", TWISTS[st["twist"]])
        elif key == glfw.KEY_F: controller.set("l_echo", 0.0 if v["l_echo"] > 0.01 else DEF["l_echo"])
        elif key == glfw.KEY_TAB: controller.set("symmetry", 0.0 if v["symmetry"] > 0.5 else 1.0)
        elif key == glfw.KEY_U: controller.set("wave", 0.0 if v["wave"] > 0.01 else 0.6)
        elif key == glfw.KEY_N and "travel_mode" in PARAMS: controller.set("travel_mode", float((int(round(v["travel_mode"])) + 1) % 4))
        elif key == glfw.KEY_E and "serpent" in PARAMS: controller.set("serpent", 0.0 if v["serpent"] > 0.01 else 0.8)
        elif key in (glfw.KEY_COMMA, glfw.KEY_PERIOD): controller.tap()
        elif key == glfw.KEY_SPACE: st["next"] = True
        elif key == glfw.KEY_V: video.next()
        elif key == glfw.KEY_R and st["calib"]: corners[:] = [list(map(float, c)) for c in unit_sq]
        elif key == glfw.KEY_S: save_mapping(map_file, corners)

    def mouse_uv(w):
        x, y = glfw.get_cursor_pos(w); ww, wh = glfw.get_window_size(w)
        return min(max(x / max(ww, 1), 0.0), 1.0), min(max(1 - y / max(wh, 1), 0.0), 1.0)

    def on_click(w, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            st["drag"] = None
            return
        if not st["calib"]:
            st["drag"] = None
            if action == glfw.PRESS:             # ripple where the pointer is, in content space (through the corner pin)
                u, v = mouse_uv(w)
                q = st["hinv"] @ np.array([u, v, 1.0])
                if abs(q[2]) > 1e-9 and 0.0 <= q[0] / q[2] <= 1.0 and 0.0 <= q[1] / q[2] <= 1.0:
                    controller.ripple(float(q[0] / q[2]), float(q[1] / q[2]))
            return
        if action == glfw.PRESS:
            u, v = mouse_uv(w)
            st["drag"] = int(np.argmin([(c[0] - u) ** 2 + (c[1] - v) ** 2 for c in corners]))
        else:
            st["drag"] = None

    def on_close(w):                             # Alt+F4 / a stray close on the projector needs the same confirmation
        if not quit_guard.press(time.monotonic()):
            glfw.set_window_should_close(w, False)

    # input callbacks never take the show down: an exception in one is logged and dropped
    glfw.set_key_callback(win, lambda *a: guarded("key", on_key, *a))
    glfw.set_mouse_button_callback(win, lambda *a: guarded("mouse", on_click, *a))
    glfw.set_window_close_callback(win, lambda *a: guarded("close", on_close, *a))
    if proj_mode:
        glfw.set_monitor_callback(lambda *a: st.__setitem__("mon_check", True))

    def place_on_projector(now):
        """Hot-plug handling: park the window while the projector is gone, put it back when it returns."""
        infos = monitor_infos()
        idx = pick_monitor(infos, args.monitor, projector)
        if idx is None:
            if st["proj"]:
                st["proj"] = False
                print("[display] projector disconnected - window parked, rendering paused until it is back")
                glfw.set_window_attrib(win, glfw.FLOATING, False)      # never sit on top of the panel on the laptop
                glfw.iconify_window(win)
            return
        _, mw, mh, mx, my = infos[idx]
        if not st["proj"]:
            st["proj"] = True
            print(f"[display] projector back: monitor {idx} at {mx},{my} {mw}x{mh}")
            glfw.restore_window(win)
            glfw.set_window_attrib(win, glfw.FLOATING, True)
        if glfw.get_window_pos(win) != (mx, my) or glfw.get_window_size(win) != (mw, mh):
            glfw.set_window_pos(win, mx, my)
            glfw.set_window_size(win, mw, mh)

    spout, spout_fails = None, 0
    if args.spout:
        try:
            import SpoutGL
            spout = SpoutGL.SpoutSender(); spout.setSenderName("PsyProjector")
            print("[spout] sending as 'PsyProjector'")
        except Exception as ex:
            print(f"[spout] disabled: {ex}")

    audio = Audio("none" if headless else args.audio)
    sm, smr = np.zeros(3), np.zeros(3)           # smoothed bands; smr = what the shader sees (held while frozen)
    ph = Phases(rw / rh)
    chase = 0.0                                  # chase phase in sweeps (double)
    env = bo = 0.0                               # slow bass envelope, smoothed blackout
    prev_bar, prev_bass, last_auto = None, 0.0, -1e9
    ripples = np.full((N_RIPPLES, 3), -1.0, dtype="f4")
    ti, tj, fade_t0 = 0, 1 % len(texs), None
    last_switch = last_scan = last_mon = 0.0
    tex_pending = not headless
    pub_t, pub_frame = 0.0, 0
    t0 = time.perf_counter(); frame = 0; prev_t = 0.0; io_time = 0.0
    last_frame = Frame()
    clk = dict(bpm=124.0, beat=0.0, beat_phase=0.0, bar_phase=0.0, phrase_phase=0.0)   # what the shader sees (held while frozen)
    snap_dir = (test_dir if headless else HERE) / "snapshots"
    jobs, job_i, shots, script, shot_at = [], 0, [], {}, 0
    batch_stats = []
    if sheet:
        if args.gallery:
            rows = gallery_rows()
        else:
            spec = json.loads(Path(args.batch).read_text(encoding="utf-8"))
            rows = [("batch", [(str(j["name"]), {k: float(v) for k, v in j.get("sets", {}).items()},
                                {int(f): [tuple(a) for a in acts] for f, acts in j.get("script", {}).items()},
                                int(j.get("frames", 90))) for j in spec])]
        jobs = [(ri, ci) + cell for ri, (_, cells) in enumerate(rows) for ci, cell in enumerate(cells)]
        shots = [(label, [None] * len(cells)) for label, cells in rows]

        def start_job(i):
            nonlocal controller, script, shot_at
            controller.close()
            controller = StaticController()
            _, _, _, sets, script, shot_at = jobs[i]
            for k, v in sets.items():
                controller.set(k, v)

        start_job(0)

    def run_action(a):
        if a[0] == "trigger": controller.trigger(a[1], a[2])
        elif a[0] == "ripple": controller.ripple(a[1], a[2])
        elif a[0] == "set": controller.set(a[1], a[2])

    def read_nonfinite(fbo):
        return sum(int((~np.isfinite(np.frombuffer(fbo.read(components=3, attachment=i, dtype="f4"), dtype="f4"))).sum())
                   for i in (0, 1))

    try:
        while not glfw.window_should_close(win):
            now = time.perf_counter() - t0
            t = frame / 60.0 if headless else now
            dt = min(0.1, max(1e-3, t - prev_t)); prev_t = t
            for a in script.get(frame, ()):
                run_action(a)
            if args.test and args.snapshot and frame == args.test - 1:
                st["snap"] = True
            if args.test and args.extreme and frame == 3:
                controller.ripple(0.5, 0.5); controller.ripple(0.0, 1.0)

            # audio smoothing: fast attack, slow release
            if sheet: a = gallery_audio(t)
            elif args.test and args.test_audio == "silent": a = np.zeros(3)
            else: a = audio.get(t)
            a = clean_bands(a)
            k = np.where(a > sm, 1 - math.exp(-dt / 0.03), 1 - math.exp(-dt / 0.25))
            sm += (a - sm) * k

            # ---- controller: one Frame of final values, events and clock. Non-GL steps are guarded: a bug in
            # the control system, a MIDI device or the panel must never take the picture down.
            fr = guarded("control", controller.update, dt, sm, default=None)
            if fr is None:
                fr = Frame(values=dict(last_frame.values), events=[], clock=dict(last_frame.clock))
            last_frame = fr
            vals = dict(DEF)
            for name, v in (fr.values or {}).items():
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if name in PARAMS and math.isfinite(v):
                    vals[name] = min(max(v, PARAMS[name][0]), PARAMS[name][1])
            st["vals"] = vals
            clock = dict(bpm=124.0, beat=0.0, beat_phase=0.0, bar_phase=0.0, phrase_phase=0.0)
            clock.update({k2: float(v2) for k2, v2 in (fr.clock or {}).items() if isinstance(v2, (int, float)) and math.isfinite(v2)})

            # ---- commands: the controller takes what it owns (call it every frame: MIDI- and AI-raised
            # commands come out of it too); corner pin / grid / textures / snapshot stay here
            cmds = guarded("remote", remote.drain, default=[]) if remote.running else []
            rest = guarded("control", controller.apply, cmds, default=[])
            own = []
            for c in rest or ():
                if not isinstance(c, dict):
                    continue
                kind = c.get("cmd") or c.get("type")
                if kind == "snapshot": st["snap"] = True
                elif kind == "show_texture": st["show"] = (str(c.get("name", "")), now + 10.0)
                elif kind == "next_video": guarded("video", video.next)
                elif isinstance(c.get("cmd"), str): own.append(c)
            if own:
                if "save" in (guarded("remote", apply_commands, own, st, corners, len(FOLDS), len(PALETTES), default=None) or ()):
                    save_mapping(map_file, corners)

                def legacy():                    # old touch-remote commands -> params
                    if st["fold_i"] is not None: controller.set("folds", float(FOLDS[min(max(int(st["fold_i"]), 0), len(FOLDS) - 1)]))
                    if st["mode"] is not None: set_preset(min(max(int(st["mode"]), 0), len(MODES) - 1))
                    if st["palette"] is not None: set_palette(int(st["palette"]))
                    if st["fb"] is not None: controller.set("l_echo", DEF["l_echo"] if st["fb"] else 0.0)
                    if st["fbAmt"] is not None: controller.set("trails", float(st["fbAmt"]))

                guarded("remote", legacy)
                st.update(fold_i=None, mode=None, palette=None, fb=None, fbAmt=None)

            # ---- time: every phase is integrated on the CPU (double precision, wrapped) so speed changes never
            # jump and nothing drifts over a long show; freeze stops them all, including what the audio drives
            fz = vals["freeze"]
            smr += (sm - smr) * (1.0 - fz)
            dts, dtf = ph.advance(dt, vals, smr)
            if fz < 0.5:
                clk = clock                      # beat-synced accents hold their value while frozen
            bar = clock["beat"] / 4.0
            dbar = 0.0 if prev_bar is None or not (0.0 <= bar - prev_bar < 0.5) else bar - prev_bar
            prev_bar = bar
            rate = vals.get("chase_rate", 1.0)
            chase += dbar * rate * (1.0 - fz)
            if fz < 0.5 and round(rate) != 0 and abs(rate - round(rate)) < 0.02:   # integer rates lock to the bar
                err = ((clock["bar_phase"] * round(rate) - chase + 0.5) % 1.0) - 0.5
                chase += err * min(1.0, dt * 1.5)
            env += (smr[0] - env) * (1 - math.exp(-dtf / 1.2))
            calm = 1.0 - min(1.0, max(0.0, (env - 0.04) / 0.14))        # 1 in a breakdown (no bass for a while)
            breath = 0.5 - 0.5 * math.cos(2 * math.pi * ph.T / BREATH_PERIOD)
            bo += (vals["blackout"] - bo) * (1 - math.exp(-dt / 0.035))  # ~100 ms fade
            T = ph.T

            # ---- ripples: age the active ones, spawn from events (and from kicks when ripple is turned up)
            act = ripples[:, 2] >= 0
            ripples[act, 2] += dtf
            life = min(12.0, 5.5 / (0.35 + 3.65 * vals["ripple_decay"]))
            ripples[ripples[:, 2] > life, 2] = -1.0
            events = [e for e in (fr.events or []) if isinstance(e, (tuple, list)) and len(e) == 3 and e[0] == "ripple"]
            if vals["ripple"] > 0.75 and sm[0] > 0.6 >= prev_bass and now - last_auto > 0.35 and fz < 0.5:
                events.append(("ripple", 0.5, 0.5)); last_auto = now
            prev_bass = sm[0]
            for _, ex_, ey_ in events:
                try:
                    ex_, ey_ = float(ex_), float(ey_)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(ex_) and math.isfinite(ey_):
                    slot = int(np.argmin(ripples[:, 2])) if (ripples[:, 2] < 0).any() else int(np.argmax(ripples[:, 2]))
                    ripples[slot] = (min(max(ex_, 0.0), 1.0), min(max(ey_, 0.0), 1.0), 0.0)

            pal = palette_stops(vals["palette"])    # fractional value = blend of neighbouring palettes

            # touch remote / panel: publish state ~6x/s
            if remote.running and now - pub_t > 0.15:
                def publish():
                    state = dict(folds=FOLDS, modes=MODES, fold_i=int(np.argmin([abs(f - vals["folds"]) for f in FOLDS])),
                                 mode=st["preset"], fb=vals["l_echo"] > 0.01, fbAmt=vals["trails"])
                    try:
                        state.update(controller.state())
                    except Exception as ex:
                        state["control_error"] = str(ex)
                    state.update(grid=st["grid"], calib=st["calib"], corners=corners, res=list(glfw.get_framebuffer_size(win)),
                                 fps=(frame - pub_frame) / (now - pub_t), bands=[float(x) for x in sm],
                                 bass=float(sm[0]), mid=float(sm[1]), high=float(sm[2]),
                                 tex=texs[tj if fade_t0 is not None else ti][0], video=video.name, limiter=round(limiter.value, 3),
                                 projector=bool(st["proj"]), palettes=[n for n, _ in PALETTES],
                                 palette_colors=[[list(c) for c in stops] for _, stops in PALETTES])
                    remote.publish_state(state)

                guarded("publish", publish)
                pub_t, pub_frame = now, frame

            # ---- AI textures: hot-reload (new gen_* files show up within ~2 s; while the library is still
            # streaming in, a couple of files every quarter second), cycling, cross-fade, show-on-request
            if not headless and now - last_scan > (0.25 if tex_pending else 2.0):
                last_scan = now
                cur_name, nxt_name = texs[ti][0], texs[tj][0]
                want = wanted_textures(args.textures, bad_tex, keep=(cur_name, nxt_name))
                have = [n for n, _ in texs]
                tex_pending = bool(want) and want != have
                if tex_pending:
                    texs = load_textures(ctx, args.textures, texs, max_new=2, bad=bad_tex, keep=(cur_name, nxt_name))
                    if [n for n, _ in texs] != have:
                        ti = next((i for i, (n, _) in enumerate(texs) if n == cur_name), 0)
                        tj = next((i for i, (n, _) in enumerate(texs) if n == nxt_name), (ti + 1) % len(texs)) if fade_t0 is not None else (ti + 1) % len(texs)
                        print(f"[tex] library now {len(texs)}")
                    tex_pending = want != [n for n, _ in texs]
            tc = ph.tex
            if st["show"] is not None:
                want_name, until = st["show"]
                hit = next((i for i, (n, _) in enumerate(texs) if n == want_name or Path(n).stem == want_name), None)
                if now > until or not want_name:
                    st["show"] = None
                elif hit is not None and fade_t0 is None:
                    st["show"] = None
                    if hit != ti:
                        tj, fade_t0 = hit, tc
                    last_switch = tc
            if fade_t0 is None and (st["next"] or tc - last_switch > args.cycle):
                st["next"] = False; fade_t0 = tc; tj = (ti + 1) % len(texs)
            mix = 0.0
            if fade_t0 is not None:
                mix = min(1.0, (tc - fade_t0) / 3.0)
                if mix >= 1.0:
                    ti, fade_t0, last_switch, mix = tj, None, tc, 0.0
                    tj = (ti + 1) % len(texs)

            if not st["calib"]:
                st["drag"] = None                    # calibrate turned off (key or remote) ends any held drag
            if st["drag"] is not None:
                trial = [list(c) for c in corners]
                trial[st["drag"]] = list(mouse_uv(win))
                if corners_ok(trial):                # refuse moves that fold the quad (singular homography)
                    corners[st["drag"]] = trial[st["drag"]]

            # ---- projector hot-plug + cursor (show mode only)
            if proj_mode:
                if st["mon_check"] or now - last_mon > 2.0:
                    st["mon_check"], last_mon = False, now
                    guarded("display", place_on_projector, now)
                cur = glfw.CURSOR_NORMAL if st["calib"] else glfw.CURSOR_HIDDEN     # no pointer on the projection
                if cur != st["cursor"]:
                    st["cursor"] = cur
                    glfw.set_input_mode(win, glfw.CURSOR, cur)
                if not st["proj"]:                   # nothing to draw on: keep control / panel alive, idle cheaply
                    glfw.poll_events()
                    time.sleep(0.05)
                    continue

            # ---- video loop + optional offscreen passes (only when their layer is up: zero cost otherwise)
            has_video = video.update(dts, now, active=vals.get("l_video", 0.0) > 0.001, wait=2.0 if headless else 0.0)
            layer_on = {}
            for ex in extra:
                amount = vals.get(ex["param"], 0.0)
                layer_on[ex["param"]] = 0.0
                if ex["obj"] is None or amount <= 0.001 or ex["fails"] >= 5:
                    continue
                try:
                    if ex["param"] == "l_fractal":
                        tex = ex["obj"].render(vals, clock, T, ph.travel, [tuple(map(float, c)) for c in pal])
                    else:
                        tex = ex["obj"].render(vals, clock, T, ph.travel, dts)
                    tex.use(ex["unit"])
                    layer_on[ex["param"]], ex["fails"] = amount, 0
                except Exception as err:
                    ex["fails"] += 1
                    print(f"[{ex['param']}] render failed ({ex['fails']}/5): {err}")

            # ---- scene pass (ping-pong feedback)
            src, dst = frame % 2, (frame + 1) % 2
            fbos[dst].use()
            ctx.viewport = (0, 0, rw, rh)
            texs[ti][1].use(0); texs[tj][1].use(1); pp[src].use(2); video.tex.use(3)
            uni = dict(texA=0, texB=1, prevFrame=2, texV=3, texF=4, texR=5, mixAB=mix,
                       bass=smr[0], mid=smr[1], high=smr[2], aspect=rw / rh, pxh=1.0 / rh,
                       hasVideo=1.0 if has_video else 0.0, limiter=limiter.value, nanGuard=0.0 if args.no_nan_guard else 1.0,
                       chasePh=chase % 1.0, chaseCol=(chase * 0.15) % 1.0, breath=breath, calm=calm,
                       beat=clk["beat"] % 4096.0, beat_phase=clk["beat_phase"],
                       bar_phase=clk["bar_phase"], phrase_phase=clk["phrase_phase"])
            uni.update(ph.uniforms(vals, args.phase_offset if headless else 0))
            for name, v in vals.items():
                uni[f"p_{name}"] = v
            uni["p_blackout"] = bo
            uni.update({f"p_{k2}": v2 for k2, v2 in layer_on.items()})
            for kname, v in uni.items():
                if kname in scene_prog:
                    if isinstance(v, np.ndarray):
                        scene_prog[kname].write(v.tobytes())
                    else:
                        scene_prog[kname].value = v
            if "pal" in scene_prog:
                scene_prog["pal"].write(np.ascontiguousarray(pal, dtype="f4").tobytes())
            if "rip" in scene_prog:
                scene_prog["rip"].write(ripples.tobytes())
            scene_vao.render(moderngl.TRIANGLE_STRIP)

            if spout:
                try:
                    spout.sendTexture(disp.glo, 0x0DE1, rw, rh, True, 0)
                    spout_fails = 0
                except Exception as ex:
                    spout_fails += 1
                    LOG("spout", f"send failed ({spout_fails}/20): {ex}")
                    if spout_fails >= 20:
                        print("[spout] disabled after repeated failures")
                        spout = None

            if st["snap"]:                           # one readback per request, PNG encode off the GL thread
                st["snap"] = False
                try:
                    data = fbos[dst].read(components=3, attachment=0, dtype="f1")
                    job = threading.Thread(target=save_snapshot, args=(snap_dir / "current.png", data, (rw, rh)), daemon=True)
                    job.start()
                    if headless:
                        job.join()
                except Exception as ex:
                    LOG("snapshot", f"failed: {ex}")

            # ---- output pass with corner pin
            hinv = inverse_homography(corners, hinv)
            st["hinv"] = hinv
            target = test_fbo if headless else ctx.screen
            target.use()
            fw, fh = (rw, rh) if headless else glfw.get_framebuffer_size(win)
            ctx.viewport = (0, 0, fw, fh)
            disp.use(0)
            out_prog["scene"].value = 0
            out_prog["Hinv"].write(hinv.T.astype("f4").tobytes())
            out_prog["grid"].value = 1.0 if st["grid"] else 0.0
            out_prog["corners"].write(np.array(corners, dtype="f4").tobytes())
            out_vao.render(moderngl.TRIANGLE_STRIP)

            # ---- brightness meter (every 4th frame), on the PRE-limiter level the scene pass left in the echo
            # buffer's alpha: 64x40 box-filtered blocks -> mean level and near-black fraction -> limiter target.
            # Flash, strobe and invert are deliberate and exempt (the target simply holds).
            if frame % 4 == 0 and max(vals["flash"], vals["strobe"], vals["invert"]) < 0.02 and bo < 0.5 and vals["master"] > 0.05:
                meter_fbo.use()
                ctx.viewport = (0, 0, *METER_SIZE)
                pp[dst].use(6)
                meter_prog["src"].value = 6
                meter_prog["texel"].value = (1.0 / rw, 1.0 / rh)
                meter_vao.render(moderngl.TRIANGLE_STRIP)
                m = np.frombuffer(meter_fbo.read(components=3), dtype=np.uint8).reshape(-1, 3)
                limiter.measure(m[:, 0].mean() / 255.0 * vals["master"], m[:, 1].mean() / 255.0, vals["symmetry"])
            limiter.step(dt)
            if args.test:
                lim_hist.append(limiter.value)

            frame += 1
            if sheet:
                if frame == shot_at:
                    img = Image.frombytes("RGB", (rw, rh), test_fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
                    ri, ci, label = jobs[job_i][:3]
                    name = "".join(ch if ch.isalnum() else "_" for ch in (label if args.batch else f"{ri:02d}_{ci}_{shots[ri][0]}_{label}"))
                    img.save(gal_dir / f"{name}.png")
                    shots[ri][1][ci] = (label, img)
                    blk, mean, blown, sat = frame_stats(img)
                    print(f"[gallery] {name:58s} black {blk:5.1%}  mean {mean:5.1f}  blown {blown:5.2%}  saturated {sat:5.1%}")
                    if args.batch:
                        batch_stats.append(dict(name=label, file=f"{name}.png", nonfinite=read_nonfinite(fbos[dst]), frames=frame,
                                                limiter=limiter.value))
                    job_i += 1
                    if job_i >= len(jobs):
                        if args.batch:
                            (gal_dir / "batch.json").write_text(json.dumps(batch_stats), encoding="utf-8")
                            print(f"[batch] {len(jobs)} jobs in {time.perf_counter() - t0:.1f}s -> {gal_dir}")
                        else:
                            contact_sheet(shots, gal_dir / "contact_sheet.png")
                            print(f"[gallery] {len(jobs)} frames + contact sheet -> {gal_dir / 'contact_sheet.png'}")
                        break
                    start_job(job_i)
                    frame, prev_t, env, bo, chase = 0, 0.0, 0.0, 0.0, 0.0
                    sm[:] = 0; smr[:] = 0
                    ph, limiter = Phases(rw / rh), Limiter(mean=args.limit_mean, black=LIMIT_BLACK)
                    prev_bar, prev_bass, last_auto = None, 0.0, -1e9
                    ripples[:] = -1.0
                    ti, fade_t0, last_switch = ((jobs[job_i][0] * 3 + jobs[job_i][1]) % len(texs) if args.gallery else 0), None, 0.0
                    tj = (ti + 1) % len(texs)
                    for fb in fbos:
                        fb.clear(0.0, 0.0, 0.0, 1.0)
            elif args.test:
                if frame in (30, args.test // 2, args.test):
                    ctx.finish()
                    s0 = time.perf_counter()
                    img = Image.frombytes("RGB", (rw, rh), test_fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
                    img.save(test_dir / f"frame_{frame:04d}.png")
                    io_time += time.perf_counter() - s0
                if frame >= args.test:
                    ctx.finish()
                    el = time.perf_counter() - t0 - io_time
                    tail = lim_hist[-600:]                # the last 10 s
                    print(f"[test] {frame} frames @ {rw}x{rh} in {el:.2f}s = {frame / el:.0f} fps (uncapped)"
                          f" layers={MODES[st['preset']] if not args.mode else args.mode} palette={vals['palette']:.2f}"
                          f" nonfinite={read_nonfinite(fbos[dst])} limiter={limiter.value:.4f} limiter_span={max(tail) - min(tail):.4f}")
                    break
            else:
                glfw.swap_buffers(win)
            glfw.poll_events()
    finally:
        # release everything even when an exception is on its way out: GL / context failures are not swallowed,
        # so the process exits non-zero and the watchdog restarts it (ESC / a confirmed close exits 0)
        for fn in (remote.stop, controller.close, video.close):
            guarded("exit", fn)
        for ex in extra:
            close = getattr(ex["obj"], "close", None) or getattr(ex["obj"], "release", None)
            if close:
                guarded("exit", close)
        audio.running = False
        glfw.terminate()


if __name__ == "__main__":
    main()
