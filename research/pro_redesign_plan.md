# Pro redesign plan: clips, macros, direct manipulation (architect pass, 2026-11-25)

Author: Opus architect pass. Read-only planning; no code was changed. Inputs: HANDOFF.md (incl. the 23:10 verdict on BUILD),
research/pro_vj_software_survey.md, effects_spec_videos.md, gemini_effect_techniques.md, optimization_ideas.md,
QUICKSTART.md, WALKTHROUGH.md, and the code structure (params.py: 189 params in 13 groups; control.py; live_visuals.py
SCENE_FS + `extra` passes + PreviewTap; stack.py; filters.py; panel.html 3302 lines, 15 tabs).

**The one-line verdict:** the engine is already better than most of the paid tools at *generating* psychedelic material
(Folding Blocks, beings, 3D decks, fractals, mapping, safety, governor). What it lacks is the *instrument* layer on top:
a playable unit (the clip), a few meaningful knobs per look, real picture transitions, a grid you can play, and hands-on
manipulation of the picture. 189 raw faders across 15 tabs is a TouchDesigner network with no Perform Mode. This plan
builds the Perform Mode and keeps the network underneath it.

---

## 1. Gap analysis (blunt)

| Capability | Pro reference | Us today | Gap |
|---|---|---|---|
| Playable unit (clip) with a thumbnail | Resolume clip, Synesthesia scene, Notch block | None. A "scene" is a full snapshot of all 189 faders. | **Critical** |
| 4-8 named controls per look | Synesthesia CONTROLS, Notch Exposed Node, Resolume Dashboard | Raw faders with engineer names ("freq", "duty", "soft", "warp", "density"). No look exposes a curated set. | **Critical** (the user's main complaint) |
| Direct manipulation on the picture | Resolume 7.21 preview transform widgets, HeavyM canvas, MadMapper stage | Only for mapping corners (good there). Nothing for orientation, direction, centre, bend. There is not even a static orientation param: `rotate` is a speed. | **Critical** |
| Layer x column clip grid, column = scene | Resolume, Millumin | 16 scene pads only | High |
| Real picture transitions (dissolve, wipe, portal) | Resolume Auto Transition, VDMX two-channel mixer, gl-transitions | Param morph only (`_start_recall`, `set_xfade`). A morph between two unrelated looks passes through ugly in-between states (folds 4 -> 16 sweeps through 10). | High |
| Preview/cue vs program, blind edit | Resolume preview monitor, HeavyM Edited vs Played | Everything you touch is live on the wall | High |
| Live clip thumbnails | Resolume, Synesthesia, VDMX media bin | Per-layer thumbs exist (PreviewTap, PREVIEW off by default); no clip thumbs | Medium (cheap to add, mechanism exists) |
| Effect chains: order, wet/dry, bypass, presets | Resolume 4 scopes, VDMX chains | One master chain: 6 slots (`flt_slot1..6`), amount = wet/dry, bypass per tab. No chain presets, no per-clip FX. | Medium |
| Preset hierarchy + show files | VDMX presets at every level, MadMapper Collect and Save | `scenes.json` + `live_state.json` only. No show file, no clip library, no presets below scene level. | High |
| Timeline / cue list | MadMapper 6 Conductor, Millumin timelines | Being built now (Opus+Neon agent) | In progress: integrate |
| Modulation | VDMX data sources, Resolume parameter animation | **Strong**: 2 mod slots per param, audio bands, BPM LFOs, S&H, sequencer, crowd camera sources, stack targets | Small: per-link curve/smoothing, envelope presets |
| Audio analysis | Synesthesia syn_* uniforms | bass/mid/high + beat + kick bloom. No spectrum texture, no per-band onset "hits". | Medium (needed for ISF `audioFFT`) |
| Clock / BPM / Link / MIDI clock | all | **Strong** (tap, MIDI clock, Link, AUTO, quantize) | None |
| MIDI learn + LED feedback | Resolume, VDMX | Learn is good; LED feedback best-effort, untested on hardware | Small |
| OSC / OSCQuery | VDMX, MadMapper, Synesthesia | None | Medium (defer; HTTP `/cmd` covers TouchOSC-style use poorly) |
| Mapping | MadMapper, Resolume Advanced Output | **Strong for one projector**: multi-surface, masks, span, glow, undo, AUTO MAP. No mesh/Bezier warp, no multi-projector soft-edge. | Small for this rig |
| Shader import | ISF in VDMX/Resolume/Millumin/MadMapper | None | Medium (section 6) |
| Generative content quality | Synesthesia, Notch | **Strong and distinctive** (Folding Blocks is the user's favourite look) | None: protect it |
| Show safety | Resolume panic, TD Perform Mode | **Better than the pros**: watchdog, PANIC/SAFE scene, governor, photosensitive caps in stack/glitch/assistant. Missing: the engine-wide output flash limiter (queued). | Small, but a prerequisite for real transitions |
| Discoverability | Synesthesia (few controls, good names) | 15 tabs, SIMPLE/FULL toggle, BUILD tab the user rejected | High |
| Hardware headroom | n/a | GTX 1650 Ti 4 GB, thermal throttling, scene pass ~5.9 ms live. Two full-res scene renders do not fit next to the passes. | Constraint that shapes every decision below |

---

## 2. Target information architecture

### 2.1 The core trick: lanes own params, clips write base values

The engine is a single-instance renderer: one SCENE_FS ubershader, one FractalPass, one LandPass, one BeingsPass, one
global param namespace. Resolume's "N layers each playing any clip" model would need N engines. We do not have the GPU
for that, and we do not need it. Instead:

- **Every param belongs to exactly one LANE** (an ownership partition, enforced by a test like
  `test_every_param_is_classified_exactly_once` in test_bypass.py). A lane is a row of the clip grid.
- **A clip belongs to one lane and writes only that lane's params.** Launching it morphs (or blends, 2.4) the lane's
  params to the clip's values: params listed in the clip get the clip's value, every other param of the lane gets its
  registry default. So a clip always looks the same, whatever was on before.
- **Clips write `Controller.base`** (the same values faders write). Nothing downstream changes: mods still run on top,
  bypass still pulls to NEUTRAL after mods, the governor still overrides frame values only, scenes still snapshot
  `base`, MIDI CC numbers still address params, autosave still works. The params registry stays append-only and is
  never renamed. This is the decision that makes the redesign safe.

| Lane | Grid row | Owns (by params.py group, exceptions listed) |
|---|---|---|
| **BASE** (the world) | 1 | space, shape, color except `palette hue saturation trails`, motion except `serpent*`, fractal, art, land except `swarm_*`, layers `l_kene l_beads l_lines l_ai l_video luma_key l_fractal l_art l_land l_stack` (+ the stack JSON) |
| **FIGURES** | 2 | `serpent*`, `l_beings` + beings group, `l_swarm` + `swarm_*` |
| **TEXT** | 3 | `l_text` + text group |
| **INTERACT** | 4 | `l_camera` + camera group (camera ON/OFF stays a privacy switch, never set by a clip) |
| **MASTER** (no clips) | right column | master group + triggers, `palette hue saturation`, `l_echo trails`, filters group, `img_auto img_hold` |

Palette and FX are master by default (Synesthesia's Meta Controls). A clip may carry a `palette` and an `fx` preset,
applied only if the lane's PALETTE LOCK / FX LOCK is off (per-lane toggle, default: palette follows the clip on BASE,
FX never follows). Real kene study layers (`kene_real.py`, in progress) join BASE when they land.

### 2.2 Objects and hierarchy

| Object | What it is | Stored |
|---|---|---|
| **Param** | registry entry, frozen order (MIDI CC) | params.py |
| **Clip** | lane + partial `sets {param: value}` + optional `stack` + `mods` + 4-8 `macros` + `handles` + optional `palette`/`fx` preset + default transition + poster thumbnail + tags | `library/clips/<id>.clip.json` |
| **Clip preset** | a named snapshot of one clip's macro values (Synesthesia preset). Shown as variant dots on the tile. | inside the clip |
| **Deck** | a page of the grid (Resolume deck): columns x 4 lanes of clip refs. "Warm-up", "Peak", "Journey". | inside the show |
| **Scene** (column) | per lane: clip id / KEEP / STOP, macro overrides, master overrides (palette, fx preset), transition. Clicking a column header fires it, quantized. | inside the show |
| **Snapshot** | today's 16 scene slots (full `base` + mods + stack). Kept as-is, recalled by the old path, so cue-channel notes 0-15 and saved shows keep working. | `scenes.json` (unchanged) |
| **Show** | decks, scenes, clip copies (collect-and-save), timeline, BPM, autopilot, MIDI map ref | `shows/<name>.show.json` + `shows/<name>/thumbs/` |
| **Venue** | surfaces.json + calibration: NOT in the show (a show moves between venues) | unchanged |

Show files use the existing hardening pattern: atomic write, `.bad` quarantine on corrupt, read-only when unreadable,
headless never writes. `live_state.json` keeps the running state and gains `comp` (which clip is on each lane, macro
values) so a crash restart resumes the composition.

### 2.3 The timeline (integrate, don't duplicate)

The timeline agent owns timeline UI and transport. This plan only fixes the addresses it can target, all through
`Controller.apply` like every other command source: `{"cmd":"clip","op":"launch","id":..}`,
`{"cmd":"scene_go","id":..}`, `{"cmd":"macro","lane":"base","n":3,"v":0.7}`, plain `set`, and snapshot recall.
Ids are short stable strings (never list indices), so reordering a deck never breaks a timeline. The show file embeds the
timeline document under `timeline` (schema owned by that agent). Phase 0 checks the agent's design before Phase 1
freezes the ids; if it keys on scene slot numbers, snapshot slots stay addressable by number and clips/scenes by id.

### 2.4 Composition and transitions

On air at any moment: one clip per lane (or none) + master. Transitions are per lane, quantized to beat/bar/phrase:

- **MORPH** (today's machinery, per lane): params interpolate, stepped params flip at the midpoint. Free. Default when
  both clips use the same generator family (two Folding Blocks clips), where a morph looks musical.
- **BLEND** (new, BASE lane only): the engine renders the incoming clip's BASE values through SCENE_FS a second time
  into a private **half-scale** FBO (the PreviewTap trick at 640x400, about 1.5 ms instead of 5.9 ms; the softness is
  invisible mid-transition), and a transition pass mixes old and new pictures: Dissolve, Luma wipe, Radial, Swirl,
  Zoom-through (ported from MIT gl-transitions, attributed), plus **White Portal** (spec A7: >= 1 s swell, through the
  limiter). FIGURES/TEXT/INTERACT are identical in both renders, so they never double. The echo buffer sees only the
  blended result. The second render ends with the transition.
- **Pass conflict rule:** offscreen passes are single-instance and stateful (beings herd, swarm trails). If both clips
  need the same pass with different values, that pass's params MORPH while the rest BLENDs. If the governor reports
  strain, BLEND degrades to MORPH + a 0.3 s dip through dark (never a hard cut).
- Default transition per clip, overridable per scene column and by a transition picker in the master section (type +
  length in beats, 1/2/4/8/16).
- The existing scene A/B crossfader becomes the **composition crossfader** in MORPH mode; with BLEND on it scrubs the
  BASE transition by hand (Resolume crossfader).

---

## 3. Macro knobs

Each clip exposes 4-8 macros (Synesthesia CONTROLS, Notch exposure). A macro is 0..1 and drives 1-4 targets of its own
lane: `{"name":"Bloom","widget":"knob","default":0.3,"targets":[{"p":"zoom","a":-0.4,"b":0.5,"curve":"smooth"}]}`.

- **Curves:** `lin`, `exp` (x^2: fine control at the bottom), `log` (sqrt), `smooth` (smoothstep), `bell` (peaks at 0.5:
  "a little wobble, then calm"), `inv`, `step:N` (stepped params; snaps, with the option labels shown on the knob).
- **Widgets:** `knob`, `slider`, `xy` (two macros on one pad), `dir` (angle dial + length = direction and speed),
  `toggle`, `bang` (momentary: ripple, punch), `select` (stepped param with labelled tiles).
- **Targets** are param names or the existing dynamic targets `stack:<id>:<field>` (and later `isf:<clip>:<input>`).
  A macro may not target another lane or any master/trigger param (so no macro can strobe). Stack SAFE_HZ caps still apply.
- **Write path:** a macro move becomes `set` on each target's base value, counted in `Controller.edits` (the user wins
  over the governor), recorded in engine-side undo. A fader move on a macro's target puts that macro in "overridden"
  state (hollow ring) until it is moved again: last writer wins, nothing fights.
- **MIDI:** new targets `macro:<lane>:<1-8>`. A controller bound to macros always plays "whatever is on air" (Resolume
  Dashboard). After gate G1 the APC mini mk2 faders 1-8 get a MACRO MODE map (the current map stays selectable).
- **MOD** works on macros too (a macro LFO sweeps all its targets coherently).
- **Acceptance rule for every shipped clip:** each macro, swept 0 -> 1 with everything else at default, must change the
  picture by >= 3 % mean abs pixel difference (reuse `impact_audit.py`). A macro that does nothing does not ship.

### Examples (real param names and ranges)

**1. Folding Blocks Mandala** (BASE; the default show look; at default macro values it must be pixel-identical to the
current default, golden test)
| Macro | Widget | Targets |
|---|---|---|
| Bloom | knob | `zoom` -0.4..0.5 smooth; `density` 0.4..0.6 |
| Petals | select | `folds` step: 4, 6, 8, 12, 16 |
| Spin | dir | `spin` -0.5..0.5 bell-centred (0 at the middle) |
| Melt | knob | `swim` 0..0.8 exp; `twirl` 0..0.35 |
| Weave | knob | `l_beads` 0..0.9; `l_lines` 0.2..0.8; `l_kene` 1..0.75 |
| Light chase | knob | `chase` 0..0.75; `chase_width` 0.6..0.25 |
| Breath | knob | `ripple` 0.2..1; `bulge` 0..0.3 smooth |
| Fly in | knob | `travel` 0..0.6 exp (with `travel_mode` 0 fixed in the clip) |

**2. Folding Blocks Crawl** (BASE, `symmetry` 0): Direction+speed (`dir`: angle -> `flow_dir` 0..1, length -> `flow`
0..1), Scale (`zoom` -0.6..0.6), Wave (`wave` 0..0.7), Twirl (`twirl` -0.6..0.6 bell), Density (`density` 0.2..0.9),
Chase (`chase` 0..0.7).

**3. Tube Flight** (BASE, `travel_mode` 1): Speed (`travel` 0..1 exp), Bend (`tunnel_curve` 0..1), Distance
(`depth_fog` 0.8..0.2), Corkscrew (`spin` 0..0.6 + `spin_twist` 0..0.8), Petals (`folds` step), Warp (`bulge` -0.3..0.5).

**4. Crystal KIFS** (BASE, `l_fractal` 1, `fractal_mode` 0, pattern layers 0): Morph (`fractal_morph`), Detail
(`fractal_detail` 0.3..1; the governor may still cap it), Glow (`fractal_glow`), Iridescence (`fractal_metal`), Turn
(`rotate` -0.5..0.5 bell), Breath (`ripple` 0..0.8).

**5. Torus Wormhole** (BASE, `l_land` 1, `land_a` 1): Speed (`travel` 0.1..1), Twist (`land_twist` -1..1 bell),
Curvature (`land_curve`), Geo morph (`land_morph`), Lens (`land_fov` 0.2..0.9), Camera (select `land_cam`: Cruise /
Orbit / Audio dive), Prism (`land_disp` 0..0.8). The swarm is a FIGURES clip, not a macro here.

**6. Spirit Herd** (FIGURES, `l_beings` 1): Creature (select `being`, 17 tiles), How many (`being_count` step 1..8),
Size (`being_size`), Wildness (`being_speed` 0.2..0.9 + `being_wiggle` 0.3..1), Glow (`being_glow`), Variety
(`being_mix`), Colour (select `being_color`).

**7. Title** (TEXT): Size (`text_size`), Position (`xy`: `text_x`, `text_y`), Tilt (`text_tilt` bell), Movement (select
`text_move`), Speed (`text_speed`), Weave (`text_weave`), Dissolve (`text_dissolve`).

---

## 4. Direct manipulation (grab and drag)

### 4.1 Where the gizmos live

A **canvas** in the PLAY view (large monitor area) and in the EDITOR view. Because PREVIEW is off by default (and must
stay off by default), the canvas works in two modes:
- **Schematic** (PREVIEW off, default): black canvas in the projector's aspect with the gizmos drawn as vectors and the
  clip's poster frame at 25 % behind them. The performer looks at the wall. Zero GPU cost.
- **Over preview** (PREVIEW on): the same gizmos over the live monitor (or the cue render for an off-air clip).

Gizmos are declared per clip (`handles` in the clip JSON) from a fixed set of types; each maps to params through
`handles.py`, which holds a **forward model** (param values -> screen position) and its **inverse** (pointer -> values),
so the handle stays under the finger.

### 4.2 Gizmo set

| Gizmo | Looks like | Drag does | Params |
|---|---|---|---|
| Move | crosshair at the centre | moves the pattern centre | `pan_x`, `pan_y` |
| Orient | ring around the centre with a notch | sets a static angle (Ctrl = snap 15 deg) | **new** `orient` (0..1 turns, appended, NEUTRAL 0), added to the rotation phase |
| Spin | outer platter ring | drag around = spin speed; flick and release = keeps spinning at the flick's rate; tap = stop (eases 0.5 s) | `spin` (or `rotate` on looks that use it) |
| Scale | four corner brackets | drag out = zoom in (log, so equal drags feel equal) | `zoom` |
| Direction+speed | arrow from the centre | angle = direction, length = speed; a dead zone near the centre = stop | `flow_dir`+`flow` (flat), `travel` (tunnels: arrow points into or out of the screen as a dot/cross) |
| Bend | two curved handles on a guide circle | tangential drag = twirl, radial drag = bulge | `twirl`, `bulge` |
| Tunnel bend | the vanishing point plus a curved guide | drag the point = where the tunnel goes; the curve's sag = bend | `tunnel_curve`; vanishing point: `pan_*` if it already moves it, else **new** `vp_x vp_y` (Phase 0 checks SCENE_FS) |
| Kaleido centre | small mirror icon | moves the kaleidoscope centre; a wheel turns the fold count | **new** `flt_kal_cx flt_kal_cy`; `flt_kal_folds` |
| Stack band | two parallel guide lines across the layer, one band highlighted | drag the line spacing = band count, drag across = angle, drag the highlighted band's edge = thickness, drag the edge blur = softness | `stack:<id>:freq/angle/duty/soft`, `cx/cy` |
| Gradient | a line with stop dots along it | drag the ends = gradient direction and length; drag stops = colour positions; tap a stop = colour picker | stack `stops`, `angle`, `freq` |
| Orbit (3D decks, art relief) | trackball circle | drag = yaw/pitch, wheel / pinch = dolly | `cam_yaw cam_pitch cam_dolly`, land camera |
| Text box | bounding box | move, corner = size, side knob = tilt | `text_x text_y text_size text_tilt` |

New params (`orient`, `flt_kal_cx`, `flt_kal_cy`, maybe `vp_x`, `vp_y`) are appended at the end, each classified in
`NEUTRAL`/`BYPASS_KEEP` and in the lane table, no CUE_CC number until the user wants one.

### 4.3 Input rules (mouse, touch, pen: one Pointer Events path)

- Hit radius 12 px mouse, 24 px touch. The nearest handle wins; the canvas background drags Move.
- **Shift** = fine (1/8 speed, same as faders). **Ctrl** = snap (15 deg angles, whole folds, centre lines).
  **Alt** = no snap. Double-click / double-tap a handle = reset to the clip's value. Right-click / long-press = menu:
  reset, MOD, MIDI learn, expose as macro, lock.
- Two fingers: pinch = Scale, twist = Orient, two-finger drag = Move. Wheel = Scale, Shift+wheel = Orient.
- Arrow keys nudge the selected handle (Shift = fine), like the MAPPING tab.
- The solid handle shows the **base** value; a teal ghost shows the **live modulated** value (same language as the
  fader's teal line). A bypassed lane draws its gizmos dimmed with an OFF tag; drags still edit base.

### 4.4 Live feedback and undo

- While dragging, the panel sends `set` batches at most 30 Hz (requestAnimationFrame-coalesced), each tagged with a
  gesture id `g` (the surfaces.py pattern). The engine already slews params, so 30 Hz looks continuous.
- **Engine-side undo** (new `Controller.history`, 50 steps, gesture-coalesced, survives a panel reload, Ctrl+Z / Ctrl+Y
  in the panel). Clip launches, macro moves, gizmo drags and fader drags are all undo steps. This replaces BUILD's
  per-browser undo.
- If the clip is off air (editing in the EDITOR), drags edit the clip document only (blind edit, HeavyM style) and the
  canvas shows the cue render if PREVIEW is on.
- **Acceptance test (headless):** render a marker, drag the Move/Orient/Scale handle in a scripted panel session, render
  again: the feature that was under the pointer at drag start is within 2 % of frame width of the pointer at drag end.

---

## 5. Live thumbnails

- **Poster frames always:** every clip gets a 256x160 JPEG when it is saved, plus an 8-frame strip (2 s) rendered
  **offline** by a `clipthumbs.py --batch` job that reuses the `--batch` gallery machinery. Strips are rendered only when
  no show is live (or under `.locks/gpu_suite` with the show stopped). Hovering a tile plays its strip (pure browser).
- **Live thumbs only when PREVIEW is ON** (default OFF stays). New PreviewTap job `("clips", [ids])`: SCENE_FS drawn
  with each clip's BASE values into a 512x80 **atlas** of four 128x80 tiles: one draw sequence, one async PBO readback,
  one JPEG, one GET (`/clip-thumbs`). The panel reports which tiles are on screen; only those are scheduled, round-robin.
- **Budget:** at most 1 atlas job every 0.5 s (about 8 thumbs/s), at most one job per frame, measured cost logged by the
  governor's timers. A 128x80 SCENE_FS draw is about 1/100 of a full frame; the readback and JPEG are the real cost and
  are already async.
- **Governor-aware:** strained = 1 atlas per 2 s; critical, a BLEND transition running, or recording at 60 fps = none.
- **Pass clips** (fractal, 3D decks, art, beings, ISF): these passes are single-instance and stateful, so they never
  render off air for a thumbnail. Their tile shows the live picture only when that clip is the one on air, else the
  poster strip. This is honest and cheap. Cue preview of an off-air pass clip is allowed only while that pass is idle on
  air, at 0.25 scale and 5 fps, and only in the EDITOR.

---

## 6. ISF support

**Recommendation: YES, in Phase 8, after the clip model ships. Scope it as an import path for a curated library, not as a
live-coding feature.**

- **Libraries to bundle:** gl-transitions (MIT) for transitions, starting in Phase 3. ISF generators/filters from
  github.com/Vidvox/ISF-Files: the survey says MIT, but many files in that repo carry their own author headers and some
  are ports from Shadertoy/GLSL Sandbox. **Audit per file.** Ship only files whose header (or the repo licence, for
  VIDVOX-authored files) is MIT/BSD/CC0, and record each in `library/isf/PROVENANCE.csv` (same pattern as
  research/kene_reference). Never bulk-import editor.isf.video or Shadertoy (CC BY-NC-SA, incompatible with AGPL-3.0).
  User-imported files go into `library/isf/user/` with a licence notice and are never committed to the public repo.
- **Spec subset (ISF v2):** INPUTS `float bool long color point2D event image audio audioFFT`; PASSES with `TARGET`,
  `WIDTH/HEIGHT` expressions, `PERSISTENT`, `FLOAT`; built-ins `TIME TIMEDELTA RENDERSIZE PASSINDEX FRAMEINDEX DATE`,
  `isf_FragNormCoord`, `IMG_*` macros. Not in v1: custom `.vs` vertex shaders, `IMPORTED` from outside the shader's own
  folder, `#version` above 330.
- **Translation:** a GLSL 330 preamble (`out vec4 isf_FragColor; #define gl_FragColor isf_FragColor`,
  `texture2D -> texture`, `varying -> in`, image size uniforms). `TIME` is wrapped on the CPU in float64 like every
  other phase. `image` inputs bind to engine sources (art piece, AI texture, the previous output frame). `audioFFT`
  needs a new 256x2 R16F spectrum texture from the existing audio thread (about 1 h of work, also useful elsewhere).
- **Where it plugs in:** an ISF **generator** is a BASE clip rendered by one `ISFPass` (new appended params `l_isf`, NEUTRAL 0, and
  `isf_res`, so the governor's existing `<x>_res` rule covers it for free), composited in SCENE_FS on its own texture
  unit. An ISF **filter** is one new `FILTER_TYPES` entry `isf` (appended, plus `flt_isf` amount = wet/dry), one ISF
  filter in the chain in v1. ISF inputs are **not** params: they use the dynamic target namespace
  `isf:<clip>:<input>` (same contract as `stack:<id>:<field>`: MIDI learn, mods, macros), so the frozen registry is untouched.
- **Compile safety (this matters on a 4 GB laptop):**
  1. **Validate only while no show is live.** A GPU hang in *any* process triggers a Windows TDR that resets the
     driver under the show as well. Import is therefore an offline step: `isf_check.py` runs in a subprocess with its
     own hidden context, compiles, renders 30 frames at 160x100 with random inputs, measures GPU time with a timer
     query, checks for non-finite output, extrapolates the cost to 1280x800 at `isf_res` 0.5, and rejects anything over
     4 ms or that takes longer than 5 s (the subprocess is killed). Results are cached by file hash.
  2. **At show time, compile at show load, never at launch.** All ISF programs in the loaded show are compiled and
     warmed with one dummy frame during startup (progress shown in the panel). A clip added mid-show compiles in a
     hidden GLFW window sharing the show's context on a worker thread. If moderngl makes that unreliable, it compiles
     between frames and accepts one hitch when the clip is *added to the grid*, never when it is launched.
  3. A compile or runtime failure keeps the previous picture, shows the line-numbered error in the EDITOR, and marks the
     tile red. Five failed frames in a row unload the pass (the existing `fails` counter pattern in `extra`).

---

## 7. Curated starter library (launch set)

Cost: L < 1 ms, M 1-3 ms, H > 3 ms (GTX 1650 Ti, 1280x800, from HANDOFF measurements and gemini_effect_techniques
estimates). "Ready" = existing engine look, needs only a clip file and macros.

| # | Clip | Lane | Source | Cost | Needs |
|---|---|---|---|---|---|
| 1 | Folding Blocks Mandala (default) | BASE | engine | L | ready; golden pixel test |
| 2 | Folding Blocks Crawl (flat) | BASE | engine | L | ready |
| 3 | Zoom Tunnel | BASE | engine travel 0 | L | ready |
| 4 | Tube Flight | BASE | engine travel 1 | L | ready |
| 5 | Depth Layers | BASE | engine travel 2 | M | ready |
| 6 | Droste Spiral | BASE | engine travel 3 | L | ready |
| 7 | Beadwork Medallion | BASE | engine beads | L | ready (`l_beads` is weak in the impact audit: macro must boost it) |
| 8 | Flower-of-Life Lines | BASE | engine lines | L | ready |
| 9 | Jewel Pulse | BASE | factory scene | L | convert snapshot |
| 10 | Crystal KIFS | BASE | FractalPass 0 | M | ready |
| 11 | Mandelbox Tunnel | BASE | FractalPass 1 | M | ready |
| 12 | Hyperbolic Tiling | BASE | FractalPass 2 | M | ready |
| 13 | Julia Jewels | BASE | FractalPass 4 | M | ready |
| 14 | Sacred Corridor | BASE | LandPass 0 | M | ready |
| 15 | Torus Wormhole | BASE | LandPass 1 | M | ready |
| 16 | Prismatic Vortex | BASE | LandPass 2 | M | ready |
| 17 | Hex Dome | BASE | LandPass 3 | M | ready |
| 18 | Art Fly-through | BASE | ArtPass use 0 | M | an uploaded piece (clip stores the art name) |
| 19 | Art Kaleidoscope | BASE | ArtPass use 2 | L | same |
| 20 | Dream Tunnel Walls | BASE | ArtPass use 3, AI source | L | AI texture |
| 21 | Op-Art Rings | BASE | redesigned stack | L | Phase 7 stack fixes |
| 22 | Flower-of-Life Dome (A1) | BASE | new LandPass deck 4 | L-M | WORLDS build |
| 23 | Morphing Girih Corridor (A2) | BASE | new deck 5 | M | WORLDS build |
| 24 | Stained-Glass Hyperspace (A6) | BASE | new 2D pass or deck 6 | M | WORLDS build |
| 25 | Electric Chapel (A3) | BASE | new deck 7 | H | WORLDS build; half res |
| 26 | Void Webs (A5) | BASE | new deck 8 | M | WORLDS build |
| 27 | Breathing Bead Chamber (C10) | BASE | new deck 9 | H | ENTITIES+BEADS build |
| 28 | Fluid Silk (D11) | BASE | new particle pass | H | FLUID INK build (transform feedback) |
| 29 | Serpents | FIGURES | engine serpents | L | ready |
| 30 | Spirit Herd | FIGURES | BeingsPass | M | ready |
| 31 | Kitten Chase | FIGURES | BeingsPass 16 | M | ready |
| 32 | Gold Swarm | FIGURES | swarm | M | ready |
| 33 | Golden Filigree Entity (B8) | FIGURES | new pass | H | ENTITIES build |
| 34 | Stained-Glass Face (B9) | BASE | JFA filter on an image | M (cached JFA: L) | ENTITIES build |
| 35 | Title Drift / Tunnel Words | TEXT | TextPass | L | ready |
| 36 | Crowd Glow Outline | INTERACT | camera | L GPU, CPU worker | ready; camera ON is a separate switch |
| T | White Portal | transition | new transition shader | L | Phase 3 + engine flash limiter |

Rules: every clip passes the macro impact rule (section 3), the black-floor/limiter checks, the long-show wrap test
pattern its source already has, and a 60 s cost measurement logged in the clip file (`cost_ms`), which the grid shows as
a small heat dot so the performer knows what is heavy before launching it.

---

## 8. Phased build plan

Constraints honoured throughout: at most **3 concurrent agents** (reviewers count), GPU work only under
`.locks/gpu_suite`, no GPU work while a show is live, Opus reviews every Fable output, Gemini Neon is the second
reviewer, the default Folding Blocks look stays pixel-identical (golden test in every phase), one agent at a time edits
`live_visuals.py` and `panel.html` (declare ownership per wave in HANDOFF.md). New panel code goes into separate files
served by remote_control.py (`panel_play.js`, `panel_editor.js`, `panel_gizmo.js`), not into the 3302-line panel.html,
so parallel agents stop colliding. Estimates are agent working hours, excluding review.

**Vertical slice first.** The user rejected BUILD after seeing it. The first gate shows a small, finished-feeling slice
(4 clips, macros, gizmos, a mini grid) before anything is mass-produced.

### Phase 0: prep (Opus, 3 h)
- Land in-flight work, full suite under the lock, record baseline costs (scene ms, PreviewTap ms).
- Read the timeline agent's design and agree the ids in section 2.3 with it (a note in HANDOFF.md).
- Check in SCENE_FS whether `pan_*` moves the tunnel vanishing point and the mandala centre (decides `vp_x/vp_y`).
- Acceptance: baseline numbers + the two decisions written into HANDOFF.md.

### Phase 1: clip model (Opus, 10 h) — TDD
- Files: `clips.py` (Clip, Macro, Lane table, `validate`, `resolve(clip, macro values) -> base updates + stack + mods`),
  `show.py` (show save/load/migrate, collect-and-save), control.py hooks (`clip`, `macro`, `scene_go`, `comp` state,
  `macro:<lane>:<n>` and `clip:<id>` MIDI targets, `Controller.history` undo), remote_control.py validation.
- Tests first: every param in exactly one lane; a BASE launch never touches FIGURES/master params; unlisted owned params
  go to defaults; macro curves (all 7) and step snapping; macro targets outside the lane refused; macro moves count in
  `Controller.edits`; launch does not change bypass (same decision as scenes); snapshot migration keeps all 16 slots and
  cue notes; show round trip; corrupt show quarantined; undo coalesces a gesture.

### Phase 2: vertical slice (Fable UI + Opus engine bits, 12 h) — GATE G1
- 2a (Opus, 3 h): append `orient` (+ `vp_x vp_y` if Phase 0 says so), hook into the rotation phase, bypass/lane classes;
  `handles.py` forward/inverse models for Move, Orient, Spin, Scale, Direction; headless marker-tracking test.
- 2b (Fable, 9 h): `panel_play.js` minimal PLAY view: 4 BASE clips (Mandala, Crawl, Tube Flight, Torus Wormhole) + 1
  FIGURES clip (Spirit Herd), macro bank (8 knobs, named, MOD indicator), schematic canvas with the five gizmos above,
  MORPH transitions with beat quantize, poster frames. Macros tuned by eye.
- Acceptance: golden test (Mandala clip at defaults = current default look); each macro >= 3 % impact; the 2 % drag
  tracking test; a headless-browser test for touch + mouse drags.
- **G1 (user, on the projector):** do the names make sense, do the gizmos feel right, does the motion feel right.
  Rework here is cheap. Nothing below starts until G1 passes (Phase 3 engine work may start in parallel).

### Phase 3: real transitions + engine flash limiter (Fable engine, Opus control, 14 h) — GATE G2
- 3a (Opus, 4 h): per-lane transition state machine (quantize, beats, curve, MORPH/BLEND choice, pass-conflict rule,
  governor fallback), crossfader scrub.
- 3b (Fable, 7 h): `transitions.py` (half-scale second SCENE_FS render into a private FBO, 5 gl-transitions ports with
  attribution, White Portal), echo isolation.
- 3c (Fable, 3 h): the queued engine-wide photosensitive limiter (output luminance change + saturated red, <= 3
  flashes/s, opt-out per show), required before BLEND and Portal ship.
- Acceptance: pixel-identical when no transition is running; BLEND adds <= 2 ms at B scale 0.5 (measured); echo never
  sees the B render; a forced strain falls back to MORPH + dip; the limiter caps a 1/16 strobe at 3 flashes/s.
- **G2:** the user watches MORPH vs BLEND vs Portal between 4 pairs of clips.

### Phase 4: full PLAY grid + thumbnails (Fable UI 10 h, Opus engine 5 h) — GATE G3
- 4a (Fable): decks, 4 lanes x N columns, scene column headers, clip presets as variant dots, master section (palette,
  FX chain with drag reorder = `flt_slot*` order, wet/dry = `flt_<type>`, ALL FX), transition picker, cost heat dots,
  hover strips, PANIC/BLACKOUT unchanged. Fold PERFORM and SCENES into it.
- 4b (Opus): PreviewTap `clips` atlas job, visibility reporting, budget + governor rules, `clipthumbs.py --batch`.
- Acceptance: PREVIEW off = zero clip jobs (extends `test_preview_is_off_by_default_and_one_state_poll`); <= 1 atlas
  per 0.5 s; strained = 1 per 2 s; pass clips never rendered off air; grid usable at 1366x768 without scrolling.
- **G3:** screenshot + 10 min of play by the user.

### Phase 5: starter library, engine looks (Fable 10 h, Opus spec 2 h) — GATE G4
- Clips 1-21, 29-32, 35-36 from section 7, with 6-8 macros each, handles, poster strips, `cost_ms`.
- Acceptance: macro impact rule, black floor, contact sheet of all clips + a macro sweep strip per clip.
- **G4:** the user reviews the contact sheet and macro names; cuts or renames.

### Phase 6: full direct manipulation (Fable 8 h, Opus 4 h) — GATE G5
- Remaining gizmos: Bend, Tunnel bend, Kaleido centre (append `flt_kal_cx/cy`), Orbit, Text box, Stack band, Gradient;
  two-finger gestures; right-click/long-press menu with "expose as macro"; over-preview mode.
- Acceptance: tracking test for every positional gizmo; undo of every gizmo; works on the touch laptop and a phone.
- **G5:** the user maps a box and plays with gizmos on the wall.

### Phase 7: clip EDITOR (the TouchDesigner layer) (Opus 8 h, Fable 8 h) — GATE G6
- Opus: `panel_editor.js` logic: clip document editing (blind edit off air), inspector of every owned param grouped as
  today's tabs, Notch-style "expose to macro N" (sets the range from two captured values), mod routings, FX preset,
  save as new clip / overwrite / preset, cue render (rules in section 5).
- Fable: the math stack becomes the "Pattern Stack" generator inside the editor: plain names (Bands, Line thickness,
  Edge softness, Wobble, Wobble size; raw field names stay internal), motion in beats (bands per beat) with a slew on
  changes, DUP makes a variant (phase + hue offset, Screen blend) instead of an identical copy, a per-generator preset
  shelf, the Stack band and Gradient gizmos.
- Acceptance: a clip built from scratch in the editor round-trips through save/launch/show file; BUILD tab gone with
  its features reachable; stack `test_stack*.py` still green.
- **G6:** the user builds one clip from scratch.

### Phase 8: ISF (Opus 8 h, Fable 6 h) — GATE G7
- 8a (Opus): `isf.py` parser + translator, `isf_check.py` sandbox, `isf:<clip>:<input>` targets, spectrum texture,
  PROVENANCE audit of the bundle.
- 8b (Fable): `ISFPass` (generators, multipass, persistent buffers), `isf` filter type, 15-25 curated shaders as clips.
- Acceptance: a malformed shader is refused with line numbers; an infinite-loop shader is killed in the sandbox and
  never reaches the show; nothing compiles at launch time (instrumented test); licences audited file by file.
- **G7:** the user picks which imported shaders stay.

### Phase 9: new worlds from the effects spec (3 Fable builds, already planned, each about 12-16 h)
WORLDS (A1-A7), ENTITIES+BEADS (B8-B9, C10), FLUID INK (D11). They now deliver **clips** (new LandPass deck indices
appended, never reordered) instead of bare faders. They are independent of Phases 1-8 except for the clip file format
(available after Phase 1), so they fill free agent slots. Each one has its own gate with stills + a short recording.

### Phase 10: show + timeline integration and hardening (Opus, 6 h)
Show browser (open / save as / recent), timeline embedded in the show, autopilot over scenes and clips (bag shuffle),
crash-restart resume of the composition, docs (QUICKSTART, WALKTHROUGH), full suite, a 2-hour soak on the projector.

### Agent schedule (at most 3 at a time; a reviewer occupies a slot)

| Wave | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|
| A | Opus P0 -> P1 | Fable P3b/3c transitions + limiter (engine; owns live_visuals.py) | Fable P9 WORLDS (own files: fx_stack decks) |
| B | Opus P2a + reviews | Fable P2b vertical slice (owns panel files) | Opus review of P3 / P9, then Neon |
| — | **G1, G2** | | |
| C | Opus P3a + P4b | Fable P4a grid | Fable P5 library |
| — | **G3, G4** | | |
| D | Opus P6 models + P7 logic | Fable P6 gizmos -> P7 stack | Fable P9 ENTITIES+BEADS / reviews |
| — | **G5, G6** | | |
| E | Opus P8a -> P10 | Fable P8b | Fable P9 FLUID INK / reviews |

Rough total: about 120-140 agent hours plus reviews, i.e. 2-3 weeks of evening sessions at this laptop's thermal pace.
G1 is about 2 sessions away.

---

## 9. Deprecate / fold

| Today | Becomes |
|---|---|
| BUILD tab, NEW CANVAS, RESTORE PREVIOUS | The EDITOR's Pattern Stack generator; NEW CANVAS = "New blank clip"; RESTORE = engine undo / "back to what was on air". `--fresh` opens the EDITOR on a blank clip. |
| SIMPLE / FULL switch | Two views: **PLAY** (grid, macros, master, canvas) and **EDIT** (clip editor). The old param tabs stay reachable as the editor's inspector and as an "All params" expert drawer, never as the default surface. |
| PERFORM tab | Folded into PLAY (its scene pads, macros row, live layers strip and quick boxes all have a home there). |
| SCENES tab (16 slots) | Snapshot row in PLAY + scene columns in decks. scenes.json is untouched. |
| Scene A/B crossfader | Composition crossfader (MORPH mode = today's behaviour). |
| Layers card (EYE / SOLO / thumbs) | Lane strip in PLAY: EYE = lane bypass, SOLO = preview solo (unchanged semantics). |
| 3D DECKS internal A/B (`land_xfade`) | Stays, as an in-clip feature of land clips (it is a good look generator, not a mixer). |
| MOD BUS, CONTROLLERS, MAPPING, AI DREAM, ASSISTANT, CAMERA | Stay as they are (utility tabs, reachable from a menu in both views). The assistant later gains `clip`/`macro` commands in its allowlist. |
| Raw stack field names in the UI (freq, duty, soft, warp) | Plain names; raw names remain the internal contract (`stack:<id>:<field>`). |
| APC mini default map | Kept; MACRO MODE map added after G1 and made default only if the user agrees. |

**Not deprecated, protected:** Folding Blocks and the overall look (golden test in every phase), params registry order,
bypass semantics, governor, PANIC/SAFE scene, PREVIEW off by default, ONLINE off by default, photosensitive caps.

## Addendum (2026-11-26, user request): environment SKINS + layer WINDOWING
User: "swap out the skin on visual 'environments' with images and AI generations, and control how the windowing occurs between the various parts that can be constructed in the visual space (the layers)."

### A. Environment skins (materials)
- Every environment/world (3D decks: corridor, torus, vortex, domes; tunnels/travel modes; future worlds from effects_spec; beings' body fill; Folding Blocks fill; kene study fill) gets 1-3 MATERIAL SLOTS (e.g. walls / floor / accents) whose source is: procedural (today's look, default), an uploaded art piece, an AI texture, a DREAMIFIED image, live camera (optional), or a math-stack layer rendered to texture.
- Per slot controls: mapping (UV along the surface / triplanar / screen-space / tunnel log-polar), scale, rotation, scroll speed + direction (beat-syncable), mirror/kaleido tiling (seamless: mirrored repeat so any image tiles without seams), tint/palette-follow, mix with the procedural pattern (multiply, screen, overlay, "pattern lines reveal the skin"), and AI-assisted seamless tiling (SD-Turbo circular-padding generation already exists → "generate a skin" button with a prompt).
- Engine: one shared material sampler path (texture array or atlas of up to N skins, mipmapped, anisotropic where cheap), no per-frame uploads; swaps crossfade; governor-aware resolution.
- Panel: drag an image/AI thumbnail from the Sources picker onto an environment's slot (or onto the live canvas region showing that surface) = assign; "skin presets" saved in show files.

### B. Layer windowing (compositing regions)
- Each layer/lane gets a WINDOW: shape (full, rectangle, rounded rect, circle/ellipse, polygon, star, ring, split halves/thirds, custom drawn path, or a MATTE from another layer's luminance/alpha or from the camera silhouette or a mapped surface), feather, invert, position/size/rotation (gizmos: drag edges/corners on the canvas), and animation (beat-synced iris open/close, wipe, slide, orbit, breathing).
- Composition rules: layer order + blend mode + window → e.g. tunnel inside a circular portal over kene, beings only inside the right third, art through a star window, split-screen two worlds, picture-in-picture.
- Window transitions: iris/wipe/portal between clips reuse the transition system (Phase 3).
- Per-surface windows already exist in mapping (masks); keep those for physical geometry and these for composition — documented difference; optionally "use mapped surface as window".
- Engine: window evaluation as an SDF per layer in the composite shader (cheap, analytic) + optional matte textures; zero cost when full-frame.
### Phasing
- Skins: after Phase 1 (clip model) as Phase 5b, or earlier as a self-contained engine feature (LandPass first) — owner Fable, review Opus.
- Windowing: with Phase 6 (direct manipulation) since it is gizmo-driven — owner Fable (engine+panel), Opus for the composition model.
