# TouchVDMX Explorer inventory (merge into the one engine + one panel)

Source: `touchvdmx_explorer.html` (2407 lines, one self-contained browser page: Tailwind CDN + WebGL2). Read whole, never run
outside a headless check. It is a separate browser renderer with its own projector pop-out window, so nothing of it ships as is.
Verdicts: **HAVE** (ours already does it), **THEIRS** (better in theirs, ported), **MISSING** (we lacked it, ported),
**SKIP** (not worth it / not safe).

## Visual generators and effects

| Feature in the prototype | Verdict | Reason / where it went |
|---|---|---|
| Engine 1 "Sacred Corridor": raymarched vaulted corridor, star tessellation reliefs, axis glow | MISSING, ported | `fx_stack.py` CORRIDOR_FS, `land_a/land_b = 0`. Recoloured from the show palette, seam closed |
| Engine 2 "Inside-Out Torus" wormhole, 3 surface styles (electric chapel, floral glass, neon gears) | MISSING, ported | TORUS_FS, `land_style`. Their SDF overwrote itself four times and clipped the camera; rebuilt as tube + torus chambers |
| Engine 3 "Prismatic Vortex": Voronoi stained glass funnel, chromatic dispersion, speed streaks | MISSING, ported | VORTEX_FS. Their `time * audio` scroll jumped on every kick; scroll is now integrated on the CPU |
| Engine 4 "Hex DMT Dome": periodic honeycomb domes, sphere to cube to octahedron morph | MISSING, ported | HEX_FS, `land_morph`. Added a porthole on the flight axis: theirs flashed flat colour crossing each shell |
| Dual-landscape mixer: engine A + engine B with a crossfader | MISSING, ported | `land_a`, `land_b`, `land_xfade`; deck B costs nothing until the fader moves |
| 8 GPU blend modes (mix, add, screen, difference, multiply, exclusion, divide, luma mask) | MISSING, ported | `land_blend`, COMP_FS; crossfade ends are always the pure decks |
| Cross-modulation "A warps B" (deck A's picture bends deck B's rays) | MISSING, ported | `land_warp` |
| Curl-noise particle swarm, 8192 line ribbons, trail buffer, 2 palettes, vortex attractor | MISSING, ported | `fx_stack.Swarm` (numpy, 6000 filaments) + trail ping-pong; layer `l_swarm`, `swarm_vortex/turb/trail/tint` |
| Swarm shockwave pad | MISSING, ported | rides the existing `punch` trigger (no new trigger, no CC change) |
| "Hyperspace warp jump" pad (FOV kick + forward lurch) | MISSING, ported | `punch` widens the landscape FOV and lurches the camera |
| "Fold invert" pad (mirror-fold the surface pattern) | MISSING, ported | `land_mirror` |
| Twist / Curvature / Geo morph / Cam FOV knobs | MISSING, ported | `land_twist`, `land_curve`, `land_morph`, `land_fov` |
| Symmetry (folds) and Color cycle knobs | HAVE | our `folds` and `hue` drive the landscapes too |
| Feedback knob (ping-pong persistence) | HAVE | our echo/trails pass already sees the new layers |
| Strobe pad = chromatic inversion flash | HAVE | our strobe + invert + flash triggers, beat-gated |
| Filmic highlight roll-off | THEIRS, ported | in COMP_FS; our limiter still has the last word |
| Radial chromatic split of the whole picture | MISSING, ported | `land_disp` ("Prism split"), applies to every landscape |
| Camera modes: auto-cruise, orbit, audio dive | MISSING, ported | `land_cam` |
| Free fly (WASD, mouse look, Q/E roll) | HAVE | our fly pad + gamepad (`cam_yaw/pitch/roll`) now steer the landscape camera too |
| 4 camera waypoint bookmarks with fly-to | HAVE | scenes with morph do this for every param, not only the camera |
| Adaptive resolution scale selector | HAVE | same idea as `fractal_res`; added `land_res` |
| Webcam optical flow bending the rays | SKIP | needs a camera on the show laptop and a capture thread in the engine; note for later |
| Live GLSL editor (compile shader text from the page) | SKIP | compiling network-supplied shader text on the live GL thread is a crash and security hole mid-show |

## Control and modulation

| Feature | Verdict | Reason / where it went |
|---|---|---|
| BPM field, tap tempo, 4 beat LEDs | HAVE | ours adds MIDI clock, Ableton Link, downbeat, bar lights |
| "VDMX data-source bus": bind any knob to audio / LFO / sequencer / MIDI | HAVE | our 2-slot mod matrix on every param |
| Min/max window per binding | HAVE | our signed amount + fader position cover it |
| 2 BPM-synced LFOs (sine, tri, saw) | HAVE | ours: 4 shapes + sample-and-hold, per param, any rate |
| LFO scope canvas (see the modulation move) | THEIRS, ported | live waveform + playhead in the MOD editor; new MOD BUS tab lists every active modulation |
| 16-step sequencer (float lane + gate lane) as a mod source | MISSING, ported | mod sources `seq` and `seq_gate` (appended), pattern editor in MOD BUS, saved in live_state.json |
| 5-band audio (sub, low, mid, high, peak) | SKIP | engine analysis is 3 bands; sub and peak add little for visuals |
| Audio band bars | HAVE | plus a scrolling 3-band scope in MOD BUS |
| WebMIDI learn for knobs, notes 36-43 to pads | HAVE | ours is engine-side, hot-plug, with APC / LCXL maps and a cue channel |
| 8 "LPD8" pads: 4 scene presets + 4 momentary FX | HAVE | 16 scenes + 6 triggers + ripple |
| A/B crossfader as the centre of the rig | THEIRS, ported | new scene crossfader: load two saved scenes on decks A/B, blend by hand (panel + MIDI-learnable) |
| Save / load preset JSON download | HAVE | scenes.json + live_state.json autosave |
| PANIC = freeze toggle | HAVE | ours jumps to a safe scene and clears triggers; freeze is its own trigger |
| Built-in EDM synth as fake audio | SKIP | we capture real system audio; the demo server already fakes bands |

## UI ideas

| Feature | Verdict | Reason / where it went |
|---|---|---|
| The output is visible inside the control page | THEIRS, ported | output monitor in the top bar (click = large), fed by a 320x200 JPEG tap only while a panel watches |
| Projector pop-out window (captureStream) | SKIP | second window over the show is forbidden; our engine owns the projector |
| "Perform mode" hiding all chrome (F1) | SKIP | theirs hides UI over its own canvas; our panel is only UI |
| Rotary knobs (vertical drag) | SKIP | tall faders are faster and safer on a touchscreen |
| Collapsible right drawer with 3 tabs | HAVE | tab bar; new tabs 3D DECKS and MOD BUS appended after MAPPING so F1-F10 keep their meaning |
| Toasts instead of alert() | HAVE | banner |
| XYZ / YPR camera telemetry HUD | SKIP | debugging aid, no live value |
| FPS badge | HAVE | |
| Tailwind + Google Fonts from CDNs | SKIP | the panel must work with no internet at a venue |

## Bugs found in theirs (not carried over)
- Torus SDF assigns `d` four times; only the last line matters and it puts the camera outside the surface for part of each period.
- Hex domes: the camera passes through shell walls, giving full-frame flat flashes.
- Voronoi and star tilings have a visible seam at angle +-pi (cell ids not wrapped).
- `u_time * (2.2 + audioLow * 1.4)` makes the vortex jump by an amount that grows with show length.
- Time and camera z grow without bound in float32; ours wraps every phase on the CPU.
- Panic only pauses the page's render loop; pads and MIDI still mutate state underneath.
