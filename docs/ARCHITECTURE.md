# Architecture

A concise map for developers. The guiding rule: **one engine, one control surface.** Every visual is drawn by `live_visuals.py` in one OpenGL context. Every control lives in `panel.html`, served by `remote_control.py`. New effects become params and passes inside the engine, never a second window over the show.

## Modules

| Module | Role |
|---|---|
| `params.py` | The shared contract: every param `(min, max, default, group, label)`, triggers, mod sources, option labels, and the bypass tables |
| `live_visuals.py` | Render engine (moderngl, GLSL 330): window / projector handling, audio analysis, the main scene shader `SCENE_FS`, the pass registry, limiter, output, all GL hooks |
| `control.py` | `Controller`: faders, mod matrix, clock (tap / MIDI / Link), scenes and morphs, autopilot, bypass, timeline evaluation, show / preset commands, command validation, `live_state.json` autosave |
| `remote_control.py` | Token-gated HTTP server: serves `panel.html`, `/state`, `/cmd`, previews, thumbnails, uploads, shows / presets / media routes |
| `panel.html` | The whole panel in one self-contained page, generated from `params` meta plus self-contained `<script>` blocks per feature |
| `midi.py` | MIDI ports, hot-plug scan, default APC mini mk2 / Launch Control XL maps, LED feedback |
| `ai_link.py` | Client of the AI worker (job queue, hold flag) |
| `ai_worker.py` | Separate process: SD-Turbo txt2img / img2img, Depth Anything V2 depth maps, Dreamify jobs (HTTP on 127.0.0.1:8766) |
| `dreamify.py` | DeepDream on GoogLeNet, run by the AI worker |
| `math_layers.py` | `FractalPass` |
| `art_layer.py` | `ArtPass`: art fly-through, art use modes, art library loader |
| `fx_stack.py` | `LandPass` (3D decks + swarm) and `PreviewTap` (panel monitor, layer thumbnails, solo) |
| `beings.py` | `BeingsPass` and the `Herd` motion model |
| `text_layer.py` | `TextState`, `TextPass` (SDF glyphs), the text compositor GLSL |
| `stack.py` | The BUILD math stack: model, validation, photosensitive caps, `StackPass` |
| `kene_real.py` | The two generative study layers (`KenePass`, `DesignBuilder`) |
| `materials.py` | Skins: model, library (texture array), engine side, GLSL chunk |
| `filters.py` | `FilterPass`: the post-FX chain plus `FINAL_FS` (limiter / master / blackout after filters) |
| `flashguard.py` | The photosensitive limiter (`FlashGuard`, `FlashTracker`, `GuardLogic`) |
| `surfaces.py`, `surface_render.py` | Multi-surface mapping model (pure Python) and its renderer |
| `automap.py` | AUTO MAP: patterns, decoding, calibration, detection, refinement, wizard, overlay |
| `camera_layer.py`, `camera_worker.py` | Crowd camera: engine side, and a separate process (segmentation, flow, pose) on 127.0.0.1:8767+ |
| `governor.py` | Resource governor and power balance |
| `recorder.py` | MP4 recording, replay buffer, crash recovery |
| `shows.py`, `timeline.py` | Show / preset files and the show worker thread; the timeline model, evaluator and transport |
| `clips.py` | Clip / macro / lane / deck model (controller side; panel not built yet) |
| `plugins/online/` | ONLINE plugin: assistant, safety allowlist, Gemini images and vision. Only `protocol.py` is imported while ONLINE is off. |
| `run_show.ps1` | Launcher and crash watchdog |

## Processes and threads

**Processes**: the engine; the AI worker (started by `run_show.ps1`); the camera worker (spawned only while the camera is on, exits by itself 15 s after losing its reader or its parent); ffmpeg encoders; a detached recording finalizer at exit.

**The render (GL) thread** is the engine's main thread. It owns every GL object and also runs `Controller.update()`, command application, timeline evaluation and the governor's per-frame hook. Rule: **nothing on this thread may block.** Anything slow goes to a worker and hands back results that the render thread picks up at the start of a frame.

**Daemon threads** (all non-blocking hand-offs to the render thread): audio capture, video decoder, HTTP server, MIDI scan and LEDs, Ableton Link, governor sensors, AI link, art loader, skin loader, text glyph rasteriser, kene design builder, camera link / start / stop, AUTO MAP wizard steps, recorder audio and writer, shows worker, online assistant / images / log, snapshot saves.

**Failure policy**: non-GL calls on the render thread go through `guarded(tag, fn, ...)`, which logs at a limited rate and never raises. A pass that fails 5 frames in a row is switched off for the session. Only GL failures exit non-zero, so the watchdog restarts the engine.

## One frame, in order

1. **Audio bands**: bass / mid / high, smoothed.
2. **Camera pre-update** (only while it is on): mod sources and hand ripples.
3. **`controller.update(dt, bands)`** returns a `Frame` of final, clamped values for every param, one-shot events and the clock. Inside it, in order: faders, mods, scene morph and drift, timeline automation (its own layer on top of the faders), then bypass pulls covered params to their neutral values.
4. **Commands**: the controller takes the ones it owns; engine commands (corner pin, snapshot, `show_texture`, `art_pick`, `surf`, `camera`, `automap`, `rec`, `gov`, `flashguard`) are handled here. AUTO MAP edits are applied on this thread.
5. **State publish**: the dict is stored and only serialised when a panel asks for `/state`.
6. **Governor** `frame(vals, base)`: may override frame values (never the stored faders).
7. **Offscreen passes** (`extra`), each only while its layer is above 0.001: fractals, art, 3D decks + swarm, beings, kené studies, text, math stack. Each renders into its own texture.
8. **Scene draw** (`SCENE_FS`): the warp chain (ripples, serpentine warp, wave, pan / rotate, punch, bulge, twirl, zoom / travel), the pattern layers, composites of the offscreen textures, camera, text, echo / feedback, then the performance stage (invert, flash, strobe), limiter, master and blackout. A skinned variant of the program is used only while a tunnel skin is active.
9. **Filters** (only while one is active): the scene is drawn with limiter, master and blackout neutral, the chain runs, then `FINAL_FS` applies limiter, master and blackout. On any failure the scene is redrawn the normal way in the same frame, so blackout is always black.
10. **Flash guard** `process(disp)`: rewrites the frame in place only while engaged.
11. **Taps**: Spout, panel preview / thumbnails / solo (only while a panel asks), snapshot.
12. **Output**: corner pin, or the multi-surface renderer (falls back to the pin after 5 failures and retries every 2 s).
13. **AUTO MAP overlay** (only during a calibration step).
14. **Recorder**: reads the program frame (before mapping) or the projector target (after).
15. **Brightness meter** (every 4th frame, pre-limiter level) and limiter step.
16. **Pacing** (power-balance cap or standby trickle), then swap.

Everything a viewer can see, record or receive passes the flash guard, except the AUTO MAP overlay, which has its own brightness cap and second-tap rule.
<!-- TODO(flashguard): output-order fixes (surfaces / calibration overlays) are in flight. Update steps 10-13 when they land. -->

### Texture units in use
Offscreen passes bind fixed units. Check this list before claiming one.

| Unit | User |
|---|---|
| 0-3 | Scene inputs (re-bound after filters) |
| 6 | Beings |
| 7 | 3D decks + swarm |
| 9-15 | ArtPass range (text uses 10 / 11 inside it, a known overlap); camera mask 12, flow 13; AUTO MAP overlay 14; PreviewTap private 15 |
| 16 | Math stack |
| 17, 20, 21 | Kené studies (output, private inputs) |
| 22 | Skins texture array |

## The params registry

`params.PARAMS` is the contract between the engine, the controller, the panel, MIDI, scenes and shows.
- **Never rename, never reorder.** Append new params at the end of the file (or the end of a group block). Names are stored in scenes, shows, MIDI maps and presets. The cue-channel CC numbers are frozen.
- Every param becomes a float uniform `p_<name>` in `SCENE_FS`, unless it is in `CPU_ONLY` (consumed by the CPU or by its own pass). A test fails if a param is neither.
- A stepped param with labels goes in `params.OPTIONS`. The controller merges it and the panel shows a button grid.
- **Every param must be classified for bypass**: either in `NEUTRAL` (the value at which it does nothing) or in `BYPASS_KEEP` (selectors, counts, palette, quality, master, triggers). `tests/test_bypass.py::test_every_param_is_classified_exactly_once` fails otherwise.
- A new group gets its own panel tab and power button automatically (appended after the existing tabs). Map it in `GROUP_TAB` if it belongs on an existing tab.
- Defaults must keep the default look unchanged. Prove it with a pixel-identity or golden test.

## Bypass
`Controller.bypass` is a set of keys: one per tab / group, one per bypassable param, and `fx` (ALL FX). In `update()`, after everything else, each covered param's output is pulled to `NEUTRAL` with a per-key fade (`bypass_fade`, default 0.3 s). Faders, mods, MIDI bindings and scenes are never edited. `BYPASS_LINKS` ties a tab's switch to its layer. Bypass is saved in `live_state.json`, not in scenes. PANIC leaves it alone, except that it clears the whole-`layers` bypass.

## Governor and power
`governor.py` watches frame timing against the display refresh, per-pass GPU timer queries, GPU utilisation / temperature / VRAM / throttling, CPU / RAM and battery, on its own sensor thread. Strain degrades in steps: render scales, then detail and counts, then AI hold, then minimum scales. Cheap passes are skipped. Overrides apply to frame values only, so scenes and autosave never capture a degraded look, and a direct fader move by the performer always wins. Power balance adds a separate ladder (fps caps 50 / 45 / 30, AI off the GPU, render scales, camera rate, advice) driven by net battery power, plus a standby trickle while nothing is visible. Passes can report cost through `*_source` hooks (camera, skins).

## Safety rules for new code
- Black is the default. Light only where a layer puts it.
- Nothing may blink faster than 2.5-3 Hz on its own: cap rates, slew brightness, ramp events. The flash guard is the last line, not the first.
- PANIC must always reach a non-black safe look. If your feature holds state that could keep the wall black or flashing, clear it in `Controller.panic()`.
- Long shows: integrate phases in float64 on the CPU and send them wrapped at a period where the pattern repeats. Test with `--phase-offset`.
- Zero cost when off: a layer at 0 must not render, allocate or upload.
- Privacy: camera frames stay in memory; ONLINE off means no import, socket, thread or log.

## Tests and TDD

Tests live in `tests/` (`pytest.ini` points there). Write the test first, see it fail, then implement.
- **Pure-Python tests** (models, validators, clocks, timelines, governor with fake sensors) run anywhere and fast.
- **GL tests** render tiny frames headlessly or run the real engine through `live_visuals.py --batch` / `--test`. They need the GPU.
- **Panel tests** drive `panel.html` in headless Chromium through Playwright against a real controller (`python -m playwright install chromium` once).
- **Golden / pixel-identity tests** prove a change did not move the default look.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_stack.py      # one file while developing
.\.venv\Scripts\python.exe -m pytest -q                           # the full suite
```

**GPU lock convention.** Only one process at a time may run the full suite or long GPU renders. Before starting, create `.locks\gpu_suite` containing your name. If it already exists, wait and retry every 60 s. Delete it when done. Never run GPU work while a live show is on the projector. Prefer targeted test files during development.

## How to add a layer
1. **Test first**: a test that the layer is off by default, costs nothing at 0, never outputs NaN, keeps black black, and survives a phase wrap.
2. **Params**: append `l_<name>` (group `layers`, default 0) and its controls in a new group at the end of `params.py`. Add every new param to `NEUTRAL` or `BYPASS_KEEP`, and option labels to `OPTIONS`. Add `BYPASS_LINKS["<group>"] = ["l_<name>"]`.
3. **Pass**: a module with `class XPass(ctx, out_size)` and `render(values, clock, t, dt, bands, palette_rgb) -> Texture` plus `release()`. RGB = light with the layer amount applied; alpha = how much it hides beneath. Keep CPU work off the render thread.
4. **Engine hook** (`live_visuals.py`, small and marked): a guarded import; register in `extra` with its param, sampler uniform and a free texture unit; add the group to `CPU_ONLY`; a composite line in `SCENE_FS` (for example `col = col*(1-a) + rgb`) at the right depth.
5. **Panel**: usually nothing. The group's tab and faders appear by themselves. Bespoke UI goes in one self-contained `<script>` block at the end of `panel.html` that listens to the shared `panelstate` event (never poll `/state` yourself).
6. **Governor**: if the layer has a `<name>_res` or count param, the governor can already scale it. Report cost through a source hook if needed.
7. **Look-check**: a `--gallery` in your module writing to `test_frames/<name>/`, and a run through `live_visuals.py --test`.
8. **Docs**: add the layer to [QUICKSTART.md](../QUICKSTART.md) section 5.

## Commands
Everything the panel, MIDI, the timeline or the assistant can do is a JSON command `{"cmd": ..., ...}` posted to `/cmd`. Each command family has a validator: the server validates, and the controller validates again (validators must be idempotent). Control commands are in `control.CONTROL_CMDS`. Engine commands pass through the controller to the render thread. Allowlists (timeline events, assistant) are explicit sets, so a new command is refused by them until added on purpose.
