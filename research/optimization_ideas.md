# Optimization ideas: GPU / CPU / power / VRAM / smoothness (read-only audit, 2026-11-25)

Scope: `live_visuals.py`, `fx_stack.py`, `beings.py`, `math_layers.py`, `art_layer.py`, `text_layer.py`,
`surface_render.py`, `governor.py`, `recorder.py`, `control.py`, `remote_control.py`, `panel.html`, `ai_worker.py`,
`ai_link.py`, `ai_gen.py`, `run_show.ps1`. Code was only read. **Nothing was run or measured** (battery was nearly
empty). Every saving below is a GUESS from code reading unless tagged `HANDOFF` (a number someone measured earlier
and wrote into HANDOFF.md). Treat the ranking as "what to measure first", not as results.

Not read in full: the bodies of the beings / fractal / art / text shaders (only loops and render paths), most of
`surfaces.py`, `midi.py`, `recorder.py` beyond the readback ring, most of `panel.html`.

## What is already fine (checked, so nobody re-audits it)

- **Vsync is on**: `glfw.swap_interval(1)` (live_visuals.py:1967). The show is capped at the projector's 60 Hz, it
  is not rendering uncapped. Headless tests are uncapped on purpose.
- **Layers at 0 cost nothing**: every offscreen pass is skipped while its fader is <= 0.001
  (live_visuals.py:2532-2547); SCENE_FS gates every layer on `if (p_l_x > 0.0)` (dynamic uniform branches, which
  the GPU skips coherently, so shader variants would gain little).
- **Raymarched passes already default to half res**: `land_res` 0.5, `fractal_res` 0.5 (params.py:91, 128); deck B
  only renders while `land_xfade` > 0; the swarm is fixed half res.
- **Readbacks are mostly async already**: PreviewTap (fx_stack.py:823-876) and the recorder (recorder.py:562-608)
  use PBOs. Remaining synchronous reads: the limiter meter (item 6) and the one-shot snapshot (on request only).
- **Projector unplugged**: rendering stops, loop sleeps 50 ms (live_visuals.py:2519-2522).
- **No Optimus copy for the projector**: HANDOFF says the projector output is driven by the NVIDIA GPU directly.
  (Check the `[gl]` line in the show log says GTX 1650 Ti, not Intel UHD.)
- Video decode stops when `l_video` is 0; art uploads are striped and time-budgeted; `land_res` changes do not reallocate.
- VRAM for render targets is small (about 8 MB per full-size RGBA16F target; all engine targets together are well
  under 200 MB). The VRAM problem is the AI worker, not the engine.

## Ranked table

Score = (expected saving x confidence) / effort, judged by eye. Effort S = under 1 h, M = half a day, L = days.

| # | Idea | Where | Estimated saving (all guesses) | Risk to the look | Effort |
|---|---|---|---|---|---|
| 1 | Check for a pegged CPU core in the vsync wait; cap CPU turbo for shows (Windows power mode / max processor state 99 %) | OS setting; `swap_buffers` live_visuals.py:2721 | Possibly the largest heat item: one i9 core spinning in turbo is 15-25 W. Unknown until Task Manager is looked at | none | S (no code) |
| 2 | `pip install nvidia-ml-py` so the governor stops spawning `nvidia-smi` every 2 s | governor.py:297-328 | one process spawn per 2 s: about 2-5 % of a core, and a periodic hitch source | none | S (no code) |
| 3 | Skip texB taps while not cross-fading (`mixAB == 0`, about 75 % of the time) | live_visuals.py:401-406, 229 | 3 of 6 anisotropic `textureGrad` taps per pixel in the AI layer, which is ON in the default look: 0.2-0.5 ms/frame | none (pixel-identical) | S |
| 4 | Echo "melt" noise runs for every pixel even when melt is 0 | live_visuals.py:857-858 | 9 value-noise evaluations per pixel, always: 0.1-0.3 ms/frame (3-6 % of the default scene) | none (pixel-identical) | S |
| 5 | Standby when dark: master 0 or blackout held > 2 s -> skip offscreen passes, drop to about 15 fps | live_visuals.py:2524-2597 | about 75 % of all render power while the picture is black (between sets) | none while dark; up to 70 ms extra latency on the first frame back | S |
| 6 | Limiter meter: PBO readback instead of a synchronous `fbo.read` 15x per second | live_visuals.py:2660-2668 | removes 15 pipeline syncs per second; no average GPU saving, fewer late frames under load | none (limiter tau is 0.8 s) | S |
| 7 | Preview OFF by default (the queued user request) | panel.html:1727-1743, 1607-1618; remote_control.py:896-914; live_visuals.py:2610 | 3-5 extra 320x200 scene draws + JPEG encodes per second whenever any panel is open, on any tab: 1-3 % GPU, 1-3 % of a core, GIL contention | none | S |
| 8 | Power-saver frame cap: 30 fps (`swap_interval(2)`) with dt-normalised echo constants | live_visuals.py:1967, 855-865 | 40-50 % of render GPU + CPU energy. The single biggest lever for heat that is certain to work | 30 fps motion is choppier on fast spin / strobe; echo length changes unless normalised | M |
| 9 | Move state publishing and autosave off the render thread; publish only while a panel is polling | live_visuals.py:2443-2463; control.py:1529-1531, 1560-1592; midi.py:350-358 | publish: 0.5-2 ms spike 6.7x per second; autosave: 1-50 ms file write every 5 s on the render thread | none | S |
| 10 | Drop the per-pass `glIsEnabled` / `glGetIntegerv` state queries | fx_stack.py:642-645; math_layers.py:831-833; beings.py:1196; text_layer.py:1137-1138 | 0.05-0.5 ms per active pass (each query syncs the driver's worker thread); low confidence | none | S |
| 11 | `--render 1024x640` as the power-saver profile (already a flag) | live_visuals.py:1879; run_show.ps1:12 | 36 % fewer pixels in every full-size pass (scene, beings, art, text, land composite) | slightly softer lines (they are analytically anti-aliased, so no shimmer) | S (no code) |
| 12 | Half-rate offscreen passes (land / fractal / art every other frame) as a governor step | live_visuals.py:2532-2565; governor.py degrade order | half of those passes' cost: 0.3-1.3 ms/frame per active pass (`HANDOFF`: deck 1.7-2.7 ms at res 1.0, 0.3-0.7 at 0.5) | 30 Hz motion inside that layer only; the echo hides most of it | S-M |
| 13 | AI worker: do not preload, unload to system RAM after idle | ai_worker.py:421-422; ai_gen.py:70; run_show.ps1:20 | about 2.5 GB VRAM freed while idle (no power saving by itself; removes the VRAM-spill stutter risk) | none; first image after idle 1-3 s slower | S-M |
| 14 | Python per-frame overhead: cached uniform handles, skip unchanged, no per-frame f-strings / dict rebuilds | live_visuals.py:2351-2358, 2572-2594; control.py:1486-1527; fx_stack.py:565-573 | 0.3-0.8 ms CPU per frame (2-5 % of one core at 60 fps) | none | M |
| 15 | GC and GIL hygiene: `gc.freeze()` after init, higher thresholds, `sys.setswitchinterval(0.001)` | live_visuals.py main() start of loop | removes occasional 5-30 ms pauses; low-medium confidence | none | S |
| 16 | Decode AI textures on a worker thread; scan the folder off the render thread | live_visuals.py:1489-1498, 2467-2479 | removes a 10-30 ms hitch each time a generated image arrives | none | S-M |
| 17 | Corridor raymarch: evaluate the `atan` relief only near the wall; composite at deck resolution | fx_stack.py:98-118, 670-685 | 15-25 % of a corridor deck; 0.1-0.2 ms for the composite | none if the near-wall threshold exceeds the 0.08 relief amplitude | S |
| 18 | `art_res` param for ArtPass (it always renders 1280x800 with a 128k-triangle mesh, x6 with slices) | art_layer.py:930, 956-963, 1423-1452 | unknown (never benchmarked here); the governor would pick up an `art_res` param by itself | softer relief at lower scale | M |
| 19 | Live internal resolution scaling for the scene pass (viewport scale, like LandPass) | live_visuals.py:1982-1989, 2568-2596, OUT_FS | pixel-proportional (0.75 scale = 44 % fewer scene pixels); closes the governor's "main scene heavy" gap | softer picture when engaged | M-L |
| 20 | One `/state` poll instead of three; slower poll when idle; run the panel on a phone, not the show laptop | panel.html:1880-1888, 2098-2101, 2211-2214 | browser on the show laptop: maybe 3-8 % of a core + iGPU heat in the same chassis | none | S |

## Notes on the top 10

### 1. Check for a busy-waiting vsync and cap turbo (no code)
Could not be verified by reading: NVIDIA's OpenGL driver is known to spin a core inside `SwapBuffers` on some
configurations instead of sleeping until vblank. **Measure first**: with the show on a cheap look, open Task Manager.
If `python.exe` sits near one full logical core (about 6 % of 16 threads) while the frame needs 2-4 ms, the wait is
spinning. Fix in that case: sleep to about 2 ms before the expected vblank ahead of `glfw.swap_buffers(win)`
(live_visuals.py:2721), or turn "Threaded optimization" off for python.exe in the NVIDIA control panel and re-check.
Independently of that: the render loop needs only a few ms of CPU per frame, so the i9 does not need turbo. Setting
Windows power mode to "Best power efficiency" or "Maximum processor state 99 %" for the show's power plan removes
turbo heat that shares the cooling with the GPU. Risk: AI image generation and startup get slower.

### 2. pynvml instead of nvidia-smi (no code)
`make_gpu_reader()` (governor.py:297-311) already prefers `pynvml` and only falls back to running `nvidia-smi` as a
subprocess every 2 s (`gpu_interval=2.0`, governor.py:371). The package is just not installed. Installing
`nvidia-ml-py` in the venv switches it to in-process NVML calls. If that is not wanted, raise `gpu_interval` to 5-10 s.

### 3. Skip texB while not cross-fading
`shade()` samples texA and texB three times each with `textureGrad` (mipmapped, anisotropy 4) and mixes by `mixAB`
(live_visuals.py:401-406). `mixAB` is 0 except during the 3 s fade every `--cycle` 12 s. Add a helper
`vec3 aiTap(vec2 uv, float gr)` that returns texA alone when `mixAB <= 0.0` and the mix otherwise; use it for the
three taps and in `srcImg` (line 229). Output is identical. `l_ai` defaults to 0.5, so this is in the default look,
and in depth-layer / symmetry-blend modes `shade()` runs 3-6 times per pixel.

### 4. Gate the echo melt noise
Line 858 computes `fbm2(...)` + `fbm(...)` and multiplies by `melt`, which is 0 unless `art_use` is 5. The compiler
cannot remove it. Wrap the line in `if (melt > 0.0) { ... }`. Same idea, smaller: `mandalaG` (lines 457-459) runs
two `fbm2` even when `p_swim` is 0; gate on `p_swim > 0.0`. Both are pixel-identical.

### 5. Standby while dark
With master at 0 or blackout held, the engine still runs every offscreen pass and the full scene at 60 fps. Add a
`dark_since` timer: when `bo > 0.99 or vals["master"] < 0.005` for more than 2 s, skip the `extra` passes, keep
drawing the (cheap, black) scene so the echo and limiter stay coherent, and `time.sleep(0.05)` before the swap.
Leave standby on the first frame where the condition is false. The same branch could cover a minimised preview window.

### 6. Async limiter meter
`meter_fbo.read()` (live_visuals.py:2667) is a synchronous `glReadPixels`: the CPU blocks until the GPU has finished
the whole frame, 15 times a second. Copy the pattern from `PreviewTap`: `meter_fbo.read_into(pbo, components=3)` on
frame N, `pbo.read()` on frame N+4 (the next meter frame) and feed `limiter.measure` then. 7680 bytes, one PBO.

### 7. Preview off by default (queued user request, confirmed still open)
No toggle exists yet: the top-bar monitor runs `setInterval(mon, 100)` and fetches `/preview` every 300 ms on every
tab while the page is visible (panel.html:1727-1743); each fetch keeps `preview_job()` handing out captures. After
the async-readback fix the remaining cost is modest (a 320x200 scene redraw is 1/16 of a frame; JPEG encode is off
thread but takes the GIL in parts), so expect a small, real saving rather than a dramatic one. Implementation: a
`PREVIEW: OFF/ON` switch stored in `localStorage` (default off); when off, `mon()` and `thumbPoll()` return at once
and show a placeholder; nothing is needed engine side because no GET means no job after the 3 s linger.

### 8. Power-saver 30 fps cap
`glfw.swap_interval(2)` halves everything at once. The catch: the echo stage uses per-frame constants
(live_visuals.py:855-865: zoom `0.9955`, rotation `0.0015`, decay `p_trails*0.94`, hue step), so at 30 fps trails
would be half as long and drift half as fast. Pass `uniform float fdt` (= dt x 60) and use `pow(decay, fdt)`,
`fr*fdt`, `pow(z, fdt)`; at 60 fps `fdt` is 1 and the look is unchanged. Expose it as a panel switch and as a
governor step for thermal throttle / battery, ahead of the 0.25 render scale. Swarm, text trails and the limiter
already scale by dt.

### 9. Publishing and autosave off the render thread
- `publish()` (live_visuals.py:2443-2463) runs every 0.15 s whenever the server is on, panel connected or not. It
  builds `controller.state()` (about 110 rounded values, base, mods, scenes, midi, ai), `mapper.set.state()`,
  `gov.status()`, static lists (palettes, palette colours, folds, modes), then `json.dumps` on the render thread
  (remote_control.py:843-850). Record the time of the last `GET /state` in `RemoteControl` and skip publishing when
  nobody asked for 2 s; hand the dict to the server thread and serialise there; send the static parts once in `meta`.
- `Controller._autosave` (control.py:1529-1531 -> midi.py:350-358) writes a temp file and renames it every 5 s
  inside `update()`. In a Documents folder (Defender, possibly OneDrive) that can take tens of ms. Queue the dict to
  a writer thread and skip the write when it equals the last one. `mapper.tick` autosave is the same pattern.

### 10. Remove GL state queries from the hot path
LandPass, FractalPass, BeingsPass and TextPass each ask the driver for the blend / depth state every frame so they
can restore it. With NVIDIA's threaded driver each `glGet*` waits for the driver thread to drain. The engine never
leaves blending or depth test on between passes (SurfaceRenderer and ArtPass disable them again), so the passes can
simply end with `ctx.disable(BLEND)` and default blend func. Keep the query path for the standalone tests if wanted
(a constructor flag). Low confidence on the size of the saving; high confidence it is free of visual risk.

## Quick wins under 30 minutes

- Install `nvidia-ml-py` (item 2). No code.
- Windows power mode / turbo cap for shows, and look at python.exe CPU % once (item 1). No code.
- `if (melt > 0.0)` around live_visuals.py:858 and `if (p_swim > 0.0)` in `mandalaG` (item 4).
- texB taps only while `mixAB > 0.0` (item 3).
- `gc.collect(); gc.freeze(); gc.set_threshold(50000, 20, 100)` and `sys.setswitchinterval(0.001)` right before the
  main loop (item 15).
- Start the AI worker with `--no-preload` in run_show.ps1:20, or run with `$env:NO_AI = '1'` when no AI images are
  planned (item 13, first half).
- Power-saver launch line: `run_show.ps1 --monitor 1 --panel --render 1024x640` (item 11).
- Fold the two extra `/state` polls (panel.html:2098, 2211) into the main `poll()`'s data (item 20).
- `self._sbuf.write(verts)` instead of `verts.tobytes()` (fx_stack.py:598): saves a 192 KB copy per swarm frame.

## Don't bother

- **Shader variants / specialisation of SCENE_FS**: the layer branches are uniform-coherent and already skipped at
  run time. Variants would multiply compile time and test surface for maybe 5 %.
- **Tile / bounding culling for beings**: the fragment loop already rejects by distance and per-form box before
  any SDF work (beings.py:905-913), and the pass drops to 0.75x / 0.6x under load. Instanced quads would save only
  the 8-iteration reject loop on empty pixels.
- **RGBA16F -> RGBA8 for the echo buffers**: three 8 MB targets are not a bandwidth problem on this GPU, and 8-bit
  trails are known not to decay to black (the swarm had exactly that bug).
- **Checkerboard / temporal reprojection for the raymarchers**: they are already half res at 0.3-0.7 ms per deck
  (`HANDOFF`). Half-rate (item 12) gets most of the gain for a tenth of the work.
- **Audio FFT**: a 2048-point FFT 94 times a second on its own thread is noise (well under 1 % of a core).
- **Mipmap generation**: only at texture load (live_visuals.py:1495; text atlas once), never per frame.
- **Recorder readback**: already PBO-based, 30 fps default, NVENC; only costs while recording.
- **Surface renderer overdraw**: one bounding-box draw per surface, tested under 1 ms for 12 surfaces (`HANDOFF`).
- **Moving the swarm to transform feedback**: 6000 particles in numpy is about 0.3-0.6 ms of CPU; a rewrite is days.
- **Optimus copy**: not on the projector path.

## Honest confidence summary

- High confidence the code does what is described: items 2, 3, 4, 5, 6, 7, 8, 9, 13, 16 (read directly).
- High confidence it is worth doing, size uncertain: 3, 4, 8, 9.
- Speculative until measured: 1 (may be nothing), 10, 14, 15, 20.
- Biggest certain heat lever: 8 (30 fps) and 11 (lower render size). Biggest possible one: 1.
- Suggested first measurement when the laptop is on mains: the governor already has per-pass GPU timers
  (`gov.status()`), so one run of a default look and one of a heavy look gives real numbers for items 3, 4, 12, 17.
