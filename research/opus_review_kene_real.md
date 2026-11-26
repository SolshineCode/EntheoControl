# Opus review: real kene study layers (kene_real.py + hooks)

2026-11-25. Read-only review. I ran CPU-only checks (the 53 non-GL tests pass, plus a timing script and the scratch checks noted below). No GPU, no show_* folders, no ports.
Scope: `kene_real.py`, the `params.py` block, the `live_visuals.py` hook, the `panel.html` KENÉ STUDY block, `tests/test_kene_real.py`.

## Ranked findings

### H1. Building a design takes 10-65 ms on the render thread, not "a few ms" (confidence: high, measured)
`kene_real.py:785-806` (`_design` -> `shipibo_design`, Python Dijkstra + fill loops). I timed it on this machine:
default single panel (22x14) about 37 ms, single panel at scale 1 about 64 ms, Panels (x4 variants) 47-68 ms, band 10-40 ms, medallion 8-42 ms.
- **When it hurts:** the first frame the layer comes up, every NEW DESIGN click, every pen cycle with `kene_pen_new` (the default is 1), and every
  `kene_scale` or `kene_seed` move. At 60 fps each one drops 2-4 frames. A knob sweep hits the 5-per-second cap and loses about 185-325 ms of
  every second, which stutters the whole show and not just this layer.
- **Fix:** build designs on a worker thread (numpy releases the GIL only partly, so a process or a precompute is safer). Keep the old design until
  the new one is ready. Also prebuild the next pen seed `(seed+n+1)` during the hold phase, and build all 32 seeds for the current dims in the
  background once the layer is first raised (each one is tiny, about 20 KB). Update the HANDOFF cost line.

### H2. Polar panels: the pen order jumps every time the ring scroll wraps (confidence: high)
`kene_real.py:842` wraps `self.polar` at 4.0. In the shader at `kene_real.py:658`, `rank = mod(id.x*3 + id.y*5, 8)/8`. Shifting `id.y` by 4 adds
20 ≡ 4 (mod 8), so every panel's rank moves by half (checked for all four cases).
- **When it hurts:** layout Follow the show with `symmetry` >= 0.5, plus travel, with `kene_pen` running or `kene_draw` < 1. Every few seconds
  (about every 3.5 s at travel 1 with 8 folds), the half-drawn panels all switch drawn/undrawn in one frame. That is a full-field pop.
- **Fix:** wrap `polar` at 8.0. Then `5*8 ≡ 0 (mod 8)` and `mod(id.y,2)` still holds. Add a test that renders `polar = x` and `x + period` and
  compares the frames.

### H3. Speed is applied twice: scroll and morph run at speed² (confidence: high)
`kene_real.py:833-846, 865`. The engine passes `dts`, which already includes freeze and speed (`live_visuals.py:1863, 2715`). `_advance`
multiplies by `speed` again. Beings and text take `dts` as it is.
- **When it hurts:** at speed 2, panels, band and tunnel scroll 4x. At 0.5 they scroll 0.25x. The bead motif morph ("over two beats") also slows
  with speed and stalls under freeze.
- **Fix:** drop `sp` from `_advance`. Drive the morph from the wall `dt` or from `clock["beat"]`. Bench and gallery pass wall dt with speed 1, so
  they never showed this.

### H4. The NEW DESIGN rate limit runs on the show clock and stalls at speed 0 or freeze (confidence: high)
`kene_real.py:790`: `t - self._dwant_t < 0.2`. Here `t` is `ph.T`, which only advances by `dts` (`live_visuals.py:1866`).
- **When it hurts:** with speed 0 or freeze held, after one new design no uncached design is ever built. NEW DESIGN, the seed fader and pen-new
  all silently show the old design. At speed 0.1 the cap becomes one design per 2 s.
- **Fix:** time the limiter with `time.perf_counter()`.

### M1. The pen cycle hard-cuts the lit cloth ground back in (confidence: high; also a photosensitivity concern)
`kene_real.py:467-468` together with `573` and `628`. `fade` goes to 0 over the last 6 % of the cycle, then jumps to 1 at `fr = 0`
(seen in the check: beat 15.99 fade 0.0003, beat 16.0 fade 1.0). With progress 0, "Light on black" shows nothing, which is fine. But
Mud-dyed cloth and Ceramic draw `uGround*cloth` whatever the progress is.
- **When it hurts:** a full panel of cream (about 0.8 luminance) cuts on in one frame once every pen cycle (about 5.5 s at 174 BPM on 4 bars).
  `test_pen_cycle...` claims "no cut" but only checks the fade-out.
- **Fix:** `fade = min(fade_out, smoothstep(0, 0.06, fr))`, or scale cloth by `smoothstep(0, 0.08, prog)` while the pen is on.

### M2. The pen seed also flips the bead tile, so the Huni Kuin layer cuts every pen cycle (confidence: high)
`kene_real.py:906` `self._beads(pen["seed"])` and `812-814`. The atlas variant is `seed % 2`, and pen-new adds 1 to the seed each cycle.
- **When it hurts:** txede bedu and kapë hina switch band width with no morph every N bars. This happens even with `l_kene_shipibo = 0` whenever
  a stored `kene_pen` is non-zero. NEW DESIGN does the same half of the time.
- **Fix:** give the beads their own seed (or `kene_seed` without the pen offset), and send variant changes through the existing morph.

### M3. Photosensitivity: the seed and motif knobs can make full-field pattern changes at up to 5 Hz (confidence: medium)
`kene_real.py:790` caps designs at 5 per second. That is above the 3 Hz flash guideline, and each swap moves every line in a full-field
high-contrast pattern. `hk_motif` sweeps are worse. A change during a morph snaps `now = next` (`kene_real.py:859-862`): the beads not yet
changed jump at once and the morph restarts, so a knob sweep gives pops at near frame rate. The engine flash limiter has not landed yet
(HANDOFF 01:20).
- **Fix:** cap at 2 designs per second, or better, cross-fade old and new designs over about 0.3 s (two `_dtex`, or fade out and in through
  `uAmtS`). For hk, queue the newest target and let the running morph finish.
- **Other animations, all fine at high BPM:** flow is one pulse per bar per point. The hk_seq run is one sweep per 8 beats per bead, but its
  sparkle term (`kene_real.py:680`, 7 % of beads re-rolled every beat, ±0.25·high) reaches 3.3 Hz at 200 BPM. It is scattered and small in
  area, so low risk. You could re-roll it every 2 beats instead. Breathing is 7.5 s. The chase contrast is 10:1 at the show's own rate, the
  same as the other layers.

### M4. Respect: "Sing it in: N bars" points at icaros (confidence: medium)
`kene_real.py:57` and `params.py:325`. README 1.4 and 4.3 ("avoid sacred-claim misrepresentation", Brabec de Mori's caution against literal
"designs are songs") argue against implying that the software sings designs in. The banned-word test (`tests/test_kene_real.py:201`) misses it.
- **Fix:** "Draw in: 4 bars" and so on. Keep the option count the same so stored values and MIDI stay valid.

### M5. Respect: bare motif names in the bead dropdown (confidence: medium)
`kene_real.py:63` and `params.py:311`. The names do match the README 2.2 table (tested, correct). But a performer who picks "txede bedu" gets
told the output *is* txede bedu, while HANDOFF says the geometry is "my reading … say study" and nobody from the community has checked it.
- **Fix:** label the param "Bead motif (study)", or suffix each option with " (study)". The longest result fits under 40 characters.

### L1. Tests that do not test what they claim (confidence: high)
- `test_primary_lines_never_dead_end` (`tests/test_kene_real.py:41`) cannot fail. Partition edges on a square grid can never give a degree-1
  corner. I checked: 200 random labellings, 0 dead ends. Only the `> 40` check carries weight. Either call it a by-construction property or
  test something that can break (for example, no 1-tile island that the secondary line doubles over, or the shader's drawn lines).
- `design_stats()["filled"]` (`kene_real.py:325`) is `interior[interior].mean()`, which is always 1.0. The `filled == 1.0` assertion
  (`tests/test_kene_real.py:74`) checks nothing.
- `test_gl_long_show_phases_wrap` (`tests/test_kene_real.py:320`) uses a fresh pass with travel, flow and rotate at 0, so no accumulator ever
  wraps. It would not catch H2. Drive `polar`, `pan`, `band_scroll` and `hk_*` to just below and just above their periods instead.
- `test_band_design_wraps_seamlessly`: `(lab[:,0] != lab[:,-1]).mean() < 0.5` is weak. The east/west fill-bit check is the real one.
- The append-only test (`tests/test_kene_real.py:181`) only checks "after crowd_serpent". Assert that the new keys are the tail of `PARAMS`.
  They are today: this is the only `PARAMS.update`.
- `test_gl_no_shimmer` compares whole-frame means, so local shimmer averages out. A per-pixel max diff on a crop would catch it.

### L2. Shipibo generation runs while only the bead layer is up (confidence: high)
`kene_real.py:905` calls `_design` even when `amt_s` is 0, so pure bead shows still pay H1's hitches on seed and pen changes.
- **Fix:** skip it when `amt_s <= 0.001`.

### L3. Small items
- Reversed `smoothstep` edges (`kene_real.py:524, 568, 571, 708, 709, 717`) are undefined in the GLSL spec. Desktop NVIDIA, AMD and Intel use the
  formula, so this is low risk. `1.0 - smoothstep(lo, hi, x)` is the portable form.
- The 4 MB RGBA8 output texture and the shader compile are allocated at startup even if the layer is never raised. That is fine, and the cost
  while off is truly zero: the engine skips `render()` at amount <= 0.001, and the composite branch is uniform-gated.
- `CORE_LAYERS` now includes both study layers, so raising one on a clean canvas brings back the core jewel. That matches beings and land, but
  check it is wanted.
- Texture units: no collision. The scene uses 0-8; art uses 9-15 (text 10-12, camera 12-13, automap 14 and fx_stack 15 already overlap inside
  that range, pre-existing and not kene's); the stack uses 16; kene uses 17; kene's private units are 20 and 21; filters use 0-3 and re-bind.
  Unit 17 stays bound between frames and nothing else touches it. GL state (FBO, viewport, BLEND) is restored in `finally`.
- NaN/inf: inputs go through `_num`/`_val`, the palette is checked with `isfinite` and clipped, `log(r)` is guarded at r >= 1e-4,
  `stroke()` has `ww > 0` whenever `fw > 0`, and the output is hue-normalised to <= 1. Labels top out at 210 (64 seeds x 4 layouts at scale 1),
  so they never collide with FRAME 255.
- Determinism: `default_rng([seed, gx, gy, wrap])` with no use of Python `hash()`. Good. The mixed-type `rng.choice(["cross", 0.0, …])` gives
  strings that `float()` turns back into numbers. It works but is fragile, so pick an index instead.
- Panel XSS: none. Everything goes through `h()` -> `textContent` (`panel.html:405`), and the labels and options are static strings.

## Respect check summary
- **Good:** the layer labels say "study" and "inspired". The CREDIT constant is in the panel (tested). Folding Blocks is untouched. The motif names
  appear exactly as README 2.2 gives them, with nawan kenê and kashi left out. Yawanawá was skipped because the README gives no grammar for it.
- **Fix:** M4 and M5 above.
- **Out of scope, but they contradict README 4.3.1:**
  - `panel.html:1269-1270` holds AI prompt presets titled "Huni Kuin kene lattice" and "Shipibo kene neon". Those label generated images as kene.
    The panel test only greps for `authentic\s+ken`, so it misses them.
  - `live_visuals.py:10, 34, 148` still call Folding Blocks "kene" (docstring, help text and the layer preset name shown for key M).
- `reference_pairs.png` puts generated output next to cropped photos, including a named living artist's textile
  (`wikimedia_olinda_silvano_textile.jpg`, `kene_real.py:1039`). Keep it as an internal study aid. Don't tune toward one artist's piece, which
  README 4.3.2 warns against ("not … scrape-and-clone"). If the sheet is ever shared, the CC BY/BY-SA credit has to be on the image itself, not
  only in PROVENANCE.csv.
