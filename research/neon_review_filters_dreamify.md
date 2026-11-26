# Code Review: Post-FX Filters + DREAMIFY

- **Reviewer:** Gemini Neon (read-only static review; no code/GPU executed)
- **Target Hardware / Stack:** Windows, ThinkPad 20TK (i9-10885H, 32 GB RAM, GTX 1650 Ti Max-Q **4 GB VRAM**), ModernGL / GLSL 330, PyTorch / Diffusers / Torchvision
- **Scope Reviewed:**
  - `filters.py`
  - `dreamify.py`
  - `live_visuals.py` (`[filters> ... <filters]` blocks + surrounding render/meter/cleanup paths)
  - `ai_worker.py` (`/dreamify`, `/dream-cancel`, `/status`, `_dream`, `RealPipes.offload`/`restore`)
  - `remote_control.py` (`validate_dream_request`, `ai_call`, `/dreamify`, `/dream-cancel`, `/dream-status`)
  - `panel.html` (`/* [filters> ... <filters] */` FILTERS tab + DREAMIFY popover/badge + `.pwbar` / `Escape` interactions)
  - `tests/test_filters.py`, `tests/test_dreamify.py`

## Summary

| Severity | Count | Highlights |
|---|---|---|
| **HIGH** | 5 | Blackout/master/limiter bypass on filter exception; queued DREAMIFY cancel broken; VRAM juggling (`acts` autograd leak + stranded CPU SD-Turbo on `restore()` OOM); `safe()` zeroing negative `hueRot` channels into black holes in `DREAM_FS`; Halftone RGB-screen lighting up pure black backgrounds |
| **MED** | 8 | Glitch photosensitive bypass under MOD modulation & frame stalls + neon/bloom surge vs limiter; `DREAM_FS` broken pause-reseed & 26.7 min wrap pop; `DREAM_FS` eye motif popping & Voronoi clipping; `SORT_FS`/`OIL_FS` half-res aliasing, flicker, & premultiplied-alpha fringing; FILTERS tab hiding its own bypass power bar inside `<details>`; DREAMIFY popover/badge stuck-poll, `Esc`->PANIC, & 1 Hz DOM rebuild; missing `filt.release()` on exit & cross-job batch leak; tests with tautologies / order-dependence / missing proxy & cancel coverage |
| **LOW** | 4 | `NEON_EDGE_FS` `atan(0,0)` UB & `pal[i]` bounds; `PreviewTap` solo-mix ignoring blackout/master/limiter while filters run; minor path/symlink/tmp/slug/pixel-limit hardening in `dreamify.py`/`ai_worker.py`; continuous `#main` `MutationObserver`, 41x `draw()` on reset, `FINAL_FS` `-0.004` dark-floor mismatch |

---

## HIGH Findings

### H1. Blackout, master, and limiter are completely bypassed (full-brightness flash on projector) if `FilterPass.apply()` raises or `FINAL_FS` fails to compile
- **Location:** `live_visuals.py:2720-2723`, `live_visuals.py:2737-2749`, `filters.py:674-681`, `filters.py:793-794`
- **Failure scenario:**
  1. When `filt_on` is `True`, `live_visuals.py:2722` overrides `uni.update(limiter=0.0, p_master=1.0, p_blackout=0.0)` and renders `scene_vao` directly into `disp` at full brightness, expecting `FilterPass.apply()` (`FINAL_FS`) to apply all three afterwards.
  2. If `filt.apply()` raises ANY exception—for example, VRAM OOM during the lazy ~40 MB `_alloc()` on first filter use while SD-Turbo/Dreamify are resident on the 4 GB GPU, a driver error, or `"final"` failing to compile at startup (which `filters.py:679-681` catches without disabling `FilterPass`, causing `KeyError("final")` at `filters.py:793`)—`live_visuals.py:2742-2745` catches the exception, sets `filt_on = False`, and continues straight to Spout, Preview, Recorder, and the **projector output pass** (`live_visuals.py:2788-2803`).
  3. Because `disp` still holds the un-limited, `master=1.0`, `blackout=0.0` scene render, a blackout (`bo == 1.0`) or zeroed master (`master == 0.0`) **blasts full brightness onto the wall for up to 5 consecutive frames** before `filt_fails >= 5` disables filters.
- **Fix:**
  1. In `FilterPass.__init__`, raise if `"final" not in self.progs` (so `live_visuals.py` sets `filt = None`).
  2. In `live_visuals.py:2720`, skip the filter pass entirely when output is blacked out (`bo >= 0.999` or `vals["master"] <= 1e-4`), letting `SCENE_FS` output true black directly (or clear `disp` to black).
  3. In `live_visuals.py:2742-2749`, on `filt.apply()` failure, restore `fbos[dst]`, viewport, and texture units 0–3, then **re-render `scene_vao`** with the real `limiter=limiter.value, p_master=vals["master"], p_blackout=bo` (idempotent because `scene_vao` reads `pp[src]`), or explicitly clear `disp` to `(0,0,0,1)` if `bo > 0.5 or vals["master"] <= 0.01`.
- **Confidence:** High

### H2. `POST /dream-cancel` fails to cancel queued DREAMIFY jobs (and races with dequeue / `submit`)
- **Location:** `ai_worker.py:295-302`, `ai_worker.py:309-312`, `ai_worker.py:362-372`, `ai_worker.py:441-452`
- **Failure scenario:**
  1. `submit()` puts `(jid, job, self.cancel_gen)` into `self.q` (reading `self.cancel_gen` outside `self.lock`) and does **not** record `self.dream_cancel_gen`.
  2. When a DREAMIFY job is queued behind startup warmup, an SD-Turbo generation, a depth map job, or another dream, calling `POST /dream-cancel` calls `cancel_dream()`, which removes the job from `self.pending` and increments `self.dream_cancel_gen`—but leaves `(jid, job, gen)` in `self.q` with `gen == self.cancel_gen`.
  3. When `_loop` later pops that job, line 363 checks `gen != self.cancel_gen` (`False`), enters `_dream()`, and line 443 captures `g0 = self.dream_cancel_gen` **after** the cancel already occurred. `cancelled()` returns `False`, and the cancelled job runs to completion and writes its PNG to disk (violating `ai_worker.py:21`).
  4. The same lost-cancel race occurs if `cancel_dream()` fires between `_loop` popping `item` (`:350`) and `_dream` reading `g0` (`:443`), or if `cancel()` runs between releasing `self.lock` (`:300`) and `self.q.put` (`:301`).
- **Fix:** Inside `self.lock` in `submit()`, capture `(cgen, dgen) = (self.cancel_gen, self.dream_cancel_gen)` and enqueue `(jid, job, cgen, dgen)`. In `_loop`, under `self.lock`, skip the job if `cgen != self.cancel_gen` or (`job.get("kind") == "dream"` and `dgen != self.dream_cancel_gen`). Pass `dgen` into `_dream()` and check `dgen != self.dream_cancel_gen` in `cancelled()`.
- **Confidence:** High

### H3. DREAMIFY VRAM juggling pins GoogLeNet autograd graphs across `pipes.restore()` and permanently strands SD-Turbo half-on-CPU if `restore()` OOMs
- **Location:** `dreamify.py:173, 188-208`, `ai_worker.py:108-115, 453-462, 500-516`
- **Failure scenario:**
  1. `RealDream._grad()` (`dreamify.py:200-208`) stores CUDA activation tensors (holding the full backward autograd graph) in `self.acts`, and **never clears `self.acts`** after `loss.backward()` or inside `RealDream.release()` (`dreamify.py:188-195`).
  2. In `ai_worker.py:507-516`, `dreamer` is still referenced by the local variable when `pipes.restore()` runs on line 514, so the last tile's activations/graph stay pinned in VRAM right as `pipes.restore()` attempts to move ~2 GB of SD-Turbo back onto the 4 GB GTX 1650 Ti.
  3. If `pipes.restore()` (`self.pipe.to("cuda")`) raises `CUDA out of memory` (e.g., because `live_visuals.py` allocated ~40 MB filter targets or new textures during the 30–90 s dream, CUDA allocator fragmentation, or pinned `dreamer.acts`), `diffusers`' `.to("cuda")` moves submodules sequentially and aborts partway through—leaving UNet/VAE/text_encoder split across CPU and CUDA. Because `Worker` and `RealPipes` do not persist an `offloaded` flag, **every subsequent SD-Turbo `/job` fails permanently** with `RuntimeError: Expected all tensors to be on the same device` until `ai_worker.py` is restarted.
  4. Additionally, if `dreamer.dream()` hits OOM on CUDA (`ai_worker.py:500-505`), there is no `torch.cuda.empty_cache()` or fallback to 512 px / CPU, and `RealDepth` (~100–200 MB VRAM, triggered automatically after every `art` dreamify at `ai_worker.py:495-499`) is never offloaded.
- **Fix:**
  1. Wrap `RealDream._grad()` in `try ... finally: self.acts.clear()`, and also clear `self.acts.clear()` in `RealDream.release()`.
  2. In `ai_worker.py:507-516`, `del dreamer` (or `dreamer = None`), call `gc.collect()` and `torch.cuda.empty_cache()` **before** calling `pipes.restore()`.
  3. Track `self._offloaded` inside `RealPipes`. If `restore()` fails, move `self.pipe.to("cpu")` + `gc.collect()` + `torch.cuda.empty_cache()` so it stays cleanly on CPU, keep `self._offloaded = True`, and automatically attempt `restore()` at the start of the next `txt2img`/`img2img` job. Consider also offloading `depth` when free VRAM is below `NEED_MB[768]`.
- **Confidence:** High

### H4. `safe()` in `filters.py` turns any negative RGB component into pure black, punching black holes into `DREAM_FS` eyes, fur, and scales
- **Location:** `filters.py:153, 158-159, 205-212, 267, 288, 301, 309-310, 170`
- **Failure scenario:**
  1. `COMMON` defines `safe(vec3 c)` as `(c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0 && c.r < 64.0 && c.g < 64.0 && c.b < 64.0) ? c : vec3(0.0)`. If **any single channel** is even slightly negative (`-0.0001`), the **entire pixel** is replaced with `(0, 0, 0)`.
  2. `hueRot(c, a)` uses Rodrigues rotation around `(1, 1, 1)/sqrt(3)`, which routinely outputs negative components (down to `-1/3`) for saturated inputs: e.g. `hueRot(vec3(0, 0, 1), PI) = vec3(0.667, 0.667, -0.333)`.
  3. In `DREAM_FS` `eyeAt()` (`:205`), `irisC = normalize(hueRot(tint, 2.2 + 1.6*rnd) + 0.05) * 1.25` rotates saturated show-palette hues right across `PI` (`2.2..3.8 rad`), yielding negative RGB components (down to `-0.34`) inside the iris. Similarly, `amp` (`:267`), fur (`:288`), and scales (`:301`) add unclamped `hueRot(...)` terms.
  4. At line 310, `f = vec4(safe(o), 1.0)` detects the negative channel and **blanks the bright iris / motif pixels to dead black holes**. Any negative value written into signed `RGBA16F` ping-pong targets `T["A"]`/`T["B"]` is likewise blacked out by `FINAL_FS:170`.
- **Fix:** Make `safe(vec3 c)` reject only NaN/Inf/extreme values while clamping negatives to `0.0`:
  ```glsl
  vec3 safe(vec3 c){
    bvec3 bad = isnan(c) | isinf(c);
    return any(bad) ? vec3(0.0) : clamp(c, 0.0, 64.0);
  }
  ```
  And wrap `hueRot(...)` calls in `eyeAt()` and `DREAM_FS` with `max(..., 0.0)` before normalizing/adding.
- **Confidence:** High

### H5. Halftone "RGB screen" mode (`flt_ht_style = 1`) draws a grid of 50%-brightness RGB dots across pure black backgrounds
- **Location:** `filters.py:397-403`, `filters.py:409-414`
- **Failure scenario:**
  1. `dotCover(pp, a, v)` computes `r = cell * sqrt(clamp(v, 0.0, 0.785) / PI)` and returns `clamp(r - d + 0.5, 0.0, 1.0)`.
  2. On a pure black pixel (`v == 0.0`), `r == 0.0`, so `dotCover` returns `clamp(0.5 - d, 0.0, 1.0)`—emitting up to **`0.5` (~128/255) brightness** at every grid cell center where `d < 0.5 px`!
  3. Style 0 (`:427`) masks this by multiplying by `cc / max(v, 1e-4)` (= 0 on black), and style 2 (`:422`) explicitly adds `* step(0.003, v)`, but style 1 (`:413`: `o[k] = dotCover(pp, a, cc[k]);`) has no zero gate.
  4. Switching Halftone to "RGB screen" lights up every black area of the projection with a bright red/green/blue dot grid, breaking the "black = no light" invariant. (`tests/test_filters.py:259` missed this because it only tests default `flt_ht_style = 0`.)
- **Fix:** Gate `dotCover` on `v > 0`:
  ```glsl
  if (v <= 0.001) return 0.0;
  return clamp(r - d + 0.5, 0.0, 1.0) * smoothstep(0.001, 0.008, v);
  ```
- **Confidence:** High

---

## MEDIUM Findings

### M1. Photosensitive safety gaps: `GLITCH_FS` breaks ramp/rate limits under MOD modulation and frame stalls; Neon/Bloom can cause ~2.5x–3.5x brightness surges before limiter reacts
- **Location:** `filters.py:91-118`, `filters.py:462, 505, 608-619, 869-872`
- **Failure scenario:**
  1. **MOD-matrix bypass on `glitch`:** `GlitchClock` caps onset frequency (`<= 2.5 Hz`) and latches `seed` only while `env == 0`, but `flt_glitch` (`amt`) and `flt_glitch_block` (`blk`) are modulated live every frame. In `GLITCH_FS:612`, `sel = step(1.0 - 0.20*amt, h) * env` is a hard binary `step()` on `amt`, and `row = floor(pp.y / blk)` changes block boundaries immediately. During an active glitch window (up to 2.5 s at `rate=0`), a fast LFO or bass mod on `flt_glitch` hard-pops blocks >3 Hz, and modulating `flt_glitch_block` scrambles the entire screen layout at 60 Hz.
  2. **Frame-stall hard cut:** At the engine's `dt` cap (`0.1 s`, `live_visuals.py:2443`), `lim = dt / GLITCH_RAMP_S = 1.67 > 1.0`, allowing `env` to jump `0.0 -> 1.0 -> 0.0` in single frames without ramping.
  3. **Neon / Bloom surge vs limiter:** `NEON_OUT_FS:462` (`c*keep*0.85 + e*0.7 + g*gl`) and `BLOOM_OUT_FS:505` (`c + b*amt*2.2`) can output >3.0 into RGBA16F (and compound further if stacked, since only `FINAL_FS` clips). Adding Neon in the panel immediately sets `flt_neon = 0.7`, creating a ~2.5x brightness jump that persists for ~0.8–1.5 s while `Limiter` (`tau = 0.8 s`, sampled every 4th frame) catches up.
- **Fix:** Latch `blk` (and optionally slew-limit `amt` at `1 / GLITCH_RAMP_S`, or smoothstep `sel`) on glitch onset while `env == 0.0`; clamp `GlitchClock.step` `dt` to `min(dt, GLITCH_RAMP_S * 0.5)` so frame hiccups cannot produce single-frame 0→1→0 cuts; consider a soft shoulder between filter passes (`o / (1 + max(max3(o) - 1.5, 0))`) and starting newly added Neon at a gentler default (e.g. `0.45`).
- **Confidence:** High

### M2. `DREAM_FS` pause-reseed fails when all filters are toggled off/on, and falsely pops every 26.7 minutes at `T_WRAP = 1600.0`
- **Location:** `filters.py:33, 132, 234, 256, 284, 801-806`
- **Failure scenario:**
  1. `_run_dream` checks `abs(self.state.t - self._dream_last) > 0.5` to re-seed `D0`/`D1` after a pause. Because `FilterPass.apply()` is skipped when all filters are off, `self.state.t` stops advancing while filters are off. Re-enabling `dream` 10 minutes later on a different scene advances `self.state.t` by one `dt` (`0.017 s <= 0.5`), so **no re-seed happens** and the stale 10-minute-old dream frame flashes into the live picture via `prev` (`keep` up to 0.955).
  2. Conversely, every 26.7 minutes (`T_WRAP = 1600.0 s`), `self.state.t` wraps `1599.98 -> 0.00`, making `abs(now - self._dream_last) == 1599.98 > 0.5` and abruptly resetting `D0`/`D1` mid-performance.
  3. At that same `T_WRAP = 1600.0` wrap, `DREAM_FS` also jumps discontinuously in `sin(t * 0.4)` (`640 / TAU = 101.86`, non-integer) and in non-periodic `vnoise(pa*2.3 + vec2(t*0.031, -t*0.023))` / `vnoise(pf*3.0 + t*0.05)`.
- **Fix:** Use `time.monotonic()` (not `self.state.t`) for `_dream_last` pause detection (updating `_dream_last` even when `d == 0` during freeze so a >0.5 s freeze does not re-seed). Make `vnoise` periodic (`mod(floor(p), 64.0)`, matching `live_visuals.py:242`) and align `DREAM_FS` time multipliers to `T_WRAP`.
- **Confidence:** High

### M3. `DREAM_FS` eyes pop abruptly when `flt_dream_scale` or `flt_dream_motif` moves, and clip along Voronoi borders
- **Location:** `filters.py:216-239, 279, 304-305`
- **Failure scenario:**
  1. `eyes()` uses a hard binary cutoff `if (rnd > dens) return vec4(0.0);` (`:228`). Sweeping or modulating `flt_dream_motif` pops eyes in and out at full opacity in a single frame.
  2. Sweeping or modulating `flt_dream_scale` changes `cells = mix(11.0, 3.5, scl)`, continuously shifting `gi = floor(p * cells)` so every eye's `hash12(id)` changes every frame—causing rapid strobe-like popping of eye positions, sizes, angles, and iris colors across the frame.
  3. `eyes()` picks only the single nearest cell `best` before checking `rnd > dens` / `e <= 0`; any empty/unlit cell slices straight off adjacent eyes (`w * s` up to `0.45` cell units + `0.25` jitter) that cross the Voronoi bisector.
- **Fix:** Skip cells with `rnd > dens` inside the 3×3 loop (or alpha-blend valid eyes) and fade opacity smoothly (`smoothstep(dens, dens - 0.08, rnd)`); for `flt_dream_scale`, crossfade two adjacent octave scales (as `live_visuals.py` does for flat-field `zoom`) so eyes scale smoothly without cell-ID thrashing.
- **Confidence:** High

### M4. Half-res aliasing & temporal flicker in `SORT_FS` / `OIL_FS`, plus premultiplied-alpha dark fringing in `SORT_OUT_FS`
- **Location:** `filters.py:335-359, 565-601, 826-832, 861-867`
- **Failure scenario:**
  1. **Unfiltered half-res sampling & flicker:** Both `OIL_FS` and `SORT_FS` render into half-res `T["H"]` (640×400) while sampling full-res `src` (1280×800, no mipmaps) directly. In `OIL_FS`, Sobel gradients at 2 full-res px stride alias on 1–2 px kene/sacred lines, making brush orientation `tv` jitter frame-to-frame. In `SORT_FS`, `dir` steps by `mix(1.0, 3.0, len)` half-res px (**up to 6 full-res px per tap**) with a hard `if (k < thr) break;`, so moving thin lines jump between taps and cross `thr` discontinuously, causing heavy temporal flicker.
  2. **Half-res upsample haloing in `SORT_OUT_FS`:** `T["H"]` stores non-premultiplied `(rgb, a)` where unsorted pixels (`a = 0`) store `c0` (often black or a different color). Bilinear filtering in `SORT_OUT_FS` (`:597-599`) blends `(bright, 1)` with `(c0, 0)` into `(0.5*bright + 0.5*c0, 0.5)`, then `mix(c, s.rgb, 0.5*amt)` darkens/muddies streak edges and leaks half-res `c0` onto crisp full-res `c`.
- **Fix:** Box-downsample `src` into `T["d1"]` (via `down`, `first=0`) before `OIL_FS` and `SORT_FS`; cap `SORT_FS` stride to <= 1.5 half-res px with soft threshold weighting; store premultiplied `(rgb * a, a)` in `T["H"]` and unpremultiply or composite `c * (1.0 - s.a * amt) + s.rgb * amt` in `SORT_OUT_FS`.
- **Confidence:** High

### M5. FILTERS tab swallows its own BYPASS power-switch bar (`.pwbar`) into the collapsed "All filter params" `<details>` pool
- **Location:** `panel.html:2199`, `panel.html:3179-3189`, `panel.html:3234-3239`
- **Failure scenario:**
  1. The main panel script runs `pages[t].insertBefore(bar, pages[t].firstChild)` (`:2199`), placing `<div class="pwbar">` at the top of `#pg-filters`.
  2. The `[filters>` block then executes `var gen = [].slice.call(pg.children)` (`:3179`) and appends **every** child—including `.pwbar`—into the collapsed `<details id="fltpool">`.
  3. If the performer switches the FILTERS tab off via the tab-strip power icon or MIDI, the prominent "FILTERS is switched OFF" banner/button is hidden inside `<details>`. Meanwhile, `:3235-3236` inspects base values `B` rather than live modulated/bypassed values `V`, so the chain header still claims `"1 of 1 running"` even while bypassed (and conversely shows `"AMOUNT 0: NOT RUNNING"` when a filter is driven solely by a MOD-matrix source).
- **Fix:** Exclude `.pwbar` when building `gen` (`!el.classList.contains('pwbar')`), and read live `A.state().V` (plus bypass state) for the running/amount-0 indicator.
- **Confidence:** High

### M6. DREAMIFY popover & badge: stuck polling on worker crash/non-matching error, `Esc` triggers PANIC, 1 Hz DOM wipe drops CANCEL taps
- **Location:** `panel.html:3265-3302`, `panel.html:2155-2162`
- **Failure scenario:**
  1. **Stuck `job` / infinite polling (`:3296`):** `job` only clears on failure if `s.error` matches `/cancel|fail/i`. If `ai_worker.py` crashes/restarts mid-dream (`s.available === false` or `s.error === ""`), or fails with `"SD-Turbo could not move back to the GPU: ..."` (`ai_worker.py:516`), `job` never clears, `#dbadge` stays stuck on `"DREAMIFY queued"` forever, and `poll()` hits `/dream-status` every 1 s indefinitely.
  2. **`Esc` triggers PANIC (`:2155-2161`):** `#dpop` (`z-index: 60`) is not registered with the global `Escape` handler, so pressing/holding `Escape` to close the popover invokes `panicDown()` instead.
  3. **1 Hz `badge.innerHTML = ''` (`:3286`):** Rebuilding the badge every second destroys and recreates the `CANCEL` button (causing touch/click taps to drop across poll boundaries) and resets the `<div class="bar"><i>` width to 0, breaking `transition: width .4s`.
  4. **Page reload / second tab (`:3283, 3294`):** When a dream is already running (`job === null`), `#dpop` shows no progress text (`if (pop._msg && job && d)`) and no completion banner fires.
- **Fix:** Clear `job` when `!s.dream && !(s.queue || []).length && !s.busy && Date.now() - job.t > 3000` (or `s.available === false`); add a capture-phase `keydown` listener closing `#dpop` on `Escape`; create `#dbadge`'s children once and mutate text/width in place; render progress whenever `d` is non-null regardless of `job`.
- **Confidence:** High

### M7. Missing `filt.release()` on shutdown, batch-job state leakage, non-idempotent `FilterPass.release()`, and live-loop allocation hitch
- **Location:** `live_visuals.py:2146-2152, 2852-2861, 2880-2901`, `filters.py:695-731`
- **Failure scenario:**
  1. `live_visuals.py`'s `finally:` block (`:2883-2901`) closes every subsystem except `filt` (`filt.release()` is never called).
  2. Between `--batch` / `--gallery` jobs (`:2852-2861`), `ph`, `limiter`, and `fbos` are reset, but `filt.state`, `filt._dream_last`, and `D0`/`D1` are **not** reset—so consecutive batch jobs bleed `dream` feedback images, `kal_phase`, and `GlitchClock` state into each other.
  3. `FilterPass.release()` (`filters.py:716-731`) does not clear `self.fbo`, `self.vaos`, `self.progs`, or `self.vbo`; calling `release()` twice raises `moderngl.Error` on already-released objects.
  4. `_alloc()` allocates all 14 targets (~39.3 MB) synchronously on the render thread on the very first frame any filter activates, risking a frame hitch (or OOM -> triggering H1) mid-show.
- **Fix:** Add `if filt is not None: guarded("exit", filt.release)` to `live_visuals.py`'s `finally:`; add `FilterPass.reset()` and call it between batch jobs; clear `self.fbo`/`vaos`/`progs` in `FilterPass.release()`.
- **Confidence:** High

### M8. Tests that do not test what they claim (tautology, order dependence, missing proxy & queued-cancel coverage)
- **Location:** `tests/test_filters.py:157-160, 244-259, 272-290`, `tests/test_dreamify.py:194-218, 245-257, 282-290`
- **Failure scenario / gaps:**
  1. `test_glitch_survives_huge_dt_without_bursts` (`test_filters.py:157-160`) asserts `max(abs(b - a)) <= 1.0`. Because `env` is always in `[0.0, 1.0]`, this is a **tautology** that would pass even for a 60 Hz full-amplitude strobe and never verifies onset rate or ramp bounds after stalls.
  2. `test_each_filter_changes_the_picture_keeps_black_and_never_blows_out` (`test_filters.py:244-259`) shares a module-scoped `FilterPass` (`gl`) across all 10 filters and renders only 12 frames (`0.2 s`). `GlitchClock.__init__` starts with `wait = 0.5 s`, so `ft="glitch"` **only passes because the 7 earlier filters drained `wait`**; running `pytest -k glitch` alone fails (`diff == 0.0`). It also only tests default sub-params, missing H5 (`flt_ht_style=1` on black).
  3. `test_final_stage_applies_limiter_master_and_blackout` (`test_filters.py:272-290`) only checks `assert fp.level is not None` without reading back `fp.level`'s alpha to verify it carries the pre-limiter post-filter level; and `test_engine_hook_is_wired` only greps strings, never testing `live_visuals.py` blackout behavior on filter failure.
  4. `test_remote_control_validates_and_proxies_dreamify` (`test_dreamify.py:282-290`) **never tests proxying**—it only calls `validate_dream_request()`, never starting `RemoteControl` or testing auth/proxying on `/dreamify`, `/dream-cancel`, or `/dream-status`.
  5. `test_cancel_stops_a_running_dream_without_writing` (`test_dreamify.py:245-257`) only cancels a running dream, missing broken queued-dream cancellation (H2); `test_low_vram_offloads_sd_turbo_then_restores` (contains dead `class W(ai_worker.Worker): pass`) never tests `restore()` failure; and `_worker()` sets `depth_factory=None`, never exercising `ai_worker.py:495-499`.
- **Fix:** Replace the tautology with real onset-rate/ramp checks; reset `fp.state` per test and run enough frames for `glitch`, plus parametrize discrete filter styles (`flt_ht_style=1`, `flt_chroma_mode=2`, etc.) on black; read back `fp.level` alpha; add live HTTP tests through `RemoteControl` for all three dream routes (including 403 auth); add queued-cancel and `restore()`-OOM tests.
- **Confidence:** High

---

## LOW Findings

### L1. `NEON_EDGE_FS` has `atan(0.0, 0.0)` UB on isoluminant chromatic edges and unguarded `pal[i]` indexing
- **Location:** `filters.py:437-438, 445-451`
- **Failure scenario:** On an isoluminant chromatic boundary (`gm > 0.15` from RGB differences, but `luma(gx) == luma(gy) == 0.0`), `atan(0.0, 0.0)` is undefined in GLSL 330 and can yield NaN -> `NEON_OUT_FS`'s `safe()` turns that edge pixel black when `colormode > 0`. Also `pal[i]` (`:438`) lacks `% 5` (unlike `live_visuals.py:249`), risking an out-of-bounds read if `fract()` rounds to `1.0` on tiny negative floats.
- **Fix:** Use `atan(luma(gy), luma(gx) + 1e-6)`, index `pal[i % 5]`, and wrap `NEON_EDGE_FS` output in `safe(...)`.
- **Confidence:** High

### L2. `PreviewTap` solo-mix redraw ignores `limiter`, `master`, and `blackout` when filters are active
- **Location:** `live_visuals.py:2720-2723, 2762-2764`, `fx_stack.py:789-825`
- **Failure scenario:** Because `live_visuals.py:2722` writes `limiter=0.0, p_master=1.0, p_blackout=0.0` into `uni` and `scene_prog` when `filt_on` is `True`, `preview.service`'s `"mix"` redraw renders soloed layers at full brightness even during blackout or `master=0` (inconsistent with when filters are off).
- **Fix:** Restore `scene_prog["limiter"]`, `p_master`, and `p_blackout` (and `uni`) to their real values before calling `preview.service` (after `filt.apply()`).
- **Confidence:** High

### L3. Minor hardening gaps in `dreamify.py`, `remote_control.py`, and `ai_worker.py` (symlink check, `.tmp` cleanup, filename collision, leading `-` slug, pixel-count guard)
- **Location:** `dreamify.py:71-74, 96-98, 108-129`, `remote_control.py:427-429`, `ai_worker.py:471-499`
- **Failure scenario:**
  - `dreamify.validate` does not verify `(folder / name).resolve().parent == folder.resolve()` (unlike `remote_control.py:367, 713`), and `_plain_name` / `validate_dream_request` do not explicitly reject `"\0"` (relies on `is_file()` raising `ValueError`).
  - `dreamify.save_png` leaves `<name>.tmp` behind if `img.save()` or `os.replace()` fails; `out_name()` uses 1-second resolution with no collision check or random suffix.
  - `_slug("-foo")` strips only `_`, leaving a leading `-` that fails `ai_worker.ART_NAME` (`^[A-Za-z0-9]`), silently skipping auto depth-map submission (`ai_worker.py:495-499`).
  - `ai_worker.py:471` loads `folder / job["name"]` without checking a decoded pixel cap (`MAX_ART_PIXELS` / `MAX_SOURCE_PIXELS`) prior to `im.load()`.
- **Fix:** Add `.resolve().parent == folder.resolve()` and `"\0" not in name`; clean up `tmp` in `save_png` `except:` and add a short hex suffix on collision; change `.strip("_")` to `.strip("_-")` in `_slug`; verify `im.size[0] * im.size[1] <= 60_000_000` before `im.load()`.
- **Confidence:** High

### L4. Minor UI/shader polish: broad `#main` `MutationObserver`, 41x synchronous `draw()` on reset, `FINAL_FS` dark-floor clamp, `KALEIDO_FS` zero-fold guard, test screenshot side-effect
- **Location:** `panel.html:3312, 3198`, `filters.py:175, 631`, `tests/test_filters.py:332-333`
- **Failure scenario:**
  - `panel.html:3312` observes all of `#main` (`childList: true, subtree: true`), firing on every 6 Hz meter/clock/fader text update and running `querySelectorAll('.pick .art')` every 300 ms indefinitely.
  - `RESET CHAIN` (`panel.html:3198`) calls `A.setParam` 41 times synchronously, triggering 41 full `draw()` passes per click.
  - `FINAL_FS:175` outputs `clamp(o, 0.0, 1.0)` without `SCENE_FS:953`'s `- 0.004` dark-floor crush, so tiny filter residuals `< 0.004` are not clamped to true zero.
  - `KALEIDO_FS:631` computes `TAU / floor(folds + 0.5)` without `max(2.0, ...)` (safe via `params.py` clamping, risky if `FilterPass` is called directly).
  - `tests/test_filters.py:332-333` writes Playwright PNGs into the repo's `test_frames/filters/` instead of `tmp_path`.
- **Fix:** Observe only `.pick` nodes; batch resets or defer `draw()`; match `SCENE_FS`'s `- 0.004` floor in `FINAL_FS` (when `nanGuard` is on); guard `max(2.0, floor(folds + 0.5))`; write test screenshots to `tmp_path`.
- **Confidence:** High

---

## Verified Clean / Notable Positives

- **Texture unit hygiene:** `FilterPass._draw` only binds units `0..2`, and `live_visuals.py:2747-2749` restores `fbos[dst]`, viewport, and scene units `0..3` even if `filt.apply()` raises. `BeingsPass` (unit 6) re-binds unit 6 before each scene draw, so `meter_prog`'s unit-6 binding (`live_visuals.py:2820`) does not collide.
- **No concurrent SD-Turbo vs DREAMIFY execution:** `ai_worker.Worker._loop` runs all `/job`, `/depth`, and `/dreamify` work sequentially on a single worker thread, preventing simultaneous GPU contention between SD-Turbo and GoogLeNet.
- **Route authentication & LAN safety:** `/dreamify`, `/dream-cancel`, and `/dream-status` in `remote_control.py` all pass through `_gate()` (constant-time token check + `active` check), body size is capped at 4096 B, and `ai_worker.py` binds strictly to `127.0.0.1` with constant-time `X-Token` authentication on every route.
- **Atomic PNG writes:** `dreamify.save_png` writes to `<name>.tmp` and promotes via `os.replace`, and both `art_layer._is_art` and `remote_control.TEX_NAME` ignore `.tmp` files.
- **Zero-cost default path:** When no filter slot has `amount > 0.001`, `filt_on` is `False`, no filter targets are allocated, and `live_visuals.py` executes the untouched pre-filter render path.
- **CSS `.art .dmb` positioning:** The earlier `top:3px` inheritance note in `HANDOFF.md` is already fixed at `panel.html:3146` (`top:auto;left:auto;right:3px;bottom:3px`, specificity `0,2,0` > `.art button`'s `0,1,1`).
