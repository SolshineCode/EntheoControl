# Code Review: Resource Governor & Visual Polish Pass

Scope: `governor.py` + hooks (`live_visuals.py`, `control.py`, `ai_link.py`, `remote_control.py`, `panel.html`), and visual polish pass (`live_visuals.py`, `art_layer.py`, `params.py`). Read-only review for live-show safety (GTX 1650 Ti 4 GB, Windows, Python 3.11, ModernGL / GLSL 330).

---

## HIGH

### 1. Stale `thermal=True` / sensor lock-in on `nvidia-smi` failure + unthrottled timeout loop and shutdown stall
- **Location:** `governor.py:317-327`, `governor.py:382-395`, `governor.py:411-415`
- **Concrete failure scenario:**
  1. **Permanent false-strained lock-in:** During a hot show, `nvidia-smi` reports `thermal=True, throttle=0x20`, which `Sensors.poll_once()` merges into `self._snap` via `self._snap.update(d)` (`governor.py:394`). If a single subsequent 5-field `nvidia-smi` call returns non-zero or empty stdout under GPU contention, `d` is `{}`, and `governor.py:325` permanently flips `state["throttle"] = False`. Every later query omits `clocks_throttle_reasons.active` and returns a 4-field dict without `"thermal"` or `"throttle"`. Because `self._snap.update(d)` never clears omitted keys, `self._snap["thermal"]` stays `True` for the rest of the show: `Governor._pressure()` permanently sees `strained`, `self.headroom` stays `False` (`governor.py:643`), and the governor degrades all the way to minimum render scales and can never recover. Likewise, if `nvidia-smi` begins timing out while `vram_mb > 3584` or `gpu_temp > 87` was last recorded, `if d:` skips updating `self._snap`, leaving stale critical values active indefinitely.
  2. **Timeout spin, sensor starvation, and exit stall:** `self._next[i] = now + every` (`governor.py:387`) is scheduled *before* `fn()` runs. Because `gpu_interval = 2.0 < timeout = 4`, a timing-out `nvidia-smi` is already overdue when it returns, so it retries every cycle with no backoff and starves battery/CPU/RAM polling on the same thread. On Windows, `subprocess.run(..., capture_output=True, timeout=4)` calls `TerminateProcess` followed by an unbounded `communicate()` — if `nvidia-smi.exe` blocks in uninterruptible WDDM/NVML kernel I/O under heavy GPU load, the sensor thread hangs indefinitely and `Governor.close()` blocks shutdown for the full 5 s `t.join(timeout=5)`.
- **Suggested fix:**
  - Store each reader's latest reading separately with a monotonic timestamp (`self._by_reader[i] = (now, d)`), rebuild `snapshot()` from non-stale entries (e.g. expire after `max(3 * every, 15.0)` s), and replace that reader's whole dict rather than `update()`-merging so omitted keys clear.
  - Only disable `state["throttle"]` after confirming the driver rejects the field (e.g. stderr indicates invalid field, or 3 consecutive 5-field failures while the 4-field query succeeds), and explicitly emit `throttle=None, thermal=None` when disabled.
  - Set `self._next[i]` *after* `fn()` finishes with exponential backoff on failure/timeout, pass `stdin=subprocess.DEVNULL`, and prefer `pynvml` over spawning `nvidia-smi` in production.
- **Confidence:** High

### 2. Broken "user wins" cooldown & misclassified restores across `res:`/`floor:` pairs, scene morphs, autopilot, and PANIC
- **Location:** `governor.py:522-528`, `governor.py:749-752`, `governor.py:858-861`, `control.py:801-822`, `control.py:936-938`, `control.py:1011-1027`
- **Concrete failure scenario:**
  1. **User MIDI/panel move or RESTORE on `res:*` immediately overridden by `floor:*`:** `self.cooldown` (`governor.py:752`) and `self.step_snooze` (`governor.py:860`) are keyed by `step.id` (`"res:fractal_res"`), not by `step.param` (`"fractal_res"`). If `"res:fractal_res"` is active and the performer moves `fractal_res` up via MIDI/panel (or clicks RESTORE on `"res:fractal_res"`), only `"res:fractal_res"` gets the 60 s cooldown / 10 min snooze. `"floor:fractal_res"` has neither, so under continued strain (~8 s later in `auto` mode) the governor fires `"floor:fractal_res"` and slams the user's `fractal_res` straight down to `0.25`. (Similarly, if both `res:` and `floor:` are active and the user clicks RESTORE on `res:`, `floor:` keeps the param pinned at `0.25`.)
  2. **Scene morphs immediately cancel newly applied reductions:** `_step_morph()` (`control.py:1020-1025`) writes `controller.base[n]` every frame during a morph. If the governor applies a reduction mid-morph on a morphing param, `r.base_ref` is recorded on frame N (`governor.py:525`) and `abs(b - r.base_ref) > 1e-4` fires `_user_override()` on frame N+1 (`governor.py:526-527`), immediately dropping the reduction and putting it on 60 s cooldown.
  3. **Autopilot / scene recall / xfade / PANIC misclassified:** Any scene recall, autopilot step, xfade drag, or PANIC that changes `controller.base[p]` triggers `_user_override()`, silently dropping even `"battery"` reductions (while `self.low_power` remains `True` in status!) and imposing a 60 s cooldown. Conversely, if PANIC switches to a safe scene (`Home`) whose `fractal_res`/`serpent_count` matches `r.base_ref` while turning those layers off, old reductions and any active countdown toast stay live — an `auto` countdown can still fire right after PANIC.
- **Suggested fix:**
  - Key `cooldown` by `step.param` (or apply cooldown/snooze to all steps sharing `r.step.param` on `_user_override` and panel `restore`), and hide/unlink `res:` when `floor:` supersedes the same param (or release both together on user move).
  - Have `Controller` track a per-param manual-edit counter bumped only by direct user actions (`set()`/`nudge()`/MIDI param moves — not `_step_morph()`, `_start_recall()`, `set_xfade()`, or `panic()`) and pass that to `Governor.frame()` for user-override detection; update `r.base_ref` on scene transitions and clear pending perf suggestions on `panic()`.
  - Keep `self.low_power` consistent with whether any `"battery"` reductions remain active.
- **Confidence:** High

### 3. Single sticky step at the top of the LIFO stack blocks all earlier reductions from recovering for up to 32 minutes
- **Location:** `governor.py:596-605`, `governor.py:720-724`
- **Concrete failure scenario:**
  - Suppose `res:fractal_res` (order 1) and `count:being_count` (order 2) are active. `count:being_count` is restored first (LIFO), brings strain back within 30 s, and becomes sticky for 2 -> 4 -> 8 -> 16 -> 32 min. Later, the performer switches to a lighter look with sustained headroom. On every recovery check, `governor.py:599-600` inspects only `r = perf[-1]` (`count:being_count`), sees it is sticky, and returns without checking `perf[-2]`. Every earlier reduction (`res:fractal_res`, `ai:pause`, etc.) is permanently blocked from recovering until the top step's sticky timer expires. Furthermore, the backoff counter `k` in `self.sticky[sid]` is never reset after clean recovery, so a single regression hours later jumps straight to 32 minutes.
- **Suggested fix:**
  - Scan `reversed(perf)` and restore the most recent step whose sticky timer has expired (`for r in reversed(perf): if now >= self.sticky.get(r.step.id, (0, -1e9))[1]: ...; break`).
  - Decay or clear `self.sticky[sid]` once a step stays restored without regression for e.g. `5 * REGRESS_S` (150 s).
- **Confidence:** High

---

## MEDIUM

### 4. `spin_twist` is bypassed in flat Droste mode (`symmetry < 1.0`, `travel_mode == 3`)
- **Location:** `live_visuals.py:639-644`, `live_visuals.py:758-760`
- **Concrete failure scenario:**
  - Line 639 folds `spin_twist` into `tw = p_twirl*1.1 - p_spin_twist*spinV*2.2` and applies it to `pf_`, which drives flat travel modes 0–2. Flat Droste (`tmode > 2.5`, lines 758–760), however, recomputes polar coordinates from raw `p` and hardcodes `p_twirl*1.1*lr` instead of `tw*lr`. Turning `spin_twist` up in flat Droste produces zero helical twist and only shifts the AA derivative scale `ps` via `tk` (`live_visuals.py:644`), slightly mis-estimating line antialiasing.
- **Suggested fix:**
  - Use `tw*lr` instead of `p_twirl*1.1*lr` at `live_visuals.py:760`.
- **Confidence:** High

### 5. `spinV` is not held during `freeze` (and is unslewed), causing frozen-frame twitching and single-frame snaps when `spin_twist > 0`
- **Location:** `live_visuals.py:639-643`, `live_visuals.py:1888`, `live_visuals.py:2425-2428`
- **Concrete failure scenario:**
  - Holding **FREEZE** (`p_freeze = 1.0`) stops `Phases.spin`, `smr`, and `clk`, and holds the echo buffer (`held = hold > 0.999`). However, `Phases.uniforms()` (`live_visuals.py:1888`) passes raw `vals["spin"]` directly as `spinV`. If `spin_twist > 0` and `spin` has audio or LFO modulation, the "frozen" scene twists back and forth every frame against the frozen echo buffer. Likewise, an abrupt `spin` change (square/random mod, instant scene recall) snaps `tw` by up to `4.4 rad` per log-radius unit in a single frame.
- **Suggested fix:**
  - Store a smoothed `self.spin_v` in `Phases`, updated with `dtf` (so `dtf = 0` during freeze holds it locked) and a short time constant (~0.06–0.08 s), and pass `spinV=self.spin_v` in `Phases.uniforms()`.
- **Confidence:** High

### 6. Non-finite sensor values (`NaN`/`Inf`) in `gov.status()` silently break `/state` JSON for the entire panel
- **Location:** `governor.py:269-277`, `governor.py:336`, `governor.py:363`, `governor.py:888-896`, `remote_control.py:859`
- **Concrete failure scenario:**
  - `parse_smi()` (`governor.py:269-273`) accepts `"NaN"`/`"inf"` (since `float("nan")` does not raise `ValueError`), and `Governor.status()`'s `r1` helper (`governor.py:888`) returns `nan`/`inf` unmodified. Because `RemoteControl._state_bytes()` encodes the combined state dict with `json.dumps(obj, ..., allow_nan=False)` (`remote_control.py:859`), any single non-finite value in `state["gov"]` raises `ValueError`, causing `_state_bytes()` to drop every state update and freeze the entire control panel on its last good snapshot.
- **Suggested fix:**
  - Guard `math.isfinite(v)` in `parse_smi()`, `battery_from_psutil()`, `read_host()`, and `Governor.status()` (`r1`, `frm`, `to`), mapping non-finite values to `None`.
- **Confidence:** High

### 7. Stale suggestions deadlock `suggest` mode while strain persists; `auto` countdown and `apply` do not re-check mode/candidates
- **Location:** `governor.py:562-575`, `governor.py:832-839`
- **Concrete failure scenario:**
  1. In `suggest` mode, governor suggests `res:fractal_res` while `lvl > 0`. The performer turns `l_fractal` off, but `l_land` keeps `lvl > 0`. Line 563 only clears `self.suggestion` when `lvl == 0`, and line 576 only runs `_pick()` when `sug is None`, so the obsolete `res:fractal_res` toast stays stuck on screen and `land_res` is never suggested.
  2. In `auto` mode, turning `l_fractal` off during the 5 s countdown does not abort the countdown; line 573 applies `res:fractal_res` anyway to an off layer.
  3. A delayed panel `{"op": "apply"}` arriving after switching to `mode == "off"` still applies the reduction (`governor.py:839`), and because `tick()` returns early at line 550 when `mode == "off"`, auto-recovery never runs.
- **Suggested fix:**
  - In `tick()`, re-validate any active `"perf"` suggestion against `self._candidate(self._step[sug["id"]], now)` and clear `self.suggestion = None` if it is no longer applicable.
  - In `command()`, ignore `op == "apply"` when `self.mode == "off"` and check `_candidate` before applying.
- **Confidence:** High

### 8. `PassTimer` missing `stop()` after each offscreen pass misattributes GPU cost; `q.elapsed` can stall under heavy strain
- **Location:** `live_visuals.py:2555-2589`, `live_visuals.py:2618-2620`, `governor.py:238-245`
- **Concrete failure scenario:**
  - `live_visuals.py:2572` calls `gov.timer.start(ex["param"])` before each active pass, but never calls `gov.timer.stop()` after it finishes (or in a `finally:` on error). The last active offscreen pass's `GL_TIME_ELAPSED` query remains open across subsequent skipped passes — including `l_art`'s `source()` GPU texture stripe uploads (`live_visuals.py:2561`) — and all scene uniform/buffer uploads (`live_visuals.py:2591-2617`) until `gov.timer.start("scene")` closes it. This inflates that pass's `pass_ms` and can cause `_pick()` (`governor.py:680`) to degrade the wrong layer first.
  - Additionally, `PassTimer._read()` (`governor.py:239`) reads `q.elapsed` (`GL_QUERY_RESULT`) unconditionally after `lag=3` frames without checking result availability; on Windows WDDM with a 3-frame swapchain queue under heavy strain, this can stall the render thread every 30 frames.
- **Suggested fix:**
  - Wrap each offscreen `render()` call in `try: ... finally: if gov: gov.timer.stop()`.
  - Increase `lag` to `4`–`5` frames (or poll `GL_QUERY_RESULT_AVAILABLE` and defer reading by one frame if not ready).
- **Confidence:** High (misattribution) / Medium-High (`lag=3` stall under load)

### 9. `res:land_res` / `floor:land_res` gates on `l_swarm`, which ignores `land_res`
- **Location:** `governor.py:117`, `governor.py:142-145`, `fx_stack.py:585`, `fx_stack.py:628-629`
- **Concrete failure scenario:**
  - `build_steps()` gates `res:land_res` and `floor:land_res` on `("l_land", "l_swarm")`, but `LandPass._swarm_pass()` always renders at fixed half-resolution `self.swarm_size` regardless of `land_res`. When `l_land == 0` and `l_swarm > 0`, the governor spends two full degrade cycles (~16–22 s in `auto`) lowering `land_res` to `0.25` for zero GPU savings while telling the user the 3D landscape layer is being reduced.
- **Suggested fix:**
  - Change `land_res`'s gate in `governor.py:117` to `("l_land",)`.
- **Confidence:** High

### 10. `art_use == 5` ("Full frame melt") forces echo decay to `0.972` (above the `0.960` pastel washout ceiling) regardless of `l_echo`/`trails`
- **Location:** `live_visuals.py:880-888`
- **Concrete failure scenario:**
  - When `art_use == 5` and `l_art = 1.0` (`melt = 1.0`), `SCENE_FS:885` overrides echo decay to `0.972` (above the `0.960` cap explicitly noted on the same line as the threshold where trails wash out to pastel) and `SCENE_FS:888` overrides echo display gain to `0.92`, even when `l_echo = 0` and `trails = 0`. Under high audio (`high = 1.0`), per-channel `max(col, trail)` combined with `hueRot(prev, 0.047)` rotates bright full-frame images across R/G/B faster than `0.972` decays, washing the frame toward pale pastel and forcing the brightness limiter to pull down overall output.
- **Suggested fix:**
  - Cap `melt` decay at `<= 0.958`–`0.960` and damp `hueRot` rate while `melt > 0` (e.g. `*(1.0 - 0.6*melt)`).
- **Confidence:** Medium-High

---

## LOW

### 11. Hard cut + brief AI-texture flash on art load in `art_use` modes 1–5; `speed == 0` stalls piece switches
- **Location:** `live_visuals.py:2558-2568`, `art_layer.py:1092-1107`, `art_layer.py:1270-1274`
- **Concrete failure scenario:**
  - In `art_use` modes 1–5 with `art_source = 0`, `ArtPass.source()` returns a single texture and hard-cuts at `_mixv > 0.5` rather than crossfading. While the initial uploaded piece is decoding/uploading, `source()` returns `None` and `live_visuals.py:2566` sets `src_ai = True`, flashing an AI texture until upload completes and then snapping to the art piece. Also, `live_visuals.py:2561` passes speed-scaled `dts` to `source()`, so setting `speed = 0` stops `_mixv` from advancing and prevents switching uploaded pieces in modes 1–5.
- **Suggested fix:**
  - Pass wall `dtf` (or `dt`) rather than `dts` to `ArtPass.source()`. Only fall back to `src_ai = True` when `ex["obj"].library()` is empty rather than while a selected piece is mid-decode.
- **Confidence:** High

### 12. `ai_hold` mutates `AIClient.hold` without lock or wake signal
- **Location:** `governor.py:918`, `ai_link.py:33`, `ai_link.py:108`, `ai_link.py:122`, `ai_link.py:135`
- **Concrete failure scenario:**
  - `governor.for_engine()` sets `ai.hold` directly via `setattr(ai, "hold", bool(on))` without acquiring `ai._lock` or calling `ai._wake.set()`. When the governor releases the hold, queued AI jobs wait up to `poll_s` (1 s) before waking.
- **Suggested fix:**
  - Add `AIClient.set_hold(self, on)` under `self._lock` that calls `self._wake.set()` when `not on`, and bind `ai_hold=ai.set_hold`.
- **Confidence:** High

### 13. `governor.validate_command()` coerces non-boolean `on` for `low_power` via `bool()`
- **Location:** `governor.py:446-447`
- **Concrete failure scenario:**
  - Unlike `control.py` / `remote_control.py`, `validate_command()` uses `bool(c.get("on", True))`, so malformed input like `{"cmd": "gov", "op": "low_power", "on": "false"}` silently enables `low_power` instead of raising `ValueError` (HTTP 400).
- **Suggested fix:**
  - Check `v = c.get("on", True); if not isinstance(v, bool): raise ValueError("'on' must be true or false")`.
- **Confidence:** High

### 14. Unbounded `uniform float t` (`self.T`) loses float32 precision in `kene` / `keneM` shimmer over multi-hour shows
- **Location:** `live_visuals.py:295`, `live_visuals.py:349`, `live_visuals.py:1809`, `live_visuals.py:1887`
- **Concrete failure scenario:**
  - `self.T` is never wrapped before upload as float32 `uniform float t`. After ~5–10 hours at `speed = 2.0` (`t ≈ 36,000–72,000`), `t * 3.0` in `vnoise(q*3.0 + t*3.0)` reaches float32 ULP `0.008–0.016`, slightly stepping the high-frequency shimmer.
- **Suggested fix:**
  - Pass a wrapped uniform `(self.T * 3.0) % 64.0` for `vnoise` shimmer (matching `vnoise`'s period 64).
- **Confidence:** High

### 15. Unclamped `max3(col)` in `art_use == 1` ("Inside the lines") amplifies the `+ 0.06` floor under stacked layers + max audio
- **Location:** `live_visuals.py:818-819`
- **Concrete failure scenario:**
  - `col = mix(col, (im*1.55+0.06)*m*(0.55+0.45*max3(col)), p_l_art)` does not clamp `max3(col)`. When several layers stack under max audio (`max3(col) ≈ 3–5`), the `+ 0.06` constant scales up proportionally, lifting dark regions of `im` toward grey before the final clip.
- **Suggested fix:**
  - Use `min(max3(col), 1.0)` inside line 819.
- **Confidence:** High

---

## Checked and fine

- **`params.py` append-only contract & registries:** Original 72 params (`master`..`art_glow`) and `control.CUE_CC` (1..66) are unchanged; `art_use`, `art_source`, `spin`, `spin_twist` are appended after `art_glow` and properly listed in `NEUTRAL`/`BYPASS_KEEP`, `CPU_ONLY` (`live_visuals.py:152-154`), and `control.OPTIONS`/`DISCRETE`.
- **`art_layer.limit_distortion()` (`art_layer.py:891-907`):** Continuous, NaN/Inf/missing-key safe via `_f()` + `_clamp()`, budget bounded in `[0.8, 1.5]` (no divide-by-zero), preserves displacer ratios, and leaves single effects and structural remaps untouched.
- **Serpents (`live_visuals.py:538-584`, `1749-1896`):** `seen[:, 0] = seen[:, -1] = False` prevents out-of-bounds indexing; forward and reverse (`serpent_speed < 0`) lap handoffs between `serpA` and `serpC` are continuous with no visible teleports; `serpB[:, 0:2]` and `serpW` wrap modulo `TAU` in float64 before float32 upload; `serpents()` over-blends cleanly without blowout; `serpent == 0` freezes advancement.
- **Flat `kene()` lattice wrap math (`live_visuals.py:308-356`, `473`, `1800`):** `G_kl = HSF/6.0` gives `pf / G_kl = 1920` lattice units per wrap, an exact multiple of 64 (super-tile + `hash21`/`vnoise`), 16 (`chaseKey`), and 2 (parity + fill motifs).
- **Ground-plane horizon (`live_visuals.py:720-739`):** `ay = max(abs(yh), 1e-3)` avoids divide-by-zero; `trSway * 2.0` wraps at `200 * PI` (exact multiple of `TAU`) so `strata` is seamless; `-0.045` knee decays to true black.
- **`art_use` modes 2–4 math & texture unit 8 (`live_visuals.py:226-239`, `608-614`, `820-841`):** `srcPh` and `srcTr` wrap at `2.0` seamlessly (period-2 trig / mirrored-repeat `srcImg`); `art_use == 3` uses explicit `textureLod` with `rK >= 1e-5` and exact period-2 `thK/TAU*4.0` across the `atan` branch cut; `srcFit` aspect math is exact; unit 8 (`texS`) does not collide with any pass.
- **`spin` integration & cross-pass sign parity (`live_visuals.py:1815`, `2573-2583`, `art_layer.py:1368-1391`, `fx_stack.py:552-554`):** Double-precision `Phases.spin` wraps modulo `TAU`, respects `freeze` and `speed`, and matches rotation direction across `SCENE_FS`, `ArtPass`, and `LandPass`.
- **Governor render-loop isolation, Windows console safety & UI wiring:** `Governor.frame()` only mutates the per-frame `vals` dict (never `controller.base`, so `live_state.json` and saved scenes never store degraded values); `nvidia-smi` uses `CREATE_NO_WINDOW` (`0x08000000`) on Windows; `live_visuals.py` wraps governor calls in `guarded()`; `remote_control.py` validates `cmd == "gov"` via `governor.validate_command()` with strict regex/allowlist checks; `panel.html` consumes the single shared `panelstate` event and tolerates `gov == null`.
