# Opus review: math stack + canvas, PERFORM redesign (2026-11-26, read-only)

Method: read stack.py, the control.py stack/canvas/scene/panic paths, the live_visuals.py hook, run_show.ps1, the
PERFORM block of panel.html and its shared helpers. Ran `tests/test_stack.py` (48 passed, CPU only). I did not run the
GL tests because the GPU was off limits. I did not run `test_panel_browser.py` because it binds random ports, so I used
my own headless probes against a test Controller on port 8799 instead. The probe scripts are in the session scratchpad.

## HIGH

**H1. PANIC leaves the math stack on.** `control.py:850-870`, `panic()`.
- What happens: `panic()` never touches `self.stack`, `l_stack` or the canvas bypasses. The factory safe scenes have no
  `stack` key, so `_start_recall` (`control.py:1124-1132`) leaves the stack alone. With no safe scene, the base goes
  back to defaults and `l_stack` defaults to 1.
- Probe: NEW CANVAS, then PANIC, gives 1 layer still drawn, `l_stack` 1.0 and 16 bypasses still set.
- Second bug: with no safe scene, `_stack_morph` stays True while `_morph` is cleared, so a running stack morph freezes
  half way.
- `test_panic_and_init` only tests `init`.
- Fix: in `panic()` add `self._stack_morph=False; self.stack.morph_cancel(); self.stack.load(dict(layers=[]))`. Muting
  every layer would also do. Add a test.
- Confidence: high.

**H2. The photosensitive caps can be bypassed through mods and the brightness slew.** `stack.py:43, 615-631`.
- Brightness slews at `2.0*lim`, which is 10 units/s, not the documented 5. Probe: a 5 Hz square on `bright` passes at
  full 0 to 1 amplitude.
- `duty`, `phase`, `hue`, `sat`, `grad`, `freq`, `zoom`, `cx` and `cy` are not slewed at all after modulation. An
  `lfo_square` (rate up to 64) on `duty` (0.02 to 0.98) or on `phase` (with fewer than 2 bands) flips the layer's mean
  level every half cycle. That is a full-frame flash. `hue` gives saturated colour flicker.
- The HANDOFF says "no LFO can blink a layer faster than 2.5 Hz". That holds only for opacity.
  `test_opacity_and_brightness_cannot_square_wave_faster_than_the_limit` asserts on opacity only.
- Fix:
  - Slew `bright` at `SLEW`.
  - Rate-limit every modulated NUM value at about `(hi-lo)*2*SAFE_HZ` per second. Alternatively, cap the LFO rate on
    `stack:*` targets so it cannot exceed SAFE_HZ.
  - Extend the test to cover `bright` and `duty`.
- Confidence: high.

**H3. An open PERFORM drawer blocks PANIC, both the button and hold-Esc.** `panel.html:638` (`.pfdw` z-index 18, the
full-screen overlay over `#top` at z 5) and `panel.html:2155-2161`.
- Probe 1: with a drawer open, `elementFromPoint` over `#panic` returns `.pfdw`. A tap there only closes the drawer.
- Probe 2: holding Esc closes the drawer on the first keydown. The repeats are then ignored (`!e.repeat`), so PANIC
  never fires (0 panics). It takes a release and a second 0.5 s hold.
- The same happens with focus in the new PERFORM text or AI quick box: Esc only blurs. That logic is older, but these
  inputs now sit on the main show screen.
- The MOD editor already had this behaviour. Drawers are now on the primary screen, though.
- Fix:
  - Start `.pfdw` below the top bar (`top:` the height of `#top`), or raise `#top` above it, so PANIC and BLACKOUT stay
    clickable.
  - In the Esc handler, close the overlay or blur and still arm `panicDown()` on that same press. A hold should
    always reach PANIC.
- Confidence: high.

## MEDIUM

**M1. A morph between two stacks whose union is more than 8 layers drops layers and pops at the midpoint.**
`stack.py:398-405`.
- Cause: the other side's ids are inserted at the front of `order`, then the list is cut to 8.
- Probe: morphing a (ids 1-8) into b (ids 11-18):
  - From t=0.1 to 0.4, only b is drawn, at opacity t. All of a is ghosted at the start.
  - At t=0.6, a comes back at opacity 0.4 and b vanishes.
  - At t=1, b fades back in.
- Ids are never reused, so two independently built scenes always have disjoint ids. 5 + 4 layers is enough to trigger
  this. The same applies to the A/B crossfader (`set_xfade`).
- Fix: when the union is more than MAX_LAYERS, keep the layers with the highest weight (`1-t` for a, `t` for b).
  Alternatively, size the shader arrays to 16.
- Confidence: high.

**M2. `frame[-MAX_LAYERS:]` cuts live bottom layers while ghosts are fading.** `stack.py:644`.
- Probe: with 8 layers, a delete followed by an add within 0.2 s makes the bottom layer vanish instantly, then come back.
- An 8-layer `replace` or UNDO does the same.
- Fix: when over the limit, drop ghosts first, not the lowest positions.
- Confidence: high.

**M3. A scene saved on a canvas does not reproduce the build.** `control.py:1062-1071`.
- Cause: scenes store base, mods and stack, but not bypass. On a canvas, the hidden engine faders are still at their
  pre-canvas values.
- Scenario: SAVE TO a pad while building on the canvas. Recall it later, off the canvas, and every engine layer comes
  back under the stack.
- Fix: while `canvas_prev` is set, save `CANVAS_KEYS` layers at their neutral values. Alternatively, store `bypass` in
  the scene.
- Confidence: high on the mechanism. Whether it was intended is for the user to decide.

**M4. `edits` bumps every frame during a stack morph or crossfade.** `stack.py:452` (`morph_to` goes through `load`).
- Effect: the BUILD pile rebuilds on every poll while a recall morphs (`panel.html:2810`, `innerHTML=''`). That drops
  focus from typed boxes and interrupts drags. It also costs a `clean_stack` and a deepcopy per frame.
- Fix: in `morph_to`, bump only when the ids or the stepped fields change.
- Confidence: medium-high.

**M5. `stack.validate` raises TypeError for an unhashable field.** `stack.py:304-305, 332`.
- Trigger: `{"cmd":"stack","op":"set","id":1,"f":[1],"v":1}`. The `/cmd` handler (`remote_control.py:796`) catches only
  ValueError and RecursionError. The exception escapes the handler, so the client gets no 400. Probe confirmed.
  `Controller.apply` swallows it, so there is no crash.
- Fix: check `isinstance(c.get("f"), str)` before using it as a key, or wrap the validator to turn TypeError into
  ValueError.
- Confidence: high. Low impact, since the endpoint needs the token.

## LOW

- **L1. Polar and Tunnel scroll phase jumps when `freq` or `angle` changes.** `stack.py:742-743`. W comes from the
  snapped angular frequency. When W changes, `scroll % W` jumps by a whole number of bands. Probe: phase 4.0 became
  6.0 when W went from 7 to 8. Colours and the hash re-index. A mod on `freq` re-colours the layer on every crossing.
  Fix: carry a wrapped phase and rebase it on a W change instead of recomputing it from the raw scroll.
- **L2. The stack shader has no NaN guard on alpha.** `stack.py:829, 953`. `pow(c, ...)` on a tiny negative `c`, or
  `atan(0,0)` at an exact centre pixel, can make `val` NaN. In the mask, over or replace modes that makes `keep` NaN.
  The rgb guard exists. Fix: use `pow(max(c,0.0), ...)` and guard `keep`.
- **L3. `coreOn` also drops the core jewel outside the canvas.** `live_visuals.py:180, 2720`. Any look with every
  CORE_LAYER at 0, such as text only or camera only, now loses the jewel it had before. Fix: drive `coreOn` from the
  canvas being active, or document the change.
- **L4. `StackPass.render` leaves state behind.** `stack.py:1006-1026`. It leaves its FBO bound and its viewport set,
  and leaves BLEND disabled if the raw `glIsEnabled` read fails. The scene pass rebinds today, so nothing breaks yet.
  A future pass that assumes the default framebuffer would. Fix: save and restore `ctx.fbo` and the viewport.
- **L5. `PF.left` closes any drawer.** `panel.html:734, 1859`. It runs `pfClose()` on every draw while the tab is not
  PERFORM. `mkDrawer` is exported on `window.__panel` (`panel.html:2219`), so a drawer used by any other tab or script
  would close within 250 ms. Fix: act only on the tab-change edge, or only on drawers that PERFORM owns.
- **L6. Touch targets are under 44 px.** With `pointer:coarse`, the morph, quantize and autopilot segment buttons
  measure 27x40 px at 1366x768, 820x1180 and 390x844. That is 27 small targets. Fix: `min-width:44px`.
- **L7. A crash just after a `--fresh` start can resume the old look.** `control.py` autosave. `run_show.ps1` correctly
  drops `--fresh` on a restart. But if the crash comes within 5 s of NEW CANVAS or a stack edit, the restart resumes
  the pre-canvas look. Fix: in `new_canvas`, `restore_canvas` and `stack.do`, pull `_save_t` forward as `set_bypass`
  already does.
- **L8. `restore_canvas` restarts the stack's motion.** Restored layers lose their scroll and spin phases, because
  `_ph` is pruned once the ghost has faded, and they fade in from 0 over 0.2 s. This is cosmetic. The settings
  themselves are restored exactly.

## Checked and OK
- **Canvas round trip.** RESTORE gives back base, mods, bypass, autopilot and stack layers exactly; only `next` moves
  on, as designed. This also holds across a restart while on a canvas. `run_show.ps1` drops `--fresh` on restarts.
- **Cost when empty.** An empty, all-muted or faded-out stack never draws: the pass is skipped, `p_l_stack` is 0 and
  texK is never sampled.
- **Samplers and texture units.** SCENE_FS uses 14 samplers, under the limit of 16. Texture unit 16 is fine on a
  GTX 1650.
- **Wrap and continuity.** The design is sound for fixed W: W is a multiple of D, of hp and of 2 for XOR. `fwidth`
  takes the min across both atan cuts. Kaleidoscope folds are whole numbers.
- **Validation.** It is idempotent for add, set, mod and replace. `stack:<id>:<field>` targets go through
  `valid_target` and `_v_mod` correctly.
- **PERFORM, no XSS found.** Labels, names and status go through `textContent` or `setTxt`, and URLs are
  `encodeURIComponent`-ed.
- **PERFORM, no listener or DOM growth.** 6884 DOM nodes before and after 40 tab switches.
- **No page errors.** LAYERS, SHAPE (`#rip` kept), SCENES (64 and 128 bar options present), AI DREAM, BUILD and
  PERFORM all load clean.
- **Hold-Esc with nothing open** still fires PANIC.
