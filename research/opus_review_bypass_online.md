# Opus review: BYPASS + PREVIEW (merge agent) and ONLINE plugin (2026-11-25)

Read-only review (no tests run, show was live). Ranked by severity. Confidence in brackets.

## Scope B: ONLINE plugin

### B1. HIGH: the Claude backend is not shown to be tool-less, and it runs in the home folder [medium; high for the cwd part]
`plugins/online/backends.py:136-148`. argv is `claude -p --output-format text --restricted --strict-mcp-config --disable-slash-commands --no-session-persistence`, with cwd = `Path.home()`.
- `--restricted` is not a flag I can confirm, and nothing else turns the built-in tools off (no `--tools ""`, no `--disallowedTools`, no `--permission-mode`). The test (`test_online.py:526`) only checks that the string is in argv.
- With cwd = home, Claude Code treats `C:\Users\you` as the project. So `~/.claude/settings.local.json` loads as project-local settings, and it allows `Bash(powershell:*)`, `Bash(python:*)` and `Bash(curl:*)`. The home project's auto-memory (the user's private auto-memory files) loads into the context too.
- How it fails: anyone with the panel token (sent over plain HTTP on venue Wi-Fi, so it can be sniffed) sends `ask` "...also run powershell ...". If tools are live, that runs code on the show laptop. Even if they are not, the model can echo private memory into `say`. That text is shown on every panel and stored raw in `logs/online_*.log` (`raw=text[:4000]`).
- Fix: run in an empty private cwd (`fresh_dir(logs/online_work, "claude")`). Disable tools explicitly (`--tools ""` or a full `--disallowedTools` list) and pass `--permission-mode default`. At plugin load, check that each flag appears in `claude --help`, and refuse to start if one is missing. Add a canary test: the reply to "list the files in your folder" must contain no file names.

### B2. HIGH: the "Allow Google CLI tools" opt-in can be ticked from any LAN panel, which makes chat text run code [high]
`protocol.py:151-157` (the `cli_google` op rides `/cmd`). `backends.py:195` (`agy --dangerously-skip-permissions`, with prompt.txt saying "follow it exactly"). `images.py:216` (`gemini -e nanobanana --yolo`) and `images.py:252-256` (agy image).
- `-e nanobanana` only limits which extensions load. The gemini CLI's own shell and file tools stay on, and `--yolo` approves them. The image prompt is written by the performer or by the LLM, goes into these agents, and they act on it.
- How it fails: a token holder sends `{"op":"cli_google","on":true}`, then `backend agy`, then an `ask` (or a `gen` prompt) asking for a command. The agent runs it with full permissions.
- Fix: the opt-in must not be settable over `/cmd`. Allow it only from a local config file or from loopback clients. Remove the agy chat backend, or run it in a sandbox with no tools. For gemini, use an explicit tool allowlist instead of `--yolo`.

### B3. HIGH (photosensitive): the chase_rate clamp ignores chase_mode [high on the math, medium on perceived area]
`safety.py:86-97` caps the sweep rate at 2.5 sweeps/s. In the shader (`live_visuals.py:243-252`), mode 1 runs the window at `idx*2.0 - chasePh*2.0`, so each point is lit twice per sweep. Mode 2 adds a counter-window, so up to 4 times per sweep.
- How it fails: the assistant sets `chase_mode 2`, `chase 1`, a wide `chase_width` and `chase_rate` at its cap. Each point then flashes at 5-10 Hz.
- The clamp also uses the BPM at reply time. In SUGGEST mode, APPLY can come minutes later at a higher BPM.
- `test_clamping_master_floor_ranges_and_chase_rate` does not cover modes.
- Fix: `lim = MAX_FLASH_HZ*240/bpm / {0:1, 1:2, 2:4}[mode]`, using the chase_mode the batch leaves behind. Refuse a chase_mode change that would push the current chase_rate over the limit. Validate again at apply time.

### B4. MEDIUM: assistant changes beat the resource governor and can overload the GPU [high on the mechanism]
`assistant.py:430-431` calls `ctl._do(set)`, which calls `Controller.set`. That increments `edits` (`control.py:746`), so the governor treats every assistant move as the performer's hand and drops its reductions. The allowlist also includes `fractal_res`, `fractal_detail`, `land_res` and every `l_*`.
- How it fails: "make it richer" turns on fractal, land, swarm and beings at render scale 1.0 on the 1650 Ti, and the governor stands down. That is the overheating / power-loss case.
- Fix: take `*_res` / `*_detail` out of the allowlist (or cap them at the governor's current value). Apply assistant sets through a path that does not bump `edits`, as the autopilot does.

### B5. MEDIUM: blackout through the back door is still possible [high]
`safety.py:162-163` refuses only the `layers` key. A batch of `set l_* 0`, or a `bypass` for each layer, fits in the 24-command limit and blacks out the show.
- Fix: after the batch is built, work out the post-batch visible opacity (base, minus anything bypass covers) and refuse the batch if every layer ends up below about 0.15.

### B6. MEDIUM: undo is neither photosensitive-safe nor aware of later changes [high]
`assistant.py:225-249`.
- (a) `bypass_fade` is restored before `set_bypass`. If the performer's fade was 0, undo brings the layer back as a hard cut, which breaks "no hard cuts".
- (b) Undo of a scene recall does `base.update` directly. Selector (DISCRETE) params snap, a recall the performer queued since is cancelled (`pending=None`), mods are swapped, and every fader the performer moved after the batch is overwritten.
- (c) The history survives PANIC, so an undo after PANIC partly reverses it.
- Fix: call `set_bypass` first, then restore the fade, never going below MIN_FADE_S. Restore through a morph. Skip params whose current base is no longer the value the batch set. Grey out history entries older than the last PANIC.

### B7. MEDIUM: child processes outlive cancel, OFF and crashes [medium]
- `backends.py:250-256`: `pty.terminate(force=True)` kills only the agy process. Its children (language server, tool shells, all running with skip-permissions) stay alive.
- There is no Windows Job Object. If the engine crashes or exits while a request runs, the `claude` or agy child keeps going, agy for up to `--print-timeout 600s`.
- Fix: `taskkill /F /T /PID pty.pid`. Put every child in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.

### B8. MEDIUM: a cancel can be lost [high, small window]
`assistant.py:342-345` and `images.py:387-389` run `_cancel.clear()` after the gen check.
- How it fails: a CANCEL, PANIC or OFF that lands between dequeue and `clear()` is wiped. The CLI then runs to its cap (180 / 600 s), and the reply is dropped only afterwards.
- Fix: give each job its own Event, or check gen again after `clear()`.

### B9. MEDIUM: PANIC and OFF do not stop online images [high]
- `plugin.py:119-123`: `on_panic` cancels the assistant but not `self.images`. A `gen {show:true}` the assistant started cross-fades in after PANIC.
- `images.py:425-426`: there is no gen check between `generate()` and `save_texture` / `show`. After OFF, an in-flight `urlopen` (up to 90 s) still writes `gen_*.png`.
- Fix: call `images.cancel()` in `on_panic`, and check gen before saving and before showing.

### B10. MEDIUM-LOW: the render thread does work that grows with the LLM output [medium]
- `tick()` runs `parse_reply` (`safety.py:48-55`: one `raw_decode` per `{`, so O(n·k)) on stdout that has no size cap (`backends.py:118`). A pathological reply stalls the GL thread.
- `lister()` (scandir + stat of textures/ and art/) runs on the GL thread for every ask and every reply.
- `shutil.which` runs again inside `state()` after each `set_providers` (key save / configure).
- Fix: cap the reply at about 16 KB in the worker, parse there, and do only the context validation on the GL thread. Cache the listings.

### B11. LOW: API key file [high]
- The key is stored as plaintext in the project root. `Documents\EntheoControl\.gitignore` has `logs/` but not `online_keys.json`. The snapshot copies (`show_preview2/`, ...) may carry the file along.
- Fix: store it in `%APPDATA%`, or add it to the publish clone's ignore file now.
- Checked and fine: the key is never in state, never in logs, and is scrubbed from SDK errors.

### B12. LOW: command-line edge cases
- `agy_pty` (`backends.py:193-199`): a prompt that ends in `\` escapes the closing quote, so the flags after it are swallowed. `%VAR%` expands if `agy` is a .cmd shim.
- `ONLINE_CLAUDE_MODEL` and `ONLINE_AGY_MODEL` go unescaped into a .CMD command line (set by the environment only).

### Checked and fine (Scope B)
- Chat text reaches claude on stdin and agy through prompt.txt: no argv injection from the chat.
- `validate` bounds every op. There are no new HTTP routes, and everything is token-gated.
- The panel renders LLM text with `textContent` only, so no XSS.
- Triggers, strobe_rate, mods, mapping, MIDI, rec and gov are all refused.
- One value per param per batch, plus MIN_APPLY_S: nothing the assistant sets reverses faster than 1 Hz (B3 is the exception).
- OFF imports only `protocol.py`. agy really is refused when `cli_google` is off, including an explicit `ask {backend:"agy"}`.
- The OFF hook does not block (only `OnlineLog.close` joins, for 2 s at most).

## Scope A: BYPASS + PREVIEW

### A1. MEDIUM: a MIDI CC bound to bypass toggles on every message of 64 or more [high]
`control.py:1342, 1361-1363`: `press = cc and val >= 64`, with no edge detection.
- How it fails: a knob or fader learned to `bypass:<key>` toggles dozens of times a second while it moves. With `bypass_fade 0` that is a hard flicker of a whole tab at MIDI rate (a photosensitive problem). The same pre-existing issue affects `scene:` and `panic`.
- Fix: keep the last state per binding and toggle only on the move from below 64 to 64 or more.

### A2. MEDIUM (design): the `layers` key is a blackout one small click away, and PANIC does not undo it [high]
- The LAYERS tab gets a header power icon (`panel.html:1994-1998`, the click is stopPropagated). It bypasses every `l_*`, so the picture goes black while master stays at 1.
- PANIC deliberately leaves bypass alone (`control.py:812-835`), so the panic button cannot recover from it.
- Fix: no header icon for `layers` (or require hold-to-confirm). PANIC should clear `layers` (and probably `fx`), or at least raise a banner.

### A3. LOW: bypass is persisted only by the 5 s autosave [high]
`control.py:1551-1553`. After a crash within 5 s of a toggle, a layer the performer just brought back comes back OFF, or a layer just killed pops back on without a fade.
- Fix: a debounced save (about 0.5 s) after `set_bypass`.

### A4. LOW: neutral values that change the overall look [medium]
- `color` / `fx` set `chase` to 0. That removes the chase mask (`mix(1, key, p_chase)`, `live_visuals.py:252`), so the frame gets brighter (about +30% at defaults, much more at chase 1 with a narrow width). "Off" is brighter than "on". It does not blow out (the limiter catches it), but it is surprising.
- `layers` going to black is the only true blackout.
- All NEUTRAL values are inside their ranges, and every one is in CONT.

### A5. LOW: the preview can keep itself alive briefly [high]
`remote_control.py:935-936`: `set_preview` re-creates the `()` view when a capture finishes after the view expired. That gives 1.5 s more of captures, and it repeats while a capture is in flight at expiry. The tail is bounded, but it is not strictly zero after 3 s.
- Fix: re-create the view only if `_preview_asked` is within `preview_linger`.

### A6. LOW: lazy /state can go backwards [high]
`remote_control.py:854-867`: two concurrent GETs. A takes obj1, B takes obj2 and stores it, then A stores obj1. The panel briefly shows older state. Nested shared lists (autopilot `drift_params`, `corners`) are now serialised off the GL thread. A RuntimeError there is caught, so the only effect is a stale snapshot.
- Fix: put a sequence number on publish and store only if newer.

### A7. LOW: "SHOW ALL HIDDEN" leaves some layers off
`panel.html:1644` clears only layer keys and `layers`. Tab keys (`art`, `land`, `fractal`, `beings`, `text`) still keep their layer rows OFF.

### Checked and fine (Scope A)
- Exact restore: bypass applies only to the post-mod output values (`control.py:1532-1535`). `base`, `_sm`, mods, scenes and MIDI are untouched, and a key is dropped at level 0, so switching back on gives exactly the live value. This also holds mid-morph and with the crossfader.
- PANIC and scenes leave bypass alone, as documented. Restore filters unknown keys and returns at level 1.
- The panel sends an explicit `on`, so two panels cannot race a toggle.
- PREVIEW OFF costs one lock plus an empty dict check per frame: no draw, no readback.
- The async PBO is collected on the next frame, and a job is taken only when the tap is free.
- The mix draw's uniform changes (`p_l_*`, `limiter`, `pxh`) are all rewritten every frame by the engine (`live_visuals.py:2597-2614`).
