# Timeline, Shows & Presets Code Review

Scope: `timeline.py`, `shows.py`, `control_methods.py`, `remote_excerpts.py`. Absent modules (`control.py`, `recorder.py`, panel HTML/JS) are marked **unverified**.

## 1. Security

- **1.1 [HIGH] Zip bomb OOM & unpacked-size bypass** (`shows.py:import_zip`, L441–483)
  Pass 1 trusts header `i.file_size`. L456 calls `z.read("show.json")` unbounded—forged `file_size=100` decompresses gigabytes into RAM and OOM-kills the engine. Pass 2 caps each entry at 128 MiB but never sums actual decompressed bytes vs `MAX_UNPACKED` (2,000 entries can unpack ~256 GiB) and buffers each file twice (`parts` + `b"".join(parts)`).
  *Fix:* Stream `show.json` with a hard `MAX_SHOW_JSON` cap; track cumulative actual bytes vs `MAX_UNPACKED`; stream entries directly to temp file (`fsync` + `os.replace`).

- **1.2 [HIGH] `/media-upload` truncates live audio before body check** (`remote_excerpts.py:_library_post`, L112–127; `shows.py:save_media`, L1064–1071)
  L115 calls `shows.save_media(..., name, b"")` for name validation, but `save_media` runs `write_atomic(f, b"")`, truncating any existing `shows/_media/<name>` to 0 bytes before `_stream_body` checks `Content-Length` or reads data. Any `400`/`413` or dropped upload wipes the track.
  *Fix:* Call `shows.check_media_name(name)` + `shows.inside(root/"shows"/"_media", dst)`; `fsync` `tmp` before `os.replace`.

- **1.3 [HIGH] Unchecked extensions & oversized media imported** (`shows.py:import_zip`, L445–451; `_restore_media`, L828–849)
  `ZIP_ALLOWED`/`MEDIA_NAME` allow any extension (`.exe`, `.py`, `.html`) under `media/{art,tex,audio}/` up to 128 MiB (> `MAX_MEDIA_FILE` 64 MiB, `MAX_AUDIO` 120 MiB) and 128 MiB non-JPEG thumbs. `_restore_media` copies them into `art_dir`/`tex_dir`/`_media` without extension checks or `inside()`.
  *Fix:* Enforce `IMG_EXT` (`art`/`tex`), `AUDIO_EXT` (`audio`), small cap + JPEG check for thumbs, per-kind limits, and `inside(folders[k], dst)` in `_restore_media`.

- **1.4 [MEDIUM] `audio_peaks` unbounded ffmpeg OOM & sticky empty cache** (`remote_excerpts.py:audio_peaks`, L144–169)
  `subprocess.run(..., capture_output=True)` buffers unlimited PCM before slicing `[:200000]`; a tiny crafted Opus/FLAC of silence can OOM. Unchecked `returncode` caches `{"rate":50,"peaks":[]}` forever on failure.
  *Fix:* Pass `-t 4000` (or bound stdout via `Popen`); require `returncode == 0` and `len(out) >= 40` before caching.

- **1.5 [MEDIUM] Show files can modify `safe` & extra `recorder` keys** (`shows.py:clean_settings`, L290; `clean_recorder`, L259–264)
  `clean_settings` imports `"safe"` (**unverified** in `control.py`; if photosensitive safe mode, a file can disable it). `clean_recorder` passes all `raw` keys to `recorder.validate_cmd` (**unverified** if output paths are accepted) instead of only the 5 saved keys.
  *Fix:* Drop `"safe"` (or forbid disabling it); filter `clean_recorder` to `("source","fps","res","audio","quality")`.

## 2. Live-Show Failure Modes

- **2.1 [HIGH] `RE-ENABLE` broken in `LATCH` mode; overrides stuck** (`control_methods.py:_step_timeline`, L155–158; `_tl_do`, L330–332)
  `reenable` sets `o[0] = -1e18`, but L156 checks `tl.ovr_mode == "latch" or clock - o[0] < self.tl_hold`—always `True` in `latch` mode, so overridden lanes never return to automation. Muted/empty lanes `continue` at L153 without pruning `_tl_ovr`.
  *Fix:* Set `o[0] = None` on `reenable` and check `o[0] is not None and (tl.ovr_mode == "latch" or clock - o[0] < self.tl_hold)`; prune `_tl_ovr` for empty/removed lanes.

- **2.2 [HIGH] Recording breaks on loop wrap & punch-out (erases automation outside punch)** (`control_methods.py:_step_timeline`, L106, L114–118, L184–193; `_tl_pass_end`, L76–91)
  *Loop wrap:* `_tl_pass_end()` runs after `advance()` wraps `tl.pos` (`< g["last"]`, `< t0`), skipping handback keys, emptying `simplify_range`, shrinking triggers to `REC_GAP`, and skipping `replace` `clear_range`.
  *Punch-out:* `rec` is sampled before `advance()`. Crossing `punch["b"]` writes past `punch["b"]`, then `rec` turns `False` next frame without calling `_tl_pass_end()`. `_rec_g` stays leaked; stopping later (e.g., `t=100`) wipes all automation in `(punch["b"], 100]`.
  *Fix:* Sample `was_rec` before and `now_rec` after `advance()`; on `was_rec and (wrapped or not now_rec)`, pass `end_pos` (`lp["b"]` on wrap, `punch["b"]` on punch-out, else `tl.pos`) to `_tl_pass_end(end_pos)`. Clear `[p0, lp["b"])` and `[lp["a"], tl.pos)` on wrap.

- **2.3 [HIGH] Preset cue validation mismatch crashes show open & startup session** (`timeline.py:preset_ref`, L84–91; `shows.py:_load_preset_ref`, L973–975, L820–821, L883–884)
  `timeline.preset_ref` permits uppercase, leading `-`/`_`, length >48, and unknown `kind` (`foo/My_Preset`), passing `clean_show`/`import_zip`. `_do_open`/`_session_plan` then call `_load_preset_ref` -> `preset_path`, raising uncaught `ValueError` and aborting open/session/prefetch.
  *Fix:* Align `preset_ref` with `shows.KINDS`/`SLUG`; catch `ValueError` in `_load_preset_ref` and return `None`.

- **2.4 [HIGH] Blocking I/O & O(N) stalls on 60 fps GL thread** (`control_methods.py:_apply_preset`, L677; `_apply_show`, L548, L553; `_new_show`, L483; `timeline.py:write_key`, L795–803; `active_triggers`, L563–577)
  `midi` preset cues call `self.midi.save()` synchronously inside `_tl_fire` (`_apply_show`/`_new_show` also call `_save_scenes()`/`midi.save()`). `write_key` rebuilds `ts` and slices `items` (O(20k) every 20 ms/param) and never enforces `MAX_KEYS` (excess keys silently truncate on reload). One long trigger (`len=4096`) forces `active_triggers` to scan up to 4,000 items every frame.
  *Fix:* Offload `midi.save()`/`_save_scenes()` to worker; enforce `MAX_KEYS` and cache timestamps in `write_key`; bucket triggers by end time.

- **2.5 [HIGH] Thread races: torn `rev`/`blobs`, unlocked dicts, HTTP/worker file collisions, RAM spikes** (`shows.py:_list_shows`, L630–631; `_list_presets`, L665–666; `state`, L569–575; `export_zip`, L407–421; `remote_excerpts.py:_library_get`, L24, L30; `_library_post`, L89–127)
  `rev[k]` increments before `dumps()` assigns `blobs[k]`, advertising a new `rev` with stale bytes. Worker mutates `cache` while GL reads; `state()` iterating `thumbs` races off-GL. HTTP imports/uploads write into `presets/`/`shows/`/`_media/` concurrently with worker jobs (`unique_slug` TOCTOU `FileExistsError`; Windows `os.replace` `PermissionError` during `ffmpeg`/`file_hash`). `export_zip` and `/media` buffer hundreds of MiB in RAM without HTTP `Range` (breaks `<audio>` seeking).
  *Fix:* Assign `blobs[k]` before incrementing `rev[k]` under a lock; lock `cache`/`thumbs`; route writes through worker; stream exports via temp file and support `Range` on `/media`.

## 3. Correctness

- **3.1 [MEDIUM] `/preset-import` overwrites presets & exceeds `SLUG`** (`remote_excerpts.py:_library_post`, L91–93)
  `unique_slug` checks `(d/slug).exists()` (no `.json`), overwriting `<slug>-i.json` on the 3rd duplicate and pushing 47–48 char slugs past `SLUG`'s 48-char limit (invisible to `_list_presets`/`check_slug`).
  *Fix:* Use `shows.free_preset_slug(d, shows.slugify(p["name"]))`.

- **3.2 [MEDIUM] `chase_items` picks first track over latest & skips cue at `pos`** (`timeline.py:chase_items`, L583–601)
  `if key not in best:` lets an earlier track-1 item beat a later track-2 item; `bisect_left` (`t < pos`) skips a cue at exact `pos` when seeking paused.
  *Fix:* Replace when `it["t"] > best[key][1]["t"]`; use `bisect_right` when `not tl.playing`.

- **3.3 [MEDIUM] Units, discrete RDP, multi-timeline, undo & autosave** (`control_methods.py:_tl_do`, L337–362; `_step_shows`, L418–420; `timeline.py:convert_units`, L762–780; `do`, L654–658; `simplify_range`, L838–847)
  `units` skips `_tl_pass_end()` (mixing units in live gestures), omits clamps on `loop`/`punch`/`trigger.len`/`pos`, and `undo()` leaves `pos` unscaled. Linear RDP drops intermediate `hold` steps (`0->1->2` loses `1`). `tl_new`/`tl_dup`/`tl_select`/`tl_delete` always call `_tl_load_docs()`, halting playback/undo on no-ops or deleting an inactive timeline. `_tl_do` never calls `_changed()` (**unverified** in `control.py`); `"saved"` clears `dirty`/`_show_dirty_t` unconditionally, dropping mid-save edits. Invalid `loop`/`punch` snapshots undo before failing.
  *Fix:* End recording before `convert_units`, clamp fields, restore/clamp `pos` on undo; deduplicate equal neighbors for `ctx.discrete`; reload only when active timeline changes; tag `"saved"` with a dirty seq counter; validate `loop`/`punch` before `snapshot()`.

- **3.4 [MEDIUM] Partial open hijacks `m.current`; `collect all` skips >64 MiB audio; partial morph** (`shows.py:_do_open`, L822–824; `_do_save`, L757; `control_methods.py:_apply_show`, L562; `_morph_base`, L492–503; `_apply_preset`, L651–658; `timeline.py:clean_audio`, L251)
  Partial open (`parts=["timeline"]`) switches `m.current`/`.session.json` to Show B, so Save overwrites Show B. `collect all` caps at `MAX_MEDIA_FILE` (64 MiB), skipping 64–120 MiB audio, and `not dst.exists()` keeps stale copies. `clean_audio` allows `,`/`'`/`[` rejected by `MEDIA_NAME`. `_morph_base` replaces `self._morph` (cancelling other morphs), and stack-only `look` presets (`vals` empty) never start `_morph`, skipping `stack.morph_start`.
  *Fix:* Keep `m.current`/`.session.json` on partial opens; use `MAX_AUDIO` for audio and refresh on hash mismatch; align `clean_audio` with `MEDIA_NAME`; merge `_morph` targets and start `_morph` for stack-only morphs.

## 4. Determinism & 5. Photosensitive Limits (<= 2.5 Hz)

- **4.1 [MEDIUM] Photosensitive bypasses: held `strobe`, rapid cuts, un-slewed snaps** (`control_methods.py:_step_timeline`, L127–140, L150–170; `_tl_fire`, L203–221)
  `strobe` rate is checked only at span start (`i not in _tl_span_ids`); raising BPM or lowering `strobe_rate` mid-span exceeds 2.5 Hz for up to 4,096 beats/s. Alternating `morph=0` cues or `bypass`/`show_texture` events flash at 10–30 Hz (**unverified** if `control.py` slews base). Muting/overriding a `LUMA` lane (`au.pop`) skips `TL_LUMA_RATE`.
  *Fix:* Re-check `strobe` rate every frame while held; apply `TL_LUMA_RATE` post-composition; rate-limit `bypass`/`show_texture` events.

- **4.2 [LOW] Determinism & unverified external boundaries** (`control_methods.py:_tl_fire`, L211–215)
  Uncached preset cues load async on a variable frame vs warm cache. **Unverified:** verify `Controller.panic()` stops `tl.playing`/`tl.rec`, ends gestures, clears `_auto`/`_tl_fade`/`_tl_ovr`/`_tl_trig`/`_morph`, sets `tl_engaged=False`; verify `_apply_state` bounds on `settings`; verify panel DOM escapes `name`/`notes`/`tags`/`text`/`marker`/`notice` (all allow `<>"'&`, XSS risk with `innerHTML`).
