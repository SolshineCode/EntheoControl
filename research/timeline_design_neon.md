# Timeline, Presets & Show-File Design Research + Critique

## 1. Sources Table

| Product | Official Documentation URLs (Verified) | Topics Covered |
|---|---|---|
| **Resolume Arena 7** | https://resolume.com/support/en/composition<br>https://resolume.com/support/en/media-manager<br>https://resolume.com/support/en/presets<br>https://resolume.com/support/en/autopilot<br>https://resolume.com/support/en/envelopes | `.avc` compositions (path refs), Media Manager (`Locate All...` / `Collect Media`), XML presets, Autopilot, phase Envelopes |
| **Ableton Live 12** | https://www.ableton.com/en/live-manual/12/managing-files-and-sets/<br>https://www.ableton.com/en/live-manual/12/automation-and-editing-envelopes/<br>https://www.ableton.com/en/live-manual/12/working-with-the-browser/<br>https://www.ableton.com/en/live-manual/12/arrangement-view/<br>https://www.ableton.com/en/live-manual/12/session-view/ | `.als` Sets, Collect All and Save, Missing Files, Crash Recovery, Browser tags/Collections, Automation Arm, Re-Enable Automation, Back to Arrangement, Punch-In/Out, Chase MIDI Notes |
| **Bitwig Studio 5** | https://www.bitwig.com/userguide/latest/working_with_projects_and_exporting/<br>https://www.bitwig.com/userguide/latest/the_project_panel/<br>https://www.bitwig.com/userguide/latest/automation/<br>https://www.bitwig.com/userguide/latest/the_clip_launcher/<br>https://www.bitwig.com/userguide/latest/browsers_in_bitwig_studio/ | Project folder + Collect and Save, Touch / Latch / Write automation, Restore Automation Control, Automation Follow, absolute/additive/multiplicative clip automation, Clip Launcher vs Arranger |
| **Millumin 5** | https://help.millumin.com/docs/playback/timeline/<br>https://help.millumin.com/docs/playback/dashboard/<br>https://help.millumin.com/docs/connect/monitor/<br>https://help.millumin.com/docs/connect/timecodes/<br>https://help.millumin.com/docs/general/tips/ | Dashboard (column cue sequencer) vs Timeline (linear + Segments), embedded timelines, Monitor Cues, Timecode, Collect Files & Optimize |
| **TouchDesigner** | https://docs.derivative.ca/Animation_COMP<br>https://docs.derivative.ca/Keyframe_CHOP<br>https://docs.derivative.ca/Virtual_File_System<br>https://docs.derivative.ca/Safe_Mode<br>https://docs.derivative.ca/Palette | `.toe`/`.tox`, Animation COMP + Keyframe CHOP, Virtual File System (`vfs:`), `CrashAutoSave.toe` + Safe Mode (non-cooking), Palette |
| **Notch** | https://manual.notch.one/1.0/en/docs/reference/user-interface/timeline/<br>https://manual.notch.one/1.0/en/docs/reference/user-interface/curve-editor/<br>https://manual.notch.one/1.0/en/docs/reference/user-interface/properties/<br>https://manual.notch.one/1.0/en/docs/reference/user-interface/preferences/ | `.dfx` projects, Timeline & layers/compositions, Auto Key & Curve Editor (Bezier/easing, snapping), Property Presets & Assets, Autosave & recovery |
| **Synesthesia** | https://app.synesthesia.live/docs/manual/presets/<br>https://app.synesthesia.live/docs/manual/settings/<br>https://app.synesthesia.live/docs/faq/midi/<br>https://app.synesthesia.live/docs/faq/troubleshooting/ | Presets with muteable channels (Scene/Meta/Media), 10 FavSlots, Playlists (timers & audio-reactive switches), MIDI Pickup & Feedback, `.synprofile` backups |

---

## 2. Research Summary per Product

| Product | Project / Bundling / Recovery | Presets & Partial Recall | Timeline, Automation & Live Override |
|---|---|---|---|
| **Resolume Arena 7** | `.avc` XML path refs; Media Manager (`Locate All...` relink, `Collect Media` copy); Advanced Output & MIDI stored separately; native autosave: *unverified* | XML presets per effect/source/param/envelope/layout; no whole-look partial-recall matrix | No master timeline; clip/column cues + Autopilot + per-param phase Envelopes (BPM/Timeline/FFT) + clip Cue Points |
| **Ableton Live 12** | Project folder + `.als`; `Collect All and Save`; File Manager relink; auto + manual `Crash/` recovery; Template Sets | Browser with tags, filters, Collections (favorites), search; device/rack presets | Session (quantized live) + Arrangement (beats); Automation Arm, Punch-In/Out (on loop brace), override + **Re-Enable Automation** & **Back to Arrangement**; Chase MIDI Notes |
| **Bitwig Studio 5** | Project folder + `Collect and Save` (Project Panel); templates; plugin crash sandboxing; autosave interval: *unverified* | Unified Browser with tags, favorites, search; device & clip presets | Clip Launcher + Arranger; **Touch / Latch / Write** modes; **Restore Automation Control**; **Automation Follow**; absolute/additive/multiplicative clip automation |
| **Millumin 5** | Project file + relative-folder auto-search + `Collect files...` (copy) + `Optimize`; custom autosave interval: *unverified* | Reusable templates; Dashboard columns act as cue snapshots | **Dashboard** (cue columns) + **Timeline** (seconds/timecode, keyframes, looping **Segments**); timelines embedded in Dashboard; Monitor Cues; declarative seek |
| **TouchDesigner** | `.toe`/`.tox`; **VFS (`vfs:`)** embeds media; **`CrashAutoSave.toe` boots in Safe Mode (cooking disabled)** | **Palette** drags `.tox` modules; native tagged param-preset browser: *unverified* | **Animation COMP** + **Keyframe CHOP** (curves evaluated by frame/seconds/lookup); deterministic on seek |
| **Notch** | `.dfx` packages graphs/timelines/assets; configurable **Autosave** + recovery dialog; standalone external collect menu: *unverified* | Node Property Presets + reusable **Assets** (exposed params); Resource Browser thumbnails & favorites | Layer Timeline + **Curve Editor** (Bezier/easing, snap to grid/keys/markers, loop, **Auto Key**); multi-composition (Isolated/Composited) |
| **Synesthesia** | Local `Data` dir + single-file **`.synprofile`** backups; bulk/selective preset export | Presets with **3 muteable channels (Scene / Meta / Media)** for 1-tap partial recall + **10 FavSlots** | No linear timeline; **Playlists** (scene+preset pairs, timers, audio-reactive transitions); **MIDI Pickup Mode** |

### Core Patterns to Adopt
- **Venue Rig vs. Show Content (Resolume):** Keep projection warping and MIDI mappings decoupled from show loads so switching shows mid-festival never overwrites stage calibration.
- **Safe-Boot Crash Recovery (TouchDesigner):** Recovering into frozen/safe output prevents GPU crash loops and accidental blinding flashes.
- **Collect + Relink Together (Resolume, Ableton, Millumin):** Lightweight refs during editing, non-destructive *Collect* (copy) for export, and recursive folder relink for backup laptops.
- **1-Tap Partial Recall (Synesthesia):** Muteable category chips right on the preset bar beat modal dialogs live.
- **Clean Live/Timeline Handoff (Ableton, Bitwig, Synesthesia):** Explicit override tracking, **Re-Enable Automation**, **Automation Follow**, and **MIDI Pickup** make live tweaking safe.
- **Looping Segments (Millumin):** Named loopable sections bridge fixed linear timelines and improvised DJ/band sets.

---

## 3. Critique of Claude's Draft Design

### MUST (Data Loss, 3 AM Live Failures, Safety)

1. **Non-atomic writes & mtime recovery risk total show corruption on power loss.**
   - *Change:* Write `show.json` and `autosave.json` via `<file>.tmp.<pid>` -> `flush()` + `os.fsync()` -> atomic `os.replace()` (with Windows `PermissionError` retry). Detect unclean exit via a `session.lock` file + embedded `base_saved_ts` instead of filesystem mtimes.
   - *Why:* Standard file truncation yields 0-byte JSON on power loss/BSOD; mtimes skew across exFAT USB sticks, zip extraction, and DST changes.

2. **Crash recovery must boot in Safe Mode, never straight into playback.**
   - *Change:* On RECOVER after an unclean exit, load with **transport STOPPED, timeline disarmed, camera OFF, master at 0 / PANIC-safe look**, and a "RECOVERED — TAP TO RESUME" banner.
   - *Why:* If heavy GPU state or bad params triggered the crash, auto-resuming causes an instant crash loop or blinding crowd flash (TouchDesigner Safe Mode: https://docs.derivative.ca/Safe_Mode).

3. **Photosensitive protection has bypass holes, and `camera on` is a privacy risk.**
   - *Change:* (a) Enforce luminance slew and flash spacing (`<= 2.5 Hz` reversals in a rolling 1 s window, flash gap `>= 0.4 s`) in a **post-composition frame-level governor** (after scene morph + timeline + mods + triggers, right before GPU upload)—not just on automation lanes. (b) Remove `camera on` from timeline events (allow `camera off`, or require per-session opt-in defaulting to OFF).
   - *Why:* A 0.2 s lane slew still permits 2.5 Hz waves that, summed with audio mods, `morph=0` scene cuts, or `flash` triggers, exceed safe limits. Allowing `camera on` contradicts `(never camera ON)` in `show.json` for shared `.entheo` files.

4. **Loading a show must NOT silently overwrite venue `surfaces` and `midi`.**
   - *Change:* Keep `surfaces` and `midi` in `show.json` as optional snapshots, but default Open Show to **Keep Current Venue Rig (Surfaces + MIDI)** with explicit opt-in checkboxes.
   - *Why:* Opening a home-built show on a mapped festival stage would wipe out hours of projection warping and controller bindings (Resolume separates Advanced Output & MIDI from `.avc`: https://resolume.com/support/en/composition).

5. **Missing-media relink is omitted, and `.entheo` import can freeze/OOM the HTTP server.**
   - *Change:* (a) Always open shows even with missing `refs` (neutral/transparent fallback + **Media Manager** with **"Scan Folder to Relink"** matching `sha256` then filename). (b) Run `ThreadingHTTPServer`, stream `.entheo` uploads to disk in chunks, and enforce a **streamed uncompressed byte cap** during extraction (don't trust `ZipInfo.file_size` headers).
   - *Why:* Backup laptops always have shifted media paths (Resolume Media Manager: https://resolume.com/support/en/media-manager). Single-threaded `http.server` + in-memory zip reads stall control heartbeats or OOM.

6. **Writing automation directly into BASE after morph breaks morphs and dirties manual state.**
   - *Change:* Separate `manual_base[p]` from `auto_value[p]` (see Q1), evaluate lanes only across active spans/enabled states, and badge automated faders in the UI.
   - *Why:* Held keyframes written into BASE after morph silently cancel later scene morphs and permanently corrupt manual faders after STOP (Ableton/Bitwig model: https://www.ableton.com/en/live-manual/12/automation-and-editing-envelopes/).

7. **Edge-triggered duration items stick on STOP/SEEK/LOOP and break determinism.**
   - *Change:* Model sustained triggers, text items, and state toggles (`bypass`, `filter on`) as **declarative spans `[t0, t1)`** evaluated purely from playhead `t`; reserve edge firing strictly for one-shots (scene recall, next video). Pass injected `dt` everywhere (never `time.monotonic()`).
   - *Why:* Paired start/end edges leak stuck states on seek/stop/loop wrap; wall-clock reads break the determinism test.

8. **Browser audio breaks on phone sleep and duplicates across multiple connected devices.**
   - *Change:* Enforce a single **"Audio Host"** client lock (off by default on secondary devices), compute waveform peaks server-side (tiny JSON, never decoding huge WAVs in phone RAM), and strongly consider optional engine-side audio playback on the show laptop.
   - *Why:* Mobile browsers suspend `AudioContext` on screen lock/tab switch; phone + laptop connected simultaneously would play double out-of-phase audio.

### SHOULD (Strong Architecture & UX Fixes)

9. **Phone/touch safety: Timeline Edit Lock & coalesced Undo.**
   - *Change:* Add a **Timeline Edit Lock** (default ON live and on narrow phone viewports) so scrolling/scrubbing can't drag keyframes accidentally. Coalesce continuous drags/recordings into single undo commits on pointer-up/record-stop; add **Automation Follow** (Bitwig: https://www.bitwig.com/userguide/latest/automation/).
   - *Why:* Tiny handles on a phone in a dark booth cause accidental edits; per-tick undo entries exhaust 50 steps in a second.

10. **Decouple Recording Write Modes from Live Override, and add MIDI Pickup.**
    - *Change:* Add **Automation Arm** separate from Transport Record (default **Touch-Overdub**, never destructive Replace), use real `pointerup` + glide for touch sliders vs. configurable 2–3 s timeout for non-touch MIDI knobs, and add **MIDI Pickup** (Synesthesia: https://app.synesthesia.live/docs/faq/midi/).
    - *Why:* A hard-coded 1 s timeout reclaims control if a VJ pauses mid-sweep; raw MIDI CC causes sudden value jumps.

11. **Versioned presets, unknown-param tolerance, and 1-tap recall chips.**
    - *Change:* Add `{"format":"entheo-preset","version":1}`, warn-and-ignore unknown params on load, place Synesthesia-style **partial-recall chips** (Visuals / Mods / Stack / FX) directly in the preset bar (https://app.synesthesia.live/docs/manual/presets/), and run async thumbnail readback on save only.
    - *Why:* Param registries evolve across versions; modal dialogs are unusable mid-set.

12. **GIL-safe autosave throttling.**
    - *Change:* Debounce autosaves (15 s after last edit, max 60 s) and defer JSON serialization during active recording or heavy frames (>12 ms).
    - *Why:* Python's GIL during `json.dumps` on a large timeline can cause visible 60 fps hitches even from a worker thread.

### COULD (Polish & V2)

13. **Quantized Section jump + looping Segments:** Tap a Marker/Section on PERFORM to queue a bar-quantized jump or hold-loop until released (Millumin: https://help.millumin.com/docs/playback/timeline/).
14. **Segment curve tension:** Optional 4th tuple element `[t, v, ease, curve]` (`-1..1`) for musical exponential sweeps without dense keyframes (Resolume Envelopes / Notch Curve Editor).
15. **Relative/Additive automation lanes:** Optional `offset` mode (Bitwig) so a VJ rides base levels while curves modulate on top.
16. **Pre-export "Optimize / Prune Unused Media":** Strip unreferenced assets before zipping `.entheo` (Millumin `Optimize`).

---

## 4. Answers to Open Questions

1. **Automation into BASE vs. separate output layer: which is safer live?**
   - **Use a virtual `effective_base` stage between `manual_base` and downstream mods/slew/bypass/governor.** Keep `manual_base[p]` intact; active un-overridden lanes drive `effective_base[p] = auto_value[p]` (fader visually tracks with an `AUTO` badge). Touch/Latch override hands `effective_base[p]` to the performer; **Re-Enable Automation** glides smoothly back to `auto_value[p]`. Stopping the timeline never mutates `manual_base`, and mods, BYPASS, slew, and photosensitive caps stay inviolable downstream.

2. **Should a timeline scene cue respect global recall quantize, or fire exactly where placed?**
   - **Fire exactly where placed.** Timeline cues are already snapped on the grid (or intentionally placed at exact audio hits). Layering global launch quantize delays a bar-16 cue to bar 17 on a 1 ms slip and breaks determinism. Reserve global recall quantize strictly for manual/live triggers (Ableton Session vs. Arrangement).

3. **Chase on seek: recall the last cue before the playhead, or not? What do Millumin / Ableton do?**
   - **Yes—chase continuous state + latest cue on seek by default (toggleable), and NEVER chase one-shot triggers/flashes.** Ableton chases automation/audio and MIDI notes by default (`Chase MIDI Notes`); Millumin and Notch evaluate all layers/keyframes declaratively at `t`. On seek to `t`, resolve the latest cue `<= t` (interpolating if `t` lands mid-morph), evaluate active lanes and declarative spans at `t`, apply a brief eye-safe 100–150 ms crossfade, and skip past flashes/one-shots.

4. **One show-level timeline, or several timelines ("songs"/"sets") per show with a playlist? Minimum for v1?**
   - **Use `timelines: [...]` + `active_timeline_id` in the v1 schema, but ship v1 UI with one active timeline + a simple switcher (New / Duplicate / Select).** Multi-song sets rarely fit one 2-hour timeline (Millumin Timelines / Notch Compositions); putting an array in v1 costs negligible code, avoids a v1->v2 migration, and lets v2 add playlist chaining.

5. **Scenes inside the show file vs. global `scenes.json`: should opening a show replace the 16 slots?**
   - **Yes—opening a show replaces the active 16 slots** (prompting/backing up if dirty), while **`presets/` stays the global cross-show library.** Timeline cues depend on the show's scene bank (same split as Resolume `.avc` decks vs. presets and Ableton `.als` vs. User Library). Migrate legacy `scenes.json` into a `"Scratchpad"` show on first boot, add "Save Slot to Look Preset" / "Import Slot from Show", and let cues reference snapshots/preset IDs or warn before clearing a referenced slot.

6. **Best way to sync browser-played audio to a beat-unit playhead when BPM can change (tap / MIDI clock)?**
   - **Separate the two workflows—don't force variable-BPM audio warping in v1:**
     - **Audio-locked (`seconds` unit):** Make **audio the clock master** (engine-side playback preferred, or one locked browser `AudioContext` sending `(audio_time, server_ts)` sync beacons with RTT estimation; engine slews playhead gently). Optional tempo map converts `seconds <-> beats` for grid snapping.
     - **Variable-BPM (`beats` unit, v2):** Require clip `native_bpm` + `beat_offset`, drive a **proportional PLL** on `playbackRate = (bpm / native_bpm) + k_p * phase_error` (clamped `0.85–1.15`, smoothed via `setTargetAtTime`, crossfaded hard-seek only if `|error| > 150 ms`). Because `AudioBufferSourceNode.playbackRate` shifts pitch, pitch-preserved warping requires `<audio>` (`preservesPitch = true`) or a WASM/AudioWorklet time-stretcher.

---

## 5. Recommended v1 Scope

### Ship in v1
- **Shows (`shows/<slug>/show.json`)**: Versioned manifest (`format`, `version:1`, `timelines:[]`), atomic write+`fsync`+rename, 10 backups, debounced autosave + `session.lock` **Safe Recovery Mode**, **Keep Current Venue Rig (Surfaces + MIDI)** default on Open, templates, streaming `.entheo` zip export/import (zip-slip + uncompressed byte cap + Collect Media), **Media Manager** with recursive relink (`sha256` + filename).
- **Presets (`presets/<kind>/<slug>.json` + `.jpg`)**: Versioned JSON (`look/layer/fx/mod/mapping/midi`), unknown-param tolerance, tags/favorites/search, 1-tap partial-recall chips, morph/snap recall, async thumbnail on save, single-preset import/export.
- **Timeline Core**: Single active timeline (multi-timeline schema), `beats` or `seconds`, exact-time Cue track (with morph), Automation lanes (`[t, v, ease]`), Declarative duration spans (triggers, text, bypass), Markers, Loop region, **Chase on Seek**, deterministic pure evaluator (`dt`-injected), **post-composition frame-level photosensitive governor**, PANIC stops transport & clears holds.
- **Live Override & Recording**: Non-destructive `manual_base` vs. `auto_value`, **Touch** (`pointerup` + glide; configurable MIDI timeout) & **Latch** + global/per-fader **Re-Enable Automation**, **Automation Arm** + **Touch-Overdub** with RDP thinning, coalesced 50-step Undo/Redo.
- **UI**: PERFORM transport + next-cue + Re-Enable badge; TIMELINE tab with **Edit Lock** (default ON live/phone), drag/snap/scrub, server-computed waveform peaks + single-client Audio Host lock (`seconds` mode).

### Defer to v2
- Playlist chaining across timelines & quantized looping Segments (Millumin-style).
- Variable-BPM pitch-preserved audio warping (PLL + AudioWorklet).
- Relative/additive lanes, curve tension handles, Automation Follow, MIDI Pickup & Punch-In/Out.
