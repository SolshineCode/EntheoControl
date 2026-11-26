# Opus review: multi-surface mapping, text layer, recorder (2026-11-25, read-only)

Scope: surfaces.py, surface_render.py, text_layer.py, recorder.py, their hooks in live_visuals.py / remote_control.py /
control.py / panel.html, tests/test_recorder.py. Static review only: nothing was run (live show on the laptop).
Confidence: H = traced in code, M = depends on driver / OS behaviour I could not run, L = plausible.

## HIGH

**H1. A 0x0 framebuffer permanently turns off multi-surface output for the rest of the show** (H)
`surfaces.py:1012-1013` computes `sx = fb_w / output[0]` and then `1.0 / sx`. When the projector window is minimised,
`glfw.get_framebuffer_size` returns (0, 0) (`live_visuals.py:2654`). That raises ZeroDivisionError. `live_visuals.py:2660-2663`
counts it as a failure, and after 5 frames `map_ref["fails"]` stays at 5 forever, because it is only reset on success and
success is never tried again. Ways to get a 0x0 framebuffer: Win+D / "show desktop", clicking the taskbar, the preview
window minimised, or possibly a frame or two after `restore_window` on projector re-plug (M). After that the engine draws the
old corner pin through the selected surface only: masks, the other faces, black outside the mask, glow and span are all gone,
so light lands on the wall or the audience. Only a restart fixes it.
Fix: skip the output pass (and `recorder.frame`) when `fw * fh == 0`. Also guard `render_items` (return [] if fb_w or fb_h
<= 0). Make the fallback retry, e.g. try multi again every 2 s while fails >= 5, and show it in the panel state.

**H2. On crash or exit, `Recorder.close()` blocks for minutes before the process can exit** (H)
`live_visuals.py:2760` calls `recorder.close()` (default `wait=True`) in the engine's `finally`, before `glfw.terminate()`.
`close()` (`recorder.py:1242-1262`) joins bg threads for 10 s each, then waits up to 120 s for `_finalize`. `_finalize` itself
does `writer.join(60)`, `proc.wait(60)` and a full mux (`recorder.py:797-830`: AAC encode of the whole show, then
`+faststart` rewrites the whole file). With an hour or more recorded, a crash while recording leaves the projector frozen
for 30 to 120+ s and the watchdog cannot restart. Without the recorder that would take seconds.
Fix: on the exception path, close ffmpeg's stdin, wait a few seconds at most, and leave `.parts/` for `recover()`. The parts
are fragmented MP4, built for exactly this. Only Shift+Esc should finalize, and it should print a "saving recording" line.

**H3. A failed mux loses the audio for good: the next start deletes it** (H)
When `_mux` fails (`recorder.py:837-846`), the video is moved out to `recordings/<name>.mp4` and the notice says the parts
are kept. On the next start, `recover()` (`recorder.py:1148-1151`) finds `.parts/<name>/` with no `video.mp4` and runs
`rmtree`. That deletes `audio.pcm`, the only copy of the show audio. A mux failure is likely on long shows: free space is only
checked against 500 MB (`recorder.py:56, 1215`), but the mux writes a second full-size copy (`.muxing.mp4`).
Fix: copy the video instead of moving it, or have `recover()` keep any dir that still has a non-empty `audio.pcm`. Refuse or
stop recording when free space < (current part size + 1 GB), so the mux always fits.

## MEDIUM

**M1. A failed PBO allocation leaks the pool count; after that, projector-source capture is dead for the session** (H code / M trigger)
`GLCapture._pbo` (`recorder.py:572-581`) does `count += 1` before `ctx.buffer(...)`. With a (0, 0) projector reader, moderngl
raises ("buffer cannot be empty"). The real buffers in `free` get released (count--) and are replaced by phantom counts. Once
the window is back, `count == pool`, `free` and `pending` are empty, and `grab()` calls `pending.popleft()` -> IndexError on
every frame. The video freezes on the last frame (the writer pads). Same trigger as H1, source=projector only.
Fix: increment `count` only after the buffer is created, refuse `nbytes == 0`, and guard `popleft` on an empty deque.

**M2. A MIDI CC learned to "rec" toggles recording on every message** (H)
`control.py:1342` treats any CC >= 64 as a press with no edge detection, and `control.py:1395` emits a toggle. A fader or knob
(or a pad sending CC with aftertouch-style repeats) learned to `rec` starts and stops dozens of sessions per second. Each one
spawns ffmpeg and an NVENC session, a parts dir and a finalize thread. `Recorder.command` (`recorder.py:1000`) has no debounce.
Fix: rising-edge detection for CC presses (last value < 64 -> >= 64), and ignore `toggle` within ~1 s of the last one.

**M3. The replay buffer restart races on one shared directory** (H)
Every replay session uses `recordings/.replay` (`recorder.py:647-649`). Changing any setting while the replay is armed
(`recorder.py:1016-1023`) stops the old session and starts a new one right away. The new `start()` runs `rmtree` + `mkdir`
on the dir (`:657-658`) while the old ffmpeg may still be writing `seg%03d.ts` / `list.csv`. Then the old `_finalize` runs
`rmtree(self.parts)` (`:821-822`) on the new session's live ring. Result: corrupt or missing segments, and "save replay" fails.
Fix: give each replay session its own directory (`.replay/<stamp>`), and clean up only its own.

**M4. Racing or failed encoder probes silently fall back to libx264 on the CPU (thermal)** (M)
The background probe's result is only cached (`recorder.py:892-893`). A REC press before the probe finishes probes again in
parallel (`:660`), so two NVENC probe sessions run at once. A transient NVENC failure is cached for the life of the process
(`:137`), and the show then encodes on the CPU at 30/60 fps on a laptop that already overheats.
Fix: one probe guarded by a lock or Event that `start()` waits on (it runs off the render thread anyway), no caching of a
result without NVENC (re-probe at the next start), and a loud panel warning when the encoder is not NVENC.

**M5. Heavy colour conversion runs on the CPU** (M)
`-pix_fmt yuv420p` (`recorder.py:147`) plus `vflip`, and for 1080p a bicubic `scale` + `pad` (`:163-167`), all run in ffmpeg's
swscale on the CPU: 1080p60 means about 480 MB/s of raw RGB converted on the CPU. On top of that, `b.read()` allocates and
copies 4-8 MB per frame on the render thread (`:599`).
Fix: feed `rgb0` straight to h264_nvenc (it accepts rgb0 / bgr0) and drop `-pix_fmt yuv420p` for NVENC. Do the flip and the
1080p letterbox in GL (one blit into a recorder FBO) so ffmpeg does no filtering. Reuse one bytearray via `read_into`.

**M6. A failed mapping autosave is never retried, and exit skips it** (H)
`Mapper.save()` sets `_edit_t = None` even when the write fails (`surfaces.py:1065-1071`). On Windows `os.replace` can fail
for a moment (AV / indexer sharing violation). `flush()` (`:1078`) only acts when `_edit_t` is set, so the edits are lost at
exit with only a console line. The write also runs on the render thread (`tick()`, `live_visuals.py:2531`), which is usually
sub-ms but can stall when AV scans it.
Fix: on failure keep `_edit_t` (retry after another 1.5 s with backoff), `flush()` on `set.dirty`, show `can't save` in panel state.

**M7. Two capture sources, or a size change, churn PBOs and force early maps (GPU stall)** (H code / M impact)
`_pbo` releases any free buffer of the wrong size (`recorder.py:573-577`). With pool 4 and latency 2, two grabs per frame
exhaust the pool and `grab()` maps the oldest PBO early (`:587-590`), which is a synchronous readback. This happens when the
file recording uses program and the replay buffer was restarted with source=projector (settings changes restart only the
replay). Fix: key the free lists by size, pool >= sources x (latency + 1). Optionally latency 3, since the NVIDIA driver queues up to 3 frames.

**M8. A projector unplug while recording causes a burst of duplicate frames on resume** (H)
While `st["proj"]` is False the loop `continue`s before `recorder.frame` (`live_visuals.py:2542-2545`). On resume, the first
slot is far ahead and the writer pushes every missing slot as a duplicate (`recorder.py:268-271`). For 5 minutes at 30 fps
that is 9000 frames x 4 MB through the pipe. NVENC is busy for many seconds, new frames are dropped (queue full), and the
CPU memcpy load is high. Fix: cap the fill (e.g. > 2 s gap -> write 1 frame per second of gap, or keep calling a cheap
`recorder.tick()` during the pause that feeds a black/last frame at 1 fps).

**M9. Font enumeration is not pre-warmed and runs GIL-heavy work during the show** (M)
`list_fonts()` opens every font file with Pillow under `_FONTS_LOCK` (`text_layer.py:337-357`). It first runs when the TEXT tab
calls `/fonts` or the first `set_text` reaches the worker. That is seconds of Pillow work competing with the render thread for
the GIL, and the lazy `/font-preview/*` requests (one FreeType load each, `:374-405`) add a burst while scrolling.
Fix: warm `list_fonts()` in a bg thread at engine start (before the show), and persist the name->path list to a small cache
keyed by the folder signature.

## LOW

- **L1** `surfaces.py:671`: if surfaces.json exists but cannot be read (OSError), `mapping.json` is not read either, so the
  show runs full frame instead of the user's corner pin. Reading legacy is harmless because saving is disabled anyway. Drop `and can_save`.
- **L2** `surfaces.py:628-632`: unusable surfaces are dropped silently and the file is rewritten without them on the next edit. If
  `_quarantine` fails (`:551`), the corrupt file is overwritten. Copy the original to `.bak` before the first save after a lossy load.
- **L3** `surface_render.py:80,82`: `pow(x, 2.0)` with negative x is undefined in GLSL (NaN on some drivers). It works on this
  NVIDIA, but `x*x` costs nothing.
- **L4** `surfaces.py:294-323, 703-710`: a concave mask clipped by the quad (Sutherland-Hodgman) can produce zero-width bridge
  edges along the quad border. The SDF then puts edge glow / feather along a line where there is no surface.
- **L5** `surfaces.py:741`: the span canvas is the bbox of the *enabled* span faces, so switching one face off mid-show
  rescales and shifts the picture on all the others.
- **L6** `surface_render.py:127-149, 184-196`: name tags load a TrueType font and rasterize on the render thread (one hitch of
  ~tens of ms when overlay=ALL is turned on with 12 surfaces, and on every rename keystroke). `_labels` is never pruned for deleted ids.
- **L7** `text_layer.py:1163-1184`: if the instanced draw raises between `enable(DEPTH_TEST)` and `disable`, `finally` only
  re-enables what was on, so DEPTH_TEST leaks. The default framebuffer has a depth buffer, so the old corner-pin path could
  fail the depth test. Always `disable` it in `finally` unless it was on.
- **L8** `recorder.py:89`: `bool(c["replay"])` makes the string "false" true. Require a real bool, as surfaces/control do.
- **L9** `recorder.py:440-449`: drift steering repeats or drops up to 16 samples with no crossfade, which can tick on tonal
  material. A short linear crossfade over the splice is enough.
- **L10** `remote_control.py:52`: MAX_BODY 16 KB is below a full 24-surface `replace` with 8-point masks (~17 KB).
- **L11** `panel.html:2227`: rec toasts compare engine `time.time()` with the phone's `Date.now()`. With more than 20 s of clock
  skew the encoder-death toast never shows. Use the state's own arrival time.
- **L12** `text_layer.py:269-286`: one invalid phrase in live_state.json drops the whole phrase list (it is validated as one
  command) and the next save persists the loss.

## Checked and fine
- Homography: `is_convex` (cross > 4 px^2, either winding) gates every corner edit, load and replace, so a convex quad never
  makes the solve singular, and w keeps one sign inside the quad (no NaN inside a surface). Near-collinear corners only stretch.
  `_solve` / `invert3` raise and are caught, and `render_items` skips those surfaces.
- Command validation: finite, typed, bounded on the server thread. Positions are clamped to overscan when applied, the gesture
  id is length-capped, and names are stripped of non-printables.
- Panel XSS: surface names, phrases, font names and rec notices all go through `textContent` / `encodeURIComponent`.
  `innerHTML` is only used with engine numbers.
- Text: rasterization and the font scan run on the worker, newest request wins, NaN-guarded field in both GLSL stages.
  Texture unit 11 is shared by the text trails and the surface labels, but `p_l_text` comes from `layer_on`, so the scene only
  samples it when TextPass bound it that frame.
- Migration: the old engine already required a convex pin (`corners_ok`, which is stricter in px than `is_convex`), so any pin it
  accepted migrates. `mapping.json` is never written unless an edit or SAVE happens.
- Blend state: SurfaceRenderer restores BLEND off + DEFAULT_BLENDING. The other passes set their own blend func.

## tests/test_recorder.py: what is thin or missing
All 28 tests listed in HANDOFF exist. The MIDI test (`:536`) and the remote test (`:551`) are complete but minimal:
- MIDI: only a note_on press -> toggle. No test that note_off / a CC fader do not toggle repeatedly (M2).
- Remote: only start / set(fps "60") / a bad op. No bool type checks (L8), no `type` alias for the other ops, no oversize body.
- Not covered: real `GLCapture` (0-size reader, pool exhaustion, two sources: M1/M7), the crash-path `close()` time (H2),
  mux failure followed by `recover()` (H3), replay restart while the old ring is finalizing (M3), resume after a long pause (M8),
  Ctrl+R gating while calibrating (engine key handler, `live_visuals.py:2181-2182`).
