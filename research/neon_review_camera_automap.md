# Code Review: Crowd Camera + AUTO MAP

- **Reviewer role:** Realtime-graphics / Python / Computer-vision (read-only)
- **Scope:** `camera_worker.py`, `camera_layer.py`, `automap.py`, hooks in `live_visuals.py`, `control.py`, `remote_control.py`, `governor.py`, `panel.html`, `params.py`, `surfaces.py`, and tests `tests/test_camera.py`, `tests/test_automap.py`
- **Summary:** 9 **HIGH**, 8 **MED**, 5 **LOW** findings across render-loop blocking, worker lifecycle/Windows process handling, privacy & token exposure, CV/homography correctness, photosensitive safety, thread safety, validation, and test gaps.

---

## HIGH

### H1. Synchronous render-loop freeze (up to 6.5 s) on `CAMERA OFF`, device switch, or worker crash
- **Location:** `camera_layer.py:592-610`, `camera_layer.py:376-385`, `camera_layer.py:529-530`, `camera_layer.py:629-632`, `live_visuals.py:2457, 2495`
- **Failure scenario:** `CameraLayer.stop()` executes synchronously on the GL render thread via `cam.command()` (`op: "off"`, `"toggle"`, or changing `"device"` in `_set`) and via `cam.pre_update() -> _watch()` when the worker exits. `stop()` calls `link.quit()` (blocking `POST /quit` up to `1.0 s` + `_thread.join(timeout=2.5)` while waiting for a 0.5 s `/frame` long-poll or 2.0 s socket timeout), then `proc.wait(timeout=2.0)`, then `proc.terminate(); proc.wait(timeout=1.0)`. Even in the happy path, waiting for `/frame` + DirectShow `cap.release()` stalls the 60 fps live show for 0.5–2.0 s; on a hung webcam driver it freezes the projector for up to **6.5 seconds**.
- **Fix:** Detach `self.link` and `self.proc` immediately on the GL thread and hand shutdown (`link.quit()`, `proc.wait()`, kill fallback) to a daemon teardown thread so `stop()` returns in `<0.1 ms`. On device change (`_set`), serialize the new `start()` behind the teardown thread's completion so the old worker releases DirectShow before the new one opens it.
- **Confidence:** High

### H2. Privacy: `/automap-image` streams raw camera photos over cleartext LAN HTTP (`0.0.0.0`) and retains them in RAM indefinitely after wizard end / `CAMERA OFF`
- **Location:** `remote_control.py:668-673, 843`, `automap.py:595, 636, 653, 752, 764-785`, `panel.html:3052, 3357`
- **Failure scenario:** `RemoteControl` binds `0.0.0.0:8765` over unencrypted HTTP with `?k=<token>` in URLs (visible on venue Wi-Fi, browser history, and any phone that scanned the QR code). Unlike `/camera-preview` (mask + skeleton only), `GET /automap-image` serves raw 640×360 grayscale photographs of the room/crowd (`white`/`grey`/`out` from `_picture`), contradicting `panel.html:3052` (*"Frames stay on this laptop... never sent anywhere"*). Worse, `AutoMap.image` and `AutoMap.scene` (`grey`, `black`) are **never cleared** when calibration/detection finishes or even when the performer turns **CAMERA OFF**—the raw room photo stays in memory and downloadable at `/automap-image` until manual `RESET` or engine exit.
- **Fix:** Gate `GET /automap-image` to `LOCAL_ADDRS` (`127.0.0.1`, `::1`, matching `remote_control.py:66, 822`) by default (or require an explicit local-only opt-in with a clear LAN warning); automatically clear `AutoMap.image` and `AutoMap.scene` whenever `CameraLayer.stop()` runs, when the wizard finishes/idles out (e.g. 5 min), or when the user closes/leaves AUTO MAP.
- **Confidence:** High

### H3. Privacy & Auth: `GET /still` is permanently enabled while crowd camera is ON and reuses the LAN-exposed `.remote_token` (plus `--remote-token` mismatch)
- **Location:** `camera_worker.py:10-12, 451, 470-476, 580`, `camera_layer.py:572, 856-872`, `live_visuals.py:1182-1200, 2204-2205`
- **Failure scenario:**
  1. Whenever crowd camera is ON, `GET http://127.0.0.1:8767/still?skip=1` serves raw 640×360 camera frames continuously even when AUTO MAP is idle, contradicting `camera_worker.py:10-12`. Because `camera_worker.py` reuses the 48-bit `.remote_token` file shared with the LAN web remote (printed to console/QR codes), any local process with that token can siphon 30 fps raw webcam video.
  2. If `live_visuals.py` is launched with `--remote-token CUSTOM`, `load_remote_token()` returns `CUSTOM` without writing `.remote_token`. `CameraLayer.start()` and `camera_worker.py` read `.remote_token` from disk, causing either startup failure (`"no .remote_token next to the code"`) or token mismatch.
  3. `CameraLayer.still()` re-reads `.remote_token` from disk on every capture instead of using the token handed to the running worker.
- **Fix:** Generate a fresh per-spawn ephemeral secret (`secrets.token_urlsafe(24)`) in `CameraLayer.start()`, pass it to `camera_worker.py` via stdin or an inherited env var (never reuse `.remote_token` and never pass it in `argv`), and store it on `self.link` for `still()`. Gate `/still` behind an explicit time-limited opt-in (`POST /config {"allow_still_s": 60}`) enabled only during active AUTO MAP capture steps and cleared in `finally:`.
- **Confidence:** High

### H4. Worker lifecycle: Windows `SO_REUSEADDR` port hijacking (`8767-8775`) and orphaned grandchild holding webcam behind `venvlauncher.exe`
- **Location:** `camera_worker.py:503`, `camera_layer.py:337-347, 449, 602-610`
- **Failure scenario:**
  1. `camera_worker.py:503` uses stock `ThreadingHTTPServer`, where `allow_reuse_address = True`. On Windows (unlike POSIX; compare `remote_control.py:538-540`), `SO_REUSEADDR` lets a second worker bind the same `127.0.0.1:8767` port without `OSError`, silently splitting/hijacking HTTP traffic if an old worker is still exiting. Additionally, `free_port()` has a TOCTOU race between closing its probe socket and `camera_worker.py` binding ~300 ms later.
  2. Inside a Windows Python venv, `sys.executable` runs `venvlauncher.exe`, which spawns the real `python.exe` as a **grandchild**. When `/quit` fails or times out (e.g., hung DirectShow driver), `proc.terminate()` (`camera_layer.py:607`) only kills the launcher shim. Because the engine (`self.parent`) is still alive, `parent_alive()` returns `True`, leaving the orphaned `camera_worker.py` holding the webcam locked for up to `IDLE_EXIT_S` (15 s)—blocking restart/device switching and violating *"OFF stops the worker and releases the webcam"*.
- **Fix:** Subclass `ThreadingHTTPServer` in `camera_worker.py` with `allow_reuse_address = sys.platform != "win32"` (and set `SO_EXCLUSIVEADDRUSE` on Windows in both `free_port()` and the server, or bind port `0` / retry `8767-8775` inside the worker and emit the bound port on stdout). Assign spawned worker processes to a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (matching `plugins/online/backends.py`) and terminate the job/tree (`taskkill /T /F`) on hard stop.
- **Confidence:** High

### H5. Photosensitive safety: AUTO MAP overlay bypasses `BLACKOUT`, `PANIC`, and `master`; `_project()` can hold patterns `< HOLD_S` under frame lag; no confirm gate
- **Location:** `live_visuals.py:2805-2806`, `automap.py:562-578, 590-591, 615-616, 623-624, 965-990`, `panel.html:3336-3345`
- **Failure scenario:**
  1. `amap.draw(target, (fw, fh))` draws directly over the output target *after* `out_vao`/`surf_draw`, completely bypassing `p_blackout`, `p_master`, and `PANIC`. If someone enters the room or reacts badly during a 30 s Gray-code sequence of full-frame `0 ↔ 170` inversions, hitting **BLACKOUT** (`K`) or **PANIC** (`Esc`) does **not** stop the flashing patterns.
  2. In `AutoMap._project()` (`automap.py:568-577`), `t0 = self.clock()` is recorded *before* waiting for `shown_id() > n0`, and `AutoMapOverlay.draw()` increments `self.drawn` *before* `glfw.swap_buffers()`. If a frame stalls for 0.35 s before drawing/swapping, the pattern stays on screen for `< 0.15 s` (> 3 Hz transition rate).
  3. On a touch panel, a single accidental tap on `1 CHECK CAMERA` or `2 CALIBRATE` immediately overrides the live show with full-frame white/black patterns without any audience-absent confirmation.
- **Fix:** Cancel `amap` immediately on `Controller.panic()` and suppress/abort `amap.draw()` whenever `bo > 0.05` or `vals["master"] < 0.05` (and scale overlay output by `(1 - bo) * master`). Record `t0` *after* `shown_id() > n0` and increment `drawn` *after* `glfw.swap_buffers()` so every pattern is genuinely visible for `>= HOLD_S`. Add a two-tap or modal confirmation in `panel.html` before starting steps 1, 2, or 4.
- **Confidence:** High

### H6. CV correctness: Degenerate homographies cause `NaN`/`Inf` in `apply_h`, uncaught `LinAlgError` crashes `Calibration.load`, and unchecked RANSAC inliers / `inf` RMS break `/state` JSON
- **Location:** `automap.py:50-67, 152-155, 196-206, 209-235, 395-396`, `remote_control.py:965`
- **Failure scenario:**
  1. `automap.apply_h()` (`:67`) computes `q[:, :2] / q[:, 2:3]` with no guard for `q[:, 2] == 0` (vanishing line) or `q[:, 2] < 0` (projective wrap-around), silently producing `NaN`/`Inf` that survive `np.clip` (`:396`) and cause surface creation to fail.
  2. `Calibration.__init__` (`:154`) calls `np.linalg.inv(self.H)` without catching `np.linalg.LinAlgError`. In NumPy, `LinAlgError` inherits directly from `Exception`, **not** `ValueError`, so `Calibration.load()` (`:205`) does not catch it! A corrupt or singular `calibration/calibration.json` crashes `EngineBridge.__init__` on startup, leaving `amap = None` until the file is deleted manually.
  3. `fit_h()` does not check inlier count/fraction: with 10 noisy markers (40 corners), RANSAC can pick any 4 corners (fitting them with exact `0.00 px` residual), and `calib_from_aruco` accepts a completely bogus homography as `"error 0.00 px"`.
  4. `calib_from_gray()` never checks `rms` or `math.isfinite(rms)`: if RANSAC returns 0 inliers (`rms = inf`) or `NaN`, `Calibration.summary()` places `Infinity`/`NaN` into `state["automap"]`, causing `remote_control.py:965` (`json.dumps(..., allow_nan=False)`) to fail and **freeze all `/state` updates for the entire panel**.
- **Fix:** Guard `abs(q[:, 2]) > 1e-9` (and `q[:, 2] > 0` where appropriate) in `apply_h()`; validate shape `(3, 3)`, finiteness, and `abs(det)` / condition number in `Calibration.__init__` (raising `CalibError`) and catch `np.linalg.LinAlgError` in `Calibration.load()`; enforce minimum inliers (`inl.sum() >= max(8, len(src) // 2)` for ArUco, `inl.sum() >= 20` for Gray code) and `math.isfinite(rms)` in `fit_h` / `calib_from_*`.
- **Confidence:** High

### H7. CV & validation: Unvalidated `ai_polys` on server thread, concave 4-point polygons scrambled by `order_quad()`, and dense-warped/invalid quads reject entire `create` batches
- **Location:** `automap.py:279-286, 337-379, 382-407, 686-700, 863-868, 905-910`, `surfaces.py:435-438, 498-500`
- **Failure scenario:**
  1. `automap.validate_command` (`:863-868`) for `op == "ai_polys"` only checks `isinstance(p, list)` and `len(json.dumps(p)) <= 20000` (without `allow_nan=False`), forwarding unvalidated nested JSON to the render thread.
  2. `validate_polys()` (`:362`) checks `_simple(a)` (not convexity), yet marks every 4-point polygon `quad=True` and passes it to `order_quad(a)`. Sorting a **concave** 4-gon by polar angle around its mean scrambles its vertices into a different/self-crossing shape, and `to_surfaces()` emits it as `shape="quad"`.
  3. Even for a convex camera quad, warping through a non-planar `dense` Gray-code map (`:395`) or clamping to `[-0.4W, 1.4W]` (`:396`) can produce a non-convex or collinear quad in projector space. `surfaces.validate_cmd(..., strict=True)` (`surfaces.py:499`) then raises `"corners must form a convex quad"` and rejects the **entire** `replace` command—creating zero surfaces, while `AutoMap.create()` (`:698-699`) already falsely set `self.made` and `self.msg = "N surface(s) created"`.
  4. The same whole-batch rejection happens if `len(keep) + len(made) > 24` (`surfaces.MAX_SURFACES`), or if a polygon has `name: None` (`str(None) -> "None"`) or whitespace-only `name: "   "` (`surfaces._name` raises ValueError). Also `_simple()` (`:377`) uses strict `< 0`, missing figure-8 polygons touching at a shared vertex or collinear overlaps.
- **Fix:** Validate `ai_polys` point types, counts, and finiteness on the server thread; in `to_surfaces()`, check `surfaces.is_convex(p)` in projector space before emitting `shape="quad"` (otherwise convert to `shape="poly"` with bounding-box `corners` and `is_simple` check, skipping degenerate polygons with a warning); sanitize `name` (`None` / whitespace -> default `"Auto N"`); cap total surfaces at `surfaces.MAX_SURFACES`; validate with `surfaces.validate_cmd` before setting `self.made`/`self.msg`; strengthen `_simple()` for touching vertices and collinear segments.
- **Confidence:** High

### H8. Thread safety: Unsynchronized `AutoMap` state (`image`, `found`, `scene`, `report`) causes TOCTOU crashes and violates `publish_state()` immutability
- **Location:** `automap.py:526-534, 652, 667, 679, 691, 772, 777-792, 804-823`, `camera_layer.py:840-848`, `remote_control.py:953-966`
- **Failure scenario:**
  1. `AutoMap` has no lock. `image_jpeg()` (HTTP thread) reads `self.image` twice (`if self.image is None` at `:781`, then `self.image[1]` at `:784`). A concurrent `op == "reset"` on the GL thread (`:822`) setting `self.image = None` triggers `TypeError: 'NoneType' object is not subscriptable` on the HTTP thread.
  2. `pick`, `drop`, `ai_polys`, `create`, and `reset` on the GL thread do not check `self.busy` and mutate `self.found` / `self.scene` while the wizard thread (`_detect` / `_picture`) iterates `self.found` (`:772`), risking `RuntimeError: list changed size during iteration` or a cancelling thread overwriting `reset`'s clean state in `except`/`finally`.
  3. `AutoMap.state()` returns `report=self.report` by reference, and `CameraLayer.state()` returns `settings=self.s` (with mutable `crop` list), `devices=self.devices`, and `ms=ws.get("ms")` by reference—violating `RemoteControl.publish_state()`'s immutability contract (`remote_control.py:953`) so concurrent mutations during `GET /state` JSON serialization raise `RuntimeError` and drop state updates.
- **Fix:** Protect `AutoMap`'s shared fields with a `threading.Lock`, snapshot `img = self.image` into a local variable in `image_jpeg()`, reject `pick`/`drop`/`ai_polys`/`create` while `self.busy` (and guard writes with a run generation ID so a cancelled thread cannot clobber `reset`), and return copies (`list(...)` / `dict(...)`) of all mutable containers in `AutoMap.state()` and `CameraLayer.state()`.
- **Confidence:** High

### H9. CV Stage 3 (`_verify` / `refine_step`): Undamped planar-`H` steps can oscillate/diverge on non-planar boxes, multi-surface outlines interfere, `proj_to_cam` misses folded walls, and stale `cur` clobbers concurrent edits
- **Location:** `automap.py:422-507, 705-755`
- **Failure scenario:**
  1. `refine_step()` (`:505`) applies an undamped `1.0` step using `calib.cam_to_proj(..., dense=False)` (the global planar homography). On a closer/tilted box face whose local scale differs from the back-wall plane, a full planar-`H` step overshoots and oscillates/diverges; `_verify()` never tracks best-so-far or rejects a step that increases error, and never checks `surfaces.is_convex(newq)` (one folded quad fails `replace` for all surfaces).
  2. `_verify()` projects all surfaces' outlines simultaneously (`:724`); for adjacent box faces, `projected_quad` (`search=12` cam px) locks onto the neighboring surface's bright outline.
  3. `physical_quad(grey, self.calib.proj_to_cam(q))` (`:717`) passes raw `grey` (ambient light + wall texture instead of `grey - black`) and uses planar `Hinv` (`proj_to_cam`) to center the `±12 px` search—on a folded wall/box offset by `>12` cam px (such as `test_gray_code_dense_map_handles_a_folded_wall`'s 25 px shift), the search window misses the physical edges completely.
  4. `_edge_lines()` (`:461-464`) uses isotropic Sobel magnitude and never checks `len(P) >= 2` or direction agreement (`abs(vt[0] @ d)`) after median filtering, so a single remaining point or perpendicular texture edge yields a bogus line.
  5. `_verify()` snapshots `cur` before the 5–10 s capture loop (`:707`) and writes it back with `op: "replace"` (`:749`), silently erasing any manual edits the performer made in MAPPING during verification.
- **Fix:** Damp updates (`alpha ≈ 0.65`) or estimate local Jacobian / use dense map when available; enforce convexity and keep the best-error quad per surface (stop/rollback on divergence); project outlines one surface at a time (or spatially separated batches); run `physical_quad` on `np.clip(grey - black, 0, 255)` with directional normal gradient; require `len(P) >= 3` and `abs(vt[0] @ d) > 0.7` in `_edge_lines`; merge updated corners onto the latest `surfaces()` (or send `corners` ops under one gesture id `g`) instead of overwriting with stale `cur`.
- **Confidence:** High

---

## MED

### M1. Render-loop stalls on `automap` `pick`/`drop`/`ai_polys` (OpenCV on GL thread) and `CameraLayer.start()`
- **Location:** `automap.py:311-329, 656-684, 802-809`, `camera_layer.py:566-590`, `live_visuals.py:2495, 2497`
- **Failure scenario:** `op: "pick"` executes `lit_mask`, `_regions` (GaussianBlur + Canny + dilate + morph open + connectedComponentsWithStats), `fit_poly`, and `_picture` synchronously on the GL render thread (plus lazy `import cv2`, ~200–500 ms on first call). `CameraLayer.start()` also opens log files, probes up to 9 TCP ports, and spawns a child process synchronously on the GL thread.
- **Fix:** Cache the `(n, lab, st)` segmentation from `_detect()` and run `pick`/`drop`/`ai_polys` + `_picture` on a background thread; spawn `camera_worker.py` off the GL thread.
- **Confidence:** High

### M2. Photosensitive safety: Low-light 10 fps mask EMA (`tau=0.07 s`) permits 76% step jumps at 5 Hz, and `Signals` slew permits ~45–55% swings at 3.5–6 Hz
- **Location:** `camera_layer.py:22-25, 47, 228-236, 703-706`, `tests/test_camera.py:138-149`
- **Failure scenario:** In low light (10 fps webcam rate measured in `HANDOFF.md:978`, `ddt = 0.10 s`) with `crowd_soft = 0.0`, `tau = 0.07 s` gives EMA step `a = 1 - exp(-0.10/0.07) = 0.76`, allowing 76% instant luminance jumps at 5 Hz in `Bodies reveal` / `Knockout` / `Glow outline` (and stepped 10 Hz updates instead of smooth 60 fps interpolation). Meanwhile, `Signals.step` (`ATTACK_S=0.10`, `RELEASE_S=0.35`) allows ~40–45% peak-to-peak swings at 6 Hz (>50% at 3.5 Hz); routed via MOD to `master`, layer opacity, or discrete selectors (`crowd_mode`, `palette`), this can trigger >3 Hz high-contrast flashes.
- **Fix:** Raise minimum mask `tau` to `>= 0.16 s` and slew `self.mask` toward the target mask every 60 fps frame (`pre_update`) rather than only on `_ingest`; increase `Signals` `ATTACK_S` to `>= 0.18 s` (or add a slew-rate cap `<= 2.0 / s`) and tighten the 3–6 Hz ripple/flicker test ceiling to `< 0.15`.
- **Confidence:** High

### M3. GLSL `camUV` division by zero (`NaN`) and inverted ghost silhouettes across vanishing line (`c.z <= 0`) under keystone
- **Location:** `camera_layer.py:910-912`
- **Failure scenario:** With strong keystone (`key_x = 1.0`, `scale = 0.25`, `pos_y = -0.5`), the homography vanishing line `c.z == 0` crosses the visible screen. `camUV` computes `c.xy / c.z` without checking `c.z > 0`: pixels on `c.z == 0` produce `Inf`/`NaN` (corrupting `p` in `camWarp` before `nanGuard`), and pixels where `c.z < 0` projectively wrap back into `[0, 1]`, projecting an upside-down ghost silhouette in the region beyond the vanishing line.
- **Fix:** Guard `c.z > 1e-4` in `camUV` (returning a sentinel outside `[0, 1]` such as `vec2(-1.0)` when `c.z <= 1e-4`) so `camIn()` evaluates to `0.0` everywhere on or past the vanishing line.
- **Confidence:** High

### M4. Exposure-lock race & silent config drop in `WorkerLink.configure()`, unsynchronized `_exposure`, and locking exposure on the wrong frame
- **Location:** `camera_layer.py:372-374, 395-399, 874-876`, `camera_worker.py:227-234, 420-421`, `automap.py:605-617`
- **Failure scenario:** (a) `WorkerLink.configure()` overwrites `self._cfg = dict(cfg)` instead of merging; a governor quality step or user fps change overwrites a queued `{"lock_exposure": True/False}`, potentially leaving exposure permanently locked. (b) `camera_worker.py:228` (`lock, self._exposure = self._exposure, None`) races with `cam-http` setting `self._exposure = False`. (c) `AutoMap._calibrate` calls `self._exposure(True)` *before* projecting a neutral/white frame (while the live show or black screen is on the wall) and before the async `/config` POST even runs—locking at dark/max exposure clips all calibration patterns to 255. (d) `black` captured at `automap.py:615` is never used by `detect_aruco`.
- **Fix:** Merge pending config keys in `WorkerLink.configure()` (`self._cfg = {**(self._cfg or {}), **cfg}`) and guard `self._exposure` with `self._lock` in `camera_worker.py`; project a neutral/white settle frame and apply `lock_exposure` synchronously (wait for `POST /config` 200) before capturing patterns; either subtract `black` or drop the dead capture at `automap.py:615`.
- **Confidence:** High

### M5. `camera_worker.py` startup errors lost (`"worker exited (1)"`), `restarts` never resets, unhandled `analyse()` exception kills worker, and `do_POST` header hardening
- **Location:** `camera_worker.py:321-329, 354-355, 434-436, 487-496, 522-535`, `camera_layer.py:627-639`
- **Failure scenario:** (a) When `_open()` fails (e.g., webcam busy), `_process_loop` calls `self._stop.set()` and `run()` shuts down immediately before `WorkerLink` polls `/status`, so the user sees `"worker exited (1)"` instead of the real driver error. (b) `CameraLayer.restarts` is never reset on manual OFF/ON or after healthy uptime, permanently disabling crash recovery after 2 glitches in a long show. (c) `_process_loop` has no `try/except` around `self.analyse(...)`, so a single transient MediaPipe/OpenCV error terminates the worker without setting `self.stats["error"]`. (d) `do_POST` (`:487`) parses `Content-Length` without `try/except ValueError` or `n < 0` check (`rfile.read(-1)` hangs HTTP/1.1), and handler `H` has no socket `timeout` on keep-alive connections.
- **Fix:** Reset `self.restarts = 0` on manual `start()` (and after 5 min healthy running); read the last `[camera]` log line on early worker exit when `link.status["error"]` is empty; wrap `self.analyse()` in `try/except` setting `self.stats["error"]` (or skipping transient bad frames); validate `0 <= n <= 4096` and set `timeout = 10` on `H`.
- **Confidence:** High

### M6. `Signals.on_data` cross-pairs left/right wrists (`hand_speed` spikes) and `CameraLayer._vals` lags by 1 frame in `pre_update` / `attach_swarm`
- **Location:** `camera_layer.py:207-220, 696, 704, 736, 824`, `live_visuals.py:2457, 2686, 2717`
- **Failure scenario:** (a) `Signals.on_data` stores `self._wrists` as a positional list (`wr = [b[n] for n in ("l_wrist", "r_wrist") if n in b]`). When `l_wrist` drops below `min_vis=0.5` on frame `t`, `zip([r_wrist], [l_wrist, r_wrist])` computes the distance from the **previous left wrist to the current right wrist**, firing a huge false `hand_speed` spike. (b) `self._vals` is only updated in `cam.uniforms(vals, uni)` (`live_visuals.py:2717`), *after* `pre_update` (`:2457`) and `l_land`/`Swarm.step` (`:2686`), so `crowd_ripples`, `crowd_soft`, and `crowd_swarm` always use the prior frame's values (empty `{}` on frame 1).
- **Fix:** Store `self._wrists` as a dict `{name: pos}` and compute speed per matching landmark key; pass/store `vals` right after `controller.update()` before `extra` passes.
- **Confidence:** High

### M7. Panel UI bugs: CAMERA `<select>` rebuilt every 300 ms, AUTO MAP candidate rows stale when `found.length` unchanged, and overlapping crop resets all 4 edges
- **Location:** `panel.html:3103, 3363-3365`, `camera_layer.py:81-86, 104-114`
- **Failure scenario:** (a) `panel.html:3103` wipes and rebuilds `<select id="camdev">` on every 300 ms `panelstate` event while `C.devices` is set, collapsing the open dropdown while the performer is choosing a USB camera. (b) `panel.html:3364` returns early when `ui.rows.length === f.length`, so re-running `3 DETECT SURFACES` (yielding the same count of different faces) or dropping 1 + picking 1 never refreshes the list. (c) Dragging `Crop left` within `0.1` of `Crop right` triggers `clean_settings()` (`camera_layer.py:84`) to reset **all four** crop edges back to `[0, 0, 1, 1]`.
- **Fix:** Rebuild `ui.dev` only when the serialized device list or `show_ir` changes; key AUTO MAP `rows()` rebuild on a candidate-list signature or revision counter; clamp/reject invalid crop edge moves instead of resetting all 4 values.
- **Confidence:** High

### M8. Test suite gaps & tests that don't verify what they claim
- **Location:** `tests/test_camera.py:47-57, 138-149, 202-204, 260-285, 350-364`, `tests/test_automap.py:134-141, 160-177, 180-210, 225-228`
- **Failure scenario:**
  - `tests/test_camera.py` mocks both `WorkerLink` and `FakeProc` (`wait()` unconditionally sets `returncode = 0`) and passes `ctx=None` to `CameraLayer`, leaving `Worker.start_server()`/`run()`/HTTP auth/`/still`/`/debug.jpg`/`/quit`, `WorkerLink`, GL `_upload()`/`uniforms()`, `attach_herd()`, `attach_swarm()`, and `_serpent()` completely untested.
  - `test_signal_cannot_follow_a_fast_flicker` asserts `< 0.45` at 6 Hz—a 44% full-scale 6 Hz oscillation is not photosensitive-safe.
  - `test_worker_analyse_with_fake_models` rolls an all-zero frame (`np.roll(a, 4, axis=1)`) and never asserts optical flow or `info["motion"]` is non-zero.
  - `test_refine_converges_with_a_biased_calibration` uses a pure translation bias (where local Jacobian is exact, masking oscillation on non-planar scale bias) and asserts `< 1.5` despite its docstring claiming `"within REFINE_TOL"` (`0.8`).
  - `test_patterns_are_photosensitive_safe` only asserts two module constants, not runtime hold duration under render lag, `aruco_board`/`outline_image` max brightness, or blackout/panic cancellation.
  - `test_ai_polygons_are_validated` omits concave 4-gons, `name: None` / `"   "`, and `validate_command({"op": "ai_polys", ...})`.
- **Fix:** Add loopback HTTP tests covering `Worker` + `WorkerLink` (403 auth, `/frame`, `/still`, `/config`, `/quit`, idle exit); test `CameraLayer._serpent`/`attach_herd`/`attach_swarm`; test `_verify`/`refine_step` under scale/non-planar bias and adjacent faces; test `_project()` hold timing with a delayed renderer; test concave 4-gons and malformed `ai_polys` end-to-end through `validate_command -> to_surfaces -> surfaces.validate_cmd`.
- **Confidence:** High

---

## LOW

### L1. `CameraLayer.live` stays stuck at `1.0` on `CAMERA OFF` when `signals.idle()` is True, and `stop()` leaves `port` / `_data_t` stale
- **Location:** `camera_layer.py:592-616, 661-665`
- **Failure scenario:** Turning `CAMERA OFF` while no motion/bodies are detected (`signals.idle() == True`) causes `pre_update()` to return early before decaying `self.live` to `0.0`. On the next `CAMERA ON`, the silhouette snaps in immediately at `live = 1.0` instead of fading over 0.5 s.
- **Fix:** Reset `self.live = 0.0`, `self._data_t = None`, and `self.port = None` in `stop()` (or check `and self.live == 0.0` in the `pre_update` fast-path guard).
- **Confidence:** High

### L2. `crowd_serpent` ignores `l_camera == 0` and uses frame-rate-dependent smoothing
- **Location:** `camera_layer.py:738-741, 756-771`
- **Failure scenario:** Unlike `crowd_beings`, `crowd_swarm`, and `crowd_ripples`, `_serpent()` still overrides serpent #1 when `l_camera` is pulled to `0` (or bypassed). Also `_serp_w += (want - self._serp_w) * 0.06` and the `0.35` track lerp are per-frame constants rather than `dt`-based, so glide speed changes with FPS.
- **Fix:** Gate `want` on `min(vals.get("l_camera", 0.0) * 4.0, 1.0)` and pass `dt` into `_serpent()` for `1 - math.exp(-dt / tau)` smoothing.
- **Confidence:** High

### L3. `camShow` mode 1 (`Glow outline`) bypasses projector `edge` vignette and hue-preserving clip
- **Location:** `camera_layer.py:943`, `live_visuals.py:908-911, 929, 953`
- **Failure scenario:** `camShow` executes after `col *= edge` and `col /= max(1.0, max3(col))`, so additive glow (`o += g`) can light the hard projector rectangle border and hard-clip RGB channels to white.
- **Fix:** Pass `edge` into `camShow` (as `textMix` does at `live_visuals.py:928`), multiply `g` by `edge`, and hue-preserve clip `o /= max(1.0, max3(o))`.
- **Confidence:** High

### L4. `AutoMapOverlay` texture assumes 4-byte row alignment; `panel.html` slider `pointercancel` and `/automap-image` fetch timeout missing
- **Location:** `automap.py:977-979`, `panel.html:3078, 3379`
- **Failure scenario:** (a) `ctx.texture((w, h), 1, dtype="f1")` defaults to `alignment=4`, which corrupts stride or raises if `output[0] % 4 != 0`. (b) Releasing a `.camsl` alignment slider outside the input (or on touch cancel) leaves `dataset.busy = '1'` stuck permanently. (c) `fetch('/automap-image?...')` has no `AbortController` timeout.
- **Fix:** Use `ctx.texture((w, h), 1, dtype="f1", alignment=1)`; add `pointercancel`/`blur` (or `setPointerCapture`) on `.camsl` inputs; add a 5 s `AbortController` timeout on `/automap-image`.
- **Confidence:** High

### L5. `GET /camera-preview` over LAN `0.0.0.0` lets any remote token-holder activate debug mask/skeleton encoding
- **Location:** `remote_control.py:674-679`, `camera_layer.py:878-882`, `camera_worker.py:376-380`
- **Failure scenario:** While `/camera-preview` never sends raw camera pixels (mask + skeleton only), any LAN client with the panel token can poll it and enable `worker.debug_want`, viewing silhouettes/poses even when the operator left preview OFF locally.
- **Fix:** Optionally gate `/camera-preview` to `LOCAL_ADDRS` or an explicit setting if strict venue-Wi-Fi privacy is desired.
- **Confidence:** Medium
