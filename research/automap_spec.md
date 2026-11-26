# AUTO-MAP spec (camera-assisted projection mapping), from Gemini's suggestion, adapted to ONE system
Source idea: Gemini 4 Neon suggestion (ArUco/Gray-code calibration → scene understanding → warped rendering → verify & auto-correct). Adaptation rules: NO separate pygame/projector app or second window (one engine, one panel: memory rule). Everything runs inside live_visuals.py (patterns drawn by the engine on the projector output), camera_worker.py (capture + CV), surfaces.py (results become normal editable surfaces), and the MAPPING tab (wizard UI).

Physical prerequisite: the camera must SEE the projected area. The laptop's built-in webcam faces the user, so either turn the laptop / lid toward the wall or use a USB webcam on a fixed mount. If camera or projector moves, re-run calibration (wizard detects large drift and offers re-calibration).

## Stage 1: calibration (camera ↔ projector)
- Engine shows calibration patterns full-frame on the projector (a "calibration overlay" mode, above everything, mapping bypassed): ArUco grid (fast, planar) and Gray-code structured light (dense per-pixel correspondence, handles non-planar sculptures; horizontal + vertical bit planes + inverses for robust thresholding).
- camera_worker captures each pattern in sync (engine signals pattern id + frame; worker waits N frames for exposure; lock camera exposure/white balance during calibration if the driver allows).
- Outputs: camera→projector homography (planar walls), plus a dense camera→projector correspondence map (Gray code) saved as calibration/ files; reprojection error reported in the panel; refuse low-quality results with a reason ("only 5 of 12 markers seen, move the camera to see the whole wall").
- Photosensitive safety: patterns are static, shown at moderate brightness, stepping no faster than ~2 Hz; the wizard warns the audience isn't present.

## Stage 2: scene understanding → surfaces
- Capture a clean photo with the projector showing neutral grey, then black (ambient), giving a projector-lit difference image.
- Classical first (offline, fast, private): edge/contour + planar-face detection; the user clicks a face on the camera image in the panel → region grow / quad fit → snaps to edges; candidate quads/polygons listed with confidence.
- Optional AI assist (only when ONLINE is on, via the hardened Online plugin; Gemini API image input preferred; never send images unless the user presses the button; show the image being sent): "find the front faces of the three boxes" → JSON polygons in camera pixels; validated (convex/simple, inside frame) before use.
- Convert camera-pixel polygons → projector pixels via the calibration → create/update surfaces (named, with masks for non-quads).

## Stage 3: verify & auto-correct (closed loop, classical; LLM optional)
- Project bright thin outlines of each surface, capture, measure distance between projected outline edges and the physical edges (edge map), nudge corners to minimise it (a few iterations, sub-pixel), stop when error < threshold. Show before/after error per surface; one-click accept / undo (surfaces undo stack).

## UI
MAPPING tab: AUTO MAP wizard (1 Camera check: live small preview + "camera sees the whole projection?" 2 Calibrate 3 Detect surfaces (click-to-pick + optional AI) 4 Verify & refine 5 Done), all steps undoable; manual corner editing stays available afterwards.

## Tests
Synthetic: render known patterns, warp with a known homography + noise into fake camera frames → recover H within tolerance; Gray-code decode on synthetic non-planar warp; quad fit on synthetic boxes; refinement converges on synthetic misalignment.
