# Improvement backlog: what live projection-mapping tests taught us

**Bottom line:** the camera-driven look loop works. Gemini Neon, looking at the real wall through the webcam, took a
five-surface mapped room from 3.2/10 to 9.0/10 in six rounds. To get there it had to work around several engine
problems, and those are now the most valuable fixes. In priority order:

1. **Per-surface skins can render dim or black.** The shared layer renderer reuses the main show's layer at the main
   show's strength, and the kené and bead studies share one renderer.
2. **Dark materials need more gain.** Dark cloth, wood and foliage need much more light than white walls, and the
   current controls make that awkward to balance.
3. **AI-detected surfaces can be wrong.** They can land on the wrong object or carry the wrong label, and there is no
   quick way to audit them.

This document is the working backlog. It comes from two sources: real-wall tests on 2026-11-26 (two AUTO MAP sessions
with the built-in webcam, and three runs of `tools/wall_look_loop.py`), and the loop's own notebook. That notebook is
kept locally in `.wall_loop/NOTES.md` and is not published.

## 1. Engine bugs (fix first)

| # | Problem | Evidence | Fix direction |
|---|---|---|---|
| E1 | A layer used as a per-surface skin renders at the main show's amount. The 3D deck at `l_land=0.01` made a near-black skin. | Setting `l_land=0` lifted cloth brightness from a mean of 14.6 to 34.5 | Render skins at full strength. Reuse the main pass only when its amount is at least 0.85, or render unscaled. |
| E2 | Kené / kené 4-channel skins render black or flat grey while the bead study uses the same renderer. | Cloth went from black to a crisp image after switching to `lines4` | Give LayerHub its own KenePass instance, or force `kene_draw=1` for skins. |
| E3 | Kené 4-channel adds a grey "floor" under other layers. | Visible haze on the camera | Check alpha and premultiplication for the 4-channel sources. |
| E4 | Surface edge glow looks like a hard white wireframe, even at 0.05–0.1, and doubles up at shared seams. | Seen in every round that used glow | Make glow softer and falloff-based, and suppress it on edges shared by two surfaces. |
| E5 | Fixed: Windows display sleep turned the projector off mid-show. | Projector showed "No source detected" | `keep_display_awake()` in live_visuals (done). |
| E6 | Fixed: the AUTO MAP live view dropped pictures after 10 idle minutes, and auto exposure clipped the projection to white. | Loop stopped at round 3 | The live view keeps its pictures and locks exposure on a dim frame (done). |

## 2. AUTO MAP accuracy

| # | Problem | Fix direction |
|---|---|---|
| A1 | AI outlines can land on the wrong object. A "plant pot" outline covered the bare wall and ran past the projector frame. | Reach checks are done. Next, check each AI polygon against the dense calibration and the lit mask before CREATE. |
| A2 | Labels can be swapped between surfaces: 3 of 5 were mislabelled in one session. | Add a flash-ID audit: light each surface in turn, have the camera confirm which object lit up, then relabel. |
| A3 | Surface outlines include background. The plant surface took in bare wall behind the leaves. | Tighten it with the thin-part mask path. Optionally subtract other surfaces' areas. |
| A4 | The camera sees only about 72% of the projection from the laptop position. | A coverage alert in the panel (marker count plus which edges are cut). Second-pose calibration. |
| A5 | Thin-part (grass blade) masks are untested on a real plant. | Run DETECT THIN PARTS on the real plant and tune the thresholds. |
| A6 | The AI only sees greyscale stills. | Keep a colour still for AI detect, AI verify and the look loop. |
| A7 | AI verify only proposes corner nudges. | Add a per-surface pass/fail verdict with a description. |

## 3. Controls the look director (and the performer) asked for

- **Per-surface contrast / gamma,** or a brightness ceiling above 2.0. Dark cloth and foliage currently need duplicate
  additive layers to match white drywall.
- **Per-surface brightness gradient** (vignette or falloff compensation) for angled surfaces that are lit unevenly.
- **More of the loop's settings table:** `fractal_mode`, `land_a`, and each surface's `src` / `rect`.
- **A pre-round health check in the loop:** projector connected, output window on the projector, camera exposure locked.

## 4. What worked (keep doing this)

- **Balance surfaces by material, not by number.**
  - Under locked exposure: white drywall about 0.65–0.75, wood panels about 0.8–2.0, dark cloth and foliage 1.0–2.0
    plus a bold source.
  - Never judge brightness under auto exposure.
- **Start each surface with one clean, high-contrast layer** (for example `lines`). Add texture only after brightness
  and placement are right.
- **Doubling the same layer in `add` blend** (same scale, scroll and direction) boosts contrast past the brightness cap
  for free, because LayerHub renders each source once per frame.
- **Apollonian circles** (`fractal_mode=3`, `fractal_glow≈0.45`) read sharply on white walls. Volumetric fractals at
  high glow turn into a grey fog there.
- **Bead constellations plus sweeping lines** read well on wood panels. A woven diamond lattice reads well on dark cloth.
- **Keep unique skin sources at 8 or fewer** (LayerHub limit). Match each surface's crop to the real surface's shape.

## 5. Running the loop well

- **Start state:** run with the show live, the camera on, and the surfaces mapped: `entheo loop 6` (personal
  launcher), or `python -u tools/wall_look_loop.py --rounds 6`.
- **Cost:** each round costs roughly 30–80k agy tokens, mostly agy's own agent overhead. For tight loops, add a direct
  Gemini API route (about 5–6k tokens per round).
- **Notebook:** the loop's notebook is fed back each round. Turn its BUGS and FEATURE_REQUESTS into rows here.
