# Visual Critique: Why the Current Output Falls Far Short of the Vision

> Based on direct visual inspection of `test_frames/frame_0090.png`, `frame_0300.png`, `frame_0600.png`, `test_frames/textures_contact.png`, and `test_frames/gallery/contact_sheet.png` (plus `live_visuals.py` & `ai_gen.py`) against the stated inspirations: **TAS Visuals, Johnathan Singer, Android Jones, Alex Grey, Ayahuasca swimming beadwork, Burning Man sculpture mapping, and Above & Beyond at The Gorge**.

---

## Summary Verdict
The **engineering infrastructure** (tests, touch remote, borderless monitor window, audio smoothing, gallery harness) is clean and reliable.
**Artistically, however, the current visuals do not approach the vision at all.** Looking at all 75 frames in `contact_sheet.png`, the entire engine renders **one single flat 2D circular coaster (`Rs = 0.475`) stamped dead-center in a black void**, made of a checkerboard of rounded-square pods (`sin(folds*uu) * cos(PI*v)`) filled with chunky Lite-Brite dots and postage-stamp cutouts of muddy 1-step AI textures. After 90 seconds on a wall, the viewer has seen 100% of what it can do.

---

## 6 Concrete Failure Modes (With Code & Frame Receipts)

### 1. Single Circular Doily Sticker vs. Full-Space Immersion
- **What's happening (`live_visuals.py:120-121`):** `Rs = 0.475` hard-crops the entire visual into a centered circle surrounded by dead black bars. Every single thumbnail in `contact_sheet.png` has the exact same centered circular silhouette and the exact same pink/orange star dot (`cg = exp(-r*r/0.0016)`) sitting dead-center.
- **Why it misses the vision:** Aya swimming beadwork, TAS stages, Singer's Tipper sets, and Gorge productions **transform the entire room/architecture**—wrapping walls, edges, ceilings, and portals—rather than stamping one spinning coaster in the middle of the wall.

### 2. Flat 2D Polar Math Instead of Real 3D Depth & Architecture
- **What's happening (`live_visuals.py:94`):** Everything (`beads`, `ai`, `stairs`) shares one flat 2D sinusoidal checkerboard `lobe(uu, v) = sin(folds*uu) * cos(PI*v)`.
- **Why it misses the vision:** Zero 3D raymarching, zero perspective camera flight through space, zero vaulted sacred architecture (Alex Grey *Net of Being* pillars/arches, DMT cathedral halls), zero multi-plane parallax or occlusion. Even the "luminance relief" (`lines 159-167`) is invisible because it's squeezed into tiny alternating petal squares.

### 3. Chunky "Lite-Brite" Dots Instead of Fine Organic Huichol / Shipibo Tapestry
- **What's happening (`live_visuals.py:126-148`):** `nbh = floor(hs / 0.060)` puts only ~5–8 large, spaced-out circular dots per rounded-square pod. Under twist + feedback (`extra_both_3fold_twist.png`), they smear into chunky plastic Lego cylinders. Similarly, `stairK` (`lines 95, 175-190`) quantizes `sin*cos` into 3 thick boxy squares rather than intricate, unbroken Shipibo *kené* filigree rivers.
- **What's needed:** Thousands of **fine, tightly packed micro-beads** flowing along continuous **curl-noise / domain-warped ribbons, vines, spirals, and bilateral totems** across the whole wall, plus delicate multi-scale filigree (macro structure + intricate micro detail).

### 4. Muddy 1-Step AI Textures — Then Buried in Tiny Postage Stamps
- **What's happening (`test_frames/textures_contact.png` + `live_visuals.py:152-172`):**
  1. `ai_gen.py` uses **1-step SD-Turbo (`steps=1`, `guidance_scale=0.0`) + TAESD at 512×512**. The 8 generated textures are murky, brownish-gray, low-detail mush—nowhere near visionary painting quality.
  2. `live_visuals.py` then chops those textures into tiny alternating `sin*cos` petal windows, keys out most pixels, and overwrites 72% of the color with a flat 5-stop tint (`mix(..., 0.72)`).
- **What's needed:**
  - Since `ai_gen.py` runs **offline/async**, there is **zero reason to cripple image quality with 1-step SD-Turbo!** Use **SD 1.5 (or SDXL-Turbo / SD 2.1) at 15–30 steps with CFG 7–8, a proper VAE (`sd-vae-ft-mse` in fp32), negative prompts (`muddy, blurry, brown, beige, low contrast, gray background`), and high-res 768–1024px** (even 8–15 seconds per offline texture is instant for a pre-show library!), OR pre-rendered **Blender 4.5 Geometry Nodes** crystalline/beadwork loops.
  - Display AI/visionary artwork **full-frame** using Johnathan Singer's actual tricks: **conformal Möbius + Escher Droste warps**, **strong multi-step parallax occlusion relief**, and **lush crossfading**—not cookie-cutter squares!

### 5. Missing True Multi-Pass Bloom & Post-Processing
- **What's happening (`live_visuals.py:217`):** Hard hue-preserving clip (`col /= max(1, max(r,g,b))`) with no separable Gaussian/Kawase bloom pass and no filmic tone-mapping.
- **Why it matters:** Luminous psychedelic visuals (TAS, Alex Grey, Gorge lasers/LEDs) feel *alive and glowing* because bright filaments bleed soft HDR light halos into true `#000000` black via a real **multi-mip Kawase/Gaussian bloom + ACES tone-mapping + subtle radial chromatic dispersion** pass (costs <1 ms at 1280×800 on GTX 1650 Ti!).

### 6. No Real Multi-Surface Projection Mapping & Only One Visual World
- **Mapping (`OUT_FS`):** Only supports **one single 4-corner quad**. Real projection mapping onto a room (doorframes, shelves, beams) or TAS-style layered foam-core stage totems requires **multiple independent polygon slices & masks** (or actually installing/wiring **TouchDesigner 2025 KantanMapper/CamSchnappr** and **Resolume Arena Advanced Output** via Spout).
- **Scene variety:** Modes `beads / ai / both / lines / all` are all variations of one shader. A real show needs **4–5 genuinely distinct 3D/generative worlds** you transition between across a set:
  1. **3D Raymarched Sacred Temple / DMT Hyperspace** (flying *through* infinite folded 3D fractal columns, Kleinian/Mandelbox/gyroid crystalline halls, glowing Alex Grey *Net of Being* arches + volumetric fog).
  2. **Full-Wall Living Beadwork & Shipibo Tapestry** (edge-to-edge curl-noise swimming flow field of dense micro-beads & delicate multi-scale *kené* filigree—no circular coaster crop!).
  3. **Johnathan Singer Liquid Visionary Portal** (full-frame high-res multi-step AI / painting textures warped by **conformal Möbius + Escher Droste** complex-plane math + true parallax 3D relief + orbiting light).
  4. **Gorge / Festival Monumental Light Architecture** (clean sweeping geometric arches, chevrons, horizon beams, and deep emotional 8-bar color arcs—warm sunset amber → deep sapphire → electric cyan/magenta).
  5. **TAS Motion-Key Multi-Slice Stage Mode** (multiple mapped polygon/totem masks with hue-encoded chasing light waves hugging physical room/sculpture edges).

---

## Priority Order to Actually Reach the Vision (for Claude / Fable)
1. **Add a real multi-pass HDR Kawase Bloom + ACES Filmic post-process pass** in `live_visuals.py` (immediately elevates *every* scene from flat MS-Paint neon to luminous light projection).
2. **Replace the single-coaster shader with a multi-scene shader architecture** (keys `F1–F4` or scene selector on the touch remote):
   - Scene 1: **3D Raymarched DMT Cathedral / Hyperbolic Fractal Flight** (real 3D depth & camera motion!).
   - Scene 2: **Full-Frame Curl-Noise Micro-Bead & Shipibo Flow** (dense, fine, edge-to-edge living wall!).
   - Scene 3: **Singer Conformal Möbius / Droste Liquid Relief** (full-frame seamless warping over crisp high-step textures!).
   - Scene 4: **Gorge Architectural Portal / Geometric Horizon** (clean monumental stage energy!).
3. **Upgrade `ai_gen.py` offline quality** (stop throwing away quality for 1-step speed when generating offline textures: 20–30 step SD 1.5 / DreamShaper / visionary model or 4–8 step SD-Turbo/SDXL-Turbo with fp32 VAE, negative prompts, and true-black jewel contrast).
4. **Upgrade mapping beyond 1 quad**: either finish installing **TouchDesigner 2025.33230** (`Downloads\projection-installers\`) + wire `live_visuals.py --spout` → **KantanMapper / CamSchnappr**, OR add multi-polygon slices/masks so the user can actually map doorframes, shelves, and layered foam-core totems.
