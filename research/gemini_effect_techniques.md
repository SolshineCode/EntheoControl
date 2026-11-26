# Realtime GLSL 330 Technique Research (Effects 1–12)

Target platform: **NVIDIA GeForce GTX 1650 Ti Mobile (4 GB GDDR6)** — Turing TU117, 1024 CUDA cores, ~3.04 TFLOPS FP32, 48 GPixel/s, 95 GTexel/s, 192 GB/s memory bandwidth — running **OpenGL 3.3 / GLSL 330** via `moderngl` (fragment shaders, ping-pong FBOs, MRT, instancing, and **Transform Feedback**; no GLSL 430 compute shaders or SSBOs) at **1280×800** (~1.024 Mpix), **60 fps** (16.67 ms budget; target **≤ 6 ms/scene** so A/B crossfades + bloom + photosensitive limiter stay comfortably under 12 ms with laptop thermal headroom).

> **Licensing convention used below:** Permissive sources (MIT, public-domain math/papers) are preferred for direct adaptation. Any default-licensed Shadertoy shader (**CC BY-NC-SA 3.0**), **Mixbox (CC BY-NC 4.0)**, or **The Book of Shaders (All Rights Reserved)** is explicitly flagged as **do not copy, idea only**.

---

## Summary Table

| Effect | Technique | Est. ms at 1280×800 on GTX 1650 | Difficulty |
| :--- | :--- | :--- | :--- |
| **1. Flower-of-Life Dome** | Analytic ray–sphere intersection + tangent/stereographic hex-lattice circle SDFs + `sdVesica` portal + `fwidth` AA | **0.6 – 1.1 ms** | Low–Medium |
| **2. Infinite Tiled Corridor** | Analytic ray–slab box intersection + Hankin Polygons-in-Contact / dihedral symmetry-fold tiling library with continuous angle morphing | **0.9 – 1.8 ms** | Medium |
| **3. Inside-Out Torus Worlds** | Inverted-SDF raymarching (torus / horn-torus hourglass / twin figure-8 blend) + toroidal `(u, v)` domain repetition + half-res radial god-rays | **2.4 – 4.8 ms** | Medium–High |
| **4. Dark Gothic Set** | KIFS space-folding SDF + MatCap/analytic Fresnel specular + domain-warped vertical Rorschach SDF relief | **2.5 – 4.5 ms** | Medium–High |
| **5. Void Webs** | Exact two-pass Voronoi bisector-edge SDF (`voronoilines`) + domain-warped web glow + bipolar/Möbius twin-hole warp | **1.4 – 2.5 ms** | Medium |
| **6. Hyperspace Stained-Glass Vortex** | Off-centre corkscrew log-polar coordinate warp + radial-stretched Voronoi glass shards + backlit radial streak accumulator | **1.0 – 1.8 ms** | Low–Medium |
| **7. White Portal (Transition)** | Analytic radial SDF iris + Dual Kawase bloom ramp + temporal luminance slew-rate / WCAG 2.3.1 photosensitive limiter | **0.5 – 0.9 ms** | Low |
| **8. Golden Filigree Entity** | Iterated symmetric space-folding / branching segment SDF (`sdCapsule` + fold matrices) + additive volumetric glow accumulation | **2.2 – 4.2 ms** | High |
| **9. Holographic Stained-Glass Face** | (A) Symmetric sculpted 3D face SDF (`smin` primitives) and/or (B) Edge-seeded Jump Flooding Algorithm (JFA) Voronoi stained-glass image filter | **1.5 – 3.8 ms** | High |
| **10. Breathing Honeycomb-Bead Chamber** | Analytic corridor box + raymarched superquadric/`smin` morphing centrepiece (`sdRoundedX` ↔ sphere ↔ 4-lobe clover) + Spherical Fibonacci / triplanar hex bead SDF | **2.6 – 4.5 ms** | Medium–High |
| **11. Fluid Silk / Ink Filaments** | Transform Feedback GPU particles (500k–1.5M) advected by 3D analytic Curl Noise + line-segment / ribbon rendering into ping-pong trail FBO + multiplicative paper / Kubelka-Munk-lite composite | **2.8 – 6.2 ms** | High |
| **12. Choreographed Show Sequencer** | CPU/Python beat-synced timeline driving uniform morph curves + dual-FBO A/B crossfade & unified-SDF morph pass | **0.2 – 0.4 ms** (+ A+B cost during blends) | Medium |

*(Shared post-processing budget: 4-level Dual Kawase Bloom ~0.4–0.6 ms + photosensitive brightness/slew limiter ~0.15–0.25 ms.)*

---

## Detailed Technique Breakdown (Effects 1–12)

### 1. Flower-of-Life Dome
- **Technique:** Analytic Ray–Sphere Intersection + Spherical Hex-Lattice Circle SDFs + Analytic Vesica Halo (`sdVesica`).
- **How it works:** Rather than raymarching a sphere, intersect the camera ray analytically with an inverted dome sphere (`dot(ro, rd)` quadratic, 1 step) and project the hit point onto a seamless dome chart (stereographic projection from the nadir for the zenith dome, or icosahedral/octahedral folding for pole-free full-sphere coverage). Evaluate a 2D hexagonal lattice (`hexCoords`) and compute the exact minimum unsigned distance `abs(length(p - center) - r)` across the local 7-circle neighborhood (center + 6 hex neighbors) to form overlapping six-petal Flower-of-Life stars. In the zenith/center region, blend an opening pearl-almond halo using Inigo Quilez’s exact `sdVesica(p, r, d)` SDF, animating `d` to open the oval. Shade razor-thin neon filaments with inverse-distance glow plus `fwidth(d)`-clamped smoothstep AA, and drive palette transitions (magenta/lime/cyan → royal purple/antique gold) via cosine palettes.
- **Cost estimate:** **0.6 – 1.1 ms** at 1280×800 (zero raymarch loop; ~7–9 distance evaluations per fragment, heavily ALU-light and ~10% of the GTX 1650 Ti's 3.04 TFLOPS capacity).
- **Pitfalls:**
  - *Thin-line aliasing / moiré near horizon:* Screen-space derivatives (`fwidth`) spike at grazing angles on a dome; clamp line width to `max(width, 1.2 * fwidth(d))` and attenuate filament intensity proportionally (`width / effectiveWidth`) to conserve energy and eliminate shimmer.
  - *Polar pinching:* Standard `(theta, phi)` equirectangular UVs crush circles at the zenith; use stereographic projection (`p = hit.xz / (1.0 + hit.y)`) or tangent-plane hex tiling so circles remain strictly conformal (circles stay true circles on the sphere!).
  - *Seam discontinuity in `fwidth`:* Never run `fwidth` across a `fract()` or `atan()` wrap; compute `fwidth` on the continuous SDF value `d` or pass analytic Jacobian scale.
- **References & licenses:**
  - Inigo Quilez, *2D Distance Functions (`sdVesica`)* — https://iquilezles.org/articles/distfunctions2d/ (**MIT License** for code snippets).
  - Inigo Quilez, *Filterable Procedurals / Anti-Aliasing* — https://iquilezles.org/articles/filterableprocedurals/ (**MIT License**).
  - Patricio Gonzalez Vivo & Jen Lowe, *The Book of Shaders (Shapes & Patterns)* — https://thebookofshaders.com/09/ (**All Rights Reserved — educational reference only, do not copy code**).

---

### 2. Infinite Tiled Corridor (Morphing Islamic / Girih Tessellations)
- **Technique:** Analytic Ray–Corridor Intersection + Hankin’s Polygons-in-Contact & Dihedral Fold Tiling Library with Continuous Morphing.
- **How it works:** Intersect the camera ray analytically with 4 infinite planes (floor, ceiling, left/right walls; 4 divisions, zero raymarching) to obtain exact planar `(u, v)` coordinates scrolling along `z + speed * time`, with a pinpoint sun halo at the vanishing point. On the corridor planes, implement a unified symmetry-fold tiling evaluator based on **Hankin’s Polygons-in-Contact (PIC)** method (formalized by Craig Kaplan) and Wythoff/dihedral reflection folding: fold tile coordinates into a fundamental triangle/kaleidoscope wedge using plane reflections (`p -= 2.0 * min(0.0, dot(p, n)) * n`), then compute exact line-segment SDFs (`sdSegment`) for rays cast from edge midpoints at contact angle $\theta$. Morphing seamlessly across the library (5-point Persian stars, nested pentagons/lotus, 8- and 12-fold Islamic girih rosettes, concentric hexagons) is done by smoothly interpolating the PIC contact angle $\theta$, ray-intersection bevels, and blending two tile SDFs via `mix(dA, dB, t)` or smooth-min during symmetry-order changes (e.g., 8-fold ↔ 12-fold ↔ 6-fold).
- **Cost estimate:** **0.9 – 1.8 ms** at 1280×800 (analytic plane intersection is < 0.15 ms; folding + 4–8 segment SDF evaluations per tile — or 2× during cross-order morphs — takes ~0.8–1.5 ms).
- **Pitfalls:**
  - *Severe horizon moiré/flickering:* Infinite corridors compress thousands of tiles into a single pixel near the vanishing point. Apply distance-based band-limiting (analytical box-filtered line coverage or smooth fog/sun-bloom fade as `fwidth(uv)` exceeds tile frequency) so high-frequency patterns fade cleanly to their average color before aliasing.
  - *Non-Euclidean / 5-fold tiling mismatch:* Regular pentagons/5-point stars cannot tile the Euclidean plane edge-to-edge alone; use Cairo/pentagonal dual grids, decagonal girih super-tiles (10-fold rosette + bowtie/rhombus), or concentric/nested radial-grid cells for the 5-point Persian carpet & lotus motifs.
  - *Precision drift on long runs:* Wrap corridor `z` travel modulo the tiling super-period (`mod(z, PERIOD)`) on the CPU before uploading uniforms so `float32` precision never degrades during a 60-minute show.
- **References & licenses:**
  - Craig S. Kaplan & David H. Salesin, *Islamic Star Patterns in Absolute Geometry* (ACM TOG 2004) — https://cs.uwaterloo.ca/~csk/other/starpatterns/ (**Academic paper — math/algorithm freely implementable**).
  - mattz, *Wythoff Explorer* (Shadertoy `Md3yRB`) — https://www.shadertoy.com/view/Md3yRB (**Default Shadertoy CC BY-NC-SA 3.0 — do not copy, idea only**).
  - Inigo Quilez, *2D Distance Functions (`sdSegment`, `sdStar`, `sdHexagram`)* — https://iquilezles.org/articles/distfunctions2d/ (**MIT License**).

---

### 3. Inside-Out Torus Worlds (Torus, Twin Tunnels, Electric Hourglass Chapel)
- **Technique:** Inverted-SDF Raymarching (Torus / Horn Torus / Smooth-Blended Figure-8) + Toroidal `(u, v)` Surface Mapping + Half-Res Screen-Space God Rays.
- **How it works:** Place the camera inside a hollow surface defined by an inverted SDF: `d = -sdTorus(p, vec2(R, r))` for the standard torus, `R ≈ r` (horn torus) to form the breathing central **hourglass pillar** where floor flows up the waist and blooms across the ceiling, or `-smin(sdTorus(p - offset), sdTorus(p + offset), k)` for the glossy **figure-eight / twin receding tunnels**. Sphere-trace in 48–72 steps with relaxed over-stepping (1.0× safe, Lipschitz = 1) and analytical/tetrahedral normals. Map surface artwork (stained-glass flowers, floral hexes, electric cyan/magenta lotus wireframes) directly in toroidal angles `u = atan(p.z, p.x)`, `v = atan(p.y, length(p.xz) - R)` animated along `v` to create continuous breathing flow through the waist. Add glossy reflections via a fast analytic environment/Fresnel term (or 1 short reflection ray of 16–24 steps) and optional sweeping diagonal god-rays via a half-resolution (640×400) 24-tap radial scattering pass (GPU Gems 3 Ch. 13).
- **Cost estimate:** **2.4 – 4.8 ms** at 1280×800 (~1.8–3.2 ms for 60-step primary raymarch + procedural stained-glass/hex shading; +0.6–1.0 ms for single-bounce glossy mirror reflection; +0.4–0.6 ms for half-res radial god-rays).
- **Pitfalls:**
  - *`atan` derivative seam (`±π` jump):* Computing `fwidth(u)` across the `±π` wrap causes a bright 1-pixel glitch line along the torus meridian. Fix by passing `vec2(cos(u), sin(u))` or taking `min(fwidth(u), fwidth(fract(u + 0.5)))`.
  - *Horn-torus singularity at the waist:* When `R == r`, the inner hole pinches to a point and `length(p.xz) == 0` breaks `atan`; keep `R >= r + 0.08` (a narrow glowing waist pillar) so surface flow stays smooth and normals remain well-defined.
  - *Grazing-angle step exhaustion:* Looking tangentially down twin tunnels burns steps; increase hit epsilon proportionally with distance (`eps = 0.0008 * t`) to keep step counts bounded and prevent distance-shimmer.
- **References & licenses:**
  - Inigo Quilez, *3D Distance Functions (`sdTorus` & smooth blends)* — https://iquilezles.org/articles/distfunctions/ (**MIT License**).
  - Kenny Mitchell, *GPU Gems 3, Ch. 13: Volumetric Light Scattering as a Post-Process* — https://developer.nvidia.com/gpugems/gpugems3/gpugems3_ch13.html (**NVIDIA GPU Gems open online reference; algorithm freely implementable**).
  - fb39ca4, *Torus Interior* (Shadertoy `MsX3Wj`) — https://www.shadertoy.com/view/MsX3Wj (**Default Shadertoy CC BY-NC-SA 3.0 — do not copy, idea only**).

---

### 4. Dark Gothic Set (Star-Blades, Bioluminescent Ribs, Liquid-Metal Totem, Ivory Engraving)
- **Technique:** Kaleidoscopic IFS (KIFS) Folding SDF + Analytic Fresnel/MatCap Obsidian Shading + Symmetric Domain-Warped Relief Engraving.
- **How it works:** Unify all four dark-gothic looks under a single symmetry-folded SDF raymarcher (56–80 steps). For **razor star-blades & gothic totems**, apply 4–6 iterations of Knighty’s **KIFS** (mirror-fold `p.x = abs(p.x)`, tetrahedral/octahedral plane folds, scale + offset) to carve crisp architectural ribs and mirrored obsidian blades around a central sphere or vertical totem spine. For **bioluminescent wireframe ribs**, intersect invisible sphere shells (`abs(sdSphere) - thickness`) with polar meridian/latitude SDF grids so only glowing amber/violet structural ribs render on near-black. For **liquid-metal obsidian**, shade with sharp Schlick Fresnel, procedural studio-strip MatCap reflections (zero extra reflection rays), and thin neon-cyan/green curvature highlights from `dot(N_a, N_b)`. For **ivory/charcoal Rorschach engraving**, raymarch (or bump-map a curved wall with) vertically scrolling mirrored (`abs(p.x)`) domain-warped fBM + KIFS masks shaded with high-contrast directional AO and hatching.
- **Cost estimate:** **2.5 – 4.5 ms** at 1280×800 (4–6 KIFS folds per step × 64 steps is ALU-bound; MatCap + curvature shading avoids secondary ray bounces, staying well under 5 ms on GTX 1650 Ti).
- **Pitfalls:**
  - *Under-stepping / over-stepping in warped SDFs:* Domain warping (for liquid-metal / Rorschach) breaks the Lipschitz-1 property, causing slicing artifacts or banding; multiply the returned distance by `0.5–0.7` (step relaxation) and cap warp amplitude.
  - *Aliasing on razor-thin chrome speculars:* Mirror-smooth obsidian with tiny roughness (`roughness < 0.04`) sparkles wildly on high-curvature KIFS edges; apply specular anti-aliasing (clamping roughness upward by `length(fwidth(N))`).
  - *Crushed blacks on projectors:* Pure black (`0.0`) next to blazing highlights loses gothic detail; keep dark obsidian albedo in `0.02–0.05` with subtle cool-violet/blood-red rim Fresnel so silhouettes read cleanly on a projection wall.
- **References & licenses:**
  - Mikael Hvidtfeldt Christensen (Syntopia), *Distance Estimated 3D Fractals (III): Folding Space (KIFS)* — https://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-iii-folding-space/ (**Public educational blog post; Knighty's KIFS math is public domain**).
  - Inigo Quilez, *Domain Warping* — https://iquilezles.org/articles/warp/ (**MIT License**).
  - Inigo Quilez, *3D Distance Functions* — https://iquilezles.org/articles/distfunctions/ (**MIT License**).

---

### 5. Void Webs (Stretchy Neuron Lace & Ember Honeycomb Twin Eyes)
- **Technique:** Exact Two-Pass Voronoi Bisector-Edge SDF (`voronoilines`) + Domain-Warped Filament Glow + Bipolar Twin-Hole Conformal Warp.
- **How it works:** Implement Inigo Quilez’s exact **Voronoi border distance** algorithm (Pass 1: 3×3 cell search for closest seed `mr`; Pass 2: 5×5 search projecting onto the exact perpendicular bisector `dot(0.5*(mr + r), normalize(r - mr))`), which returns a true Euclidean SDF to every web strand. For **stretchy glowing neuron/lace webs**, animate seed points smoothly along Lissajous paths, warp input UVs with low-frequency sine/curl displacement, and render multiple parallax depth layers (2–3 scales) with bright acid-yellow/magenta/white inverse-distance neon glow (`exp(-k * d)` + core line). For the **ember honeycomb with twin dark sapphire eyes**, reduce Voronoi jitter to `0.0` (exact hexagonal lattice) and apply a **bipolar / two-pole conformal warp** (`log((z - a) / (z + a))` in complex coordinates) or two radial sink SDFs that pull honeycomb rings around two dark ringed sapphire voids.
- **Cost estimate:** **1.4 – 2.5 ms** at 1280×800 (one 2-pass exact Voronoi evaluation is 9 + 25 = 34 cheap hash/dot iterations, ~0.5–0.7 ms; 3 parallax layers take ~1.8–2.2 ms; pure hex lattice + twin-hole warp takes < 0.8 ms).
- **Pitfalls:**
  - *Using naive `F2 - F1` instead of exact bisector distance:* `F2 - F1` pinches and thickens unevenly at oblique cell junctions, ruining the razor-thin lace look; always use the 2-pass perpendicular-bisector formula (or shrink Pass 2 to 3×3 when jitter ≤ 0.5 for a 2× speedup with rare corner error).
  - *Hash artifacts from `sin()` hashes on NVIDIA vs Intel:* Classic `fract(sin(dot(...))*43758.5453)` can exhibit precision banding at large offsets; use an integer/bit-mixing hash (e.g., Dave Hoskins' MIT `hash22` without `sin()`).
  - *Line flicker at thin neon cores:* Ensure filament core width is `>= 1.0` pixel via `fwidth(d)` and feed HDR glow values into the shared Dual Kawase bloom rather than hard-clamping at `1.0`.
- **References & licenses:**
  - Inigo Quilez, *Voronoi Edges / Lines* — https://iquilezles.org/articles/voronoilines/ (**MIT License**).
  - Inigo Quilez, *Smooth Minimum* (for organic webbed junctions via `smin`) — https://iquilezles.org/articles/smin/ (**MIT License**).
  - Patricio Gonzalez Vivo & Jen Lowe, *The Book of Shaders, Ch. 12: Cellular Noise* — https://thebookofshaders.com/12/ (**All Rights Reserved — educational reference only, do not copy code**).

---

### 6. Hyperspace Stained-Glass Vortex
- **Technique:** Off-Centre Corkscrew Log-Polar Mapping + Speed-Stretched Voronoi Petal Shards + Backlit Radial Streak Pass.
- **How it works:** Compute polar coordinates around a slowly orbiting off-centre vortex origin `c(t)`: `r = length(p - c)`, `theta = atan(p.y - c.y, p.x - c.x)`, then map to corkscrew log-polar space `uv = vec2(theta * N / TAU + twist * log(r), log(r) * radialScale - speed * t)`. Evaluate 2–3 concentric depth layers of angularly tiled **Voronoi / petal SDF shards** (`voronoilines` wrapped periodically in `theta` modulo `N`) with heavy black lead lines (`smoothstep` on exact edge SDF) and jewel-like backlit transmission (`pow(1.0 - r, 2.0)` sun-behind-glass falloff + animated HSV/cosine hue rotation across white/yellow/pink → rainbow → molten ruby/violet). As hyperspace velocity increases, anisotropically stretch the radial frequency (`radialScale`) at outer radii and optionally run a lightweight 12–16 tap radial motion-blur / streak pass toward `c(t)` for crisp warp-speed light streaks.
- **Cost estimate:** **1.0 – 1.8 ms** at 1280×800 (pure 2D analytic transform + 2 layers of periodic Voronoi shards ~0.9–1.3 ms; optional 12-tap radial streak post-pass ~0.3–0.5 ms).
- **Pitfalls:**
  - *Angular seam at `atan` wrap (`-π` to `+π`):* Wrap Voronoi integer cell X-indices modulo `N` (`mod(cell.x, N)`) so shards stitch seamlessly across the branch cut, and compute `fwidth` using `min(fwidth(theta), fwidth(fract(theta/TAU + 0.5)*TAU))`.
  - *Singularity / strobing at the vortex center (`r → 0`):* As `r → 0`, `log(r) → -∞` and shards shrink below 1 pixel, causing violent white/black flicker right at the focal point. Fade shard contrast smoothly to the dark corkscrew hole (or backlit core glow) for `r < r_min` and attenuate lead lines when `fwidth(uv.y)` grows large.
  - *Motion sickness / strobing at high speed:* Keep outer radial streaks smooth via analytical streak length scaling rather than discrete frame-to-frame particle jumps.
- **References & licenses:**
  - Inigo Quilez, *Voronoi Edges* — https://iquilezles.org/articles/voronoilines/ (**MIT License**).
  - Shadertoy *Hyperspace / Warp Tunnel* (`4tjSDt`) — https://www.shadertoy.com/view/4tjSDt (**Default Shadertoy CC BY-NC-SA 3.0 — do not copy, idea only**).
  - Kenny Mitchell, *GPU Gems 3, Ch. 13: Volumetric Light Scattering* (radial streak sampling kernel) — https://developer.nvidia.com/gpugems/gpugems3/gpugems3_ch13.html (**Open reference; math freely implementable**).

---

### 7. White Portal (Photosensitive-Safe Tunnel Exit Transition)
- **Technique:** Analytic Radial Iris SDF + Dual Kawase Bloom Ramp + Temporal Luminance Slew-Rate & WCAG 2.3.1 Limiter.
- **How it works:** Render the portal as an analytic glowing disc/tunnel exit SDF (`d = length(p - center) - radius(t)`) composited over the outgoing scene, where `radius(t)` swells smoothly over **≥ 1.0 s** using a C² S-curve (`smootherstep`) so luminance acceleration is zero at start and end. Drive a 4-level **Dual Kawase Bloom** pyramid (Marius Bjørge, SIGGRAPH 2015: 5-tap downsample, 9-tap upsample) so the expanding white disc bleeds organically across tunnel walls before filling the frame, then dissolve cleanly into the incoming scene. Enforce a dedicated final-stage **photosensitive safety pass** (or CPU-tracked exposure uniform using a 1×1 downsampled mip of previous frame luminance) that hard-clamps full-frame luminance slew rate below WCAG 2.3.1 / ITU-R BT.1702 thresholds (never > 3 flashes/s, smooth ≥ 1 s rise to peak white, capped max wall brightness for projector comfort).
- **Cost estimate:** **0.5 – 0.9 ms** at 1280×800 (~0.15 ms for portal composite + ~0.4–0.55 ms for 4-level Dual Kawase bloom + ~0.15 ms for luminance limiter).
- **Pitfalls:**
  - *Band/posterization on smooth white gradients:* Expanding soft white glows in 8-bit `RGBA8` FBOs show ugly concentric rings on dark walls; render intermediate HDR/bloom buffers in `RGBA16F` and apply triangular-PDF blue-noise dithering (`±0.5/255`) on final output.
  - *Photosensitive seizure hazard:* Never trigger the white portal on an instantaneous beat spike or snap-cut back to black; enforce `duration >= 1.0 s` in the sequencer and ramp both rise and decay smoothly.
  - *Bloom fireflies / flickering:* Clamp pre-bloom HDR values (soft knee tone curve) before downsampling so 1-pixel specular sparks do not pulse the bloom pyramid.
- **References & licenses:**
  - Marius Bjørge (ARM), *Bandwidth-Efficient Rendering (Dual Kawase Filter)*, SIGGRAPH 2015 Mobile Graphics Course — summarized & GLSL reference at https://learnopengl.com/Advanced-Lighting/Bloom (**LearnOpenGL CC BY-NC 4.0 — do not copy verbatim; Dual Kawase algorithm from SIGGRAPH 2015 slides is freely implementable**).
  - W3C WAI, *Understanding Success Criterion 2.3.1: Three Flashes or Below Threshold* — https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html (**W3C Public Standard**).
  - ITU-R, *Recommendation BT.1702-3: Guidance for the reduction of photosensitive epileptic seizures caused by television* — https://www.itu.int/rec/R-REC-BT.1702/en (**Public ITU-R Standard**).

---

### 8. Golden Filigree Entity (Branching Glow Figure)
- **Technique:** Iterated Symmetric Space-Folding / Branching Capsule SDF (Pseudo-L-System in Fragment Shader) + Volumetric Glow Accumulation.
- **How it works:** True recursive string-rewriting L-systems cannot run natively per-pixel in GLSL 330, so implement a **branching KIFS / space-folding SDF** directly inside a single raymarching loop. Start by mirroring `p.x = abs(p.x)` for bilateral humanoid/mandala symmetry, anchor a torso/head/wing armature with a few `sdCapsule` primitives and a faceted blazing white-pink octahedron jewel (`sdOctahedron`) at the heart and third eye. For the nervous-system / chandelier filigree wings, run a fixed 6–8 iteration folding loop: in each step, evaluate `sdCapsule` for the current branch segment, translate to the branch tip, mirror-fold (`p.x = abs(p.x)`), rotate by branch angle $\alpha$ (modulated subtly by breath/audio), and scale by `1.4–1.6` while tracking total scale for a valid distance bound (`d / scale`). Instead of solid opaque shading alone, accumulate soft volumetric glow along the ray (`glow += exp(-k * abs(d)) * stepSize`) so filaments shimmer like golden-orange spun light and dissolve seamlessly into the surrounding architecture via a 3D noise/height threshold.
- **Cost estimate:** **2.2 – 4.2 ms** at 1280×800 (48–64 raymarch steps × 6–8 branch folds per step; cheap bounding-sphere / bounding-box early-out skips empty background pixels, cutting average pixel cost by ~50–65%).
- **Pitfalls:**
  - *Exponentially thin branches disappearing / aliasing:* After 7–8 scale folds, terminal twigs drop below 1 pixel and flicker; clamp minimum branch radius to `max(r, 0.0015 * rayDist)` (cone-tracing radius floor) and rely on soft volumetric glow accumulation rather than hard surface hits for the finest tips.
  - *Over-bright white-out in additive glow accumulation:* Fixed step counts near grazing branches can blow out to flat white; normalize glow with a soft Reinhardt/ACES shoulder (`col = col / (1.0 + col)`) and clamp step-contribution spikes when `d` is very small.
  - *Dissolve popping:* Morphing the entity out of the architecture looks abrupt with a hard alpha fade; smoothly blend the entity SDF with the background architecture SDF via `smin(dArch, dEntity, k)` or drive the branch fold scale/iteration fade continuously.
- **References & licenses:**
  - Mikael Hvidtfeldt Christensen (Syntopia), *Distance Estimated 3D Fractals (III): Folding Space* — https://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-iii-folding-space/ (**Public educational post; folding math freely implementable**).
  - Inigo Quilez, *3D Distance Functions (`sdCapsule`, `sdOctahedron`)* & *Primitives Template* (`Xds3zM`) — https://iquilezles.org/articles/distfunctions/ and https://www.shadertoy.com/view/Xds3zM (**MIT License**).
  - Shadertoy KIFS Branching Fractal Example (`Wfs3W2`) — https://www.shadertoy.com/view/Wfs3W2 (**Default Shadertoy CC BY-NC-SA 3.0 — do not copy, idea only**).

---

### 9. Holographic Stained-Glass Face (Procedural SDF + Image Faceting Filter)
- **Technique:** Dual-Mode: **(A) Procedural Symmetric Face SDF** (`smin`-blended ellipsoids/capsules + lattice contour overlay) and/or **(B) Edge-Aware Jump Flooding Algorithm (JFA) Voronoi Stained-Glass Image Filter**.
- **How it works:**
  - *Mode A (Procedural Holographic Face):* Mirror `p.x = abs(p.x)` and sculpt a serene symmetric mask/head using ~12–16 analytic SDF primitives (`sdEllipsoid` cranium/cheeks/jaw, `sdCapsule` nose bridge/brows, `smin`/`smax` eye sockets & serene lips, plus a glowing forehead bindi/third-eye dot and crown headdress arches). Project translucent electric-blue/violet contour isolines (`abs(fract(p.y * freq) - 0.5)`) and Voronoi glass facets onto the raymarched surface normal with strong Fresnel rim glow.
  - *Mode B (Image → Backlit Stained-Glass Filter):* Upload any source image (e.g., Buddha, portrait, AI art), optionally mirror horizontally (`uv.x = abs(uv.x - 0.5) + 0.5`), run a 1-pass **Sobel edge detector** into a seed-init FBO placing dense Voronoi seeds along strong image edges and jittered grid seeds in flat areas, then execute **Jump Flooding Algorithm (JFA)** across ping-pong `RG32F` FBOs ($\lceil \log_2 1280 \rceil = 11$ passes, step sizes `1024 → 1`, or 10 passes at half-res). A final pass computes exact lead-line borders (neighbor seed-ID difference + distance-to-bisector), samples average/centroid cell color shifted into silver/lavender/cyan, and adds beveled glass normal highlights + backlit radial glow. *(Crucial optimization: for a static uploaded image, run JFA **once on upload** and cache the Voronoi map — runtime rendering is then a single 0.4 ms relighting/animation pass!)*
- **Cost estimate:**
  - *Mode A (Procedural 3D Face SDF):* **2.2 – 3.8 ms** at 1280×800 (56-step raymarch inside a tight bounding ellipsoid).
  - *Mode B (Live JFA Image Filter every frame):* **1.5 – 2.4 ms** (11 JFA passes at 1280×800 = 11 × 9 texel fetches, memory-bandwidth bound ~1.4 ms + Sobel + final glass composite ~0.6 ms); **0.4 – 0.6 ms** if the source image is static and JFA is cached!
- **Pitfalls:**
  - *JFA coordinate precision errors:* Never store JFA seed coordinates in `RGBA8` or `RG16F` (at 1280 px, `fp16` loses integer accuracy above 2048 and quantizes cell edges); always use `RG32F` (`dtype='f4'`) ping-pong FBOs with `NEAREST` filtering.
  - *Jagged 1-pixel stair-stepped lead lines from JFA:* Simple neighbor-ID inequality produces aliased binary edges; compute exact Euclidean distance to the perpendicular bisector between the pixel's seed and any differing neighbor seed in a 3×3 neighborhood, then apply `smoothstep` + `fwidth` AA.
  - *Uncanny procedural faces:* Keep Mode A stylized/sculptural (calm closed eyes, clean brow/cheek planes, symmetric headdress) and use smooth polynomial `smin` (`k = 0.04–0.08`) so contours flow gracefully without harsh creases.
- **References & licenses:**
  - Guodong Rong & Tiow-Seng Tan, *Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform* (ACM I3D 2006) — https://www.researchgate.net/publication/220791895_Jump_flooding_in_GPU_with_applications_to_Voronoi_diagram_and_distance_transform (**Academic paper — algorithm freely implementable**).
  - Inigo Quilez, *Smooth Minimum (`smin`)* — https://iquilezles.org/articles/smin/ (**MIT License**).
  - Inigo Quilez, *Selfie Girl* (`WsSBzh`) & *Happy Jumping* (`3lsSzf`) — https://www.shadertoy.com/view/WsSBzh and https://www.shadertoy.com/view/3lsSzf (**Inigo Quilez SDF sculpting references; check per-shader header — default Shadertoy CC BY-NC-SA 3.0 unless MIT header is present; treat as idea only**).

---

### 10. Breathing Honeycomb-Bead Chamber (Lite-Brite Morphing Form)
- **Technique:** Analytic Motion-Blurred Box Chamber + Raymarched Superquadric/`smin` Morphing Form + Spherical Fibonacci / Hex-Lattice Bead Surface SDF.
- **How it works:** Render the recessed square box chamber analytically (4 corridor planes + crisp black foreground frame SDF) with packed hex-lattice glowing bead dots (`hexCoords` → circular bead SDF with dark inter-bead gaps) stretched along `z` to simulate rushing motion-blurred walls. Inside the box, raymarch (48–64 steps) a 3D morphing centerpiece looping on a beat-synced phase $\phi \in [0, 1)$: interpolate via `mix()` / `smin()` between (1) an extruded/revolved **X-shaped faceted pillow** (`sdRoundedX` extruded + rounded), (2) **pinched parenthesis sides** (subtracting two lateral spheres via smooth-max `-smin(-d, dPinch, k)`), (3) an **inflated sphere** (`sdSphere`), and (4) **four clover lobes** (4 symmetric spheres joined by `smin` or an angle-modulated superquadric `(cos(4*theta))`), snapping elastically back to the X. To map glowing Lite-Brite beads over the morphing 3D belly *without polar pinching* and with authentic stretching as it inflates, evaluate **Keinert et al.’s Inverse Spherical Fibonacci Mapping** (constant-time nearest-dot lookup on the unit direction `normalize(p)` — or 3D triplanar hex-grid blending near high curvature), rendering each bead as a crisp glowing circular disc with dark gaps and glass specular dome highlight.
- **Cost estimate:** **2.6 – 4.5 ms** at 1280×800 (~0.5 ms analytic beaded box walls; ~2.0–3.5 ms for 56-step raymarch of the morphing shape + Inverse Spherical Fibonacci bead lookup & shading on hit pixels only).
- **Pitfalls:**
  - *Triplanar seam blending ruins discrete beads:* Standard triplanar blending creates ghosted half-dots at 45° diagonals; **Inverse Spherical Fibonacci Mapping** (`Keinert 2015`) maps the entire star-convex morphing surface seamlessly with uniform circular beads and zero seams, naturally stretching dots across the inflated belly!
  - *Morphing SDFs breaking distance bounds:* Linearly blending two distinct SDFs (`mix(dA, dB, t)`) remains a safe lower bound (Lipschitz ≤ 1), whereas superquadric algebraic functions `(x^e + y^e + z^e)^(1/e) - 1` do not; scale superquadric steps by `0.5` or stick to exact primitives (`sdRoundedX`, `sdSphere`, `smin`).
  - *High-frequency dot aliasing on distant chamber walls:* Apply analytical coverage AA (`smoothstep` width scaled by `fwidth`) and fade bead contrast slightly toward the back of the box so distant walls glow smoothly without pixel-crawl.
- **References & licenses:**
  - Benjamin Keinert, Matthias Innmann, Michael Sänger, Marc Stamminger, *Spherical Fibonacci Mapping* (ACM TOG / SIGGRAPH Asia 2015) — https://dl.acm.org/doi/10.1145/2816795.2818131 (**Academic paper — constant-time GPU inverse mapping math freely implementable**).
  - Inigo Quilez, *2D Distance Functions (`sdRoundedX`)* & *Smooth Minimum (`smin`)* — https://iquilezles.org/articles/distfunctions2d/ and https://iquilezles.org/articles/smin/ (**MIT License**).
  - Alan H. Barr, *Superquadrics and Angle-Preserving Transformations* (IEEE CG&A 1981) — https://ieeexplore.ieee.org/document/1673799 (**Academic paper — superquadric math freely implementable**).

---

### 11. Fluid Silk / Ink Filaments (500k–2M GPU Particles + Ink-on-Paper / Watercolour)
- **Technique:** GLSL 330 **Transform Feedback** (or Ping-Pong `RGBA32F` Position Textures) + Analytic 3D Curl Noise + Prev→Curr Line-Segment Rendering into Ping-Pong Trail FBO + Multiplicative Paper / Kubelka-Munk-Lite Composite.
- **How it works:** Store 500k–1.5M particles either in two ping-pong VBOs updated via **Transform Feedback** (`vao.transform(vbo_dst, moderngl.POINTS)`, 32 bytes/particle = 32 MB for 1M particles) or in ping-pong `RGBA32F` position/velocity textures (e.g., 1024×1024 = 1.05M particles, sampled in the render vertex shader via `texelFetch` by `gl_VertexID`). Each frame, advect particles through divergence-free **3D Curl Noise** (Bridson 2007: finite-difference curl of 3D simplex noise, or a fast `64³` `RGBA16F` curl-noise LUT) plus real-time **MIDI impulse/vortex uniforms** (point attractors, curl vortices, velocity bursts). Render particles as anti-aliased hair-thin segments (`GL_LINES` or instanced 2-triangle screen-aligned hair quads connecting `prev_pos → pos`) into a ping-pong `RGBA16F` trail FBO that decays exponentially (`trail *= decay`) and optionally advects/diffuses 1 tap per frame for silky/watercolour bleed. For **dark cosmic silk / gold sparklers**, composite with **additive blending** (`GL_ONE, GL_ONE`); for **sumi-e ink / watercolour / charcoal on rice paper**, accumulate ink density into the trail FBO and composite **multiplicatively / subtractively** over procedural paper grain using **Kubelka-Munk-lite** (Beer-Lambert absorption `paperColor * exp(-density * K_rgb)` + Curtis 1997 capillary edge-darkening, or MIT Spectral.js pigment curves). Enforce a **≥ 0.25 s smooth crossfade** on polarity flips (dark void ↔ bright rice paper) through the brightness limiter.
- **Cost estimate:** **2.8 – 6.2 ms** at 1280×800:
  - *500k particles:* ~0.8 ms Transform Feedback (curl noise) + ~1.4 ms line rasterization/blending + ~0.6 ms trail decay/paper composite = **~2.8 ms**.
  - *1.2M–1.5M particles:* ~1.8 ms Transform Feedback + ~3.2–3.8 ms line rasterization (sensitive to line-length ROP overdraw!) + ~0.6 ms composite = **~5.6–6.2 ms**. *(2M particles is achievable at 60 fps if segment lengths are short and simplex curl uses a pre-baked 3D curl-noise lookup texture (`64³` `RGBA16F`, 2 MB!), which cuts advection cost by ~4×.)*
- **Pitfalls:**
  - *Overdraw bottleneck at 1M+ particles:* On a 48 GPixel/s GPU, 1.5M long streaks averaging 25 px each = 37.5 Mpx/frame (~0.8–2.5 ms pure ROP blend cost, worse if wide quads). Keep line width 1 px (`GL_LINES` or thin screen-space quads) and clamp max segment stretch per frame; use a **precomputed 3D Curl-Noise LUT texture (`64³` `RGBA16F`)** sampled with hardware trilinear filtering instead of evaluating 6× 3D simplex noise per particle to hit 1.5M–2M particles effortlessly.
  - *8-bit trail quantization ("ghost stuck trails"):* Multiplying an `RGBA8` trail buffer by `0.97` stalls at `1/255` and leaves permanent gray smudges; always use `RGBA16F` for the ping-pong trail FBO.
  - *Muddy RGB blending in watercolour mode:* Standard alpha-blending yellow + blue gives dull gray; use Beer-Lambert / Kubelka-Munk absorption (`exp(-sum(c_i * K_i))`) or MIT **Spectral.js** spectral curves, and add subtle paper-fiber normal/height modulation for capillary edge darkening.
  - *Polarity-flip wall flash:* Flipping instantly from black void to white rice paper triggers a massive luminance jump; crossfade polarity over `≥ 0.25 s` (ideally `0.5–1.0 s`) and gate bright-paper modes behind an explicit user toggle.
- **References & licenses:**
  - Robert Bridson, Jim Hourihan, Marcus Nordenstam, *Curl-Noise for Procedural Fluid Flow* (ACM SIGGRAPH 2007) — https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07.pdf (**Academic paper + author public-domain reference code** at https://www.cs.ubc.ca/~rbridson/).
  - Cassidy J. Curtis et al., *Computer-Generated Watercolor* (ACM SIGGRAPH 1997, Kubelka-Munk glazing & pigment pooling) — https://grail.cs.washington.edu/projects/watercolor/ (**Academic paper — algorithm freely implementable**).
  - Ronald van Wijnen, *Spectral.js* (Kubelka-Munk pigment mixing) — https://github.com/rvanwijnen/spectral.js (**MIT License**). *(Note: Secret Weapons' **Mixbox** at https://github.com/scrtwpns/mixbox is **CC BY-NC 4.0 — do not copy for commercial use, idea only**.)*
  - ModernGL Documentation, *Transform Feedback / Particle System* — https://moderngl.readthedocs.io/ (**MIT License**).

---

### 12. Choreographed Show Sequencer (Beat-Synced 20–60 s Journeys)
- **Technique:** Beat-Locked Parameter Timeline (Python/CPU) + Unified Dual-Deck FBO Crossfade / SDF-Morph Pipeline.
- **How it works:** Drive the entire 20–60 s journey (e.g., Dome → Filigree Entity → Corridor Tiling Morphs → Stained-Glass Face → White Portal → loop) from a declarative beat-synced keyframe timeline on the CPU (quantized to BPM / bar phase from MIDI clock or Ableton Link, emitting normalized `u_beat`, `u_bar`, `u_section_t`, and eased envelope uniforms). Avoid recompiling shaders mid-show: precompile all effect programs at startup and route two active decks (`Deck A` and `Deck B`) into two `RGBA16F` FBOs, transitioning via (1) **in-shader parameter morphs** for intra-family transitions (corridor tiling angles, bead-shape loop, polarity flips — zero double-render cost) and (2) **timed transition shaders** (portal iris wipe, luminance-keyed dissolve, or smooth A/B crossfade over 1–4 bars) between distinct scene architectures. Append new decks to `fx_stack` LandPass indices monotonically (never reordering existing indices) so saved presets and choreographies remain stable.
- **Cost estimate:** **0.2 – 0.4 ms** for the compositor/transition pass itself; during a 1–2 s dual-deck crossfade, GPU cost is `cost(Deck A) + cost(Deck B) + 0.4 ms` (staying **≤ 11 ms** if each individual scene respects the ≤ 5–6 ms budget).
- **Pitfalls:**
  - *Shader compilation stutters ("hitching"):* Never compile GLSL programs lazily on scene entry; compile all 12 deck programs and warm up 1 dummy frame per program during startup.
  - *Crossfade GPU spike:* Avoid crossfading the two heaviest effects simultaneously (e.g., 1.5M particles + heavy KIFS raymarch); use the **White Portal (Effect 7)** or a brief fade-through-void/bloom bridge between heavy worlds, or scale particle count/raymarch steps down 25% during overlaps.
  - *Abrupt loop seams:* Ensure every time-driven parameter (torus rotation, corridor `z` mod period, curl-noise phase, hue cycle) wraps cleanly over an exact integer number of bars/beats so 20–60 s journeys loop invisibly.
- **References & licenses:**
  - Inigo Quilez, *Smoothstep Functions* & *Cosine Color Palettes* — https://iquilezles.org/articles/smoothsteps/ and https://iquilezles.org/articles/palettes/ (**MIT License**).
  - ModernGL Documentation, *Framebuffers & Multi-Pass Rendering* — https://moderngl.readthedocs.io/ (**MIT License**).

---

## Cross-Cutting GLSL 330 Engineering Notes (GTX 1650 Ti 4 GB)

1. **Universal Anti-Aliased Thin-Line Pattern (for Effects 1, 2, 3, 5, 6, 8, 10):**
   Compute an exact Euclidean SDF `d` (never un-normalized algebraic values), then filter with:
   ```glsl
   float fw = max(fwidth(d), 1e-4);
   float effW = max(lineHalfWidth, 0.75 * fw);
   float coverage = smoothstep(effW + fw, effW - fw, abs(d)) * (lineHalfWidth / effW);
   ```
   The `(lineHalfWidth / effW)` factor preserves energy when lines become sub-pixel, preventing sparkling/shimmering on distant geometry.
2. **HDR Pipeline & Precision:** Keep scene FBOs, ping-pong trails, and Dual Kawase bloom mip chain in `RGBA16F` (at 1280×800, one `RGBA16F` FBO is only **8.2 MB** — 10 full-res FBOs use < 85 MB of the 4 GB VRAM). Use `RG32F` / `RGBA32F` strictly for JFA coordinates (Effect 9) and particle positions (Effect 11).
3. **Shared Photosensitive Safety Pass:** Place a final lightweight pass before display output that tracks exponential moving-average frame luminance (via a downsampled mip or 16×10 reduction FBO) and limits whole-wall brightness slew rate (`<= 0.35` normalized luminance per second during rapid transitions, and `>= 1.0 s` for full white portal swells), complying with WCAG 2.3.1 / ITU-R BT.1702.
