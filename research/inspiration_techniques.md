# Artistic Inspirations: Publicly Documented Techniques, Pipelines & Rigs

> Practical translation of how our core visual inspirations work in production — and how to capture their signature aesthetic on a single 1280×800 short-throw projector + 4 GB GPU laptop.

---

## 1. TAS VISUALS (Georg — *The Adventurous Spark*)
- **Publicly documented pipeline** ([TAS Visuals Patreon](https://www.patreon.com/tasvisuals)):
  - **Stage & Vector UV Blueprint:** Physical festival stages (multi-layered wooden lattices + stretched translucent **Lycra** tension panels at Mo:Dem, Ozora, Boom) are drafted in **Adobe Illustrator**. Custom TAS scripts export those exact shapes directly into **Resolume Arena Advanced Output** polygons (millimeter-accurate stage masks) and **Resolume Wire** vector patches (`TAS SVG Strokes`, `TAS SVG Gradient`).
  - **The Secret Weapon — Hue-Encoded Animation (`Color Shapes` + `TAS Motion Key`):** Instead of keyframing thousands of geometric elements, TAS's Illustrator script paints stacked vector shapes sequentially across the full **HSV rainbow spectrum** (`0° → 360°`, locking first/last to red). Inside Resolume, **`TAS Motion Key`** sweeps a soft-edged sliding hue window (with optional dual-hue repetition) around that rainbow map — turning a *single static high-detail PNG/SVG* into infinite 60 FPS cascading light waves, crawling snakes, and breathing bead rings locked to the physical architecture (pure greyscale/black stays untouched).
  - **3D Heavy-Hitter Loops:** Uses **3ds Max** + **tyFlow** (procedural particle/beadwork simulation) + **Redshift** (GPU raytracing) to pre-render hyper-dense seamless crystalline/iridescent loops — and recommends **Blender Geometry Nodes** as the free equivalent.
  - **Live Engine & Custom FX:** **Resolume Arena** mixes the pre-rendered 3D loops with live **Resolume Wire** patches (`TAS Motion Key`, `TAS Mirror` dynamic symmetry folding, `TAS Slice Color`) + feedback loops, played via compact MIDI faders/knobs (e.g., Korg nanoKONTROL2).
- **Direct GLSL / Shader Translation for Fable 5.2:**
  - **Motion Key shader trick:** Compute a scalar phase `p = fract(ring_or_layer_index)` (0..1), then apply a smooth wrapped pulse `wave = pow(0.5 + 0.5 * cos(TAU * (p * repeats - t * speed)), sharpness)` to drive both **luminance/alpha** and **spectral jewel-tone recoloring** — gives instant TAS-grade chasing waves across concentric beadwork/Shipibo rings at almost zero GPU cost.
  - **Physical depth trick:** Mount 3–6 layered matte-white foam-core geometric shields at 1–3 inch depth offsets on a dark wall and map each layer as its own slice — instantly recreates the miniature festival-stage illusion.

---

## 2. Johnathan Singer
- **Publicly documented pipeline** ([Johnathan Singer](https://www.johnathansinger.com/), [Johnathan Singer Art](https://www.johnathansingerart.com/), Mettle Q&A interview):
  - **Famous live collaborations:** **Tipper**, **Shpongle** (including the Shpongle Live Band Red Rocks finale), **Dead & Company / Grateful Dead**, plus live visual animation of paintings by **Alex Grey**, **Android Jones**, **Chris Dyer**, and **Autumn Skye**.
  - **Secret Weapon #1 — 2D Paintings as 3D Displaced Relief Meshes (`After Effects` + `Mettle FreeForm Pro` & `ShapeShifter`):** Takes flat visionary paintings and feeds high-contrast luminance or fractal noise into **3D displacement maps**, pushing pixels along the Z-axis into living, undulating 3D topography/bas-relief with virtual camera motion and specular lighting.
  - **Secret Weapon #2 — Conformal Complex-Plane Recursion (`Mettle Mantra VR` & `FLUX`):** Rather than simple mirror tiles, uses mathematical recursion — **Escher Droste** infinite spirals, **Möbius transformations**, and 3D fractal nebulae (**FLUX**) — so visionary paintings melt continuously into themselves with zero harsh seams.
  - **Secret Weapon #3 — Improvisational "Video Mixology" (`ArKaos GrandVJ` + MIDI):** Notably uses **ArKaos GrandVJ** (with hardware redundancy/backup servers) instead of rigid timecoded playlists. Plays layers like a jazz instrument alongside improvised Tipper sets, maintaining **deliberate, cohesive color narratives** (smooth chromatic arcs instead of random clashing colors) and organic speed/blend modulation.
- **Direct GLSL / Shader Translation for Fable 5.2:**
  - **Luminance 3D Relief / Parallax Shading:** Treat the AI texture's brightness `h = dot(col, vec3(0.299, 0.587, 0.114))` as a heightfield! Offset UVs gently by view vector (`parallax`) and compute finite-difference screen normals `N = normalize(vec3(dFdx(h), dFdy(h), depth))` for dynamic moving virtual rim/specular light — makes flat AI textures look like breathing liquid-metal / oil-paint 3D reliefs!
  - **Complex-Plane Möbius + Escher Droste Warp:** Treat UV as complex number `z = vec2(x, y)`:
    - **Escher Droste:** `log_z = vec2(log(length(z)), atan(z.y, z.x))` → apply a slight spiral rotation/skew matrix `(r1, r2)` before wrapping `fract()` → infinite seamless logarithmic recursion with twisting spiral arms.
    - **Möbius transform:** `w = cdiv(cmul(a, z) + b, cmul(c, z) + d)` (slowly orbiting poles `b, d` with mid-frequency audio) → smooth, alien, non-Euclidean folding that preserves local angles (so intricate detail stays crisp instead of smearing!).

---

## 3. Android Jones
- **Publicly documented pipeline** ([Android Jones](https://www.androidjones.com/)):
  - **Tools:** Pioneered "Electro-Mineralism" digital painting (Corel Painter, Photoshop, ZBrush) and co-created **Microdose VR** — a real-time VR psychedelic painting/performance engine where artists paint symmetrical 3D audio-reactive light sculptures live and pipe the output via **Spout / Syphon** into Resolume or TouchDesigner for concert projection.
  - **Large-scale mapping:** Fulldome 360° projection mapping (*Samskara* dome exhibition, collaborating with 3D animators and dome media servers) and monumental architectural projection (e.g., Sydney Opera House, Empire State Building).
- **Takeaway for our rig:** Strong **bilateral & radial symmetry** (2-fold vertical mirror for iconic totem/face/portal presence, transitioning smoothly into 6/8/12-fold mandalas), high-contrast luminous highlights against deep cosmic shadows, and slow multi-layer parallax.

---

## 4. Alex Grey (Aesthetic Reference Only — Do Not Train On)
- **Visual principles to translate mathematically into shaders/3D:**
  - **Translucent energy lattices:** Layered semi-transparent grids of luminous meridians and intersecting arcs (*Net of Being* infinite mirrored columns / arches).
  - **Radiant focal points:** Glowing concentric haloes/flames/chakra nodes pulsing gently on sub-bass swells, surrounded by crisp golden-amber, fiery orange, violet, and cyan filigree.
  - **Multi-scale detail:** Macro architectural symmetry visible from across the room + intricate micro-filigree visible up close.

---

## 5. Burning Man Sculpture & Art Car Projection Mapping
- **Publicly documented specs & rigs** (e.g., large playa installations, [Tempest Enclosures](https://www.tempest.biz/)):
  - **Projector Lumens & Counts:** Massive outdoor pyramids, geodesic domes, and art cars typically use **multiple 15,000–32,000+ ANSI lumen laser projectors** (often 2–8+ blended units) to cut through blowing playa dust, distant camp light, and 50–150 ft throw distances.
  - **Dust & Thermal Protection:** Projectors sit inside pressurized/filtered outdoor enclosures (Tempest Cyclone/Blizzard) to survive fine alkaline dust and 50°F desert night swings.
  - **Mapping Workflow:** 3D CAD/photogrammetry model of the sculpture → calibrated in **TouchDesigner (`CamSchnappr`)**, **Resolume Arena**, or **MadMapper** with soft edge-blending across overlapping projector frustums.
- **Scaled-down lesson for our 3,200-lumen BenQ MW632ST:**
  - Brightness scales with `1 / area`. At **3–5 ft** (short-throw `0.72:1`), 3,200 lumens delivers **106–295 lumens/sq ft** — brighter per square foot than a 20,000-lumen projector hitting a 30 ft Burning Man sculpture! Keep surfaces **matte white** (foam core, flat white paint, white ripstop nylon) and kill ambient light to maximize perceived contrast.

---

## 6. The Gorge Amphitheatre — Above & Beyond Group Therapy Weekender
- **Publicly documented production** (Lighting & visual director **Neil Marsh**, [Live Design](https://www.livedesignonline.com/), visual content creators including Dylan Byrne):
  - **Why it looks the way it does:** Open-air amphitheaters at twilight/dusk rely primarily on **massive high-nit LED walls, synchronized laser arrays, and architectural moving-head rigs** (controlled via consoles like **ChamSys MagicQ**) paired with **Resolume** media servers locked via **SMPTE timecode**, rather than long-throw projection (ambient sunset light washes out projection).
  - **Aesthetic & Pacing Lessons for Our Shaders:**
    - **Emotional arc over 4-bar / 8-bar phrases:** Quiet, breathable, minimal geometric horizons during trance/melodic breakdowns → gradual build in saturation, symmetry complexity, and zoom velocity → clean, sweeping bloom on the drop (never muddy visual clutter).
    - **Bold architectural framing:** Crisp glowing geometric outlines (ribbon arches, chevrons, portals) framing dark negative space, paired with warm sunset-amber → deep sapphire-indigo → electric cyan/magenta palette progressions.

---

## 7. Insider FOH & Collaborator Notes (Online Crew / VJ Accounts)

### Johnathan Singer (FOH & Collaborator Breakdowns)
- **Multi-computer + Hardware Switcher Architecture:** Rather than running everything on one overloaded laptop, Singer stated (*Mettle* Q&A) he runs playback engines (**ArKaos GrandVJ / MediaMaster**, **Hippotizer**, or **disguise d3**) mixed through a **Roland V-800HD hardware video switcher** and multiple **MIDI controllers**. FOH observers on `r/Tipper` / `r/vjing` note he splits **Machine 1 (pre-rendered visionary art loops — including *Psychoportation* co-created with Glass Crane, plus Alex Grey / Android Jones / Amanda Sage / Chris Dyer works)** from **Machine 2 (a live 3D spatial/line/fractal simulator he literally "flies" in real time)**, blending and luma-keying them in hardware for zero dropped frames and instant failover.
- **No-Setlist Improvisation & Color Narratives:** Both Singer and `r/Tipper` insiders emphasize that Tipper rarely gives a locked setlist. The secret to keeping improvised visuals hypnotic rather than chaotic is **cohesive color narratives** (holding one harmonic palette family for minutes before gliding into the next) and heavy emphasis on **hardware redundancy** (backup server, spare cables/paths).
- **Stage Architecture Collaborations:** Worked alongside stage sculptor **Carey Thompson** and **Android Jones** on enveloping room-wide architectural mappings (e.g., The Fillmore).

### TAS Visuals (Georg Peinhaupt) & Close Collaborators (*Delta Process*, *Alien Art*)
- **Co-design with Delta Process (Federico Petrei & Fay Fernandes):** TAS and Italian mapping collective **Delta Process** regularly collaborate on monumental psychedelic mapping — including **Alex Grey's *Entheon* grand opening at CoSM**, **Mo:Dem's *The Hive* stage**, and **Alien Art** (Ace Ventura & Captain Hook).
- **Ableton Live → `loopMIDI` → Resolume Precision Sync (*Alien Art – Mutation* breakdown):** Instead of relying solely on raw mic/FFT reactivity, TAS splits a track in **Ableton Live** into ~10 structural blocks (intro/build/drop/breakdown), maps **1 Resolume column per musical segment**, and routes **MIDI Notes + CC automation envelopes** over Windows **`loopMIDI`** straight into Resolume Arena — driving clip triggers, layer opacities, playback speed, and `TAS Motion Key` hue ramps with zero human latency.
- **Physical-First 3D Camera Matching & GPU Codecs:** On festival stages (CNC-milled wood + stretched Lycra), virtual 3D cameras in 3ds Max/Blender are matched directly to physical projector lens positions/throws, loops are encoded in **DXV3 / HAP** (GPU-decoded for stutter-free multi-layer 4K), and multi-projector setups use Illustrator-exported slices or camera calibration (e.g., VIOSO).

---

## Sources
- TAS Visuals Patreon (3ds Max, tyFlow, Resolume Arena/Wire workflows): https://www.patreon.com/tasvisuals
- Delta Process (Federico Petrei & Fay Fernandes — Entheon / Mo:Dem / TAS collaborations): https://www.deltaprocess.it/
- Johnathan Singer official sites: https://www.johnathansinger.com/ and https://www.johnathansingerart.com/
- Glass Crane × Johnathan Singer (*Psychoportation* VJ pack): https://iamglasscrane.com/
- Android Jones official site (Microdose VR, Samskara): https://www.androidjones.com/
- Tempest outdoor projector enclosures (Burning Man / extreme environment protection): https://www.tempest.biz/
- Live Design (Above & Beyond / Neil Marsh lighting & visual production): https://www.livedesignonline.com/
- Resolume Arena documentation: https://resolume.com/support/en/advanced-output
