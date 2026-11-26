# Professional VJ & Projection-Mapping Software Survey (2025–2026)

> **Scope:** Web-researched technical and UX analysis of 12 professional VJ, generative visual, and projection-mapping applications (2025–2026 versions), shader/generator ecosystems with **AGPL-3.0** open-source licensing analysis, and a ranked Top-15 architecture/UX blueprint for a Python + `moderngl` VJ engine with a browser-based control panel.

---

## Table of Contents
1. [Executive Summary & Comparative Matrix](#1-executive-summary--comparative-matrix)
2. [Product Deep-Dives](#2-product-deep-dives)
   - [2.1 Resolume Arena 7 (+ Wire)](#21-resolume-arena-7--wire)
   - [2.2 TouchDesigner (2025–2026 Builds)](#22-touchdesigner-20252026-builds)
   - [2.3 VDMX6 (6.0–6.3)](#23-vdmx6-6063)
   - [2.4 MadMapper 5 & MadMapper 6 (6.0–6.2)](#24-madmapper-5--madmapper-6-6062)
   - [2.5 Millumin 5](#25-millumin-5)
   - [2.6 Notch (0.9.x / 1.0 / 2026.1)](#26-notch-09x--10--20261)
   - [2.7 Synesthesia (`synesthesia.live`)](#27-synesthesia-synesthesialive)
   - [2.8 Magic Music Visuals](#28-magic-music-visuals)
   - [2.9 HeavyM 2](#29-heavym-2)
   - [2.10 CoGe (Discontinued Legacy)](#210-coge-discontinued-legacy)
   - [2.11 Modul8 (Legacy)](#211-modul8-legacy)
   - [2.12 Lumen (Paracosm)](#212-lumen-paracosm)
3. [Shader & Generator Ecosystems and AGPL-3.0 Licensing](#3-shader--generator-ecosystems-and-agpl-30-licensing)
4. [Top 15 Pro Capabilities & UX Patterns for a Python + moderngl + Browser VJ Engine](#4-top-15-pro-capabilities--ux-patterns-for-a-python--moderngl--browser-vj-engine)

---

## 1. Executive Summary & Comparative Matrix

Across the 2025–2026 live-visual landscape, professional VJ and projection-mapping software splits into four distinct paradigms—yet top performers routinely bridge them via **Syphon (macOS)**, **Spout (Windows)**, and **NDI**:

1. **Clip-Grid & Layer Mixers** (*Resolume Arena 7*, *VDMX6*, *Modul8*, legacy *CoGe*): Built for split-second improvisational mixing, live thumbnails, deck/page switching, A/B crossfading, stacked FX chains, and hardware MIDI controller muscle memory.
2. **Shader / Generative Instruments** (*Synesthesia*, *Magic Music Visuals*, *Lumen*): Centered on real-time GPU synthesis (GLSL/ISF/SSF or analog-modeled oscillators), tight out-of-the-box audio reactivity, and a **small, curated set of named macro controls** rather than raw code or massive clip libraries.
3. **Projection Mapping & Cue/Timeline Show Servers** (*MadMapper 5/6*, *Millumin 5*, *HeavyM 2*): Centered on physical geometry (quads, meshes, Bézier masks, 3D surfaces, soft-edge blending), spatial calibration, and structured cues/timelines synchronized to LTC/MTC/Art-Net.
4. **Real-Time Node Authoring Engines** (*TouchDesigner*, *Notch 1.0*): Blank-canvas GPU/3D/particle programming environments that either run custom Perform Mode UIs or export parameterized **Blocks / `.tox` components** into host media servers.

### Comparative Matrix (2025–2026)

| Product | Primary Paradigm | OS | Native Shader Format | Audio Reactivity | Output Mapping | Status (2025–2026) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Resolume Arena 7 (+ Wire)** | Clip grid + layers + node patcher | Win / macOS | FFGL, ISF, Wire patches | External/Comp FFT + BPM | Slices, masks, warp, soft-edge, DMX | Active (v7.21–7.24, 10-bit, REST API) |
| **TouchDesigner** | Node-based visual programming | Win / macOS | GLSL TOPs, POPs, Python | Deep CHOP FFT / multi-channel | KantanMapper, CamSchnappr, Stoner | Active (2025/2026 POPs, Gaussian Splats) |
| **VDMX6** | Modular DIY VJ instrument | macOS (Metal) | ISF (originator), Metal, TD/Vuo | Universal Audio Analysis plugin | Surface/Quad warp, multi-screen | Active (v6.1 CV/OCR, v6.3 Expressions) |
| **MadMapper 5 / 6** | Projection mapping + show timeline | Win / macOS | GLSL Materials + ISF passes | FFT / BPM / Ableton Link | 3D surface soft-edge, Spatial Scanner | Active (v6.0–6.2 Timeline/Conductor/Tasks) |
| **Millumin 5** | Cue-column + timeline media server | macOS | ISF shaders | Audio input / LTC / timeline | Mesh/Bézier warp, multi-proj soft-edge | Active (v5) |
| **Notch (1.0 / 2026.1)** | Real-time 3D/VFX node engine | Win (Builder) | Custom GLSL / NURA nodes | Audio reactive nodes + host feeds | Hosted in Disguise/Resolume/Pixera | Active (v1.0 / 2026.1 NURA overhaul) |
| **Synesthesia** | Psychedelic GLSL shader instrument | Win / macOS | SSF (`.synScene`), ISF/Shadertoy import | Best-in-class auto FFT/hits/BPM | Single output + Spout/Syphon | Active (Standard & Pro tiers) |
| **Magic Music Visuals** | Modular audio-reactive graph | Win / macOS | ISF, GLSL | Multi-stem / multi-band FFT | Basic multi-display / quad warp | Active |
| **HeavyM 2** | Vector face projection mapper | Win / macOS | Generative & post-FX GLSL | Tempo/TAP/Link + audio input | Faces, maps, warping, edge-blend | Active (v2, AI generators) |
| **CoGe** | Semi-modular ClipSynth VJ | macOS ≤10.14 | Quartz Composer, ISF, Vuo | Audio Analysis, LFO, MIDI clock | Basic / Syphon to mapper | **Discontinued** (Oct 2022) |
| **Modul8** | Legacy layer mixer + Python modules | macOS | OpenGL filters / Python modules | Audio FFT + BPM | Basic (paired w/ MadMapper) | Maintenance/Legacy (v3 via Rosetta) |
| **Lumen** | Analog-style modular video synth | macOS | Analog-modeled GPU synth | External via MIDI/Syphon/OSC | Single output + Syphon | Active niche synth |

---

## 2. Product Deep-Dives

### 2.1 Resolume Arena 7 (+ Wire)

* **Official Documentation:** [Resolume Support Manual](https://resolume.com/support/en/), [Clips](https://resolume.com/support/en/clips), [Layers](https://resolume.com/support/en/layers), [Effects](https://resolume.com/support/en/effects), [Dashboard](https://resolume.com/support/en/dashboard), [Envelopes](https://resolume.com/support/en/envelopes), [Parameter Animation](https://resolume.com/support/en/parameter-animation), [MIDI Shortcuts](https://resolume.com/support/en/midi-shortcuts), [Advanced Output](https://resolume.com/support/en/advanced-output).
* **2025–2026 Versions:** Resolume 7.21 through 7.24 introduced Wire **Input Groups**, interactive preview transform widgets, REST API clip/group clearing (7.21), column beat-snap & autopilot upgrades (7.22), reusable **Parameter Animation Presets** (7.23), and **native 10-bit color output**, faster high-volume MIDI processing without UI lag, and Wire's Patch Navigator & hover texture previews (7.24).

#### Core Paradigm & Moment-to-Moment Performer Workflow
Resolume Arena is the de facto live club and festival VJ instrument. Its layout is a horizontal **Layer × Column clip matrix** above an inspector pane. Moment to moment, a performer taps BPM (or syncs via Ableton Link/MIDI clock), fires clips on independent layers (e.g., Layer 1 background texture, Layer 2 geometric focal loop, Layer 3 logo/strobe mask), rides layer opacity faders and an A/B crossfader, punches momentary effects or twists Dashboard macro knobs, and triggers entire Columns to change looks in lockstep with song drops.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** Clips live inside **Decks** (tabs acting like record crates—e.g., Intro, Techno, Ambient) organized in a horizontal Layer × Column grid with visual **clip thumbnails**. Clicking a thumbnail launches the clip on that layer; clicking a Column header triggers all clips vertically across layers, with optional beat-snap quantization (expanded in 7.22).
* **Generator / Shader Parameter Exposure:** Built-in Sources, ISF shaders, FFGL plugins, and **Resolume Wire** patches appear as drag-and-drop clips or effects exposing clean, named UI sliders, toggles, dropdowns, and color pickers. Since v7.21, Wire patches support **Input Groups**, allowing patch authors to fold and organize controls into clean, minimal sections inside Arena.
* **Effects Chains:** Over 100 built-in GPU effects (plus FFGL/ISF/Wire) can be stacked at four scopes: **Clip**, **Layer**, **Group**, and **Composition** (master). Effects execute top-to-bottom in drag-reorderable stacks; every effect provides an **Opacity (dry/wet)** slider, bypass toggle, blend mode, and saveable **Effect Presets**.
* **Parameter Animation:** Any slider can switch its animation driver to **Timeline**, **BPM Sync**, **Audio FFT** (External, Composition, or Clip audio with selectable frequency band window, gain, falloff, and direction), **Dashboard** dial, or **Crossfader Phase**. Custom **Envelopes** (multi-point curves) shape any animated parameter, and v7.23 lets performers save/recall **Parameter Animation Presets** globally. Each Clip, Layer, Group, and Composition has an 8-dial **Dashboard** for multi-parameter macro control.
* **Transitions:** Every layer includes an **Auto Transition** mixer (0–10 s duration) supporting Alpha fade plus dozens of GPU transitions (Push, Wipe, Jitterbug, Luma Key, or any Wire/FFGL transition). An **A/B Crossfader** blends assigned layers/groups with selectable crossfader curves and can also drive parameter modulation via Crossfader Phase.
* **Layer Compositing:** Photoshop-style bottom-to-top compositing with ~50+ blend modes (Alpha, Add, Screen, Multiply, Difference, Luma/Chroma keys, 50-mask/stencil modes), per-layer opacity, Layer Groups (sub-mixes), and **Layer Routers** for routing one layer's texture as an input or mask to another.
* **Media Browser / Library Management:** The right-hand **Files Panel** browses local drives with toggleable thumbnails, folder **Favorites** (heart icon), search filtering, and drag-and-drop into clip slots. The **Media Manager** tracks missing files, supports "Locate All," and collects composition media for touring portability.
* **Preset / Scene Save, Recall & Morph:** Compositions save decks/columns/routing; Effects and Parameter Animations have individual presets. Columns function as instant full-stage scene recalls (with optional per-layer transition fades acting as visual morphs).
* **MIDI / OSC Mapping UX:** Dedicated **Shortcuts** mode overlays color-coded mappable targets directly on the UI; click a control and move a MIDI knob/pad to learn. Supports **bi-directional MIDI LED color feedback** with built-in velocity/color lookup tables (e.g., Akai APC40mk2 clip-state swatches), MIDI shortcuts scoped by *Composition*, *Layer*, *Selected Layer*, or *Clip*, plus full OSC address trees and a **REST API** (Arena).
* **Output / Mapping:** **Advanced Output** separates Input Selection (cropping/sampling composition or specific layers) from Output Transformation: arbitrary **Slices**, **Polygon/Bézier Masks**, grid/Bézier **Warping**, multi-projector **Edge Blending (soft-edge)** with gamma/luminance curves, **Fixture/DMX/Art-Net** pixel mapping, and multi-display/NDI/Spout/Syphon routing.
* **Performance UX Details:** Dual **Preview Monitor** (audition clips, adjust trim/FX, or preview files from the browser before pushing live) and **Output Monitor**; one-click **Composition Bypass / Blackout (Panic)** commonly bound to spacebar or MIDI; per-layer and per-column **Autopilot** (Forward, Backward, Random, Bag) for hands-free sets.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Industry-standard reliability; intuitive clip-grid ergonomics; **perpetual (non-subscription) license** with famous 50%-off Black Friday sales; buttery scrubbing via the frame-based **DXV** codec; Wire keeps the ecosystem modern.
  * *Complaints:* Heavy CPU/GPU resource consumption (particularly noted on macOS Sequoia/Apple Silicon even on light compositions); enormous DXV file sizes; basic stereo audio engine; heavy reliance on pre-rendered clips unless paired with TouchDesigner or Wire.

---

### 2.2 TouchDesigner (2025–2026 Builds)

* **Official Documentation:** [Derivative TouchDesigner Docs](https://docs.derivative.ca/), [Release Notes & Learn Portal](https://derivative.ca/learn).
* **2025–2026 Versions:** The 2025/2026 Official builds introduced **POPs (Point Operators)**—GPU-accelerated 3D geometry/point-cloud processing—native **Gaussian Splat** loading (`.gsz`/`.ply` via Point File In POP), `.pop` lossless geometry files, Blob Track TOP enhancements (lifting non-commercial blob limits), Blackmagic 64-channel audio, and multi-monitor Perform Mode fixes.

#### Core Paradigm & Moment-to-Moment Performer Workflow
TouchDesigner is not turnkey VJ software; it is a real-time procedural node environment split into operator families: **TOPs** (2D GPU textures), **CHOPs** (channels/audio/MIDI/LFOs), **SOPs/POPs** (CPU/GPU 3D geometry), **DATs** (Python/tables/OSC), **MATs** (materials), and **COMPs** (UI panels & containers). Performers either build custom show-control interfaces and enter **Perform Mode** (single-window GPU-optimized full-screen mode hiding the node editor) or use TouchDesigner as a generative engine piped into Resolume/VDMX via Spout/Syphon/NDI.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** No built-in clip grid out of the box. VJs build custom Switch TOP / Movie File In TOP / `.tox` dynamic-loader banks (or adopt community VJ shells like `tdVJ` or Palette UI widgets) with live TOP viewer thumbnails.
* **Generator / Shader Parameter Exposure:** Any COMP (`.tox` component) defines **Custom Parameters** (floats, sliders, RGBA, menus, pulses, Python callbacks) grouped onto clean pages via the Component Editor—turning a 200-node network into a 6-knob instrument panel.
* **Effects Chains:** Built by wiring TOPs sequentially (Level, Blur, Feedback, GLSL TOP, Composite). Dry/wet is explicitly constructed with Cross TOP or Comp TOP; effect ordering is patched via wires or dynamic Switch/Select TOP routers.
* **Parameter Animation:** Unrivaled modulation via **CHOPs**: `Audio Device In` → `Audio Spectrum` / `Audio Filter` / `Analyze` CHOPs (multi-band FFT, RMS, onset detection), `LFO CHOP`, `Beat CHOP`, `Timeline/Animation COMP` (keyframe curves), `Lag/Filter/Math CHOPs` for smoothing, and direct Python expressions on any parameter.
* **Transitions:** Built explicitly with `Cross TOP`, `Switch TOP` (with Blend Between Inputs), custom GLSL transition shaders, or CHOP-driven envelopes.
* **Layer Compositing:** `Composite TOP` (30+ blend modes), `Over TOP`, `Multiply TOP`, `Layout TOP`, and `Render TOP` (3D camera compositing).
* **Media Browser / Library Management:** File system + built-in **Palette** (curated `.tox` tools, UI widgets, image filters, generators, mapping tools) plus community registries like **Olib** (`olib.amb-service.net`). No native media-tag browser unless built in Python/UI COMPs.
* **Preset / Scene Save, Recall & Morph:** Built-in Palette **Preset Engine / Tau / TDAbleton** or custom Python JSON/DAT state engines (`TDStoreTools`, `Parameter Execute`) support instant recall and smooth CHOP-interpolated morphing between parameter snapshots.
* **MIDI / OSC Mapping UX:** Low-level, ultra-flexible via `MIDI In Map / MIDI In CHOP`, `MIDI Out CHOP` (full LED feedback via custom scripts/CHOPs), and `OSC In/Out CHOP & DAT`. Supports Ableton Live (`TDAbleton`), GrandMA3/DMX/sACN/Art-Net, Kinect, MediaPipe, LiDAR, and custom Python protocols—though MIDI "Learn" is less point-and-click than Resolume unless using `MIDI Map Dialog`.
* **Output / Mapping:** Built-in Palette mapping tools: **KantanMapper** (2D quad/freeform/mask projection mapping), **Stoner** (grid/keystone corner-pin warping), **CamSchnappr** (3D physical-to-virtual camera calibration), and `Window COMP` spanning multi-projector arrays, plus POP-aware Camera COMP Quad Reprojection (2026).
* **Performance UX Details:** **Perform Mode** disables editor overhead for locked 60+ fps touring stability. Preview/program, blackout panic switches, and autopilot logic are entirely user-architected (a strength for stadium tours, a burden for quick club prep).
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Infinite creative ceiling; avoids the "stock preset pack" look; unmatched protocol/sensor/lighting integration; rock-solid on stadium tours once locked down; POPs (2025/2026) unlock GPU-scale particles and Gaussian Splats.
  * *Complaints:* "Not VJ software out of the box"—requires months to build a gig-ready clip/transition/panic UI; **single-core CPU cooking bottleneck** (heavy CPU logic drops frame rate unless split across multiple TD instances via shared memory/TouchIn-Out); paid TouchPlayer licenses required for commercial multi-machine deployment.

---

### 2.3 VDMX6 (6.0–6.3)

* **Official Documentation:** [VIDVOX VDMX Docs](https://docs.vidvox.net/), [Plugins](https://docs.vidvox.net/vdmx/plugins.html), [Media Management](https://docs.vidvox.net/vdmx/media_management.html), [OSCQuery](https://docs.vidvox.net/vdmx/oscquery.html), [Supported Media](https://docs.vidvox.net/vdmx/supported_media.html).
* **2025–2026 Versions:** VDMX6 moved to a **native Apple Metal rendering engine** (late 2024), followed by **VDMX 6.1 (2025)** adding native Apple Vision computer vision (face/hand/body tracking), live OCR/QR scanning, color scopes, and AI-assisted Color Transfer FX, and **VDMX 6.3 (mid-2026)** adding **Control Surface Expressions** (inline math expressions for custom LFOs/gates/crossfades), reusable **Plugin Templates**, and centralized macOS permissions diagnostics.

#### Core Paradigm & Moment-to-Moment Performer Workflow
VDMX6 is a **modular, build-your-own VJ instrument** for macOS. Instead of a fixed layout, performers assemble floating/docked plugins—Media Bins, Layer Controls, Two Channel Mixers, Preview Windows, Control Surfaces, LFOs, Audio Analysis, Cue Lists—inside a **Workspace** (or start from ready-made Workspace Templates like Simple Mixer or APC40). Its defining philosophy is **"universal data sources"**: *every* slider, button, and color picker can subscribe to any internal or external data stream.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** The **Media Bin** plugin displays clips, images, ISF generators, live cameras, and Syphon/NDI feeds as customizable thumbnail grids organized into **Pages**. Media Bins can target a specific layer, cycle across layers, or trigger multiple layers simultaneously. Supports RAM preloading for zero-latency launch.
* **Generator / Shader Parameter Exposure:** VIDVOX invented **ISF (Interactive Shader Format)** specifically for VDMX. Dropping any `.fs` ISF generator or filter automatically builds clean named sliders, buttons, color pickers, and XY pads from its JSON `INPUTS` header. VDMX6 Plus also natively runs **TouchDesigner (`.tox`)** and **Vuo** compositions with exposed inputs.
* **Effects Chains:** Each Layer, Group, or Main Output has a drag-reorderable **FX Chain** supporting ISF shaders, Metal-accelerated filters, CoreImage, and FFGL. Every effect has an individual **Wet/Dry** slider, bypass toggle, and chain-level **FX Presets** (save/recall entire stacks).
* **Parameter Animation:** Dedicated modular plugins publish named data sources: **Audio Analysis** (multi-band FFT with peak/smoothing/threshold), **LFO** (multi-waveform, BPM-synced, arbitrary custom curves), **Clock** (BPM tap, Ableton Link, MIDI clock), **Step Sequencer**, **Control Surface** (custom macro knobs/sliders/XY pads, now with inline math **Expressions** in v6.3), and **Vision/CV Tracking** (v6.1). Right-clicking *any* UI slider lets you bind it to any data source with custom min/max scaling, inversion, and easing curves.
* **Transitions:** The **Two Channel Mixer** plugin delivers DJ-style A/B crossfading with selectable crossfade curves, auto-fade durations, and **ISF transition shaders** (wipes, dissolves, luma masks, glitches) between Layer A and Layer B.
* **Layer Compositing:** Arbitrary multi-layer and nested **Layer Group** compositing with OpenGL/Metal/CoreImage blend modes (Add, Over, Multiply, Screen, Difference, Luma/Chroma masks, etc.), per-layer opacity, 2D/3D transforms, and layer-as-mask routing.
* **Media Browser / Library Management:** Drag directly from macOS Finder into Media Bin pages; supports live/custom thumbnails, clip-level metadata, per-clip default playback/FX settings, and HAP/ProRes/MP4/WebM hardware-accelerated playback.
* **Preset / Scene Save, Recall & Morph:** Hierarchical **Presets** exist at every level—Individual FX, FX Chains, Layers, Plugins, and **Workspace Presets** (entire show states)—with smooth **timed fade/interpolation** between presets and linear automation via the **Cue List** plugin.
* **MIDI / OSC Mapping UX:** Best-in-class **Hardware Detect / Learn**: right-click any control → "Detect MIDI/OSC," wiggle a knob, done. Full **bi-directional MIDI LED talk-back** (color/velocity state sync for Launchpad/APC/OhmRGB), native **OSCQuery** automatic parameter discovery, and downloadable controller templates.
* **Output / Mapping:** Built-in **Canvas / Main Output** supports multiple physical displays, quad/corner-pin warping, layer masking, and edge blending; complex 3D architectural mapping is commonly piped via zero-copy **Syphon** into MadMapper.
* **Performance UX Details:** Unlimited dockable **Preview Windows** (monitor any individual layer, group, pre-FX feed, or final output); Media Bin hover/cue preview; instant master fade/blackout via Control Surface or shortcut; Media Bin auto-trigger/random playback modes.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Metal rewrite runs effortlessly and cool on Apple Silicon; extraordinary value ($99 base / $249 Plus perpetual license); native ISF + TD/Vuo + CV/OCR in v6.1–6.3; unmatched data-source modulation flexibility and friendly developer support.
  * *Complaints:* **Strictly macOS-only** (single biggest complaint on `r/vjing`); modular blank-canvas learning curve intimidates beginners used to Resolume's fixed grid; strict 3-machine activation policy.

---

### 2.4 MadMapper 5 & MadMapper 6 (6.0–6.2)

* **Official Documentation:** [MadMapper Documentation](https://madmapper.com/documentation/), [Resources](https://madmapper.com/resources/), GarageCube release notes (v6.0.0 Nov 2025 – v6.2 2026).
* **2025–2026 Versions:** **MadMapper 6** entered public beta in late 2024 and officially shipped **v6.0.0 in November 2025**, with **v6.0.8–6.2 through 2026**. It transformed MadMapper from a pure surface mapper into a full show-control suite by adding a **Timeline & Conductor** (MTC/LTC/Art-Net timecode), **Tasks (CueList with GO button)** in v6.2, **ISF multi-pass shaders**, **Mad AI** text-to-GLSL material generation, offline library caching, and self-contained **"Collect and Save"** project folders.

#### Core Paradigm & Moment-to-Moment Performer Workflow
MadMapper is the industry standard for **architectural projection mapping, LED pixel mapping, and laser control**. Its core workspace splits into an **Input UV preview** (left) and **Output Stage canvas** (right) populated with geometric **Surfaces** (Quads, Triangles, Circles, Meshes, 3D Objects, Lines, Masks). In live shows, performers switch **Scenes** (full project state) and **Cues** (partial parameter/media overrides) on a grid, run sequential **Tasks** (v6.2), or drive timecoded shows from **The Conductor** timeline—frequently ingesting live generative feeds from Resolume/TouchDesigner/VDMX via Syphon/Spout/NDI.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** Media Bin on the left holds video clips, images, live inputs (Syphon/Spout/NDI/cameras), **Generators**, and **GLSL Materials** with live thumbnails. Clicking a media asset assigns it to the currently selected surface(s); live switching during performance happens via the **Scenes & Cues grid** (with BPM / Ableton Link launch quantization in v6).
* **Generator / Shader Parameter Exposure:** Built-in **Generators** and **GLSL Materials** (using MadMapper's `materialColorForPixel` boilerplate or **ISF** JSON headers) automatically expose clean named sliders, color pickers, checkboxes, and speed controls in the Media Inspector. The integrated **Material Editor** allows live GLSL editing and **Mad AI** prompt-to-GLSL generation (v6).
* **Effects Chains:** Per-surface color correction, blending, feathering, and **Surface FX** (plus Material-level shader passes via ISF render passes in v6). Less oriented toward deep DJ-style stacked VJ effect chains than Resolume or VDMX.
* **Parameter Animation:** Every surface, material, and fixture parameter can be modulated by **Audio FFT / Beat Detection**, **BPM / Ableton Link**, built-in **LFO/Generator controls**, or animated with linear/Bézier/Catmull-Rom keyframes on the **MadMapper 6 Timeline**.
* **Transitions:** Scenes and Cues support configurable **smooth crossfade durations** that interpolate both media opacity and geometric/parameter states; Timelines support montage fades and follow actions.
* **Layer Compositing:** Surfaces stack in Z-order on the stage canvas with per-surface opacity, blend modes (Over, Add, Multiply, Screen, etc.), and vector **Bézier Masks** (including inverted masks).
* **Media Browser / Library Management:** Built-in Media panel + online **Public Material Library** (download community GLSL materials directly in-app, now cached for **offline access** in v6), plus **Collect and Save** project bundling.
* **Preset / Scene Save, Recall & Morph:** Row 1 of the cue bank holds **Scenes** (complete snapshots; auto-hides surfaces added later to prevent leakage); subsequent rows hold **Cues** (targeted subsets—e.g., Row 1 stage, Row 2 side walls, Row 3 DMX LEDs) that crossfade/morph independently. v6.2 adds **Tasks** for sequential theater-style GO cue lists.
* **MIDI / OSC Mapping UX:** Unified **Control List** with instant **Learn** for MIDI Notes, CC, and 14-bit Pitch Bend, plus **OSC** and **OSCQuery** (automatic network parameter discovery with Chataigne or other apps), DMX/Art-Net/sACN, and Keyboard shortcuts.
* **Output / Mapping:** Industry-leading: **Quad/Triangle/Mesh/3D OBJ** warping, **surface-level 3D Soft-Edge Blending** with Auto-Setup and test cards, **Spatial Scanner** (automated structured-light camera calibration), multi-projector outputs, DMX/Art-Net LED fixture mapping, ILDA laser output, and standalone export to **MiniMAD** hardware players (plus v6 Timeline export to video/ILDA/MadLight).
* **Performance UX Details:** Dual Input/Stage preview; Cue Scheduler & `.ics` Calendar Scheduler for unattended installations; global blackout / master dimmer; Conductor timecode sync (LTC/MTC/Art-Net).
* **User Praise & Complaints (2025–2026):**
  * *Praise:* **Spatial Scanner** saves hours of manual calibration; rock-solid Syphon/Spout pipeline; v6 Timeline & Tasks close the gap with linear show servers; MiniMAD standalone installation export.
  * *Complaints:* High entry price (~$579+) for hobbyists; Spatial Scanner favors Canon cameras; prior to v6.2 Tasks, linear theatrical cueing felt awkward; still less nimble than Resolume for improvised multi-layer club VJing.

---

### 2.5 Millumin 5

* **Official Documentation:** [Millumin Help Center](https://help.millumin.com/).
* **2025–2026 Versions:** Millumin 4 & **Millumin 5** maintain its reputation as the premier macOS theatrical, dance, museum, and corporate projection-mapping media server.

#### Core Paradigm & Moment-to-Moment Performer Workflow
Millumin bridges **QLab-style theatrical cueing** and **Resolume-style visual mapping**. Its primary interface is the **Dashboard**: a horizontal layer stack intersected by numbered vertical **Columns** (cues). During a live show, an operator steps forward cue-by-cue with `SPACEBAR` (or via Stream Deck + Bitfocus Companion / ETC Eos OSC), while pre-timed sequences run inside embedded **Timelines** dropped into columns.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** Media sits in the **Library** and is dragged into Dashboard Layer × Column cells with clear visual thumbnails. Clicking a column or hitting `SPACEBAR` launches the entire column state; columns support color-coding, labels, cue notes, and bulk editing (*Manage Multiple Columns*).
* **Generator / Shader Parameter Exposure:** Native **ISF shaders** (generators, effects, transitions) expose their JSON-defined named parameters directly in the Layer Properties inspector and can be animated column-to-column or on a timeline.
* **Effects Chains:** Per-layer drag-reorderable **Effects stack** organized into Coloring (LUTs, RGBA/HSLA, levels), Standard, Special, and **ISF Shaders**, with parameter animation across columns and individual bypass/mix controls.
* **Parameter Animation:** Smooth automatic state interpolation between columns, full keyframe **Timelines** (which can target multi-canvas setups or embed as a single Dashboard block), and real-time external modulation via the **Interactions** panel (audio level, MIDI, OSC, DMX, Arduino, Kinect/Leap Motion).
* **Transitions:** Per-column and per-layer **Natural Transitions** (custom fade curves/durations) with a standout **Monitor tab** allowing live manual override of active video, audio, and light transitions during rehearsals or unpredictable live theatre cues.
* **Layer Compositing:** Vertical layer stack per Canvas (supports multiple independent Canvases for different stages/screens), standard blend modes, opacity, masks, and live camera/Syphon/NDI layers.
* **Media Browser / Library Management:** Clean project Library organizing videos, images, audio, live inputs, lights, and ISF shaders, with missing-media management and color-coded organization.
* **Preset / Scene Save, Recall & Morph:** Each Dashboard **Column** is a complete recallable scene snapshot; moving between columns automatically morphs animated parameters, masks, and opacities over the configured transition time.
* **MIDI / OSC Mapping UX:** Centralized **Interactions Panel** maps MIDI, OSC, DMX, keyboard, and sensors to any column or layer property. Features a comprehensive, well-documented **OSC API** heavily used with **Bitfocus Companion / Stream Deck** and theater lighting consoles.
* **Output / Mapping:** Strong built-in **Video Mapping** (grid meshes, Bézier curves, masks), **Soft-Edge (Edge Blending)** with overlap, gamma, and black-level calibration across multi-projector canvases or splitters (e.g., Datapath FX4), plus integrated DMX light/fixture control.
* **Performance UX Details:** Dedicated **Monitor** view for live transition override, live stage preview vs output, LTC/Art-Net timecode sync, instant blackout column/master dimmer.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Effortless column-based theatrical show calling; hits the sweet spot between QLab and enterprise Disguise/Pixera servers at a fraction of the complexity; rock-solid multi-projector soft-edge blending and Stream Deck integration.
  * *Complaints:* **macOS-only**; not built for spontaneous, improvisational club VJing or heavy real-time FX jamming; smaller tutorial ecosystem than Resolume or QLab; price can be steep for hobbyists.

---

### 2.6 Notch (0.9.x / 1.0 / 2026.1)

* **Official Documentation:** [Notch Manual](https://manual.notch.one/).
* **2025–2026 Versions:** **Notch 1.0** and **Notch 2026.1** delivered a ground-up engine rewrite built on **NURA (Notch Unified Rendering Architecture)** with GPU sparse-grid fluid/smoke solvers, upgraded raytracing/path-tracing, tighter After Effects (`[VFX]`) workflows, and expanded Disguise/media-server integration.

#### Core Paradigm & Moment-to-Moment Performer Workflow
Notch is a **real-time 3D motion graphics, particle, and IMAG VFX authoring engine** (Windows Builder) designed specifically for arena/stadium concerts, broadcast, and live events. Rather than performing *inside* Notch Builder directly on show night, artists author complex real-time 3D/particle/camera-FX node graphs in Builder, **expose a tight, curated set of show parameters**, and compile a self-contained **Notch Block (`.dfxdll`)** that runs natively inside media servers (**Disguise/d3**, **Resolume Arena**, **Pixera**, **TouchDesigner**, **Smode**).

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** Inside host media servers, each compiled Notch Block acts like a video clip or layer containing one or more internal **Layers/Scenes** selectable by the show operator.
* **Generator / Shader Parameter Exposure (Best-in-Class Pattern):** Notch sets the gold standard for **separating authoring complexity from operator control**. In Builder, right-clicking any property (float, color, dropdown, text string, video texture input) and selecting **`[RFX] Expose to Block/Standalone`** adds it to the centralized **Exposed Node**, where the author assigns clean operator-facing **Display Labels, Group Names, clamped Min/Max Ranges, and unique IDs**. When loaded into Resolume or Disguise, *only* those curated controls appear.
* **Effects Chains:** Massive node library for real-time 3D lighting, procedural fields, GPU particles, depth-of-field, and **IMAG (live camera) stylization**—chained visually in the node graph with post-FX nodes and blend/mix controls.
* **Parameter Animation:** Internal timeline keyframes, procedural modifiers (oscillators, math, noise, audio-reactive modifiers), and **host-driven automation** (all Exposed Parameters can be keyframed on Disguise's timeline or driven by Resolume's FFT/BPM/MIDI).
* **Transitions:** Handled either inside the Block (via exposed Layer Blend/Morph parameters) or at the host media server layer level.
* **Layer Compositing:** Full 3D scene graph + 2D Render Layer compositing, alpha/depth outputs, and live video texture inputs (IMAG cameras fed into 3D particle emitters or stylization chains).
* **Media Browser / Library Management:** Self-contained `.dfx` projects and packaged `.dfxdll` Blocks bundling 3D meshes, textures, and shaders cleanly for touring servers.
* **Preset / Scene Save, Recall & Morph:** Multi-layer Blocks allow switching looks cleanly; host media servers store Exposed Parameter presets/cues; **Remote Network Editing** lets an artist tweak a live Block on a media server directly from Builder.
* **MIDI / OSC Mapping UX:** Native MIDI/OSC/Art-Net/PSN (PosiStageNet) support inside standalone/Builder, and seamless pass-through via host media servers (lighting desks like GrandMA3 commonly drive Notch Exposed Parameters via Art-Net/sACN or OSC).
* **Output / Mapping:** Mapping, multi-projector warping, and LED wall pixel routing are typically delegated to the host media server (Disguise, Pixera, Resolume), though Notch supports virtual cameras and 3D stage pre-visualization.
* **Performance UX Details:** Engineered for zero-frame-drop arena reliability; real-time GPU profiling stats inside Builder; live camera (IMAG) low-latency processing.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Cinema-4D-grade real-time 3D/particles and IMAG effects without coding GLSL; unmatched speed-to-result for touring deadlines; seamless Disguise/media-server Block workflow; responsive dev team.
  * *Complaints:* **Brutal subscription pricing** (Indie/Pro/RFX tiers) + physical USB dongle requirements on Pro/Playback; expensive friction converting indie/trial projects to RFX; the recognizable **"Notch Look"** when relying on stock nodes; less open-ended than TouchDesigner.

---

### 2.7 Synesthesia (`synesthesia.live`) — *Highly Relevant Shader-Instrument Reference*

* **Official Documentation:** [Synesthesia Docs](https://docs.synesthesia.live/), [Control Panel](https://app.synesthesia.live/docs/manual/controls.html), [Presets & Playlists](https://app.synesthesia.live/docs/manual/presets.html), [SSF Spec](https://app.synesthesia.live/docs/ssf/index.html), [Audio Uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html), [Standard Uniforms](https://app.synesthesia.live/docs/ssf/standard_uniforms.html), [OSC](https://app.synesthesia.live/docs/manual/osc.html), [MIDI/OSC FAQ](https://app.synesthesia.live/docs/faq/midi_osc.html).

#### Core Paradigm & Moment-to-Moment Performer Workflow
Synesthesia is the premier **psychedelic GLSL shader instrument** for live VJs—and arguably the single closest architectural role model for a Python + `moderngl` shader-first VJ engine. Rather than mixing dozens of pre-rendered video loops, a performer "plays" real-time audio-reactive GLSL **Scenes** (`.synScene`) like a musical synthesizer: launching scenes or saved **Presets/FavSlots** from playlists, twisting a **small bank of named Scene Controls** (sliders, knobs, XY pads, bangs), layering master **Meta Controls** (brightness, saturation, palette shift, mirror, strobe, global audio reactivity), mixing optional video/webcam/logo overlays through the **Media / FX Mixer**, and crossfading or running **Autopilot** hands-free.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** Scenes appear in a clean visual **Scene Library** and curated **Playlists** with `thumbnail.png` previews, search/filtering, and **FavSlots** (one-click or MIDI-pad slots that jump directly to a specific *Scene + Preset* pairing).
* **Generator / Shader Parameter Exposure (Gold Standard for Shader Engines):** Every `.synScene` bundle defines a **`scene.json`** `"CONTROLS"` array mapping curated, artist-named parameters straight to GLSL uniforms. Supported widget types include `slider`, `knob`, `toggle`, `bang` (momentary pulse), `xy` (2D dual-axis pad), `dropdown`, `color`, plus **`speed` / `smooth` variants** (`slider smooth`, `knob speed`, `xy speed`) so the host automatically integrates velocity/smoothing without boilerplate shader math! Typically a scene exposes **4 to 12 thoughtfully grouped controls**, keeping live performance playable rather than overwhelming.
* **Effects Chains:** Separated cleanly into **Scene-internal passes** (`PASSES` in `scene.json`), **Media / FX Mixer** (blending imported HAP/WebM/GIF/image/webcam/Spout/Syphon media into the shader with color correction and blend modes), and global **Meta Controls** (post-processing effects like Mirror, Kaleidoscope/Invert, Color Grading, Strobe, Alpha/Color Limiting applied uniformly across *every* scene).
* **Parameter Animation:** Synesthesia's secret weapon is its **automatic, zero-config Audio Analysis Engine**. Rather than forcing VJs to wire raw FFT bins manually, Synesthesia pre-computes rich musical features every frame and injects them as standard GLSL uniforms: `syn_BassLevel`, `syn_MidLevel`, `syn_HighLevel`, `syn_Level`, onset triggers (`syn_BassHits`, `syn_MidHits`, `syn_HighHits`), beat-synced oscillators (`syn_BPMSin`, `syn_BPMSin2`, `syn_BPMSin4`, `syn_BPMTri`, `syn_BPMConfidence`), audio-driven integrated time (`syn_Time`, `syn_BassTime`, `syn_HighTime`), and data textures (`syn_Spectrum` with raw/juiced/smooth FFT + waveform rows, `syn_LevelTrail` history buffer). Optional `script.js` (`setup()` / `update(dt)`) handles stateful CPU animations.
* **Transitions:** Smooth **Crossfade / Transition** blending between scenes and presets (with configurable transition duration and optional `HARD_TRANSITIONS` flag per scene), plus audio-reactive auto-transitions in Playlists.
* **Layer Compositing:** Primarily single-active-scene (with dual-scene overlap during transitions) plus **Media Layer compositing** (injecting logos, clips, or webcam feeds *into* the shader via `syn_UserImage` or overlaying/blending them in the FX Mixer). Pro VJs needing 6-layer compositing pipe Synesthesia via **Spout/Syphon** into Resolume or VDMX.
* **Media Browser / Library Management:** Built-in **Scene Marketplace**, local Scene Library, **Media Library** (HAP, GIF, WebM, MP4, images, live cameras, Syphon/Spout), **Playlists** for show organization, and **FavSlots** for rapid favorites.
* **Preset / Scene Save, Recall & Morph:** **Presets** capture exact Scene Controls, Meta Controls, and active Media state. Recalling presets smoothly interpolates/morphs parameters or switches scenes via **FavSlots** and **Playlists**.
* **MIDI / OSC Mapping UX:** Click-to-learn MIDI mapping for any Scene Control, Meta Control, FavSlot, or Playlist trigger (heavily paired with **Akai APC40 / APC40 MkII**). Supports **bi-directional OSC** (e.g., TouchOSC tablets) AND **broadcasts Synesthesia's own audio-analysis variables over OSC** so external apps (TouchDesigner/Resolume/lights) can ride Synesthesia's beat/hit detection!
* **Output / Mapping:** Dedicated fullscreen output window with Pro resolution/aspect-ratio scaling (for LED walls/projectors) and zero-copy **Spout (Windows) / Syphon (macOS)** output. No native multi-surface projection warping—intended to feed Resolume/MadMapper when mapping is required.
* **Performance UX Details:** Top-right live **Preview** window; built-in **Autopilot** (countdown timers + beat/energy-aware automatic scene/playlist switching); built-in **live-coding GLSL editor** with hot-reload; Shadertoy URL auto-importer and ISF importer.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Effortless, musical **out-of-the-box audio reactivity**; gorgeous festival-grade GLSL visuals without coding; feels like playing a synth; rock-solid Spout/Syphon integration into Resolume/VDMX.
  * *Complaints:* Recognizable **"stock Synesthesia look"** if scenes aren't customized or layered with personal media; MIDI/TouchOSC advanced template setup can feel fiddly; locking full OSC and custom resolution scaling to the **Pro tier** frustrates budget users; deep geometry edits still require GLSL coding.

---

### 2.8 Magic Music Visuals

* **Official Documentation:** [Magic Music Visuals User's Guide](https://magicmusicvisuals.com/documentation/).

#### Core Paradigm & Moment-to-Moment Performer Workflow
Magic Music Visuals (MMV) is a **lightweight, node-based audio-reactive visual synthesizer** (Windows & macOS) occupying the sweet spot between Resolume and TouchDesigner. Users build directed left-to-right **Module graphs** inside tabbed **Scenes** (connecting generators, images, 3D models, ISF/GLSL shaders, and transform/FX modules to the `Magic` output node) and link any module parameter directly to audio frequency bands, stems, or MIDI with modifiers (Scale, Smooth, Peak, Trigger, Sine, Ramp).

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** Scenes live as horizontal tabs and are sequenced in the **Playlist Window** (manual hotkeys, MIDI/OSC triggers, or timed/beat-synced progression).
* **Generator / Shader Parameter Exposure:** Built-in modules and dropped-in **ISF shaders** expose compact inline numeric fields directly on each graph node—minimalist, functional, though less "tactile hardware knob" than Synesthesia.
* **Effects Chains:** Constructed visually by chaining Effect/ISF modules left-to-right; dry/wet and bypass are controlled via `Mix` modules or module power toggles.
* **Parameter Animation:** Standout **per-parameter modifier stacks**! Any numeric box can link to an audio source (multiple independent audio inputs / **multi-stem audio files** for music videos, configurable FFT frequency ranges or volume/pitch) and stack chainable math modifiers: `Smooth`, `Peak`, `Scale`, `Offset`, `Invert`, `Wrap`, `Sine`, `Triangle`, `Ramp`, `Expression`—delivering ultra-clean audio reactivity without writing code.
* **Transitions:** Built-in **Playlist Transitions** (dissolves, wipes, zooms, ISF transitions) with configurable durations when switching Playlist entries.
* **Layer Compositing:** `MultiMix`, `Blend`, `AlphaMask`, and multi-input `Magic` output node support standard blend modes and 3D camera layering.
* **Media Browser / Library Management:** Relies on OS folder paths + an **Additional Module Folders** setting where dropping `.fs` ISF files immediately populates the module menu. No visual thumbnail media library browser.
* **Preset / Scene Save, Recall & Morph:** Scenes act as full presets; scenes can even be embedded *inside* other scenes as sub-modules (`SceneModule`) for reusable looks or global post-FX chains.
* **MIDI / OSC Mapping UX:** Simple **MIDI Learn** and **OSC** input binding per parameter or Playlist entry (linking MIDI note velocity, CC, or note-on triggers with modifier stacks). Limited bi-directional MIDI LED feedback compared to Resolume/VDMX.
* **Output / Mapping:** Multi-display fullscreen output, `CornerPin` / `Keystone` / `Mask` modules, plus Spout/Syphon output. Basic compared to MadMapper or Resolume Arena.
* **Performance UX Details:** Preview window with fullscreen toggle, Playlist **Auto-Advance** (autopilot), keyboard/MIDI blackout scene, and offline MP4/PNG movie export for music videos.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Ideal **no-code/low-code sweet spot**; unmatched multi-stem & FFT audio reactivity via modifier stacks; generous full-featured demo; native **ISF** folder drop-in extensibility; low cost & lightweight footprint.
  * *Complaints:* Managing **global FX or overrides across multiple Playlist scenes** feels clunky without workarounds (embedding scenes inside a master scene); utilitarian UI lacks visual clip thumbnails and modern dashboard ergonomics; smaller YouTube tutorial ecosystem.

---

### 2.9 HeavyM 2

* **Official Documentation:** [HeavyM Help Center](https://help.heavym.net/).

#### Core Paradigm & Moment-to-Moment Performer Workflow
HeavyM 2 is a **geometry-first projection-mapping tool** built for speed and visual accessibility. Performers draw vector **Faces**, **Masks**, and **Media Players** directly over a live projector canvas, organize shapes into **Groups** and swappable **Maps**, apply procedural **Group Effects** and **Shaders** that automatically respect polygon boundaries/edges, and perform by triggering **Sequences** (grid cues), running **Autopilot**, or programming a **Timeline** (Pro).

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals:** The **Sequencer** presents a grid of **Sequences** (each storing maps, group effects, media, and transitions). HeavyM distinguishes cleanly between the **Edited Sequence** (what you are tweaking in the UI) and the **Played Sequence** (what is live on the projector)—enabling safe blind editing mid-show!
* **Generator / Shader Parameter Exposure:** Built-in **Group Effects** (border chase, snake, repeat, kaleidoscope, particle fills) and **Generative/Post-Processing GLSL Shaders** expose clean, minimal sliders, toggles, and color palettes tailored to polygon geometry.
* **Effects Chains:** Group Effects and Post-Processing Shaders can be applied at **Media Player**, **Group**, **Sequence**, or **Master Output** scope with blend modes and opacity.
* **Parameter Animation:** Built-in **LFOs** on group effects, **Tempo module** (manual BPM, **TAP**, downbeat **Resync**, **Ableton Link**), live **Audio Analysis** (mic/system audio frequency bands), and a Pro **Timeline** synced to a master audio track.
* **Transitions:** Per-sequence **Transitions** with configurable duration, starting angle, and easing curves.
* **Layer Compositing:** Shapes live inside layered **Groups** with blend modes, opacity, **Masks**, and Media Players clipped cleanly to vector Faces.
* **Media Browser / Library Management:** Built-in Sources panel (videos, images, Spout/Syphon/NDI, Generative Shaders, and newer **AI content generation** tools) with visual thumbnails.
* **Preset / Scene Save, Recall & Morph:** Sequences function as complete mapping presets with smooth transitions between them; **Maps** allow swapping underlying shape layouts across sequences.
* **MIDI / OSC Mapping UX:** MIDI and OSC Learn for triggering sequences, controlling group effect intensity, tempo tap/resync, and master controls.
* **Output / Mapping:** Vector Faces, Bézier/Mesh **Warping**, **Edge Blending**, calibration test cards, and multi-display Pro output; frequently paired via Spout/Syphon with Resolume for club stages.
* **Performance UX Details:** **Edited vs. Played** blind-edit separation, **Autopilot** timed sequence cycling, instant blackout/mute controls.
* **User Praise & Complaints (2025–2026):**
  * *Praise:* Fastest, most "fun" learning curve for drawing physical polygon maps; geometry-aware built-in shaders/effects look great immediately without external video loops; solid Spout/Syphon companion to Resolume.
  * *Complaints:* Heavily watermarked free trial/lower tiers and steep Pro pricing for hobbyists; poor fit for **LED pixel mapping** or deep multi-layer video mixing; stock geometric looks repeat quickly once mastered, pushing pros toward MadMapper or Resolume Arena.

---

### 2.10 CoGe (Discontinued Legacy)

* **Official Status:** [Imimot Blog — CoGe Discontinuation Announcement (Oct 2022)](https://imimot.com/) (Imimot now focuses on active macOS tools **Mitti** and **Vezér**).

#### Core Paradigm, Historical Significance & Lessons Learned
CoGe VJ (macOS only, incompatible beyond **macOS 10.14 Mojave**) was a beloved **semi-modular VJ mixer** notable for its **ClipSynth** clip-mashing engine, floating/docked workspace modules, **Layer Groups**, stacked FX chains, and deep modulation routing (Audio Analysis, LFOs, MIDI Clock, bi-directional MIDI/OSC). Its architecture relied heavily on Apple's **Quartz Composer (`.qtz`)** and QuickTime before adding **ISF** and **Vuo** support too late to save the codebase when Apple deprecated Quartz Composer and 32-bit/QuickTime APIs.

* **Key Architectural Lesson for Modern VJ Engines:** Never tie your plugin/shader ecosystem to a single vendor's deprecated proprietary runtime (like Quartz Composer); open, text-based GLSL standards (**ISF** + clean JSON metadata) survive OS transitions intact.

---

### 2.11 Modul8 (Legacy)

* **Official Site:** [GarageCUBE Modul8](https://www.garagecube.com/modul8/) (same creators as MadMapper).

#### Core Paradigm, Historical Significance & Lessons Learned
Originating in the mid-2000s for macOS, Modul8 pioneered the ultra-fast **Photoshop-style Layer Stack + MediaSet grid** for improvisational club VJing. Notable innovations still worth studying:
* **MediaSet & Dual-Group Layers:** Instant clip firing from numbered MediaSet banks onto independently transformable 2D/3D layers with built-in OpenGL filters, particle/color FX, and audio/BPM reactivity.
* **Python-Scriptable Modules & Online Library:** Modul8 embedded a **Python scripting engine** letting VJs build custom UI panels ("Modules" for generative visuals, OSC routers, LFOs, macros) and share them via an in-app community library.
* **Logical Record:** Instead of recording gigabytes of compressed video frames, **Logical Record** captured *performer actions and parameter automation* as lightweight metadata scripts—enabling lossless replay, editing, and remixing of live sets.
* **MIDI/OSC:** Direct MIDI mapping (including bespoke hardware like FaderFox's `micromodul8`) and OSC via Modules.
* **Legacy Hurdles (2025–2026):** Aging OpenGL codebase (v3 runs on Apple Silicon via Rosetta); overshadowed by GarageCUBE's own **MadMapper** and modern tools like Resolume/VDMX6/TouchDesigner.

---

### 2.12 Lumen (Paracosm)

* **Official Site:** [Paracosm Lumen](https://lumen-app.com/).

#### Core Paradigm & Moment-to-Moment Performer Workflow
Lumen (macOS only) is a **semi-modular, analog-style video synthesizer** inspired by classic hardware like the Sandin Image Processor and LZX Eurorack systems. It embodies a brilliant two-sided UI pattern:
* **Front Panel (Playable Performance Mode):** Clean, tactile **knobs, switches, and colorizers** for visual oscillators, waveshapers, keyers, and external camera/video inputs—designed to be played blind with a MIDI controller.
* **Back Panel (Modular Patch Bay):** Flipping to the back exposes virtual **patch cables** routing oscillator/LFO/video signals into modulation inputs, feedback loops, and mixers.

#### Feature-by-Feature Breakdown
* **Picking / Launching Visuals & Presets:** Organized by **Patches** and an **XY Snapshot Morphing Pad**—performers save 4+ distinct visual states onto corners/points of a 2D pad and smoothly morph continuously between organic analog textures in real time!
* **Feedback & External Sources:** Ingests live USB webcams, video files, and **Syphon Input** to create self-oscillating analog-style **video feedback loops** (infinite recursion, smearing, color cycling) without expensive fragile CRT/hardware rigs.
* **MIDI & Output:** Full MIDI Learn across every knob/button (and OSC via OSCulator); native **Syphon Output** makes Lumen a favorite organic "analog texture generator & live-camera texturizer" piped straight into VDMX or Resolume.
* **User Praise & Complaints:**
  * *Praise:* Democratizes expensive analog video synthesis with software conveniences (instant recall, undo, XY morphing, zero hardware fragility); beloved Syphon companion.
  * *Complaints:* Strictly macOS-only; rigid aspect/resolution quirks in older workflows; extreme feedback patches can spike GPU load; modular patching curve for video editors.

---

## 3. Shader & Generator Ecosystems and AGPL-3.0 Licensing

For an open-source **AGPL-3.0** Python + `moderngl` VJ engine, choosing the right shader format and legally compatible bundled libraries is critical. **AGPL-3.0 is a strong copyleft license that explicitly permits commercial use and redistribution** (provided full source code is made available under AGPL-3.0). Therefore, any bundled shader or library **must permit commercial use and AGPL-compatible redistribution**—which immediately disqualifies Non-Commercial (`CC BY-NC-*`) and restrictive source-available licenses.

### 3.1 Interactive Shader Format (ISF) — *Recommended Primary Standard*

* **Official Spec & Docs:** [isf.video](https://isf.video/), [Quickstart](https://docs.isf.video/quickstart), [JSON Reference](https://docs.isf.video/ref_json), [Variables](https://docs.isf.video/ref_variables), [Multipass](https://docs.isf.video/ref_multipass), [Audio](https://docs.isf.video/ref_audio), [Integrations](https://docs.isf.video/integrations), [GitHub `Vidvox/isf`](https://github.com/Vidvox/isf), [GitHub `Vidvox/ISF-Files`](https://github.com/Vidvox/ISF-Files).
* **How the Spec Works:** Every ISF shader is a plain GLSL fragment shader (`.fs`, with optional `.vs`) starting with a JSON metadata block inside a C-style comment `/*{ ... }*/`:
  * **`INPUTS`**: Declares typed, named UI controls automatically rendered by the host: `float` (with `MIN`/`MAX`/`DEFAULT`), `bool`, `long` (enum dropdown via `VALUES`/`LABELS`), `event` (momentary trigger), `color` (RGBA `vec4`), `point2D` (XY pad), `image` (texture input — presence of `inputImage` classifies a shader as a **Filter/FX**, `startImage` + `endImage` + `progress` as a **Transition**, and neither as a **Generator**), **`audio`** (raw PCM waveform packed into a 2D texture), and **`audioFFT`** (frequency spectrum packed into a 2D texture).
  * **`PASSES`**: Defines multi-pass FBO rendering (`TARGET` buffer name, `WIDTH`/`HEIGHT` expressions like `"$WIDTH/2.0"`, **`PERSISTENT`: `true`** for frame-to-frame feedback/cellular automata/motion blur, and **`FLOAT`: `true`** for 32-bit floating-point buffers).
  * **`IMPORTED`**: Loads static image/LUT textures by path before execution.
  * **Built-in Uniforms & Macros:** `TIME`, `TIMEDELTA`, `RENDERSIZE`, `PASSINDEX`, `FRAMEINDEX`, `DATE`, `isf_FragNormCoord`, and target-agnostic sampling macros **`IMG_NORM_PIXEL(tex, normCoord)`**, `IMG_PIXEL(tex, pxCoord)`, and `IMG_THIS_PIXEL(tex)`—which map cleanly to `moderngl` `sampler2D` uniforms via a small `#define` preamble!
* **App Adoption Matrix:**
  * *Full Native Support:* **VDMX6** (creator), **Resolume Arena 7 / Wire**, **Millumin 5**, **Magic Music Visuals**.
  * *Direct/Partial Support:* **MadMapper 5/6** (expanded ISF render passes in v6), **Synesthesia** (auto-importer to SSF), **TouchDesigner** (community parsers), **Vuo 2.0+**, **CoGe** (legacy).
* **Licensing Nuance (`Vidvox/ISF-Files` vs. `editor.isf.video`):**
  * **`github.com/Vidvox/ISF-Files` (~200+ official generators, filters & transitions): MIT License** — **100% safe and legal to ship inside an AGPL-3.0 app** (preserve MIT copyright notices).
  * **`editor.isf.video` (community upload library): Mixed/Variable Licenses.** Many user uploads are ports from Shadertoy carrying `CC BY-NC-SA` or unstated copyright headers. **Do NOT bulk-scrape or bundle `editor.isf.video` shaders** without auditing each individual file's header license; instead, let *end users* import their own local `.fs` files into a user shaders folder!

### 3.2 Synesthesia Shader Format (SSF / `.synScene`)

* **Official Docs:** [SSF Overview](https://app.synesthesia.live/docs/ssf/index.html), [Audio Uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html).
* **How the Format Works:** Folder-based `.synScene` packages containing `scene.json` (metadata, `CONTROLS` with rich widgets like `slider`, `knob`, `toggle`, `bang`, `xy`, `dropdown`, `color`, plus `smooth`/`speed` auto-integrators, `PASSES`, `IMAGES`), `main.glsl` (fragment shader consuming ~40+ pre-analyzed musical uniforms like `syn_BassLevel`, `syn_BassHits`, `syn_HighHits`, `syn_BPMSin4`, `syn_Spectrum`, `syn_LevelTrail`), optional `script.js` (`setup()`/`update(dt)`), and `thumbnail.png`.
* **Licensing & Takeaway:** Synesthesia itself and its bundled/marketplace scenes are **proprietary commercial software/assets**—you **cannot** legally ship Synesthesia's stock `.synScene` files in an AGPL-3.0 app. However, **you CAN adopt the best ideas of SSF** in your own open JSON/ISF-extended metadata schema: specifically, pre-computed musical uniforms (bass/mid/high energy, onset hits, audio-integrated time, BPM sine/tri waves, spectrum + trail texture) and rich control widget hints (`knob`, `xy`, `bang`, `smooth`, `speed`).

### 3.3 Shadertoy Import & The `CC BY-NC-SA 3.0` Trap

* **Default License:** Unless an author explicitly states a permissive license inside the shader's top comments, every Shadertoy shader defaults to **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (`CC BY-NC-SA 3.0`)**.
* **AGPL-3.0 Incompatibility:** Because `CC BY-NC-SA 3.0` prohibits commercial use while AGPL-3.0 explicitly grants commercial freedom to all recipients, **default Shadertoy shaders CANNOT legally be bundled in an AGPL-3.0 repository or distribution.**
* **The "Derivative Work" Trap:** Running a Shadertoy shader through an automated converter (`iTime`/`iResolution`/`mainImage` → ISF/GLSL) creates a *derivative work* that remains bound by `CC BY-NC-SA 3.0`—conversion does **not** strip the license.
* **Safe Strategy:**
  1. **Never bundle** default Shadertoy shaders unless the author explicitly placed an **MIT / BSD / CC0 / Unlicense** header in the source (and document the exact URL + author license header in a `THIRD_PARTY_LICENSES` file).
  2. **Optionally offer a user-initiated local import/wrapper** (similar to Synesthesia or ISF Editor) with a clear UI notice reminding users to respect each Shadertoy author's license when loading shaders locally for personal use.

### 3.4 TouchDesigner Palette & `.tox` Components

* **Official Terms:** Governed by [Derivative's EULA](https://derivative.ca/). Built-in Palette `.tox` components (KantanMapper, Stoner, CamSchnappr, image filters, UI widgets) belong to Derivative and are licensed strictly for use *within* TouchDesigner—they **cannot** be extracted or redistributed in an external open-source app.
* **Community `.tox` / Olib (`olib.amb-service.net`):** Individually licensed by third-party authors (MIT, GPL, CC, or proprietary/Privacy-locked). Also structurally tied to TouchDesigner's operator runtime, though MIT-licensed GLSL TOP snippets inside open GitHub repos can be referenced if properly attributed.

### 3.5 AGPL-3.0 Compatibility Matrix: Shader & Generator Libraries You CAN (and CANNOT) Ship

| Library / Source | Repository / URL | License | Ship in AGPL-3.0? | Best Role in a Python + `moderngl` VJ Engine |
| :--- | :--- | :--- | :---: | :--- |
| **Vidvox ISF-Files** (~200+ shaders) | [github.com/Vidvox/ISF-Files](https://github.com/Vidvox/ISF-Files) | **MIT** | **YES** | Turnkey starter pack of generators, FX filters, and transitions |
| **gl-transitions** (~70+ transitions) | [github.com/gl-transitions/gl-transitions](https://github.com/gl-transitions/gl-transitions) | **MIT** (2–3 BSD-2/3 files) | **YES** | Pro-grade GPU crossfader & deck transition library |
| **webgl-noise / glsl-noise** | [github.com/stegu/webgl-noise](https://github.com/stegu/webgl-noise) / `hughsk/glsl-noise` | **MIT** | **YES** | Shared `#include` simplex/Perlin/Worley noise library for built-in generators |
| **glsl-blend** | [github.com/jamieowen/glsl-blend](https://github.com/jamieowen/glsl-blend) | **MIT** | **YES** | Accurate Photoshop-style layer compositing blend modes (25+ modes) |
| **glsl-easings** | [github.com/glslify/glsl-easings](https://github.com/glslify/glsl-easings) | **MIT** | **YES** | Penner easing curves for transitions, envelopes, and preset morphs |
| **Three.js Post-Processing Shaders** | [github.com/mrdoob/three.js](https://github.com/mrdoob/three.js) (`examples/jsm/shaders`) | **MIT** | **YES** | Bloom, RGB shift, film grain, halftone, Sobel, kaleido, vignette FX |
| **Butterchurn Presets** | [github.com/jberg/butterchurn-presets](https://github.com/jberg/butterchurn-presets) | **MIT** | **YES** | Reference/inspiration (or MilkDrop-style preset pack if implementing MilkDrop) |
| **Hydra Synth** | [github.com/hydra-synth/hydra-synth](https://github.com/hydra-synth/hydra-synth) | **AGPL-3.0** | **YES** | Same license! Great reference for chainable GLSL video synth ops |
| **Inigo Quilez MIT-header snippets** | `iquilezles.org` / `tuxalin/procedural-tileable-shaders` | **MIT** (verify per file!) | **YES (audited)** | SDF primitives, cosine color palettes, domain warping |
| **projectM (`libprojectM`)** | [github.com/projectM-visualizer/projectm](https://github.com/projectM-visualizer/projectm) | **LGPL-2.1** (presets mixed provenance) | **CAUTION** | Engine is LGPL-2.1 (compatible), but bulk community presets have murky provenance—audit before bundling |
| **LYGIA Shader Library** | [github.com/patriciogonzalezvivo/lygia](https://github.com/patriciogonzalezvivo/lygia) | **Prosperity / Patron Dual** | **NO (avoid)** | Prosperity restricts commercial use → **conflicts with AGPL-3.0** unless commercial/patron agreement covers downstream users |
| **Shadertoy (default)** | `shadertoy.com` | **CC BY-NC-SA 3.0** | **NO** | Non-Commercial clause is strictly incompatible with AGPL-3.0 |
| **editor.isf.video (community)** | `editor.isf.video` | **Mixed (many CC-NC)** | **NO (bulk)** | Only ship files with explicit MIT/BSD/CC0/Apache-2.0 headers |
| **Synesthesia / TD Palette** | `synesthesia.live` / `derivative.ca` | **Proprietary EULA** | **NO** | Study UX patterns only; never redistribute assets |

*(Note: Apache-2.0, MIT, BSD-2/3, ISC, CC0, Unlicense, LGPL-2.1/3.0, and GPL-3.0 are all legally compatible for inclusion inside an AGPL-3.0 application, provided original copyright/license notices are preserved.)*

---

## 4. Top 15 Pro Capabilities & UX Patterns for a Python + `moderngl` + Browser VJ Engine

To make a compact Python + `moderngl` engine controlled via a browser UI feel genuinely **"pro"** (play-all-night reliable, tactile, musical, and zero-friction) rather than a developer tech demo, adopt these 15 patterns—ranked by live-performance impact:

### 1. Native ISF Parser + Curated Named Control Panels (4–12 Controls per Shader)
* **How the Best Do It:** **VDMX6** (ISF `INPUTS` auto-UI), **Synesthesia** (`scene.json` `CONTROLS` with `knob`/`slider`/`xy`/`bang`/`color` + `smooth`/`speed` modes), and **Notch 1.0** (`Exposed Node` with grouped labels and clamped ranges).
* **How to Build It:** Implement an ISF `.fs` parser in Python (`/*{...}*/` JSON header + `moderngl` GLSL preamble defining `IMG_NORM_PIXEL`, `TIME`, `RENDERSIZE`, `PASSINDEX`) with optional extended widget metadata (`widget: "knob" | "xy" | "bang"`, `group`, `smooth`). Auto-render clean, dark, high-contrast browser controls—never dump 40 raw uniforms onto a performer. Ship the MIT-licensed `Vidvox/ISF-Files` pack out of the box.

### 2. Zero-Config, Musical Audio-Reactivity Engine (Pre-Cooked Uniforms + Per-Parameter FFT Binding)
* **How the Best Do It:** **Synesthesia** (automatic `syn_BassLevel`, `syn_BassHits`, `syn_HighHits`, `syn_Time`, `syn_BPMSin4`, `syn_Spectrum` + `syn_LevelTrail` textures) combined with **Resolume Arena 7** & **Magic Music Visuals** (click any parameter → bind to selectable FFT band with Gain, Falloff/Smoothing, Range Min/Max, and Invert).
* **How to Build It:** Run a low-latency Python audio thread (`sounddevice` / WASAPI loopback / PortAudio) computing normalized sub-bass/bass/mid/high RMS, spectral flux onset "hits" with decay envelopes, BPM tap/sync + Ableton Link (`aalink`), audio-integrated time, and a 2-row 1D/2D `moderngl` FFT + waveform texture every frame. Expose these automatically to *all* shaders AND let the browser UI bind any slider to any audio channel with a 1-click modulation drawer.

### 3. Dual-Monitor Preview vs. Program + Cue/Blind-Edit Workflow
* **How the Best Do It:** **Resolume Arena 7** (Preview Monitor + Output Monitor; click clip title to audition/tweak before pushing live) and **HeavyM 2** (explicit **Edited vs. Played** sequence separation).
* **How to Build It:** Render live low-res (e.g., 480×270 @ 20–30 fps JPEG/WebP over WebSocket or WebRTC) **Program** and **Preview (Deck A / Deck B / Cue)** thumbnails directly in the top bar of the browser UI. Let performers load and tweak a shader/preset on the inactive deck (Preview) before fading it into Program.

### 4. Dual-Deck A/B Crossfader with GPU `gl-transitions` & Selectable Curves
* **How the Best Do It:** **VDMX6** (`Two Channel Mixer` with ISF transition shaders and auto-fade) and **Resolume Arena 7** (A/B Crossfader + per-layer Auto Transition 0–10 s + Crossfader Phase).
* **How to Build It:** Render Deck A and Deck B into two `moderngl` FBOs and composite them through a dedicated **Transition Pass** shipping the MIT-licensed **`gl-transitions`** library (Dissolve, Luma Wipe, Glitch, Zoom, Radial, Swirl) plus linear/equal-power/S-curve crossfade math and a 1-click **Auto-Fade (e.g., 1, 2, 4 beats or 0.5–8 s)** button.

### 5. Reorderable Post-FX Chains with Per-Effect Wet/Dry, Bypass & Chain Presets
* **How the Best Do It:** **Resolume Arena 7** (drag-reorderable Clip/Layer/Composition FX stacks with Opacity dry/wet, bypass, and Effect Presets) and **VDMX6** (Layer & Master FX chains with saved chain presets).
* **How to Build It:** Use ping-pong `moderngl` FBOs so any number of ISF Filter shaders (Bloom, Kaleidoscope, RGB Shift, Feedback Trails, Color Grade, Strobe) run in user-defined order per Deck and on Master Output. Every FX slot MUST have: **drag handle (reorder)**, **power toggle (bypass)**, **Wet/Dry mix slider** (computed uniformly in the wrapper shader!), and **Save/Load FX Chain Preset**.

### 6. Universal Global "Meta Controls" & 8-Knob Macro Dashboard
* **How the Best Do It:** **Synesthesia** (**Meta Controls**—global Master Brightness, Contrast, Saturation, Hue/Palette Shift, Mirror, Invert, Global Speed, Global Audio Reactivity applied consistently across *all* scenes) and **Resolume Arena 7** (8-dial **Dashboard** mapping multiple disparate parameters with custom min/max curves to one knob).
* **How to Build It:** Pin a permanent **Master Meta Bar** (Brightness, Contrast, Hue Shift, Saturation, Speed Multiplier, Audio Reactivity Gain, Mirror/Symmetry, Strobe) + an **8-Knob Macro Bank** in the browser UI so a performer's MIDI knobs always work identically no matter which shader is active.

### 7. Instant Blackout / Panic, Tap-Tempo & Strobe/Flash Performance Strip
* **How the Best Do It:** **Resolume Arena 7** (one-key Spacebar/MIDI Blackout/Bypass, Tap BPM + Resync downbeat) and **HeavyM 2** (TAP + Resync).
* **How to Build It:** Place an always-visible top performance bar with: **BLACKOUT (Panic)** (instant hard cut to black + optional fade-to-black, bound to `Esc`/Spacebar), **FREEZE FRAME**, **BPM Readout + TAP + RESYNC + ÷2 / ×2**, and momentary **FLASH / INVERT / STROBE** hold-buttons (WebSocket `pointerdown`/`pointerup` for zero-latency rhythmic punching).

### 8. Unified Modulation Bus (BPM LFOs, Custom Envelopes & Smoothing)
* **How the Best Do It:** **VDMX6** (universal right-click data-source assignment with scaling/smoothing) and **Resolume 7.23** (reusable **Parameter Animation Presets** with BPM-synced envelopes).
* **How to Build It:** Run a lightweight Python modulation engine ticking every frame before uniform upload: assign any parameter to **Manual**, **BPM LFO** (Sine/Tri/Saw/Square/Noise at `1/4` to `16` beats), **Audio FFT Band/Hit**, or **Macro Knob**, with per-link **Min/Max clamp, Invert, and exponential Smoothing/Lag**. Display live animated value rings directly on browser knobs so the performer *sees* modulation in real time.

### 9. Live Visual Scene/Clip Grid with Real Thumbnails, Tags, Favorites & Playlists
* **How the Best Do It:** **Resolume Arena 7** (Decks + clip thumbnails + Files Panel Favorites) and **Synesthesia** (Scene Library with thumbnails + **Playlists** + **FavSlots**).
* **How to Build It:** Auto-render a cached GPU thumbnail (`thumbnail.jpg` or short animated WebP preview) for every shader/clip on first load. Provide a dark-mode responsive grid with **instant search**, **mood/color/style tags**, **Star/Favorites**, **Decks/Playlists**, and a pinned row of **8–16 FavSlots** (one-click `Scene + Preset` launchers).

### 10. Preset Save, Instant Recall & Timed / XY Snapshot Morphing
* **How the Best Do It:** **Lumen** (**XY Snapshot Morphing Pad** interpolating smoothly between 4 patch states), **VDMX6** (timed Workspace/FX Preset fades), and **Synesthesia** (Presets storing Scene + Meta + Media states).
* **How to Build It:** Serialize complete Deck/Scene/FX state as clean JSON presets. Add a **Morph Time slider** (0 s instant snap → 8 s smooth cosine/easing interpolation of all float/color uniforms) AND an optional **2D XY Morph Pad** assigning 4 presets to the 4 corners—turning 1 shader into an endlessly evolving live instrument.

### 11. Frictionless MIDI Learn + Bi-Directional LED Feedback + OSC / OSCQuery
* **How the Best Do It:** **Resolume Arena 7** (color-coded Shortcuts overlay + **MIDI LED color feedback** lookup tables for APC40mk2/Launchpad), **VDMX6** (right-click Detect + **OSCQuery**), and **MadMapper 6** (Control List + OSCQuery).
* **How to Build It:** Support **both** browser **WebMIDI API** (zero-install controller plug-in from any laptop/tablet!) and Python backend **`mido`/`python-rtmidi` + `python-osc` (+ OSCQuery JSON endpoint)**. Implement a single **"MIDI LEARN" toggle** that highlights every mappable browser control; click → wiggle hardware knob → bound, with automatic CC/Note **LED feedback** sent back on state change and bundled JSON templates for APC40 MkII, Launchpad, and nanoKONTROL2.

### 12. Multi-Layer Compositing with Accurate Photoshop Blend Modes & Layer Routing
* **How the Best Do It:** **Resolume Arena 7** (Layer stack + 50 blend modes + Layer Groups + Layer Routers/Masks) and **VDMX6** (Layers + Groups + Layer-as-Mask).
* **How to Build It:** Support 2–4 active layers per deck (or 4 composited layers + A/B crossfader) using the MIT-licensed **`glsl-blend`** shader functions (Normal, Add, Screen, Multiply, Overlay, Difference, Exclusion, Luma-Key Mask) with per-layer opacity, 2D transform (scale/rotate/tile/mirror), and ability to route Layer N as a mask or `inputImage` into another layer's shader.

### 13. Built-In Projection Mapping Stage (Quads/Slices, Masks, Corner-Pin Warping & Soft-Edge)
* **How the Best Do It:** **Resolume Arena 7 Advanced Output** (Input crop → Output Slice/Mask/Warp/Edge-Blend) and **MadMapper 5/6** (interactive geometric stage canvas with test cards).
* **How to Build It:** Add a final **`moderngl` Output Mapping Pass** controlled from an interactive SVG/HTML5 Canvas tab in the browser UI: define multiple **Slices** (input UV crop → output **homography / corner-pin quad** or subdivided Bézier warp grid), **polygon blackout masks**, **edge-blend feathering + gamma**, a toggleable **Calibration Test Card** (grid + crosshairs + slice IDs), and multi-window / **Spout (Win) / Syphon (macOS) / NDI** output for integration with external servers.

### 14. Intelligent Hands-Free Autopilot (Beat- & Energy-Aware Playlist Progression)
* **How the Best Do It:** **Synesthesia** (Playlist Autopilot with countdown timers and audio-reactive auto-transitions) and **Resolume 7.22** (Column & Layer Autopilot with Forward, Backward, Random, and **Bag** non-repeating shuffle).
* **How to Build It:** Provide an **Autopilot** toggle with selectable trigger interval (every N bars/beats or N seconds), **Bag shuffle** (plays every scene in a Playlist once before repeating), automatic **drop/phrase-aligned transitions** (wait for next downbeat or energy change), and an automatic **safe-override** rule (touching a manual scene launch smoothly resets the Autopilot timer instead of fighting the performer).

### 15. Rock-Solid Performance Ergonomics, Zero-Stutter Hot-Reload & Crash-Safe Show State
* **How the Best Do It:** **TouchDesigner Perform Mode** (decoupled UI/render stability), **MadMapper 6** (`Collect and Save` self-contained workspaces), and **Resolume 7.24** (high-throughput non-blocking MIDI & fast composition loading).
* **How to Build It:**
  * **Decouple Render Loop from Browser/Network I/O:** Keep `moderngl` locked to VSync/60 fps in the main thread while FastAPI/aiohttp WebSockets, MIDI, and OSC run asynchronously and update lock-free state buffers—browser disconnects or slow Wi-Fi must *never* drop a single output frame!
  * **Zero-Glitch Shader Compilation:** Compile/validate new or hot-reloaded GLSL shaders off-path; if a syntax error occurs, keep the previous valid shader running live and show the line-numbered GLSL error cleanly in the browser UI.
  * **Dark-Booth UI Ergonomics:** High-contrast dark theme (pure black/charcoal backgrounds to avoid blinding a dark venue), large touch-friendly sliders/pads for iPad/tablet remote control, double-click-to-reset default on every knob, FPS/frame-time HUD badge, and single-file JSON show-file save/export.

---

## References & Official Documentation Links
* **Resolume Arena 7 & Wire:** [resolume.com/support/en](https://resolume.com/support/en/) — [Clips](https://resolume.com/support/en/clips), [Layers](https://resolume.com/support/en/layers), [Effects](https://resolume.com/support/en/effects), [Dashboard](https://resolume.com/support/en/dashboard), [Envelopes](https://resolume.com/support/en/envelopes), [Parameter Animation](https://resolume.com/support/en/parameter-animation), [MIDI Shortcuts](https://resolume.com/support/en/midi-shortcuts), [Advanced Output](https://resolume.com/support/en/advanced-output).
* **TouchDesigner:** [docs.derivative.ca](https://docs.derivative.ca/), [derivative.ca/learn](https://derivative.ca/learn).
* **VDMX6 (VIDVOX):** [docs.vidvox.net](https://docs.vidvox.net/), [Plugins](https://docs.vidvox.net/vdmx/plugins.html), [Media Management](https://docs.vidvox.net/vdmx/media_management.html), [OSCQuery](https://docs.vidvox.net/vdmx/oscquery.html).
* **MadMapper 5 & 6:** [madmapper.com/documentation](https://madmapper.com/documentation/), [madmapper.com/resources](https://madmapper.com/resources/).
* **Millumin 5:** [help.millumin.com](https://help.millumin.com/).
* **Notch (0.9.x / 1.0 / 2026.1):** [manual.notch.one](https://manual.notch.one/).
* **Synesthesia:** [docs.synesthesia.live](https://docs.synesthesia.live/), [Controls](https://app.synesthesia.live/docs/manual/controls.html), [Presets & Playlists](https://app.synesthesia.live/docs/manual/presets.html), [SSF Spec](https://app.synesthesia.live/docs/ssf/index.html), [Audio Uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html), [OSC](https://app.synesthesia.live/docs/manual/osc.html).
* **Magic Music Visuals:** [magicmusicvisuals.com/documentation](https://magicmusicvisuals.com/documentation/).
* **HeavyM 2:** [help.heavym.net](https://help.heavym.net/).
* **CoGe (Discontinued) & Modul8 (Legacy):** [imimot.com](https://imimot.com/), [garagecube.com/modul8](https://www.garagecube.com/modul8/).
* **Lumen (Paracosm):** [lumen-app.com](https://lumen-app.com/).
* **ISF & Open-Source Shader Libraries:** [isf.video](https://isf.video/), [docs.isf.video/ref_json](https://docs.isf.video/ref_json), [docs.isf.video/ref_multipass](https://docs.isf.video/ref_multipass), [docs.isf.video/ref_audio](https://docs.isf.video/ref_audio), [docs.isf.video/integrations](https://docs.isf.video/integrations), [github.com/Vidvox/ISF-Files](https://github.com/Vidvox/ISF-Files), [github.com/gl-transitions/gl-transitions](https://github.com/gl-transitions/gl-transitions), [github.com/stegu/webgl-noise](https://github.com/stegu/webgl-noise), [github.com/jamieowen/glsl-blend](https://github.com/jamieowen/glsl-blend), [github.com/glslify/glsl-easings](https://github.com/glslify/glsl-easings), [github.com/mrdoob/three.js](https://github.com/mrdoob/three.js), [github.com/hydra-synth/hydra-synth](https://github.com/hydra-synth/hydra-synth), [github.com/patriciogonzalezvivo/lygia](https://github.com/patriciogonzalezvivo/lygia).
