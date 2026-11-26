# TouchDesigner 2025: Audio Loopback, `audioAnalysis`, Projection Mapping & Spout

> Target setup: a hybrid-GPU laptop (Intel iGPU + NVIDIA GTX 1650 Ti Max-Q 4 GB), Windows, TouchDesigner 2025.33230 (Non-Commercial — 1280×1280 cap fits the BenQ MW632ST native **1280×800** 1:1 with zero downscaling).

---

## (a) Capturing Windows System Audio in `Audio Device In CHOP`

### Important fact about WASAPI Loopback in TouchDesigner
TouchDesigner's **Audio Device In CHOP** exposes only two Windows driver backends in its `Driver` parameter: **`DirectSound/CoreAudio`** and **`ASIO`** ([Audio Device In CHOP docs](https://docs.derivative.ca/Audio_Device_In_CHOP)). It does **not** expose a native Windows WASAPI Loopback driver dropdown (unlike OBS, Audacity, or Python's `soundcard` library used in `live_visuals.py`).

### Exact Steps — Option 1 (Recommended for Home Practice): VB-Audio Virtual Cable
1. Download and install **VB-Cable** from [https://vb-audio.com/Cable/](https://vb-audio.com/Cable/) (extract ZIP → right-click `VBCABLE_Setup_x64.exe` → **Run as administrator** → reboot if prompted).
2. Route desktop audio into VB-Cable:
   - Open **Windows Settings → System → Sound** (or `mmsys.cpl` → **Playback** tab).
   - Set **`CABLE Input (VB-Audio Virtual Cable)`** as the default playback device (or per-app in *Volume Mixer* for Spotify/Chrome/DJ software).
3. Keep hearing audio on your speakers/headphones simultaneously:
   - Open `mmsys.cpl` → **Recording** tab → double-click **`CABLE Output (VB-Audio Virtual Cable)`** → **Listen** tab.
   - Check **Listen to this device**, pick your real speakers/headphones under *Playback through this device*, click **Apply**.
4. Capture inside TouchDesigner:
   - Create an **Audio Device In CHOP** (`Tab → CHOP → Audio Device In`).
   - Set **Active** = `On`.
   - Set **Driver** = `DirectSound/CoreAudio`.
   - Set **Device** = `CABLE Output (VB-Audio Virtual Cable)`.
   - Set **Rate** = `44100` or `48000` (match Windows device format in `mmsys.cpl → Advanced`), **Buffer Length** = `0.05`–`0.1` s (lower for tighter sync, raise if clicks occur).

### Exact Steps — Option 2: VoiceMeeter (cleaner multi-bus monitoring)
If you want separate monitor/Visuals mixes without Windows' "Listen" latency, install **VoiceMeeter** ([https://vb-audio.com/Voicemeeter/](https://vb-audio.com/Voicemeeter/)), route Windows playback to *VoiceMeeter Input*, send bus **A1** to your headphones/speakers, and select **`VoiceMeeter Out B1`** in TouchDesigner's `Audio Device In CHOP` (`DirectSound`).

### Exact Steps — Option 3: Live Gig (DJ Mixer Line-In)
Connect DJ mixer `Rec Out` / `Booth Out` to a USB audio interface (see `hardware_next.md`), set **Driver** = `ASIO` (or `DirectSound` for UCA222), choose the interface under **Device**, and select the stereo input pair in **Inputs**.

---

## (b) Using `Palette:audioAnalysis` for Kick / Snare / High

TouchDesigner ships with a tuned multiband + percussive onset detector in the Palette ([Palette:audioAnalysis docs](https://docs.derivative.ca/Palette:audioAnalysis)).

### Exact Wiring Steps
1. Open the Palette (`Alt+L`), navigate to **`Tools → audioAnalysis`**, and drag it into your network.
2. Connect your **Audio Device In CHOP** output directly into `audioAnalysis`'s input.
3. Click `audioAnalysis` and inspect its custom parameters:
   - **Analyze page:** ensure **Low**, **Mid**, **High**, **Kick**, **Snare** (and optionally **Spectral Centroid** / **Rhythm**) are enabled.
   - **Gain / Threshold tuning:** play a reference EDM track and open the component's viewer (or attach a **Trail CHOP**). Adjust **Low / Kick Threshold** and **Snare Threshold** so `kick` fires cleanly on the 4-on-the-floor pulse without chattering during breakdowns.
   - **Smoothing (Lag):** `audioAnalysis` exposes smoothing/lag controls per band; for psychedelic swimming visuals (non-jittery), attach a **Lag CHOP** after the output with fast attack (`Lag 1 = 0.03–0.05 s`) and smooth release (`Lag 2 = 0.25–0.45 s`).
4. Extract individual channels cleanly:
   - Attach a **Select CHOP** → set **Channel Names** to `kick snare low mid high` (or pick one channel per Select CHOP for clarity).
   - Attach a **Math CHOP** to rescale `0..1` into target shader/TOP ranges (e.g., `kick` → `Feedback Scale` `1.002..1.018`, `mid` → `Noise Period/Translate`, `high` → `HSV Hue Offset` or `Bloom Intensity`).
   - Reference cleanly via Python expressions, e.g. `op('null_audio')['kick']` (always terminate audio chains with a **Null CHOP** named `null_audio`).

---

## (c) Single 1280×800 Projector Mapping: Kantan Mapper vs. CamSchnappr

### Pre-flight Display Setup (Both Tools)
1. Plug in the BenQ MW632ST via HDMI → Windows Display Settings → **Extend these displays** (never Duplicate).
2. Confirm BenQ resolution is set to native **1280 × 800 @ 60 Hz**, Windows Display Scale = **100%**.
3. On the BenQ projector OSD menu: turn **Auto Keystone OFF** and set **Keystone = 0** (do all warping in software to preserve sharpness).
4. In TouchDesigner, output via a **Window COMP**:
   - **Window Operator** = your final mapped `Null TOP` (resolution `1280 × 800`).
   - **Monitor** = `1` (the BenQ), **Borders** = `Off`, **Size** = `Fill`, **Opening Size** = `Fill`, **Always on Top** = `On`.
   - Click **Open as Separate Window** (or press `F1` for Perform Mode).

### Option A: `Palette:kantanMapper` (Best for 2D surfaces, walls, doors, layered flat totems)
Docs: [https://docs.derivative.ca/Palette:kantanMapper](https://docs.derivative.ca/Palette:kantanMapper)
1. Drag **`Mapping → kantanMapper`** from the Palette into your network.
2. Set `kantanMapper`'s **Output Resolution** to **`1280 × 800`** and **Monitor** to `1` (BenQ).
3. Click **Open Kantan Window** on the parameter page.
4. In the Kantan UI:
   - Use the **Rectangle**, **Freeform (Polygon)**, or **Bézier** tool to trace physical surfaces on your wall/room/foam-core shapes while watching the live projector output.
   - Select each shape → drag any TOP (e.g., your Spout In TOP or generative beadwork Null TOP) onto the shape's **Texture** parameter.
   - Use **Mask** shapes (black fill on top layer) to block light spill onto doorways, DJ gear, or windows.
   - Enable **Soft Edge** / adjust UV crop per shape for independent visuals per surface.
5. Take `kantanMapper`'s TOP output (`out1`) straight into your Window COMP.

### Option B: `Palette:camSchnappr` (Best for 3D sculptures, cubes, pyramids, polyhedra)
Docs: [https://docs.derivative.ca/Palette:camSchnappr](https://docs.derivative.ca/Palette:camSchnappr)
1. Model (or photogrammetry-scan) the exact physical 3D sculpture in **Blender 4.5 LTS** and export as `.fbx` or `.obj` with clean UVs.
2. Import into TouchDesigner via **File In SOP** → place inside a **Geometry COMP** (`geo_sculpture`).
3. Drag **`Mapping → camSchnappr`** from the Palette.
4. Set `camSchnappr`'s **Render Geo** parameter to `geo_sculpture`, **Monitor** to `1`, and output resolution to **`1280 × 800`**.
5. Click **Open CamSchnappr**:
   - In the 3D viewport, select **at least 6 non-coplanar vertices** on the virtual mesh (corners/apexes).
   - For each point, drag its crosshair on the projector output until it lands dead-center on the corresponding physical corner of the sculpture.
   - `camSchnappr` runs OpenCV `cameraCalibrate` to compute the physical projector's exact 3D pose, lens FOV, and distortion, outputting a calibrated **Camera COMP**.
6. Feed `camSchnappr` into a **Render TOP** (`1280 × 800`) referencing `geo_sculpture` + your psychedelic material/TOP, then send that Render TOP to your Window COMP.

**Rule of thumb:** Start with **Kantan Mapper** for flat walls, doorframes, shelves, and TAS-style layered flat foam-core shields. Switch to **CamSchnappr** once you build a 3D faceted sculpture with a matching Blender mesh.

---

## (d) Receiving Spout Sender `PsyProjector` (`Syphon Spout In TOP`)

Docs: [https://docs.derivative.ca/Syphon_Spout_In_TOP](https://docs.derivative.ca/Syphon_Spout_In_TOP)

### Exact Steps
1. Start the Python sender:
   ```powershell
   .\.venv\Scripts\python.exe live_visuals.py --spout
   ```
   Confirm console prints `[spout] sending as 'PsyProjector'`.
2. In TouchDesigner, drop a **Syphon Spout In TOP** (`Tab → TOP → Syphon Spout In`).
3. Set **Active** = `On`.
4. Click the **Sender Name** dropdown and select **`PsyProjector`** (or type `PsyProjector` verbatim).
5. Verify texture arrives at `1280 × 800` (middle-click the TOP to check resolution and pixel format). If vertically inverted, follow it with a **Flip TOP** (`Flip Y = On`).

### Critical Hybrid-GPU (Optimus) Laptop Gotcha
Derivative's docs explicitly note that Spout on Windows requires an **NVIDIA or AMD GPU** (Intel GPUs are not supported for Spout texture sharing), and **both sender and receiver MUST run on the same physical GPU**.
On a hybrid-GPU laptop (tested: Intel UHD + NVIDIA GTX 1650 Ti Max-Q):
1. Open **Windows Settings → System → Display → Graphics**.
2. Add BOTH:
   - `<your EntheoControl folder>\.venv\Scripts\python.exe`
   - `C:\Program Files\Derivative\TouchDesigner.2025.33230\bin\TouchDesigner.exe`
3. Set both to **High Performance (NVIDIA GeForce GTX 1650 Ti with Max-Q Design)**.
4. Restart both apps if you changed this setting while either was running.

---

## Sources
- TouchDesigner `Audio Device In CHOP` documentation: https://docs.derivative.ca/Audio_Device_In_CHOP
- TouchDesigner `Palette:audioAnalysis` documentation: https://docs.derivative.ca/Palette:audioAnalysis
- TouchDesigner `Palette:kantanMapper` documentation: https://docs.derivative.ca/Palette:kantanMapper
- TouchDesigner `Palette:camSchnappr` documentation: https://docs.derivative.ca/Palette:camSchnappr
- TouchDesigner `Syphon Spout In TOP` documentation: https://docs.derivative.ca/Syphon_Spout_In_TOP
- VB-Audio Virtual Cable: https://vb-audio.com/Cable/
- VB-Audio VoiceMeeter: https://vb-audio.com/Voicemeeter/
