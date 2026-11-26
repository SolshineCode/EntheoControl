# Resolume Arena 7 (7.29): Demo Limits, Advanced Output, Spout, Audio FFT & Sync

> Target setup: BenQ MW632ST (1280×800 native) on Monitor 1, GTX 1650 Ti Max-Q 4 GB, Windows.

---

## 1. Resolume Arena 7 Demo Limitations

Resolume Arena runs in unrestricted **Demo Mode** without a license key ([Resolume Demo Limitations](https://resolume.com/support/en/demo-limitations)):
- **Features:** **100% unlocked** with **no time limit** — Advanced Output (slices, warping, masks, edge blending), Spout In/Out, External Audio FFT, MIDI/OSC, Ableton Link, and saving compositions all work identically to the paid license.
- **Video limitation:** A translucent **Resolume logo watermark** periodically fades in and out over the main output and preview monitors.
- **Audio limitation:** A periodic **robotic voice** plays over Resolume's audio output announcing demo mode.
  - *Practical tip for practice/rehearsal:* If you only use Resolume for **video** (muted audio output in Arena while listening to music through your DJ mixer or system player), the robotic audio prompt won't interrupt your music — only the intermittent visual watermark appears on the projector. Do **not** use Demo Mode for a paid/public show.

---

## 2. Advanced Output: Slices & Masks Workflow

Official docs: [https://resolume.com/support/en/advanced-output](https://resolume.com/support/en/advanced-output)

### Composition & Display Prep
1. Set Windows Display to **Extend**; confirm BenQ MW632ST is **1280 × 800 @ 60 Hz**, BenQ **Auto-Keystone OFF**.
2. In Arena: **Composition → Settings** → set size to **`1280 × 800`** (matches projector 1:1 and keeps VRAM usage low on 4 GB).
3. Go to **Output → Advanced...** (`Ctrl+Shift+A`).

### Step-by-Step Mapping Workflow
1. **Assign the Screen:**
   - Select `Screen 1` in the left panel → **Device** → choose **`Display 2 (1280×800)`** (the BenQ).
2. **Input Selection Tab (what to grab from your composition):**
   - Click `+` → add **Slice** (rectangle) or **Polygon** (custom shapes like diamonds, hexagons, doorframes).
   - Set each slice's **Input Source** to either `Composition` (entire mix) or a specific **Layer** (e.g., Layer 1 = background AI kaleidoscope, Layer 2 = crisp TAS beadwork borders on separate physical shapes).
3. **Output Transformation Tab (where to project it in the room):**
   - Select a slice → enable **Show Slice Grid** (or **Output → Show Test Card**).
   - Drag the 4 corner pins to hug a physical wall panel, foam-core totem, or doorframe.
   - For curved/irregular surfaces, switch to **Edit Points** → enable **Bezier** or increase **Mesh** subdivisions.
4. **Masks (blocking unwanted light spill):**
   - Inside **Output Transformation**, select your Screen or Slice → click the **Mask** tool (polygon/circle/crop).
   - Trace pillars, windows, picture frames, or DJ tables so light is masked to pure `#000000` black.
   - Toggle **Invert Mask** if you want to project *only* inside a traced cutout instead of blocking it out.
5. **Save & Close:** Click **Save & Close** (presets are stored in `Documents\Resolume Arena\Presets\Advanced Output\`).

---

## 3. Receiving a Spout Source (`PsyProjector`)

Official docs: [https://resolume.com/support/en/syphon-and-spout](https://resolume.com/support/en/syphon-and-spout)

1. Start `live_visuals.py --spout` (or TouchDesigner with a `Syphon Spout Out TOP`).
2. In Resolume Arena, open the **Sources** panel (bottom-right) → scroll to **Spout Servers**.
3. Your live sender **`PsyProjector`** appears automatically in the list.
4. Drag **`PsyProjector`** into any empty Clip Slot on a Layer and trigger it.
   - *Tip:* Set that clip's **Resize** mode to `Fill` or `Stretch` if the composition resolution differs from the Spout feed, and stack Resolume's built-in **Mirror / Kaleidoscope / Hue Rotate / Feedback** effects right on top!
5. **Sending Spout OUT of Arena:** `Output → Texture Sharing (Spout)` (or assign a Virtual Screen to Spout inside Advanced Output) if you ever want to feed Resolume into TouchDesigner/StreamTurbo.
6. **Hybrid-GPU (ThinkPad Optimus) Requirement:** Just like TouchDesigner, pin **both** `Arena.exe` and `.venv\Scripts\python.exe` to **High Performance (NVIDIA GTX 1650 Ti)** in *Windows Settings → System → Display → Graphics*, otherwise Spout frames will appear black.

---

## 4. Audio FFT Effect Routing

Official docs: [https://resolume.com/support/en/audio-fft](https://resolume.com/support/en/audio-fft)

1. **Select Audio Input:**
   - Open **Arena → Preferences → Audio**.
   - Set **Audio Input Device** to your USB audio interface (gig) or **VB-Cable `CABLE Output`** / laptop mic (home practice).
   - Set **FFT Input** to the stereo channel pair receiving music (e.g., `Ch 1+2`).
2. **Map FFT to Any Parameter:**
   - Pick any clip, layer, or effect parameter (e.g., *Scale*, *Opacity*, *Bloom Amount*, *Hue Rotate*, *Displace Amount*).
   - Click the small **gear / dropdown arrow** next to the parameter → choose **`External FFT`** (for live external audio/DJ mixer) or **`Composition FFT`** (for audio tracks playing inside Arena).
3. **Sculpt the Band & Smoothing:**
   - Expand the FFT panel beneath the slider:
     - **L / M / H buttons** (or drag the frequency window): isolate **Low** (40–120 Hz kick), **Mid** (250 Hz–2 kHz synths/vocals), or **High** (4 kHz+ hi-hats/shimmer).
     - **Gain:** boost/attenuate until peaks reliably touch 80–95% without pegging flat at 100%.
     - **Fall (Decay):** raise **Fall** to `0.3–0.6` so visuals swell and glide smoothly instead of jittering.
     - **Range handles (`< >`):** clamp the parameter's min/max (e.g., scale `100% → 112%` rather than `0% → 500%`).
4. **Resolume 7.29 + Wire note:** Arena 7.29 includes expanded audio FFT routing into **Resolume Wire** patches via the `Spectrum In` node (allowing custom shader/generative effects driven directly by raw FFT bins).

---

## 5. Ableton Link & BPM Tap Sync

Official docs: [https://resolume.com/support/en/ableton-link](https://resolume.com/support/en/ableton-link)

### Ableton Link (Auto-Sync over Wi-Fi / Ethernet / Localhost)
1. Click the **`Link`** button in the top-left BPM toolbar of Resolume Arena.
2. Any Link-enabled app on the same laptop or local network (Ableton Live, Traktor, Rekordbox via Link, TouchDesigner's `Ableton Link CHOP`, iPad apps) immediately shares **BPM and 4-beat bar phase**.
3. On any clip or effect parameter, switch Timeline mode to **`BPM Sync`** and set beats (`4`, `8`, `16`, `32` beats) so psychedelic cycles breathe on exact 4-bar / 8-bar EDM phrases.

### Manual BPM Tap (When Syncing to a Live DJ Without Network Link)
1. Tap the **`Tap`** button in the BPM bar (or map a pad on your MIDI controller to **Tap Tempo**) on the beat 4–8 times.
2. Hit **`Resync`** right on the downbeat (beat 1 of a bar) whenever phase drifts.
3. Use **`/2`** and **`*2`** buttons for quick half-time (ambient breakdowns) or double-time (drops).

---

## Sources
- Resolume Demo Limitations: https://resolume.com/support/en/demo-limitations
- Resolume Advanced Output (Slices & Masks): https://resolume.com/support/en/advanced-output
- Resolume Syphon and Spout: https://resolume.com/support/en/syphon-and-spout
- Resolume Audio FFT: https://resolume.com/support/en/audio-fft
- Resolume Ableton Link: https://resolume.com/support/en/ableton-link
