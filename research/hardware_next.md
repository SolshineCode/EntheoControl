# Hardware Upgrades & BenQ MW632ST Throw-Distance Math

> Ranked by impact-per-dollar for moving from home wall testing to live EDM / house-party / art-installation gigs.

---

## 1. USB Audio Interface for DJ Mixer Line-In

At a live show, **never rely on your laptop microphone** for FFT audio reactivity — crowd chatter and room reverb muddy the kick/bass separation. Always tap a clean stereo **Line-Level** feed from the DJ mixer's **`Rec Out`** (RCA, fixed volume — ideal!) or **`Booth Out`** (1/4" TRS).

| Model | Approx. Price | Inputs | Best For | Pros / Caveats |
|---|---|---|---|---|
| **Behringer UCA222** | ~$25–$35 | Stereo **RCA Line-In** + RCA Out + headphone monitor | **Cheapest gig-ready DJ tap** | Plug-and-play USB class-compliant (DirectSound/WASAPI); accepts DJ mixer `Rec Out` directly with a standard stereo RCA cable; fixed line-level gain (zero preamp clipping hassle). No ASIO low-latency driver needed for FFT. |
| **Behringer UMC202HD** | ~$80–$100 | 2× XLR / 1/4" TRS combo (with Pad + Gain) | Budget pro interface | Balanced TRS inputs resist long-cable club hum; dedicated ASIO driver for TouchDesigner; needs RCA-to-dual-1/4"-TS cable for `Rec Out`. |
| **Focusrite Scarlett 2i2 (Gen 3/4)** | ~$130–$199 | 2× XLR / 1/4" TRS combo | Touring standard | Rock-solid Windows ASIO drivers, visual gain halos (green/amber/red) make setting FFT input level effortless in a dark booth. |

**Essential $15 gig pouch extras:**
- 1× **Stereo RCA-to-RCA** cable (10–15 ft) for DJ `Rec Out` → UCA222.
- 1× **Dual 1/4" TS-to-RCA** cable (if using UMC202HD / Scarlett).
- 1× **Ground-loop isolator** (RCA or 3.5 mm, ~$10) — eliminates 60 Hz hum that would otherwise falsely trigger the 40–120 Hz bass/kick FFT band!

---

## 2. MIDI Controller: Akai APC Mini mk2 vs. Novation Launch Control XL

Playing visuals live feels 10× more musical with physical faders and knobs than keyboard hotkeys.

| Spec | **Akai APC Mini mk2** (~$99) | **Novation Launch Control XL** (~$169) |
|---|---|---|
| **Controls** | **64 RGB pads** (8×8 grid) + **9 faders** (8 ch + 1 master) | **24 rotary knobs** (3 rows × 8, RGB LEDs) + **8 faders** + **16 buttons** |
| **Build & Portability** | Ultra-compact, lightweight plastic; fits in any backpack | Sturdy chassis, long-throw faders, clear center-detent knobs |
| **Best Software Fit** | **Resolume Arena** (clip/column launching + layer opacities) | **TouchDesigner & `live_visuals.py`** (continuous shader uniforms + feedback/warp tweaking) |
| **Main Weakness** | Zero rotary knobs (harder to dial fine shader parameters) | Only 16 buttons (no big clip-trigger grid) |

### Recommended Starter Mappings (for `live_visuals.py` / TD / Resolume)
- **Faders 1–4:** `Beadwork Opacity`, `AI Texture Opacity`, `Feedback Decay (0.0–0.96)`, `Master Dimmer / Blackout`.
- **Faders 5–8 (or Knobs Row 1):** `Bass Reactivity Gain`, `Noise Swim/Warp Depth`, `Hue Cycle Speed`, `Zoom/Tunnel Speed`.
- **Buttons / Pads:**
  - Row 1: Kaleidoscope Fold presets (`3, 4, 5, 6, 8, 10, 12, 16`).
  - Row 2: Next/Prev AI texture, Crossfade trigger, Tap Tempo, Instant Blackout (kill switch for drops!).

**Recommendation:** If you plan to lean heavily on **Resolume Arena** clip/loop decks → **APC Mini mk2**. If you plan to perform primarily with **generative GLSL shaders (`live_visuals.py` / TouchDesigner)** where continuous parameters drive the magic → **Novation Launch Control XL**.

---

## 3. HDMI Cables: Passive Copper vs. Active Optical (AOC)

Our BenQ MW632ST runs at **1280 × 800 @ 60 Hz** (low bandwidth compared to 4K60 — ~2.2 Gbps vs 18 Gbps), which is very forgiving, *but* long dark-room cable runs still fail if you buy cheap passive wire:
- **Up to 25 ft (7.5 m):** Standard certified **High-Speed HDMI 2.0 passive copper** (~$15–$25, 24–26 AWG). Rock-solid for bedrooms, house parties, and nearby booths.
- **25–50 ft (7.5–15 m):** Passive copper is borderline at 50 ft (prone to HDMI handshake drops when club lights/subwoofers spike power). Prefer **Active Optical HDMI (AOC)** from 35 ft upward.
- **50–150 ft (15–45 m):** **Active Optical HDMI (fiber optic AOC)** (~$40–$90). Zero signal loss, immune to electrical/lighting interference, thin & lightweight.
  - **Critical AOC rules:** Always **directional** — plug the **`SOURCE`** end into the laptop and **`DISPLAY`** into the BenQ. Never bend sharper than a ~2-inch radius or let crowds step on it (tape down with matte gaffer tape).
- **Port saver ($8):** Keep a short 6-inch HDMI male-to-female pigtail on the laptop so an accidental yank on a long cable never snaps the laptop's soldered HDMI port.

---

## 4. BenQ MW632ST Throw-Distance Math (Exact Specs)

Official specs ([BenQ MW632ST Specifications](https://www.benq.com/en-us/business/projector/mw632st/specifications.html), [BenQ Projector Calculator](https://projectorcalculator.benq.com/)):
- **Native Resolution:** `1280 × 800` (16:10 native aspect ratio)
- **Brightness:** `3,200 ANSI lumens` | **Contrast:** `13,000:1`
- **Throw Ratio:** **`0.72:1`** (max zoom / widest image) to **`0.87:1`** (min zoom / tightest image) via **1.2× optical zoom**
- **Formulas:**
  - $\text{Image Width} = \dfrac{\text{Throw Distance}}{\text{Throw Ratio}}$
  - $\text{Image Height} = \text{Width} \times \dfrac{800}{1280} = \text{Width} \times 0.625$
  - $\text{Diagonal} = \sqrt{W^2 + H^2} = \text{Width} \times 1.1779$
  - $\text{Approx. Surface Brightness} = \dfrac{3200\text{ lm}}{W_{\text{ft}} \times H_{\text{ft}}}\text{ (lm/ft}^2\text{; }\times 10.764 \approx \text{lux)}$

### Throw Table at 3 ft, 5 ft, and 10 ft

| Throw Distance | Zoom Setting (Throw Ratio) | Image Width | Image Height | Diagonal | Area | Pixel Density | Approx. Brightness (3200 lm) |
|---|---|---|---|---|---|---|---|
| **3 ft** (0.91 m) | **Max Zoom (0.72)** | **4.17 ft** (50.0") | **2.60 ft** (31.3") | **4.91 ft** (58.9") | 10.85 ft² | 25.6 px/in | ~295 lm/ft² (~3,170 lux) — blazing bright |
| **3 ft** (0.91 m) | Min Zoom (0.87) | 3.45 ft (41.4") | 2.16 ft (25.9") | 4.06 ft (48.7") | 7.43 ft² | 30.9 px/in | ~431 lm/ft² (~4,640 lux) |
| **5 ft** (1.52 m) | **Max Zoom (0.72)** | **6.94 ft** (83.3") | **4.34 ft** (52.1") | **8.18 ft** (98.2") | 30.14 ft² | 15.4 px/in | ~106 lm/ft² (~1,140 lux) — ideal wall size |
| **5 ft** (1.52 m) | Min Zoom (0.87) | 5.75 ft (69.0") | 3.59 ft (43.1") | 6.77 ft (81.2") | 20.64 ft² | 18.6 px/in | ~155 lm/ft² (~1,670 lux) |
| **10 ft** (3.05 m) | **Max Zoom (0.72)** | **13.89 ft** (166.7") | **8.68 ft** (104.2") | **16.36 ft** (196.3") | 120.56 ft² | 7.7 px/in | ~26.5 lm/ft² (~286 lux) — full dark room/stage |
| **10 ft** (3.05 m) | Min Zoom (0.87) | 11.49 ft (137.9") | 7.18 ft (86.2") | 13.54 ft (162.5") | 82.57 ft² | 9.3 px/in | ~38.8 lm/ft² (~417 lux) — punchy small stage |

*Sweet spot notes:*
- **3–5 ft** is the magic zone for **hyper-crisp TAS beadwork** and foam-core 3D stage totems (high pixel density + intense jewel-tone pop even with minor ambient light).
- **8–10 ft** fills an entire bedroom/living-room wall or a festival canopy/rear-projection portal (make sure the room/night is dark; at 10 ft max zoom the image is ~14 ft wide × 8.7 ft tall, which exceeds a standard 8 ft ceiling!).

---

## 5. Cheap Physical Staging Upgrades (<$50 total)
- **Matte white foam-core boards** ($15–$25): Cut diamonds, hexagons, and overlapping chevrons; mount with 1–2" foam spacers for instant 3D festival-totem depth.
- **White ripstop nylon or frosted shower curtain** ($15): Hang across a doorway/canopy and set BenQ to **Rear Projection** mode for a walk-up glowing portal with zero shadows.
- **Projector tripod stand with lipped tray** ($30–$45): Eliminates table-bump misalignment so your Kantan/corner-pin mapping stays locked all night.

---

## Sources
- BenQ MW632ST Specifications: https://www.benq.com/en-us/business/projector/mw632st/specifications.html
- BenQ Official Projector Throw Calculator: https://projectorcalculator.benq.com/
- Resolume Audio Preferences & External FFT: https://resolume.com/support/en/audio-fft
