# Walkthrough: your first session, step by step

This walkthrough takes you from a dark room to a performance you can play live. It takes about 45 minutes.
For installation and the full reference, see [QUICKSTART.md](QUICKSTART.md).

## 0. Before you start (2 min)
- **Power and position:** keep the laptop on its charger. Raise the back of the laptop so the vents breathe, since the GPU works hard.
- **Projector:** in the projector's menu, turn Auto Keystone OFF. Use the native resolution (for example 1280x800). On the laptop, press **Win+P** and choose **Extend**.
- **Windows tweaks for shows (optional, they cut heat):**
  - Set Power Options > Processor maximum state to **99%** (turbo off).
  - In the NVIDIA Control Panel, set **Threaded optimization = Off** for `python.exe`.

## 1. Start the show (1 min)
```
powershell -ExecutionPolicy Bypass -File run_show.ps1
```
- **What opens:**
  - the visuals, on the projector
  - the **control panel**, in your browser
  - the AI worker, which loads only when you first ask for an image
- **Quitting:** use **Shift+Esc**. A single Esc never quits.
- **Crashes:** if anything crashes, the show restarts within a second with the same look.

Play some music on the laptop. The visuals breathe with the bass, mids and highs.

## 2. Learn the panel (5 min)
- **Top bar:**
  - MASTER fader
  - **BLACKOUT**, and **PANIC** (hold for half a second to reach the safe look)
  - **BPM / TAP**
  - the **health meter**: fps, GPU, temperature, battery
  - **REC**
  - **PREVIEW**, off by default to save the GPU
  - **ONLINE**, off by default
- **Tabs:** each tab that changes the picture has a **power icon** that turns it fully off. Turning it back on restores exactly the same settings. **ALL FX** does this for everything at once.
- **Faders:**
  - drag to set, Shift-drag for fine control, scroll wheel to nudge
  - double-click to reset
  - **MOD** makes the fader follow the bass, mids, highs, a beat-synced LFO or a random walk

## 3. Shape the pattern (10 min)
1. **PERFORM** (everything on one screen): in **Macros** move **Zoom**, **Spin**, **Symmetry** (flat vs mandala) and **Speed**, tap the palette swatches, tap the **ripple pad**. **MORE CONTROLS…** has Rotate speed, Swim, Hue and the rest.
2. **SHAPE:** **Twirl**, **Bulge**, **Wave**. Click the XY pad to drop **ripples**.
3. **COLOR:** turn up the **light chase**, the TAS-style sweep of a hue window across the pattern.
4. **MOTION:**
   - Pick a **travel mode** (zoom tunnel, 3D tube, ground plane, depth layers) and push **Travel** to fly into the visual.
   - Add **Spin** to barrel-roll, and **Corkscrew** to make the far end spin faster.
5. **BEINGS (on MOTION):**
   - Raise the **Beings** layer and pick a creature: anaconda spirit, jaguar, jellyfish, all-seeing eyes, whale, kitten and others.
   - **Variety** mixes forms. **Colour mode** recolours them and the serpents.

## 4. Add your art and AI images (10 min)
1. **ART:** upload a painting or photo. A depth map is made automatically.
2. **Pick it:** click its thumbnail in **Sources**. If its layer was off, it fades in.
3. **Choose how it's used:**
   - **Relief**: fly around a 3D version with Yaw, Pitch and Dolly
   - **Stencil**: seen through the pattern lines
   - **Kaleidoscope**
   - **Tunnel walls**
   - **Displace**: the art warps the pattern
   - **Melt**
4. **AI DREAM:** type a prompt such as *"glowing jade buddha, sacred geometry, black background"*. About 2.5 s later the new image appears in Sources.

## 5. Layer it up (5 min)
- **Live layers** (PERFORM): only the layers that are showing; **EYE** shows or hides a layer on the wall, the fader sets its strength, "+ N OFF" brings another one in. The full **Layers card** is on LAYERS, the next tab: with PREVIEW on, **SOLO** previews a layer in the panel only, without the audience seeing it.
- **3D DECKS:** two raymarched worlds (corridor, torus wormhole, stained-glass vortex, honeycomb domes) with a crossfader and blend modes, plus a particle **swarm**.
- **TEXT:** type words and pick a font. Then choose movement (drift, orbit, wave, spiral into the tunnel) and how they weave in (on top, engraved into the pattern, mask, knockout, warp, melt with the scene).

## 6. Map real objects (10 min)
1. **One rectangle:** tape a rectangle on the wall. In **MAPPING**, drag the 4 corners onto it while watching the wall. The magnifier and arrow keys give pixel precision. Each corner shows its angle and each edge its length.
2. **A box:** click **+ ADD** for each visible face, and turn on **TEST GRID** to see the numbered outlines on the wall. Drag each face's corners; neighbouring corners snap together. Then try **EDGE GLOW**, **SPANNED** (one picture wrapping the box) or **WHOLE FRAME** for each face.
3. Everything outside your surfaces stays black.

## 7. Save looks and perform (5 min)
- **SCENES:**
  - Shift-click a slot to **save** the current look, and click to **recall** it. Recall morphs over the chosen bars and lands on the beat.
  - Use **Deck A / B + crossfader** to blend two scenes by hand.
- **Autopilot** steps through your saved scenes on its own. It runs locally and calls no AI.
- **MIDI:** click **MIDI LEARN**, click any control, then move a knob or hit a pad. Default maps exist for the APC mini mk2 and Launch Control XL.
- **Record:** press **REC**. Choose clean program or the mapped wall, with or without laptop audio. Files go to `recordings\`.

## 7b. Save your show, keep presets, build a timeline (10 min)
- **Save everything:** top bar **SHOW** > type a name > **SAVE AS**. From then on **Ctrl+S**. If the laptop crashes, open SHOW and press **RECOVER**.
- **Keep a look for later:** top bar **PRESETS** > name it > **SAVE PRESET**. Click its card any time to bring it back, or tick only COLOR first to take just its colours.
- **Make a timeline:** open **TIMELINE**. Double-click the *Sections* row to mark "intro", "build", "drop". Double-click *Cues* and type a scene number. Add a lane with **+ AUTOMATION LANE** (try Zoom) and double-click it to set points; drag them.
- **Or record it live:** press **REC**, then **PLAY**, and perform: fader moves, scene pads, triggers and layer switches are written in. Press PLAY again to stop; **UNDO** removes the whole take.
- **Play it back:** **PLAY**. Grab a fader at any moment and it is yours; let go and the timeline takes it back (or press **RELEASE**). **PANIC** always stops it.

## 8. Optional: the online assistant
Turn **ONLINE** on and open **ASSISTANT**, then describe what you want, for example *"slower and deeper, emerald palette, bring in the jaguar"*.
- **Modes:** in **SUGGEST** mode it proposes changes for you to **Apply**. **AUTOPILOT** applies them, and each batch can be undone.
- **Cost:** it only calls the AI when you press SEND.
- **Limits:** it can never strobe, flash, black out or touch your mapping.
- **Nano Banana images:** paste your own Gemini API key in AI DREAM.

## When the laptop struggles
The health meter turns amber or red and suggests a fix, such as lowering a heavy layer's resolution, with **Apply** and **Ignore** buttons. On battery it warns you, and below 10% it shows a red bar. For long shows, stay plugged in.

## Respect
The kene-style patterns are generative and *inspired by* Shipibo-Konibo, Huni Kuin and Yawanawá traditions. Credit them, and collaborate with Indigenous artists where you can. Use only artwork you own or have rights to.
