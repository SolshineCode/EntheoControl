# Quickstart: getting hands-on

All commands run in PowerShell from the project folder:
```
cd C:\path\to\EntheoControl
```

## 1. Hook up the projector (5 min)
1. HDMI from the ThinkPad to the BenQ, power on, dark room, 3-6 ft from a wall.
2. Press **Win + P** and choose **Extend** (not Duplicate). The projector becomes display #2.
3. BenQ menu: turn **Auto Keystone OFF** and set keystone to 0. All warping happens in software, which keeps the image sharp.
4. Settings > Display: set the projector to **1280 x 800** (its native resolution).

## 2. Start the show (one command)
```
powershell -ExecutionPolicy Bypass -File run_show.ps1
```
This starts everything: the visuals on the projector, the **control panel** in your laptop browser (one panel for everything), and the AI worker. A crash restarts automatically and resumes the same look. The panel link stays the same between restarts.

- **Quit:** **Shift+Esc**, or Esc twice within 1 s. A single Esc does nothing, so you can't end a show by accident.
- **Unplugging the projector** pauses the visuals; plug it back in and they return. If you launch before plugging in, it waits for the projector.
- Play music on the laptop (Spotify, YouTube, a DJ set) and the visuals react to it. Add `--audio mic` to use the room mic instead (e.g. `run_show.ps1 --monitor 1 --panel --audio mic`).
- Add `--spout` to send the output into TouchDesigner or Resolume.

**The panel tabs:** PERFORM (main faders + layers + scenes) · MOTION (travel into the visual, serpents) · SHAPE (zoom, pan XY pad, twirl, bulge, ripples) · COLOR (palettes, TAS light chase) · LAYERS (kene, beads, lines, AI, video, fractals, art) · FRACTAL (pure-math modes) · ART (upload paintings, fly through them) · SCENES (16 slots, morph, autopilot) · AI DREAM (type a prompt) · CONTROLLERS (MIDI learn, gamepad, clock) · MAPPING (several surfaces: boxes, sculpture faces; corners, angles, masks, edge glow) · 3D DECKS (flown-through 3D landscapes on an A/B deck with blend modes, particle swarm) · MOD BUS (audio scope, 16-step sequencer, every active modulation). Press **?** in the panel for shortcuts.

**Layers card (top of PERFORM and LAYERS):** one row per layer with a live thumbnail. **EYE** shows / hides the layer on the projector, **SOLO** shows only the soloed layers in the panel's monitor without changing the projector, and the fader is the opacity. Uploaded art not showing? Its layer ("Art pieces") starts at 0: click a thumbnail in **Sources** and it fades in.

**Power switches:** every tab that changes the picture has a power icon on its tab button and a big ON / OFF switch at its top; a layer's **EYE** does the same for one layer, **ALL FX** for every effect. Off and on again brings back exactly the same settings. **PREVIEW** (top bar) is OFF by default to save GPU: switch it on to see the output monitor, solo layers and layer thumbnails in the panel.

**Also in the panel:** a small **output monitor** in the top bar (click it for a large one), a **Sources** picker on PERFORM / ART / AI DREAM (thumbnails of uploaded art and AI images: click one to use it), and a **scene crossfader** on PERFORM / SCENES (click DECK A, click a scene, same for DECK B, then drag to blend the two looks by hand).

Every fader has a **MOD** button: make it follow the bass, mids, highs, a beat-synced LFO or random.

**Projector-window keys** (click the projector window first; the panel is easier): `1`-`9` folds · `M` layer preset · `P` palette · `TAB` flat/mandala · `N` travel mode · `E` serpents · `-`/`=` zoom · hold `X` flash, `Z` strobe, `I` invert, `Q` freeze, `K` blackout, `J` punch · `,` tap tempo · `Ctrl+C` calibrate (drag corners with the mouse) · `Ctrl+G` outlines + grid + names · while calibrating `Ctrl+Z` undo, `Ctrl+R` reset that surface · `S` save mapping · `Ctrl+Backspace` reset · `Ctrl+W` panel server.

**First exercise (one surface):** tape a big rectangle on the wall (or hold up a foam board). In the panel's MAPPING tab drag the 4 corners on the canvas onto its corners while you watch the wall. A magnifier shows the exact pixel; arrow keys nudge the selected corner 1 px (Shift = 0.1 px). The visuals now live exactly on that object. It saves by itself (`surfaces.json`); Ctrl+Z undoes.

**Second exercise (a box, three faces):**
1. Stack a cardboard box so the projector sees its front, one side and the top.
2. MAPPING > **+ ADD** twice: you now have three surfaces. Click one in the list and type a name ("Box front").
3. Set "Show on the projector" to **ALL** and switch **TEST GRID** on: every surface shows up on the wall in its own colour with its number and name, and a white dot marks its top-left corner.
4. Drag each surface's corners onto its face. Corners snap onto each other, so neighbouring faces share an edge exactly. The numbers on the canvas are the angle at each corner and the length of each edge in pixels; **SQUARE IT UP** makes a wonky quad a true rectangle.
5. Overlay **OFF**. Now try, per surface: **EDGE GLOW** (a neon frame on the box edges, dark face), **SPANNED** on all three (one continuous picture wrapping round the box) or **WHOLE FRAME** (the full visual on every face), **Soft edge**, **Brightness** (dim the face that catches the most light).
6. A face that is not four-sided (a triangle, a sculpture facet): **MASK**, then **+ POINT** / drag the diamond points.
Everything outside the surfaces stays black, so nothing spills onto the wall behind.

## Recording the show
Hit the red **REC** button in the panel's top bar (or **Ctrl+R** on the projector window, when not calibrating). The dot blinks and the elapsed time and file size count up; press it again to stop. A few seconds later the MP4 is in `recordings\` (for example `show_20261125_213000.mp4`).

Settings live in **CONTROLLERS > RECORD**:
- **Source:** PROGRAM = the clean visual (best for sharing), PROJECTOR = exactly what is on the wall, mapping included.
- **Audio:** OFF, LAPTOP AUDIO (whatever the laptop is playing: Spotify, YouTube, a DJ set) or MIC.
- **30 / 60 fps**, **native or 1920x1080** (letterboxed), **quality** draft / standard / high.
- **Replay buffer:** switch it on and the last 10-120 s are always kept; **SAVE LAST N s** writes them out after something great happened.
- **OPEN FOLDER** opens `recordings\` on the laptop; the list shows the latest files with length and size.
- MIDI: in MIDI LEARN click REC, then hit a pad.

It never slows the show down: if the laptop can't keep up, it drops a frame rather than stutter. It warns when the disk is under 2 GB free and stops (saving what it has) under 0.5 GB. Quitting with Shift+Esc or a crash still saves the file.

## 3. Grow the AI texture library
```
.\.venv\Scripts\python.exe ai_gen.py --n 16
.\.venv\Scripts\python.exe ai_gen.py --prompt "your own vision here, glowing, black background" --n 6
```
Easier: use the panel's **AI DREAM** tab during the show. About 2.5 s per image, and the engine crossfades to it automatically.
Prompt tips: "black background", "glowing", "jewel tones", "intricate", "symmetrical" project best.

## Online assistant + Nano Banana (optional, OFF by default)
The show never needs the internet. If you want help from an AI while playing, press **ONLINE OFF** in the top bar so it
reads **ONLINE ON** (amber). Press it again and nothing leaves the laptop.

- **ASSISTANT tab** (and the small box on PERFORM): type what you want, e.g. "slower and deeper, emerald palette, bring
  in a jaguar". Claude answers in about 4 s with a list of changes.
  - **SUGGEST** (default): you see the list first; **APPLY** or **DISMISS**.
  - **AUTOPILOT**: it applies them straight away. Either way every batch has an **UNDO** in the list on the right.
  - It can move faders, pick palettes, scenes and images, switch layers, put words on the text layer and start AI images.
    It can never flash, strobe, black out, PANIC or touch the mapping; PANIC also cancels whatever it was doing.
- **AI DREAM > Image engine > NANO BANANA**: recognisable subjects (a Buddha, a woman, a jaguar) instead of textures.
  Paste your own Gemini API key (https://aistudio.google.com) into the box under the selector and press SAVE KEY.
  Prompts (and frames, for dream-from-frame) are sent to Google; usage limits apply. If it fails or you are
  offline, the local engine makes the image instead. Add "on a pure black background" to prompts.
- **Allow Google CLI tools** (checkbox on ASSISTANT, off by default): lets the plugin use your `agy` login for chat
  (about 14 s) and for images without an API key (about 40 s each). Read https://antigravity.google/terms first: they may
  not permit other programs driving the CLI. Your call.
- Everything it does is written to `logs\online_<date>.log`.

## 4. TouchDesigner (the deep, pro tool)
1. Open TouchDesigner from the Start menu, then sign in and activate a free **Non-Commercial** key (1280x1280 cap, which fits the BenQ).
2. Starter tutorials and step-by-step audio and mapping recipes: `research/td_audio_mapping.md`.
3. To bring this app's visuals into TD: run `live_visuals.py --spout`, then in TD add a **Syphon Spout In TOP** and pick `PsyProjector`.

## 5. Resolume Arena (VJ-style live mixing)
Demo mode (watermark). See `research/resolume_spout_mapping.md`. Its built-in web remote (Preferences > Webserver) also works from your phone.

## Suggested learning path
- **Day 1:** sections 1-3. Get a feel for folds, palettes and trails while music plays, and map one taped rectangle.
- **Day 2:** generate your own textures with your own prompts; tape 3-5 foam-core shapes or stack boxes and give each its own surface in MAPPING.
- **Week 1:** TouchDesigner basics (Noise > Feedback > Kaleidoscope > Out), audioAnalysis, Kantan Mapper.
- **Later:** train your own LoRA (`research/lora_dataset.md`), MIDI controller (`research/hardware_next.md`), Resolume sets.
