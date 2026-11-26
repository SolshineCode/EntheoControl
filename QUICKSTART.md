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

**The panel tabs:** PERFORM (main faders + layers + scenes) · MOTION (travel into the visual, serpents) · SHAPE (zoom, pan XY pad, twirl, bulge, ripples) · COLOR (palettes, TAS light chase) · LAYERS (kene, beads, lines, AI, video, fractals, art) · FRACTAL (pure-math modes) · ART (upload paintings, fly through them) · SCENES (16 slots, morph, autopilot) · AI DREAM (type a prompt) · CONTROLLERS (MIDI learn, gamepad, clock) · MAPPING (corner pin) · 3D DECKS (flown-through 3D landscapes on an A/B deck with blend modes, particle swarm) · MOD BUS (audio scope, 16-step sequencer, every active modulation). Press **?** in the panel for shortcuts.

**Layers card (top of PERFORM and LAYERS):** one row per layer with a live thumbnail. **EYE** shows / hides the layer on the projector, **SOLO** shows only the soloed layers in the panel's monitor without changing the projector, and the fader is the opacity. Uploaded art not showing? Its layer ("Art pieces") starts at 0: click a thumbnail in **Sources** and it fades in.

**Also in the panel:** a small **output monitor** in the top bar (click it for a large one), a **Sources** picker on PERFORM / ART / AI DREAM (thumbnails of uploaded art and AI images: click one to use it), and a **scene crossfader** on PERFORM / SCENES (click DECK A, click a scene, same for DECK B, then drag to blend the two looks by hand).

Every fader has a **MOD** button: make it follow the bass, mids, highs, a beat-synced LFO or random.

**Projector-window keys** (click the projector window first; the panel is easier): `1`-`9` folds · `M` layer preset · `P` palette · `TAB` flat/mandala · `N` travel mode · `E` serpents · `-`/`=` zoom · hold `X` flash, `Z` strobe, `I` invert, `Q` freeze, `K` blackout, `J` punch · `,` tap tempo · `Ctrl+C` calibrate · `Ctrl+G` grid · `Ctrl+Backspace` reset · `Ctrl+W` panel server.

**First exercise:** tape a big rectangle on the wall (or hold up a foam board). In the panel's MAPPING tab (or Ctrl+C on the projector window), drag the 4 corners onto its corners, then save. Now the visuals live exactly on that object.

## 3. Grow the AI texture library
```
.\.venv\Scripts\python.exe ai_gen.py --n 16
.\.venv\Scripts\python.exe ai_gen.py --prompt "your own vision here, glowing, black background" --n 6
```
Easier: use the panel's **AI DREAM** tab during the show. About 2.5 s per image, and the engine crossfades to it automatically.
Prompt tips: "black background", "glowing", "jewel tones", "intricate", "symmetrical" project best.

## 4. TouchDesigner (the deep, pro tool)
1. Open TouchDesigner from the Start menu, then sign in and activate a free **Non-Commercial** key (1280x1280 cap, which fits the BenQ).
2. Starter tutorials and step-by-step audio and mapping recipes: `research/td_audio_mapping.md`.
3. To bring this app's visuals into TD: run `live_visuals.py --spout`, then in TD add a **Syphon Spout In TOP** and pick `PsyProjector`.

## 5. Resolume Arena (VJ-style live mixing)
Demo mode (watermark). See `research/resolume_spout_mapping.md`. Its built-in web remote (Preferences > Webserver) also works from your phone.

## Suggested learning path
- **Day 1:** sections 1-3. Get a feel for folds, palettes and trails while music plays, and map one taped rectangle.
- **Day 2:** generate your own textures with your own prompts; tape 3-5 foam-core shapes and map them.
- **Week 1:** TouchDesigner basics (Noise > Feedback > Kaleidoscope > Out), audioAnalysis, Kantan Mapper.
- **Later:** train your own LoRA (`research/lora_dataset.md`), MIDI controller (`research/hardware_next.md`), Resolume sets.
