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

**The panel tabs:** PERFORM (one screen: scene grid, macro faders, live layers, quick boxes) · MOTION (travel into the visual, serpents) · SHAPE (zoom, pan XY pad, twirl, bulge, ripples) · COLOR (palettes, TAS light chase) · LAYERS (kene, beads, lines, AI, video, fractals, art) · FRACTAL (pure-math modes) · ART (upload paintings, fly through them) · SCENES (16 slots, morph, autopilot) · AI DREAM (type a prompt) · CONTROLLERS (MIDI learn, gamepad, clock) · MAPPING (several surfaces: boxes, sculpture faces; corners, angles, masks, edge glow) · 3D DECKS (flown-through 3D landscapes on an A/B deck with blend modes, particle swarm) · MOD BUS (audio scope, 16-step sequencer, every active modulation). Press **?** in the panel for shortcuts.

**PERFORM, one screen (no scrolling on a 1920x1080 laptop; stacked on a phone or tablet):** master, BLACKOUT, PANIC, TAP / BPM and AUTOPILOT are the top bar right above it. **Scenes** (top left): the 16 pads (tap = recall, SAVE TO… or Shift-click / long-press = save), with morph length, quantize, and the autopilot period (4 to 128 bars) and order. **Macros** (top right): Zoom, Speed, Spin, Travel, Symmetry, Light chase, Trails, each with its MOD button (lit = modulated, teal line = live value), plus travel mode, palette swatches and a ripple pad; MORE CONTROLS… opens a drawer with the rarer faders. **Live layers**: only the layers that are showing, each with EYE + fader (tap the name to open that layer's tab); "+ N OFF" lists the rest (tap one to bring it in), + ADD opens BUILD. **Quick boxes** (HIDE / SHOW): Sources thumbnails, a text box, an AI prompt box (and the assistant). Esc closes a drawer. SIMPLE mode hides the jumps to other tabs.

**Layers card (top of LAYERS):** one row per layer with a live thumbnail. **EYE** shows / hides the layer on the projector, **SOLO** shows only the soloed layers in the panel's monitor without changing the projector, and the fader is the opacity. Uploaded art not showing? Its layer ("Art pieces") starts at 0: click a thumbnail in **Sources** and it fades in.

**Power switches:** every tab that changes the picture has a power icon on its tab button and a big ON / OFF switch at its top; a layer's **EYE** does the same for one layer, **ALL FX** for every effect. Off and on again brings back exactly the same settings. **PREVIEW** (top bar) is OFF by default to save GPU: switch it on to see the output monitor, solo layers and layer thumbnails in the panel.

**Also in the panel:** a small **output monitor** in the top bar (click it for a large one), a **Sources** picker on PERFORM / ART / AI DREAM (thumbnails of uploaded art and AI images: click one to use it), and a **scene crossfader** on SCENES and behind A / B FADER on PERFORM (click DECK A, click a scene, same for DECK B, then drag to blend the two looks by hand).

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
  Paste your own Gemini API key (https://aistudio.google.com) into the box under the selector and press SAVE KEY
  (it is kept in your Windows user profile, not in the show folder).
  Prompts (and frames, for dream-from-frame) are sent to Google; usage limits apply. If it fails or you are
  offline, the local engine makes the image instead. Add "on a pure black background" to prompts.
- **Allow images through the agy CLI** (checkbox on ASSISTANT, off by default): images without an API key (about 30 s
  each) through your `agy` login. It can only be ticked on the laptop itself (open the panel at address 127.0.0.1),
  never from a phone. Read https://antigravity.google/terms first: they may not permit other programs driving the
  CLI. Your call. Chat is Claude Code only.
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


## Start simple and build it up (BUILD tab)
Start on a clean black canvas with one moving stripe layer:
```
powershell -ExecutionPolicy Bypass -File run_show.ps1 --monitor 1 --panel --fresh
```
or press **NEW CANVAS** on the **BUILD** tab at any time (it crossfades; **RESTORE PREVIOUS** brings your earlier look
back exactly, nothing is lost).

- **+ ADD LAYER**: pick a generator (stripes, rings, spiral, rays, checker, waves, polygons, noise bands). Up to 8.
  Further down the same menu you can bring the engine's own layers back in gently (Folding Blocks, Beadwork, Beings,
  fractals, art, 3D landscapes, text...).
- Click a layer to shape it on the right. Every number can be dragged, typed, reset (the round arrow, or double-click
  the fader), modulated (**MOD**: bass, LFOs, sequencer) and MIDI-learned.
  - **Shape**: how many bands, band width, edge softness, waveform.
  - **Space**: straight, polar, tunnel, kaleidoscope, mirrors, perspective floor; direction, zoom, centre.
  - **Motion**: scroll speed (FREE = per second, BEAT = per beat), rotation, pulse, wave.
  - **Colour**: click a stop on the gradient to change its colour, drag it to move it, **+ STOP** to add one (2 to 6);
    or FOLLOW THE SHOW PALETTE. "Colour blend" 0 = each stripe takes the next colour, 1 = one smooth gradient.
  - **Blend**: add, screen, multiply, difference, mask (the layers below show only where this one is lit), replace, over.
- In the list: arrows reorder, EYE mutes, SOLO shows only that layer, DUP copies, the cross deletes (click twice).
  **UNDO** steps back.
- **SIMPLE / FULL** (top bar): SIMPLE keeps only BUILD, PERFORM and SCENES in view; FULL shows every tab.
- Save what you built into a scene slot as usual (SCENES): the stack is stored with it and morphs on recall.
- Safety: very wide bands cannot scroll faster than 2.5 per second and pulses never beat faster than that; the panel
  says "speed limited" when that holds a layer back.


## Crowd camera (CAMERA tab)
Dancers become part of the picture: their silhouettes, movement and arms drive the visuals. **Off by default**; it never
switches itself on, not even after a restart.
1. Open the **CAMERA** tab, press **CAMERA OFF (click = ON)**. The top bar shows a red blinking **CAMERA ON** while the
   webcam is in use; one click there is CAMERA OFF and frees the webcam.
2. Raise **Crowd camera** (`l_camera`, on the LAYERS tab and at the top of the CAMERA faders) and pick an **Integration**:
   Bodies reveal (visuals only on the people), Glow outline, Patterns inside (bodies carry a hue-turned pattern),
   Knockout (people black), Motion trails (silhouettes smear into the echo), Flow push (movement pushes the pattern).
   **Flow push** also works on top of any mode.
3. Align: Glow outline, stand in front of the wall, move Position / Scale / Keystone / Crop until the outline sits on you.
   MIRROR / AS SEEN depends on where the camera points. **PREVIEW: ON** shows a small mask + skeleton view (no camera image).
4. Dancer effects: **Hand ripples**, **Beings follow hands** (negative = they flee), **Hands pull swarm** (needs the
   particle swarm up), **Serpent on arm** (needs serpents up; serpent 1 rides the most raised arm).
5. Any fader can follow the crowd: open its **MOD** editor and pick `crowd_motion`, `crowd_left` / `centre` / `right`,
   `crowd_presence`, `hand_height`, `hands_spread`, `hip_sway` or `hand_speed`. The meters on the CAMERA tab show them live.

Frames are analysed in memory by a separate low-priority process and are never saved, sent or recorded. A dark room lowers
the webcam's own frame rate (about 10 fps measured at night): more light on the dancers = smoother tracking.


## Filters + Dreamify (FILTERS tab, DREAM buttons)
Post-effects on the finished picture, before mapping (every surface, Spout, the recording and the panel monitor get them).
1. **FILTERS** tab -> **+ ADD FILTER** -> pick one: Deep dream, Oil paint, Halftone, Neon edges, Prism split, Posterize,
   Pixel sort, Glitch, Bloom, Mirror / kaleido. It comes in with its amount up; drag the faders under its name.
2. Up to 6 filters run top to bottom: **▲ / ▼** reorder, **✕** takes one out (its settings stay for next time).
   **RESET CHAIN** (tap twice) empties the chain and puts every filter setting back to default.
3. The power button on the FILTERS tab (and **ALL FX**) switches every filter off and back on to the same settings.
   With no filter running the pass is skipped completely: no cost at all.
4. Every filter fader works with **MIDI LEARN** and **MOD** like any other; scenes store and morph the chain.
   Filters that are not in the chain: open "All filter params" at the bottom of the tab.
5. Glitch never fires more than 2.5 times a second and every glitch fades in and out (photosensitive-safe).

**DREAMIFY** (real DeepDream, offline): on the ART, AI DREAM or PERFORM sources picker every image has a small **DREAM**
button. Pick a preset (**Dog-slugs**, **Eyes**, **Architecture**, **Psychedelic swirl**), a strength, **DREAM IT**.
A progress badge (bottom right, with CANCEL) shows while it runs, roughly half a minute to a minute and a half per image.
The result is a NEW image next to the original (art pieces stay art pieces, AI textures stay AI textures) and appears
in the picker by itself. Needs the AI worker (started by run_show.ps1). The first use downloads the 50 MB GoogLeNet
weights once (internet needed that one time).


## AUTO MAP (MAPPING tab): let the camera find the boxes
1. Point a camera at the wall (turn the laptop lid toward it, or a USB webcam on a fixed mount) and switch the CAMERA on.
2. MAPPING tab, AUTO MAP card: **1 CHECK CAMERA** (does it see the whole projection?), **2 CALIBRATE** (black-and-white
   test patterns for about 30 s: do it before people arrive), **3 DETECT SURFACES** (faces are outlined and listed; click a
   face in the picture to add one), tick / name them, **CREATE SURFACES**, then **4 VERIFY & REFINE** (moves the corners
   onto the real edges). UNDO puts things back; afterwards edit corners by hand as usual.

## Real kene layers (generative studies, off by default)
The old "kene" layer is now called **Folding Blocks** and is unchanged. Two new layers are studies of the real visual
grammar documented in `research/kene_reference/` (not copies, and not to be called authentic kené):
- **BUILD > + ADD LAYER > "Kené study (Shipibo-Konibo inspired)"** or **"Bead study (Huni Kuin inspired)"**, or raise them on LAYERS.
- Tab **KENÉ STUDY**: Medium (light on black, mud-dyed cloth, embroidery, ceramic, beadwork), Colours (first = follow the
  show palette), Field (single panel, panels, band, medallion, follow the show's mandala / tunnel), Design 1-32 and
  **NEW DESIGN**, Pen ("sing it in" over 4-32 bars, a new design each time), Flowing light, Breathing; for the beads:
  motif, medium (glass beads / woven / painted), colours, motif morph, beads lighting in sequence.
- For mapping a box face or a strip: Field = Single panel or Band (they stay still and fill the surface cleanly).
- Credit: "Generative study inspired by Shipibo-Konibo kené / Huni Kuin kenê kuin. For public shows, credit and
  collaborate with the originating communities — see research/kene_reference."
- Stills: `test_frames/kene_real/contact_sheet.png` and `reference_pairs.png`.
- **With the laptop's own webcam:** pick BUILT-IN WEBCAM in the AUTO MAP card, turn the laptop so the lid faces the wall
  (it can sit next to the projector), run the panel from your phone, press LIVE FRAMING and follow the hints (tilt up /
  down, turn left / right) until it says "good". Mirroring and the wide-angle lens are handled. If it cannot see all of
  the projection: calibrate POSE 1, turn the laptop to see the rest (with some overlap), POSE 2, calibrate again.
- **AI assist** (needs ONLINE and your Gemini key on AI DREAM): type what to look for, AI: FIND SURFACES or AI: CHECK
  OUTLINES, look at the exact picture that would be sent (people greyed out), then SEND or DISCARD.

## Shows, presets and the timeline (save it all, play it back)
- **SHOW** (top bar) = the whole project in one place: look, 16 scenes, timelines, settings (BPM, clock, morph, autopilot,
  camera + governor settings), recorder settings, MIDI map and mapping. **Ctrl+S** saves; **SAVE AS** names it. Shows live in
  `shows\<name>\show.json`, every save keeps the previous version in `backups\`, and unsaved changes autosave about 15 s
  after you stop editing (the list then offers **RECOVER** after a crash). **OPEN** loads the parts you tick; the venue rig
  (MIDI map, mapping) is left alone unless you tick it. **NEW SHOW**: clean canvas, factory looks, or "from what's on now";
  **NEW FROM IT** uses any saved show as a template. **EXPORT** downloads one `.entheo` file (with "collect media" it carries
  the art / textures / audio too); **IMPORT .entheo** brings one in. Files are pure data and are checked on the way in.
- **PRESETS** (top bar) = smaller pieces: a LOOK, one LAYER, the EFFECTS chain, a MOD routing, a MAPPING, a MIDI map. Type a
  name, SAVE PRESET (a thumbnail is grabbed from the output). Click a card to recall it; tick groups first for a partial recall
  (only COLOR, say), pick CUT or a morph length. Star = favourite, search finds names and tags, `...` = rename, tags, export
  (a small `.json` to share), **+ CUE** (put it on the timeline), delete. **IMPORT PRESET** loads someone's `.json`.
- **TIMELINE tab**: a playhead in bars (follows TAP / MIDI clock / Link) or seconds. Tracks: Sections (markers), Cues (scene or
  preset recalls with a morph), automation lanes (any fader), triggers (flash / punch / freeze ... with a length), events
  (layer or tab on / off, next video, texture / art picks), text. Double-click a lane to add, drag to move, Shift-drag on the
  ruler for the loop range. **REC** + PLAY records what you do live (fader moves, scene recalls, triggers, layer switches).
  While it plays, touching a fader the timeline drives hands it to you (TOUCH: back after 1.5 s; LATCH: until
  **RE-ENABLE AUTOMATION**); **RELEASE** gives every fader back. PANIC stops the timeline. Flashes it plays are held to
  <= 2.5 per second and brightness lanes cannot swing faster than 0.2 s.
- Optional **audio**: + AUDIO uploads a song (`shows\_media\`), pick it in the list, then press **AUDIO HOST** on the ONE device
  that should play it (the laptop with the speakers). Use SECONDS units for a song with a fixed timing.

## Flash guard (photosensitive limiter, always on)
The engine will not put out more than 3 flashes a second over a quarter of the picture, whatever you press: strobe at
1/8 or 1/16, flash, invert, blackout hammering, rapid cuts, glitch. Above the limit it holds the picture at its dark
phase instead of letting it flash; the picture never stops. Red flashing is held earlier.
- **Top bar:** FLASH GUARD is green while it leaves the picture alone, amber "HOLDING" with a count while it steps in.
  Click it for the card in CONTROLLERS (flashes per second coming in and going out, the white mark is the limit).
- **A 1/4 strobe is fine** up to 180 BPM (3 per second). Faster strobes are what gets held.
- **Override for one show** ("the audience has been warned"): CONTROLLERS > Flash guard > hold the button for 2 s,
  then press YES, AUDIENCE WARNED. A red FLASH GUARD OFF badge stays on screen; tap it to switch the guard back on.
  It is back on by itself after every restart.
- Check it without a projector: `.\.venv\Scripts\python.exe flashguard.py --demo` (before / after counts) and `--bench`.

## Skins: put your own images on the worlds (3D DECKS and MOTION tabs)
Any image in the library (uploaded art, an AI texture, a dreamified picture) can become the surface of a world.
1. **3D DECKS tab > SKINS card.** Each world has slots: Sacred corridor (walls, floor), Torus wormhole (tube, chambers),
   Prismatic vortex (walls), Hex domes (domes). **MOTION tab > SKINS card** has the travel tunnel (walls).
2. **Assign:** drag a thumbnail from any Sources picker (or the strip in the card) onto a slot. On a touch screen: tap the
   slot, then tap an image. PROCEDURAL on a slot goes back to the built-in look (its settings are kept).
3. **Shape it** (tap a slot first): Mapping (Surface / Triplanar / Screen; the tunnel has Follow the travel mode / Log-polar /
   Tube / Screen), Mix with the pattern (Replace, Multiply, Screen, Overlay, Lines reveal the skin), Tiling (Mirror =
   seamless with any picture, Repeat = for AI textures, which already tile), Amount, Scale, Rotation, Scroll speed +
   direction (Free or On the beat), Follow the show palette, Brightness, Tint.
4. **GENERATE A SKIN:** type what you want ("molten gold filigree"), press the button. The local AI worker makes a seamless
   tile when it has nothing else to do and the picture lands on the selected slot by itself.
5. Every fader has a MOD button and is MIDI-learnable (`skin:<world>:<slot>:<field>`). Skins are saved with scenes, look
   presets and show files. PANIC puts every slot back to procedural.
- **Cost:** a world with no skin costs exactly what it did before. A skin is one texture fetch per pixel. Triplanar is
  three fetches: use it when Surface stretches too much, not by default.
- **Memory:** 8 skins are kept on the GPU (512 px by default, about 11 MB; `--skin-size 1024` for 43 MB). Under load the
  governor switches skins off (critical) until the show recovers; the card says so.
- Look-check without the projector: `.\.venv\Scripts\python.exe materials.py --gallery` -> `test_frames/skins/contact_sheet.png`.
