# Walkthrough: your first hour

This takes you from a box with a projector in it to a short recorded show, in about 60 minutes. Each step is "do this, and you should see that". If something does not match, the [reference manual](QUICKSTART.md) has the details and a troubleshooting table.

You will need: the laptop with EntheoControl installed (see the [README](README.md)), a projector and HDMI cable, some music, a wall, and a cardboard box. A phone is handy for step 9.

---

## 1. Set up (5 min)

1. Put the projector square to a blank wall and plug it into the laptop. Plug the laptop into its **stock charger**, directly, not through a dock.
2. Press **Win + P** and choose **Extend**.
   You should see the projector light up with your desktop wallpaper, not a copy of your screen.
3. In the projector's menu, turn **Auto Keystone off** and set keystone to 0. In Windows display settings, set the projector to its native resolution.
4. Focus it:
   ```powershell
   cd C:\path\to\EntheoControl
   .\.venv\Scripts\python.exe test_pattern.py --monitor 1
   ```
   You should see a grid, three circles and a crosshair. Turn the focus ring until the fine line pairs in the corners are crisp, then press Esc.
5. Optional, but worth it for any show longer than an hour: set **Maximum processor state to 99 %** (Power Options) and, in the NVIDIA Control Panel for `python.exe`, **Max Frame Rate 60**, **WhisperMode off** and **Threaded optimization off**.

## 2. Start the show (2 min)

```powershell
powershell -ExecutionPolicy Bypass -File run_show.ps1
```
You should see the mandala appear on the wall and the **control panel** open in your browser. Play some music on the laptop: the picture breathes with the bass, and the three meters in the panel's top bar move.

Two things to know right away:
- **Quit** with **Shift+Esc** in the projector window. A single Esc never ends a show.
- If anything crashes, the show restarts within a second with the same look.

## 3. Learn PERFORM (8 min)

PERFORM is the screen you play from. Everything fits on one laptop screen.

1. **Top bar**: find **Master**, **BLACKOUT**, **PANIC**, **TAP** and the BPM. Tap TAP in time with the music four times.
   You should see the BPM change and the beat lights follow the music.
2. **PANIC**: press and hold it for half a second.
   You should see the picture jump straight to the safe look. Remember this: holding **Esc** does the same from anywhere.
3. **Macros**: drag **Zoom**, **Speed**, **Spin** and **Symmetry**. Symmetry at 0 turns the mandala into a flat field that crawls across the whole wall.
4. Tap the **palette swatches**, then tap and drag on the **ripple pad**.
   You should see ripples spread from where you touched.
5. **Trigger row** at the bottom: hold **FLASH**, then **STROBE**, then **FREEZE**.
6. Click **MOD** under Zoom, pick **bass** and an amount.
   You should see Zoom pulse with the kick, and a teal line on the fader showing the live value.

## 4. Shape the look (7 min)

1. **SHAPE** tab: try **Twirl**, **Bulge / pinch** and **Wave**. Tap the ripple pad here too.
2. **COLOR** tab: raise **Light chase**.
   You should see a window of light sweep across the pattern in time with the bar.
3. **MOTION** tab: pick a **Travel mode** (zoom tunnel, tube, depth layers, Droste) and push **Travel into**.
   You should feel like you are flying into the picture. Add **Spin** to barrel-roll.

## 5. Layers and power buttons (5 min)

1. Back on **PERFORM**, look at **Live layers**: Folding Blocks, Beadwork, Sacred lines, AI textures, Echo and Math stack are showing.
2. Click a layer's **EYE**.
   You should see that layer fade off the wall. Click again and it comes back exactly as it was.
3. Click **+ N OFF** and bring in **Math fractals**.
4. Look at the tab strip: tabs that change the picture have a **power icon**. Click the one on **COLOR**.
   You should see the colour effects switch off and the tab struck through. Click again to restore them.
5. **ALL FX** switches every effect off at once. Try it, then switch it back.

Nothing you switch off is lost. Power buttons never edit your faders.

## 6. Art, AI and skins (8 min)

1. **ART** tab: drop a picture you own into the **Art library**.
   You should see its thumbnail appear. A depth map is made in the background.
2. Click its thumbnail in **Sources**. Try the **Art use** buttons: Fly-through relief, Inside the lines, Kaleidoscope, Tunnel walls, Displace pattern, Full frame melt. Drag the **Fly pad** to fly around the relief.
3. **AI DREAM** tab: type *glowing jade lotus, sacred geometry, black background* and press **GENERATE**.
   You should see a new image appear in the results after a few seconds.
4. Press the small **DREAM** button on any thumbnail, pick **Eyes** and **DREAM IT**. It runs in the background (up to a minute or so), and a new dreamified copy appears in Sources.
5. **3D DECKS** tab: raise **3D landscapes**. In the **SKINS** card press **SKINS: OFF** to switch skins on, then drag your AI image onto **Sacred corridor: walls**.
   You should see your image become the walls of the corridor you are flying through.

## 7. Beings, kené study and text (6 min)

1. **MOTION** tab, **Beings** card: raise **Beings**, pick **Whale**, then **Jellyfish**. Raise **Variety** to mix forms and try a **Colour mode**.
2. **KENÉ STUDY** tab: raise **Kené study (Shipibo-Konibo inspired)**, set **Pen** to *Draw in: 8 bars* and press **NEW DESIGN**.
   You should see a design drawn line by line over eight bars. Read the credit card on this tab: these are generative studies, not authentic kené, and public shows should credit the originating communities.
3. **TEXT** tab: type a word, press **SHOW**, raise **Text**. Try **Movement** = Tunnel spiral and **Integration** = Woven.

## 8. Camera fun (4 min)

1. **CAMERA** tab: press **CAMERA OFF (click = ON)**.
   You should see a red blinking **CAMERA ON** in the top bar.
2. Raise **Crowd camera**, choose **Glow outline**, and stand in front of the wall. Adjust **Position** and **Scale** until the outline sits on you.
3. Raise **Hand ripples** and wave.
   You should see ripples come off your hands.
4. Switch the camera **off** again (click the red CAMERA ON). Frames never leave the laptop and are never recorded.

## 9. Map a box (8 min)

**By hand first.**
1. Put a cardboard box in front of the wall so the projector sees its front, one side and the top.
2. **MAPPING** tab: click **+ ADD** until you have three surfaces, and name them ("front", "side", "top").
3. Set "Show on the projector" to **ALL** and switch **TEST GRID** on.
   You should see three numbered outlines in different colours on the wall.
4. For each surface, drag its four corners on the canvas onto that face's corners, watching the wall. Corners snap to each other so the faces share edges. Use the arrow keys for single pixels.
5. Overlay **OFF**. Try **EDGE GLOW** on all three, then **SPANNED** on all three.
   You should see the visuals only on the box, black everywhere else, with one continuous picture wrapping round it.

**Then let the camera do it (optional).** Turn the laptop lid toward the wall (or use a USB webcam), switch the camera on, open the panel on your phone, and in the **AUTO MAP** card pick **BUILT-IN WEBCAM**, press **LIVE FRAMING** and follow the hints. Then **1 CHECK CAMERA**, **2 CALIBRATE**, **3 DETECT SURFACES**, **CREATE SURFACES**, **4 VERIFY & REFINE**. Buttons that flash patterns on the wall need a second tap. **UNDO** puts your hand-made surfaces back.

## 10. Scenes (4 min)

1. Make a look you like. On **SCENES**, choose **SAVE TO…** and click pad 9. Make a different look and save it to pad 10.
2. Set **Morph** to 4 bars and **Quantize** to BAR, then click pad 9.
   You should see the look glide back over four bars, starting on the next bar.
3. Click **SET SAFE…** and then pad 9: that is now where PANIC goes.
4. Switch **AUTOPILOT** on (top bar) and watch it step through your scenes. Switch it off again.

## 10b. Themes and sound tags (5 min)

1. Open **LAYERS**, make the beings look the way you like, and tap the small camera on the Beings row.
   You should see a new card in the THEMES box on PERFORM, named "Beings", the time and the palette.
2. Change the beings completely. Tap the card: they glide back on the next bar. Change them again and **hold** the card: they snap back at once. Nothing else on the wall moved.
3. Play a song through the laptop. Open the **THEMES** tab, select the card and press **TAG** (10 s). Let it listen.
4. Press **LEARN / TEST** and watch the meter: high while that song plays, low for other music.
5. Switch **SOUND TAGS ARMED** on, change the beings, play another song, then go back to the first one.
   You should see the beings come back by themselves a few seconds after the first song returns. Touch any fader and it holds off for 20 s: you always win.
6. Tap the star to keep the theme in PRESETS. Switch SOUND TAGS off again for now.

## 11. Timeline (5 min)

1. Open **TIMELINE**. Double-click the **Sections** row to mark "intro", "build" and "drop".
2. Double-click the **Cues** row near "build" and choose scene 10.
3. Add an automation lane for **Zoom** and double-click it to set two points.
4. Press **PLAY**.
   You should see the playhead move in bars, scene 10 arrive at the cue and Zoom follow the lane.
5. Grab the Zoom fader while it plays: it is yours until you let go, then the timeline takes it back (or press **RELEASE**). **PANIC** always stops the timeline.

## 12. Record (3 min)

1. **CONTROLLERS > Record**: Source **PROGRAM**, Audio **LAPTOP AUDIO**.
2. Press **REC** in the top bar, press **PLAY** on the timeline, and perform for a minute: a scene change, a strobe, a ripple.
3. Press **REC** again.
   You should see the file in `recordings\` a few seconds later (**OPEN FOLDER** opens it).

## 13. Save the show (2 min)

1. Top bar **SHOW** > type a name > **SAVE AS**. From now on, **Ctrl+S** saves.
2. Save a look you like as a preset: **PRESETS** > type a name > **SAVE PRESET**.
3. For a backup on a USB stick: SHOW > **EXPORT** with "collect media" ticked gives you one `.entheo` file.

If the laptop ever crashes, open **SHOW** and press **RECOVER**.

## 14. Before a real gig (3 min)

Open **CONTROLLERS** and look at three cards:
- **Health > Power**: pick **BALANCED**, and on battery set the **Show length**. The checklist under it repeats the Windows and NVIDIA settings from step 1.
- **Health**: governor on **SUGGEST** (or AUTO if you will not be watching the laptop).
- **Flash guard**: green. Leave it on.

Then print [docs/SHOW_DAY_CHECKLIST.md](docs/SHOW_DAY_CHECKLIST.md) and run through it at the venue. Practise holding PANIC until it is muscle memory.

You are ready. Quit with **Shift+Esc**.
