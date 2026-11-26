# EntheoControl

A live, audio-reactive visual engine for psychedelic projection mapping. It runs on a modest laptop GPU and is controlled entirely from one browser panel.

The visual language draws on visionary art and live VJ work: kene-inspired line labyrinths, swimming beadwork, light chases, tunnels you fly into, pure-math fractals, and fly-throughs of 2D artwork turned into 3D relief.

It was built for and tested on a GTX 1650 Ti (4 GB) driving a 1280x800 short-throw projector.

## Features
- **One engine:** `live_visuals.py` uses moderngl with GLSL 330 shaders. It runs as a borderless window on the projector, so you can keep using the laptop while the show runs.
- **One control surface:** `panel.html` is served by `remote_control.py` on `localhost:8765`, behind a per-install token. Its faders are generated from the `params.py` registry.
- **Layers:**
  - kene pattern fields, beadwork and sacred lines
  - AI textures and video loops
  - segmented serpents
  - six fractal modes: KIFS, Mandelbox, hyperbolic tiling, Apollonian, Julia and a 4D polytope
  - art fly-through, where your own images become depth-mapped 3D reliefs you can fly around and distort
- **Motion:**
  - four travel-into-the-visual modes: zoom tunnel, 3D tube, ground plane and depth layers
  - twirl, bulge, wave and ripples
  - a TAS-style hue-window light chase
- **Performance:**
  - a mod matrix on every fader: audio bands, beat-synced LFOs and random
  - tap tempo, MIDI clock and Ableton Link
  - 16 scenes with quantized morphs, plus autopilot
  - flash, strobe, freeze, blackout and a PANIC button
  - MIDI learn, with default maps for the APC mini mk2 and Launch Control XL
  - gamepad flying
- **Mapping:** corner-pin keystone that you drag from the panel.
- **Local AI (optional):**
  - `ai_worker.py` runs SD-Turbo for prompt-to-texture in about 2.5 s on a 4 GB card.
  - It can also dream from the current frame.
  - Depth Anything V2 Small makes the depth maps for uploaded art.
- **Output:** Spout output (`--spout`) for TouchDesigner or Resolume.
- **Reliability:** a crash watchdog (`run_show.ps1`) restarts the show and resumes the last live state.

## Quick start (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # install CUDA torch from pytorch.org first if you want the AI features
powershell -ExecutionPolicy Bypass -File run_show.ps1
```
- **Projector:** extend your desktop onto it first. The show opens on monitor 1 and the panel opens in your browser.
- **Quit:** Shift+Esc.
- **More:** see [QUICKSTART.md](QUICKSTART.md) for setup, panel tabs, keys and a first mapping exercise.
- **Headless checks:**
  - `python live_visuals.py --test 90` renders 90 frames and reports fps and frame health.
  - `--gallery` renders sample stills.
- **Tests:** `python -m pytest -q`

## Respect and rights
The kene-style patterns are generative and *inspired by* Shipibo-Konibo, Huni Kuin and Yawanawá design traditions. They are not copies of specific designs.

For public shows, credit these traditions, and where possible collaborate with or commission Indigenous artists. Use only artwork you own, have licensed, or that is in the public domain.

## License
[AGPL-3.0](LICENSE). Modified versions you distribute or run as a network service must make their source available.
