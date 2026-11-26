"""Shared contract between the render engine (live_visuals.py) and the control system (control.py).

The control system owns *how* values are produced (faders, modulation, BPM, scenes, autopilot, MIDI).
The render engine owns *what they look like*. Each frame:

    frame = controller.update(dt, bands)        # bands = np.array([bass, mid, high]) smoothed 0..1
    frame.values  -> {param_name: float}         # final, modulated, clamped values for every PARAMS entry
    frame.events  -> [("ripple", x, y), ...]     # one-shot events this frame (x, y in 0..1 content UV, origin bottom-left)
    frame.clock   -> {"bpm", "beat", "beat_phase", "bar_phase", "phrase_phase"}   # beat = float beats since start

The shader receives every PARAMS entry as a float uniform named "p_<name>" (unused uniforms are fine).
Do not rename existing params; add new ones at the end of a group.
"""
from dataclasses import dataclass, field

# name: (min, max, default, group, label)
PARAMS = {
    # ---- master / performance
    "master":     (0.0, 1.0, 1.0, "master", "Master"),
    "flash":      (0.0, 1.0, 0.0, "master", "Flash"),          # momentary envelope, 0 = off
    "strobe":     (0.0, 1.0, 0.0, "master", "Strobe"),         # 0 = off, >0 = on (rate set by control, gated by clock)
    "invert":     (0.0, 1.0, 0.0, "master", "Invert"),
    "freeze":     (0.0, 1.0, 0.0, "master", "Freeze"),         # 1 = hold the current frame (time stops, feedback holds)
    "blackout":   (0.0, 1.0, 0.0, "master", "Blackout"),       # 1 = full black (render engine fades ~100 ms)
    "punch":      (0.0, 1.0, 0.0, "master", "Zoom punch"),     # momentary envelope

    # ---- camera / space
    "zoom":       (-1.0, 1.0, 0.0, "space", "Zoom"),           # log scale, -1 = far out, +1 = deep in
    "pan_x":      (-1.0, 1.0, 0.0, "space", "Pan X"),
    "pan_y":      (-1.0, 1.0, 0.0, "space", "Pan Y"),
    "rotate":     (-1.0, 1.0, 0.1, "space", "Rotate speed"),
    "symmetry":   (0.0, 1.0, 1.0, "space", "Symmetry"),        # 0 = flat field crawling across the surface, 1 = full mandala
    "folds":      (2.0, 24.0, 8.0, "space", "Folds"),
    "speed":      (0.0, 2.0, 1.0, "space", "Speed"),           # global time scale
    "flow":       (-1.0, 1.0, 0.3, "space", "Crawl / flow"),   # pattern crawl speed + direction
    "flow_dir":   (0.0, 1.0, 0.0, "space", "Crawl angle"),     # 0..1 -> 0..360 deg for the flat field

    # ---- motion: travelling INTO the visual + serpents
    "travel":       (-1.0, 1.0, 0.0, "motion", "Travel into"),      # forward (+) / backward (-) flight speed, 0 = still
    "travel_mode":  (0.0, 3.0, 0.0, "motion", "Travel mode"),       # rounded: 0 zoom tunnel (log-polar), 1 3D tube tunnel,
                                                                    #          2 depth layers (parallax planes), 3 Droste spiral
    "tunnel_curve": (0.0, 1.0, 0.2, "motion", "Tunnel bend"),       # path curvature / camera sway while flying
    "depth_fog":    (0.0, 1.0, 0.5, "motion", "Depth fade"),        # distant geometry fades to black (sense of depth)
    "serpent":      (0.0, 1.0, 0.0, "motion", "Serpents"),          # amount: 0 off
    "serpent_mode": (0.0, 2.0, 0.0, "motion", "Serpent mode"),      # rounded: 0 segmented serpents slither over the scene,
                                                                    #          1 serpentine warp of the whole field, 2 both
    "serpent_count": (1.0, 8.0, 3.0, "motion", "Serpent count"),
    "serpent_segments": (4.0, 64.0, 24.0, "motion", "Segments"),     # body segmentation (beaded / scaled bands)
    "serpent_speed": (-1.0, 1.0, 0.3, "motion", "Slither speed"),
    "serpent_wiggle": (0.0, 1.0, 0.5, "motion", "Wiggle"),          # amplitude/frequency of the S-curve
    "serpent_size":  (0.0, 1.0, 0.4, "motion", "Serpent size"),

    # ---- contortion / shape
    "swim":       (0.0, 1.0, 0.3, "shape", "Swim"),            # domain-warp amount
    "ripple":     (0.0, 1.0, 0.5, "shape", "Ripple strength"),
    "ripple_decay": (0.0, 1.0, 0.5, "shape", "Ripple decay"),
    "twirl":      (-1.0, 1.0, 0.0, "shape", "Twirl"),
    "bulge":      (-1.0, 1.0, 0.0, "shape", "Bulge / pinch"),
    "wave":       (0.0, 1.0, 0.0, "shape", "Wave"),
    "density":    (0.0, 1.0, 0.5, "shape", "Density"),         # pattern scale

    # ---- colour
    "palette":    (0.0, 4.0, 0.0, "color", "Palette"),         # FRACTIONAL = glide between adjacent palettes (colour narratives)
    "hue":        (0.0, 1.0, 0.0, "color", "Hue shift"),
    "saturation": (0.0, 1.5, 1.0, "color", "Saturation"),
    "chase":      (0.0, 1.0, 0.5, "color", "Light chase"),     # TAS motion-key wave amount
    "chase_width": (0.05, 0.95, 0.25, "color", "Chase width"), # hue-window width: narrow = sharp line of light
    "chase_rate": (-4.0, 4.0, 1.0, "color", "Chase rate"),     # sweeps per bar (beat-synced); negative = inward
    "chase_mode": (0.0, 2.0, 0.0, "color", "Chase mode"),      # 0 single window, 1 dual, 2 dual counter-chase (rounded)
    "trails":     (0.0, 0.98, 0.85, "color", "Trails"),

    # ---- layers (opacity)
    "l_kene":     (0.0, 1.0, 1.0, "layers", "Kene patterns"),  # crawling labyrinth / Shipibo-Huni Kuin-inspired line fields
    "l_beads":    (0.0, 1.0, 0.7, "layers", "Beadwork"),
    "l_lines":    (0.0, 1.0, 0.5, "layers", "Sacred lines"),
    "l_ai":       (0.0, 1.0, 0.5, "layers", "AI textures"),
    "l_echo":     (0.0, 1.0, 0.6, "layers", "Echo"),
    "l_video":    (0.0, 1.0, 0.0, "layers", "Video loops"),    # pre-rendered clips (videos/ folder), Singer "machine 1"
    "luma_key":   (0.0, 1.0, 0.3, "layers", "Video luma key"), # keys out dark parts of the video so black stays black
    "l_fractal":  (0.0, 1.0, 0.0, "layers", "Math fractals"),   # pure-math generator layer (no AI): see fractal group
    "l_art":      (0.0, 1.0, 0.0, "layers", "Art pieces"),      # uploaded art as a flyable 3D relief (art_layer.ArtPass)

    # ---- pure-math fractal generators (math_layers.FractalPass)
    "fractal_mode":   (0.0, 5.0, 0.0, "fractal", "Fractal"),    # rounded: 0 crystal KIFS flythrough, 1 Mandelbox tunnel,
                                                                #  2 hyperbolic tiling, 3 Apollonian / circle inversion,
                                                                #  4 Julia orbit-trap jewels, 5 4D polytope (hypercube/600-cell)
    "fractal_morph":  (0.0, 1.0, 0.5, "fractal", "Morph"),      # slowly morphs the fractal's constants
    "fractal_detail": (0.0, 1.0, 0.5, "fractal", "Detail"),     # iterations / march steps (cost!)
    "fractal_glow":   (0.0, 1.0, 0.5, "fractal", "Glow"),       # edge / orbit-trap glow
    "fractal_metal":  (0.0, 1.0, 0.4, "fractal", "Iridescence"),# metallic / thin-film iridescent shading
    "fractal_res":    (0.25, 1.0, 0.5, "fractal", "Render scale"),  # internal resolution (perf safety)

    # ---- uploaded art: Jonathan Singer-style fly-through + distortion (art_layer.ArtPass)
    "art_index":   (0.0, 63.0, 0.0, "art", "Art piece"),        # rounded index into the art library (control sets it)
    "art_relief":  (0.0, 1.0, 0.4, "art", "Relief depth"),      # displacement from depth map (or luminance fallback)
    "cam_yaw":     (-1.0, 1.0, 0.0, "art", "Fly yaw"),
    "cam_pitch":   (-1.0, 1.0, 0.0, "art", "Fly pitch"),
    "cam_roll":    (-1.0, 1.0, 0.0, "art", "Fly roll"),
    "cam_dolly":   (-1.0, 1.0, 0.0, "art", "Fly in/out"),       # camera distance; travel also pushes the camera forward
    "cam_drift":   (0.0, 1.0, 0.3, "art", "Camera drift"),      # slow automatic orbit / handheld float
    "art_melt":    (0.0, 1.0, 0.0, "art", "Melt"),              # flow-field drip/drift of pixels
    "art_liquify": (0.0, 1.0, 0.0, "art", "Liquify"),           # swirling liquid displacement
    "art_slices":  (0.0, 1.0, 0.0, "art", "Depth slices"),      # segment by depth and pull layers apart (parallax)
    "art_breathe": (0.0, 1.0, 0.3, "art", "Breathe"),           # relief pulses with the beat/bass
    "art_fold":    (0.0, 1.0, 0.0, "art", "Kaleido fold"),      # mirror/fold the piece into symmetry
    "art_droste":  (0.0, 1.0, 0.0, "art", "Droste"),            # recursive picture-in-picture spiral
    "art_smear":   (0.0, 1.0, 0.0, "art", "Slit-scan smear"),   # time-displacement smear
    "art_glow":    (0.0, 1.0, 0.3, "art", "Edge glow"),         # luminous edges / rim light on the relief
}

TRIGGERS = ["flash", "strobe", "invert", "freeze", "blackout", "punch"]   # momentary buttons in the panel / MIDI
EVENTS = ["ripple"]                                                        # ("ripple", x, y)
MOD_SOURCES = ["none", "bass", "mid", "high", "lfo_sine", "lfo_tri", "lfo_saw", "lfo_square", "random"]


def defaults():
    return {k: v[2] for k, v in PARAMS.items()}


def clamp(name, x):
    lo, hi = PARAMS[name][0], PARAMS[name][1]
    return min(max(float(x), lo), hi)


@dataclass
class Frame:
    values: dict = field(default_factory=defaults)
    events: list = field(default_factory=list)
    clock: dict = field(default_factory=lambda: dict(bpm=124.0, beat=0.0, beat_phase=0.0, bar_phase=0.0, phrase_phase=0.0))


class StaticController:
    """Stand-in until control.py exists. control.Controller must provide the same API:

    update(dt, bands) -> Frame          called once per frame on the GL thread
    set(name, value)                    base (fader) value, clamped
    nudge(name, delta)                  base value += delta, clamped (keyboard)
    trigger(name, on: bool)             momentary TRIGGERS (press/release)
    ripple(x, y)                        queue a ripple event at content UV
    tap()                               tap tempo
    apply(cmds) -> list                 consume remote/panel commands it understands, return the rest
                                        (corner pin / grid / calib / next-texture / save stay with live_visuals)
    state() -> dict                     JSON-able snapshot for the panel (values, mods, clock, scenes, midi...)
    close()                             stop MIDI threads etc.
    """

    def __init__(self, bpm=124.0):
        self.bpm, self.beat, self.base, self.events = bpm, 0.0, defaults(), []

    def set(self, name, value):
        if name in PARAMS:
            self.base[name] = clamp(name, value)

    def nudge(self, name, delta):
        self.set(name, self.base.get(name, 0.0) + delta)

    def trigger(self, name, on):
        self.set(name, 1.0 if on else 0.0)

    def ripple(self, x, y):
        self.events.append(("ripple", float(x), float(y)))

    def tap(self):
        pass

    def apply(self, cmds):
        return list(cmds)

    def state(self):
        return dict(values=dict(self.base), bpm=self.bpm)

    def close(self):
        pass

    def update(self, dt, bands):
        self.beat += dt * self.bpm / 60.0
        b = self.beat
        ev, self.events = self.events, []
        return Frame(values=dict(self.base), events=ev,
                     clock=dict(bpm=self.bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0,
                                phrase_phase=(b / 32) % 1.0))
