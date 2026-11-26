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
    "l_kene":     (0.0, 1.0, 1.0, "layers", "Folding Blocks"),  # crawling stepped labyrinth (renamed 2026-11-25: it is NOT real kene; internal name kept for MIDI/state compat)
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

    # ---- TouchVDMX merge (fx_stack.LandPass), APPENDED 2026-11-25: raymarched 3D landscapes on an A/B deck with
    #      blend modes + a curl-noise particle swarm. Both layers default to 0 (off = no cost).
    "l_land":       (0.0, 1.0, 0.0, "layers", "3D landscapes"),
    "l_swarm":      (0.0, 1.0, 0.0, "layers", "Particle swarm"),
    "land_a":       (0.0, 3.0, 0.0, "land", "Deck A"),          # rounded: 0 sacred corridor, 1 torus wormhole,
    "land_b":       (0.0, 3.0, 3.0, "land", "Deck B"),          #          2 prismatic vortex, 3 hex dome
    "land_xfade":   (0.0, 1.0, 0.0, "land", "A / B crossfade"), # 0 = deck A only (deck B is not rendered)
    "land_blend":   (0.0, 7.0, 2.0, "land", "A / B blend"),     # rounded: 0 mix, 1 add, 2 screen, 3 difference,
                                                                #          4 multiply, 5 exclusion, 6 divide, 7 luma mask
    "land_warp":    (0.0, 1.0, 0.2, "land", "A warps B"),       # cross-modulation: deck A's picture bends deck B's rays
    "land_twist":   (-1.0, 1.0, 0.25, "land", "Twist"),
    "land_curve":   (0.0, 1.0, 0.4, "land", "Curvature"),
    "land_morph":   (0.0, 1.0, 0.3, "land", "Geo morph"),
    "land_fov":     (0.0, 1.0, 0.45, "land", "Field of view"),  # 45..125 degrees; punch widens it (warp jump)
    "land_cam":     (0.0, 2.0, 0.0, "land", "Camera"),          # rounded: 0 cruise, 1 orbit, 2 audio dive
    "land_style":   (0.0, 2.0, 0.0, "land", "Torus glass"),     # rounded: 0 electric chapel, 1 floral glass, 2 neon gears
    "land_mirror":  (0.0, 1.0, 0.0, "land", "Fold invert"),     # rounded: mirror-fold the surface pattern
    "land_disp":    (0.0, 1.0, 0.25, "land", "Prism split"),    # chromatic dispersion of the landscape picture
    "land_res":     (0.25, 1.0, 0.5, "land", "Render scale"),   # internal resolution (perf safety)
    "swarm_vortex": (0.0, 1.0, 0.35, "land", "Swarm vortex"),   # swirl attractor strength
    "swarm_turb":   (0.0, 1.0, 0.4, "land", "Swarm turbulence"),
    "swarm_trail":  (0.0, 1.0, 0.6, "land", "Ribbon trails"),   # persistence of the filament trails
    "swarm_tint":   (0.0, 1.0, 0.0, "land", "Swarm gold tint"), # 0 = show palette, 1 = warm gold filaments

    # ---- how the chosen image is used (APPENDED 2026-11-25). Amount = l_art. The image is the uploaded art piece
    #      (art_index) or the AI texture on show, per art_source. Pick by name: engine command "art_pick" (HANDOFF.md).
    "art_use":      (0.0, 5.0, 0.0, "art", "Art use"),          # rounded: 0 fly-through relief, 1 inside the pattern lines
                                                                #  (stencil), 2 kaleidoscope source, 3 tunnel walls,
                                                                #  4 displaces the pattern, 5 full frame + feedback melt
    "art_source":   (0.0, 1.0, 0.0, "art", "Image source"),     # rounded: 0 uploaded art piece, 1 AI texture

    # ---- barrel roll around the flight axis (APPENDED 2026-11-25): works in every travel mode and stacks with rotate
    "spin":         (-1.0, 1.0, 0.0, "motion", "Spin"),         # roll speed (bipolar); the engine integrates the phase
    "spin_twist":   (0.0, 1.0, 0.0, "motion", "Spin twist"),    # corkscrew: the far end of the tunnel leads the spin

    # ---- beings layer (beings.BeingsPass), APPENDED 2026-11-25: living / supernatural creatures as luminous line-art.
    #      Off by default (l_beings 0 = the pass is not rendered). Form names: OPTIONS["being"] below.
    "l_beings":     (0.0, 1.0, 0.0, "layers", "Beings"),
    "being":        (0.0, 16.0, 0.0, "beings", "Being"),        # rounded index into OPTIONS["being"]
    "being_count":  (1.0, 8.0, 3.0, "beings", "How many"),      # rounded
    "being_size":   (0.0, 1.0, 0.5, "beings", "Size"),
    "being_speed":  (0.0, 1.0, 0.4, "beings", "Speed"),         # travel + body rhythm (0 = they stay put, still breathing)
    "being_wiggle": (0.0, 1.0, 0.5, "beings", "Wiggle"),        # undulation / sway of bodies, tails, tentacles
    "being_glow":   (0.0, 1.0, 0.5, "beings", "Glow"),          # halo + stroke brightness
    "being_detail": (0.0, 1.0, 0.5, "beings", "Ornament"),      # density of the inner beadwork / labyrinth patterning
    "being_mix":    (0.0, 1.0, 0.0, "beings", "Variety"),       # 0 = all the chosen form, 1 = every being a different form

    # ---- text layer (text_layer.TextPass), APPENDED 2026-11-25: words typed in the panel, SDF-rendered. Off by default
    #      (l_text 0 = the pass is not rendered). The words / font / phrase list travel by command, not by param
    #      (control command "text", see HANDOFF.md). Mode names: OPTIONS below.
    "l_text":        (0.0, 1.0, 0.0, "layers", "Text"),
    "text_size":     (0.0, 1.0, 0.45, "text", "Size"),            # letter height, log scale (about 3 % .. 60 % of the frame)
    "text_x":        (-1.0, 1.0, 0.0, "text", "Position X"),
    "text_y":        (-1.0, 1.0, 0.0, "text", "Position Y"),
    "text_track":    (-0.2, 1.0, 0.0, "text", "Letter spacing"),  # tracking, in letter heights
    "text_line":     (0.6, 2.5, 1.0, "text", "Line spacing"),
    "text_tilt":     (-1.0, 1.0, 0.0, "text", "Tilt"),            # rotation of the whole block, -180 .. 180 degrees
    "text_color":    (0.0, 3.0, 0.0, "text", "Colour mode"),      # rounded: 0 show palette, 1 fixed colour (text_hue / text_sat),
                                                                  #          2 rainbow per letter, 3 follows the light chase
    "text_hue":      (0.0, 1.0, 0.52, "text", "Hue"),
    "text_sat":      (0.0, 1.0, 1.0, "text", "Colour depth"),     # 0 = white
    "text_fill":     (0.0, 1.0, 1.0, "text", "Fill"),             # 0 = hollow outline letters
    "text_outline":  (0.0, 1.0, 0.0, "text", "Outline"),          # bright outline width
    "text_glow":     (0.0, 1.0, 0.35, "text", "Glow"),
    "text_move":     (0.0, 6.0, 0.0, "text", "Movement"),         # rounded: 0 static, 1 drift, 2 orbit (ring of letters), 3 marquee,
                                                                  #          4 wave per letter, 5 spiral into the tunnel, 6 bounce on beat
    "text_speed":    (-1.0, 1.0, 0.4, "text", "Movement speed"),  # negative = the other way
    "text_wave":     (0.0, 1.0, 0.5, "text", "Movement amount"),  # wave height / drift reach / orbit radius / bounce height
    "text_pulse":    (0.0, 1.0, 0.3, "text", "Audio pulse"),      # size + brightness kick with the bass / beat
    "text_mode":     (0.0, 7.0, 0.0, "text", "Integration"),      # rounded: 0 overlay, 1 woven / engraved, 2 stencil mask, 3 knockout,
                                                                  #          4 displaces the visuals, 5 rides the scene's warps,
                                                                  #          6 kaleidoscope, 7 echo trails
    "text_weave":    (0.0, 1.0, 0.5, "text", "Weave"),            # how strongly the mode's integration acts (see HANDOFF.md)
    "text_dissolve": (0.0, 1.0, 0.0, "text", "Dissolve"),         # letters break up into the pattern

    # ---- colours of the beings and of the serpents, APPENDED 2026-11-25. Mode 0 with saturation 1 and brightness 1
    #      (the defaults) is exactly the look before these existed. Mode names: OPTIONS below.
    "being_color":   (0.0, 5.0, 0.0, "beings", "Colour mode"),    # rounded: 0 show palette, 1 single hue, 2 two-tone
                                                                  #  (head -> tail), 3 rainbow along the body,
                                                                  #  4 each creature its own hue, 5 follows the light chase
    "being_hue":     (0.0, 1.0, 0.52, "beings", "Hue"),
    "being_hue2":    (0.0, 1.0, 0.85, "beings", "Hue 2 (tail)"),  # two-tone only
    "being_sat":     (0.0, 1.0, 1.0, "beings", "Colour depth"),   # 0 = white light
    "being_bright":  (0.0, 2.0, 1.0, "beings", "Brightness"),
    "serpent_color": (0.0, 5.0, 0.0, "motion", "Serpent colours"),  # rounded: same six modes as being_color
    "serpent_hue":   (0.0, 1.0, 0.52, "motion", "Serpent hue"),
    "serpent_hue2":  (0.0, 1.0, 0.85, "motion", "Serpent hue 2 (tail)"),
    "serpent_sat":   (0.0, 1.0, 1.0, "motion", "Serpent colour depth"),
    "serpent_bright": (0.0, 2.0, 1.0, "motion", "Serpent brightness"),

    # ---- math stack (stack.StackPass), APPENDED 2026-11-25: master amount of the hand-built procedural layers.
    #      The layers themselves are structured state (control command "stack", see HANDOFF.md), not params.
    #      Default 1 with an EMPTY stack: nothing is drawn (the pass is not even run), so the default look is unchanged.
    "l_stack":       (0.0, 1.0, 1.0, "layers", "Math stack"),

    # ---- post-FX filter chain (filters.FilterPass), APPENDED 2026-11-25. Display only, between the scene composite
    #      and the brightness limiter, before mapping. Six ordered slots (flt_slotN = 0 empty, else 1 + index into
    #      FILTER_TYPES); each filter type has its own amount + 2-3 params, so a MIDI knob / mod keeps its meaning
    #      whatever slot the filter sits in. A type in two slots runs once (first slot). Every amount defaults to 0:
    #      with nothing on, the pass is skipped entirely (zero cost, pixel-identical show). Names: OPTIONS below.
    "flt_slot1":       (0.0, 10.0, 0.0, "filters", "Slot 1"),
    "flt_slot2":       (0.0, 10.0, 0.0, "filters", "Slot 2"),
    "flt_slot3":       (0.0, 10.0, 0.0, "filters", "Slot 3"),
    "flt_slot4":       (0.0, 10.0, 0.0, "filters", "Slot 4"),
    "flt_slot5":       (0.0, 10.0, 0.0, "filters", "Slot 5"),
    "flt_slot6":       (0.0, 10.0, 0.0, "filters", "Slot 6"),
    "flt_dream":       (0.0, 1.0, 0.0, "filters", "Deep dream"),
    "flt_dream_scale": (0.0, 1.0, 0.5, "filters", "Dream motif size"),
    "flt_dream_swirl": (0.0, 1.0, 0.5, "filters", "Dream swirl"),       # feedback flow + persistence (recursion)
    "flt_dream_motif": (0.0, 1.0, 0.3, "filters", "Dream motifs"),      # 0 eyes .. 0.5 fur / dog-slugs .. 1 scales / arches
    "flt_oil":         (0.0, 1.0, 0.0, "filters", "Oil paint"),
    "flt_oil_size":    (0.0, 1.0, 0.5, "filters", "Brush size"),
    "flt_oil_relief":  (0.0, 1.0, 0.4, "filters", "Impasto"),
    "flt_halftone":    (0.0, 1.0, 0.0, "filters", "Halftone"),
    "flt_ht_size":     (0.0, 1.0, 0.35, "filters", "Dot size"),
    "flt_ht_angle":    (0.0, 1.0, 0.125, "filters", "Screen angle"),
    "flt_ht_style":    (0.0, 2.0, 0.0, "filters", "Screen"),              # rounded: OPTIONS["flt_ht_style"]
    "flt_neon":        (0.0, 1.0, 0.0, "filters", "Neon edges"),
    "flt_neon_glow":   (0.0, 1.0, 0.5, "filters", "Neon glow"),
    "flt_neon_keep":   (0.0, 1.0, 0.25, "filters", "Neon keeps picture"),
    "flt_neon_color":  (0.0, 1.0, 0.0, "filters", "Neon colour"),        # 0 the picture's own hue .. 1 show palette
    "flt_chroma":      (0.0, 1.0, 0.0, "filters", "Prism split"),
    "flt_chroma_mode": (0.0, 2.0, 0.0, "filters", "Prism mode"),         # rounded: OPTIONS["flt_chroma_mode"]
    "flt_chroma_angle": (0.0, 1.0, 0.0, "filters", "Prism angle"),
    "flt_posterize":   (0.0, 1.0, 0.0, "filters", "Posterize"),
    "flt_post_levels": (2.0, 12.0, 4.0, "filters", "Levels"),
    "flt_post_palette": (0.0, 1.0, 0.0, "filters", "To show palette"),
    "flt_sort":        (0.0, 1.0, 0.0, "filters", "Pixel sort"),
    "flt_sort_thresh": (0.0, 1.0, 0.35, "filters", "Sort threshold"),
    "flt_sort_len":    (0.0, 1.0, 0.5, "filters", "Sort length"),
    "flt_sort_angle":  (0.0, 1.0, 0.25, "filters", "Sort angle"),        # 0..1 -> 0..180 degrees (0.25 = down)
    "flt_glitch":      (0.0, 1.0, 0.0, "filters", "Glitch"),
    "flt_glitch_rate": (0.0, 1.0, 0.3, "filters", "Glitch rate"),        # 0.2 .. 2.5 events / s (photosensitive cap)
    "flt_glitch_block": (0.0, 1.0, 0.4, "filters", "Glitch blocks"),
    "flt_bloom":       (0.0, 1.0, 0.0, "filters", "Bloom"),
    "flt_bloom_thresh": (0.0, 1.0, 0.55, "filters", "Bloom threshold"),
    "flt_bloom_radius": (0.0, 1.0, 0.5, "filters", "Bloom radius"),
    "flt_kaleido":     (0.0, 1.0, 0.0, "filters", "Mirror / kaleido"),
    "flt_kal_mode":    (0.0, 2.0, 2.0, "filters", "Mirror mode"),        # rounded: OPTIONS["flt_kal_mode"]
    "flt_kal_folds":   (2.0, 16.0, 6.0, "filters", "Kaleido folds"),     # rounded in the shader
    "flt_kal_spin":    (-1.0, 1.0, 0.1, "filters", "Kaleido spin"),

    # ---- crowd camera (camera_layer.CameraLayer + camera_worker.py), APPENDED 2026-11-25: webcam silhouettes, motion
    #      and dancer pose woven into the visuals. The CAMERA itself is switched on in the panel (command "camera");
    #      while it is off these do nothing and cost nothing. Mode names: OPTIONS["crowd_mode"] below.
    "l_camera":      (0.0, 1.0, 0.0, "layers", "Crowd camera"),
    "crowd_mode":    (0.0, 5.0, 1.0, "camera", "Integration"),     # rounded: 0 bodies reveal the visuals, 1 glowing outline,
                                                                   #  2 patterns flow inside bodies, 3 knockout (bodies black),
                                                                   #  4 motion trails (into the echo), 5 crowd pushes the pattern
    "crowd_glow":    (0.0, 1.0, 0.6, "camera", "Silhouette glow"),
    "crowd_hue":     (0.0, 1.0, 0.0, "camera", "Silhouette hue"),  # hue shift of the light inside / around bodies
    "crowd_soft":    (0.0, 1.0, 0.4, "camera", "Smoothing"),       # temporal smoothing of the silhouettes (slower = calmer)
    "crowd_push":    (0.0, 1.0, 0.0, "camera", "Flow push"),       # crowd movement pushes the pattern (works in every mode)
    "crowd_swarm":   (0.0, 1.0, 0.0, "camera", "Hands pull swarm"),
    "crowd_beings":  (-1.0, 1.0, 0.0, "camera", "Beings follow hands"),   # + follow, - flee
    "crowd_ripples": (0.0, 1.0, 0.0, "camera", "Hand ripples"),    # fast hand moves spawn ripples (max 2 per second)
    "crowd_serpent": (0.0, 1.0, 0.0, "camera", "Serpent on arm"),  # serpent 1 rides the dancer's arm, head at the hand
}

# Option names for selectable (rounded) params that live here; index = rounded value - min. The panel shows them as a
# dropdown / tiles (control.py merges this into its own OPTIONS table when it builds the panel meta).
OPTIONS = {
    "being": ["Anaconda spirit", "Feathered dragon", "Jaguar spirit", "Hummingbird", "Butterflies & moths", "Jellyfish",
              "Manta rays", "Koi", "Firebird", "All-seeing eyes", "Light beings", "Jester sprites", "Octopus",
              "Spirit vine", "Fireflies", "Whale", "Kittens"],
    "being_color": ["Show palette", "Single hue", "Two-tone", "Rainbow", "Each its own hue", "Light chase"],
    "serpent_color": ["Show palette", "Single hue", "Two-tone", "Rainbow", "Each its own hue", "Light chase"],
    "text_color": ["Show palette", "Fixed colour", "Rainbow letters", "Light chase"],
    "text_move": ["Static", "Drift", "Orbit ring", "Marquee", "Letter wave", "Tunnel spiral", "Beat bounce"],
    "text_mode": ["Overlay", "Woven", "Stencil mask", "Knockout", "Displace", "Rides the scene", "Kaleidoscope",
                  "Echo trails"],
    "crowd_mode": ["Bodies reveal", "Glow outline", "Patterns inside", "Knockout", "Motion trails", "Flow push"],
}

TRIGGERS = ["flash", "strobe", "invert", "freeze", "blackout", "punch"]   # momentary buttons in the panel / MIDI
EVENTS = ["ripple"]                                                        # ("ripple", x, y)
MOD_SOURCES = ["none", "bass", "mid", "high", "lfo_sine", "lfo_tri", "lfo_saw", "lfo_square", "random",
               "seq", "seq_gate",     # 16-step sequencer (control.Controller.seq): step value 0..1 / gate envelope
               # crowd camera (camera_layer.Signals, APPENDED): 0..1 except hip_sway -1..1; all 0 while the camera is off
               "crowd_motion", "crowd_left", "crowd_centre", "crowd_right", "crowd_presence",
               "hand_height", "hands_spread", "hip_sway", "hand_speed"]


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


# ---------------------------------------------------------------------------------------------------------------
# BYPASS (APPENDED 2026-11-25): "click a layer / tab entirely off and on again to the same settings".
# control.Controller keeps a set of bypassed keys and, AFTER faders + mods + scenes, pulls the OUTPUT value of every
# covered param to its neutral value (crossfaded). Stored values, mods, MIDI bindings and scenes are never edited.
# EVERY param must be in exactly one of the two tables below (tests/test_bypass.py fails on a new, undecided param).
NEUTRAL = {     # the value at which the param does nothing
    # camera / space (SHAPE tab)
    "zoom": 0.0, "pan_x": 0.0, "pan_y": 0.0, "rotate": 0.0, "flow": 0.0,
    # motion
    "travel": 0.0, "tunnel_curve": 0.0, "serpent": 0.0, "spin": 0.0, "spin_twist": 0.0,
    # contortion
    "swim": 0.0, "ripple": 0.0, "twirl": 0.0, "bulge": 0.0, "wave": 0.0,
    # colour
    "saturation": 1.0, "chase": 0.0,
    # layers: opacity 0
    "l_kene": 0.0, "l_beads": 0.0, "l_lines": 0.0, "l_ai": 0.0, "l_echo": 0.0, "l_video": 0.0, "l_fractal": 0.0,
    "l_art": 0.0, "l_land": 0.0, "l_swarm": 0.0, "l_beings": 0.0, "l_text": 0.0, "l_stack": 0.0,
    # art fly-through + distortion
    "cam_yaw": 0.0, "cam_pitch": 0.0, "cam_roll": 0.0, "cam_dolly": 0.0, "cam_drift": 0.0, "art_melt": 0.0,
    "art_liquify": 0.0, "art_slices": 0.0, "art_breathe": 0.0, "art_fold": 0.0, "art_droste": 0.0, "art_smear": 0.0,
    "art_glow": 0.0,
    # 3D decks
    "land_warp": 0.0, "land_twist": 0.0, "land_disp": 0.0,
}
NEUTRAL = {k: v for k, v in NEUTRAL.items() if k in PARAMS}
BYPASS_KEEP = {  # bypass leaves these alone: "off" has no meaning (selectors, counts, sizes, colours, quality, triggers)
    "master", "flash", "strobe", "invert", "freeze", "blackout", "punch",
    "symmetry", "folds", "speed", "flow_dir",
    "travel_mode", "depth_fog", "serpent_mode", "serpent_count", "serpent_segments", "serpent_speed", "serpent_wiggle",
    "serpent_size",
    "ripple_decay", "density",
    "palette", "hue", "chase_width", "chase_rate", "chase_mode", "trails",
    "luma_key",
    "fractal_mode", "fractal_morph", "fractal_detail", "fractal_glow", "fractal_metal", "fractal_res",
    "art_index", "art_relief", "art_use", "art_source",
    "land_a", "land_b", "land_xfade", "land_blend", "land_curve", "land_morph", "land_fov", "land_cam", "land_style",
    "land_mirror", "land_res", "swarm_vortex", "swarm_turb", "swarm_trail", "swarm_tint",
    "being", "being_count", "being_size", "being_speed", "being_wiggle", "being_glow", "being_detail", "being_mix",
    "being_bright", "being_color", "being_hue", "being_hue2", "being_sat",
    "serpent_bright", "serpent_color", "serpent_hue", "serpent_hue2", "serpent_sat",
    "text_size", "text_x", "text_y", "text_track", "text_line", "text_tilt", "text_color", "text_hue", "text_sat",
    "text_fill", "text_outline", "text_glow", "text_move", "text_speed", "text_wave", "text_pulse", "text_mode",
    "text_weave", "text_dissolve",
}
BYPASS_KEEP = {k for k in BYPASS_KEEP if k in PARAMS}
# post-FX filter chain (filters.py): slot value i (1..10) = FILTER_TYPES[i-1]; the type's amount param is flt_<type>.
# Bypass: the amounts go to 0 (the FILTERS tab's power button, and ALL FX), everything else is kept.
FILTER_TYPES = ["dream", "oil", "halftone", "neon", "chroma", "posterize", "sort", "glitch", "bloom", "kaleido"]
FILTER_NAMES = ["Deep dream", "Oil paint", "Halftone", "Neon edges", "Prism split", "Posterize", "Pixel sort", "Glitch",
                "Bloom", "Mirror / kaleido"]
OPTIONS.update({f"flt_slot{i}": ["Empty"] + FILTER_NAMES for i in range(1, 7)})
OPTIONS.update({"flt_ht_style": ["Colour dots", "RGB screen", "Line engraving"],
                "flt_chroma_mode": ["Radial", "Linear", "Prism ghosts"],
                "flt_kal_mode": ["Mirror", "Quad mirror", "Kaleidoscope"]})
NEUTRAL.update({f"flt_{t}": 0.0 for t in FILTER_TYPES if f"flt_{t}" in PARAMS})
BYPASS_KEEP |= {n for n, v in PARAMS.items() if v[3] == "filters" and n not in NEUTRAL}
BYPASS_GROUP_KEY = {"space": "shape"}            # the SHAPE tab holds two groups: one switch for both
BYPASS_LINKS = {                                 # switching a tab off also switches its layer(s) off
    "fractal": ["l_fractal"], "art": ["l_art"], "land": ["l_land", "l_swarm"], "beings": ["l_beings"], "text": ["l_text"],
}
# crowd camera (camera_layer.py): amounts go to 0, the look settings are kept; the CAMERA tab's switch also drops
# l_camera. (Bypass never switches the webcam itself off: that is the CAMERA ON / OFF switch.)
NEUTRAL.update({n: 0.0 for n in ("l_camera", "crowd_push", "crowd_swarm", "crowd_beings", "crowd_ripples",
                                 "crowd_serpent") if n in PARAMS})
BYPASS_KEEP |= {n for n in ("crowd_mode", "crowd_glow", "crowd_hue", "crowd_soft") if n in PARAMS}
BYPASS_LINKS["camera"] = ["l_camera"]


def bypass_cover():
    """{bypass key: (params it pulls to neutral)}. Keys: one per tab / group (never master), one per bypassable
    param (every layer, serpent, chase, ...), and "fx" = every effect but no layer (ALL FX OFF)."""
    cover = {}
    for name, v in PARAMS.items():
        group = v[3]
        if group == "master":
            continue
        key = BYPASS_GROUP_KEY.get(group, group)
        cover.setdefault(key, [])
        if name in NEUTRAL:
            cover[key].append(name)
    for group, layers in BYPASS_LINKS.items():
        if group in cover:
            cover[group] += [n for n in layers if n in NEUTRAL and n not in cover[group]]
    cover = {k: tuple(v) for k, v in cover.items() if v}
    cover["fx"] = tuple(n for n in NEUTRAL if not n.startswith("l_"))
    for name in NEUTRAL:
        cover.setdefault(name, (name,))
    return cover
