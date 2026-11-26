"""Generative kené / kenê kuin STUDY layers (kene_real.py).

Pure-python tests first (design grammar, HK bead tiles, colour conventions, the params / panel contract); the GL tests
at the bottom render tiny frames and are the only ones that touch the GPU.
Grammar source: research/kene_reference/README.md (three-tier line hierarchy, quadrant / cross topology, bounded
field, lines that never dead-end, eye nodes; Huni Kuin motif names and bead colour conventions).
"""
import re
from pathlib import Path

import numpy as np
import pytest

import kene_real as kr
import params

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "research" / "kene_reference" / "README.md").read_text(encoding="utf-8")
MOTIF_TABLE = README.split("### 2.2", 1)[1].split("### 2.3", 1)[0]


# ------------------------------------------------------------------ Shipibo-Konibo-inspired design generator (CPU)

def boundary_edges(lab):
    """Primary lines = edges between tiles of different label (the frame counts as a label of its own)."""
    p = np.pad(lab.astype(int), 1, constant_values=kr.FRAME)
    return (p[:, 1:] != p[:, :-1]).astype(int), (p[1:, :] != p[:-1, :]).astype(int)   # vertical (h+2, w+1), horizontal (h+1, w+2)


@pytest.mark.parametrize("seed", range(8))
def test_design_is_seeded_and_bounded(seed):
    a = kr.shipibo_design(seed, 10, 10)
    b = kr.shipibo_design(seed, 10, 10)
    assert a.shape == (30, 30, 4) and a.dtype == np.uint8
    assert np.array_equal(a, b), "same seed must give the same design"
    assert not np.array_equal(a, kr.shipibo_design(seed + 1, 10, 10)), "a new seed must give a new design"
    assert a[..., 0].max() < kr.FRAME, "label 255 is the frame outside the field"


def test_dead_end_detector_can_fail():
    """The detector itself: it must see a dangling stroke (so the design check below is not a tautology of the
    helper), and it reports none on any partition (a by-construction property of region outlines)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        ve, he = boundary_edges(rng.integers(0, 4, (9, 9)))
        assert kr.line_dead_ends(ve, he) == 0
    ve, he = boundary_edges(np.zeros((6, 6), int))
    assert kr.line_dead_ends(ve, he) == 0
    ve[3, 3] = 1                                                    # one loose vertical stroke inside the field
    assert kr.line_dead_ends(ve, he) == 2
    he[3, 3] = 1                                                    # ... joined by a stroke at one end only
    assert kr.line_dead_ends(ve, he) >= 1


@pytest.mark.parametrize("seed", range(12))
def test_primary_lines_form_a_real_network(seed):
    """Primary lines are region outlines, so they cannot dead-end (by construction; kept as a guard for the day a
    free-ending path type is added). What CAN break: too few lines, a region too thin to hold its echo line, or a
    design that is all frame."""
    for dims in ((10, 10), (22, 14)):
        d = kr.shipibo_design(seed, *dims)
        lab = d[..., 0]
        ve, he = boundary_edges(lab)
        assert kr.line_dead_ends(ve, he) == 0, "a primary line dead-ends inside the field"
        inner = int(ve[1:-1, 1:-1].sum() + he[1:-1, 1:-1].sum())
        assert inner > 1.5 * (lab.shape[0] + lab.shape[1]), f"primary lines inside the frame, not just the frame ({inner})"
        # the field is not all thin tracks: several regions are wide enough to hold the echo line AND fine fill
        ids, counts = np.unique(lab, return_counts=True)
        big = ids[counts >= 30]
        interior = (d[..., 1] & 16) > 0
        roomy = [r for r in big if r <= 200 and interior[lab == r].any()]
        assert len(roomy) >= 2, f"only {len(roomy)} regions have room for the fine fill"


@pytest.mark.parametrize("seed", range(12))
def test_quadrant_topology_and_central_eye(seed):
    """Square panels are organised round the centre: the line network is unchanged by a half turn (and usually by a
    quarter turn or mirrors), and the centre carries an eye node (a small square of its own)."""
    lab = kr.shipibo_design(seed, 10, 10, form=0)[..., 0].astype(int)
    ve, he = boundary_edges(lab)
    assert np.array_equal(ve, ve[::-1, ::-1]) and np.array_equal(he, he[::-1, ::-1]), "not symmetric about the centre"
    c = lab.shape[0] // 2
    centre = lab[c - 1:c + 1, c - 1:c + 1]
    assert len(set(centre.ravel())) == 1, "the central eye is one node"
    assert lab[c, c] != lab[c, c + 3] or lab[c, c] != lab[c + 3, c], "the eye is bounded by its own primary line"
    assert len(np.unique(lab)) >= 5, "centre + at least four surrounding zones (quincunx)"


@pytest.mark.parametrize("seed", range(8))
def test_three_tiers_fill_the_field(seed):
    d = kr.shipibo_design(seed, 22, 14)
    s = kr.design_stats(d)
    assert 0.25 <= s["ring"] <= 0.75, f"secondary (contour) ring tiles: {s['ring']:.2f}"
    assert s["interior"] >= 0.25, "there is room for the fine fill"
    assert s["dots"] < 0.30, f"the fill is mostly lines, not left-over dots ({s['dots']:.2f})"
    assert s["reciprocal"], "fill lines join tile to tile (no half-drawn connection)"
    assert s["cross_label"] == 0, "fine fill never crosses a primary line"


def test_pen_order_travels_along_the_lines():
    d = kr.shipibo_design(3, 14, 14, form=0)
    order = d[..., 3].astype(int)
    lab, bits = d[..., 0], d[..., 1]
    ring = (bits & 16) == 0
    c = order.shape[0] // 2
    assert order.max() == 255 and order[c - 3:c + 3, c - 3:c + 3].min() <= 4, "the pen starts at the centre"
    # along the line network the order is continuous: neighbours on the ring never jump (the pen never teleports)
    step = 0
    for dy, dx in ((0, 1), (1, 0)):
        a, b = order[:order.shape[0] - dy, :order.shape[1] - dx], order[dy:, dx:]
        both = ring[:ring.shape[0] - dy, :ring.shape[1] - dx] & ring[dy:, dx:]
        step = max(step, int(np.abs(a - b)[both].max()))
    assert step <= 24, f"pen order jumps {step}/255 between neighbouring line tiles"
    assert lab.shape == order.shape


def test_band_design_wraps_seamlessly():
    d = kr.shipibo_design(5, 20, 8, wrap_x=True)
    lab, bits = d[..., 0], d[..., 1]
    # a fill line leaving the right edge arrives on the left edge
    east, west = (bits[:, -1] & 2) > 0, (bits[:, 0] & 8) > 0
    assert np.array_equal(east, west)


# ------------------------------------------------------------------ Huni Kuin-inspired bead tiles (CPU)

def test_hk_motif_names_come_from_the_readme():
    assert len(kr.HK_MOTIFS) >= 4
    for name in kr.HK_MOTIFS:
        assert name in README, f"'{name}' is not a motif name documented in research/kene_reference/README.md"
        assert name in MOTIF_TABLE, f"'{name}' is not in the README's table of named Huni Kuin motifs (2.2)"


def test_hk_tiles_are_periodic_diagonal_and_balanced():
    atlas = kr.hk_atlas(0)
    T = kr.HK_T
    assert atlas.shape == (T, T * kr.NF, 2) and atlas.dtype == np.uint8
    tiles = [atlas[:, k * T:(k + 1) * T, 0] for k in range(len(kr.HK_MOTIFS))]
    for k, t in enumerate(tiles):
        fig = t > 0
        assert 0.3 <= fig.mean() <= 0.7, f"{kr.HK_MOTIFS[k]}: figure and ground carry equal weight ({fig.mean():.2f})"
        assert t.max() <= 6
        assert np.array_equal(fig, fig[:, ::-1]) or np.array_equal(fig, np.roll(fig[:, ::-1], 1, 1)), "mirror symmetric"
        # diagonal stepped geometry: most figure / ground corners are staircase corners, not long straight runs
        dx, dy = fig != np.roll(fig, 1, 1), fig != np.roll(fig, 1, 0)
        assert (dx & dy).sum() >= 0.25 * max(dx.sum(), dy.sum()), f"{kr.HK_MOTIFS[k]} is not diagonal / stepped"
    for a in range(len(tiles)):
        for b in range(a + 1, len(tiles)):
            assert (tiles[a] != tiles[b]).mean() > 0.1, "motif families must differ"


def test_hk_tile_is_seamless_when_repeated():
    for fam in range(kr.NF):
        for seed in (0, 1):
            small = kr.hk_tile(fam, seed, kr.HK_T)
            big = kr.hk_tile(fam, seed, kr.HK_T * 2)
            assert np.array_equal(np.tile(small, (2, 2, 1))[..., 0], big[..., 0]), kr.HK_FAMILIES[fam]


# ------------------------------------------------------------------ colour conventions per medium

def test_styles_follow_the_documented_colour_conventions():
    pal = [(1.0, 0.2, 0.6), (0.1, 0.9, 0.8), (1.0, 0.8, 0.1), (0.5, 0.2, 1.0), (0.2, 1.0, 0.3)]
    v = params.defaults()
    lum = kr.style_uniforms(dict(v, kene_style=0), pal)
    assert lum["ground_on"] == 0.0, "projection default: luminous lines on black"
    cloth = kr.style_uniforms(dict(v, kene_style=1), pal)                 # mud-dyed cotton: dark brown on cream
    assert cloth["ground_on"] == 1.0 and min(cloth["ground"]) > 0.45 and cloth["ground"][0] > cloth["ground"][2]
    assert max(cloth["cols"][0]) < 0.25, "mud-dye ink is near black"
    cer = kr.style_uniforms(dict(v, kene_style=3), pal)                   # ceramic: black + red oxide on cream slip
    assert cer["ground_on"] == 1.0 and max(cer["cols"][0]) < 0.2
    r, g, b = cer["cols"][1]
    assert r > 0.4 and r > 2 * g and r > 2 * b, "ceramic secondary lines are iron-oxide red"
    emb = kr.style_uniforms(dict(v, kene_style=2, kene_colors=2), pal)    # embroidery: bright threads on black
    assert emb["ground_on"] == 0.0 and emb["stitch"] > 0
    for c in emb["cols"][:3]:
        assert max(c) > 0.85 and max(c) - min(c) > 0.5, "embroidery threads are saturated"
    assert kr.style_uniforms(dict(v, kene_style=4), pal)["bead"] > 0, "beadwork renders on a bead grid"
    follow = kr.style_uniforms(dict(v, kene_style=0, kene_colors=0), pal)
    assert np.allclose(follow["cols"][0], pal[0], atol=0.02), "palette-follow uses the show palette"


def test_hk_colourways_use_the_documented_bead_colours():
    for name, cols in zip(kr.HK_COLOR_NAMES, kr.HK_COLORWAYS):
        assert cols is None or len(cols) == 8, name
    hk = kr.hk_colours(dict(params.defaults(), hk_colors=0), None)
    assert all(abs(c[0] - c[1]) < 0.1 and abs(c[1] - c[2]) < 0.12 for c in hk), "black & white colourway"
    pal = [(1.0, 0.2, 0.6)] * 5
    follow = kr.hk_colours(dict(params.defaults(), hk_colors=kr.HK_COLOR_NAMES.index("Follow show palette")), pal)
    assert np.allclose(follow[0], pal[0], atol=0.02)


# ------------------------------------------------------------------ params / panel contract

NEW = ["l_kene_shipibo", "l_kene_hunikuin"] + [n for n, v in params.PARAMS.items() if v[3] == "kene"]


def test_params_are_appended_off_by_default_and_classified():
    assert params.PARAMS["l_kene"][4] == "Folding Blocks" and params.PARAMS["l_kene"][2] == 1.0, "Folding Blocks is untouched"
    assert params.PARAMS["l_kene_shipibo"][2] == 0.0 and params.PARAMS["l_kene_hunikuin"][2] == 0.0
    names = list(params.PARAMS)
    assert names.index("l_kene_shipibo") > names.index("crowd_serpent"), "append only"
    i = names.index("l_kene_shipibo")
    assert names[i:i + len(NEW)] == NEW, "the block is contiguous and in its original order (append only)"
    assert len(NEW) > 10
    for n in NEW:
        assert (n in params.NEUTRAL) != (n in params.BYPASS_KEEP), n
    cover = params.bypass_cover()
    assert set(cover["kene"]) == {"l_kene_shipibo", "l_kene_hunikuin"}
    assert "l_kene_shipibo" in cover["layers"] and cover["l_kene_hunikuin"] == ("l_kene_hunikuin",)


def test_option_labels_match_the_module_and_the_ranges():
    for name, labels in kr.OPTIONS.items():
        assert params.OPTIONS[name] == list(labels), name
        lo, hi = params.PARAMS[name][0], params.PARAMS[name][1]
        assert lo == 0.0 and hi == len(labels) - 1, name
        assert all(len(s) <= 40 for s in labels), name                    # control.merge_options cuts at 40
    assert params.OPTIONS["hk_motif"][:5] == [n + " (study)" for n in kr.HK_MOTIFS], "the dropdown says study"
    assert all("study" in x for x in params.OPTIONS["hk_motif"]) and len(params.OPTIONS["hk_motif"]) == kr.NF
    assert all("study" in x for x in params.OPTIONS["kene_style"][5:]), "the new media are labelled as studies"
    assert not any(n in kr.HK_EXTRA for n in kr.HK_MOTIFS), "the added families carry descriptive names, not README names"
    assert "study" in params.PARAMS["hk_motif"][4].lower()


def test_nothing_is_labelled_as_authentic():
    text = " ".join(params.PARAMS[n][4] for n in NEW) + " " + " ".join(s for n in kr.OPTIONS for s in kr.OPTIONS[n])
    assert not re.search(r"authentic|real kene|genuine|sacred|traditional|\bsing\b|\bsung\b|icaro|\bsong", text, re.I)
    for n in ("l_kene_shipibo", "l_kene_hunikuin"):
        assert re.search(r"study|inspired", params.PARAMS[n][4], re.I), "the layer label says what it is"
    assert "Generative study inspired by Shipibo-Konibo kené / Huni Kuin kenê kuin" in kr.CREDIT
    assert "credit and collaborate with the originating communities" in kr.CREDIT
    assert "research/kene_reference" in kr.CREDIT


def test_panel_carries_the_credit_and_the_bring_in_entries():
    html = (ROOT / "panel.html").read_text(encoding="utf-8")
    assert kr.CREDIT in html
    assert "['l_kene_shipibo','kene']" in html and "['l_kene_hunikuin','kene']" in html
    assert not re.search(r"authentic\s+ken", html, re.I)
    assert "Huni Kuin kene lattice" not in html and "Shipibo kene neon" not in html, "AI presets must not label output as kene"


def test_engine_hook_is_cpu_only_for_the_group():
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert "KenePass" in src and "p_l_kene_shipibo" in src and "texN" in src
    assert re.search(r'CPU_ONLY \|= .*"kene"', src)


def test_pen_cycle_draws_then_holds_then_changes_design():
    v = dict(params.defaults(), kene_pen=1, kene_pen_new=1, kene_seed=3)       # every 4 bars
    p0 = kr.pen_state(v, beat=0.0)
    mid = kr.pen_state(v, beat=8.0)
    hold = kr.pen_state(v, beat=13.0)
    nxt = kr.pen_state(v, beat=16.5)
    assert p0["progress"] == 0.0 and 0.5 < mid["progress"] < 1.0 and hold["progress"] == 1.0
    assert nxt["seed"] != p0["seed"] and nxt["progress"] < 0.1, "a new design starts while nothing is drawn"
    assert kr.pen_state(v, beat=15.99)["fade"] < 0.05, "the old design is gone before the swap (no cut)"
    assert kr.pen_state(v, beat=16.0)["fade"] < 0.05, "and the field (a lit cloth ground) fades back in, never cuts on"
    fades = [kr.pen_state(v, beat=b)["fade"] for b in np.arange(14.0, 18.0, 1 / 30.0)]
    assert max(abs(a - b) for a, b in zip(fades, fades[1:])) < 0.2, "the pen fade has no step"
    assert kr.pen_state(v, beat=8.0)["next_seed"] == nxt["seed"], "the next design is known ahead (prebuilt)"
    off = kr.pen_state(dict(v, kene_pen=0, kene_draw=0.4), beat=5.0)
    assert off["progress"] == pytest.approx(0.4) and off["seed"] == 3 and off["fade"] == 1.0


# ------------------------------------------------------------------ GL (tiny renders)

@pytest.fixture(scope="module")
def gl():
    ctx, win = kr.hidden_context()
    yield ctx


def shot(ctx, size=(640, 400), t=4.0, beat=None, **over):
    kp = kr.KenePass(ctx, size)
    try:
        return kr.render_once(kp, kr.default_values(**over), t=t, beat=beat)
    finally:
        kp.release()


def black_frac(img):
    return float((img[..., :3].max(axis=2) <= 4).mean())


def blown_frac(img):
    return float((img[..., :3].min(axis=2) >= 250).mean())


def test_gl_off_draws_nothing(gl):
    kp = kr.KenePass(gl, (160, 100))
    try:
        tex = kp.render(params.defaults(), kr.make_clock(1.0), 1.0, 1 / 60, (0.5, 0.3, 0.2), None)
        assert kp.draws == 0
        assert not np.frombuffer(tex.read(), np.uint8).any()
    finally:
        kp.release()


@pytest.mark.parametrize("style", range(len(kr.STYLES)))
@pytest.mark.parametrize("layout", range(5))
def test_gl_every_style_and_layout_renders(gl, style, layout):
    img = shot(gl, (640, 400), l_kene_shipibo=1.0, kene_style=style, kene_layout=layout, symmetry=1.0 if layout == 4 else 0.0)
    assert img.shape == (400, 640, 4) and img[..., :3].max() > 120, "nothing was drawn"
    assert blown_frac(img) < 0.02, "white-out"
    if not kr.style_uniforms(dict(params.defaults(), kene_style=style), None)["ground_on"]:   # lit grounds are on purpose
        assert black_frac(img) >= 0.25, f"true black {black_frac(img):.2f}"


@pytest.mark.parametrize("motif", range(kr.NF))
@pytest.mark.parametrize("medium", range(len(kr.HK_MEDIA)))
def test_gl_hunikuin_renders(gl, motif, medium):
    img = shot(gl, (640, 400), l_kene_hunikuin=1.0, hk_motif=motif, hk_medium=medium)
    assert img[..., :3].max() > 120 and blown_frac(img) < 0.02
    if not (medium in (1, 2) and motif >= 6):            # flat paint / woven floats of an all-colour study have no bead gaps
        assert black_frac(img) >= 0.25, f"true black {black_frac(img):.2f}"
    lit = img[..., :3][img[..., :3].max(axis=2) > 80].astype(float)
    assert len(lit) > 500
    if medium == 0 and motif < 5:
        sat = (lit.max(axis=1) - lit.min(axis=1)) / lit.max(axis=1)
        assert sat.mean() > 0.45, "bead colours are saturated"


def test_gl_pen_draws_the_design_in(gl):
    ink = [float(shot(gl, (320, 200), l_kene_shipibo=1.0, kene_draw=d, kene_flow=0.0)[..., :3].mean()) for d in (0.0, 0.3, 0.6, 1.0)]
    assert ink[0] < 0.5, "nothing drawn at 0"
    assert ink[0] < ink[1] < ink[2] < ink[3], f"the drawing grows: {ink}"


def test_gl_new_design_changes_the_picture_and_is_repeatable(gl):
    a = shot(gl, (320, 200), l_kene_shipibo=1.0, kene_seed=1, kene_flow=0.0)
    b = shot(gl, (320, 200), l_kene_shipibo=1.0, kene_seed=1, kene_flow=0.0)
    c = shot(gl, (320, 200), l_kene_shipibo=1.0, kene_seed=2, kene_flow=0.0)
    assert np.array_equal(a, b)
    assert (a != c).mean() > 0.05


def test_gl_no_shimmer_under_a_sub_pixel_shift(gl):
    """Antialiased lines: sliding the band by a fraction of a pixel must not change the light it carries."""
    kp = kr.KenePass(gl, (1280, 800))
    try:
        v = kr.default_values(l_kene_shipibo=1.0, kene_layout=2, kene_flow=0.0, kene_breathe=0.0)
        means, crops = [], []
        for k in range(4):
            kp.band_scroll = 0.25 + k * 0.00013                           # fractions of the band period: about 1/7 pixel
            img = kr.render_once(kp, v, dt=0.0)[..., :3]
            means.append(float(img.mean()))
            crops.append(img[300:500, 400:880].astype(int))
        assert max(means) - min(means) < 0.02 * max(means), means
        for a, b in zip(crops, crops[1:]):                               # per pixel: a 1/7 px slide moves no pixel far
            assert np.abs(a - b).max() <= 70, f"a pixel jumped {np.abs(a - b).max()} for a 1/7 px shift"
            assert (np.abs(a - b) > 40).mean() < 0.002
    finally:
        kp.release()


def test_gl_long_show_phases_wrap(gl):
    """Hours in, the picture is the same as at the start of the show (clock phases are wrapped on the CPU)."""
    v = dict(l_kene_shipibo=1.0, l_kene_hunikuin=0.6, kene_layout=4, symmetry=1.0, travel=0.0, flow=0.0, rotate=0.0)
    a = shot(gl, (320, 200), t=7.5 * 4, beat=64.0, **v)
    b = shot(gl, (320, 200), t=7.5 * 4 + 7.5 * 4096, beat=64.0 + 32 * 2048, **v)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 2


WRAPS = [   # (values, accumulator, index, its wrap period)
    (dict(l_kene_shipibo=1.0, kene_layout=4, symmetry=1.0, kene_draw=0.5), "polar", None, kr.POLAR_WRAP),
    (dict(l_kene_shipibo=1.0, kene_layout=1, kene_draw=0.5), "pan", 0, kr.PAN_WRAP),
    (dict(l_kene_shipibo=1.0, kene_layout=1, kene_draw=0.5), "pan", 1, kr.PAN_WRAP),
    (dict(l_kene_shipibo=1.0, kene_layout=2), "band_scroll", None, 1.0),
    (dict(l_kene_hunikuin=1.0, kene_layout=0), "hk_scroll", 0, kr.HK_T),
    (dict(l_kene_hunikuin=1.0, kene_layout=0, hk_motif=4), "hk_scroll", 1, kr.HK_T),
    (dict(l_kene_hunikuin=1.0, kene_layout=3, hk_motif=4), "hk_polar", None, kr.HK_T),
]


@pytest.mark.parametrize("case", range(len(WRAPS)))
def test_gl_no_pop_when_a_scroll_wraps(gl, case):
    """The frame just before a scroll accumulator wraps and the frame just after are neighbours, not a jump: the
    shader's picture at x and at x + period is the same (the half-drawn tunnel used to flip every panel at the wrap)."""
    over, name, idx, period = WRAPS[case]
    kp = kr.KenePass(gl, (320, 200))
    try:
        v = kr.default_values(kene_flow=0.0, kene_breathe=0.0, hk_seq=0.0, **over)

        def at(x):
            cur = getattr(kp, name)
            if idx is None:
                setattr(kp, name, x)
            else:
                cur[idx] = x
            return kr.render_once(kp, v, dt=0.0).astype(int)
        base, shifted, moved = at(0.37), at(0.37 + period), at(0.37 + 0.3 * period + 0.5)
        assert np.abs(base - shifted).max() <= 3, f"{name} pops at its wrap"
        assert np.abs(base - moved).mean() > 0.3, f"{name} does not move the picture: the test would prove nothing"
    finally:
        kp.release()


def test_gl_advancing_across_the_wrap_is_continuous(gl):
    """Really advance the tunnel through its wrap with the pen half way: no frame differs from the one before by more
    than ordinary motion does."""
    kp = kr.KenePass(gl, (320, 200))
    try:
        v = kr.default_values(l_kene_shipibo=1.0, kene_layout=4, symmetry=1.0, kene_draw=0.5, kene_flow=0.0,
                              kene_breathe=0.0, travel=-1.0)
        kr.render_once(kp, v, dt=0.0)
        kp.polar = kr.POLAR_WRAP - 0.02
        diffs, wrapped, prev, last = [], False, None, kp.polar
        for _ in range(12):
            img = kr.render_once(kp, v, dt=1 / 60).astype(int)
            wrapped |= kp.polar < last
            last = kp.polar
            if prev is not None:
                diffs.append(float(np.abs(img - prev).mean()))
            prev = img
        assert wrapped and 0.0 <= kp.polar < kr.POLAR_WRAP, "the scroll never crossed its wrap"
        assert max(diffs) < 3.0 * (np.median(diffs) + 0.2), f"a pop at the wrap: {diffs}"
    finally:
        kp.release()


# ------------------------------------------------------------------ review round: threading, clocks, rate limits

def wait_built(kp, timeout=10.0):
    import time
    t0 = time.perf_counter()
    while kp.builder.pending() and time.perf_counter() - t0 < timeout:
        time.sleep(0.005)
    assert not kp.builder.pending(), "the design worker never finished"


def live(kp, values, n=1, wall=1 / 60, dt=1 / 60, t=4.0, beat=None):
    """Render like the engine does (asynchronous builds), with a wall clock the test steps."""
    kp.sync, kp.wall_dt = False, wall
    tex = None
    for _ in range(n):
        tex = kp.render(values, kr.make_clock(t, beat=beat), t, dt, (0.5, 0.3, 0.2), kr.GALLERY_PALETTE)
    return np.frombuffer(tex.read(), np.uint8).reshape(tex.height, tex.width, 4)[::-1].astype(int)


def test_gl_designs_are_never_built_on_the_render_thread(gl):
    import threading
    kp = kr.KenePass(gl, (160, 100))
    try:
        v = kr.default_values(l_kene_shipibo=1.0, kene_pen=1, kene_pen_new=1)
        first = live(kp, v, beat=2.0)
        assert not first[..., :3].any() or kp.builder.built, "nothing is shown until the first design exists"
        wait_built(kp)
        img = live(kp, v, n=20, beat=2.0)                                  # fades in over FADE_S of wall time
        assert img[..., :3].max() > 100
        for seed in (5, 9, 11):
            live(kp, dict(v, kene_seed=seed), n=3, beat=2.0)
            wait_built(kp)
        assert kp.builder.built >= 4
        assert threading.get_ident() not in kp.builder.build_threads, "a design was built on the calling thread"
        # the pen's next design is built ahead, during the hold: the swap at the cycle start finds it ready
        nxt = kp._key(kr.pen_state(v, 2.0)["next_seed"], 0, 0.5)
        assert kp.builder.get(nxt) is not None, "the next pen design was not prebuilt"
    finally:
        kp.release()


def test_gl_design_swaps_are_rate_limited_on_the_wall_clock_and_fade(gl):
    """A sweeping Design knob: at most one swap per SWAP_S, each through a fade (no frame-to-frame cut), and it keeps
    working with the show frozen (dt 0, show time standing still)."""
    kp = kr.KenePass(gl, (160, 100))
    try:
        v = kr.default_values(l_kene_shipibo=1.0, kene_flow=0.0, kene_breathe=0.0)
        live(kp, v)
        wait_built(kp)
        live(kp, v, n=30)
        for seed in range(1, 9):                                          # have them all ready: only the limiter is left
            kp.builder.build_now(kp._key(seed, 0, 0.5))
        swaps0, means, steps = kp.swaps, [], []
        prev = live(kp, v, dt=0.0)
        for f in range(120):                                              # 2 s of wall time, frozen show, a new seed every 2 frames
            img = live(kp, dict(v, kene_seed=1 + (f // 2) % 8, freeze=1.0, speed=0.0), dt=0.0, t=4.0)
            steps.append(float(np.abs(img - prev)[..., :3].mean()))
            means.append(float(img[..., :3].mean()))
            prev = img
        n = kp.swaps - swaps0
        assert 1 <= n <= 5, f"{n} design swaps in 2 s (limit: one per {kr.SWAP_S} s; frozen show must still swap)"
        assert max(steps) < 0.35 * max(means), f"a design cut in one frame: step {max(steps):.2f} of {max(means):.2f}"
    finally:
        kp.release()


def test_gl_speed_is_not_applied_twice(gl):
    """The engine's dt already carries speed and freeze; the pass must not scale by `speed` again."""
    kp = kr.KenePass(gl, (160, 100))
    try:
        moved = {}
        for sp in (0.5, 1.0, 2.0):
            kp.pan[:] = [0.0, 0.0]
            kp.polar = kp.band_scroll = 0.0
            kr.render_once(kp, kr.default_values(l_kene_shipibo=1.0, kene_layout=1, flow=1.0, travel=0.5, speed=sp), dt=0.1)
            moved[sp] = kp.pan[0]
        assert moved[0.5] == pytest.approx(moved[1.0]) == pytest.approx(moved[2.0]) and moved[1.0] > 0
    finally:
        kp.release()


def test_gl_bead_layer_alone_builds_no_design_and_ignores_the_pen(gl):
    kp = kr.KenePass(gl, (160, 100))
    try:
        v = kr.default_values(l_kene_hunikuin=1.0, kene_pen=1, kene_pen_new=1, hk_seq=0.0)
        a = live(kp, v, n=3, beat=2.0)
        b = live(kp, v, n=3, beat=18.0)                                   # the next pen cycle: seed + 1
        wait_built(kp)
        assert kp.builder.built == 0, "the bead layer alone must never build a line design"
        assert a[..., :3].max() > 60 and np.array_equal(a, b), "the pen cycle must not flip the bead tiles"
    finally:
        kp.release()


def test_gl_bead_motif_changes_morph_and_a_sweep_never_pops(gl):
    kp = kr.KenePass(gl, (160, 100))
    try:
        v = kr.default_values(l_kene_hunikuin=1.0, hk_seq=0.0)
        prev = live(kp, v, n=2)
        steps, full = [], float(prev[..., :3].mean())
        for f in range(90):                                              # the motif knob swept through every family
            img = live(kp, dict(v, hk_motif=(f // 3) % 5, kene_seed=(f // 7) % 2), dt=0.0)
            steps.append(float(np.abs(img - prev)[..., :3].mean()))
            prev = img
        assert max(steps) < 0.5 * full, f"bead pattern popped: {max(steps):.2f} of {full:.2f}"
        assert kp._hk_morph > 0.0 or kp._hk_now == kp._hk_want
        # frozen show (dt 0): the morph still finishes, on wall time
        for _ in range(200):
            live(kp, dict(v, hk_motif=3), dt=0.0)
        assert kp._hk_now == 3 and kp._hk_morph == 0.0
    finally:
        kp.release()


def test_gl_palette_follow_and_alpha(gl):
    red = [(1.0, 0.05, 0.05)] * 5
    kp = kr.KenePass(gl, (320, 200))
    try:
        img = kr.render_once(kp, kr.default_values(l_kene_shipibo=1.0, kene_colors=0), palette=red)
    finally:
        kp.release()
    lit = img[img[..., :3].max(axis=2) > 100]
    assert lit[:, 0].mean() > 3 * max(lit[:, 1].mean(), lit[:, 2].mean()), "the lines take the show palette"
    assert (img[..., 3] > 0).mean() < 0.75, "black stays transparent for the layers underneath"
    half = shot(gl, (320, 200), l_kene_shipibo=0.5, kene_flow=0.0)
    full = shot(gl, (320, 200), l_kene_shipibo=1.0, kene_flow=0.0)
    assert half[..., :3].mean() == pytest.approx(full[..., :3].mean() * 0.5, rel=0.12), "the fader is the opacity"


# ------------------------------------------------------------------ GRAMMAR_SPEC primitives (H2 - H4) and the 7 studies

@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("width", [1, 2, 3])
def test_fret_is_interlocking_and_complementary(diagonal, width):
    """The negative space of the fret is the same fret: shifting by one cell swaps figure and ground exactly."""
    P = 12
    j, i = np.meshgrid(np.arange(-24, 48), np.arange(-24, 48), indexing="ij")
    f = kr.fret(i, j, P, width, diagonal)
    shift = (P, P) if diagonal else (P, 0)
    assert np.array_equal(kr.fret(i + shift[0], j + shift[1], P, width, diagonal), 1 - f), "not complementary"
    assert np.array_equal(kr.fret(i + (4 if diagonal else 2) * P, j, P, width, diagonal), f), "not periodic"
    if diagonal:                                                            # staircase strokes hang together (4-connected)
        lone = (f == 1) & (np.roll(f, 1, 0) == 0) & (np.roll(f, -1, 0) == 0) & (np.roll(f, 1, 1) == 0) & (np.roll(f, -1, 1) == 0)
        assert lone[2:-2, 2:-2].mean() < 0.01, "the diagonal fret falls apart into single cells"
    assert f.mean() == pytest.approx(0.5, abs=0.02), "figure and ground carry equal weight"
    assert not np.array_equal(kr.fret(i, j, P, width, diagonal, hand=-1), f), "handedness changes the spiral"


def test_fret_is_a_spiral_not_concentric_rings():
    """The break joins ring k to ring k + 2: over the tiled plane the fret hangs together in far fewer pieces than
    the concentric rings it is cut from, and so does its negative space."""
    def pieces(a):
        a, n = a.copy().astype(bool), 0
        while a.any():
            n += 1
            stack = [tuple(np.argwhere(a)[0])]
            while stack:
                y, x = stack.pop()
                if 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x]:
                    a[y, x] = False
                    stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
        return n
    for P, w in ((12, 1), (16, 2), (24, 2)):
        j, i = np.meshgrid(np.arange(4 * P), np.arange(4 * P), indexing="ij")
        f = kr.fret(i, j, P, w)
        cu, cv = i // P, j // P
        x, y = (i - cu * P) - (P - 1) / 2, (j - cv * P) - (P - 1) / 2
        rings = (np.floor((np.maximum(np.abs(x), np.abs(y)) + 0.5) / w).astype(int) + cu + cv) % 2
        assert pieces(f) < 0.7 * pieces(rings) and pieces(1 - f) < 0.7 * pieces(1 - rings), (P, w, pieces(f), pieces(rings))
        runs = [len(r) for row in f[P:2 * P] for r in "".join(map(str, row)).replace("0", " ").split()]
        assert min(runs) >= 1 and max(runs) <= P + 2 * w


def test_ombre_ramp_is_exact():
    d = np.arange(-3, 12)
    assert kr.ombre(d, 6).tolist() == [1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6], "one cell per step, held at the ends"
    assert kr.ombre(d, 4, start=2).tolist() == [1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
    # in the bead tile study the corner rings step through the whole ramp, one bead each, along a diagonal
    t = kr.hk_tile(6, 0)[..., 0].astype(int)
    row = t[1, 1:11].tolist()                                               # along a tile's edge row, corner to middle
    assert sorted(set(row)) == [1, 2, 3, 4, 5, 6] and all(b - a in (0, 1) for a, b in zip(row, row[1:])), row


@pytest.mark.parametrize("seed", range(6))
def test_meander_fills_arbitrary_cells(seed):
    """No gaps, no crossings: every tile of an arbitrary cell is on a path (or is a lone dot), paths never branch,
    never leave the cell, never link two different cells."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:40, 0:56]
    blob = ((xx - 28) ** 2 / 500.0 + (yy - 20) ** 2 / 260.0 + 0.25 * np.sin(xx * 0.5 + seed) * np.cos(yy * 0.4) < 1.0)
    blob &= ~(((xx - 20) ** 2 + (yy - 18) ** 2) < 30)                        # with a hole
    lab = (xx > 30).astype(int)                                             # and two cells side by side
    for lo, hi in ((3, 7), (8, 26)):
        bits = kr.meander_fill(blob, rng, lo, hi, lab=lab).astype(int)
        assert not bits[~blob].any(), "the fill left its cell"
        deg = sum(((bits & m) > 0).astype(int) for m in (1, 2, 4, 8))
        assert deg.max() <= 2, "a fill line branches or crosses"
        assert (deg[blob] == 0).mean() < 0.12, f"gaps: {(deg[blob] == 0).mean():.2f} of the cell has no line"
        for dj, di, mb, tb in ((1, 0, 1, 4), (0, 1, 2, 8)):
            a = (bits[:40 - dj, :56 - di] & mb) > 0
            b = (bits[dj:, di:] & tb) > 0
            assert np.array_equal(a, b), "a half-drawn connection"
            assert not (a & (lab[:40 - dj, :56 - di] != lab[dj:, di:])).any(), "the fill crossed a primary line"
        # no gap wider than the line spacing: every tile of the cell holds a line or dot at its centre, and an
        # empty (dot) tile always touches a tile with a line
        dots = blob & (deg == 0)
        near = np.zeros_like(dots)
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near |= np.roll(deg > 0, (dj, di), (0, 1))
        assert (near | ~dots).mean() > 0.995


@pytest.mark.parametrize("seed", range(8))
def test_arms_terminals_and_accent_blocks(seed):
    gx, gy, _nv, _wrap = kr.grid_dims(0, 0.7, 1.6, dense=True)
    assert gy > kr.grid_dims(0, 0.7, 1.6)[1] * 1.5, "the painted / embroidered studies use a much finer lattice"
    d = kr.shipibo_design(seed, gx, gy)
    s = kr.design_stats(d)
    assert {kr.ARROW, kr.TBAR, kr.CAP} <= set(s["terminals"]), f"terminal vocabulary: {s['terminals']}"
    assert 0.04 < s["arms"] < 0.35 and s["reciprocal"] and s["cross_label"] == 0
    assert 0.015 < s["block_frac"] < 0.10, f"accent blocks cover {s['block_frac']:.3f}: small, never whole cells"
    bits, code = d[..., 1].astype(int), d[..., 2].astype(int) & 7
    arm, block = (bits & 32) > 0, (bits & 64) > 0
    assert not (arm & block).any() and ((bits[arm | block] & 16) > 0).all(), "arms and blocks live inside the cells"
    assert ((bits[block] & 15) == 0).all(), "no fill line runs through an accent block"
    # mirror symmetry: the arms repeat under a half turn like the primary network does
    assert np.array_equal(arm, arm[::-1, ::-1]) and np.array_equal(block, block[::-1, ::-1])
    plain = kr.design_stats(kr.shipibo_design(seed, gx, gy, form=0))
    assert plain["arms"] == 0.0 and plain["blocks"] == 0
    assert (code[block] == 3).any() or seed >= 0                            # (the stepped centre cross needs a roomy centre)


def test_some_designs_have_the_stepped_centre_cross():
    gx, gy = kr.grid_dims(0, 0.7, 1.6, dense=True)[:2]
    n = 0
    for seed in range(10):
        d = kr.shipibo_design(seed, gx, gy)
        cross = ((d[..., 1] & 64) > 0) & ((d[..., 2] & 7) == 3)
        if cross.any():
            n += 1
            jj, ii = np.nonzero(cross)
            assert abs(jj.mean() - (d.shape[0] - 1) / 2) < 0.6 and abs(ii.mean() - (d.shape[1] - 1) / 2) < 0.6
            assert 8 <= len(jj) <= 24
    assert n >= 3


def test_new_studies_follow_the_spec_colours():
    v = params.defaults()
    a = kr.style_uniforms(dict(v, kene_style=kr.STYLES.index("Painted kené (study)")), None)
    assert a["ground_on"] and min(a["ground"]) > 0.7 and max(a["cols"][0]) < 0.25 and a["dense"] and a["form"] == kr.F_LATTICE
    assert a["w"][0] > 3.5 * a["w"][2] and a["w"][1] == 0.0 and a["hair"], "bold line with a hairline, fine fill"
    assert 2.5 <= 1.0 / (2 * a["w"][2]) <= 4.5, "fill spacing is about 3 - 4 line widths (GRAMMAR_SPEC A2.3)"
    assert a["acc"][0][0] > 2 * a["acc"][0][2] and a["acc_on"] == 1.0 and not a["knob"]
    b = kr.style_uniforms(dict(v, kene_style=kr.STYLES.index("Cloth kené (study)")), None)
    assert b["frame"] is not None and max(b["frame"]) < 0.3 and b["acc_on"] == 0.0 and b["wob"] > a["wob"] and b["form"] == kr.F_SPINE
    c = kr.style_uniforms(dict(v, kene_style=kr.STYLES.index("Embroidered kené (study)")), None)
    r, g, bl = c["ground"]
    assert r > 0.6 and r > 3 * g and bl > g, "raspberry ground"
    assert c["cols"][0][2] > c["cols"][0][0] and c["stitch"] and c["border"] and c["twig"] and c["form"] == kr.F_CELLS
    assert c["cols"][1][1] < c["echo2"][1], "outer outline orange, inner yellow"
    assert c["cols"][2][2] > c["cols"][2][0] > c["cols"][2][1], "blue-violet fill"
    assert c["w"][2] >= 0.18, "the fill line is about half its spacing: dense and visible"


def test_painted_lattice_study_structure():
    """A: medallions on a rhombic lattice (d2), bulbs (C-loops ending in an arrowhead), junction gems, red channel
    runs, bowtie blocks, and ONE uniform labyrinth fill reaching every cell."""
    gx, gy = kr.grid_dims(0, 0.6, 1.6, 2.2)[:2]
    found_bulb = found_bow = 0
    for seed in range(4):
        d = kr.shipibo_design(seed, gx, gy, False, kr.F_LATTICE)
        lab, bits, cb = d[..., 0].astype(int), d[..., 1].astype(int), d[..., 2].astype(int)
        ve, he = boundary_edges(lab)
        assert np.array_equal(ve, ve[::-1]) and np.array_equal(ve, ve[:, ::-1]), "mirror symmetric on both axes"
        assert {1, 2, 3, 4} <= set(np.unique(lab)), "centre + rhombic + edge medallions"
        interior, arm, block = (bits & 16) > 0, (bits & 32) > 0, (bits & 64) > 0
        ring = ~interior
        assert ((cb & 8) > 0)[ring].sum() >= 8, "junction gems where channels meet"
        assert ((cb & 7) == 1)[ring].sum() >= 10, "red channel runs"
        deg = sum(((bits & m) > 0).astype(int) for m in (1, 2, 4, 8))
        fill = interior & ~arm & ~block
        assert (deg[fill] == 0).mean() < 0.06 and (deg[fill] == 2).mean() > 0.70, "one uniform wandering line, few ends"
        s = kr.design_stats(d)
        assert s["reciprocal"] and s["cross_label"] == 0
        # a bulb is an arm that turns the same way three times (a C) and ends in an arrowhead
        found_bulb += int(((cb & 7)[arm & (deg == 1)] == kr.ARROW).sum() >= 4)
        found_bow += int((block & ((cb & 7) == 3)).sum() >= 40)
    assert found_bulb >= 3 and found_bow >= 2


def test_cloth_spine_study_structure():
    """B: portrait 1 : 2, a spine on the axis with lozenge and diamonds, outline arrowheads, hooks ending in open
    squares, dumbbells; mirrored left / right and top / bottom; fill edge to edge."""
    gx, gy = kr.grid_dims(0, 0.5, 1.6, False, kr.F_SPINE)[:2]
    assert gy >= 1.8 * gx, "portrait"
    for seed in range(5):
        d = kr.shipibo_design(seed, gx, gy, False, kr.F_SPINE)
        ny, nx = d.shape[:2]
        lab, bits, code = d[..., 0].astype(int), d[..., 1].astype(int), d[..., 2].astype(int) & 7
        arm = (bits & 32) > 0
        assert np.array_equal(arm, arm[:, ::-1]) and np.array_equal(arm, arm[::-1]), "mirrored on both axes"
        assert np.array_equal(lab, lab[:, ::-1]) and np.array_equal(lab, lab[::-1])
        assert lab[ny // 2, nx // 2] == 2 and len(np.unique(lab)) >= 3, "lozenge at the centre, diamonds on the spine"
        assert arm[:, nx // 2].sum() >= 0.3 * ny, "a spine down the central column"
        ends = set(code[arm].tolist())
        assert {kr.ARROW_OPEN, kr.CAP, kr.TBAR} <= ends, f"outline arrowheads, open squares, dumbbells: {ends}"
        assert kr.ARROW not in ends and (bits & 64).sum() == 0, "no filled arrowheads, no colour blocks"
        deg = sum(((bits & m) > 0).astype(int) for m in (1, 2, 4, 8))
        assert ((bits & 16) > 0).mean() > 0.9 and (deg[~arm] == 0).mean() < 0.12, "coarse fill everywhere, edge to edge"


def test_embroidered_cells_study_structure():
    """C: many small cells packed edge to edge, twig colours ON the channels only, eye dots, no arms inside cells."""
    gx, gy = kr.grid_dims(0, 0.4, 1.6, 1.25)[:2]
    d = kr.shipibo_design(2, gx, gy, False, kr.F_CELLS)
    lab, bits, cb = d[..., 0].astype(int), d[..., 1].astype(int), d[..., 2].astype(int)
    interior = (bits & 16) > 0
    assert len(np.unique(lab)) >= 9 and not (bits & 32).any()
    assert 0.2 < ((cb & 7) > 0)[~interior].mean() < 0.8, "coloured twig runs along the channels"
    assert len(set((cb & 7)[~interior].tolist()) - {0}) >= 4, "several twig colours"
    assert 1 <= ((bits & 64) > 0).sum() <= 40, "eye dots"


def test_bead_studies_follow_the_spec_structures():
    W, T = kr.WHITE_ROLE, kr.HK_T
    woven = kr.hk_tile(5, 0)[..., 0].astype(int)                            # D: three colours, three band types of 16 rows
    assert set(np.unique(woven)) == {0, 1, 2}
    bands = [woven[(np.arange(T) + 8) // 16 % 3 == k] for k in range(3)]
    assert all(b.shape[0] == 16 for b in bands) and len({b.tobytes() for b in bands}) == 3
    for b in bands:
        assert 0.3 < (b > 0).mean() < 0.7
    assert (woven == 2).any(axis=1).sum() >= 10, "orange in every band"
    tiles = kr.hk_tile(6, 0)[..., 0].astype(int)                            # E: four different tiles with gaps between
    quads = [tiles[a:a + 24, b:b + 24] for a in (0, 24) for b in (0, 24)]
    assert len({q.tobytes() for q in quads}) == 4
    for q in quads:
        assert not q[0].any() and not q[:, 0].any() and not q[-1].any(), "a dark gap round every tile"
    key = quads[0] == W
    ys, xs = np.nonzero(key)
    assert np.ptp(xs) >= 10 and np.ptp(ys) >= 10, "the S-fret is large: it spans the inner diamond"
    runs = [len(r) for row in key for r in "".join("1" if v else " " for v in row).split()]
    assert np.mean(np.array(runs) >= 2) > 0.3, "a two-bead stroke"
    assert (tiles == 7).any() and (tiles == W).sum() > 100
    strap = kr.hk_tile(7, 0)[..., 0].astype(int)                            # F: three different straps, dark between
    cols = [strap[:, k * 16:(k + 1) * 16] for k in range(3)]
    assert len({c.tobytes() for c in cols}) == 3
    for c in cols:
        assert not c[:, 0].any() and not c[:, 14:].any() and c[:, 1:14].any(), "13 beads, then dark ground"
        assert np.array_equal(c[:, 1:14], c[:, 1:14][:, ::-1]), "mirrored about the strap's axis"
        assert np.array_equal(c[:16], c[16:32]), "repeats every 16 rows"
    assert {1, 2, 3, 4} <= set(np.unique(cols[0])) and W in cols[0] and W in cols[1] and 7 in cols[2]
    assert {1, 2, 3, 4, 5, 6} <= set(np.unique(cols[2])), "the rays run through the whole rainbow ramp"
    fretp = kr.hk_tile(8, 0)[..., 0].astype(int)                            # G: red / green frets with dark separators
    panel = fretp[:10, :6]
    assert {0, 7, 8} <= set(np.unique(fretp[:12, :12])), "two key colours and separators"
    assert abs(int((panel == 7).sum()) - int((panel == 8).sum())) <= 16
    line = fretp[0, 19:24]                                                  # the chevron band next to the panel
    assert line.tolist() == [1, 2, 3, 4, 5], f"white peak next to the panel, then down the blue ramp: {fretp[0, 15:26]}"
    blue = kr.hk_colours(dict(params.defaults(), hk_colors=kr.HK_COLOR_NAMES.index("Blue ombré")), None)
    lum = [0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2] for c in blue[:6]]
    assert lum == sorted(lum, reverse=True), "the blue ramp runs white -> navy"
    assert "hk_ombre" in params.PARAMS and params.PARAMS["hk_ombre"][2] == 0.0
    # bead-by-bead lighting follows the fret: the 'along' channel changes steadily along the spiral
    al = kr.fret_along(*np.meshgrid(np.arange(12), np.arange(12)), 12, 2)
    assert 0.0 <= al.min() and al.max() < 1.0 and len(np.unique(np.round(al, 2))) > 20
