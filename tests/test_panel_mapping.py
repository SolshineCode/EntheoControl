"""The MAPPING tab in headless Edge/Chrome against a real controller + surfaces.Mapper. Skipped without Playwright."""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pw_api = pytest.importorskip("playwright.sync_api")
import control  # noqa: E402
import surfaces as S  # noqa: E402
from remote_control import RemoteControl  # noqa: E402

SHOTS = ROOT / "test_frames" / "surfaces"


@pytest.fixture
def served(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    mapper = S.Mapper(tmp_path / "surfaces.json")
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            for c in ctl.apply(rc.drain()):
                if c.get("cmd") == "surf":
                    mapper.command(c)
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            rc.publish_state(dict(ctl.state(), corners=mapper.corners(), surfaces=mapper.set.state(), res=[1280, 800], fps=60))
            time.sleep(1 / 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/?k=tok", mapper
    stop.set()
    t.join(timeout=2)
    ctl.close()
    rc.stop()


def launch(pw):
    for channel in ("msedge", "chrome", None):
        try:
            return pw.chromium.launch(channel=channel, headless=True) if channel else pw.chromium.launch(headless=True)
        except Exception:
            continue
    pytest.skip("no Chromium-family browser available")


def until(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.03)
    return False


def open_tab(pw, url, size=(1680, 1000)):
    br = launch(pw)
    page = br.new_page(viewport=dict(width=size[0], height=size[1]))
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(url)
    page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
    page.click('#tabs button[data-tab="mapping"]')
    page.wait_for_function("window.__map.state() && !window.__map.state().local", timeout=5000)
    return br, page, errors


def at(page, x, y):
    """Projector pixel -> page coordinates on the canvas."""
    v = page.evaluate("window.__map.view()")
    box = page.locator("#mapcv").bounding_box()
    return box["x"] + v["ox"] + x * v["s"], box["y"] + v["oy"] + y * v["s"]


def drag(page, a, b, steps=8, **mods):
    x0, y0 = at(page, *a)
    x1, y1 = at(page, *b)
    page.mouse.move(x0, y0)
    page.mouse.down()
    for k in range(1, steps + 1):
        page.mouse.move(x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps)
    page.wait_for_timeout(60)
    page.mouse.up()


@pytest.mark.parametrize("size", [(1680, 1000), (390, 844)])
def test_mapping_tab_loads_clean_and_fits(served, size):
    url, mapper = served
    with pw_api.sync_playwright() as pw:
        br, page, errors = open_tab(pw, url, size)
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
        assert page.locator(".sfrow").count() == 1 and page.locator("#mapcv").count() == 1
        assert "Tilt +0.00" in page.locator("#mread").text_content() and "Aspect 1.600" in page.locator("#mread").text_content()
        br.close()
    assert errors == []


def test_panel_geometry_matches_the_python_model(served):
    url, _ = served
    quads = [[[0, 0], [1280, 0], [1280, 800], [0, 800]], [[128, 80], [1150, 40], [1020, 700], [260, 680]],
             [[250, 0], [750, 0], [1000, 1000], [0, 1000]], [[500, 200], [900, 200], [800, 600], [400, 600]]]
    with pw_api.sync_playwright() as pw:
        br, page, errors = open_tab(pw, url)
        for q in quads:
            js, py = page.evaluate("q => window.__map.geo(q, [1280, 800])", q), S.describe(q)
            for key in ("angles", "edges", "keystone", "center"):
                assert js[key] == pytest.approx(py[key], abs=1e-6), key
            for key in ("rotation", "skew", "aspect", "width", "height", "area"):
                assert js[key] == pytest.approx(py[key], abs=1e-6), key
            assert sum(js["pct"], []) == pytest.approx(sum(py["pct"], []))
        br.close()
    assert errors == []


def test_surface_list_add_rename_duplicate_reorder_enable_delete(served):
    url, mapper = served
    st = mapper.set
    with pw_api.sync_playwright() as pw:
        br, page, errors = open_tab(pw, url)
        page.click('#mlist button:text-is("+ ADD")')
        assert until(lambda: len(mapper.set.surfaces) == 2 and mapper.set.sel == 2)
        page.wait_for_function("window.__map.state().surfaces.length === 2 && window.__map.state().sel === 2")
        page.fill("#mname", "Box front")
        page.press("#mname", "Enter")
        assert until(lambda: st.get(2)["name"] == "Box front")
        page.click('#mlist button:text-is("DUPLICATE")')
        assert until(lambda: len(st.surfaces) == 3 and st.sel == 3)
        page.wait_for_function("window.__map.state().sel === 3")
        page.click('#mlist button:text-is("▲")')
        assert until(lambda: [s["id"] for s in st.surfaces] == [1, 3, 2])
        page.wait_for_timeout(900)
        page.locator(".sfrow").nth(0).locator(".meye").click()
        assert until(lambda: st.get(1)["enabled"] is False)
        page.wait_for_timeout(900)
        page.locator(".sfrow").nth(2).click()
        assert until(lambda: st.sel == 2)
        page.click('#mlist button:text-is("DELETE")')
        page.wait_for_timeout(150)
        assert len(st.surfaces) == 3, "one tap only arms the delete"
        page.click('#mlist button:text-is("TAP AGAIN")')
        assert until(lambda: [s["id"] for s in st.surfaces] == [1, 3])
        page.wait_for_function("window.__map.state().surfaces.length === 2")
        assert len(page.locator(".sfrow .mnm").all_text_contents()) == 2
        br.close()
    assert errors == []


def test_drag_snap_nudge_typed_values_and_undo(served):
    url, mapper = served
    st = mapper.set
    with pw_api.sync_playwright() as pw:
        br, page, errors = open_tab(pw, url)
        # drag TL of the full-frame surface inward: one drag = one undo step, whole pixels
        drag(page, (0, 0), (203.4, 151.7))
        assert until(lambda: st.surfaces[0]["corners"][0] != [0, 0])
        page.wait_for_timeout(250)
        tl = st.surfaces[0]["corners"][0]
        assert abs(tl[0] - 203) <= 2 and abs(tl[1] - 152) <= 2 and tl == [round(tl[0]), round(tl[1])]
        assert st.state()["undo"] == 1
        # arrows nudge the selected point by the step; Shift = 0.1 px
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowDown")
        assert until(lambda: st.surfaces[0]["corners"][0] == [tl[0] + 1, tl[1] + 1])
        page.keyboard.press("Shift+ArrowLeft")
        assert until(lambda: st.surfaces[0]["corners"][0] == pytest.approx([tl[0] + 0.9, tl[1] + 1]))
        assert st.state()["undo"] == 2, "a run of nudges on one point is one undo step"
        page.wait_for_timeout(900)
        # typed coordinates and a typed angle
        box = page.locator(".geo input").first                       # TL x
        box.fill("128")
        box.press("Enter")
        assert until(lambda: st.surfaces[0]["corners"][0][0] == 128)
        ang = page.locator(".geo tr").nth(2).locator("input").nth(2)  # TR angle
        ang.fill("80")
        ang.press("Enter")
        assert until(lambda: abs(S.interior_angles(st.surfaces[0]["corners"])[1] - 80) < 1e-3)
        page.wait_for_timeout(900)
        page.click('#msel button:text-is("SQUARE IT UP")')
        assert until(lambda: max(abs(a - 90) for a in S.interior_angles(st.surfaces[0]["corners"])) < 1e-4)
        # undo walks back, redo forward (Ctrl+Z / Ctrl+Y with the tab open)
        n = st.state()["undo"]
        page.locator("#mapcv").hover()
        page.keyboard.press("Control+z")
        assert until(lambda: st.state()["undo"] == n - 1 and st.state()["redo"] == 1)
        page.keyboard.press("Control+y")
        assert until(lambda: st.state()["undo"] == n and st.state()["redo"] == 0)
        # a second surface: drag its TL next to surface 1's TL and it snaps exactly onto it
        page.wait_for_timeout(800)
        page.click('#mlist button:text-is("+ ADD")')
        assert until(lambda: len(st.surfaces) == 2)
        page.wait_for_function("window.__map.state().surfaces.length === 2 && window.__map.state().sel === 2")
        page.wait_for_timeout(800)
        target = st.surfaces[0]["corners"][2]                         # BR of surface 1
        src = st.get(2)["corners"][2]
        drag(page, src, (target[0] - 4, target[1] + 3), steps=12)
        assert until(lambda: st.get(2)["corners"][2] == pytest.approx(target, abs=1e-9)), (st.get(2)["corners"], target)
        # dragging inside a surface moves the whole surface
        page.wait_for_timeout(800)
        before = [list(p) for p in st.get(2)["corners"]]
        cx, cy = S.centroid(before)
        drag(page, (cx, cy + 40), (cx + 30, cy + 60))
        assert until(lambda: st.get(2)["corners"] != before)
        page.wait_for_timeout(200)
        after = st.get(2)["corners"]
        d = [after[0][0] - before[0][0], after[0][1] - before[0][1]]
        assert all(abs(after[k][0] - before[k][0] - d[0]) < 1e-9 and abs(after[k][1] - before[k][1] - d[1]) < 1e-9 for k in range(4))
        assert S.edge_lengths(after) == pytest.approx(S.edge_lengths(before))
        br.close()
    assert errors == []


def test_look_controls_overlay_mask_and_screenshot(served):
    url, mapper = served
    st = mapper.set
    for name, pts in (("Box front", [[330, 330], [650, 370], [650, 700], [330, 640]]),
                      ("Box side", [[650, 370], [900, 310], [900, 620], [650, 700]]),
                      ("Box top", [[330, 330], [590, 280], [900, 310], [650, 370]])):
        if name != "Box front":
            st.apply(S.validate_cmd(dict(op="add")))
        sid = st.surfaces[-1]["id"]
        st.apply(S.validate_cmd(dict(op="rename", id=sid, name=name)))
        st.apply(S.validate_cmd(dict(op="corners", id=sid, pts=pts)))
    with pw_api.sync_playwright() as pw:
        br, page, errors = open_tab(pw, url)
        page.wait_for_function("window.__map.state().surfaces.length === 3")
        page.locator(".sfrow").nth(0).click()
        assert until(lambda: st.sel == 1)
        page.wait_for_timeout(800)
        page.click('#msel button:text-is("EDGE GLOW")')
        assert until(lambda: st.get(1)["content"] == "glow" and st.get(1)["glow"] == 1.0)
        page.click('#msel button:text-is("SPANNED")')
        assert until(lambda: st.get(1)["source"]["mode"] == "span")
        page.click('#mlist button:text-is("ALL")')
        page.click('#mlist button:text-is("TEST GRID")')
        assert until(lambda: st.overlay == dict(mode="all", grid=True))
        page.wait_for_timeout(800)
        page.click('#msel button:text-is("MASK")')
        assert until(lambda: st.get(1)["shape"] == "poly" and len(st.get(1)["mask"]) == 4)
        page.wait_for_timeout(800)
        page.click('#msel button:text-is("+ POINT")')
        assert until(lambda: len(st.get(1)["mask"]) == 5)
        page.wait_for_timeout(800)
        sl = page.locator("#msel .msl .sl").nth(3).bounding_box()      # soft edge 0..120 px
        page.mouse.click(sl["x"] + sl["width"] * 0.25, sl["y"] + sl["height"] / 2)
        assert until(lambda: 20 < st.get(1)["feather"] < 40)
        page.locator(".mcol").nth(4).click()
        assert until(lambda: st.get(1)["color"] == pytest.approx(list(S.COLORS[4])))
        page.wait_for_timeout(900)
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / "panel_mapping.png"))
        # the magnifier while a corner is held
        x, y = at(page, 650, 370)
        page.locator(".sfrow").nth(1).click()
        page.wait_for_timeout(800)
        page.mouse.move(*at(page, 900, 310))
        page.mouse.down()
        page.mouse.move(*at(page, 905, 300))
        page.mouse.move(*at(page, 912, 296))
        page.wait_for_timeout(100)
        page.screenshot(path=str(SHOTS / "panel_mapping_loupe.png"))
        page.mouse.up()
        br.close()
    assert errors == []
