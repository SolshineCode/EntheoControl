"""Loads the real panel in headless Edge/Chrome and checks it runs clean. Skipped without Playwright or a browser."""
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
from remote_control import RemoteControl  # noqa: E402


@pytest.fixture
def served(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art",
                       tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            ctl.apply(rc.drain())
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60))   # flat, as the engine publishes it
            time.sleep(1 / 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/?k=tok", ctl
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


@pytest.mark.parametrize("size", [(1920, 1080), (390, 844)])
def test_panel_loads_and_drives_the_controller(served, size):
    url, ctl = served
    errors = []
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        page = br.new_page(viewport=dict(width=size[0], height=size[1]))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(url)
        page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
        tabs = page.eval_on_selector_all("#tabs button[data-tab]", "els => els.map(e => e.dataset.tab)")
        assert {"perform", "motion", "shape", "color", "layers", "scenes", "ai", "midi", "mapping"} <= set(tabs)
        for t in tabs:
            page.click(f'#tabs button[data-tab="{t}"]')
            page.wait_for_timeout(120)
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2, t
        # every non-trigger param has a control somewhere (generated from PARAMS)
        have = set(page.eval_on_selector_all("[data-mt^='param:']", "els => els.map(e => e.dataset.mt.slice(6))"))
        want = {n for n in control.CONT if n != "art_index"}
        assert want <= have, want - have
        page.click('#tabs button[data-tab="perform"]')
        zoom = page.locator('#pg-perform [data-mt="param:zoom"] .sl')
        zoom.scroll_into_view_if_needed()                 # phone: the macros sit below the scene grid
        box = zoom.bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.1)
        page.keyboard.down("KeyB")
        page.wait_for_timeout(500)
        assert ctl.base["zoom"] > 0.6 and ctl.held["blackout"] is True
        page.keyboard.up("KeyB")
        page.keyboard.press("Space")
        page.wait_for_timeout(400)
        assert ctl.held["blackout"] is False
        br.close()
    assert errors == []


def until(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.03)
    return False


def test_triggers_cannot_stick_and_scene_commands_are_retried(served):
    url, ctl = served
    ctl.quantize, ctl.morph_bars = "off", 0.0
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        page = br.new_page(viewport=dict(width=1920, height=1080))
        page.add_init_script("window.__pads=[];navigator.getGamepads=function(){return window.__pads;};")
        page.goto(url)
        page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
        page.wait_for_timeout(300)
        # 1. the release is lost on the network: after the connection is back the trigger is let go
        page.keyboard.down("KeyB")
        assert until(lambda: ctl.held["blackout"])
        page.route("**/cmd", lambda route: route.abort())
        page.keyboard.up("KeyB")
        page.keyboard.press("Digit3")                       # a scene recall made while the link is down
        page.wait_for_timeout(600)
        assert ctl.held["blackout"] is True and ctl.cur != 2
        page.unroute("**/cmd")
        assert until(lambda: ctl.held["blackout"] is False), "blackout stayed stuck after reconnect"
        assert until(lambda: ctl.cur == 2), "scene recall was dropped instead of retried"
        # 2. a full queue sheds fader traffic, never a trigger release
        n, has_off = page.evaluate("""(() => {
            const P = window.__panel;
            for (let i = 0; i < 300; i++) P.send({cmd:'ripple', x:0.5, y:0.5});
            P.send({cmd:'trigger', name:'strobe', on:false});
            for (let i = 0; i < 300; i++) P.send({cmd:'ripple', x:0.5, y:0.5});
            const q = P.pend();
            return [q.length, q.some(p => p.c.cmd === 'trigger' && p.c.name === 'strobe' && p.c.on === false)];
        })()""")
        assert n <= 97 and has_off
        # 3. a gamepad that vanishes with a button down releases it
        page.evaluate("window.__pads=[{connected:true,id:'Fake Pad',axes:[0,0,0,0],"
                      "buttons:Array.from({length:17},(_, i)=>({pressed:i===0,value:i===0?1:0}))}]")
        assert until(lambda: ctl.held["flash"]), "gamepad A did not hold flash"
        page.evaluate("window.__pads=[]")
        assert until(lambda: ctl.held["flash"] is False), "flash stayed on after the gamepad disconnected"
        br.close()


def test_bypass_switches_on_tabs_pages_and_layer_rows(served):
    """Power switches drive the controller's bypass set; nothing stored changes and the tab does not switch."""
    url, ctl = served
    errors = []
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        page = br.new_page(viewport=dict(width=1920, height=1080))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_timeout(900)
        keys = set(control.BYPASS_COVER)
        tabs = page.eval_on_selector_all("#tabs button[data-tab]", "els => els.map(e => [e.dataset.tab, !!e.querySelector('.pw')])")
        for t, has in tabs:
            want = t in keys and t != "layers"     # LAYERS has no header icon: one click there blacked out everything (Opus review A2)
            assert has == want, f"tab {t}: power switch {'missing' if want else 'unexpected'}"
        assert dict(tabs)["motion"] and not dict(tabs)["mapping"] and not dict(tabs)["perform"]
        base = dict(ctl.base)
        page.click('#tabs button[data-tab="motion"] .pw')                # header icon: toggles, does not switch tab
        page.wait_for_timeout(500)
        assert ctl.bypass == {"motion"} and page.evaluate("window.__panel.state().tab") == "perform"
        assert "byp" in page.get_attribute('#tabs button[data-tab="motion"]', "class")
        page.click('#tabs button[data-tab="motion"]')
        page.wait_for_timeout(300)
        assert "byp" in page.get_attribute("#pg-motion", "class") and page.inner_text("#pg-motion .pwbar button") == "OFF"
        page.click("#pg-motion .pwbar button")                           # big switch at the top of the tab
        page.wait_for_timeout(500)
        assert ctl.bypass == set() and page.inner_text("#pg-motion .pwbar button") == "ON"
        page.click('#tabs button[data-tab="perform"]')
        row = page.locator("#pg-perform .lrow").first                    # kene
        row.locator(".ey").click()
        page.wait_for_timeout(500)
        assert ctl.bypass == {"l_kene"} and row.locator(".ey").inner_text() == "OFF"
        assert ctl.base["l_kene"] == base["l_kene"], "EYE must not edit the stored opacity any more"
        row.locator(".ey").click()
        page.click("#fxoff")
        page.wait_for_timeout(500)
        assert ctl.bypass == {"fx"} and page.inner_text("#fxoff") == "ALL FX OFF"
        page.click("#fxoff")
        page.wait_for_timeout(500)
        assert ctl.bypass == set() and ctl.base == base
        br.close()
    assert errors == []


def test_preview_is_off_by_default_and_one_state_poll(tmp_path):
    """Preview off = no /preview, no /layer-thumb, no capture job. Remembered per browser. One /state poll per page."""
    import socket as _s
    sk = _s.socket(); sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]; sk.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    stop, jobs = threading.Event(), []

    def loop():
        while not stop.is_set():
            ctl.apply(rc.drain())
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            j = rc.preview_job()
            if j:
                jobs.append(j)
                if j[0] != "thumb":                                      # stand in for the engine's capture
                    rc.set_preview(bytes([255, 216, 255, 217]), j[1] or ())
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60))
            time.sleep(1 / 60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    reqs = []
    try:
        with pw_api.sync_playwright() as pw:
            br = launch(pw)
            ctx = br.new_context(viewport=dict(width=1600, height=900))
            page = ctx.new_page()
            page.on("request", lambda r: reqs.append(r.url.split("?")[0].rsplit("/", 1)[-1] if "/layer-thumb/" not in r.url else "layer-thumb"))
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.click('#tabs button[data-tab="layers"]')
            page.wait_for_timeout(2500)
            assert "preview" not in reqs and "layer-thumb" not in reqs and jobs == [], "preview must be OFF by default"
            assert page.inner_text("#pvsw") == "PREVIEW OFF" and not page.is_visible("#mon")
            n_state = reqs.count("state")
            assert 8 <= n_state <= 22, f"one /state poll at about 6 Hz expected, saw {n_state} requests in 2.5 s"
            page.click("#pvsw")
            page.wait_for_timeout(2500)
            assert page.inner_text("#pvsw") == "PREVIEW ON" and "preview" in reqs and "layer-thumb" in reqs
            assert {"out", "thumb"} <= {j[0] for j in jobs}
            page.click('#tabs button[data-tab="perform"]')
            page.wait_for_timeout(400)
            reqs.clear()
            page.wait_for_timeout(1800)
            assert "layer-thumb" not in reqs, "thumbnails are for the LAYERS tab only"
            assert "preview" in reqs, sorted(set(reqs))
            page.reload()                                                # remembered per browser
            page.wait_for_timeout(1200)
            assert page.inner_text("#pvsw") == "PREVIEW ON"
            page.click("#pvsw")
            page.wait_for_timeout(3600)                                  # past the server's 3 s linger
            reqs.clear(); jobs.clear()
            page.wait_for_timeout(1500)
            assert "preview" not in reqs and "layer-thumb" not in reqs and jobs == [], "switched off: everything stops"
            br.close()
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()


def _open(pw, url, w=1920, h=1080):
    br = launch(pw)
    page = br.new_page(viewport=dict(width=w, height=h))
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
    page.click('#tabs button[data-tab="perform"]')
    page.wait_for_timeout(300)
    return br, page, errors


FITS = """(() => { const m = document.getElementById('main'), t = document.getElementById('trig').getBoundingClientRect();
    const low = Math.max(...[...document.querySelectorAll('#pg-perform > .pfz')].map(e => e.getBoundingClientRect().bottom));
    return [m.scrollHeight - m.clientHeight, document.documentElement.scrollWidth - innerWidth, low <= t.top + 1,
            document.documentElement.scrollHeight - innerHeight]; })()"""


def test_perform_fits_one_screen_at_1920x1080(served):
    """PERFORM is one screen on a 1920x1080 laptop: no vertical or horizontal scrolling, every zone above the trigger
    row, also with every layer switched on (the layer strip at its tallest)."""
    url, ctl = served
    with pw_api.sync_playwright() as pw:
        br, page, errors = _open(pw, url)
        zones = page.eval_on_selector_all("#pg-perform > *", "els => els.map(e => e.className)")
        assert len(zones) == 4 and all("pfz" in z for z in zones), zones      # scenes, macros, live layers, quick boxes
        assert page.locator("#pg-perform .pfs .sc").count() == control.N_SCENES
        for n in ("zoom", "speed", "spin", "travel", "symmetry", "chase", "trails"):
            assert page.locator(f'#pg-perform .pfm [data-mt="param:{n}"] .fm').count() == 1, n   # fader + its MOD button
        assert page.is_visible("#pg-perform .pfm .rip") and page.is_visible("#pg-perform .pfsrc")
        assert page.is_visible("#pg-perform .txmini input") and page.is_visible("#pfaip")
        over, wide, above, doc = page.evaluate(FITS)
        assert over <= 1 and wide <= 2 and above and doc <= 1, (over, wide, above, doc)
        layers = [n for n in control.CONT if n.startswith("l_")] + ["serpent"]
        for n in layers:
            ctl.set(n, 0.5)
        page.wait_for_timeout(700)
        assert page.locator("#pg-perform .pfl .lrow:not(.gone)").count() == len(layers)
        over, wide, above, doc = page.evaluate(FITS)
        assert over <= 1 and wide <= 2 and above and doc <= 1, ("all layers on", over, wide, above, doc)
        br.close()
    assert errors == []


def test_perform_zones_drive_the_same_commands(served):
    """Scene pads, transition + autopilot selectors, the live layer strip and the drawers of the PERFORM tab."""
    url, ctl = served
    ctl.quantize, ctl.morph_bars = "off", 0.0
    with pw_api.sync_playwright() as pw:
        br, page, errors = _open(pw, url)
        page.click("#pg-perform .pfs .sc >> nth=2")
        assert until(lambda: ctl.cur == 2)
        page.click("#pg-perform .pftr .seg >> nth=0 >> text=4")
        page.click("#pg-perform .pftr .seg >> nth=2 >> text=64")                # the slower autopilot periods
        assert until(lambda: ctl.morph_bars == 4 and ctl.auto["bars"] == 64)
        # only the layers that are showing have a row; the rest is a count with a drawer
        on = [n for n in list(control.CONT) if (n.startswith("l_") or n == "serpent") and ctl.base[n] > 0.02]
        assert page.locator("#pg-perform .pfl .lrow:not(.gone)").count() == len(on)
        assert ctl.base["l_fractal"] == 0
        page.click("#pg-perform .pfoffb")
        page.click(".pfdw.on .pfoff button:has-text('Math fractals')")
        assert until(lambda: ctl.base["l_fractal"] > 0.5), "the OFF drawer brings a layer in"
        page.wait_for_timeout(300)
        assert page.locator("#pg-perform .pfl .lrow:not(.gone)").count() == len(on) + 1
        # Esc closes a drawer; it does not start PANIC
        page.click("#pg-perform .pfm .pfh button")
        assert page.locator(".pfdw.on [data-mt='param:rotate']").count() == 1
        page.keyboard.press("Escape")
        page.wait_for_timeout(700)
        assert page.locator(".pfdw.on").count() == 0 and ctl.panics == 0
        # a row switched off with EYE stays in place so it can be switched straight back on
        row = page.locator("#pg-perform .pfl .lrow:not(.gone)").first
        row.locator(".ey").click()
        assert until(lambda: len(ctl.bypass) == 1)
        assert row.is_visible() and row.locator(".ey").inner_text() == "OFF"
        row.locator(".ey").click()
        assert until(lambda: ctl.bypass == set())
        # the ripple pad sends ripples; + ADD opens BUILD
        box = page.locator("#pg-perform .pfm .rip").bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.5)
        page.click("#pg-perform .pfadd")
        assert page.evaluate("window.__panel.state().tab") == "build"
        br.close()
    assert errors == []


def _hold_panic_button(page):
    box = page.locator("#panic").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    assert page.evaluate("([x, y]) => !!document.elementFromPoint(x, y).closest('#panic')", [x, y]), "PANIC is covered"
    page.mouse.move(x, y)
    page.mouse.down()
    page.wait_for_timeout(750)
    page.mouse.up()


@pytest.mark.parametrize("size", [(1920, 1080), (1366, 768), (390, 844)])
def test_panic_is_one_hold_away_whatever_is_open_or_focused(served, size):
    """Opus review H3: a drawer, the MOD editor or the key sheet never covers the top bar, and holding Esc reaches PANIC
    on the SAME press that closed a drawer or left a text box."""
    url, ctl = served
    with pw_api.sync_playwright() as pw:
        br, page, errors = _open(pw, url, *size)
        more = "#pg-perform .pfm .pfh button"
        # 1. the PANIC button under an open drawer, the MOD editor and the key sheet
        for n, opener in enumerate([lambda: page.click(more), lambda: page.evaluate("window.__panel.openMod('zoom')"),
                                    lambda: page.keyboard.press("?")]):
            opener()
            page.wait_for_timeout(200)
            assert page.locator(".pfdw.on, #modp.on, #ov.on").count() == 1
            _hold_panic_button(page)
            assert until(lambda: ctl.panics == n + 1), f"overlay {n}: PANIC did not fire"
            page.keyboard.press("Escape")                       # a tap only closes
            page.wait_for_timeout(200)
            assert page.locator(".pfdw.on, #modp.on, #ov.on").count() == 0 and ctl.panics == n + 1
        # 2. hold Esc with a drawer open: it closes AND the same hold is PANIC
        page.click(more)
        page.keyboard.down("Escape")
        page.wait_for_timeout(750)
        page.keyboard.up("Escape")
        assert page.locator(".pfdw.on").count() == 0 and until(lambda: ctl.panics == 4), "hold-Esc with a drawer open"
        # 3. hold Esc while typing in the PERFORM text box and the AI box: focus is released, PANIC fires
        for k, sel in enumerate(("#pg-perform .txmini input", "#pfaip")):
            page.click(sel)
            page.keyboard.type("om")
            page.keyboard.down("Escape")
            page.wait_for_timeout(750)
            page.keyboard.up("Escape")
            assert page.evaluate("document.activeElement.tagName") != "INPUT", sel
            assert until(lambda: ctl.panics == 5 + k), sel
        page.click("#pfaip")                                    # a tap of Esc leaves the box and is not a PANIC
        page.keyboard.press("Escape")
        page.wait_for_timeout(700)
        assert ctl.panics == 6 and page.evaluate("document.activeElement.tagName") != "INPUT"
        br.close()
    assert errors == []


def test_drawers_survive_other_tabs_and_touch_targets_are_44px(served):
    """Opus review L5 (a drawer opened from another tab stays open) and L6 (touch screens: 44 px targets)."""
    url, ctl = served
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        for w, h in ((1366, 768), (820, 1180), (390, 844)):
            ctx = br.new_context(viewport=dict(width=w, height=h), has_touch=True, is_mobile=True)
            page = ctx.new_page()
            page.goto(url)
            page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
            page.evaluate("window.__panel.setTab('perform')")
            page.wait_for_timeout(300)
            if not page.evaluate("matchMedia('(pointer:coarse)').matches"):
                pytest.skip("this browser does not emulate a coarse pointer")
            small = page.evaluate("""[...document.querySelectorAll('#pg-perform .seg button, #pg-perform .pfh > button, #pg-perform .ey, #pg-perform .fm, #pg-perform .sc, #pg-perform .pfdk button, #pg-perform .pfair button, #pg-perform .txmini button')]
                .filter(b => b.offsetParent).map(b => { const r = b.getBoundingClientRect(); return [b.textContent.trim(), Math.round(r.width), Math.round(r.height)]; })
                .filter(x => x[1] < 44 || x[2] < 44)""")
            assert small == [], (w, h, small)
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2, (w, h)
            ctx.close()
        page = br.new_page(viewport=dict(width=1920, height=1080))
        page.goto(url)
        page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
        page.evaluate("window.__panel.setTab('scenes'); window.__d = window.__panel.mkDrawer('probe'); window.__d.open()")
        page.wait_for_timeout(900)
        assert page.locator(".pfdw.on").count() == 1, "a drawer used outside PERFORM must stay open"
        br.close()
