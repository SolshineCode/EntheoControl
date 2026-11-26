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
        box = page.locator('#pg-perform [data-mt="param:zoom"] .sl').bounding_box()
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
            assert has == (t in keys), f"tab {t}: power switch {'missing' if t in keys else 'unexpected'}"
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
