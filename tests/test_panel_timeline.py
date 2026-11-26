"""TIMELINE tab, SHOW menu, preset browser and the PERFORM hook in a headless browser against a real Controller.
Screenshots: test_frames/timeline/*.png."""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pw_api = pytest.importorskip("playwright.sync_api")
import control  # noqa: E402
from remote_control import RemoteControl  # noqa: E402

SHOTS = Path(__file__).resolve().parents[1] / "test_frames" / "timeline"


@pytest.fixture
def served(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art",
                       tex_dir=tmp_path / "tex")
    rc.state_dir = tmp_path
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    stop = threading.Event()
    lock = threading.Lock()

    def loop():
        while not stop.is_set():
            with lock:
                ctl.apply(rc.drain())
                ctl.update(1 / 60, [0.5, 0.3, 0.1])
                rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60))
            time.sleep(1 / 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/?k=tok", ctl, lock
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
    pytest.skip("no chromium browser")


def wait(cond, s=5.0):
    t = time.time() + s
    while time.time() < t:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_timeline_tab_show_menu_presets_and_perform_hook(served):
    url, ctl, lock = served
    SHOTS.mkdir(parents=True, exist_ok=True)
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        page = br.new_page(viewport=dict(width=1400, height=880))
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(url)
        page.wait_for_selector("#shbtn")
        # PERFORM hook
        assert page.evaluate("!!window.__timeline && !!window.__timeline.card.parentNode")
        page.click("#tabs button[data-tab=timeline]")
        page.wait_for_selector("#tlcv")
        # an automation lane + keys
        page.select_option(".tladd select", "zoom")
        assert wait(lambda: "zoom" in ctl.tl.by_param)
        with lock:
            tr = ctl.tl.track(ctl.tl.by_param["zoom"].tid)
            ctl.apply([{"cmd": "tl", "op": "items_add", "track": tr["id"], "items": [
                {"t": 0, "v": -0.6, "ease": "smooth"}, {"t": 8, "v": 0.8, "ease": "linear"}, {"t": 16, "v": 0.0, "ease": "hold"}]},
                {"cmd": "tl", "op": "items_add", "track": ctl.tl.first("marker")["id"], "items": [{"t": 0, "name": "intro"},
                                                                                             {"t": 8, "name": "drop"}]},
                {"cmd": "tl", "op": "items_add", "track": ctl.tl.first("cue")["id"], "items": [{"t": 4, "kind": "scene", "slot": 2, "morph": 1}]},
                {"cmd": "tl", "op": "track_add", "kind": "trigger"}])
        time.sleep(0.6)
        # double-click on the automation lane adds a key (lock is off on a mouse)
        box = page.locator("#tlcv").bounding_box()
        n0 = len(ctl.tl.track(tr["id"])["items"])
        ry = 30 + 26 + 36 + 35                      # ruler + sections + cues + half of the zoom lane
        page.mouse.dblclick(box["x"] + 128 + 24 * 12, box["y"] + ry)
        assert wait(lambda: len(ctl.tl.track(tr["id"])["items"]) == n0 + 1)
        page.click("#tlplay")
        assert wait(lambda: ctl.tl.playing)
        time.sleep(1.0)
        page.screenshot(path=str(SHOTS / "timeline_tab.png"))
        page.click("#tlplay")
        assert wait(lambda: not ctl.tl.playing)
        # SHOW menu: save as
        page.click("#shbtn")
        page.fill(".pfdw.on input[type=text]", "Browser show")
        page.click(".pfdw.on button:has-text('SAVE AS')")
        assert wait(lambda: ctl.shows.current == "browser-show")
        page.wait_for_selector(".pfdw.on .shr")
        page.screenshot(path=str(SHOTS / "show_menu.png"))
        page.keyboard.press("Escape")
        page.mouse.click(5, 870)
        # preset browser: save + recall
        page.click("#prbtn")
        dw = page.locator(".pfdw.on")
        dw.locator("input[placeholder='preset name']").fill("Tunnel blue")
        dw.locator("button:has-text('SAVE PRESET')").click()
        dw.locator(".pcard").first.wait_for()
        page.screenshot(path=str(SHOTS / "preset_browser.png"))
        with lock:
            ctl.set("hue", 0.9)
        dw.locator(".pcard").first.click()
        assert wait(lambda: abs(ctl.base["hue"] - 0.0) < 1e-6)
        br.close()
    assert not [e for e in errors if "favicon" not in e], errors
