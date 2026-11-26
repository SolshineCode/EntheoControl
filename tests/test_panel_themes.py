"""THEMES tray / tab, camera buttons and SOUND TAGS meters in a headless browser (panel.html block [themes> ... <themes]).
The first test stubs state.themes + GET /themes (works with or without the backend); the second runs against the real
controller once themes.py has landed. Screenshots: test_frames/themes/*.png."""
import json
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

SHOTS = Path(__file__).resolve().parents[1] / "test_frames" / "themes"

TS = {"rev": 3, "n": 2, "armed": True, "test": False, "audio": True, "rec": None,
      "m": [["t1", 0.82, 0.7, 2], ["t2", 0.31, 0.6, 0]], "block": None, "susp": 0, "last": [12, "t1", "Beings 21:14 Jewel", 0.84, True],
      "pend": None}
DOC = {"rev": 3, "tray": [
    {"id": "t1", "slot": 1, "name": "Beings 21:14 Jewel", "scope": {"kind": "layer", "key": "l_beings"}, "label": "Layer: Beings",
     "star": True, "ref": "theme/beings-21-14-jewel", "t": 1764000000, "thumb": 0, "pal": 0,
     "tag": {"n": 2, "secs": 20, "mode": "vibe", "key_inv": False, "sens": 0.5, "hold": 3, "cooldown": 30, "quant": "bar",
             "morph": None, "prio": 0, "on": True, "only_auto": False}},
    {"id": "t2", "slot": 2, "name": "Look 21:20 Twilight", "scope": {"kind": "look", "key": None}, "label": "Look",
     "star": False, "ref": None, "t": 1764000300, "thumb": 0, "pal": 2, "tag": None}],
    "log": [[1764000400, "fire", "t1", "Beings 21:14 Jewel", 0.84, "bar"]],
    "suggest": [{"sid": "s1", "label": "Section A (3x)", "secs": 12, "n": 3, "at": [10, 200, 400]}],
    "scopes": {"layers": [["l_beings", "Beings"], ["l_kene", "Folding Blocks"]], "groups": [["color", "COLOR"], ["shape", "SHAPE"]]},
    "opts": {"override": 20, "gap": 8}}


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


def open_page(br, url, errors):
    page = br.new_page(viewport=dict(width=1400, height=880))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "theme-thumb" not in m.text
            and "404" not in m.text else None)
    return page


def test_themes_ui_with_stubbed_backend(served):
    url, ctl, lock = served
    SHOTS.mkdir(parents=True, exist_ok=True)
    sent = []
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        errors = []
        page = open_page(br, url, errors)

        def on_state(route):
            r = route.fetch()
            d = r.json()
            d["themes"] = TS
            route.fulfill(response=r, body=json.dumps(d), headers=dict(r.headers, **{"content-type": "application/json"}))

        def on_cmd(route):
            try:
                body = json.loads(route.request.post_data or "null")
            except ValueError:
                body = None
            for c in (body if isinstance(body, list) else body.get("cmds", [body]) if isinstance(body, dict) else []):
                if isinstance(c, dict) and c.get("cmd") == "theme":
                    sent.append(c)
            route.continue_()

        page.route("**/state*", on_state)
        page.route("**/themes*", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(DOC)))
        page.route("**/cmd*", on_cmd)
        page.goto(url)
        page.wait_for_function("document.body.classList.contains('thon') && document.querySelectorAll('.thq .thc').length >= 2"
                               " && getComputedStyle(document.querySelector('.thq .thc')).display !== 'none'", timeout=8000)
        assert page.evaluate("!!window.__themes && window.__themes.doc().tray.length === 2")
        # PERFORM chip: MIDI-learn target + tag dot + meter
        assert page.evaluate("document.querySelector('.thq .thc[data-id=t1]').dataset.mt") == "theme:1"
        assert page.evaluate("document.querySelector('.thq .thc[data-id=t1]').classList.contains('tag')")
        page.screenshot(path=str(SHOTS / "perform_tray.png"))
        # tap = restore, hold = cut
        chip = page.locator(".thq .thc[data-id=t2]")
        chip.click()
        box = chip.bounding_box()
        page.mouse.move(box["x"] + 50, box["y"] + 30)
        page.mouse.down()
        page.wait_for_timeout(650)
        page.mouse.up()
        # camera on a layer row (PERFORM strip or LAYERS card) + CAPTURE LOOK
        page.evaluate("window.__panel.setTab('layers')")
        page.wait_for_selector("#pg-layers .lrow .thcam", state="visible", timeout=5000)
        page.locator("#pg-layers .lrow .thcam").first.click()
        page.evaluate("window.__panel.setTab('color')")
        page.locator("#pg-color .pwbar .thpg").first.click()
        # THEMES tab
        page.click("#tabs button[data-tab=themes]")
        page.wait_for_selector("#pg-themes.on")
        assert page.evaluate("document.querySelectorAll('#pg-themes .thgrid .thc').length") == 2
        assert "ARMED" in page.inner_text("#pg-themes .tharm")
        page.click("#pg-themes .tharm")                      # disarm
        page.locator("#pg-themes button", has_text="CAPTURE LOOK").click()
        page.locator("#pg-themes .thdet button", has_text="TAG").first.click()
        page.locator("#pg-themes button", has_text="LEARN / TEST").click()
        page.locator("#pg-themes button", has_text="SUGGEST TAGS").click()
        page.wait_for_timeout(1200)
        page.screenshot(path=str(SHOTS / "themes_tab.png"), full_page=True)
        # another tab hides THEMES again
        page.evaluate("window.__panel.setTab('perform')")
        page.wait_for_timeout(150)
        assert not page.evaluate("document.getElementById('pg-themes').classList.contains('on')")
        # "..." menu on PERFORM
        page.locator(".thq .thc[data-id=t1] .mn").click()
        page.wait_for_selector(".pfdw.on .thdet")
        page.screenshot(path=str(SHOTS / "theme_menu.png"))
        br.close()
    ops = [c.get("op") for c in sent]
    assert "restore" in ops and any(c.get("op") == "restore" and c.get("now") for c in sent), sent
    scopes = [c.get("scope") for c in sent if c.get("op") == "capture"]
    assert "look" in scopes and any(s and s.startswith("layer:l_") for s in scopes) and "group:color" in scopes, scopes
    assert {"arm", "tag", "test", "suggest"} <= set(ops), ops
    if "theme" not in control.CONTROL_CMDS:                  # no backend yet: the server refuses the commands (400)
        errors = [e for e in errors if "status of 400" not in e]
    assert not errors, errors


def test_themes_ui_hidden_without_backend_state(served):
    url, ctl, lock = served
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        errors = []
        page = open_page(br, url, errors)

        def strip(route):
            r = route.fetch()
            d = r.json()
            d.pop("themes", None)
            route.fulfill(response=r, body=json.dumps(d))

        page.route("**/state*", strip)
        page.goto(url)
        page.wait_for_selector("#shbtn")
        page.wait_for_timeout(800)
        assert not page.evaluate("document.body.classList.contains('thon')")
        assert page.evaluate("getComputedStyle(document.querySelector('#tabs button[data-tab=themes]')).display") == "none"
        br.close()
    assert not errors, errors


@pytest.mark.skipif("theme" not in control.CONTROL_CMDS, reason="themes backend not landed yet")
def test_themes_capture_round_trip_real_backend(served):
    url, ctl, lock = served
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        errors = []
        page = open_page(br, url, errors)
        page.goto(url)
        page.wait_for_function("document.body.classList.contains('thon')", timeout=8000)
        page.locator(".thq button", has_text="LOOK").click()
        page.wait_for_function("document.querySelectorAll('.thq .thc[data-id]').length >= 1 &&"
                               " [].some.call(document.querySelectorAll('.thq .thc[data-id]'), e => e.style.display !== 'none')",
                               timeout=8000)
        page.locator(".thq .thc[data-id]").first.click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOTS / "real_backend.png"))
        br.close()
    assert not errors, errors
