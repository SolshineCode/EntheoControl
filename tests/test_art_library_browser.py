"""Panel in a real (headless) browser: combinable art mode chips + the one-library Sources picker."""
import sys
import time
from pathlib import Path

import pytest

pw_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_panel_browser import launch, served  # noqa: E402,F401  (fixture)
from PIL import Image, PngImagePlugin  # noqa: E402


def _imgs(tmp):
    (tmp / "art").mkdir(exist_ok=True)
    (tmp / "tex").mkdir(exist_ok=True)
    Image.new("RGB", (64, 48), (200, 80, 40)).save(tmp / "art" / "20260101-000000-000_bird_ab12.png")
    meta = PngImagePlugin.PngInfo()
    meta.add_text("prompt", "spirit jaguar")
    Image.new("RGB", (64, 48), (40, 200, 80)).save(tmp / "tex" / "gen_0001_a.png", pnginfo=meta)
    Image.new("RGB", (64, 48), (90, 40, 200)).save(tmp / "tex" / "gen_dream_0002_eyes.png")


def test_combine_chips_and_library_picker(served, tmp_path):
    url, ctl = served
    _imgs(tmp_path)
    errors = []
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        page = br.new_page(viewport=dict(width=1600, height=1000))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("dialog", lambda d: d.accept("Jaguar piece"))
        page.goto(url)
        page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
        page.click('#tabs button[data-tab="art"]')
        page.wait_for_timeout(300)
        card = page.locator("#pg-art .amodes").first
        # one mode: a chip = that mode only, amounts one-hot
        card.locator(".amode .tile", has_text="Kaleidoscope").click()
        page.wait_for_timeout(400)
        assert ctl.base["art_use"] == 2.0 and ctl.base["art_mix_kaleido"] == 1.0 and ctl.base["art_mix_stencil"] == 0.0
        assert card.locator(".amode .sl").first.is_hidden()                     # no amount faders in one-mode
        # COMBINE: chips toggle independently, each with an amount fader
        card.locator(".acomb").click()
        page.wait_for_timeout(300)
        card.locator(".amode .tile", has_text="Inside the lines").click()
        page.wait_for_timeout(400)
        assert ctl.base["art_combine"] == 1.0 and ctl.base["art_mix_kaleido"] == 1.0 and ctl.base["art_mix_stencil"] == 1.0
        assert card.locator(".amode .sl").first.is_visible()
        page.screenshot(path=str(ROOT / "test_frames" / "art_mix" / "panel_combine.png"), full_page=False) if (ROOT / "test_frames" / "art_mix").exists() else None
        # library picker: filters + action menu on every thumbnail
        picker = page.locator("#pg-art .card").filter(has=page.locator(".pick")).first
        page.wait_for_function("document.querySelectorAll('#pg-art .pick .art').length >= 3", timeout=8000)
        picker.locator(".srcf button", has_text="DREAMED").click()
        page.wait_for_timeout(200)
        assert picker.locator(".pick .art").count() == 1
        picker.locator(".srcf button", has_text="ALL").click()
        page.wait_for_timeout(200)
        th = picker.locator('.pick .art.tex[title="gen_0001_a.png"]')
        th.locator(".sm").click()
        menu = page.locator("#srcmenu")
        assert menu.is_visible()
        labels = menu.locator("button").all_inner_texts()
        for want in ("USE AS ART", "IN THE AI LAYER", "SKIN", "SPRITE CREATURE", "DELETE"):
            assert any(want in x for x in labels), (want, labels)
        menu.locator("button", has_text="USE AS ART").click()
        t0 = time.time()
        while time.time() - t0 < 6 and len(list((tmp_path / "art").glob("*.png"))) < 2:
            time.sleep(0.1)
        arts = sorted(p.name for p in (tmp_path / "art").glob("*.png"))
        assert len(arts) == 2 and any("spirit_jaguar" in a for a in arts)         # imported with its prompt as the name
        page.wait_for_timeout(600)
        assert ctl.base["art_source"] == 0.0
        br.close()
    assert not errors, errors
