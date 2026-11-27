"""MAPPING tab: the LAYERS ON THIS SURFACE card + AUTO MAP thin-parts row, in headless Edge/Chrome. Skipped without Playwright."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pw_api = pytest.importorskip("playwright.sync_api")
import surfaces as S  # noqa: E402
from test_panel_mapping import SHOTS, open_tab, served, until  # noqa: E402,F401


def test_panel_source_list_mirrors_the_model():
    text = (ROOT / "panel.html").read_text(encoding="utf-8")
    js = re.search(r"var SURF_SRC=\[(.*?)\];", text, re.S).group(1)
    assert re.findall(r"\['([a-z0-9]+)','", js) == list(S.SOURCE_IDS)


def test_layers_card_own_layers_add_edit_looks(served):
    url, mapper = served
    st = mapper.set
    st.apply(S.validate_cmd(dict(op="add")))
    st.apply(S.validate_cmd(dict(op="add")))
    with pw_api.sync_playwright() as pw:
        br, page, errors = open_tab(pw, url)
        page.wait_for_function("window.__map.state().surfaces.length === 3")
        page.locator(".sfrow").nth(0).click()
        assert until(lambda: st.sel == 1)
        page.wait_for_timeout(600)
        page.click('#mlay button:text-is("OWN LAYERS")')
        assert until(lambda: st.get(1)["content"] == "layers" and len(st.get(1)["layers"]) == 1)
        page.wait_for_timeout(900)
        page.click('#mlay button:text-is("+ LAYER")')
        assert until(lambda: len(st.get(1)["layers"]) == 2)
        page.wait_for_timeout(900)
        page.locator("#mlay .slrow select").nth(0).select_option("fractal")
        assert until(lambda: st.get(1)["layers"][0]["src"] == "fractal")
        page.wait_for_timeout(900)
        page.locator("#mlay .slrow select").nth(1).select_option("add")
        assert until(lambda: st.get(1)["layers"][0]["blend"] == "add")
        page.wait_for_timeout(900)
        page.locator("#mlay .slrow .ey").nth(1).click()
        assert until(lambda: st.get(1)["layers"][1]["on"] is False)
        page.wait_for_timeout(900)
        sl = page.locator("#mlay .slrow .sl").nth(0).bounding_box()          # opacity of layer 1
        page.mouse.click(sl["x"] + sl["width"] * 0.3, sl["y"] + sl["height"] / 2)
        assert until(lambda: 0.2 < st.get(1)["layers"][0]["opacity"] < 0.4)
        page.wait_for_timeout(900)
        page.click('#mlay button:text-is("SAME AS MAIN")')
        assert until(lambda: st.get(1)["content"] == "main" and len(st.get(1)["layers"]) == 2)
        page.wait_for_timeout(900)
        page.click("#mlooks")
        assert until(lambda: all(s["content"] == "layers" for s in st.surfaces))
        assert len({s["layers"][0]["src"] for s in st.surfaces}) == 3
        page.wait_for_timeout(900)
        assert page.locator("#amthin").count() == 1 and page.locator("#amthinc").is_disabled()
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.locator("#mlay").screenshot(path=str(SHOTS / "panel_surface_layers.png"))
        br.close()
    assert errors == []
