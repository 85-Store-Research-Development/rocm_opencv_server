"""新品のまとめ見せ（app/ad_lineup.py・app/ad_lineup_layout.py・layout: lineup）。
モデルと Chrome は使わない。商品・店名・色の名前はすべて架空の値。"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app import ad_compose as ac
from app import ad_lineup as lu
from app import ad_lineup_layout as ll

MS = ac.Measurer(None)
FMTS = ("4:5", "1:1", "9:16")


def png(tmp_path: Path, name: str, color, aspect=1.0, h=400) -> Path:
    """色つきの服のような切り抜き（四角い身頃と袖）を RGBA の PNG で書く。"""
    w = round(h * aspect)
    a = np.zeros((h, w), np.uint8)
    cv2.rectangle(a, (w // 6, 0), (w * 5 // 6, h), 255, -1)
    cv2.rectangle(a, (0, 0), (w, h // 3), 255, -1)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = a
    p = tmp_path / f"{name}.png"
    Image.fromarray(rgba).save(p)
    return p


def make_sprites(tmp_path, ids, aspect=1.0):
    out = []
    for i, sid in enumerate(ids):
        color = (200 - 30 * (i % 5), 90 + 25 * (i % 4), 60 + 40 * (i % 3))
        p = png(tmp_path, sid, color, aspect)
        a = np.asarray(Image.open(p))[..., 3]
        out.append(ac.Sprite(sid, "", p, np.ascontiguousarray(a)))
    return out


def raw_item(iid, color):
    return {"id": iid, "image": f"/nonexistent/{iid}.jpg", "item_type": "Sweaters", "color": color}


def lineup_raw(style="compare", **over):
    groups = [
        {
            "name": "型あ",
            "price_text": "¥10,000",
            "sizes": "S / M / L",
            "items": [raw_item("a1", "あか"), raw_item("a2", "あお")],
        },
        {
            "name": "型い",
            "price_text": "¥12,000",
            "sizes": "S / M / L",
            "items": [raw_item("b1", "みどり"), raw_item("b2", "くろ")],
        },
    ]
    raw = {
        "id": "lu1",
        "format": "4:5",
        "layout": "lineup",
        "store": {
            "name": "SAMPLE-SHOP",
            "tagline": "ここで会いましょう。",
            "area_line": "テスト町",
        },
        "headline": "同じ生地の\n2型です。",
        "aside": "新品です",
        "palette": "cream",
        "lineup": {
            "style": style,
            "title_note": "空気をふくむ生地",
            "groups": groups,
            "badges": ["新品", "ウール11%"],
        },
        "seed": 0,
        "candidates": 3,
    }
    raw.update(over)
    return raw


def build(style, tmp_path, fmt="4:5", seed=0, worn=None, raw=None):
    raw = raw or lineup_raw(style)
    brief = ac.parse_brief(raw)
    ids = [li.id for g in brief.lineup.groups for li in g.items]
    sprites = make_sprites(tmp_path, ids)
    lay = ll.build_lineup_layout(brief, sprites, fmt, seed, MS, worn)
    return brief, sprites, lay


# ------------------------------------------------------------ ブリーフ


def test_parse_lineup_groups_items_and_labels():
    b = ac.parse_brief(lineup_raw("compare"))
    assert b.layout == "lineup" and b.lineup.style == "compare"
    assert [g.name for g in b.lineup.groups] == ["型あ", "型い"]
    assert [i.id for i in b.items] == ["a1", "a2", "b1", "b2"]  # 型の順に並べた商品
    assert b.items[0].label  # 商品の文字（collage に落とすときに使う）は空でない
    assert b.lineup.badges == ["新品", "ウール11%"]


def test_parse_lineup_problems_are_collected():
    raw = lineup_raw("compare")
    raw["lineup"]["groups"] = raw["lineup"]["groups"][:1]  # compare は 2〜3 型
    with pytest.raises(ac.BriefError) as e:
        ac.parse_brief(raw)
    assert any("2〜3" in p for p in e.value.problems)
    raw = lineup_raw("compare")
    raw["lineup"]["groups"][0]["name"] = "あ" * 40
    raw["lineup"]["groups"][1]["items"].append(raw_item("a1", "かさなり"))
    with pytest.raises(ac.BriefError) as e:
        ac.parse_brief(raw)
    assert any("20" in p for p in e.value.problems) and any("重複" in p for p in e.value.problems)
    with pytest.raises(ac.BriefError):
        ac.parse_brief(lineup_raw("brand"))  # brand は題字が要る
    raw = lineup_raw("compare")
    raw["lineup"] = "x"
    with pytest.raises(ac.BriefError):
        ac.parse_brief(raw)


def test_parse_lineup_caps_badges_and_group_items():
    raw = lineup_raw("catalog")
    raw["lineup"]["badges"] = ["a", "b", "c", "d"]
    raw["lineup"]["groups"][0]["items"] = [raw_item(f"x{i}", f"色{i}") for i in range(7)]
    b = ac.parse_brief(raw)
    assert len(b.lineup.badges) == lu.BADGE_MAX
    assert len(b.lineup.groups[0].items) == lu.MAX_ITEMS_PER_GROUP
    assert len(b.lineup.notes) >= 3  # 省いたことを残す
    assert {i.id for i in b.items} == {li.id for g in b.lineup.groups for li in g.items}


def test_unknown_style_falls_back_to_collage():
    raw = lineup_raw("zigzag")
    b = ac.parse_brief(raw)
    assert b.layout == "collage" and b.lineup is None
    assert 2 <= len(b.items) <= ac.MAX_ITEMS
    assert all(i.label for i in b.items)


def test_brand_uses_first_item_of_each_group_and_collage_labels():
    raw = lineup_raw("brand")
    raw["lineup"]["brand"] = "BRAND"
    b = ac.parse_brief(raw)
    assert [i.id for i in b.items] == ["a1", "b1"]
    assert "型あ" in b.items[0].label and "¥10,000" in b.items[0].label


def test_layout_lineup_does_not_change_collage_or_single_briefs():
    raw = lineup_raw("compare")
    raw.pop("layout")
    raw["items"] = [raw_item(f"c{i}", "x") | {"label": f"L{i}"} for i in range(3)]
    b = ac.parse_brief(raw)
    assert b.layout == "collage" and b.lineup is None and len(b.items) == 3


def test_dominant_color_is_the_garment_color(tmp_path):
    p = png(tmp_path, "c", (30, 120, 200))
    assert lu.dominant_color(p) == "#1E78C8"


# ------------------------------------------------------------ 置き方と検査


@pytest.mark.parametrize("style", ["compare", "colors", "catalog"])
def test_each_style_places_every_group_and_passes_the_checks(style, tmp_path):
    raw = lineup_raw(style)
    if style == "colors":
        raw["lineup"]["groups"] = [
            {
                "name": "型あ",
                "price_text": "¥7,150",
                "sizes": "S / M / L",
                "items": [raw_item(f"k{i}", c) for i, c in enumerate(("あか", "あお", "きいろ"))],
            }
        ]
    brief, _, lay = build(style, tmp_path, raw=raw)
    c = ac.check_layout(lay, MS)
    assert lay.layout == "lineup" and lay.lineup.style == style
    assert len(lay.lineup.groups_placed) == len(brief.lineup.groups), c.problems
    assert c.verdict == "pass", c.problems
    assert c.lineup["groups_placed"] == c.lineup["groups_wanted"]
    names = [t for t in lay.lineup.texts if t.kind == "name"]
    assert names and all(t.size >= lu.NAME_MIN for t in names)
    colors = [t for t in lay.lineup.texts if t.kind == "color"]
    assert len(colors) == len(lay.items) and all(t.size >= lu.COLOR_MIN for t in colors)


def test_compare_gives_each_group_a_price_and_fans_colors(tmp_path):
    brief, _, lay = build("compare", tmp_path)
    prices = {b.text for b in lay.lineup.badges} | {
        t.text for t in lay.lineup.texts if t.kind == "price_pill"
    }
    assert prices == {"¥10,000", "¥12,000"}
    by_group = {}
    for it in lay.items:
        by_group.setdefault(lay.lineup.group_of[it.sprite.id], []).append(it)
    for its in by_group.values():  # 同じ型の色ちがいは重なる（扇）が、ずれて見える
        a, b = its
        assert abs(a.cx - b.cx) < 0.9 * a.w and abs(a.cx - b.cx) > 0.3 * a.w
    # 事実の札（新品）が入っている
    assert {"新品", "ウール11%"} <= {t.text for t in lay.lineup.texts if t.kind == "badge_pill"}


def test_all_formats_pass_for_compare(tmp_path):
    for fmt in FMTS:
        _, _, lay = build("compare", tmp_path, fmt=fmt)
        c = ac.check_layout(lay, MS)
        assert (lay.w, lay.h) == ac.FORMATS[fmt]
        assert c.verdict == "pass", (fmt, c.problems)


def test_layout_is_deterministic_per_seed_and_varies_with_seed(tmp_path):
    _, sp, a = build("compare", tmp_path, seed=3)
    _, _, b = build("compare", tmp_path, seed=3)
    assert a.lineup.to_json() == b.lineup.to_json()
    assert [(i.cx, i.cy, i.angle) for i in a.items] == [(i.cx, i.cy, i.angle) for i in b.items]
    _, _, c = build("compare", tmp_path, seed=4)
    assert [(i.cx, i.angle) for i in a.items] != [(i.cx, i.angle) for i in c.items]


def test_no_text_over_items_and_texts_inside_safe_area(tmp_path):
    _, _, lay = build("catalog", tmp_path)
    geo = ac.geometry("4:5")
    sx0, sy0, sx1, sy1 = geo.safe_rect
    size = (lay.w, lay.h)
    union = np.zeros((lay.h, lay.w), bool)
    for it in lay.items:
        union |= it.mask(size)
    for t in lay.lineup.texts:
        p = t.polygon()
        assert p[:, 0].min() >= sx0 - 8 and p[:, 0].max() <= sx1 + 8, t.text
        assert np.count_nonzero(ac.polygon_mask(t.polygon(), size) & union) <= lu.TOLERANCE_PX


def test_check_flags_small_names_and_missing_groups(tmp_path):
    brief, _, lay = build("compare", tmp_path)
    name = next(t for t in lay.lineup.texts if t.kind == "name")
    name.size = 30
    c = ac.check_layout(lay, MS)
    assert c.verdict == "fail" and any("型の名前が小さすぎる" in p for p in c.problems)
    _, _, lay2 = build("compare", tmp_path)
    lay2.lineup.groups_placed = lay2.lineup.groups_placed[:1]
    c2 = ac.check_layout(lay2, MS)
    assert any("載っていない" in p for p in c2.problems)


def test_same_group_may_overlap_but_different_groups_may_not(tmp_path):
    _, _, lay = build("compare", tmp_path)
    a, b = lay.items[0], lay.items[2]  # 別の型
    b.cx, b.cy = a.cx, a.cy  # 別の型を同じ場所へ
    c = ac.check_layout(lay, MS)
    assert any("商品どうしが重なりすぎ" in p for p in c.problems)


def test_worn_goes_to_the_side_and_keeps_groups(tmp_path):
    worn_png = png(tmp_path, "w", (80, 80, 80), aspect=0.5, h=700)
    alpha = np.asarray(Image.open(worn_png))[..., 3]
    worn = ac.Sprite(
        "a1#worn", "着るとこんな感じ", worn_png, np.ascontiguousarray(alpha), worn=True
    )
    brief, _, lay = build("compare", tmp_path, worn=worn)
    assert lay.worn_item_id == "a1" and lay.items[-1].sprite.worn
    others = [it for it in lay.items if not it.sprite.worn]
    hx, _ = lay.items[-1].half_extent()
    for it in others:  # 着画の幅の分だけ、商品は横にずれて置く
        assert abs(it.cx - lay.items[-1].cx) > hx * 0.8
    assert len(lay.lineup.groups_placed) == 2


def test_reduce_lineup_items_keeps_the_last_item_of_each_group(tmp_path):
    brief = ac.parse_brief(lineup_raw("compare"))
    sprites = make_sprites(tmp_path, ["a1", "a2", "b1", "b2"])
    group_of = {li.id: gi for gi, g in enumerate(brief.lineup.groups) for li in g.items}
    keep, removed = ac.reduce_lineup_items(sprites, 2, None, group_of)
    assert [s.id for s in keep] == ["a1", "b1"] and [s.id for s in removed] == ["a2", "b2"]
    keep, removed = ac.reduce_lineup_items(sprites, 3, "a2", group_of)
    assert [s.id for s in keep] == ["a1", "a2", "b1"]


def test_a_missing_sprite_drops_that_color_but_not_the_group(tmp_path):
    brief = ac.parse_brief(lineup_raw("compare"))
    sprites = make_sprites(tmp_path, ["a1", "b1", "b2"])  # a2 は切り抜けなかった
    lay = ll.build_lineup_layout(brief, sprites, "4:5", 0, MS)
    assert len(lay.lineup.groups_placed) == 2 and len(lay.items) == 3


# ------------------------------------------------------------ 描画・meta・ブランド特集


def test_render_html_has_names_prices_swatches_and_badges(tmp_path):
    _, _, lay = build("compare", tmp_path)
    html = ac.render_html(lay, Path("/nonexistent-fonts"))
    for word in ("型あ", "型い", "あか", "くろ", "S / M / L", "空気をふくむ生地", "新品"):
        assert word in html
    assert "¥10,000" in html and 'class="sw"' in html and "lt pill" in html
    assert "一点もの" not in html  # 新品は一点ものと言わない


def test_layout_json_and_meta(tmp_path):
    brief, _, lay = build("compare", tmp_path)
    j = lay.to_json()
    assert j["layout"] == "lineup" and j["lineup"]["style"] == "compare"
    json.dumps(j, ensure_ascii=False, default=float)
    meta = ac.lineup_meta(lay)
    assert meta["lineup_style"] == "compare" and meta["groups_placed"] == 2
    assert ac.lineup_meta(replace(lay, lineup=None))["lineup_style"] == "brand"


def test_brand_adds_a_masthead_and_lowers_the_region(tmp_path):
    raw = lineup_raw("brand")
    raw["lineup"]["brand"] = "BRAND"
    brief = ac.parse_brief(raw)
    plain = ac.parse_brief(lineup_raw("brand") | {"lineup": {**raw["lineup"], "brand": "B"}})
    fixed = ac.layout_fixed(brief, "4:5", MS)
    mast = next(b for b in fixed.blocks if b.name == "masthead")
    assert mast.lines == ["BRAND"] and mast.size >= lu.MASTHEAD_MIN
    base = ac.layout_fixed(replace(plain, layout="collage", lineup=None), "4:5", MS)
    assert fixed.region[1] > base.region[1] + mast.h * 0.9
    sprites = make_sprites(tmp_path, ["a1", "b1"])
    lay = ac.build_layout(brief, sprites, "4:5", 0, MS)
    lay.layout = lu.LAYOUT_NAME
    html = ac.render_html(lay, Path("/nonexistent-fonts"))
    assert "masthead" in html and "BRAND" in html
    assert ac.check_layout(lay, MS).problems is not None


def test_collage_and_single_layouts_are_untouched_by_lineup_code(tmp_path):
    raw = copy.deepcopy(lineup_raw("compare"))
    raw.pop("lineup")
    raw["layout"] = "collage"
    raw["items"] = [raw_item(f"c{i}", "x") | {"label": f"L{i}"} for i in range(3)]
    b = ac.parse_brief(raw)
    assert b.layout == "collage" and b.lineup is None
    sprites = make_sprites(tmp_path, ["c0", "c1", "c2"])
    lay = ac.build_layout(b, sprites, "4:5", 0, MS)
    assert lay.lineup is None and lay.to_json()["lineup"] is None
    assert ac.check_layout(lay, MS).lineup is None
