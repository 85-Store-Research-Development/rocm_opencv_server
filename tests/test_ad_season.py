"""広告画像の季節・色・沈み対策（app/ad_season.py・app/ad_palette.py・app/ad_compose.py）。
モデルと Chrome は使わない。値は架空。"""

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import ad_compose as ac
from app import ad_palette as ap
from app import ad_season as se


def raw(n=3, **over):
    r = {
        "id": "s1",
        "format": "4:5",
        "store": {"name": "SAMPLE", "tagline": "ここで会いましょう。", "area_line": "テスト町"},
        "headline": "あたらしい服、\nとどいてます。",
        "aside": "一点もの",
        "palette": "navy",
        "items": [
            {"id": f"i{k}", "image": f"/x/{k}.jpg", "label": f"ブランド{k} の シャツ"}
            for k in range(n)
        ],
        "candidates": 2,
    }
    r.update(over)
    return r


SEASON = {
    "motif": "late_autumn",
    "label": "晩秋",
    "stamp": "NOVEMBER / 晩秋",
    "weather_line": "朝晩は5℃前後。",
    "stage": "coat",
    "tmax": 12,
    "tmin": 5,
    "trend": "cooling",
    "estimated": False,
}


def sprite(k, light=None, hint=None):
    a = np.zeros((380, 280), np.uint8)
    cv2.ellipse(a, (140, 190), (130, 180), 0, 0, 360, 255, -1)
    hist = None
    if light is not None:
        rgb = np.full((380, 280, 3), round(light * 2.55), np.uint8)
        hist = ap.tone_hist(rgb, a)
    return ac.Sprite(f"s{k}", f"品{k} ¥{k}000", Path(f"/x/{k}.png"), a, None, hist, hint)


# ------------------------------------------------------------ season の読み取り


def test_parse_season_full():
    s = se.parse_season(SEASON)
    assert s.motif == "late_autumn" and s.stamp == "NOVEMBER / 晩秋"
    assert s.tmax == 12.0 and s.estimated is False and s.weather_line == "朝晩は5℃前後。"


@pytest.mark.parametrize(
    "value", [None, "autumn", [], {}, {"motif": "monsoon"}, {"motif": 3}, {"label": "秋"}]
)
def test_parse_season_none_when_unusable(value):
    assert se.parse_season(value) is None


def test_parse_season_tolerates_bad_fields():
    s = se.parse_season({"motif": "winter", "stamp": "x" * 80, "tmax": "hot", "label": 3})
    assert s.motif == "winter" and s.stamp == "" and s.tmax is None and s.label == ""


def test_brief_without_season_is_unchanged():
    b = ac.parse_brief(raw())
    assert b.season is None and b.palette_options == [] and b.palette_requested == "navy"
    assert all(it.tone is None for it in b.items)


def test_brief_with_season_options_and_tone():
    r = raw(season=SEASON, palette_options=["olive", "pink", "olive", "cream", 3])
    r["items"][0]["tone"] = "dark"
    r["items"][1]["tone"] = "purple"
    b = ac.parse_brief(r)
    assert b.season.motif == "late_autumn"
    assert b.palette_options == ["olive", "cream"]
    assert b.items[0].tone == "dark" and b.items[1].tone is None


def test_brief_with_unknown_motif_has_no_season():
    assert ac.parse_brief(raw(season={"motif": "x", "stamp": "A / B"})).season is None


# ------------------------------------------------------------ パレット


def test_ten_palettes_with_shared_names():
    assert set(ap.PALETTE_NAMES) == {
        "navy",
        "brown",
        "burgundy",
        "green",
        "mustard",
        "olive",
        "charcoal",
        "cream",
        "sky",
        "terracotta",
    }
    assert all(n in ap.PALETTES for n in ap.PALETTE_NAMES)


@pytest.mark.parametrize("name", list(ap.PALETTES))
def test_palette_contrast_and_light_ground(name):
    p = ap.PALETTES[name]
    for fg in (p.text, p.accent, p.doodle):
        assert ap.contrast_ratio(fg, p.bg) >= ap.MIN_CONTRAST
    if name in ("cream", "sky", "paper"):
        assert p.light and ap.luminance(p.text) < 0.1  # 明るい地は文字と線が濃い
    else:
        assert not p.light and ap.luminance(p.text) > 0.5


def test_light_ground_shadow_is_not_black_heavy():
    assert ap.PALETTES["cream"].shadow != ap.PALETTES["navy"].shadow


# ------------------------------------------------------------ 明るさの計測


def test_tone_hist_black_white_and_alpha_weight():
    alpha = np.full((40, 40), 255, np.uint8)
    black = ap.tone_hist(np.zeros((40, 40, 3), np.uint8), alpha)
    white = ap.tone_hist(np.full((40, 40, 3), 255, np.uint8), alpha)
    assert ap.hist_median(black) <= 1 and ap.hist_median(white) >= 99
    # 左半分が白・右半分が黒でも、白を透明にすれば黒だけを測る
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:, :20] = 255
    a = alpha.copy()
    a[:, :20] = 0
    assert ap.hist_median(ap.tone_hist(rgb, a)) <= 1
    assert ap.hist_median(ap.tone_hist(rgb, np.zeros_like(a))) is None


def test_item_lightness_prefers_pixels_then_hint():
    hist = ap.tone_hist(np.full((8, 8, 3), 255, np.uint8), np.full((8, 8), 255, np.uint8))
    assert ap.item_lightness(hist, "dark") > 90
    assert ap.item_lightness(None, "dark") == ap.TONE_L["dark"]
    assert ap.item_lightness(None, None) is None
    assert sprite(0, light=10).light < 20 and sprite(0, hint="light").light > 80


# ------------------------------------------------------------ 沈む・色の差し替え


def test_sinks_by_lightness_gap():
    assert ap.sinks(12, ap.PALETTES["brown"].bg)  # 黒い服 × 濃い地
    assert not ap.sinks(12, ap.PALETTES["cream"].bg)
    assert ap.sinks(90, ap.PALETTES["cream"].bg)  # 白い服 × 明るい地
    assert not ap.sinks(None, ap.PALETTES["cream"].bg)


def test_choose_palette_keeps_when_nothing_sinks():
    assert ap.choose_palette("navy", [90, 85, 80]) == ("navy", None)
    assert ap.choose_palette("navy", [None, None]) == ("navy", None)


def test_choose_palette_swaps_within_options():
    used, reason = ap.choose_palette("brown", [10, 14, 8], ["brown", "cream", "sky"])
    assert used in ("cream", "sky") and "brown" in reason and used in reason


def test_choose_palette_default_family_stays_dark_or_light():
    used, _ = ap.choose_palette("brown", [10, 14])
    assert used != "brown" and not ap.PALETTES[used].light
    # 明るい地の系統には白い服を沈ませない色が無い → 替えない（縁取りで救う）
    assert ap.choose_palette("cream", [92, 95]) == ("cream", None)


def test_choose_palette_mixed_keeps_when_no_option_is_better():
    # 暗い服と明るい服が混ざり、どの候補でも沈む数が減らない → 替えない（縁取りで救う）
    used, reason = ap.choose_palette("navy", [15, 92], ["navy", "cream"])
    assert (used, reason) == ("navy", None)


def test_choose_palette_ignores_unknown_options():
    assert ap.choose_palette("navy", [10], ["pink"])[0] != "pink"


def test_glow_is_lighter_on_dark_and_darker_on_light():
    nav = ap.PALETTES["navy"]
    assert ap.lstar(ap.glow_color(nav)) > ap.lstar(nav.bg)
    assert ap.lstar(ap.glow_color(nav)) - ap.lstar(nav.bg) < 45  # 「わずかに」
    cr = ap.PALETTES["cream"]
    assert ap.lstar(ap.glow_color(cr)) < ap.lstar(cr.bg)


def test_contrast_score_penalises_sunk_and_credits_glow():
    bg = ap.PALETTES["brown"].bg
    good = ap.contrast_score([90, 85], bg)
    bad = ap.contrast_score([12, 15], bg)
    assert good > bad
    assert ap.contrast_score([12, 15], bg, [True, True]) > bad
    assert ap.contrast_score([None], bg) is None


def test_compose_format_swap_is_recorded(tmp_path, monkeypatch):
    b = ac.parse_brief(raw(palette="brown", palette_options=["brown", "cream"]))
    sp = [sprite(k, light=10) for k in range(3)]
    used, reason = ap.choose_palette(b.palette, [s.light for s in sp], b.palette_options)
    assert used == "cream"
    from dataclasses import replace

    b2 = replace(b, palette=used, palette_requested="brown", palette_changed_reason=reason)
    lay = ac.build_layout(b2, sp, "4:5", 1, ac.Measurer(None))
    js = lay.to_json()
    assert js["palette_used"] == "cream" and js["palette_requested"] == "brown"
    assert js["palette_changed_reason"] == reason
    assert not any(it.glow for it in lay.items)  # 色を替えて沈みが消えた


def test_glow_set_on_remaining_sunk_items_and_score_has_contrast():
    b = ac.parse_brief(raw(palette="navy"))
    sp = [sprite(0, light=90), sprite(1, light=90), sprite(2, light=15)]
    lay = ac.build_layout(b, sp, "4:5", 3, ac.Measurer(None))
    glows = {it.sprite.id: it.glow for it in lay.items}
    assert glows == {"s0": False, "s1": False, "s2": True}
    c = ac.check_layout(lay)
    assert "contrast" in c.score_parts
    # 画素もトーンも無いスプライトでは点数に混ぜない
    plain = [ac.Sprite(f"p{k}", "x", Path("/x"), sprite(k).alpha) for k in range(3)]
    lay2 = ac.build_layout(b, plain, "4:5", 3, ac.Measurer(None))
    assert "contrast" not in ac.check_layout(lay2).score_parts


# ------------------------------------------------------------ 落書き・スタンプ


def test_every_motif_has_six_or_more_known_doodles():
    assert set(se.MOTIF_DOODLES) == set(se.MOTIFS)
    for motif, kinds in se.MOTIF_DOODLES.items():
        assert len(set(kinds)) >= 6, motif
        for k in kinds:
            sh = se.SHAPES[k]
            assert sh.layers and sh.w > 0 and sh.h > 0 and all(m in "sa" for _, m in sh.layers)


def test_no_season_doodles_use_generic_kinds_only():
    b = ac.parse_brief(raw())
    lay = ac.build_layout(b, [sprite(k) for k in range(3)], "4:5", 1, ac.Measurer(None))
    assert lay.doodles and all(d.kind in ac.DOODLES for d in lay.doodles)
    assert not any(bl.name == "stamp" for bl in lay.blocks)


def test_season_doodles_are_mostly_seasonal_and_varied():
    seasonal = total = 0
    sizes, angles = set(), set()
    for seed in range(6):
        rng = random.Random(seed)
        geo = ac.geometry("4:5")
        occ = np.zeros((geo.h, geo.w), bool)
        ds = ac.place_doodles(rng, geo, occ, (30.0, 420.0, 1050.0, 1180.0), se.parse_season(SEASON))
        total += len(ds)
        seasonal += sum(1 for d in ds if d.kind in se.MOTIF_DOODLES["late_autumn"])
        assert sum(1 for d in ds if d.kind in ac.DOODLES) <= ac.SEASON_GENERIC_MAX
        sizes |= {round(d.size) for d in ds}
        angles |= {round(d.angle) for d in ds}
    assert seasonal / total > 0.6 and len(sizes) > 8 and len(angles) > 8


@pytest.mark.parametrize("fmt", ["4:5", "1:1", "9:16"])
def test_stamp_and_doodles_do_not_overlap(fmt):
    b = ac.parse_brief(raw(season=SEASON))
    sp = [sprite(k, light=50) for k in range(3)]
    lay = ac.build_layout(b, sp, fmt, 2, ac.Measurer(None))
    stamp = next(bl for bl in lay.blocks if bl.name == "stamp")
    size = (lay.w, lay.h)
    sm = stamp.mask(size)
    others = [bl.mask(size) for bl in lay.blocks if bl is not stamp]
    assert not any(np.count_nonzero(sm & o) for o in others)
    for d in lay.doodles:
        assert not np.count_nonzero(ac.polygon_mask(d.polygon(), size) & sm)
    assert ac.check_layout(lay).doodle_overlap_px <= ac.TOLERANCE_PX
    geo = ac.geometry(fmt)
    poly = stamp.polygon(pad=0)
    assert poly[:, 1].min() >= geo.safe_top - 8 and poly[:, 0].max() <= geo.w - geo.mx


def test_stamp_absent_without_text_and_in_html():
    s = dict(SEASON, stamp="")
    b = ac.parse_brief(raw(season=s))
    lay = ac.build_layout(b, [sprite(k) for k in range(3)], "4:5", 1, ac.Measurer(None))
    assert not any(bl.name == "stamp" for bl in lay.blocks)


def test_render_html_has_stamp_glow_and_accent_layers(tmp_path):
    b = ac.parse_brief(raw(season=SEASON, palette="navy"))
    sp = [sprite(0, light=10), sprite(1, light=90), sprite(2, light=90)]
    lay = ac.build_layout(b, sp, "4:5", 1, ac.Measurer(None))
    doc = ac.render_html(lay, tmp_path)
    assert "NOVEMBER / 晩秋" in doc and 'class="stamp"' in doc
    assert doc.count("drop-shadow(0 0 ") == len(ac._GLOW_BLUR)  # 沈む1点だけ
    assert "weather" not in doc and "朝晩" not in doc  # weather_line は画像に入れない
