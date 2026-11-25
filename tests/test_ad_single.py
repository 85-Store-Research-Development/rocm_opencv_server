"""1点ものの広告（app/ad_single.py・layout: single）。モデルと Chrome は使わない。値は架空。"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import ad_compose as ac
from app import ad_single as sg

W, H = 1080, 1350
SAFE = (65.0, 47.0, 1015.0, 1303.0)


def single_raw(callouts=None, **over):
    raw = {
        "id": "one",
        "format": "4:5",
        "layout": "single",
        "store": {
            "name": "SAMPLE-SHOP",
            "tagline": "ここで会いましょう。",
            "area_line": "テスト町",
        },
        "headline": "SAMPLE の\nネルシャツ。",
        "aside": "古着です",
        "palette": "navy",
        "items": [
            {
                "id": "i1",
                "image": "/nonexistent/1.jpg",
                "label": "ブランド の シャツ ¥9,800",
                "item_type": "Shirts",
            }
        ],
        "single": {
            "price_text": "¥9,800",
            "stamp": "USA製",
            "callouts": callouts
            if callouts is not None
            else [
                {"kind": "brand", "text": "SAMPLE"},
                {"kind": "era", "text": "90年代"},
                {"kind": "made_in", "text": "USA製"},
                {"kind": "fabric", "text": "ウール100%"},
                {"kind": "size", "text": "着丈72 身幅60"},
                {"kind": "condition", "text": "状態 A"},
            ],
        },
        "seed": 0,
        "candidates": 4,
    }
    raw.update(over)
    return raw


def checker(w, h):
    a = np.indices((h, w)).sum(axis=0) // 8 % 2
    return Image.fromarray(np.where(a[..., None] == 1, 20, 220).astype(np.uint8).repeat(3, axis=2))


def hero_sprite(w=420, h=520):
    a = np.zeros((h, w), np.uint8)
    cv2.rectangle(a, (w // 5, 0), (w * 4 // 5, h), 255, -1)  # 身頃
    cv2.rectangle(a, (0, 0), (w, h // 3), 255, -1)  # 肩・袖
    return ac.Sprite("i1", "", Path("/x/hero.png"), a)


def hero_mask(cx=540, cy=760, h=620):
    sp = hero_sprite()
    return ac.item_mask(sp.alpha, cx, cy, h, 3.0, (W, H))


def top_blocked():
    b = np.zeros((H, W), bool)
    b[:300, :] = True  # 見出しなど
    b[1230:, :] = True  # コピー
    return b


MS = ac.Measurer(None)


# ------------------------------------------------------------ ブリーフ


def test_parse_single_kinds_notes_and_problems():
    raw = {
        "price_text": "¥9,800",
        "stamp": "USA製",
        "callouts": [{"kind": "brand", "text": "A"}, {"kind": "???", "text": "B"}]
        + [{"kind": "size", "text": f"S{i}"} for i in range(6)]
        + [{"kind": "size", "text": ""}],
    }
    s, problems = sg.parse_single(raw)
    assert not problems
    assert [c.kind for c in s.callouts][:2] == ["brand", "other"]
    assert len(s.callouts) == sg.MAX_CALLOUTS and s.notes  # 後ろから省き、そのことを残す
    assert s.callouts[-1].text == "S3"
    _, bad = sg.parse_single({"price_text": 5, "callouts": "x"})
    assert len(bad) == 2
    empty, none = sg.parse_single(None)
    assert not none and empty.callouts == [] and empty.price_text == ""


def test_parse_brief_layout_single_and_backward_compat():
    b = ac.parse_brief(single_raw())
    assert b.layout == "single" and len(b.single.callouts) == 6 and len(b.items) == 1
    # 1点ものは先頭の1点だけを使う
    raw = single_raw()
    raw["items"] = raw["items"] * 1 + [dict(raw["items"][0], id="i2")]
    assert len(ac.parse_brief(raw).items) == 1
    # layout が無い・知らない値は今までどおり（2〜6 点・single は読まない）
    for lay in (None, "grid", 3):
        r = single_raw()
        if lay is None:
            r.pop("layout")
        else:
            r["layout"] = lay
        try:
            ac.parse_brief(r)
            raise AssertionError("1点だと従来の形は作れない")
        except ac.BriefError as e:
            assert any("2〜6" in p for p in e.problems)
        r["items"] = [dict(r["items"][0], id=f"i{k}") for k in range(3)]
        b = ac.parse_brief(r)
        assert b.layout == "collage" and b.single is None and len(b.items) == 3


def test_parse_brief_single_without_single_block_is_ok():
    r = single_raw()
    r.pop("single")
    b = ac.parse_brief(r)
    assert b.layout == "single" and b.single.callouts == [] and b.single.price_text == ""


# ------------------------------------------------------------ メモ・線の配置


def test_leader_curve_is_not_a_ruler_line_and_is_deterministic():
    a, e = np.array([100.0, 100.0]), np.array([420.0, 320.0])
    c1 = sg.leader_curve(a, e, random.Random(1))
    c2 = sg.leader_curve(a, e, random.Random(1))
    assert np.allclose(c1, c2) and np.allclose(c1[0], a) and np.allclose(c1[-1], e)
    chord = (e - a) / np.linalg.norm(e - a)
    off = np.abs((c1 - a) @ np.array([-chord[1], chord[0]]))
    assert off.max() > 6  # 少し曲がっている


def test_polylines_cross():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert sg.polylines_cross(a, np.array([[0.0, 10.0], [10.0, 0.0]]))
    assert not sg.polylines_cross(a, np.array([[20.0, 0.0], [30.0, 10.0]]))


def test_split_lines_avoids_one_char_orphans():
    assert sg.split_lines("USA製", MS, 40) == [["USA製"]]
    two = sg.split_lines("着丈72 身幅60", MS, 40)
    assert two[0] == ["着丈72 身幅60"] and two[1] == ["着丈72", "身幅60"]


def memo(kind, text, lines, fs, cx, cy, w, h, angle, line, end, style):
    return sg.Memo(kind, text, lines, fs, cx, cy, w, h, angle, line, end, style)


def callouts(n):
    kinds = ["brand", "era", "made_in", "fabric", "size", "condition"]
    texts = ["SAMPLE", "90年代", "USA製", "ウール100%", "着丈72 身幅60", "状態 A"]
    return [sg.Callout(kinds[i], texts[i]) for i in range(n)]


def place(n, seed=0, blocked=None, mask=None):
    mask = hero_mask() if mask is None else mask
    blocked = top_blocked() if blocked is None else blocked
    return sg.place_memos(
        callouts(n), hero_mask=mask, blocked=blocked, safe=SAFE, ms=MS, rng=random.Random(seed)
    )


def test_memos_do_not_overlap_anything_and_lines_do_not_cross():
    for n in (2, 4, 6):
        for seed in range(4):
            mask, blocked = hero_mask(), top_blocked()
            memos, dropped, _ = place(n, seed, blocked, mask)
            assert len(memos) + len(dropped) == n
            assert len(memos) >= 2
            plan = sg.SinglePlan(memos=memos, dropped=dropped, wanted=n)
            c, problems = sg.check_plan(
                plan, hero_mask=mask, blocked=blocked, safe=SAFE, width=W,
                expect_price=False, expect_stamp=False,
            )  # fmt: skip
            assert problems == [], (n, seed, problems)
            assert c.line_cross == 0 and c.memo_memo_px == 0 and c.memo_on_hero_px == 0
            # 読める大きさ・画面の中・線の始まりはメモの近く
            assert all(m.fs >= sg.MEMO_MIN_FS for m in memos)
            for m in memos:
                p = m.line_pts()
                assert p[:, 0].min() >= 0 and p[:, 0].max() <= W
                assert p[:, 1].min() >= 0 and p[:, 1].max() <= H


def test_memos_are_deterministic_per_seed():
    a, _, _ = place(4, 7)
    b, _, _ = place(4, 7)
    assert [(m.text, round(m.cx, 1), round(m.cy, 1)) for m in a] == [
        (m.text, round(m.cx, 1), round(m.cy, 1)) for m in b
    ]


def test_memos_that_do_not_fit_are_dropped_from_the_back_and_reported():
    mask = hero_mask()
    blocked = np.ones((H, W), bool)
    blocked[300:1230, 65:330] = False  # 左の細い列だけが空いている
    blocked |= mask
    memos, dropped, _ = place(6, 1, blocked, mask)
    placed_kinds = [m.kind for m in memos]
    assert len(memos) < 6 and dropped
    assert all(d["reason"] and d["kind"] and d["text"] for d in dropped)
    # 優先度の高い（前の）メモが先に置かれる
    order = [c.kind for c in callouts(6)]
    assert placed_kinds == [k for k in order if k in placed_kinds]
    assert order.index(dropped[0]["kind"]) > order.index(placed_kinds[0])
    # 何も置けない空きなら全部省く
    none, dropped_all, _ = place(3, 1, np.ones((H, W), bool), mask)
    assert none == [] and len(dropped_all) == 3


def test_line_does_not_cross_the_hero_except_for_the_end_mark():
    mask = hero_mask()
    memos, _, _ = place(6, 3, mask=mask)
    for m in memos:
        body = sg.line_mask(sg.body_part(m.line_pts()), (H, W), 3)
        assert not np.count_nonzero(body & mask)
        end = np.array(m.end)
        assert mask[int(round(end[1])), int(round(end[0]))]  # 印は服の中（輪郭の少し内側）


def test_kind_targets_collar_for_brand_and_hem_for_size():
    mask = hero_mask()
    memos, _, _ = place(6, 2, mask=mask)
    x0, y0, x1, y1 = sg.mask_bbox(mask)
    by_kind = {m.kind: m for m in memos}
    if "brand" in by_kind:  # 襟元（上のほう）
        assert by_kind["brand"].end[1] < y0 + 0.35 * (y1 - y0)
    if "size" in by_kind:  # 裾（下のほう）
        assert by_kind["size"].end[1] > y0 + 0.55 * (y1 - y0)


# ------------------------------------------------------------ 検査


def test_check_plan_flags_bad_memos_lines_and_badge():
    mask, blocked = hero_mask(), top_blocked()
    on_hero = sg.Memo("brand", "A", ["A"], 48, 540, 760, 200, 60, 0.0,
                      [[100, 700], [300, 760]], [300.0, 760.0], "ring")  # fmt: skip
    c, problems = sg.check_plan(
        sg.SinglePlan(memos=[on_hero], wanted=1), hero_mask=mask, blocked=blocked, safe=SAFE,
        width=W, expect_price=False, expect_stamp=False,
    )  # fmt: skip
    assert any("服に重なっている" in p for p in problems)
    assert c.memo_on_hero_px > sg.TOLERANCE_PX
    # 線の交差・小さい文字・画面外
    m1 = memo(
        "a", "A", ["A"], 20, 150, 1000, 80, 30, 0.0, [[100, 1000], [300, 1100]], [300, 1100], "ring"
    )
    m2 = memo(
        "b", "B", ["B"], 48, 900, 1000, 80, 60, 0.0, [[100, 1100], [300, 1000]], [300, 1000], "ring"
    )
    m3 = memo(
        "c", "C", ["C"], 48, 900, 400, 80, 60, 0.0, [[900, 400], [900, 1500]], [900, 1500], "ring"
    )
    _, problems = sg.check_plan(
        sg.SinglePlan(memos=[m1, m2, m3], wanted=3), hero_mask=mask, blocked=blocked, safe=SAFE,
        width=W, expect_price=False, expect_stamp=False,
    )  # fmt: skip
    text = " ".join(problems)
    assert "交差" in text and "小さくて" in text and "画面の外" in text
    # 価格の札が要るのに無い・メモが2つ置けない
    _, problems = sg.check_plan(
        sg.SinglePlan(wanted=3), hero_mask=mask, blocked=blocked, safe=SAFE, width=W,
        expect_price=True, expect_stamp=False,
    )  # fmt: skip
    assert any("価格の札" in p for p in problems) and any("メモを2つ" in p for p in problems)
    # メモが1つしか無いブリーフは2つ置けなくても問題にしない
    _, problems = sg.check_plan(
        sg.SinglePlan(wanted=1), hero_mask=mask, blocked=blocked, safe=SAFE, width=W,
        expect_price=False, expect_stamp=False,
    )  # fmt: skip
    assert problems == ["メモを置けない"]


def test_badge_is_placed_in_free_space_with_readable_size():
    mask, blocked = hero_mask(), top_blocked()
    b = sg.place_badge(
        "¥9,800", hero_mask=mask, blocked=blocked, safe=SAFE, ms=MS, rng=random.Random(0), width=W
    )
    assert b is not None and b.size >= sg.BADGE_TEXT_MIN and b.r > 100
    disc = b.disc_mask((H, W))
    assert np.count_nonzero(disc & blocked) <= sg.TOLERANCE_PX
    assert np.count_nonzero(disc & mask) / np.count_nonzero(disc) <= sg.BADGE_HERO_OVERLAP
    s = sg.place_stamp_tag(
        "USA製",
        b,
        hero_mask=mask,
        blocked=blocked,
        safe=SAFE,
        ms=MS,
        rng=random.Random(0),
        width=W,
    )
    assert s is not None and sg.poly_overlap(s.polygon(), blocked) <= sg.TOLERANCE_PX


# ------------------------------------------------------------ 細部の写真


def test_closeup_score_and_pick_rules():
    plain = Image.new("RGB", (200, 300), (240, 240, 240))
    plain.paste(Image.new("RGB", (100, 150), (30, 30, 30)), (50, 75))  # 全身: 縁は地
    close = checker(200, 300)  # 縁まで物
    assert sg.closeup_score(plain) < 0.05 and sg.closeup_score(close) > sg.CLOSEUP_TOUCH
    scores = {0: 0.0, 1: 0.1, 2: 0.9, 3: 0.6, 4: 0.8}
    picks = sg.pick_detail_indices(scores, 0, skip={4})
    assert [i for i, _ in picks] == [2, 3]  # アップを大きい順・skip と主役は除く
    # アップが無ければ、主役の次の1枚（2枚目）
    assert [i for i, _ in sg.pick_detail_indices({0: 0.0, 1: 0.1, 2: 0.2}, 0, set())] == [1]
    assert sg.pick_detail_indices({0: 0.0}, 0, set()) == []


def test_prepare_details_skips_main_and_person_photos(tmp_path):
    import io

    def jpg(w, noisy):
        im = checker(w, 200) if noisy else Image.new("RGB", (w, 200), (120, 120, 120))
        buf = io.BytesIO()
        im.save(buf, "JPEG")
        return buf.getvalue()

    urls = [f"https://img.example/{k}.jpg" for k in range(4)]
    # 幅で見分ける: 3 枚目（幅 170）だけ人が写る
    data = {urls[0]: jpg(150, False), urls[1]: jpg(160, True), urls[2]: jpg(165, False)}
    data[urls[3]] = jpg(170, True)

    class Cutter:
        def detect_people(self, img):
            return (
                [{"kind": "person", "score": 0.9, "box": [0, 0, 1, 1]}] if img.width == 170 else []
            )

    out = sg.prepare_details(
        urls, urls[0], tmp_path, lambda: Cutter(), data.__getitem__, log=lambda *_: None
    )
    assert [p.index for p in out] == [1]  # 着画（人が写る）と主役の写真は使わない
    assert out[0].path.exists() and max(Image.open(out[0].path).size) <= sg.PHOTO_MAX_SIDE


# ------------------------------------------------------------ 組み・meta・描画


def build(n=6, fmt="4:5", seed=0, details=None, price="¥9,800"):
    raw = single_raw(callouts=[c for c in single_raw()["single"]["callouts"][:n]])
    raw["single"]["price_text"] = price
    brief = ac.parse_brief(raw)
    return brief, ac.build_single_layout(brief, [hero_sprite()], fmt, seed, MS, None, details)


def test_build_single_layout_end_to_end_meta_and_json():
    for fmt in ac.FORMATS:
        brief, lay = build(6, fmt)
        chk = ac.check_layout(lay, MS)
        assert lay.layout == "single" and lay.single.badge is not None
        assert len(lay.items) == 1 and lay.items[0].sprite.label == ""  # 輪郭の文字は使わない
        assert len(lay.single.memos) >= 2, fmt
        assert chk.verdict == "pass", (fmt, chk.problems)
        j = lay.to_json()
        assert j["layout"] == "single" and len(j["single"]["memos"]) == len(lay.single.memos)
        cand = ac.Candidate(0, lay, chk)
        fr = ac.FormatResult(fmt, "pass", cand, [cand], [], Path("x.png"), Path("x.json"), 0.0, [])
        m = fr.meta()
        assert m["layout"] == "single"
        assert m["callouts_placed"] == len(lay.single.memos)
        assert m["callouts_wanted"] == 6
        assert m["callouts_placed"] + len(m["callouts_dropped"]) == 6
        assert m["worn_used"] is False and m["detail_photos_used"] == 0


def test_two_four_six_callouts_all_place_when_there_is_room():
    for n in (2, 4, 6):
        _, lay = build(n)
        assert len(lay.single.memos) + len(lay.single.dropped) == n
        assert len(lay.single.memos) >= min(2, n)


def test_single_with_details_places_photos_at_the_edge_only(tmp_path):
    p = tmp_path / "d.jpg"
    Image.new("RGB", (300, 300), (90, 40, 40)).save(p)
    det = sg.DetailPhoto(p, "https://img.example/d.jpg", 3, 0.8, "アップ")
    _, lay = build(4, "9:16", details=[det, det])
    assert lay.single.photos
    mask = lay.items[0].mask((lay.w, lay.h))
    for ph in lay.single.photos:
        poly = ph.polygon()
        assert sg.poly_overlap(poly, mask) / sg.poly_area_px(poly) <= sg.PHOTO_HERO_OVERLAP
    _, lay45 = build(4, "4:5", details=[det, det])
    assert len(lay45.single.photos) <= 1
    assert ac.check_layout(lay45, MS).verdict == "pass"


def test_no_price_means_no_badge_and_still_passes():
    _, lay = build(4, price="")
    assert lay.single.badge is None
    assert ac.check_layout(lay, MS).verdict == "pass"


def test_render_html_for_single_has_memos_badge_and_lines(tmp_path):
    brief, lay = build(4)
    lay.items[0].sprite.png_path = tmp_path / "hero.png"
    html = ac.render_html(lay, tmp_path)
    assert html.count('class="memo"') == len(lay.single.memos)
    assert "¥9,800" in html and 'class="lead"' in html and 'class="price"' in html
    # 従来の形のレイアウトには 1点もののものが出ない
    collage = ac.build_layout(
        ac.parse_brief(
            {**single_raw(), "layout": None, "items": [
                dict(single_raw()["items"][0], id=f"i{k}") for k in range(3)
            ]}
        ),
        [ac.Sprite(f"i{k}", f"ラベル{k}", Path("/x.png"), hero_sprite().alpha) for k in range(3)],
        "4:5", 0, MS,
    )  # fmt: skip
    assert collage.layout == "collage" and collage.single is None
    page = ac.render_html(collage, tmp_path)
    assert 'class="memo"' not in page and "price" not in page


def test_collage_meta_has_no_layout_key():
    sprites = [
        ac.Sprite(f"s{k}", f"ラベル{k}", Path("/x.png"), hero_sprite().alpha) for k in range(3)
    ]
    brief = ac.parse_brief({**single_raw(), "layout": "collage", "items": [
        dict(single_raw()["items"][0], id=f"i{k}") for k in range(3)]})  # fmt: skip
    lay = ac.build_layout(brief, sprites, "4:5", 0, MS)
    cand = ac.Candidate(0, lay, ac.check_layout(lay, MS))
    fr = ac.FormatResult("4:5", "pass", cand, [cand], [], Path("x.png"), Path("x.json"), 0.0, [])
    assert "layout" not in fr.meta() and "callouts_placed" not in fr.meta()


def test_layout_collage_and_extra_keys_are_the_classic_collage():
    items = [dict(single_raw()["items"][0], id=f"i{k}") for k in range(3)]
    for lay in ("collage", None, "unknown"):
        raw = {**single_raw(), "items": items, "theme": "特集", "angle": "weekly"}
        raw["layout"] = lay
        brief = ac.parse_brief(raw)  # theme など知らないキーは読まない
        assert brief.layout == "collage" and brief.single is None and len(brief.items) == 3
        sprites = [
            ac.Sprite(f"i{k}", f"ラベル{k}", Path("/x.png"), hero_sprite().alpha) for k in range(3)
        ]
        assert ac.build_layout(brief, sprites, "4:5", 0, MS).layout == "collage"


def test_no_stamp_and_no_aside_still_build_check_and_render(tmp_path):
    # 店の決まりで「1点もの」の札は付けない。stamp・aside が無くても組めて、描けて、検査を通る
    for fmt in ac.FORMATS:
        raw = single_raw()
        del raw["single"]["stamp"], raw["aside"]
        brief = ac.parse_brief(raw)
        assert brief.aside == "" and brief.single.stamp is None
        lay = ac.build_single_layout(brief, [hero_sprite()], fmt, 0, MS, None, None)
        assert lay.single.badge is not None and lay.single.stamp is None
        assert not any(b.name in ("aside", "stamp") for b in lay.blocks)
        chk = ac.check_layout(lay, MS)
        assert chk.verdict == "pass", (fmt, chk.problems)
        lay.items[0].sprite.png_path = tmp_path / "hero.png"
        html = ac.render_html(lay, tmp_path)
        assert "¥9,800" in html and "古着です" not in html
