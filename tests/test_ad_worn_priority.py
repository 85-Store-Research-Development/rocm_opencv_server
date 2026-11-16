"""着画を優先する（入らなければ商品を減らす）・長いラベルを短くする・点線の矢印の余白。値は架空。"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app import ad_compose as ac
from tests.test_ad_compose import brief_raw, ellipse_alpha, sprites

MS = ac.Measurer(None)
LONG = "SAMPLE TERRITORY の ネルシャツ ¥6,800"


def worn_sprite(pid):
    a = np.zeros((520, 260), np.uint8)
    cv2.ellipse(a, (130, 260), (120, 250), 0, 0, 360, 255, -1)
    return ac.Sprite(f"{pid}#worn", "着るとこんな感じ", Path("/x/w.png"), a, worn=True)


# ------------------------------------------------------------ ラベルを短くする


@pytest.mark.parametrize(
    "label, short",
    [
        (LONG, LONG.split(" の ")[0] + " ¥6,800"),  # まず品名を外してブランドを残す
        ("A の B の C ¥12,000", "A ¥12,000"),
        ("ネルシャツ ¥5,800", None),  # ブランドの部分が無い
        ("ブランド の ネルシャツ", None),  # 価格が無い
        ("ブランド の ネルシャツ ¥", None),
        ("", None),
    ],
)
def test_shorten_label(label, short):
    assert ac.shorten_label(label) == short
    if short:
        assert short.split(" ")[-1] == label.split(" ")[-1]  # 価格は必ず残る


def test_long_label_is_shortened_before_going_straight_or_smaller():
    sp = ac.Sprite("s", LONG, Path("/x/s.png"), ellipse_alpha(240, 300))
    geo = ac.geometry("4:5")
    fixed = ac.layout_fixed(ac.parse_brief(brief_raw(3)), "4:5", MS)
    it = ac.ItemState(sp, 300, 540, 700, 4.0)
    ok, _ = ac.place_texts([it], fixed, geo, MS, __import__("random").Random(0), ac.LABEL_STEPS[:1])
    assert ok and it.placed
    assert it.label_full == LONG and it.sprite.label in ac.shorten_steps(LONG)
    assert it.sprite.label.endswith("¥6,800")
    assert not it.straight and it.fs == ac.LABEL_STEPS[0]  # 小さくも、まっすぐにもしていない
    # 短くしなくて入るものは、そのまま
    sp2 = ac.Sprite("s2", "ネルシャツ ¥6,800", Path("/x/s.png"), ellipse_alpha(240, 300))
    it2 = ac.ItemState(sp2, 300, 540, 700, 4.0)
    ac.place_texts([it2], fixed, geo, MS, __import__("random").Random(0), ac.LABEL_STEPS[:1])
    assert it2.label_full == "" and it2.sprite.label == "ネルシャツ ¥6,800"
    # 元のスプライトは書き換えない（別の候補で使い回すため）
    assert sp.label == LONG


def test_shortened_labels_are_reported_in_meta():
    b = ac.parse_brief(brief_raw(3))
    sp = [ac.Sprite(f"s{k}", LONG, Path("/x/s.png"), ellipse_alpha(240, 300)) for k in range(3)]
    lay = ac.build_layout(b, sp, "4:5", 0, MS)
    chk = ac.check_layout(lay, MS)
    cand = ac.Candidate(0, lay, chk)
    fr = ac.FormatResult("4:5", "pass", cand, [cand], [], Path("x.png"), Path("x.json"), 0, [])
    short = fr.meta().get("labels_shortened", [])
    assert short and all(s["from"] == LONG and s["to"].endswith("¥6,800") for s in short)
    # 短くしていなければ、キー自体を付けない（従来の meta のまま）
    plain = [
        ac.Sprite(f"p{k}", "シャツ ¥1,000", Path("/x"), ellipse_alpha(240, 300)) for k in range(3)
    ]
    lay2 = ac.build_layout(b, plain, "4:5", 0, MS)
    cand2 = ac.Candidate(0, lay2, ac.check_layout(lay2, MS))
    fr2 = ac.FormatResult("4:5", "pass", cand2, [cand2], [], Path("x.png"), Path("x.json"), 0, [])
    assert "labels_shortened" not in fr2.meta()


# ------------------------------------------------------------ 商品を減らす


def test_reduce_items_from_the_back_and_keeps_the_worn_product():
    sp = sprites(5)  # s0..s4
    keep, removed = ac.reduce_items(sp, 3, None)
    assert [s.id for s in keep] == ["s0", "s1", "s2"] and [s.id for s in removed] == ["s3", "s4"]
    keep, removed = ac.reduce_items(sp, 3, "s4")  # 後ろにある着画の商品は外さない
    assert [s.id for s in keep] == ["s0", "s1", "s4"] and [s.id for s in removed] == ["s2", "s3"]
    keep, removed = ac.reduce_items(sp, 9, None)
    assert len(keep) == 5 and removed == []


def compose(tmp_path, fmt, n, worn_id=None, monkeypatch=None, fail_when=None, cap=None):
    raw = brief_raw(n)
    raw["candidates"] = 2
    b = ac.parse_brief(raw)
    sp = sprites(n)
    worn = worn_sprite(worn_id) if worn_id else None
    if monkeypatch and cap:
        monkeypatch.setitem(ac.WORN_MAX_ITEMS, fmt, cap)
    if monkeypatch and fail_when:
        real = ac.check_layout

        def fake(lay, ms=None):
            c = real(lay, ms)
            if fail_when(lay):
                c.problems.append("テスト用の失敗")
                c.verdict = "fail"
            return c

        monkeypatch.setattr(ac, "check_layout", fake)
    return ac.compose_format(
        b, sp, [{"id": "x", "status": "fail", "reasons": ["元から外れた"]}], fmt, tmp_path,
        Path("."), render=False, worn=worn, worn_candidates=1,
    )  # fmt: skip


def test_worn_has_priority_items_are_dropped_from_the_back(tmp_path, monkeypatch):
    # 4 点を超えると検査に通らない（4:5）ことにする。着画の商品 s4 は後ろにあるが外さない
    fr = compose(
        tmp_path, "4:5", 5, "s4", monkeypatch,
        fail_when=lambda lay: len([i for i in lay.items if not i.sprite.worn]) > 3,
    )  # fmt: skip
    lay = fr.best.layout
    ids = [it.sprite.id for it in lay.items]
    assert fr.verdict == "pass" and lay.worn_item_id == "s4" and lay.worn_fallback is None
    assert "s4" in ids and "s4#worn" in ids and len(ids) == 4  # 3 点 + 着画
    omitted = [d for d in fr.dropped if d["status"] == "omitted"]
    assert [d["id"] for d in omitted] == ["s2", "s3"]
    assert all(d["reasons"] == ["着画を入れるため"] for d in omitted)
    assert fr.dropped[0]["id"] == "x"  # 元から外れた物は残る
    assert fr.meta()["dropped_items"] == fr.dropped


def test_format_caps_start_below_the_brief_count(tmp_path, monkeypatch):
    fr = compose(tmp_path, "9:16", 5, "s0", monkeypatch, cap=3)
    kept = [it.sprite.id for it in fr.best.layout.items]
    assert len(kept) <= 4 and "s0#worn" in kept
    assert {d["id"] for d in fr.dropped if d["status"] == "omitted"} >= {"s3", "s4"}


def test_worn_is_given_up_only_when_nothing_passes_and_all_items_come_back(tmp_path, monkeypatch):
    fr = compose(
        tmp_path, "4:5", 5, "s1", monkeypatch,
        fail_when=lambda lay: any(i.sprite.worn for i in lay.items),
    )  # fmt: skip
    lay = fr.best.layout
    assert lay.worn_item_id is None and "テスト用の失敗" in lay.worn_fallback
    assert len(lay.items) == 5  # 着画なしは全点に戻す
    assert not [d for d in fr.dropped if d["status"] == "omitted"]
    assert fr.verdict == "pass"


def test_without_worn_keeps_all_items_and_reduces_only_when_it_would_need_review(
    tmp_path, monkeypatch
):
    fr = compose(tmp_path, "1:1", 5)
    assert len(fr.best.layout.items) == 5 and len(fr.dropped) == 1  # 通るなら点数を保つ
    fr = compose(
        tmp_path, "1:1", 5, None, monkeypatch, fail_when=lambda lay: len(lay.items) > 4
    )  # fmt: skip
    assert fr.verdict == "pass" and len(fr.best.layout.items) == 4
    om = [d for d in fr.dropped if d["status"] == "omitted"]
    assert [d["id"] for d in om] == ["s4"] and "検査に通らなかった" in om[0]["reasons"][0]
    # どの点数でも通らないなら、全点のまま needs_review（減らして隠さない）
    fr = compose(tmp_path, "1:1", 4, None, monkeypatch, fail_when=lambda lay: True)
    assert fr.verdict == "needs_review" and len(fr.best.layout.items) == 4
    assert not [d for d in fr.dropped if d["status"] == "omitted"]


# ------------------------------------------------------------ 点線の矢印


def link_items(gap):
    geo = ac.geometry("4:5")
    w = ac.ItemState(worn_sprite("a"), 300, 200, 700, 0.0)
    t = ac.ItemState(
        ac.Sprite("a", "", Path("/x"), ellipse_alpha(240, 300)), 280, 200 + gap, 700, 0.0
    )
    return geo, w, t


def test_link_is_not_drawn_beyond_40_percent_of_the_width():
    geo, w, t = link_items(0.3 * 1080 + 300)  # 縁どうしは 0.3 倍幅より少し離れる
    size = (geo.w, geo.h)
    assert ac.plan_link(w, t, size, np.zeros((geo.h, geo.w), bool)) is not None
    geo, w, t = link_items(0.55 * 1080 + 300)
    assert ac.plan_link(w, t, (geo.w, geo.h), np.zeros((geo.h, geo.w), bool)) is None


def test_link_keeps_20px_from_other_items_and_text():
    geo, w, t = link_items(0.3 * 1080 + 300)
    size = (geo.w, geo.h)
    free = ac.plan_link(w, t, size, np.zeros((geo.h, geo.w), bool))
    assert free is not None
    # 通り道の横長の帯だけを空けて上下を塞ぐ。帯が 20px の余白 × 2 より狭ければ引かない
    for half, expect in ((14, False), (60, True)):
        taken = np.ones((geo.h, geo.w), bool)
        taken[700 - half : 700 + half, :] = False
        taken[:, :] &= ~(w.mask(size) | t.mask(size))
        got = ac.plan_link(w, t, size, taken)
        assert (got is not None) == expect, half
        if got is not None:
            pts = np.round(np.array(got["pts"])).astype(int)
            near = cv2.dilate(taken.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
            assert not near[pts[3:-3, 1], pts[3:-3, 0]].any()


def test_shorten_steps_keep_brand_first_then_name():
    assert ac.shorten_steps("ブランド の ネルシャツ ¥8,070") == [
        "ブランド ¥8,070",
        "ネルシャツ ¥8,070",
    ]
    assert ac.shorten_steps("ネルシャツ ¥5,800") == []
