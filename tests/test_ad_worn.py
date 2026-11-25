"""着画（人が着ている商品写真）の候補の選び方・コラージュへの入れ方（app/ad_worn.py・
app/ad_compose.py・app/cutout.py の人用の判定）。モデルと Chrome は使わない。値は架空。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import ad_compose as ac
from app import ad_worn as aw
from app import cutout

SIZE = (1000, 1500)  # 縦長の写真（幅, 高さ）


def person(score=0.7, box=(150, 150, 850, 1400)):
    return {"kind": "person", "score": score, "box": list(box)}


def face(score=0.8, box=(420, 220, 580, 400)):
    return {"kind": "face", "score": score, "box": list(box)}


# ------------------------------------------------------------ 候補の選び方


def test_judge_accepts_a_person_with_a_face():
    j = aw.judge([person(), face()], SIZE)
    assert j and j["full_body"] and j["area_ratio"] > 0.5 and j["person_score"] == 0.7


@pytest.mark.parametrize(
    "dets, why",
    [
        ([person()], "顔が無い（服だけ・手だけ・顔の無い写真）"),
        ([person(0.45), face()], "人の確信度が低い（平置きの服が person に出る程度）"),
        ([person(), person(0.6, (0, 0, 100, 400)), face()], "人が2人"),
        (
            [person(box=(400, 300, 520, 500)), face(box=(430, 310, 470, 350))],
            "箱が小さい（プリントの顔）",
        ),
        ([person(box=(100, 600, 900, 1100)), face(box=(400, 620, 520, 740))], "箱の高さが足りない"),
        ([person(), face(0.3)], "顔の確信度が低い"),
        ([person(), face(box=(420, 0, 580, 120))], "顔の上が画像の端で切れている（縦横比 0.75）"),
        ([person(), face(box=(420, 0, 580, 60))], "頭が画像の端で切れている"),
        ([person(), face(box=(420, 1000, 580, 1200))], "顔が箱の下のほう"),
        ([person(), face(box=(450, 220, 460, 240))], "顔が小さすぎる"),
    ],
)
def test_judge_rejects(dets, why):
    assert aw.judge(dets, SIZE) is None, why


def test_judge_ignores_weak_second_person():
    assert aw.judge([person(), person(0.3, (0, 400, 100, 900)), face()], SIZE)


def test_judge_prefers_bigger_fuller_person():
    big = aw.judge([person(0.7, (100, 50, 900, 1450)), face()], SIZE)
    small = aw.judge([person(0.7, (200, 250, 850, 1100)), face(box=(480, 270, 600, 390))], SIZE)
    assert big["score"] > small["score"]


def cand(item, idx, score, area=0.4):
    return aw.WornCandidate(item, idx, f"u{idx}", score, [0, 0, 1, 1], 0.7, area, True)


def test_best_per_item_takes_the_best_photo_per_item():
    per = aw.best_per_item(
        [cand("a", 3, 1.0), cand("a", 5, 1.2), cand("b", 2, 0.9), cand("b", 1, 0.9)]
    )
    assert per["a"].image_index == 5
    assert per["b"].image_index == 1  # 同点は前のほうの写真


def test_pick_worn_item_best_then_first_in_brief_order():
    per = aw.best_per_item([cand("a", 0, 1.1), cand("b", 0, 1.3), cand("c", 0, 1.3)])
    assert aw.pick_worn_item(per, ["a", "b", "c"]).item_id == "b"
    assert aw.pick_worn_item(per, ["c", "a", "b"]).item_id == "c"  # 同点は案の中で先頭
    assert aw.pick_worn_item({}, ["a"]) is None
    assert aw.pick_worn_item(per, ["x"]) is None


def test_normalize_images():
    assert aw.normalize_images("u", None) == []
    assert aw.normalize_images("u", "http://a") == []
    urls = ["https://x/1.jpg", "https://x/1.jpg", "ftp://x", 3, "http://x/2.jpg"]
    assert aw.normalize_images("u", urls) == ["https://x/1.jpg", "http://x/2.jpg"]
    assert (
        len(aw.normalize_images("u", [f"https://x/{i}.jpg" for i in range(30)]))
        == aw.WORN_MAX_IMAGES
    )


def test_parse_people_keeps_only_person_and_face():
    out = cutout.parse_people(
        ["person", "face", "hand", "person face"], [0.7, 0.8, 0.3, 0.4], [[0, 0, 1, 1]] * 4
    )
    assert [d["kind"] for d in out] == ["person", "face"]


# ------------------------------------------------------------ 人の切り抜きの判定


def test_person_grade():
    ok = cutout.PersonQuality(0.2, 0.6, 0.7, 1.0, 0.98, 0.6)
    assert cutout.grade_person(ok) == ("ok", [])
    st, why = cutout.grade_person(cutout.PersonQuality(0.02, 0.6, 0.7, 1.0, 0.98, 0.6))
    assert st == "fail" and "小さすぎる" in why[0]
    assert cutout.grade_person(cutout.PersonQuality(0.2, 0.1, 0.7, 1.0, 0.98, 0.6))[0] == "fail"
    assert cutout.grade_person(cutout.PersonQuality(0.2, 0.6, 0.3, 1.0, 0.98, 0.6))[0] == "check"
    assert cutout.grade_person(cutout.PersonQuality(0.2, 0.6, 0.7, 0.8, 0.98, 0.6))[0] == "check"
    assert cutout.grade_person(cutout.PersonQuality(0.2, 0.6, 0.7, 1.0, 0.5, 0.6))[0] == "check"
    assert cutout.grade_person(cutout.PersonQuality(0.2, 0.6, 0.7, 1.0, 0.98, 0.97))[0] == "check"


def test_finish_person_keeps_pixels_and_flags_small_mask():
    from PIL import Image

    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (300, 200, 3), np.uint8))
    sam = np.zeros((300, 200), bool)
    sam[30:280, 50:150] = True
    r = cutout.finish_person(img, sam, sam.astype(np.float32), [50, 30, 150, 280], 0.7)
    assert r.status in ("ok", "check") and r.method == "person"
    a = np.asarray(r.rgba.getchannel("A"))
    rgb = np.asarray(r.rgba.convert("RGB"))
    src = np.asarray(img)
    ys, xs = np.where(a > 250)
    # 画素は描き直さない: 不透明の所の色は元の写真と同じ（切り詰めた分だけずれる）
    assert a.max() == 255 and rgb[ys[0], xs[0]].tolist() in src.reshape(-1, 3).tolist()
    tiny = np.zeros((300, 200), bool)
    tiny[100:110, 100:110] = True
    assert (
        cutout.finish_person(img, tiny, tiny.astype(np.float32), [90, 90, 120, 120], 0.7).status
        == "fail"
    )


# ------------------------------------------------------------ ブリーフ（images・worn_note）


def raw(n=3, **over):
    r = {
        "id": "w1",
        "format": "4:5",
        "store": {"name": "SAMPLE", "tagline": "ここで会いましょう。", "area_line": "テスト町"},
        "headline": "あたらしい服、\\nとどいてます。",
        "aside": "古着です",
        "palette": "navy",
        "items": [
            {"id": f"i{k}", "image": f"/x/{k}.jpg", "label": f"品{k} ¥{k}000"} for k in range(n)
        ],
        "candidates": 3,
    }
    r.update(over)
    return r


def test_brief_without_images_is_unchanged():
    b = ac.parse_brief(raw())
    assert b.worn_note == "" and all(it.images == [] for it in b.items)


def test_brief_with_images_and_note():
    r = raw(worn_note="着るとこんな感じ")
    r["items"][1]["images"] = ["https://x/a.jpg", "https://x/b.jpg"]
    b = ac.parse_brief(r)
    assert b.worn_note == "着るとこんな感じ" and b.items[1].images == r["items"][1]["images"]
    with pytest.raises(ac.BriefError):
        ac.parse_brief(raw(worn_note="あ" * 40))


def test_prepare_worn_skips_when_no_images(tmp_path):
    called = []
    out = ac.prepare_worn(
        ac.parse_brief(raw()),
        [],
        tmp_path,
        lambda: called.append(1),
        lambda u: b"",
        log=lambda *_: None,
    )
    assert out == (None, 0, []) and not called  # 探さない（モデルも読まない）


# ------------------------------------------------------------ 着画をコラージュに入れる


def sprites(n):
    out = []
    for k in range(n):
        a = np.zeros((380, 280), np.uint8)
        cv2.ellipse(a, (140, 190), (130, 180), 0, 0, 360, 255, -1)
        out.append(ac.Sprite(f"i{k}", f"品{k} ¥{k}000", Path(f"/x/{k}.png"), a))
    return out


def worn_sprite(note="着るとこんな感じ"):
    a = np.zeros((900, 420), np.uint8)
    cv2.ellipse(a, (210, 450), (200, 440), 0, 0, 360, 255, -1)
    sp = ac.Sprite("i1#worn", note, Path("/x/w.png"), a)
    sp.worn = True
    return sp


@pytest.mark.parametrize("fmt", ["4:5", "1:1", "9:16"])
def test_exactly_one_worn_item_and_item_photo_stays(fmt):
    b = ac.parse_brief(raw(4))
    lay = ac.build_layout(b, sprites(4), fmt, 1, ac.Measurer(None), worn_sprite())
    worn = [it for it in lay.items if it.sprite.worn]
    assert len(worn) == 1 and len(lay.items) == 5
    assert any(it.sprite.id == "i1" for it in lay.items)  # アイテム写真も置く
    assert lay.worn_item_id == "i1"
    w = worn[0]
    assert abs(w.angle) <= ac.WORN_ANGLE[1] + 1e-6  # 傾きは小さめ
    assert w.h >= max(it.h for it in lay.items if not it.sprite.worn)  # ほかより大きい
    js = lay.to_json()
    assert js["worn_item_id"] == "i1" and sum(i["worn"] for i in js["items"]) == 1


def test_worn_without_note_has_no_label_and_passes_checks():
    b = ac.parse_brief(raw(3))
    ok = False
    for seed in range(4):
        lay = ac.build_layout(b, sprites(3), "4:5", seed, ac.Measurer(None), worn_sprite(""))
        c = ac.check_layout(lay, ac.Measurer(None))
        w = next(it for it in lay.items if it.sprite.worn)
        assert w.arc is None and w.placed and w.sprite.id not in c.unplaced_text
        ok = ok or c.verdict == "pass"
    assert ok


def test_worn_label_follows_the_worn_contour_and_is_checked():
    b = ac.parse_brief(raw(3))
    lay = ac.build_layout(b, sprites(3), "4:5", 2, ac.Measurer(None), worn_sprite())
    w = next(it for it in lay.items if it.sprite.worn)
    assert w.arc is not None and ac.arc_length(w.arc) >= w.text_len - 1e-6
    c = ac.check_layout(lay, ac.Measurer(None))
    assert w.sprite.id not in c.truncated_labels + c.reversed_labels


def test_link_arrow_avoids_other_things_when_present():
    b = ac.parse_brief(raw(3))
    found = 0
    for seed in range(6):
        lay = ac.build_layout(b, sprites(3), "4:5", seed, ac.Measurer(None), worn_sprite())
        if lay.link is None:
            continue
        found += 1
        size = (lay.w, lay.h)
        others = np.zeros((lay.h, lay.w), bool)
        for it in lay.items:
            if it.sprite.id not in ("i1", "i1#worn"):
                others |= it.mask(size)
        assert not np.count_nonzero(lay.link["mask"] & others)
        assert ac.check_layout(lay, ac.Measurer(None)).doodle_overlap_px <= ac.TOLERANCE_PX
    assert found >= 1


def test_compose_format_with_worn_records_item_and_candidates(tmp_path):
    b = ac.parse_brief(raw(3))
    fr = ac.compose_format(
        b, sprites(3), [], "4:5", tmp_path, tmp_path, render=False,
        worn=worn_sprite(), worn_candidates=2, log=lambda *_: None,
    )  # fmt: skip
    meta = fr.meta()
    assert meta["worn_item_id"] == "i1" and meta["worn_candidates"] == 2
    assert meta["worn_fallback"] is None
    js = json.loads(fr.layout_json.read_text())
    assert js["worn_item_id"] == "i1" and js["worn_candidates"] == 2


def test_compose_format_falls_back_without_worn_when_checks_fail(tmp_path, monkeypatch):
    real = ac.check_layout

    def strict(lay, ms=None):
        c = real(lay, ms)
        if any(it.sprite.worn for it in lay.items):
            c.problems.append("入らない（試験用）")
            c.verdict = "fail"
        return c

    monkeypatch.setattr(ac, "check_layout", strict)
    b = ac.parse_brief(raw(3))
    fr = ac.compose_format(
        b, sprites(3), [], "4:5", tmp_path, tmp_path, render=False,
        worn=worn_sprite(), worn_candidates=1, log=lambda *_: None,
    )  # fmt: skip
    lay = fr.best.layout
    assert not any(it.sprite.worn for it in lay.items) and len(lay.items) == 3
    meta = fr.meta()
    assert meta["worn_item_id"] is None and "入らない" in meta["worn_fallback"]
    assert meta["worn_candidates"] == 1 and fr.verdict == "pass"  # 着画なしの今の組みに戻る


def test_no_worn_is_the_old_behaviour(tmp_path):
    b = ac.parse_brief(raw(3))
    fr = ac.compose_format(b, sprites(3), [], "4:5", tmp_path, tmp_path, render=False, log=print)
    meta = fr.meta()
    assert meta["worn_item_id"] is None and meta["worn_candidates"] == 0
    assert meta["worn_fallback"] is None


def test_prepare_worn_only_searches_items_whose_photo_is_placed(tmp_path):
    r = raw(3)
    for it in r["items"]:
        it["images"] = [f"https://x/{it['id']}.jpg"]
    asked = []

    def get():
        asked.append(1)
        raise RuntimeError("モデル無し")

    out = ac.prepare_worn(
        ac.parse_brief(r), sprites(1), tmp_path, get, lambda u: b"", log=lambda *_: None
    )
    assert out[0] is None  # 失敗しても着画なしで続ける（例外を出さない）
