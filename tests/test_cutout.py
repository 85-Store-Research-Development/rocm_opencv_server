import numpy as np
from PIL import Image

from app import cutout
from scripts import cutout_trial


def blob(h=200, w=200, box=(60, 40, 140, 160)):
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_clean_mask_fills_small_hole_and_drops_speck():
    m = blob()
    m[90:93, 90:93] = False  # 内側の小さな穴
    m[5:7, 5:7] = True  # 孤立した粒
    out = cutout.clean_mask(m)
    assert out[91, 91]
    assert not out[5, 5]
    assert out[100, 100]


def test_clean_mask_keeps_big_hole_and_border_background():
    m = blob(box=(20, 20, 180, 180))
    m[60:140, 60:140] = False  # 大きな穴（襟ぐりの外など）は残す
    out = cutout.clean_mask(m)
    assert not out[100, 100]


def test_soften_edge_shrinks_and_is_soft():
    a = cutout.soften_edge(blob())
    assert a.dtype == np.uint8
    assert a[100, 100] == 255
    assert a[40, 100] < 255  # 元の縁は内側に寄っている
    assert a[10, 10] == 0


def test_rgba_keeps_pixels_untouched():
    img = Image.new("RGB", (50, 50), (10, 200, 30))
    rgba = cutout.to_rgba(img, np.full((50, 50), 255, np.uint8))
    assert rgba.mode == "RGBA"
    assert rgba.getpixel((3, 3)) == (10, 200, 30, 255)


def test_crop_to_content_adds_margin_and_clamps():
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    rgba.paste((255, 0, 0, 255), (50, 60, 150, 120))
    out = cutout.crop_to_content(rgba, margin=0.1)
    assert out.size == (120, 80)
    full = Image.new("RGBA", (40, 40), (1, 2, 3, 255))
    assert cutout.crop_to_content(full).size == (40, 40)
    assert cutout.crop_to_content(Image.new("RGBA", (9, 9), (0, 0, 0, 0))).size == (9, 9)


def test_measure_and_grade_ok():
    q = cutout.measure(blob(box=(50, 30, 150, 170)), 0.7)
    assert not q.touches_border and q.components == 1
    assert q.grade()[0] == "ok"


def test_grade_border_and_pieces_need_check():
    m = blob(box=(0, 30, 120, 170))
    q = cutout.measure(m, 0.7)
    assert q.touches_border
    status, reasons = q.grade()
    assert status == "check" and any("見切れ" in r for r in reasons)
    two = blob(box=(10, 40, 80, 160)) | blob(box=(110, 40, 190, 160))
    assert cutout.measure(two, 0.7).components == 2


def test_grade_fail_cases():
    assert cutout.Quality(0.01, False, 1, 0.8).grade()[0] == "fail"
    assert cutout.Quality(0.97, False, 1, 0.8).grade()[0] == "fail"
    assert cutout.Quality(0.3, False, 1, 0.1).grade()[0] == "fail"
    assert cutout.Quality(0.3, False, 1, 0.35).grade()[0] == "check"


def test_pick_box_prefers_large_confident_box():
    boxes = [[0, 0, 20, 20], [10, 10, 190, 190]]
    box, score = cutout.pick_box(boxes, [0.6, 0.5], (200, 200))
    assert box == boxes[1]
    assert cutout.pick_box([], [], (10, 10)) is None
    # 下限未満しか無ければ確信度の最大（判定側で fail）
    assert cutout.pick_box(boxes, [0.1, 0.05], (200, 200))[1] == 0.1


def test_prompt_for():
    assert "shirt" in cutout.prompt_for("Shirts")
    assert cutout.prompt_for(None) == cutout.GENERIC_PROMPT


def test_finish_fail_returns_no_image():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    res = cutout.finish(img, np.zeros((200, 200), bool) | blob(box=(0, 0, 4, 4)), 0.9)
    assert res.status == "fail" and res.rgba is None


def photo(garment, wall, box=(50, 30, 150, 170), size=200):
    arr = np.full((size, size, 3), wall, np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = garment
    return Image.fromarray(arr)


def test_edge_contrast_is_small_for_white_on_white_and_large_for_dark():
    m = blob(box=(50, 30, 150, 170))
    weak = cutout.edge_contrast_top(cutout.lab_image(photo((245, 245, 245), (235, 235, 235))), m)
    strong = cutout.edge_contrast_top(cutout.lab_image(photo((30, 30, 30), (220, 220, 220))), m)
    assert weak < cutout.EDGE_DELTA_CHECK < strong
    assert (
        cutout.edge_contrast_top(
            cutout.lab_image(photo((0, 0, 0), (0, 0, 0))), np.zeros((200, 200), bool)
        )
        == cutout.NO_EDGE_DELTA
    )


def test_top_roughness_is_larger_for_a_notched_collar():
    smooth = blob(h=300, w=300, box=(50, 50, 250, 250))
    ragged = smooth.copy()
    for x in range(100, 200, 14):  # 襟元のギザギザの欠け
        ragged[50:75, x : x + 7] = False
    assert cutout.top_roughness(ragged) > cutout.ROUGH_CHECK > cutout.top_roughness(smooth)


def test_grade_checks_weak_edge_and_rough_top():
    base = cutout.Quality(area_ratio=0.3, touches_border=False, components=1, box_score=0.8)
    assert cutout.grade(base)[0] == "ok"
    # 淡い色（境目が弱い）だけでは外さない。輪郭が少し荒いときだけ check
    weak = cutout.Quality(0.3, False, 1, 0.8, edge_delta=10.0, roughness=0.0003)
    assert cutout.grade(weak)[0] == "ok"
    weak_rough = cutout.Quality(0.3, False, 1, 0.8, edge_delta=10.0, roughness=0.0012)
    status, reasons = cutout.grade(weak_rough)
    assert status == "check" and "境目が弱く" in reasons[0]
    strong = cutout.Quality(0.3, False, 1, 0.8, edge_delta=40.0, roughness=0.0012)
    assert cutout.grade(strong)[0] == "ok"
    rough = cutout.Quality(0.3, False, 1, 0.8, roughness=0.004)
    assert cutout.grade(rough)[0] == "check"


def test_finish_white_on_white_with_clean_outline_stays_ok():
    """白い服 × 白い壁でも、輪郭がなめらかなら外さない（境目の弱さは品質に残る）"""
    img = photo(garment=(245, 245, 245), wall=(235, 235, 235))
    res = cutout.finish(img, blob(box=(50, 30, 150, 170)), 0.8)
    assert res.status == "ok" and res.rgba is not None
    assert res.quality.edge_delta < cutout.EDGE_DELTA_CHECK


def test_finish_ok_returns_cropped_rgba():
    img = photo(garment=(30, 30, 30), wall=(200, 200, 200))
    res = cutout.finish(img, blob(box=(50, 30, 150, 170)), 0.8)
    assert res.status == "ok"
    assert res.rgba.mode == "RGBA" and res.rgba.width < 200


def test_pick_targets_spreads_types():
    preds = {
        f"gid://x/Product/{i}": {
            "productType": t,
            "images": [f"u{i}"],
            "title": "t",
            "photo": {"answers": {"subtype": {"choice": c}}},
        }
        for i, (t, c) in enumerate(
            [("Shirts", "a")] * 5
            + [("Shirts", "b")] * 3
            + [("Coats", "x")] * 3
            + [("Knit", "y")] * 3
        )
    }
    out = cutout_trial.pick_targets(preds, 6)
    assert len(out) == 6 and len({o["id"] for o in out}) == 6
    assert {o["product_type"] for o in out} == {"Shirts", "Coats", "Knit"}
    assert out[0]["url"]


def test_page_is_self_contained_and_escapes():
    rows = [
        {
            "id": "1",
            "product_type": "Shirts",
            "source": "1_src.jpg",
            "cutout": "1.png",
            "status": "ok",
            "reasons": [],
            "seconds": 1.2,
            "quality": {"area_ratio": 0.3, "box_score": 0.7, "components": 1},
        },
        {
            "id": "2<b>",
            "product_type": "",
            "source": "2_src.jpg",
            "status": "fail",
            "reasons": ["服を検出できなかった"],
            "seconds": 0.5,
            "quality": None,
        },
    ]
    out = cutout_trial.page(rows)
    assert "http" not in out.replace("http-equiv", "")
    assert "2&lt;b&gt;" in out and "1.png" in out
    assert "ok 1 ・ check 0 ・ fail 1" in out


def test_tall_narrow_mask_needs_check():
    q = cutout.measure(blob(box=(80, 10, 120, 190)), 0.8)
    assert q.aspect > cutout.MAX_ASPECT
    status, reasons = q.grade()
    assert status == "check" and any("細長い" in r for r in reasons)


# ---------------------------------------------- BiRefNet との合わせ方（モデルを使わない）


def test_refine_alpha_is_zero_outside_the_grown_region():
    sam = blob(h=400, w=400, box=(100, 100, 300, 300))
    fg = np.ones(sam.shape, np.float32)  # BiRefNet は全面を前景と言う
    out = cutout.refine_alpha(sam, fg)
    assert out[200, 200] == 1.0
    assert out[5, 5] == 0.0  # 膨らませた領域の外は 0
    assert out[200, 20] == 0.0


def test_refine_alpha_fills_a_notch_but_not_a_distant_hanger():
    sam = blob(h=400, w=400, box=(100, 100, 300, 300))
    sam[100:112, 192:208] = False  # 襟元の欠け（へこみ）
    fg = np.zeros(sam.shape, np.float32)
    fg[100:300, 100:300] = 1.0  # 欠けた所も前景と言っている
    fg[40:90, 195:205] = 1.0  # マスクから離れたハンガーのフック
    out = cutout.refine_alpha(sam, fg)
    assert out[105, 200] > 0.5  # 欠けは埋まる
    assert out[60, 200] == 0.0  # フックは入らない


def test_refine_alpha_drops_thin_things_next_to_the_mask():
    sam = blob(h=400, w=400, box=(100, 100, 300, 300))
    fg = np.zeros(sam.shape, np.float32)
    fg[100:300, 100:300] = 1.0
    fg[93:100, 150:153] = 1.0  # 縁にくっついた細い針金（幅3px）
    out = cutout.refine_alpha(sam, fg)
    assert out[96, 151] == 0.0


def test_refine_alpha_keeps_the_core_opaque_even_if_birefnet_says_background():
    sam = blob(h=400, w=400, box=(100, 100, 300, 300))
    fg = np.zeros(sam.shape, np.float32)  # BiRefNet が何も前景と言わない
    out = cutout.refine_alpha(sam, fg)
    assert out[200, 200] == 1.0


def test_finish_alpha_keeps_pixels_and_reports_method():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 400, 3), np.uint8))
    sam = blob(h=400, w=400, box=(100, 100, 300, 300))
    r = cutout.finish_alpha(img, sam, sam.astype(np.float32), 0.9)
    assert r.method == cutout.METHOD_BIREFNET
    assert r.summary()["method"] == "sam2+birefnet"
    assert r.rgba is not None
    rgb = np.asarray(r.rgba.convert("RGB"))
    # 中心の画素は元の写真のまま（描き直さない）
    a = np.asarray(r.rgba.getchannel("A"))
    cy, cx = np.array(a.shape) // 2
    assert (rgb[cy, cx] == np.asarray(img)[200, 200]).all()


def test_finish_without_birefnet_is_sam2():
    img = Image.fromarray(np.zeros((400, 400, 3), np.uint8))
    r = cutout.finish(img, blob(h=400, w=400, box=(100, 100, 300, 300)), 0.9)
    assert r.method == cutout.METHOD_SAM2


def test_cut_falls_back_to_sam2_when_birefnet_cannot_load(tmp_path):
    class Stub(cutout.Cutter):
        def __init__(self):  # モデルは読まない
            self.birefnet_path = str(tmp_path / "none")
            self.birefnet = None
            self._birefnet_tried = False
            self.birefnet_error = ""
            self.device = "cpu"

        def detect(self, image, product_type):
            return [0.0, 0.0, 10.0, 10.0], 0.9

        def segment(self, image, box):
            return blob(h=400, w=400, box=(100, 100, 300, 300))

    img = Image.fromarray(np.zeros((400, 400, 3), np.uint8))
    r = Stub().cut(img, "Shirts", method=cutout.METHOD_BIREFNET)
    assert r.method == cutout.METHOD_SAM2  # 黙って SAM2 だけに戻る
    assert r.status in ("ok", "check")


def test_kornia_stand_in_is_only_for_missing_kornia():
    import sys

    saved = {k: sys.modules.pop(k) for k in ("kornia", "kornia.filters") if k in sys.modules}
    try:
        cutout._ensure_kornia()
        if "kornia" in sys.modules and hasattr(sys.modules["kornia"], "filters"):
            lap = sys.modules["kornia.filters"].laplacian
            try:
                lap()
            except NotImplementedError:
                pass
    finally:
        for k in ("kornia", "kornia.filters"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_method_for_splits_by_item_type(monkeypatch):
    monkeypatch.delenv("CUTOUT_METHOD", raising=False)
    assert cutout.method_for("Coats & Jackets") == cutout.METHOD_SAM2
    assert cutout.method_for(None) == cutout.METHOD_SAM2
    assert cutout.method_for("Shirts") == cutout.METHOD_BIREFNET
    assert cutout.method_for("T-Shirts") == cutout.METHOD_BIREFNET
    monkeypatch.setenv("CUTOUT_METHOD", "sam2")
    assert cutout.method_for("Shirts") == cutout.DEFAULT_METHOD
