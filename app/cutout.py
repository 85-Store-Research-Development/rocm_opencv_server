"""商品写真から服だけを切り抜いて、透過 PNG（RGBA）にする。広告の画像用の試し。

流れ: Grounding DINO で服の箱を取る → SAM2 で箱からマスク → 小さな穴埋め・孤立した粒の除去 →
縁を内側に寄せて軽くぼかす → 余白を切り詰めて RGBA。
**商品の画素は描き直さない**（生成モデルで補正・補完しない）。マスクで抜くだけで、色は元の写真のまま。

モデルは新しくダウンロードしない。手元の ~/models にあるものを、オフラインで読む
（CUTOUT_DINO_PATH / CUTOUT_SAM_PATH で変えられる）。

品質の目安（Quality）も返し、規則で ok（そのまま使える）/ check（人が見る）/ fail（切り抜かず
写真のまま使う）を付ける。しきい値は下の名前付きの定数。出どころの無い値は暫定。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DINO_PATH = os.getenv("CUTOUT_DINO_PATH", "~/models/grounding-dino-base")
SAM_PATH = os.getenv("CUTOUT_SAM_PATH", "~/models/sam2.1-hiera-large")

# 品目（Shopify の商品タイプの英語名）→ 検出に使う語。無ければ GENERIC
PROMPTS = {
    "Shirts": "shirt",
    "Coats & Jackets": "jacket . coat",
    "Sweaters": "sweater",
    "Sweatshirts": "sweatshirt . hoodie",
    "Trousers": "pants",
    "Short Trousers": "shorts",
    "Hats": "hat",
}
GENERIC_PROMPT = "clothing . garment"
# 箱を採用する確信度の下限（これ未満しか無ければ fail）
BOX_MIN_SCORE = 0.25
TEXT_THRESHOLD = 0.2

# --- 後処理 ---
# 面積がマスクのこの割合未満の連結成分は、孤立した粒として消す
SPECK_RATIO = 0.002
# 穴（服の内側の背景の小さな抜け）は、画像のこの割合未満なら埋める
HOLE_RATIO = 0.0015
# 縁を内側へ寄せる幅（短辺に対する割合。最低 1px）と、ぼかしの強さ（σ）
ERODE_RATIO = 1 / 700
FEATHER_SIGMA = 1.0
# 切り詰めるときに残す余白（切り抜いた物の長辺に対する割合）
CROP_MARGIN = 0.03

# --- 判定のしきい値 ---
FAIL_AREA_MIN = 0.03  # マスクが画像のこれ未満なら、服を捉えていない
FAIL_AREA_MAX = 0.92  # これ超なら、背景まで取っている
CHECK_AREA_MIN = 0.10
CHECK_AREA_MAX = 0.75
CHECK_BOX_SCORE = 0.40
# 画像の端 1px のうち、マスクが触れている割合がこれ以上なら「見切れている」
BORDER_TOUCH = 0.01
# 後処理のあとに残る連結成分のうち、最大のこの割合以上のものを「別の塊」として数える
COMPONENT_RATIO = 0.05


@dataclass
class Quality:
    area_ratio: float
    touches_border: bool
    components: int
    box_score: float

    def grade(self) -> tuple[str, list[str]]:
        return grade(self)


@dataclass
class Result:
    rgba: Image.Image | None
    quality: Quality | None
    status: str
    reasons: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "quality": asdict(self.quality) if self.quality else None,
            "seconds": round(self.seconds, 2),
        }


# ---------------------------------------------------------------- 後処理（モデルを使わない）


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n - 1, labels, stats[1:, cv2.CC_STAT_AREA]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """小さな穴を埋め、孤立した粒を消す。bool のマスクを返す。"""
    mask = mask.astype(bool)
    total = mask.size
    # 穴埋め: 背景の連結成分のうち、小さいものを前景にする（画像の端に触れる背景は残す）
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        on_edge = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not on_edge and area < HOLE_RATIO * total:
            out[labels == i] = True
    # 粒の除去
    count, labels, areas = _components(out)
    if count == 0:
        return out
    keep = [i + 1 for i, a in enumerate(areas) if a >= SPECK_RATIO * out.sum()]
    if not keep:
        keep = [int(np.argmax(areas)) + 1]
    return np.isin(labels, keep)


def soften_edge(mask: np.ndarray) -> np.ndarray:
    """縁を1〜2px 内側に寄せてから軽くぼかした、アルファ（uint8）。背景の色のにじみを減らす。"""
    px = max(1, round(min(mask.shape) * ERODE_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    eroded = cv2.erode(mask.astype(np.uint8) * 255, kernel)
    return cv2.GaussianBlur(eroded, (0, 0), FEATHER_SIGMA)


def to_rgba(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """元の画素はそのまま、アルファだけを付ける。"""
    rgba = image.convert("RGB").copy()
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def crop_to_content(rgba: Image.Image, margin: float = CROP_MARGIN) -> Image.Image:
    """透明でない範囲に余白を付けて切り詰める。全部透明ならそのまま返す。"""
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return rgba
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = round(max(x1 - x0, y1 - y0) * margin)
    box = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(rgba.width, x1 + pad),
        min(rgba.height, y1 + pad),
    )
    return rgba.crop(box)


def measure(mask: np.ndarray, box_score: float) -> Quality:
    h, w = mask.shape
    edges = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    count, _, areas = _components(mask)
    big = int((areas >= COMPONENT_RATIO * areas.max()).sum()) if count else 0
    return Quality(
        area_ratio=round(float(mask.mean()), 4),
        touches_border=bool(edges.mean() >= BORDER_TOUCH),
        components=big,
        box_score=round(float(box_score), 3),
    )


def grade(q: Quality) -> tuple[str, list[str]]:
    """ok / check / fail と理由（日本語）。fail なら切り抜かず写真のまま使う。"""
    fails, checks = [], []
    if q.box_score < BOX_MIN_SCORE:
        fails.append(f"服の検出の確信度が低い（{q.box_score:.2f}）")
    if q.area_ratio < FAIL_AREA_MIN:
        fails.append(f"マスクが小さすぎる（画像の{q.area_ratio:.0%}）")
    if q.area_ratio > FAIL_AREA_MAX:
        fails.append(f"マスクが大きすぎる（画像の{q.area_ratio:.0%}。背景まで含む）")
    if fails:
        return "fail", fails
    if q.touches_border:
        checks.append("服が画像の端で見切れている")
    if q.components > 1:
        checks.append(f"マスクが{q.components}つの塊に分かれている")
    if q.area_ratio < CHECK_AREA_MIN:
        checks.append(f"マスクが小さい（画像の{q.area_ratio:.0%}）")
    if q.area_ratio > CHECK_AREA_MAX:
        checks.append(f"マスクが大きい（画像の{q.area_ratio:.0%}）")
    if q.box_score < CHECK_BOX_SCORE:
        checks.append(f"検出の確信度が低め（{q.box_score:.2f}）")
    return ("check", checks) if checks else ("ok", [])


def pick_box(boxes: list[list[float]], scores: list[float], size: tuple[int, int]):
    """検出した箱から1つ選ぶ。確信度 × 面積が最大のもの（袖だけの小さな箱より全体を選ぶ）。
    採用の下限を超えるものが無ければ、確信度が最大のものを返す（判定側で fail にする）。"""
    if not boxes:
        return None
    w, h = size
    area = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) / (w * h) for b in boxes]
    ok = [i for i, s in enumerate(scores) if s >= BOX_MIN_SCORE]
    if not ok:
        best = max(range(len(boxes)), key=lambda i: scores[i])
        return boxes[best], scores[best]
    best = max(ok, key=lambda i: scores[i] * (area[i] ** 0.5))
    return boxes[best], scores[best]


def prompt_for(product_type: str | None) -> str:
    base = PROMPTS.get(product_type or "")
    return f"{base} . {GENERIC_PROMPT}" if base else GENERIC_PROMPT


def finish(image: Image.Image, mask: np.ndarray, box_score: float) -> Result:
    """生のマスクから、後処理・判定・RGBA まで（モデルを使わない部分）。"""
    mask = clean_mask(mask)
    q = measure(mask, box_score)
    status, reasons = grade(q)
    if status == "fail":
        return Result(None, q, status, reasons)
    rgba = crop_to_content(to_rgba(image, soften_edge(mask)))
    return Result(rgba, q, status, reasons)


# ---------------------------------------------------------------- モデル


class Cutter:
    """Grounding DINO + SAM2。GPU（ROCm）で動かす。モデルは最初に使うときに読む（オフライン）。"""

    def __init__(self, dino_path: str | None = None, sam_path: str | None = None) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        d = str(Path(dino_path or DINO_PATH).expanduser())
        s = str(Path(sam_path or SAM_PATH).expanduser())
        self.dino_proc = AutoProcessor.from_pretrained(d, local_files_only=True)
        self.dino = (
            AutoModelForZeroShotObjectDetection.from_pretrained(d, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.sam_proc = Sam2Processor.from_pretrained(s, local_files_only=True)
        self.sam = Sam2Model.from_pretrained(s, local_files_only=True).to(self.device).eval()

    def detect(self, image: Image.Image, product_type: str | None):
        text = prompt_for(product_type)
        inputs = self.dino_proc(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out,
            inputs["input_ids"],
            threshold=0.1,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = res["boxes"].cpu().tolist()
        scores = res["scores"].cpu().tolist()
        return pick_box(boxes, scores, image.size)

    def segment(self, image: Image.Image, box: list[float]) -> np.ndarray:
        inputs = self.sam_proc(
            images=image, input_boxes=[[[float(v) for v in box]]], return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
        return masks[0][0, 0].numpy().astype(bool)

    def cut(self, image: Image.Image, product_type: str | None = None) -> Result:
        import time

        t0 = time.time()
        image = image.convert("RGB")
        picked = self.detect(image, product_type)
        if picked is None:
            return Result(None, None, "fail", ["服を検出できなかった"], time.time() - t0)
        box, score = picked
        result = finish(image, self.segment(image, box), score)
        result.seconds = time.time() - t0
        return result
