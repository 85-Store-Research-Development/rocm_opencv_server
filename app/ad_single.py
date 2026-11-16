"""1点ものの広告（ブリーフの `layout: "single"`）の組み。出品中の商品から選んだ1点のための広告。

誌面のように、1つの物を大きく載せて、まわりに手書きのメモと引き出し線で特徴を書き込む。
  1. 主役（商品の切り抜き）を大きく置く（置き方は app/ad_compose.py の build_single_layout）
  2. 価格の札（手描きの縁の丸いシール）と「1点もの」の小さなシール
  3. 手書きのメモ（callouts）を服のまわりの空きに置き、服の輪郭の近い点へ手描きのゆるい線を引く
     （ブランド → 襟元、サイズ → 裾、素材 → 身頃 …。部位の検出はせず、輪郭の上・中・下と左右の目安）
  4. 着画が無いとき、2枚目以降の写真から細部の写真（タグ・生地のアップなど）を
     白いふちの小さな写真として添える

このモジュールはモデルも Chrome も使わず、ad_compose にも依存しない
（マスクと文字幅の測り方を受け取って動く）。
**商品の画素は描き直さない**。細部の写真は切り抜かず四角のまま、拡大縮小・回転・影だけ。
置けないメモは後ろの物から省き、省いたことを結果に残す。
検査は ad_compose.check_layout から check_plan を呼ぶ。
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import ad_season, ad_worn
from app.ad_palette import Palette, contrast_ratio

# ---------------------------------------------------------------- 定数

LAYOUT_NAME = "single"
MAX_CALLOUTS = 6
MIN_CALLOUTS = 2  # 主役の大きさを調整してでも置きたいメモの数
CALLOUT_MAX_LEN = 24
PRICE_MAX_LEN = 16
STAMP_MAX_LEN = 10
KINDS = ("brand", "era", "made_in", "fabric", "size", "condition")
KIND_OTHER = "other"
BRAND_ACCENT = True  # ブランドのメモだけアクセント色にして、目に入る順を作る

HERO_H_RATIO = {"4:5": 0.50, "1:1": 0.52, "9:16": 0.38}  # 画面の高さに対する主役の高さ
HERO_W_RATIO = 0.56  # 画面の幅に対する主役の幅の上限
HERO_W_RATIO_WORN = 0.44  # 着画を横に並べるとき
HERO_H_OF_REGION = 0.9
HERO_ANGLE = (1.5, 5.0)  # 傾き（±度）
WORN_REL_H = 0.9  # 着画の高さ ÷ 主役の高さ（少し小さいか同じくらい）
WORN_REL_H_STACK = 0.78  # 縦長（9:16）で主役の下に置くとき、残りの高さに対する割合
WORN_ANGLE = (1.0, 3.5)
HERO_GAP = 18  # 主役と着画の間

MEMO_STEPS = (48, 42, 36)  # メモの文字の大きさ（幅 1080 のとき。大きいほうから）
MEMO_MIN_FS = 32  # これ未満は読めない（幅 1080 のとき）
MEMO_ANGLE = 3.0  # メモの傾き（±度）
MEMO_PAD = 8
MEMO_GAP = 12  # メモと、ほかの物の間
MEMO_LINE_H = 1.18
GRID_STEP = 18
TOP_K = 48  # 費用の小さい順に、線まで含めて確かめる候補の数
LINE_STROKE = 3.6
LINE_CHECK_W = 9  # 線が通ってはいけない幅（ほかの線・文字・商品）
LINE_CROSS_W = 5  # 検査で線どうしの交差を測る太さ
LINE_TAIL = 30  # 線の終わりのこの長さ（px）だけは主役の上に乗ってよい（輪郭の中の印のため）
END_INSET = 10  # 線の終わりの印を、輪郭からこれだけ服の中へ入れる
RING_R = 8
LINE_MIN = 60
LINE_MAX_RATIO = 0.42  # 線の長さの上限（画面の幅に対する割合）
TARGET_PULL = 1.4  # メモが部位の目安から遠い点へ引かれるときの費用

BADGE_R_RATIO = 0.105  # 価格の札の半径（画面の幅に対する割合）
BADGE_TEXT_MAX = 70
BADGE_TEXT_MIN = 34
BADGE_HERO_OVERLAP = 0.12  # 札が主役に重なってよい割合（札の面積に対して。端にかかるだけ）
STAMP_SIZE = 34
STAMP_HERO_OVERLAP = 0.12
PHOTO_W_RATIO = 0.205  # 細部の写真の幅（画面の幅に対する割合）
PHOTO_HERO_OVERLAP = 0.18  # 主役と重なってよいのは端だけ
PHOTO_MAX = 2  # 縦長（9:16）だけ。ほかは1枚（似た写真が並ぶと散らかる）
PHOTO_ANGLE = (3.0, 7.0)
PHOTO_PAD = 12
PHOTO_PAD_BOTTOM = 34
PHOTO_MAX_SIDE = 760  # 保存する細部の写真の長辺

TOLERANCE_PX = 40
DOODLE_LIMIT = 3  # 1点ものでは落書きを少なめにする
DOODLE_SIZE_SCALE = 0.8

# 細部の写真の選び方（簡単な規則）
CLOSEUP_TOUCH = (
    0.45  # 画像の四辺の縁のうち、写っている物（地と違う色）が占める割合がこれ以上なら「アップ」
)
CLOSEUP_DIFF = 30  # 地の色との差（0〜255）がこれ以上を「物」とする
PERSON_SKIP = 0.5  # この確信度以上の人が写る写真は細部に使わない（着画）

# 部位の目安（主役の外接矩形に対する割合）。(x, y)。x は None なら、メモのある側の端（左 0・右 1）
TARGETS: dict[str, tuple[float | None, float]] = {
    "brand": (0.5, 0.03),  # 襟元
    "made_in": (0.42, 0.10),  # 襟の内側のタグのあたり
    "era": (None, 0.30),  # 胸・肩
    "fabric": (None, 0.52),  # 身頃
    "size": (None, 0.92),  # 裾・袖
    "condition": (None, 0.74),
    KIND_OTHER: (None, 0.5),
}
ARROW_KINDS = ("size",)  # 先を矢じりにする（ほかは小さな丸）

# ---------------------------------------------------------------- ブリーフの single


@dataclass
class Callout:
    kind: str
    text: str


@dataclass
class Single:
    price_text: str = ""
    stamp: str | None = None
    callouts: list[Callout] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # 読み込みで省いたこと


def parse_single(raw) -> tuple[Single, list[str]]:
    """ブリーフの single。(読めた値, 問題の一覧)。無い・空は空の Single（メモも札も無しで組む）。
    メモの kind が知らない値なら other として扱う。MAX_CALLOUTS を超える分は後ろから省く。"""
    problems: list[str] = []
    out = Single()
    if raw is None:
        return out, problems
    if not isinstance(raw, dict):
        return out, ["single はオブジェクトにしてください"]
    price = raw.get("price_text")
    if price not in (None, ""):
        if not isinstance(price, str) or len(price) > PRICE_MAX_LEN:
            problems.append(f"single.price_text は {PRICE_MAX_LEN} 字までの文字列にしてください")
        else:
            out.price_text = price.strip()
    stamp = raw.get("stamp")
    if stamp not in (None, ""):
        if not isinstance(stamp, str) or len(stamp) > STAMP_MAX_LEN:
            problems.append(f"single.stamp は {STAMP_MAX_LEN} 字までの文字列にしてください")
        else:
            out.stamp = stamp.strip() or None
    cs = raw.get("callouts")
    if cs is not None and not isinstance(cs, list):
        problems.append("single.callouts は配列にしてください")
        cs = []
    for i, c in enumerate(cs or []):
        if not isinstance(c, dict):
            problems.append(f"single.callouts[{i}] はオブジェクトにしてください")
            continue
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            continue  # 文が無いメモは置かない
        text = text.strip()
        if len(text) > CALLOUT_MAX_LEN:
            problems.append(f"single.callouts[{i}].text は {CALLOUT_MAX_LEN} 字までです")
            continue
        kind = c.get("kind")
        out.callouts.append(Callout(kind if kind in KINDS else KIND_OTHER, text))
    if len(out.callouts) > MAX_CALLOUTS:
        for c in out.callouts[MAX_CALLOUTS:]:
            out.notes.append(f"メモが多すぎて省いた（{c.kind}: {c.text}）")
        out.callouts = out.callouts[:MAX_CALLOUTS]
    return out, problems


# ---------------------------------------------------------------- 幾何の部品


def rot_matrix(angle_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[c, -s], [s, c]])


def rect_poly(cx: float, cy: float, w: float, h: float, angle: float = 0.0, pad: float = 0.0):
    half = np.array([w / 2 + pad, h / 2 + pad])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    return np.array([cx, cy]) + corners @ rot_matrix(angle).T


def _bbox(poly: np.ndarray, shape: tuple[int, int]):
    h, w = shape
    x0, x1 = int(max(0, math.floor(poly[:, 0].min()))), int(min(w, math.ceil(poly[:, 0].max()) + 1))
    y0, y1 = int(max(0, math.floor(poly[:, 1].min()))), int(min(h, math.ceil(poly[:, 1].max()) + 1))
    return x0, y0, x1, y1


def poly_overlap(poly: np.ndarray, big: np.ndarray) -> int:
    """多角形と、画面と同じ大きさの bool マスクが重なる画素数。"""
    x0, y0, x1, y1 = _bbox(poly, big.shape)
    if x1 <= x0 or y1 <= y0:
        return 0
    loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(loc, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    return int(np.count_nonzero(loc.astype(bool) & big[y0:y1, x0:x1]))


def poly_area_px(poly: np.ndarray) -> int:
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(loc, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    return int(np.count_nonzero(loc))


def paint_poly(poly: np.ndarray, big: np.ndarray) -> None:
    x0, y0, x1, y1 = _bbox(poly, big.shape)
    if x1 <= x0 or y1 <= y0:
        return
    loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(loc, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    big[y0:y1, x0:x1] |= loc.astype(bool)


def poly_outside_rect(poly: np.ndarray, rect: tuple[float, float, float, float], slack=0.0) -> bool:
    x0, y0, x1, y1 = rect
    return bool(
        poly[:, 0].min() < x0 - slack
        or poly[:, 0].max() > x1 + slack
        or poly[:, 1].min() < y0 - slack
        or poly[:, 1].max() > y1 + slack
    )


def line_mask(pts: np.ndarray, shape: tuple[int, int], width: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, width)
    return m.astype(bool)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    k = 2 * r + 1
    return (
        cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    )


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 1.0, 1.0
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def contour_points(mask: np.ndarray, step: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """主役の輪郭（いちばん大きい外周）を等間隔にとった点と、外向きの単位法線。"""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    n = max(8, int(d[-1] // step))
    s = np.linspace(0, d[-1], n, endpoint=False)
    pts = np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], axis=1)
    t = np.roll(pts, -3, axis=0) - np.roll(pts, 3, axis=0)
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    h, w = mask.shape
    probe = np.rint(pts + nrm * 8).astype(int)
    px = np.clip(probe[:, 0], 0, w - 1)
    py = np.clip(probe[:, 1], 0, h - 1)
    inside = mask[py, px]
    nrm[inside] *= -1  # 外向きへ
    return pts, nrm


def leader_curve(a: np.ndarray, e: np.ndarray, rng: random.Random, n: int = 44) -> np.ndarray:
    """メモの端 a から印 e への、少し曲がった手描きの線（3次ベジェ + ごく小さなゆらぎ）。
    定規で引いたように見えないよう、曲がりの向き・大きさと、中ほどの膨らみを種で変える。"""
    d = e - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return np.stack([a, e])
    u = d / length
    nr = np.array([-u[1], u[0]])
    b1 = rng.choice((-1, 1)) * rng.uniform(0.06, 0.16)
    b2 = b1 * rng.uniform(-0.5, 0.9)
    c1 = a + d * 0.30 + nr * length * b1
    c2 = a + d * 0.72 + nr * length * b2
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * e
    ph = rng.uniform(0, 2 * math.pi)
    wob = 1.5 * np.sin(2 * math.pi * 2.3 * t[:, 0] + ph) * np.sin(math.pi * t[:, 0])
    return pts + nr * wob[:, None]


def path_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def body_part(pts: np.ndarray, tail: float = LINE_TAIL) -> np.ndarray:
    """線のうち、終わりの tail（px）を除いた部分（主役の上を通ってはいけない所）。"""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    remain = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]
    keep = remain > tail
    return pts[keep] if keep.sum() >= 2 else pts[:2]


# ---------------------------------------------------------------- 計画（置いた物の記録）


@dataclass
class Memo:
    kind: str
    text: str
    lines: list[str]
    fs: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    line: list  # 引き出し線の点列（[[x, y], ...]。メモの端 → 印）
    end: list  # 印の位置
    end_style: str  # ring / arrow
    accent: bool = False

    def polygon(self, pad: float = 0.0) -> np.ndarray:
        return rect_poly(self.cx, self.cy, self.w, self.h, self.angle, pad)

    def line_pts(self) -> np.ndarray:
        return np.asarray(self.line, float)


@dataclass
class Badge:
    text: str
    cx: float
    cy: float
    r: float
    size: int
    angle: float
    style: str  # scallop / star
    seed_text: str = ""

    def polygon(self) -> np.ndarray:
        return rect_poly(self.cx, self.cy, 2 * self.r, 2 * self.r, 0.0)

    def disc_mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, np.uint8)
        cv2.circle(m, (round(self.cx), round(self.cy)), round(self.r), 1, -1)
        return m.astype(bool)


@dataclass
class StampTag:
    text: str
    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def polygon(self, pad: float = 0.0) -> np.ndarray:
        return rect_poly(self.cx, self.cy, self.w, self.h, self.angle, pad)


@dataclass
class Polaroid:
    path: str
    cx: float
    cy: float
    w: float  # 白いふちを含む
    h: float
    angle: float
    photo: float  # 写真の窓（正方形）の一辺
    src: str = ""
    index: int = -1

    def polygon(self, pad: float = 0.0) -> np.ndarray:
        return rect_poly(self.cx, self.cy, self.w, self.h, self.angle, pad)


@dataclass
class DetailPhoto:
    """細部の写真の候補（保存済みの JPEG）。"""

    path: Path
    url: str
    index: int
    score: float
    reason: str


@dataclass
class SinglePlan:
    memos: list[Memo] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # 置けなかったメモ {kind, text, reason}
    wanted: int = 0  # ブリーフのメモの数
    badge: Badge | None = None
    stamp: StampTag | None = None
    photos: list[Polaroid] = field(default_factory=list)
    price_text: str = ""
    stamp_text: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["memos"] = [{**asdict(m), "line": np.round(m.line_pts(), 1).tolist()} for m in self.memos]
        return d


# ---------------------------------------------------------------- メモの文字の組み


def _script(ch: str) -> str:
    return "latin" if ord(ch) < 0x250 or ch in "¥%％" else "cjk"


def split_lines(text: str, ms, fs: int) -> list[list[str]]:
    """メモの行の分け方の候補（1行・2行）。2行は、空白か、和文と英数字の境か、真ん中で割る。"""
    out = [[text]]
    if len(text) < 4:
        return out
    cuts = [i for i, ch in enumerate(text) if ch == " " and 0 < i < len(text) - 1]
    cuts = [(i, i + 1) for i in cuts]
    for i in range(1, len(text)):
        if _script(text[i - 1]) != _script(text[i]) and text[i] != " " and text[i - 1] != " ":
            cuts.append((i, i))
    if not cuts:
        mid = len(text) // 2
        cuts = [(mid, mid)]

    def cost(c):
        a, b = text[: c[0]].strip(), text[c[1] :].strip()
        if len(a) < 2 or len(b) < 2:  # 1字だけ余る割り方は読みにくい（USA / 製）
            return 1e9
        return abs(ms.width("hand", a, fs) - ms.width("hand", b, fs))

    best = min(cuts, key=cost)
    a, b = text[: best[0]].strip(), text[best[1] :].strip()
    if len(a) >= 2 and len(b) >= 2:
        out.append([a, b])
    return out


def memo_size(lines: list[str], ms, fs: int) -> tuple[float, float]:
    w = max(ms.width("hand", ln, fs) for ln in lines)
    return w * 1.04, len(lines) * fs * MEMO_LINE_H


def _integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8))


def _rect_sums(ii: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def desired_point(kind: str, bbox, side: int) -> np.ndarray:
    ux, uy = TARGETS.get(kind, TARGETS[KIND_OTHER])
    bx0, by0, bx1, by1 = bbox
    if ux is None:
        ux = 0.0 if side < 0 else 1.0
    return np.array([bx0 + ux * (bx1 - bx0), by0 + uy * (by1 - by0)])


def box_anchor(m_center, angle, w, h, target) -> np.ndarray:
    """メモの箱の縁のうち、印の方向にある点（箱の少し外側）。"""
    r = rot_matrix(angle)
    local = r.T @ (np.asarray(target, float) - m_center)
    half = np.array([w / 2 + MEMO_PAD, h / 2 + MEMO_PAD])
    if abs(local[0]) < 1e-6 and abs(local[1]) < 1e-6:
        local = np.array([1.0, 0.0])
    k = min(half[0] / max(abs(local[0]), 1e-9), half[1] / max(abs(local[1]), 1e-9))
    edge = local * k
    edge = edge + edge / max(np.linalg.norm(edge), 1e-9) * 6  # 線の始まりをメモから少し離す
    return m_center + r @ edge


def place_memos(
    callouts: list[Callout],
    *,
    hero_mask: np.ndarray,
    blocked: np.ndarray,
    safe: tuple[float, float, float, float],
    ms,
    rng: random.Random,
    scale: float = 1.0,
) -> tuple[list[Memo], list[dict], np.ndarray]:
    """メモを置く。(置けたメモ, 省いたメモ, メモ・線を書き足した占有マップ)。

    大きい文字から順に、全部置けるか試し、置けなければ小さい段へ。全部が置ける段が無ければ、
    置けた数が最も多い段（同数なら大きい段）を使い、置けなかった物は後ろ（優先度の低い物）から省く。
    blocked は、主役以外で重なってはいけない物（見出し・店名・着画・札・写真など）。"""
    H, W = hero_mask.shape
    best = None
    for fs0 in MEMO_STEPS:
        fs = max(round(fs0 * scale), round(MEMO_MIN_FS * scale))
        run_rng = random.Random(rng.random())
        memos, dropped, occ = _place_all(callouts, hero_mask, blocked, safe, ms, run_rng, fs)
        key = (len(memos), fs)
        if best is None or key > best[0]:
            best = (key, memos, dropped, occ)
        if not dropped:
            break
    _, memos, dropped, occ = best
    return memos, dropped, occ


def _place_all(callouts, hero_mask, blocked, safe, ms, rng, fs):
    H, W = hero_mask.shape
    occ = blocked.copy()  # 主役以外の占有 + 置いたメモと線
    lines = np.zeros((H, W), bool)
    memos: list[Memo] = []
    dropped: list[dict] = []
    bbox = mask_bbox(hero_mask)
    hcx = (bbox[0] + bbox[2]) / 2
    pts, nrm = contour_points(hero_mask)
    hero_pad = dilate(hero_mask, MEMO_GAP + 4)
    for c in callouts:
        placed = None
        # 左右に振り分ける: 置いた数の少ない側を好む（費用に足す）
        for lines_txt in split_lines(c.text, ms, fs):
            w, h = memo_size(lines_txt, ms, fs)
            placed = _search(
                c, lines_txt, w, h, fs, pts, nrm, bbox, hcx, hero_mask, hero_pad, occ, lines,
                safe, rng, memos, W, H,
            )  # fmt: skip
            if placed:
                break
        if placed is None:
            dropped.append({"kind": c.kind, "text": c.text, "reason": "置ける空きが無い"})
            continue
        memo, lmask = placed
        paint_poly(memo.polygon(MEMO_PAD + MEMO_GAP // 2), occ)
        lines |= lmask
        occ |= dilate(lmask, 2)
        memos.append(memo)
    return memos, dropped, occ


def _search(c, lines_txt, w, h, fs, pts, nrm, bbox, hcx, hero_mask, hero_pad, occ, lines, safe,
            rng, memos, W, H):  # fmt: skip
    """1つのメモの置き場所を探す。費用の小さい順に、線まで含めて確かめて、最初に通った物を返す。"""
    angle = rng.uniform(-MEMO_ANGLE, MEMO_ANGLE)
    ca, sa = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    aw = (w + 2 * MEMO_PAD) * ca + (h + 2 * MEMO_PAD) * sa
    ah = (w + 2 * MEMO_PAD) * sa + (h + 2 * MEMO_PAD) * ca
    sx0, sy0, sx1, sy1 = safe
    xs = np.arange(sx0 + aw / 2, sx1 - aw / 2 + 1, GRID_STEP)
    ys = np.arange(sy0 + ah / 2, sy1 - ah / 2 + 1, GRID_STEP)
    if len(xs) == 0 or len(ys) == 0 or len(pts) == 0:
        return None
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    blocked = occ | hero_pad
    ii = _integral(dilate(blocked, MEMO_GAP // 2))
    x0 = np.clip(np.floor(gx - aw / 2).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil(gx + aw / 2).astype(int), 1, W)
    y0 = np.clip(np.floor(gy - ah / 2).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil(gy + ah / 2).astype(int), 1, H)
    ok = _rect_sums(ii, x0, y0, x1, y1) == 0
    if not ok.any():
        return None
    gx, gy = gx[ok], gy[ok]
    side = np.where(gx < hcx, -1, 1)
    left_n = sum(1 for m in memos if m.cx < hcx)
    right_n = len(memos) - left_n
    # 印の候補: メモが輪郭の外側にあるときだけ（外向きの法線の向きにメモがある）
    d = gx[:, None] - pts[None, :, 0], gy[:, None] - pts[None, :, 1]
    dist = np.hypot(d[0], d[1])
    facing = (d[0] * nrm[None, :, 0] + d[1] * nrm[None, :, 1]) > 0.25 * dist
    des_l, des_r = desired_point(c.kind, bbox, -1), desired_point(c.kind, bbox, 1)
    des = np.where(side[:, None] < 0, des_l[None, :], des_r[None, :])  # (n, 2)
    pull = np.hypot(pts[None, :, 0] - des[:, 0:1], pts[None, :, 1] - des[:, 1:2])
    cost_pt = dist + TARGET_PULL * pull
    cost_pt = np.where(facing & (dist > LINE_MIN) & (dist < LINE_MAX_RATIO * W), cost_pt, 1e9)
    j = np.argmin(cost_pt, axis=1)
    base = cost_pt[np.arange(len(gx)), j]
    balance = np.where(side < 0, 70.0 * (left_n - right_n), 70.0 * (right_n - left_n))
    cost = base + np.maximum(balance, 0) + np.array([rng.uniform(0, 40) for _ in gx])
    order = np.argsort(cost)
    tried = 0
    for k in order:
        if cost[k] >= 1e8 or tried >= TOP_K:
            break
        tried += 1
        center = np.array([gx[k], gy[k]])
        poly = rect_poly(center[0], center[1], w, h, angle, MEMO_PAD)
        if poly_outside_rect(poly, safe) or poly_overlap(poly, blocked):
            continue
        p = pts[j[k]]
        e = p - nrm[j[k]] * END_INSET
        a = box_anchor(center, angle, w, h, e)
        curve = leader_curve(a, e, rng)
        lm = line_mask(body_part(curve), (H, W), LINE_CHECK_W)
        if np.count_nonzero(lm & hero_mask):
            continue  # 主役の上を通る（終わりの印のところだけは許す）
        own = rect_poly(center[0], center[1], w, h, angle, MEMO_PAD + 2)
        # 自分のメモの箱から出る所は占有ではないので、箱の中は消して測る
        lm_check = lm.copy()
        x0b, y0b, x1b, y1b = _bbox(own, lm.shape)
        sub = np.zeros((y1b - y0b, x1b - x0b), np.uint8)
        cv2.fillPoly(sub, [np.round(own - [x0b, y0b]).astype(np.int32)], 1)
        lm_check[y0b:y1b, x0b:x1b] &= ~sub.astype(bool)
        if np.count_nonzero(lm_check & occ) or np.count_nonzero(lm_check & lines):
            continue
        cp = curve
        if (
            cp[:, 0].min() < 6
            or cp[:, 0].max() > W - 6
            or cp[:, 1].min() < 6
            or cp[:, 1].max() > H - 6
        ):
            continue
        memo = Memo(
            c.kind,
            c.text,
            lines_txt,
            fs,
            float(center[0]),
            float(center[1]),
            float(w),
            float(h),
            float(angle),
            np.round(curve, 1).tolist(),
            [round(float(e[0]), 1), round(float(e[1]), 1)],
            "arrow" if c.kind in ARROW_KINDS else "ring",
            accent=BRAND_ACCENT and c.kind == "brand",
        )
        return memo, line_mask(curve, (H, W), LINE_CHECK_W)
    return None


# ---------------------------------------------------------------- 札・シール・細部の写真


def fit_price(text: str, r: float, ms) -> int:
    """札の中に収まる文字の大きさ（幅が札の直径の 0.78 以内）。"""
    size = BADGE_TEXT_MAX
    while size > BADGE_TEXT_MIN and ms.width("head", text, size) > 1.56 * r * 0.95:
        size -= 2
    return size


def place_badge(
    text: str,
    *,
    hero_mask: np.ndarray,
    blocked: np.ndarray,
    safe: tuple[float, float, float, float],
    ms,
    rng: random.Random,
    width: int,
) -> Badge | None:
    """価格の札。主役の外接矩形の四隅のあたりから、空いていて主役に重なる割合が小さい所を選ぶ
    （右下・右上を少し好む。種で上位から選ぶ）。置けなければ None。"""
    H, W = hero_mask.shape
    r = BADGE_R_RATIO * width
    size = fit_price(text, r, ms)
    bx0, by0, bx1, by1 = mask_bbox(hero_mask)
    sx0, sy0, sx1, sy1 = safe
    cands = []
    pref = {(1, 1): 0.0, (1, -1): 8.0, (-1, 1): 14.0, (-1, -1): 22.0}
    for (qx, qy), bias in pref.items():
        for ix in (0.0, 0.12, 0.25):
            for iy in (0.0, 0.12, 0.25):
                cx = (bx1 - ix * (bx1 - bx0)) if qx > 0 else (bx0 + ix * (bx1 - bx0))
                cy = (by1 - iy * (by1 - by0)) if qy > 0 else (by0 + iy * (by1 - by0))
                cx += qx * 0.35 * r
                cy += qy * 0.35 * r
                cx = min(max(cx, sx0 + r), sx1 - r)
                cy = min(max(cy, sy0 + r), sy1 - r)
                cands.append((cx, cy, bias + 30 * (ix + iy)))
    scored = []
    area = math.pi * r * r
    for cx, cy, bias in cands:
        disc = np.zeros((H, W), np.uint8)
        cv2.circle(disc, (round(cx), round(cy)), round(r), 1, -1)
        d = disc.astype(bool)
        if np.count_nonzero(d & blocked) > TOLERANCE_PX:
            continue
        ov = np.count_nonzero(d & hero_mask) / area
        if ov > BADGE_HERO_OVERLAP:
            continue
        scored.append((bias + 400 * ov + rng.uniform(0, 12), cx, cy))
    if not scored:
        return None
    scored.sort()
    _, cx, cy = scored[rng.randrange(min(2, len(scored)))]
    return Badge(
        text,
        float(cx),
        float(cy),
        float(r),
        size,
        rng.uniform(-8, 8),
        rng.choice(("scallop", "star")),
        text,
    )


def place_stamp_tag(
    text: str,
    badge: Badge | None,
    *,
    hero_mask: np.ndarray,
    blocked: np.ndarray,
    safe: tuple[float, float, float, float],
    ms,
    rng: random.Random,
    width: int,
) -> StampTag | None:
    """「1点もの」のシール。価格の札の縁に寄せて（札があれば）置く。札が無ければ主役の肩のあたり。"""
    H, W = hero_mask.shape
    size = round(STAMP_SIZE * width / 1080)
    w = ms.width("hand", text, size) * 1.12 + 26
    h = size * 1.2 + 14
    angle = rng.choice((-1, 1)) * rng.uniform(4, 9)
    if badge is not None:
        base = [
            (
                badge.cx + math.cos(a) * (badge.r + w * 0.3),
                badge.cy + math.sin(a) * (badge.r + h * 0.5),
            )
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ]
    else:
        bx0, by0, bx1, by1 = mask_bbox(hero_mask)
        base = [(bx1, by0), (bx0, by0), (bx1, by1), (bx0, by1)]
    best = None
    area = w * h
    for cx, cy in base:
        poly = rect_poly(cx, cy, w, h, angle)
        if poly_outside_rect(poly, safe):
            continue
        if poly_overlap(poly, blocked) > TOLERANCE_PX:
            continue
        ov = poly_overlap(poly, hero_mask) / area
        if ov > STAMP_HERO_OVERLAP:
            continue
        if badge is not None:
            ob = poly_overlap(poly, badge.disc_mask((H, W))) / area
            if ob > 0.22:
                continue  # 札にほとんど隠れる
            d = math.hypot(cx - badge.cx, cy - badge.cy)
        else:
            d = 0.0
        cost = d + 300 * ov + rng.uniform(0, 10)
        if best is None or cost < best[0]:
            best = (cost, cx, cy)
    if best is None:
        return None
    return StampTag(text, float(best[1]), float(best[2]), float(w), float(h), float(angle))


def place_photos(
    photos: list[DetailPhoto],
    *,
    hero_mask: np.ndarray,
    blocked: np.ndarray,
    safe: tuple[float, float, float, float],
    rng: random.Random,
    width: int,
    limit: int = PHOTO_MAX,
) -> list[Polaroid]:
    """細部の写真（白いふちの小さな写真）。主役と重なってよいのは端だけ。入らなければ置かない
    （1枚目が置けなければ2枚目も置かない。置けた分だけ返す）。"""
    H, W = hero_mask.shape
    out: list[Polaroid] = []
    occ = blocked.copy()
    pw = PHOTO_W_RATIO * width
    photo = pw - 2 * PHOTO_PAD
    ph = photo + PHOTO_PAD + PHOTO_PAD_BOTTOM
    sx0, sy0, sx1, sy1 = safe
    bx0, by0, bx1, by1 = mask_bbox(hero_mask)
    hcx, hcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    for dp in photos[: min(limit, PHOTO_MAX)]:
        angle = rng.choice((-1, 1)) * rng.uniform(*PHOTO_ANGLE)
        area = pw * ph
        best = None
        for cx in np.arange(sx0 + pw / 2, sx1 - pw / 2 + 1, 24):
            for cy in np.arange(sy0 + ph / 2, sy1 - ph / 2 + 1, 24):
                poly = rect_poly(cx, cy, pw, ph, angle)
                if poly_overlap(poly, occ):
                    continue
                ov = poly_overlap(poly, hero_mask) / area
                if ov > PHOTO_HERO_OVERLAP:
                    continue
                # 主役のそばで、下側（足元）を少し好む。端が主役に触れる配置を好む
                d = max(0.0, math.hypot(cx - hcx, cy - hcy) - 0.5 * (bx1 - bx0))
                cost = d + (0 if cy > hcy else 60) - 80 * min(ov, 0.1) / 0.1 + rng.uniform(0, 30)
                if best is None or cost < best[0]:
                    best = (cost, cx, cy)
        if best is None:
            break
        _, cx, cy = best
        pol = Polaroid(str(dp.path), float(cx), float(cy), float(pw), float(ph), float(angle),
                       float(photo), dp.url, dp.index)  # fmt: skip
        paint_poly(pol.polygon(MEMO_GAP // 2), occ)
        out.append(pol)
    return out


# ---------------------------------------------------------------- 細部の写真の選び方


def closeup_score(img: Image.Image) -> float:
    """写真が「アップ」か（服が画面いっぱい）の目安 0〜1。四辺の縁のうち、地（縁の中央値の色）と
    違う色の占める割合。全身の写真は縁が地なので小さく、タグ・生地のアップは縁まで物が写るので大きい。"""
    im = img.convert("RGB")
    im.thumbnail((96, 96))
    a = np.asarray(im).astype(np.int16)
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    bg = np.median(ring, axis=0)
    fg = np.abs(a - bg).max(axis=2) >= CLOSEUP_DIFF
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    return float(border.mean())


def pick_detail_indices(
    scores: dict[int, float], main_index: int, skip: set[int], limit: int = PHOTO_MAX
) -> list[tuple[int, str]]:
    """細部に使う写真の番号 [(番号, 理由)]。scores は {番号: アップの目安}
    （着画・取れない写真は入れない）。アップ（CLOSEUP_TOUCH 以上）を目安の大きい順に
    （同点は番号の小さい順）。1枚も無ければ、主役の写真・
    skip を除いた最初の1枚（判断がつかなければ2枚目）。"""
    cand = {i: s for i, s in scores.items() if i != main_index and i not in skip}
    close = sorted((i for i, s in cand.items() if s >= CLOSEUP_TOUCH), key=lambda i: (-cand[i], i))
    if close:
        return [(i, f"アップ（縁の {cand[i]:.0%} に物が写る）") for i in close[:limit]]
    if cand:
        i = min(cand)
        return [(i, "アップが無いので、主役の次の写真")]
    return []


def prepare_details(
    images: list[str],
    main_image: str,
    cache_dir: Path,
    cutter_get: Callable,
    fetch: Callable[[str], bytes],
    *,
    skip_urls: tuple[str, ...] = (),
    log: Callable = print,
) -> list[DetailPhoto]:
    """細部の写真を選んで、白いふちに入れる用の JPEG（長辺 PHOTO_MAX_SIDE）を cache_dir に保存する。
    人が写る写真（着画）・主役の写真は使わない。取れない写真は飛ばす（無ければ空）。"""
    if len(images) < 2:
        return []
    data: dict[int, bytes] = {}
    scores: dict[int, float] = {}
    skip: set[int] = set()
    main_index = images.index(main_image) if main_image in images else 0
    for i, url in enumerate(images):
        if i == main_index or url in skip_urls:
            skip.add(i)
            continue
        try:
            raw = ad_worn.fetch_cached(url, cache_dir, fetch)
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception as e:  # 取れない・読めない写真は飛ばす
            log(f"  細部の写真 {i + 1} を使えなかった（{type(e).__name__}）")
            continue
        try:
            dets, _ = ad_worn.detect_cached(raw, cache_dir, cutter_get)
            if any(d.get("kind") == "person" and d.get("score", 0) >= PERSON_SKIP for d in dets):
                skip.add(i)
                continue
        except Exception:  # 検出が使えなくても、アップの目安だけで選ぶ
            pass
        data[i] = raw
        scores[i] = closeup_score(img)
    picks = pick_detail_indices(scores, main_index, skip)
    out: list[DetailPhoto] = []
    for i, reason in picks:
        img = Image.open(io.BytesIO(data[i])).convert("RGB")
        img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
        key = hashlib.sha1(images[i].encode()).hexdigest()[:16]
        p = cache_dir / f"detail-{key}.jpg"
        img.save(p, "JPEG", quality=88)
        out.append(DetailPhoto(p, images[i], i, scores[i], reason))
        log(f"  細部の写真: {i + 1} 枚目（{reason}）")
    return out


# ---------------------------------------------------------------- 検査・点数


@dataclass
class SingleChecks:
    memo_on_hero_px: int = 0
    memo_on_blocked_px: int = 0
    memo_memo_px: int = 0
    line_on_hero_px: int = 0
    line_on_text_px: int = 0
    line_cross: int = 0
    lines_out: int = 0
    memos_outside_safe: int = 0
    small_memos: int = 0
    badge_missing: bool = False
    badge_on_hero: float = 0.0
    badge_problem: bool = False
    photos_bad: int = 0
    placed: int = 0
    wanted: int = 0


def segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def polylines_cross(a: np.ndarray, b: np.ndarray) -> bool:
    """2本の線（点列）が交差するか。"""
    a0, a1 = a[:-1], a[1:]
    b0, b1 = b[:-1], b[1:]

    def ccw(p, q, r):
        return (r[..., 1] - p[..., 1]) * (q[..., 0] - p[..., 0]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    A0, A1 = a0[:, None, :], a1[:, None, :]
    B0, B1 = b0[None, :, :], b1[None, :, :]
    d1, d2 = ccw(B0, B1, A0), ccw(B0, B1, A1)
    d3, d4 = ccw(A0, A1, B0), ccw(A0, A1, B1)
    return bool(np.any(((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))))


def check_plan(
    plan: SinglePlan,
    *,
    hero_mask: np.ndarray,
    blocked: np.ndarray,
    safe: tuple[float, float, float, float],
    width: int,
    expect_price: bool,
    expect_stamp: bool,
) -> tuple[SingleChecks, list[str]]:
    """メモ・線・札・写真の検査。plan と、主役のマスク・主役以外の占有（見出し・店名・着画など）から
    作り直して数える（配置の途中の値は使わない）。(数, 問題の一覧)。"""
    H, W = hero_mask.shape
    c = SingleChecks(wanted=plan.wanted, placed=len(plan.memos))
    problems: list[str] = []
    boxes = [m.polygon(0) for m in plan.memos]
    memo_union = np.zeros((H, W), bool)
    for b in boxes:
        paint_poly(b, memo_union)
    for i, m in enumerate(plan.memos):
        b = boxes[i]
        c.memo_on_hero_px += poly_overlap(b, hero_mask)
        c.memo_on_blocked_px += poly_overlap(b, blocked)
        if poly_outside_rect(b, safe, slack=8):
            c.memos_outside_safe += 1
        if m.fs < MEMO_MIN_FS * width / 1080 - 0.5:
            c.small_memos += 1
        for j in range(i + 1, len(plan.memos)):
            c.memo_memo_px += poly_overlap(b, _mask_of(boxes[j], (H, W)))
    lines = [m.line_pts() for m in plan.memos]
    for i, m in enumerate(plan.memos):
        pts = lines[i]
        if pts[:, 0].min() < 0 or pts[:, 0].max() > W or pts[:, 1].min() < 0 or pts[:, 1].max() > H:
            c.lines_out += 1
        body = body_part(pts)
        lm = line_mask(body, (H, W), 3)
        c.line_on_hero_px += int(np.count_nonzero(lm & hero_mask))
        lm2 = line_mask(pts, (H, W), 3)
        others = blocked.copy()
        for j, b in enumerate(boxes):
            if j != i:
                paint_poly(b, others)
        # 自分のメモの箱の中（線の始まり）は数えない
        own = _mask_of(m.polygon(MEMO_PAD + 3), (H, W))
        c.line_on_text_px += int(np.count_nonzero(lm2 & others & ~own))
        for j in range(i + 1, len(plan.memos)):
            if polylines_cross(pts, lines[j]):
                c.line_cross += 1
    if plan.badge is not None:
        bm = plan.badge.disc_mask((H, W))
        area = max(1, int(np.count_nonzero(bm)))
        c.badge_on_hero = float(np.count_nonzero(bm & hero_mask)) / area
        bad = poly_outside_rect(plan.badge.polygon(), safe, slack=4)
        bad = bad or np.count_nonzero(bm & blocked) > TOLERANCE_PX
        bad = bad or c.badge_on_hero > BADGE_HERO_OVERLAP + 0.02
        bad = bad or any(poly_overlap(b, bm) > TOLERANCE_PX for b in boxes)
        c.badge_problem = bool(bad)
    elif expect_price:
        c.badge_missing = True
    for p in plan.photos:
        poly = p.polygon(0)
        area = max(1, poly_area_px(poly))
        bad = poly_outside_rect(poly, (0, 0, W, H))
        bad = bad or poly_overlap(poly, blocked) > TOLERANCE_PX
        bad = bad or poly_overlap(poly, hero_mask) / area > PHOTO_HERO_OVERLAP + 0.02
        bad = bad or poly_overlap(poly, memo_union) > TOLERANCE_PX
        c.photos_bad += int(bad)
    if c.memo_on_hero_px > TOLERANCE_PX:
        problems.append(f"メモが服に重なっている（{c.memo_on_hero_px}px）")
    if c.memo_on_blocked_px > TOLERANCE_PX:
        problems.append(f"メモが見出しなどの文字・着画に重なっている（{c.memo_on_blocked_px}px）")
    if c.memo_memo_px > TOLERANCE_PX:
        problems.append(f"メモどうしが重なっている（{c.memo_memo_px}px）")
    if c.memos_outside_safe:
        problems.append(f"メモが安全域の外にある（{c.memos_outside_safe} 個）")
    if c.small_memos:
        problems.append(f"メモの文字が小さくて読みにくい（{c.small_memos} 個）")
    if c.line_on_hero_px > TOLERANCE_PX:
        problems.append(f"引き出し線が服の上を通っている（{c.line_on_hero_px}px）")
    if c.line_on_text_px > TOLERANCE_PX:
        problems.append(f"引き出し線が文字や札に重なっている（{c.line_on_text_px}px）")
    if c.line_cross:
        problems.append(f"引き出し線が交差している（{c.line_cross} か所）")
    if c.lines_out:
        problems.append(f"引き出し線が画面の外に出ている（{c.lines_out} 本）")
    if c.badge_missing:
        problems.append("価格の札を置けない")
    if c.badge_problem:
        problems.append("価格の札が見えにくい（はみ出し・文字や服への重なり）")
    if expect_stamp and plan.stamp is None:
        plan.notes.append("「1点もの」のシールを置く場所が無い")
    if c.photos_bad:
        problems.append("細部の写真が重なっている・はみ出している")
    if plan.wanted >= MIN_CALLOUTS and c.placed < MIN_CALLOUTS:
        problems.append(f"メモを{MIN_CALLOUTS}つ置けない（{c.placed}/{plan.wanted}）")
    elif plan.wanted and c.placed == 0:
        problems.append("メモを置けない")
    return c, problems


def _mask_of(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    paint_poly(poly, m)
    return m


def score_plan(
    plan: SinglePlan,
    *,
    hero_mask: np.ndarray,
    region: tuple[float, float, float, float],
    width: int,
    height: int,
    hero_target_h: float,
) -> tuple[float, dict]:
    """点数（0〜100）。置けたメモの割合・主役の大きさ・左右のつり合い・線の短さと自然さ・文字の大きさ・
    内容の縦の使い方。簡単な目安。"""
    bx0, by0, bx1, by1 = mask_bbox(hero_mask)
    hh = by1 - by0
    size = min(1.0, hh / max(1.0, hero_target_h))
    placed = (len(plan.memos) / plan.wanted) if plan.wanted else 1.0
    cx = (bx0 + bx1) / 2
    left = sum(1 for m in plan.memos if m.cx < cx)
    right = len(plan.memos) - left
    bal = 1.0 if not plan.memos else 1 - abs(left - right) / len(plan.memos)
    lens = [path_length(m.line_pts()) for m in plan.memos]
    short = 1.0 if not lens else max(0.0, 1 - (sum(lens) / len(lens)) / (0.32 * width))
    fsn = 1.0 if not plan.memos else min(1.0, float(np.mean([m.fs for m in plan.memos])) / (
        MEMO_STEPS[1] * width / 1080))  # fmt: skip
    x0, y0, x1, y1 = region
    top = min([by0] + [m.cy - m.h / 2 for m in plan.memos])
    bot = max([by1] + [m.cy + m.h / 2 for m in plan.memos])
    use = min(1.0, (bot - top) / max(1.0, 0.9 * (y1 - y0)))
    extras = 0.0
    if plan.badge is not None:
        extras += 0.5
    if plan.stamp is not None or not plan.stamp_text:
        extras += 0.25
    if plan.photos:
        extras += 0.25
    parts = {
        "placed": round(placed, 3),
        "size": round(size, 3),
        "balance": round(bal, 3),
        "short_lines": round(short, 3),
        "font": round(fsn, 3),
        "use": round(use, 3),
        "extras": round(extras, 3),
    }
    score = 100 * (
        0.30 * placed + 0.20 * size + 0.10 * bal + 0.12 * short + 0.08 * fsn + 0.10 * use
        + 0.10 * extras
    )  # fmt: skip
    return round(score, 1), parts


# ---------------------------------------------------------------- 描画（HTML / SVG の部品）


def badge_text_color(pal: Palette) -> str:
    """札（アクセント色の地）の文字色。地の色・白・濃い色のうち、いちばん読めるもの。"""
    cands = [pal.bg, "#FFFFFF", "#1B1A18"]
    return max(cands, key=lambda c: contrast_ratio(pal.accent, c))


def badge_path(b: Badge) -> str:
    """札の縁。丸いシールのふち（細かい波）か、角の丸い星形。手描きらしく少しゆらす（同じ文字なら同じ形）。"""
    rnd = random.Random(b.seed_text)
    pts = []
    if b.style == "star":
        n = 14
        for i in range(2 * n):
            a = math.pi * i / n - math.pi / 2
            rr = b.r * (1.0 if i % 2 == 0 else 0.86) * (1 + rnd.uniform(-0.015, 0.015))
            pts.append((rr * math.cos(a), rr * math.sin(a)))
    else:
        n = 56
        for i in range(n):
            a = 2 * math.pi * i / n - math.pi / 2
            rr = b.r * (1 + 0.035 * math.sin(a * 14) + rnd.uniform(-0.012, 0.012))
            pts.append((rr * math.cos(a), rr * math.sin(a)))
    return ad_season.smooth(pts, closed=True)


def svg_parts(plan: SinglePlan, pal: Palette, e: Callable[[str], str]) -> list[str]:
    """引き出し線・札・1点もののシールの SVG。"""
    out: list[str] = []
    for m in plan.memos:
        pts = m.line_pts()
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        end = pts[-1]
        if m.end_style == "arrow":
            v = pts[-1] - pts[-5]
            v = v / max(np.linalg.norm(v), 1e-9)
            for ang in (150, -150):
                q = end + rot_matrix(ang) @ v * 20
                d += f" M{end[0]:.1f} {end[1]:.1f} L{q[0]:.1f} {q[1]:.1f}"
            out.append(f'<path class="lead" d="{d}"/>')
        else:
            out.append(
                f'<path class="lead" d="{d}"/>'
                f'<circle class="lead" cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="{RING_R}"/>'
            )
    if plan.badge is not None:
        b = plan.badge
        tc = badge_text_color(pal)
        out.append(
            f'<g transform="translate({b.cx:.1f} {b.cy:.1f}) rotate({b.angle:.1f})">'
            f'<path d="{badge_path(b)}" fill="{pal.accent}" stroke="{pal.doodle}" stroke-width="3"'
            ' stroke-linejoin="round" vector-effect="non-scaling-stroke" opacity=".98"/>'
            f'<text class="price" y="{b.size * 0.34:.1f}" font-size="{b.size}" fill="{tc}">'
            f"{e(b.text)}</text></g>"
        )
    return out


def html_parts(plan: SinglePlan, pal: Palette, e: Callable[[str], str], width: int) -> list[str]:
    """メモ（手書き）・1点ものシール・細部の写真の HTML。"""
    out: list[str] = []
    for p in plan.photos:
        pad_b = PHOTO_PAD_BOTTOM
        out.append(
            f'<div class="polaroid" style="left:{p.cx - p.w / 2:.1f}px;top:{p.cy - p.h / 2:.1f}px;'
            f"width:{p.w:.1f}px;height:{p.h:.1f}px;transform:rotate({p.angle:.2f}deg);"
            f'padding:{PHOTO_PAD}px {PHOTO_PAD}px {pad_b}px {PHOTO_PAD}px">'
            f'<img src="{Path(p.path).resolve().as_uri()}"></div>'
        )
    if plan.stamp is not None:
        s = plan.stamp
        size = round(STAMP_SIZE * width / 1080)
        out.append(
            f'<div class="tag" style="left:{s.cx - s.w / 2:.1f}px;top:{s.cy - s.h / 2:.1f}px;'
            f"width:{s.w:.1f}px;height:{s.h:.1f}px;font-size:{size}px;line-height:{s.h:.1f}px;"
            f'transform:rotate({s.angle:.1f}deg)">{e(s.text)}</div>'
        )
    for m in plan.memos:
        style = (
            f"left:{m.cx - m.w / 2:.1f}px;top:{m.cy - m.h / 2:.1f}px;width:{m.w:.1f}px;"
            f"height:{m.h:.1f}px;font-size:{m.fs}px;transform:rotate({m.angle:.2f}deg);"
        )
        if m.accent:
            style += f"color:{pal.accent};"
        out.append(
            f'<div class="memo" style="{style}">' + "<br>".join(e(ln) for ln in m.lines) + "</div>"
        )
    return out


def css(pal: Palette) -> str:
    return f"""
.memo{{position:absolute;white-space:nowrap;font-family:"Caveat","Yomogi";font-weight:700;
line-height:{MEMO_LINE_H};letter-spacing:.03em;color:{pal.text}}}
.polaroid{{position:absolute;background:#FBF8F1;border-radius:5px;
box-shadow:0 10px 18px {pal.shadow}}}
.polaroid img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:2px}}
.tag{{position:absolute;text-align:center;white-space:nowrap;font-family:"Caveat","Yomogi";
font-weight:700;background:{pal.text};color:{pal.bg};border-radius:999px;letter-spacing:.06em}}
svg.ov path.lead,svg.ov circle.lead{{fill:none;stroke:{pal.doodle};stroke-width:{LINE_STROKE};
stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}}
svg.ov text.price{{font-family:"DM Sans","Noto Sans CJK JP",sans-serif;font-weight:900;
text-anchor:middle;letter-spacing:-.01em}}
"""
