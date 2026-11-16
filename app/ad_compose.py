"""切り抜いた商品の写真から、SNS 広告の画像（雑誌の誌面のようなコラージュ）を自動で組む。

入力はブリーフ（JSON。店名・コピー・商品など店に固有の値はすべてここから来る）。流れ:
  1. 切り抜き（app/cutout.py。同じ画像は data/ のキャッシュを使う）。check / fail の商品は外す
  2. 配置（規則 + 乱数の種）: 商品を散らして少し傾け、服の輪郭に沿って流れる手書きの文字（label）の
     区間を、ほかの物の上を通らないように選ぶ。見出し・店名・コピーの領域とは重ねない
  3. 描画: HTML + SVG（textPath）を作り、ヘッドレスの Chrome で PNG にする
  4. 検査（規則）: 重なり・はみ出し・安全域・商品の大きさ・見出しの行数を数え、
     pass / fail と点数を出す
  5. 候補を種を変えて作り、pass の中で点数の高い物を残す。
     無ければ違反の最も少ない物を needs_review にする

**商品の画素は描き直さない**（切り抜き・拡大縮小・回転・影だけ）。文字の大きさ・色・安全域などは
名前付きの定数にまとめてある。モデルと Chrome を使わない部分（占有マップ・配置・検査）だけでも動く。
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont

from app import ad_season
from app.ad_palette import (  # noqa: F401  色の部品（テスト・外からも ad_compose 経由で使える）
    DEFAULT_PALETTE,
    MIN_CONTRAST,
    PALETTE_NAMES,
    PALETTES,
    Palette,
    choose_palette,
    contrast_ratio,
    contrast_score,
    glow_color,
    item_lightness,
    luminance,
    palette_family,
    sinks,
    tone_hist,
)
from app.ad_palette import GLOW_BLUR as _GLOW_BLUR
from app.ad_season import Season, parse_season

# ---------------------------------------------------------------- 定数

FORMATS = {"1:1": (1080, 1080), "4:5": (1080, 1350), "9:16": (1080, 1920)}
DEFAULT_FORMAT = "4:5"
MIN_ITEMS = 2
MAX_ITEMS = 6

# 文字の大きさは比率の段（1.333 倍ずつ）から選ぶ
SCALE = (24, 32, 43, 57, 76, 101, 135, 180)
STORE_SIZE = SCALE[0]
ASIDE_ANGLE = -6.0
ASIDE_STEPS = (SCALE[3], SCALE[2])  # ひとこと（右に収まる大きい段から）
TAGLINE_STEPS = (SCALE[3], SCALE[2])
HEADLINE_STEPS = {
    "4:5": (SCALE[4], SCALE[3]),
    "1:1": (SCALE[4], SCALE[3]),
    "9:16": (SCALE[5], SCALE[4], SCALE[3]),
}
LABEL_STEPS = (SCALE[1], SCALE[0])  # 輪郭に沿う文字（大きいほうから試す）
# 下限は LABEL_STEPS[0]。入らないときは商品を縮めて間隔を広げ、最後の試行だけ小さい段も許す
HEAD_LINE_HEIGHT = 1.16
HEADLINE_MAX_LINES = 2
HEADLINE_LINE_MAX = 16  # 見出しの1行の字数の上限
PALT_FACTOR = 0.9  # 見出しの詰め組みで縮む分の見込み（測りは詰め組みなしなので少し縮める）
TEXT_LEN_SAFETY = 1.06  # 輪郭の文字の長さの見積りの余裕
ARC_SLACK = 1.15  # 描く弧は、見積りよりこれだけ長く取る（textPath は弧の終わりより先を描かない）

# 安全域: 左右 6%・上下 3.5%。9:16 は上下 14% を文字なし（ストーリーズの操作部に隠れる）
MARGIN_X_RATIO = 0.06
MARGIN_Y_RATIO = 0.035
STORY_SAFE_V = 0.14
ITEM_EDGE_RATIO = 0.03  # 商品が画面の端から離れる最小（幅に対する割合）

ANGLE_MAX = 10.0
ANGLE_MIN = 3.0
FILL_RATIO = 0.64  # 商品の外接矩形の合計が、置ける範囲に占める割合のめやす
USE_TARGET = 0.9  # 商品の縦の広がりが、置ける範囲の高さのこの割合以上なら満点
FILL_RATIO_TALL = 0.95  # 縦長（9:16）は商品を大きめにして縦に広く使う
MAX_ITEM_H_OF_REGION = 0.6
MAX_ITEM_W_OF_REGION = 0.65
ELLIPSE_FACTOR = 0.85  # 商品どうしの離し方に使う楕円の大きさ（外接矩形に対して）
ITEM_GAP = 36  # 商品どうしの間に空ける（輪郭の文字の分）
RELAX_ITERS = 90
ATTEMPTS = 6  # 文字が収まらないとき、商品を縮めてやり直す回数
SHRINK = 0.92

# 輪郭に沿う文字
CONTOUR_OFFSET = 20  # 服の輪郭から文字の土台までの距離（px）
CONTOUR_STEP = 6  # 輪郭の点の間隔（px）
BAND_UP = 0.9  # 文字の占める帯: 土台から外側へ 文字の大きさ × これ
BAND_DOWN = 0.3  # 内側へ
UPSIDE_PENALTY = 6000.0  # 逆さになる区間（長さの割合）への罰（px² 相当）
SAFE_PENALTY = 1.0
CONCAVE_PENALTY = 9000.0  # 内側へ曲がる（文字が詰まる）角への罰（曲がる角度 rad あたり）
CONCAVE_FREE = 0.06  # 1歩あたりこの角度までの曲がりは許す

# 検査のしきい値
TOLERANCE_PX = 40  # 重なりの許容（アンチエイリアスの分）
MAX_ITEM_OVERLAP = 0.12  # 商品どうしの重なり（小さいほうの面積に対する割合）
CONTRAST_WEIGHT = 0.1  # 点数に占める「商品と地の明暗の差」の割合
MIN_ITEM_H_RATIO = 0.18  # 商品の高さの下限（幅に対する割合）
ITEM_OUTSIDE_RATIO = 0.01  # 商品が画面の端の余白から出てよい割合（商品の面積に対して）

# 落書き
DOODLE_COUNT = (3, 5)
DOODLE_SIZE = (80, 120)
DOODLE_ROT = 25.0  # 汎用の落書きを傾ける角度の上限（±度）
SEASON_DOODLE_COUNT = (5, 7)  # 季節があるとき（季節の落書きが主）
SEASON_GENERIC_MAX = 2  # そのうち汎用の落書き（星・波・矢印…）は多くてもこれだけ
SEASON_DOODLE_SIZE = (84, 150)  # 大きさにばらつきを付ける
DOODLE_STROKE = 4
DOODLE_CLEAR = 18  # 落書きと商品・文字の間に空ける
DOODLE_TRIES = 400

CHROME_ARGS = (
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--hide-scrollbars",
    "--force-device-scale-factor=1",
    "--virtual-time-budget=4000",
)


FONT_FILES = {"dm": "DMSans.ttf", "caveat": "Caveat.ttf", "yomogi": "Yomogi.ttf"}
FONT_OPTIONAL = ("DMSans-Italic.ttf", "ZenKurenaido.ttf", "Gaegu-Bold.ttf")
NOTO_BLACK = os.getenv("AD_NOTO_BLACK", "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc")
NOTO_REGULAR = os.getenv(
    "AD_NOTO_REGULAR", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)


class BriefError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class FontsError(RuntimeError):
    pass


class CannotCompose(RuntimeError):
    """切り抜けた商品が足りないなど、作れないとき。"""


# ---------------------------------------------------------------- ブリーフ


@dataclass
class BriefItem:
    id: str
    image: str
    label: str
    item_type: str | None = None
    cutout: str | None = None  # 切り抜き済みの RGBA PNG があればそれを使う
    tone: str | None = None  # dark / mid / light（任意。画素から測れないときの手がかり）


@dataclass
class Brief:
    id: str
    format: str
    store_name: str
    tagline: str
    area_line: str
    headline: list[str]
    aside: str
    palette: str
    items: list[BriefItem]
    seed: int = 0
    candidates: int = 8
    orig_id: str = (
        ""  # 元の id（CRM の URL・返信・dropped_items に使う）。id はファイル名用の安全な形
    )
    revision: object = None  # CRM が古い版を見分けるための版（そのまま返す）
    formats: list[str] = field(default_factory=list)  # ブリーフが望む形式（引数が無いとき使う）
    season: Season | None = None  # 無い・知らない motif は None（季節なし。今までの見た目）
    palette_options: list[str] = field(default_factory=list)  # 沈むとき替えてよい色（任意）
    palette_requested: str = ""  # ブリーフの palette（palette は実際に使う色）
    palette_changed_reason: str | None = None


_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_id(raw_id: str, maxlen: int = 48) -> str:
    """ファイル名・キャッシュに使える形に直す（英数字と . _ - 以外は _）。直した・切ったときは、
    別の id と衝突しないよう元の id の短いハッシュを足す。"""
    safe = _UNSAFE.sub("_", raw_id)[:maxlen]
    if safe != raw_id or safe in ("", ".", ".."):
        safe = f"{safe}-{hashlib.sha1(raw_id.encode()).hexdigest()[:6]}"
    return safe


def _raw_id(value) -> str | None:
    """id は文字列か整数（CRM のブリーフは整数）。それ以外・空は None。"""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    return text if text and len(text) <= 200 and not _CTRL.search(text) else None


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _text(problems: list[str], name: str, value, *, required=False, maxlen=60) -> str:
    if value is None or value == "":
        if required:
            problems.append(f"{name} が必要です")
        return ""
    if not isinstance(value, str):
        problems.append(f"{name} は文字列にしてください")
        return ""
    if _CTRL.search(value):
        problems.append(f"{name} に制御文字があります")
    if len(value) > maxlen:
        problems.append(f"{name} が長すぎます（{maxlen} 字まで）")
    return value.strip()


def parse_brief(raw: dict) -> Brief:
    """ブリーフの検証。足りない・おかしい所を全部まとめて BriefError にする。"""
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise BriefError(["ブリーフは JSON のオブジェクトにしてください"])
    bid = _raw_id(raw.get("id"))
    if bid is None:
        problems.append("id は空でない文字列か整数にしてください")
    fmt = raw.get("format") or DEFAULT_FORMAT
    if fmt not in FORMATS:
        problems.append(f"format は {' / '.join(FORMATS)} のどれかにしてください")
    store = raw.get("store") or {}
    if not isinstance(store, dict):
        problems.append("store はオブジェクトにしてください")
        store = {}
    name = _text(problems, "store.name", store.get("name"), required=True, maxlen=30)
    tagline = _text(problems, "store.tagline", store.get("tagline"), maxlen=40)
    area = _text(problems, "store.area_line", store.get("area_line"), maxlen=30)
    head = _text(problems, "headline", raw.get("headline"), required=True, maxlen=60)
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    if len(lines) > HEADLINE_MAX_LINES:
        problems.append(f"headline は {HEADLINE_MAX_LINES} 行までです")
    # 改行が無い1行は、組むときに2行へ割る（その分だけ長くてよい）
    if any(
        len(ln) > (HEADLINE_LINE_MAX if len(lines) > 1 else 2 * HEADLINE_LINE_MAX) for ln in lines
    ):
        problems.append(f"headline の1行は {HEADLINE_LINE_MAX} 字までです（改行が無ければ2行ぶん）")
    aside = _text(problems, "aside", raw.get("aside"), maxlen=20)
    palette = raw.get("palette") or DEFAULT_PALETTE
    if palette not in PALETTES:
        problems.append(f"palette は {' / '.join(PALETTES)} のどれかにしてください")
    seed = raw.get("seed", 0)
    cands = raw.get("candidates", 8)
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append("seed は整数にしてください")
    if not isinstance(cands, int) or isinstance(cands, bool) or not 1 <= cands <= 32:
        problems.append("candidates は 1〜32 の整数にしてください")
    items: list[BriefItem] = []
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        problems.append("items は配列にしてください")
        raw_items = []
    if not MIN_ITEMS <= len(raw_items) <= MAX_ITEMS:
        problems.append(
            f"items は {MIN_ITEMS}〜{MAX_ITEMS} 点にしてください（{len(raw_items)} 点）"
        )
    seen: set[str] = set()
    for i, it in enumerate(raw_items):
        if not isinstance(it, dict):
            problems.append(f"items[{i}] はオブジェクトにしてください")
            continue
        iid = _raw_id(it.get("id"))
        if iid is None:
            problems.append(f"items[{i}].id は空でない文字列か整数にしてください")
            continue
        if iid in seen:
            problems.append(f"items[{i}].id が重複しています（{iid}）")
        seen.add(iid)
        image = _text(problems, f"items[{i}].image", it.get("image"), maxlen=2000)
        cut = _text(problems, f"items[{i}].cutout", it.get("cutout"), maxlen=2000)
        if not image and not cut:
            problems.append(f"items[{i}].image が必要です")
        label = _text(problems, f"items[{i}].label", it.get("label"), required=True, maxlen=60)
        itype = it.get("item_type")
        tone = it.get("tone")
        items.append(
            BriefItem(
                iid,
                image,
                label,
                itype if isinstance(itype, str) else None,
                cut or None,
                tone if tone in ("dark", "mid", "light") else None,
            )
        )
    if problems:
        raise BriefError(problems)
    wanted = raw.get("formats")
    formats = [f for f in wanted if f in FORMATS] if isinstance(wanted, list) else []
    opts = raw.get("palette_options")
    options = (
        [o for o in dict.fromkeys(opts) if isinstance(o, str) and o in PALETTES]
        if isinstance(opts, list)
        else []
    )
    return Brief(
        safe_id(bid),
        fmt,
        name,
        tagline,
        area,
        lines,
        aside,
        palette,
        items,
        seed,
        cands,
        orig_id=bid,
        revision=raw.get("revision"),
        formats=formats,
        season=parse_season(raw.get("season")),
        palette_options=options,
        palette_requested=palette,
    )


# ---------------------------------------------------------------- 幾何（占有マップの部品）


def rot_matrix(angle_deg: float) -> np.ndarray:
    """CSS の rotate(θ)（画面で時計回りが正・y が下）の 2x2 行列。"""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[c, -s], [s, c]])


def item_affine(sh: int, h: float, cx: float, cy: float, angle: float, w: float) -> np.ndarray:
    """切り抜き（高さ sh）を、高さ h・中心 (cx, cy)・angle 度で置く 2x3 アフィン行列。"""
    r = rot_matrix(angle)
    s = h / sh
    t = np.array([cx, cy]) - r @ np.array([w / 2, h / 2])
    return np.hstack([r * s, t[:, None]])


def item_mask(
    alpha: np.ndarray,
    cx: float,
    cy: float,
    h: float,
    angle: float,
    size: tuple[int, int],
    pad: int = 0,
) -> np.ndarray:
    """回転後の商品のマスク（画面と同じ大きさの bool）。pad は四方に広げて測る分。"""
    width, height = size
    w = alpha.shape[1] * h / alpha.shape[0]
    m = item_affine(alpha.shape[0], h, cx + pad, cy + pad, angle, w)
    out = cv2.warpAffine(alpha, m, (width + 2 * pad, height + 2 * pad), flags=cv2.INTER_LINEAR)
    return out > 40


def polygon_mask(poly: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(out, [np.round(poly).astype(np.int32)], 1)
    return out.astype(bool)


def rect_polygon(
    x: float, y: float, w: float, h: float, angle: float = 0.0, pad: float = 0.0
) -> np.ndarray:
    """左上 (x, y)・幅 w・高さ h の矩形を中心まわりに angle 度回した 4 点。"""
    c = np.array([x + w / 2, y + h / 2])
    half = np.array([w / 2 + pad, h / 2 + pad])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    return c + corners @ rot_matrix(angle).T


@dataclass
class Band:
    """輪郭に沿う文字が占める帯のラスタ（外接矩形の中だけ）。x0, y0 は画面での左上。"""

    x0: int
    y0: int
    mask: np.ndarray

    def clip(self, width: int, height: int):
        """画面と重なる範囲を (自分の slice, 画面の slice) で返す。無ければ None。"""
        h, w = self.mask.shape
        sx0, sy0 = max(0, -self.x0), max(0, -self.y0)
        sx1, sy1 = min(w, width - self.x0), min(h, height - self.y0)
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        return (
            (slice(sy0, sy1), slice(sx0, sx1)),
            (slice(self.y0 + sy0, self.y0 + sy1), slice(self.x0 + sx0, self.x0 + sx1)),
        )

    def overlap(self, big: np.ndarray) -> int:
        c = self.clip(big.shape[1], big.shape[0])
        if c is None:
            return 0
        mine, theirs = c
        return int(np.count_nonzero(self.mask[mine] & big[theirs]))

    def paint(self, big: np.ndarray) -> None:
        c = self.clip(big.shape[1], big.shape[0])
        if c is not None:
            mine, theirs = c
            big[theirs] |= self.mask[mine]

    def outside(self, rect: tuple[int, int, int, int]) -> int:
        """矩形 (x0, y0, x1, y1) の外にある画素数。"""
        h, w = self.mask.shape
        rx0, ry0 = max(0, rect[0] - self.x0), max(0, rect[1] - self.y0)
        rx1, ry1 = min(w, rect[2] - self.x0), min(h, rect[3] - self.y0)
        inside = (
            int(np.count_nonzero(self.mask[ry0:ry1, rx0:rx1])) if rx1 > rx0 and ry1 > ry0 else 0
        )
        return int(np.count_nonzero(self.mask)) - inside

    def full(self, width: int, height: int) -> np.ndarray:
        out = np.zeros((height, width), bool)
        self.paint(out)
        return out


def arc_tangents(pts: np.ndarray) -> np.ndarray:
    t = np.gradient(pts, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-9)


def raster_band(pts: np.ndarray, fs: float) -> Band:
    """弧（点列）に沿って流れる、大きさ fs の文字が占める帯。文字の上（外側）は (ty, -tx)。"""
    t = arc_tangents(pts)
    up = np.stack([t[:, 1], -t[:, 0]], axis=1)
    outer = pts + up * fs * BAND_UP
    inner = pts - up * fs * BAND_DOWN
    allp = np.vstack([outer, inner])
    x0, y0 = int(math.floor(allp[:, 0].min())) - 1, int(math.floor(allp[:, 1].min())) - 1
    x1, y1 = int(math.ceil(allp[:, 0].max())) + 2, int(math.ceil(allp[:, 1].max())) + 2
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0])
    quads = np.stack([inner[:-1], outer[:-1], outer[1:], inner[1:]], axis=1) - off
    cv2.fillPoly(local, np.round(quads).astype(np.int32), 1)
    return Band(x0, y0, local.astype(bool))


def concave_turn(pts: np.ndarray) -> float:
    """弧が文字の内側（時計回りの弧では左）へ曲がる角度の合計（rad）。大きいと文字が詰まる。"""
    t = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    d = np.diff(ang)
    return float(np.sum(np.maximum(0.0, -d - CONCAVE_FREE)))


def upside_fraction(pts: np.ndarray) -> float:
    """弧のうち、文字が逆さに近くなる（上向きの向きが下を向く）区間の割合。"""
    t = arc_tangents(pts)
    up_y = -t[:, 0]
    return float(np.mean(up_y > 0.5))


# ---------------------------------------------------------------- 輪郭


def contour_local(
    alpha: np.ndarray, h: int, off: int = CONTOUR_OFFSET, step: float = CONTOUR_STEP
) -> tuple[float, np.ndarray]:
    """高さ h にしたときの服の輪郭を、外側へ off だけずらして滑らかにした点列。

    戻り値は (幅, 点列 N×2)。点列は画面で時計回り・上半分の左端から始まり、点の間隔は step。"""
    s = h / alpha.shape[0]
    w = max(1, int(round(alpha.shape[1] * s)))
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    pad = off + 12
    m = cv2.copyMakeBorder(
        (a > 40).astype(np.uint8) * 255, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * off + 1, 2 * off + 1))
    m = cv2.dilate(m, k)
    m = (cv2.GaussianBlur(m, (0, 0), 10) > 127).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) - pad
    if signed_area(c) < 0:  # 画面（y が下）で時計回りにそろえる
        c = c[::-1]
    ker = 41
    if len(c) > 3 * ker:
        box = np.ones(ker) / ker
        c = np.stack(
            [
                np.convolve(np.r_[c[-ker:, i], c[:, i], c[:ker, i]], box, "same")[ker:-ker]
                for i in (0, 1)
            ],
            axis=1,
        )
    c = resample_closed(c, step)
    top = np.where(c[:, 1] < h * 0.45)[0]
    pool = top if len(top) else np.arange(len(c))
    start = pool[int(np.argmin(c[pool, 0]))]
    return float(w), np.roll(c, -start, axis=0)


def signed_area(c: np.ndarray) -> float:
    x, y = c[:, 0], c[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_closed(c: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    n = max(8, int(cum[-1] / step))
    u = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(u, cum, closed[:, 0]), np.interp(u, cum, closed[:, 1])], axis=1)


def to_canvas(
    pts: np.ndarray, w: float, h: float, cx: float, cy: float, angle: float
) -> np.ndarray:
    return (pts - np.array([w / 2, h / 2])) @ rot_matrix(angle).T + np.array([cx, cy])


# ---------------------------------------------------------------- 文字の測り


def _is_latin(ch: str) -> bool:
    return ord(ch) < 0x250 or ch == "¥"


class Measurer:
    """文字の幅の見積り。フォントがあれば PIL で測り、無ければ全角 1em・半角 0.55em で見積もる。"""

    def __init__(self, fonts_dir: str | Path | None = None):
        self.fonts_dir = Path(fonts_dir) if fonts_dir else None
        self._fonts: dict[tuple, ImageFont.FreeTypeFont | None] = {}
        self._w: dict[tuple, float] = {}

    def _font(self, key: str, size: int):
        k = (key, size)
        if k in self._fonts:
            return self._fonts[k]
        font = None
        try:
            if key in FONT_FILES and self.fonts_dir:
                path = self.fonts_dir / FONT_FILES[key]
                if path.exists():
                    font = ImageFont.truetype(str(path), size)
                    if key == "dm":
                        font.set_variation_by_axes([min(max(size, 9), 40), 800])
                    elif key == "caveat":
                        font.set_variation_by_axes([700])
            elif key == "noto_black" and Path(NOTO_BLACK).exists():
                font = ImageFont.truetype(NOTO_BLACK, size, index=0)
            elif key == "noto_regular" and Path(NOTO_REGULAR).exists():
                font = ImageFont.truetype(NOTO_REGULAR, size, index=0)
        except OSError:
            font = None
        self._fonts[k] = font
        return font

    def _char(self, latin: str, other: str, ch: str, size: int) -> float:
        k = (latin, other, ch, size)
        if k not in self._w:
            font = self._font(latin if _is_latin(ch) else other, size)
            if font is None:
                self._w[k] = size * (0.55 if _is_latin(ch) else 1.0)
            else:
                self._w[k] = font.getlength(ch)
        return self._w[k]

    def width(self, kind: str, text: str, size: int) -> float:
        """kind: head（太いゴシック）/ hand（手書き）/ store（店名）/ area（地域の行）。"""
        if kind == "head":
            return PALT_FACTOR * sum(self._char("dm", "noto_black", c, size) for c in text)
        if kind == "hand":
            return sum(self._char("caveat", "yomogi", c, size) for c in text) + 0.03 * size * len(
                text
            )
        if kind == "store":
            return sum(self._char("dm", "noto_regular", c, size) for c in text) + 0.12 * size * len(
                text
            )
        return sum(
            self._char("noto_regular", "noto_regular", c, size) for c in text
        ) + 0.06 * size * len(text)


def check_fonts(fonts_dir: str | Path | None) -> Path:
    """描画に要るフォントがそろっているか。無ければ分かりやすいエラー。"""
    if not fonts_dir:
        raise FontsError(
            "フォントの置き場所が分かりません。--fonts でフォントのフォルダを指定してください"
            "（DMSans.ttf・Caveat.ttf・Yomogi.ttf など）"
        )
    d = Path(fonts_dir).expanduser()
    missing = [f for f in FONT_FILES.values() if not (d / f).exists()]
    if missing:
        raise FontsError(f"フォントがありません: {', '.join(missing)}（{d}）")
    return d


# ---------------------------------------------------------------- 固定の文字


@dataclass
class Geo:
    w: int
    h: int
    mx: int
    safe_top: int
    safe_bottom: int

    @property
    def safe_rect(self) -> tuple[int, int, int, int]:
        return (self.mx, self.safe_top, self.w - self.mx, self.h - self.safe_bottom)


def geometry(fmt: str) -> Geo:
    w, h = FORMATS[fmt]
    v = STORY_SAFE_V if fmt == "9:16" else MARGIN_Y_RATIO
    return Geo(w, h, round(MARGIN_X_RATIO * w), round(v * h), round(v * h))


@dataclass
class Block:
    name: str  # store / area / headline / aside / tagline
    lines: list[str]
    kind: str  # head / hand / store / area
    size: int
    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0
    align: str = "left"
    underline: bool = False

    def polygon(self, pad: float = 6.0) -> np.ndarray:
        return rect_polygon(self.x, self.y, self.w, self.h, self.angle, pad)

    line_widths: list[float] = field(
        default_factory=list
    )  # 見出しの行ごとの幅（占有を行ごとにする）

    def polygons(self, pad: float = 6.0) -> list[np.ndarray]:
        """占有する領域。見出しは行ごとの矩形（短い行の脇を空けるため）、ほかは1つ。"""
        if self.line_widths and len(self.line_widths) == len(self.lines):
            lh = self.h / len(self.lines)
            return [
                rect_polygon(self.x, self.y + i * lh, w, lh, 0.0, pad)
                for i, w in enumerate(self.line_widths)
            ]
        return [self.polygon(pad)]

    def mask(self, size: tuple[int, int], pad: float = 6.0) -> np.ndarray:
        out = np.zeros((size[1], size[0]), bool)
        for poly in self.polygons(pad):
            out |= polygon_mask(poly, size)
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d["polygon"] = np.round(self.polygon(), 1).tolist()
        d["polygons"] = [np.round(p, 1).tolist() for p in self.polygons()]
        return d


def balance_split(text: str, ms: Measurer, size: int) -> list[str]:
    """1行のままでは収まらない見出しを、2行に割る。読点・句点・空白のあとを優先し、幅が揃うように。"""
    if len(text) < 2:
        return [text]
    cuts = [i + 1 for i, c in enumerate(text[:-1]) if c in "、，, 。！？! ?"] or list(
        range(1, len(text))
    )

    def cost(i: int) -> float:
        a, b = text[:i].strip(), text[i:].strip()
        return abs(ms.width("head", a, size) - ms.width("head", b, size))

    i = min(cuts, key=cost)
    return [text[:i].strip(), text[i:].strip()]


@dataclass
class Fixed:
    blocks: list[Block]
    region: tuple[float, float, float, float]  # 商品を置ける範囲 (x0, y0, x1, y1)
    notes: list[str] = field(default_factory=list)


STAMP_SIZE = SCALE[0] - 2  # 季節のスタンプの文字の大きさ
STAMP_ANGLE = -4.0
STAMP_GAP = 30  # 店名・地域の行との間
STAMP_TOP_DROP = 4  # 傾けた角が安全域の上へ出ないよう、行の上端より少し下げる


def place_stamp(season: Season | None, blocks: list[Block], geo: Geo, ms: Measurer) -> Block | None:
    """季節の一言（例「NOVEMBER / 晩秋」）の小さなスタンプ。店名の行の、店名のすぐ右に置く
    （地域の行にぶつかるなら置かない）。手描きの楕円の枠と少しの傾きは描画側。"""
    if season is None or not season.stamp:
        return None
    tw = ms.width("store", season.stamp, STAMP_SIZE)
    rx, ry = tw / 2 * 1.28 + 12, STAMP_SIZE * 0.85
    w, h = 2 * rx, 2 * ry
    store = next((b for b in blocks if b.name == "store"), None)
    area = next((b for b in blocks if b.name == "area"), None)
    x = (store.x + store.w + STAMP_GAP) if store else geo.mx
    y = geo.safe_top + STAMP_TOP_DROP
    limit = (area.x - STAMP_GAP) if area else geo.w - geo.mx
    if x + w > limit:
        return None
    return Block(
        "stamp", [season.stamp], "store", STAMP_SIZE, x, y, w, h, angle=STAMP_ANGLE, align="center"
    )


def layout_fixed(brief: Brief, fmt: str, ms: Measurer) -> Fixed:
    geo = geometry(fmt)
    W, H, mx = geo.w, geo.h, geo.mx
    content_w = W - 2 * mx
    blocks: list[Block] = []
    notes: list[str] = []
    row_h = round(STORE_SIZE * 1.4)
    blocks.append(
        Block(
            "store",
            [brief.store_name],
            "store",
            STORE_SIZE,
            mx,
            geo.safe_top,
            ms.width("store", brief.store_name, STORE_SIZE),
            row_h,
        )
    )
    if brief.area_line:
        aw = ms.width("area", brief.area_line, STORE_SIZE)
        blocks.append(
            Block(
                "area",
                [brief.area_line],
                "area",
                STORE_SIZE,
                W - mx - aw,
                geo.safe_top,
                aw,
                row_h,
                align="right",
            )
        )
    stamp = place_stamp(brief.season, blocks, geo, ms)
    if brief.season and brief.season.stamp and stamp is None:
        notes.append("季節のスタンプを置く場所が無い")
    if stamp is not None:
        blocks.append(stamp)
    # 見出し: 大きい段から、いちばん長い行が収まるものを選ぶ
    lines = list(brief.headline)
    steps = HEADLINE_STEPS[fmt]
    size = steps[-1]
    fit = False
    for cand in steps:
        if (
            len(lines) == 1
            and ms.width("head", lines[0], cand) > content_w * 0.92
            and cand == steps[0]
        ):
            lines = balance_split(lines[0], ms, cand)
        if max(ms.width("head", ln, cand) for ln in lines) <= content_w * 0.92:
            size, fit = cand, True
            break
    if not fit:
        size = steps[-1]
        notes.append("見出しが幅に収まらない")
    hw = max(ms.width("head", ln, size) for ln in lines)
    hy = geo.safe_top + row_h + 36
    head = Block(
        "headline",
        lines,
        "head",
        size,
        mx,
        hy,
        hw,
        len(lines) * size * HEAD_LINE_HEIGHT,
        line_widths=[ms.width("head", ln, size) for ln in lines],
    )
    blocks.append(head)
    bottom = head.y + head.h
    if brief.aside:
        lh = size * HEAD_LINE_HEIGHT
        line_rects = [
            (mx, head.y + i * lh, mx + ms.width("head", ln, size), head.y + (i + 1) * lh)
            for i, ln in enumerate(lines)
        ]
        asz, ay, beside = ASIDE_STEPS[0], head.y + head.h + 8, False
        for cand in ASIDE_STEPS:  # 見出しの行の右に収まる大きい段を選ぶ。無ければ見出しの下
            aw = ms.width("hand", brief.aside, cand)
            probe = Block(
                "aside",
                [brief.aside],
                "hand",
                cand,
                W - mx - aw,
                head.y + size * 0.1,
                aw,
                cand * 1.3,
                angle=ASIDE_ANGLE,
            )
            pp = probe.polygon()
            box = (pp[:, 0].min(), pp[:, 1].min(), pp[:, 0].max(), pp[:, 1].max())
            if not any(
                box[0] < r[2] + 20 and box[2] > r[0] - 8 and box[1] < r[3] + 8 and box[3] > r[1] - 8
                for r in line_rects
            ):
                asz, ay, beside = cand, probe.y, True
                break
        aw = ms.width("hand", brief.aside, asz)
        if aw > content_w * 0.9:
            asz = SCALE[2]
            aw = ms.width("hand", brief.aside, asz)
        if not beside:
            bottom = ay + asz * 1.3 + 10
        blocks.append(
            Block(
                "aside",
                [brief.aside],
                "hand",
                asz,
                W - mx - aw,
                ay,
                aw,
                asz * 1.3,
                angle=ASIDE_ANGLE,
                align="right",
            )
        )
    if brief.tagline:
        tsz = TAGLINE_STEPS[-1]
        for cand in TAGLINE_STEPS:
            if ms.width("hand", brief.tagline, cand) <= content_w:
                tsz = cand
                break
        tw = ms.width("hand", brief.tagline, tsz)
        th = tsz * 1.2 + 22  # 下線の分
        ty = H - geo.safe_bottom - th
        blocks.append(
            Block("tagline", [brief.tagline], "hand", tsz, mx, ty, tw, th, underline=True)
        )
        floor_y = ty - 24
    else:
        floor_y = H - geo.safe_bottom - 24
    ex = round(ITEM_EDGE_RATIO * W)
    region = (float(ex), float(bottom + 28), float(W - ex), float(floor_y))
    if region[3] - region[1] < 0.25 * H:
        notes.append("商品を置ける範囲が狭い")
    return Fixed(blocks, region, notes)


# ---------------------------------------------------------------- 商品（切り抜き）


@dataclass
class Sprite:
    id: str
    label: str
    png_path: Path
    alpha: np.ndarray  # 元の切り抜きのアルファ（高さを ALPHA_MAX_H までに縮めたもの）
    item_type: str | None = None
    hist: np.ndarray | None = None  # アルファで重み付けした L* の分布（画素から測れたとき）
    tone_hint: str | None = None  # ブリーフの tone（画素から測れないときの手がかり）

    @property
    def light(self) -> float | None:
        """商品の明るさ（L*。0〜100）。測れなければ tone の目安、それも無ければ None。"""
        return item_lightness(self.hist, self.tone_hint)


ALPHA_MAX_H = 700
# 切り抜きの判定の規則を変えたら上げる（古い判定のキャッシュを使わない）
CUTOUT_CACHE_VERSION = 4


TONE_MAX_H = 300  # 明るさを測るときに縮める高さ


def sprite_from_png(item: BriefItem, png: Path) -> Sprite:
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    alpha = np.ascontiguousarray(rgba[..., 3])
    small = rgba
    if small.shape[0] > TONE_MAX_H:
        s = TONE_MAX_H / small.shape[0]
        small = cv2.resize(
            small, (max(1, round(small.shape[1] * s)), TONE_MAX_H), interpolation=cv2.INTER_AREA
        )
    hist = tone_hist(small[..., :3], small[..., 3])
    if alpha.shape[0] > ALPHA_MAX_H:
        s = ALPHA_MAX_H / alpha.shape[0]
        alpha = cv2.resize(
            alpha, (max(1, round(alpha.shape[1] * s)), ALPHA_MAX_H), interpolation=cv2.INTER_AREA
        )
    return Sprite(
        item.id,
        item.label,
        png,
        np.ascontiguousarray(alpha),
        item.item_type,
        hist if hist.sum() > 0 else None,
        item.tone,
    )


def default_fetch(src: str) -> bytes:
    """画像の取得。URL か、ローカルのパス。外へ出るのはブリーフにある画像の URL だけ。"""
    if re.match(r"^https?://", src):
        import httpx2 as httpx

        r = httpx.get(src, timeout=60, follow_redirects=True)
        r.raise_for_status()
        return r.content
    return Path(src).expanduser().read_bytes()


def cache_key(src: str, data: bytes | None, item_type: str | None) -> str:
    h = hashlib.sha1()
    h.update(f"v{CUTOUT_CACHE_VERSION}:".encode())
    h.update((item_type or "").encode())
    h.update(b"\0")
    h.update(data if data is not None else src.encode())
    return h.hexdigest()[:20]


def prepare_sprites(
    brief: Brief,
    cache_dir: Path,
    *,
    cutter_factory: Callable | None = None,
    fetch: Callable[[str], bytes] = default_fetch,
    base_dir: Path | None = None,
    log: Callable = print,
) -> tuple[list[Sprite], list[dict]]:
    """各商品を切り抜く（キャッシュがあれば切り抜き直さない）。check / fail は外して理由を返す。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sprites: list[Sprite] = []
    dropped: list[dict] = []
    cutter = None
    for item in brief.items:
        if item.cutout:
            p = _resolve(item.cutout, base_dir)
            sprites.append(sprite_from_png(item, p))
            continue
        src = item.image
        local = None if re.match(r"^https?://", src) else _resolve(src, base_dir)
        data = local.read_bytes() if local else None
        key = cache_key(src, data, item.item_type)
        meta_path, png = cache_dir / f"{key}.json", cache_dir / f"{key}.png"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            if data is None:
                try:
                    data = fetch(src)
                    Image.open(io.BytesIO(data)).verify()
                except Exception as e:  # 取れない・画像でない商品は外して続ける
                    reason = f"画像を取得できなかった（{type(e).__name__}）"
                    dropped.append({"id": item.id, "status": "fail", "reasons": [reason]})
                    log(f"  画像 {item.id}: {reason}")
                    continue
            if cutter is None:
                cutter = (cutter_factory or _default_cutter)()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            res = cutter.cut(image, item.item_type)
            meta = {
                "status": res.status,
                "reasons": res.reasons,
                "quality": res.summary().get("quality"),
            }
            if res.rgba is not None:
                res.rgba.save(png)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False))
            log(f"  切り抜き {item.id}: {res.status} {'; '.join(res.reasons)}")
        if meta["status"] == "ok" and png.exists():
            sprites.append(sprite_from_png(item, png))
        else:
            dropped.append({"id": item.id, "status": meta["status"], "reasons": meta["reasons"]})
    if len(sprites) < MIN_ITEMS:
        raise CannotCompose(
            f"切り抜けた商品が{len(sprites)}点で、{MIN_ITEMS}点に足りません"
            f"（外した商品: {json.dumps(dropped, ensure_ascii=False)}）"
        )
    return sprites, dropped


def _resolve(path: str, base_dir: Path | None) -> Path:
    p = Path(path).expanduser()
    if p.exists() or not base_dir or p.is_absolute():
        return p
    return base_dir / p if (base_dir / p).exists() else p


def _default_cutter():
    from app import cutout

    return cutout.Cutter()


# ---------------------------------------------------------------- 配置


@dataclass
class ItemState:
    sprite: Sprite
    h: float
    cx: float
    cy: float
    angle: float
    w: float = 0.0
    contour: np.ndarray | None = None  # 輪郭（商品の座標。画面の座標へは to_canvas）
    fs: int = LABEL_STEPS[0]
    arc: np.ndarray | None = None
    text_len: float = 0.0
    cost: float = 0.0
    placed: bool = False
    glow: bool = False  # 地に沈むので、縁にやわらかい光の縁取りを足す（画素は変えない）

    def __post_init__(self):
        self.w = self.sprite.alpha.shape[1] * self.h / self.sprite.alpha.shape[0]

    def half_extent(self) -> tuple[float, float]:
        c, s = abs(math.cos(math.radians(self.angle))), abs(math.sin(math.radians(self.angle)))
        return (self.w * c + self.h * s) / 2, (self.w * s + self.h * c) / 2

    def mask(self, size: tuple[int, int]) -> np.ndarray:
        return item_mask(self.sprite.alpha, self.cx, self.cy, self.h, self.angle, size)


def ellipse_radius(a: float, b: float, ux: float, uy: float) -> float:
    return 1.0 / math.sqrt((ux / a) ** 2 + (uy / b) ** 2)


def initial_centers(rng: random.Random, n: int, region: tuple) -> list[tuple[float, float]]:
    """範囲をほぼ格子に割り、互い違いにずらして揺らした中心（順番は種で入れ替える）。"""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    cols = max(1, round(math.sqrt(n * rw / rh)))
    rows = math.ceil(n / cols)
    cells = []
    for r in range(rows):
        k = min(cols, n - r * cols)
        for j in range(k):
            cx = x0 + (j + 0.5) * rw / k
            cy = y0 + (r + 0.5) * rh / rows
            cy += (-1) ** (r + j) * 0.12 * rh / rows
            cx += rng.uniform(-0.1, 0.1) * rw / cols
            cy += rng.uniform(-0.08, 0.08) * rh / rows
            cells.append((cx, cy))
    rng.shuffle(cells)
    return cells


def make_items(
    sprites: list[Sprite], region: tuple, rng: random.Random, scale: float
) -> list[ItemState]:
    n = len(sprites)
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    mult = [rng.uniform(0.85, 1.15) for _ in sprites]
    if n >= 3:
        mult[rng.randrange(n)] *= 1.3  # 主役を1つ
    aspects = [s.alpha.shape[1] / s.alpha.shape[0] for s in sprites]
    fill = FILL_RATIO_TALL if rh > 1.15 * rw else FILL_RATIO
    k = fill * rw * rh * scale * scale / sum(m * m for m in mult)
    items = []
    centers = initial_centers(rng, n, region)
    for i, sp in enumerate(sprites):
        h = math.sqrt(k * mult[i] ** 2 / aspects[i])
        h = min(h, MAX_ITEM_H_OF_REGION * rh, MAX_ITEM_W_OF_REGION * rw / aspects[i])
        sign = 1 if (i + rng.random() > 0.5) % 2 == 0 else -1
        ang = sign * rng.uniform(ANGLE_MIN, ANGLE_MAX)
        items.append(ItemState(sp, h, centers[i][0], centers[i][1], ang))
    return items


def clamp_item(it: ItemState, region: tuple) -> None:
    hx, hy = it.half_extent()
    x0, y0, x1, y1 = region
    it.cx = min(max(it.cx, x0 + hx), max(x0 + hx, x1 - hx))
    it.cy = min(max(it.cy, y0 + hy), max(y0 + hy, y1 - hy))


def spread_vertically(items: list[ItemState], region: tuple) -> None:
    """縦長の範囲では、商品の並びを縦に引き伸ばして、範囲の上から下までいっぱいに使う。"""
    x0, y0, x1, y1 = region
    tops = min(it.cy - it.half_extent()[1] for it in items)
    bottoms = max(it.cy + it.half_extent()[1] for it in items)
    if bottoms - tops < 1 or bottoms - tops >= y1 - y0:
        return
    k = (y1 - y0) / (bottoms - tops)
    for it in items:
        it.cy = y0 + (it.cy - tops) * k + 0.0
    for it in items:
        clamp_item(it, region)


def relax(items: list[ItemState], region: tuple) -> None:
    """商品どうしが重なりすぎないよう、楕円で押し合わせる（少しの重なりは残す）。"""
    for it in items:
        clamp_item(it, region)
    for _ in range(RELAX_ITERS):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                dx, dy = b.cx - a.cx, b.cy - a.cy
                d = math.hypot(dx, dy)
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                ux, uy = dx / d, dy / d
                ha, hb = a.half_extent(), b.half_extent()
                need = (
                    ellipse_radius(ELLIPSE_FACTOR * ha[0], ELLIPSE_FACTOR * ha[1], ux, uy)
                    + ellipse_radius(ELLIPSE_FACTOR * hb[0], ELLIPSE_FACTOR * hb[1], ux, uy)
                    + ITEM_GAP
                )
                if d < need:
                    push = (need - d) / 2
                    a.cx -= ux * push
                    a.cy -= uy * push
                    b.cx += ux * push
                    b.cy += uy * push
                    moved = True
        for it in items:
            clamp_item(it, region)
        if not moved:
            break


@dataclass
class Occupancy:
    """配置の途中の占有マップ。items=商品の和、text=文字（固定の文字と輪郭の文字）の和。"""

    size: tuple[int, int]
    safe: tuple[int, int, int, int]
    items: np.ndarray
    text: np.ndarray


def window_cost(band: Band, occ: Occupancy, pts: np.ndarray) -> tuple[float, float]:
    """(重なり・はみ出しの画素数, 罰を含む費用)。"""
    bad = band.overlap(occ.items) + band.overlap(occ.text) + SAFE_PENALTY * band.outside(occ.safe)
    return bad, bad + UPSIDE_PENALTY * upside_fraction(pts) + CONCAVE_PENALTY * concave_turn(pts)


def best_window(
    it: ItemState, label_w: float, fs: int, occ: Occupancy
) -> tuple[float, float, int, np.ndarray, Band] | None:
    """輪郭の上で、文字が流れる区間をいちばん良い位置に選ぶ。(bad, cost, start, arc 点, 帯)。"""
    pts = to_canvas(it.contour, it.w, it.h, it.cx, it.cy, it.angle)
    n = len(pts)
    text_len = label_w * TEXT_LEN_SAFETY
    m = int(math.ceil(text_len * 1.04 / CONTOUR_STEP)) + 1
    if m > n - 2:
        return None
    best = None
    for s in range(0, n, 2):
        idx = (s + np.arange(m)) % n
        seg = pts[idx]
        band = raster_band(seg, fs)
        bad, cost = window_cost(band, occ, seg)
        cost += 0.01 * s  # 同じなら、上の左から始まる（読みやすい）ほうへ
        if best is None or cost < best[1]:
            best = (bad, cost, s, seg, band)
    return best


def _place_text(
    it: ItemState, ms: Measurer, occ: Occupancy, steps: tuple[int, ...] = LABEL_STEPS
) -> bool:
    """商品 it の文字の区間を選んで占有マップに書く。許容内に収まれば True。"""
    for fs in steps:
        label_w = ms.width("hand", it.sprite.label, fs)
        best = best_window(it, label_w, fs, occ)
        if best is None:
            continue
        bad, cost, start, seg, band = best
        if bad <= TOLERANCE_PX:
            pts = to_canvas(it.contour, it.w, it.h, it.cx, it.cy, it.angle)
            n = len(pts)
            m_arc = min(
                n - 1, int(math.ceil(label_w * TEXT_LEN_SAFETY * ARC_SLACK / CONTOUR_STEP)) + 1
            )
            it.arc = pts[(start + np.arange(m_arc)) % n]
            it.fs, it.text_len, it.cost, it.placed = fs, label_w * TEXT_LEN_SAFETY, cost, True
            band.paint(occ.text)
            return True
    return False


def item_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    small = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    return inter / small if small else 0.0


def place_texts(
    items: list[ItemState],
    fixed: Fixed,
    geo: Geo,
    ms: Measurer,
    rng: random.Random,
    steps: tuple[int, ...] = LABEL_STEPS,
) -> tuple[bool, float]:
    """商品ごとに、輪郭の文字の区間を選ぶ。置けない商品は、少し動かして探し直す。"""
    size = (geo.w, geo.h)
    text = np.zeros((geo.h, geo.w), bool)
    for b in fixed.blocks:
        text |= b.mask(size)
    for it in items:
        it.contour = contour_local(it.sprite.alpha, round(it.h))[1]
    ok_all = True
    placed_masks: dict[int, np.ndarray] = {}
    order = sorted(range(len(items)), key=lambda i: -items[i].h)
    occ = Occupancy(size, geo.safe_rect, np.zeros_like(text), text)
    for it in items:
        occ.items |= it.mask(size)
    for i in order:
        it = items[i]
        if _place_text(it, ms, occ, steps):
            continue
        # 動かして探し直す（近い所から）
        done = False
        orig = (it.cx, it.cy)
        for r in (30, 60, 100, 150):
            for k in range(12):
                a = 2 * math.pi * (k + rng.random()) / 12
                it.cx, it.cy = orig[0] + r * math.cos(a), orig[1] + r * math.sin(a)
                clamp_item(it, fixed.region)
                mine = it.mask(size)
                others = [items[j] for j in range(len(items)) if j != i]
                if any(
                    item_overlap_ratio(mine, o.mask(size)) > MAX_ITEM_OVERLAP * 0.8 for o in others
                ):
                    continue
                occ2 = Occupancy(size, geo.safe_rect, np.zeros_like(text), occ.text)
                for o in others:
                    occ2.items |= o.mask(size)
                occ2.items |= mine
                if np.count_nonzero(mine & occ.text) > TOLERANCE_PX:
                    continue
                if _place_text(it, ms, occ2, steps):
                    occ.items = occ2.items
                    done = True
                    break
            if done:
                break
        if not done:
            it.cx, it.cy = orig
            ok_all = False
    del placed_masks
    return ok_all, sum(it.cost for it in items if it.placed)


# ---------------------------------------------------------------- 落書き


def _spiral() -> str:
    pts = []
    for t in np.linspace(0, 2.6 * 2 * math.pi, 60):
        r = 3 + 6.2 * t / (2 * math.pi) * 3.0
        pts.append(f"{r * math.cos(t):.1f} {r * math.sin(t):.1f}")
    return "M" + " L".join(pts)


# 単位の箱（−50〜50 を中心に置く）の落書き。(パス, 横, 縦) は当たり判定の大きさ
DOODLES = {
    "star": ("M0 -50 L12 -12 L50 0 L12 12 L0 50 L-12 12 L-50 0 L-12 -12Z", 100, 100),
    "wave": ("M-50 0 q12.5 -22 25 0 t25 0 t25 0 t25 0", 100, 28),
    "spiral": (_spiral(), 64, 64),
    "arrow": ("M-45 6 q42 32 86 -6 m-24 -8 l24 8 -6 24", 92, 50),
    "burst": ("M0 -42 L0 -18 M36 -22 L18 -9 M-36 -22 L-18 -9 M0 0 L0 0", 80, 40),
}


def shape_spec(kind: str) -> ad_season.Shape:
    """落書きの形。汎用（星・波・渦・矢印）は線だけ、季節の物は app/ad_season.py の形。"""
    if kind in ad_season.SHAPES:
        return ad_season.SHAPES[kind]
    path, bw, bh = DOODLES[kind]
    return ad_season.Shape(((path, "s"),), bw, bh, DOODLE_ROT)


@dataclass
class Doodle:
    kind: str
    cx: float
    cy: float
    size: float
    angle: float

    def polygon(self, pad: float = 0.0) -> np.ndarray:
        sp = shape_spec(self.kind)
        k = self.size / 100
        return rect_polygon(
            self.cx - sp.w * k / 2, self.cy - sp.h * k / 2, sp.w * k, sp.h * k, self.angle, pad
        )


def doodle_sequence(rng: random.Random, season: Season | None) -> tuple[list[str], int]:
    """置く落書きの種類の並び（先頭から置けた分だけ使う）と、置く数。季節があれば季節の物が主。"""
    generic = list(DOODLES)
    if season is None:
        rng.shuffle(generic)
        return generic * 2, rng.randint(*DOODLE_COUNT)
    own = list(ad_season.doodle_kinds(season))
    rng.shuffle(own)
    rng.shuffle(generic)
    want = rng.randint(*SEASON_DOODLE_COUNT)
    seq: list[str] = []
    for _ in range(want * 3):  # 置けない物があるぶん多めに並べる
        pool = own if (rng.random() < ad_season.SEASON_SHARE and own) else generic
        k = pool.pop(0)
        pool.append(k)  # 一巡するまで同じ物を重ねない
        seq.append(k)
    return seq, want


def place_doodles(
    rng: random.Random,
    geo: Geo,
    occupied: np.ndarray,
    region: tuple,
    season: Season | None = None,
) -> list[Doodle]:
    """空いている所に、種から選んだ数と種類の落書きを置く（占有マップで空きを探す）。
    季節があれば季節の落書きが主（汎用の星・波・矢印も混ぜる）。大きさと傾きにばらつきを付ける。"""
    size = (geo.w, geo.h)
    d = int(DOODLE_CLEAR * 2 + 1)
    blocked = cv2.dilate(
        occupied.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    ).astype(bool)
    sx0, sy0, sx1, sy1 = geo.safe_rect
    x0, y0, x1, y1 = region
    kinds, want = doodle_sequence(rng, season)
    size_range = SEASON_DOODLE_SIZE if season else DOODLE_SIZE
    out: list[Doodle] = []
    for kind in kinds:
        if len(out) >= want:
            break
        if (
            season
            and kind in DOODLES
            and sum(1 for o in out if o.kind in DOODLES) >= SEASON_GENERIC_MAX
        ):
            continue
        rot_max = shape_spec(kind).rot if season else DOODLE_ROT
        for _ in range(DOODLE_TRIES // 4):
            dd = Doodle(
                kind,
                rng.uniform(sx0, sx1),
                rng.uniform(max(sy0, y0 - 60), min(sy1, y1 + 40)),
                rng.uniform(*size_range) * geo.w / 1080,
                rng.uniform(-rot_max, rot_max),
            )
            poly = dd.polygon(pad=4)
            if (
                poly[:, 0].min() < sx0
                or poly[:, 0].max() > sx1
                or poly[:, 1].min() < sy0
                or poly[:, 1].max() > sy1
            ):
                continue
            m = polygon_mask(poly, size)
            if np.count_nonzero(m & blocked):
                continue
            if any(
                np.hypot(dd.cx - o.cx, dd.cy - o.cy) < 0.5 * (dd.size + o.size) + 40 for o in out
            ):
                continue
            out.append(dd)
            occupied |= m
            blocked |= cv2.dilate(
                m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            ).astype(bool)
            break
    return out


# ---------------------------------------------------------------- レイアウト（出力・検査の入力）


@dataclass
class Layout:
    format: str
    w: int
    h: int
    seed: int
    palette: str
    blocks: list[Block]
    region: tuple[float, float, float, float]
    items: list[ItemState]
    doodles: list[Doodle]
    notes: list[str] = field(default_factory=list)
    attempts: int = 1
    scale: float = 1.0
    palette_requested: str = ""
    palette_changed_reason: str | None = None
    season: Season | None = None

    def to_json(self) -> dict:
        items = []
        for it in self.items:
            items.append(
                {
                    "id": it.sprite.id,
                    "label": it.sprite.label,
                    "cx": round(it.cx, 1),
                    "cy": round(it.cy, 1),
                    "w": round(it.w, 1),
                    "h": round(it.h, 1),
                    "angle": round(it.angle, 2),
                    "light": None if it.sprite.light is None else round(it.sprite.light, 1),
                    "glow": it.glow,
                    "label_size": it.fs,
                    "text_placed": it.placed,
                    "text_len": round(it.text_len, 1),
                    "text_cost": round(it.cost, 1),
                    "arc": None if it.arc is None else np.round(it.arc, 1).tolist(),
                }
            )
        return {
            "format": self.format,
            "size": [self.w, self.h],
            "seed": self.seed,
            "palette": self.palette,
            "palette_used": self.palette,
            "palette_requested": self.palette_requested or self.palette,
            "palette_changed_reason": self.palette_changed_reason,
            "season": None if self.season is None else self.season.to_json(),
            "region": [round(v, 1) for v in self.region],
            "blocks": [b.to_json() for b in self.blocks],
            "items": items,
            "doodles": [
                {**asdict(d), "polygon": np.round(d.polygon(), 1).tolist()} for d in self.doodles
            ],
            "notes": self.notes,
            "attempts": self.attempts,
            "scale": round(self.scale, 3),
        }


def build_layout(brief: Brief, sprites: list[Sprite], fmt: str, seed: int, ms: Measurer) -> Layout:
    """1つの種から、1つの候補を組む。文字が収まらなければ商品を少し縮めてやり直す。"""
    geo = geometry(fmt)
    fixed = layout_fixed(brief, fmt, ms)
    best: tuple[float, Layout] | None = None
    scale = 1.0
    for attempt in range(1, ATTEMPTS + 1):
        rng = random.Random(f"{seed}:{attempt}")
        items = make_items(sprites, fixed.region, rng, scale)
        relax(items, fixed.region)
        if fixed.region[3] - fixed.region[1] > 1.15 * (fixed.region[2] - fixed.region[0]):
            spread_vertically(items, fixed.region)
            relax(items, fixed.region)
        steps = LABEL_STEPS if attempt == ATTEMPTS else LABEL_STEPS[:1]
        ok, cost = place_texts(items, fixed, geo, ms, rng, steps)
        unplaced = sum(1 for it in items if not it.placed)
        bg = PALETTES[brief.palette].bg
        for it in items:
            it.glow = sinks(it.sprite.light, bg)
        lay = Layout(
            fmt,
            geo.w,
            geo.h,
            seed,
            brief.palette,
            fixed.blocks,
            fixed.region,
            items,
            [],
            list(fixed.notes),
            attempt,
            scale,
            brief.palette_requested or brief.palette,
            brief.palette_changed_reason,
            brief.season,
        )
        rank = unplaced * 1e6 + cost
        if best is None or rank < best[0]:
            best = (rank, lay)
        if ok:
            break
        scale *= SHRINK
    lay = best[1]
    occ = np.zeros((geo.h, geo.w), bool)
    for b in lay.blocks:
        occ |= b.mask((geo.w, geo.h))
    for it in lay.items:
        occ |= it.mask((geo.w, geo.h))
        if it.arc is not None:
            raster_band(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)], it.fs).paint(occ)
    lay.doodles = place_doodles(random.Random(f"{seed}:doodle"), geo, occ, lay.region, brief.season)
    return lay


# ---------------------------------------------------------------- 検査


@dataclass
class Checks:
    text_on_item_px: int = 0
    text_text_px: int = 0
    out_of_canvas_px: int = 0
    outside_safe_px: int = 0
    items_outside_px: int = 0
    item_overlap_max: float = 0.0
    min_item_h: float = 0.0
    small_items: list[str] = field(default_factory=list)
    unplaced_text: list[str] = field(default_factory=list)
    headline_lines: int = 0
    headline_fits: bool = True
    doodle_overlap_px: int = 0
    problems: list[str] = field(default_factory=list)
    verdict: str = "pass"
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)

    def violations(self) -> int:
        return len(self.problems)


def check_layout(lay: Layout, ms: Measurer | None = None) -> Checks:
    """配置の規則の検査。占有マップをレイアウトから作り直して数える（配置の途中の値は使わない）。"""
    W, H = lay.w, lay.h
    size = (W, H)
    geo = geometry(lay.format)
    c = Checks()
    pad = 300
    # 商品
    masks, outside = [], 0
    edge = round(ITEM_EDGE_RATIO * W)
    for it in lay.items:
        big = item_mask(it.sprite.alpha, it.cx, it.cy, it.h, it.angle, size, pad=pad)
        total = int(np.count_nonzero(big))
        inside = big[pad + edge : pad + H - edge, pad + edge : pad + W - edge]
        out_px = total - int(np.count_nonzero(inside))
        if out_px > ITEM_OUTSIDE_RATIO * total:
            outside += out_px
        masks.append(big[pad : pad + H, pad : pad + W])
    c.items_outside_px = outside
    if outside:
        c.problems.append(f"商品が画面の端からはみ出している（{outside}px）")
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            c.item_overlap_max = max(c.item_overlap_max, item_overlap_ratio(masks[i], masks[j]))
    c.item_overlap_max = round(c.item_overlap_max, 3)
    if c.item_overlap_max > MAX_ITEM_OVERLAP:
        c.problems.append(f"商品どうしが重なりすぎ（{c.item_overlap_max:.0%}）")
    min_h = MIN_ITEM_H_RATIO * W
    c.min_item_h = round(min(it.h for it in lay.items), 1)
    c.small_items = [it.sprite.id for it in lay.items if it.h < min_h]
    if c.small_items:
        c.problems.append(f"商品が小さすぎる（高さ {c.min_item_h:.0f}px < {min_h:.0f}px）")
    union = np.zeros((H, W), bool)
    for m in masks:
        union |= m
    # 文字: 固定の文字の矩形と、輪郭の帯
    texts: list[tuple[str, np.ndarray]] = []
    for b in lay.blocks:
        texts.append((b.name, b.mask(size)))
    for it in lay.items:
        if it.arc is None:
            c.unplaced_text.append(it.sprite.id)
            continue
        band = raster_band(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)], it.fs)
        c.out_of_canvas_px += band.outside((0, 0, W, H))
        c.outside_safe_px += band.outside(geo.safe_rect)
        texts.append((f"label:{it.sprite.id}", band.full(W, H)))
    if c.unplaced_text:
        c.problems.append(f"輪郭の文字を置けない商品がある（{', '.join(c.unplaced_text)}）")
    for b in lay.blocks:
        px = b.polygon(pad=0)
        sx0, sy0, sx1, sy1 = geo.safe_rect
        if (
            px[:, 0].min() < sx0 - 8
            or px[:, 0].max() > sx1 + 8
            or px[:, 1].min() < sy0 - 8
            or px[:, 1].max() > sy1 + 8
        ):  # 回した矩形の角は 8px まで許す
            c.outside_safe_px += 1000
    c.text_on_item_px = sum(int(np.count_nonzero(m & union)) for _, m in texts)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            c.text_text_px += int(np.count_nonzero(texts[i][1] & texts[j][1]))
    if c.text_on_item_px > TOLERANCE_PX:
        c.problems.append(f"文字が商品に重なっている（{c.text_on_item_px}px）")
    if c.text_text_px > TOLERANCE_PX:
        c.problems.append(f"文字どうしが重なっている（{c.text_text_px}px）")
    if c.out_of_canvas_px > TOLERANCE_PX:
        c.problems.append(f"文字が画面からはみ出している（{c.out_of_canvas_px}px）")
    if c.outside_safe_px > TOLERANCE_PX:
        c.problems.append(f"文字が安全域の外にある（{c.outside_safe_px}px）")
    head = next((b for b in lay.blocks if b.name == "headline"), None)
    c.headline_lines = len(head.lines) if head else 0
    c.headline_fits = not any("見出しが幅に収まらない" in n for n in lay.notes)
    if not head or c.headline_lines > HEADLINE_MAX_LINES or not c.headline_fits:
        c.problems.append("見出しが2行に収まらない")
    # 落書き
    occ_all = union.copy()
    for _, m in texts:
        occ_all |= m
    c.doodle_overlap_px = sum(
        int(np.count_nonzero(polygon_mask(d.polygon(), size) & occ_all)) for d in lay.doodles
    )
    if c.doodle_overlap_px > TOLERANCE_PX:
        c.problems.append(f"落書きが商品や文字に重なっている（{c.doodle_overlap_px}px）")
    c.verdict = "pass" if not c.problems else "fail"
    c.score, c.score_parts = score_layout(lay, union)
    return c


def score_layout(lay: Layout, union: np.ndarray) -> tuple[float, dict]:
    """点数（0〜100）。余白のつり合い・商品の大きさのばらつき・埋まり具合・縦の使い切り。簡単な目安。

    use = 商品が置ける範囲の高さに占める、商品の縦の広がりの割合。縦長（9:16）では大きな空きが
    片側に残る候補（上下の空きの偏り）を選ばないよう、重みを大きくする。"""
    x0, y0, x1, y1 = lay.region
    rw, rh = x1 - x0, y1 - y0
    ys, xs = np.where(union)
    if len(xs) == 0:
        return 0.0, {}
    ml, mr = xs.min() - x0, x1 - xs.max()
    mt, mb = ys.min() - y0, y1 - ys.max()
    balance = max(0.0, 1 - (abs(ml - mr) / rw + abs(mt - mb) / rh))
    hs = np.array([it.h for it in lay.items])
    cv = float(hs.std() / hs.mean())
    variety = max(0.0, 1 - abs(cv - 0.22) / 0.22)
    cover = float(np.count_nonzero(union[int(y0) : int(y1), int(x0) : int(x1)])) / (rw * rh)
    fill = max(0.0, 1 - abs(cover - 0.4) / 0.4)
    upside = float(
        np.mean([upside_fraction(it.arc) for it in lay.items if it.arc is not None] or [0.0])
    )
    readable = 1 - upside
    tall = rh > 1.15 * rw
    use = min(1.0, float(ys.max() - ys.min()) / (rh * USE_TARGET))
    vbal = max(0.0, 1 - abs(mt - mb) / (0.25 * rh))  # 上下の空きの差が範囲の 1/4 なら 0
    parts = {
        "balance": round(balance, 3),
        "variety": round(variety, 3),
        "fill": round(fill, 3),
        "readable": round(readable, 3),
        "use": round(use, 3),
        "vbalance": round(vbal, 3),
        "cover": round(cover, 3),
    }
    if tall:
        score = 100 * (
            0.2 * balance + 0.1 * variety + 0.2 * fill + 0.1 * readable + 0.2 * use + 0.2 * vbal
        )
    else:
        score = 100 * (0.35 * balance + 0.2 * variety + 0.3 * fill + 0.15 * readable)
    # 商品と地の明暗の差（沈んだ商品が多いほど下がる。光の縁取りで救った分は最低 0.6）。
    # 画素もブリーフの tone も無く測れないときは混ぜない（従来の点数のまま）
    contrast = contrast_score(
        [it.sprite.light for it in lay.items],
        PALETTES[lay.palette].bg,
        [it.glow for it in lay.items],
    )
    if contrast is not None:
        parts["contrast"] = round(contrast, 3)
        score = (1 - CONTRAST_WEIGHT) * score + 100 * CONTRAST_WEIGHT * contrast
    return round(score, 1), parts


# ---------------------------------------------------------------- 描画（HTML + SVG → Chrome）


def font_face_css(fonts_dir: Path) -> str:
    def face(family: str, file: str, extra: str = "") -> str:
        p = fonts_dir / file
        if not p.exists():
            return ""
        return f'@font-face{{font-family:"{family}";src:url({p.resolve().as_uri()});{extra}}}\n'

    return (
        face("DM Sans", "DMSans.ttf", "font-weight:100 1000")
        + face("DM Sans", "DMSans-Italic.ttf", "font-weight:100 1000;font-style:italic")
        + face("Yomogi", "Yomogi.ttf")
        + face("Zen Kurenaido", "ZenKurenaido.ttf")
        + face("Caveat", "Caveat.ttf", "font-weight:400 700")
        + face("Gaegu", "Gaegu-Bold.ttf", "font-weight:700")
    )


def _wave_path(x: float, y: float, w: float) -> str:
    seg = 90.0
    n = max(2, int(round(w / seg)))
    d = f"M{x:.1f} {y:.1f}"
    dx = w / n
    for i in range(n):
        d += f" q{dx / 4:.1f} {-8 if i % 2 == 0 else 8} {dx / 2:.1f} 0 t{dx / 2:.1f} 0"
    return d


def stamp_frame_d(rx: float, ry: float, seed_text: str) -> str:
    """スタンプの手描きの楕円の枠。少しゆらし、終わりを始めより先へ重ねる（同じ文字なら同じ形）。"""
    rnd = random.Random(seed_text)
    n = 36
    pts = []
    for i in range(n + 3):  # 1周より少し先まで（ペンの重なり）
        a = 2 * math.pi * i / n - math.pi / 2
        j = 1 + rnd.uniform(-0.02, 0.02) + 0.03 * (i / n)
        pts.append((rx * j * math.cos(a), ry * j * math.sin(a)))
    return ad_season.smooth(pts)


def arc_path_d(arc: np.ndarray) -> str:
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in arc)


def render_html(lay: Layout, fonts_dir: Path) -> str:
    pal = PALETTES[lay.palette]
    W, H = lay.w, lay.h
    e = html.escape
    css = f"""{font_face_css(fonts_dir)}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:{W}px;height:{H}px;overflow:hidden}}
body{{position:relative;background:{pal.bg};color:{pal.text};
font-family:"DM Sans","Noto Sans CJK JP",sans-serif;font-feature-settings:"palt";
line-break:strict;-webkit-font-smoothing:antialiased}}
.b{{position:absolute;white-space:nowrap}}
.store{{font-weight:700;letter-spacing:.12em;font-size:{STORE_SIZE}px;line-height:1.4}}
.area{{font-weight:500;letter-spacing:.06em;font-size:{STORE_SIZE}px;line-height:1.4}}
.head{{font-weight:900;line-height:{HEAD_LINE_HEIGHT};letter-spacing:-.01em}}
.hand{{font-family:"Caveat","Yomogi";font-weight:700;line-height:1.2;letter-spacing:.03em}}
.it{{position:absolute}}
.it img{{position:absolute;left:0;top:0;width:100%;height:100%;
filter:drop-shadow(0 14px 16px {pal.shadow})}}
svg.ov{{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}}
svg.ov .d{{fill:none;stroke:{pal.doodle};stroke-width:{DOODLE_STROKE};stroke-linecap:round;
stroke-linejoin:round;vector-effect:non-scaling-stroke}}
svg.ov .da{{fill:{pal.accent};stroke:{pal.doodle};stroke-width:{DOODLE_STROKE};
stroke-linejoin:round;vector-effect:non-scaling-stroke}}
svg.ov .stamp{{fill:none;stroke:{pal.accent};stroke-width:3;stroke-linecap:round;
vector-effect:non-scaling-stroke}}
svg.ov text.stampt{{fill:{pal.text};font-family:"DM Sans","Noto Sans CJK JP",sans-serif;
font-weight:700;letter-spacing:.1em;text-anchor:middle}}
svg.ov text{{fill:{pal.text};font-family:"Caveat","Yomogi";font-weight:700;letter-spacing:.03em}}
.acc{{color:{pal.accent}}}
"""
    body: list[str] = []
    glow = glow_color(pal)
    for it in lay.items:
        # 沈む商品だけ、地よりわずかに明るい（明るい地では暗い）やわらかい光の縁取りを足す。
        # 商品の画素は変えない（周りに光を足すだけ）
        img_style = ""
        if it.glow:
            blur = " ".join(f"drop-shadow(0 0 {r}px {glow})" for r in _GLOW_BLUR)
            img_style = f' style="filter:{blur} drop-shadow(0 14px 16px {pal.shadow})"'
        body.append(
            f'<div class="it" style="left:{it.cx - it.w / 2:.1f}px;top:{it.cy - it.h / 2:.1f}px;'
            f'width:{it.w:.1f}px;height:{it.h:.1f}px;transform:rotate({it.angle:.2f}deg)">'
            f'<img src="{it.sprite.png_path.resolve().as_uri()}"{img_style}></div>'
        )
    for b in lay.blocks:
        if b.name == "stamp":
            continue  # SVG の側で描く（枠つき）
        cls = {
            "store": "store",
            "area": "area",
            "headline": "head",
            "aside": "hand",
            "tagline": "hand",
        }[b.name]
        pos = f"top:{b.y:.1f}px;"
        pos += f"right:{W - (b.x + b.w):.1f}px;" if b.align == "right" else f"left:{b.x:.1f}px;"
        style = f"{pos}font-size:{b.size}px;"
        if b.angle:
            style += f"transform:rotate({b.angle:.1f}deg);"
        if b.name == "aside":
            style += f"color:{pal.accent};"
        if b.name == "headline":
            rows = []
            for i, ln in enumerate(b.lines):
                txt = e(ln)
                if i == len(b.lines) - 1 and ln and ln[-1] in "。！？.!?":
                    txt = e(ln[:-1]) + f'<span class="acc">{e(ln[-1])}</span>'
                rows.append(txt)
            inner = "<br>".join(rows)
        else:
            inner = "<br>".join(e(ln) for ln in b.lines)
        body.append(f'<div class="b {cls}" style="{style}">{inner}</div>')
    svg = [f'<svg class="ov" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    for d in lay.doodles:
        k = d.size / 100
        layers = "".join(
            f'<path class="{"da" if mode == "a" else "d"}" d="{path}"/>'
            for path, mode in shape_spec(d.kind).layers
        )
        svg.append(
            f'<g transform="translate({d.cx:.1f} {d.cy:.1f}) rotate({d.angle:.1f}) '
            f'scale({k:.3f})">{layers}</g>'
        )
    for b in lay.blocks:
        if b.name == "stamp":
            cx, cy = b.x + b.w / 2, b.y + b.h / 2
            svg.append(
                f'<g transform="translate({cx:.1f} {cy:.1f}) rotate({b.angle:.1f})">'
                f'<path class="stamp" d="{stamp_frame_d(b.w / 2, b.h / 2, b.lines[0])}"/>'
                f'<text class="stampt" y="{b.size * 0.36:.1f}" font-size="{b.size}">'
                f"{e(b.lines[0])}</text></g>"
            )
    for b in lay.blocks:
        if b.underline:
            svg.append(f'<path class="d" d="{_wave_path(b.x, b.y + b.h - 6, b.w)}"/>')
    for i, it in enumerate(lay.items):
        if it.arc is None:
            continue
        svg.append(
            f'<path id="a{i}" d="{arc_path_d(it.arc)}" fill="none"/>'
            f'<text font-size="{it.fs}"><textPath href="#a{i}">{e(it.sprite.label)}'
            f"</textPath></text>"
        )
    svg.append("</svg>")
    return (
        f'<!doctype html><meta charset="utf-8"><style>{css}</style><body>'
        + "".join(body)
        + "".join(svg)
        + "</body>"
    )


def find_chrome(explicit: str | None = None) -> str:
    for c in (explicit, os.getenv("AD_CHROME"), "google-chrome", "chromium", "chromium-browser"):
        if c and (shutil.which(c) or Path(c).exists()):
            return c
    raise RuntimeError("Chrome が見つかりません。--chrome か環境変数 AD_CHROME で指定してください")


def screenshot(
    html_path: Path,
    png_path: Path,
    size: tuple[int, int],
    chrome: str | None = None,
    timeout: int = 90,
) -> None:
    exe = find_chrome(chrome)
    with tempfile.TemporaryDirectory() as prof:
        cmd = [
            exe,
            *CHROME_ARGS,
            f"--user-data-dir={prof}",
            f"--window-size={size[0]},{size[1]}",
            f"--screenshot={png_path.resolve()}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    im = Image.open(png_path)
    if im.size != size:  # 新しいヘッドレスは窓の枠の分だけずれることがある
        canvas = Image.new("RGB", size, "white")
        canvas.paste(im.convert("RGB").crop((0, 0, *size)))
        canvas.save(png_path)


# ---------------------------------------------------------------- 候補の選び・出力


@dataclass
class Candidate:
    seed: int
    layout: Layout
    checks: Checks
    png: Path | None = None
    critique: dict | None = None

    @property
    def rank_score(self) -> float:
        extra = 0.0
        if self.critique and self.critique.get("mean") is not None:
            extra = 6.0 * (self.critique["mean"] - 3.0)
        return self.checks.score + extra

    def sort_key(self):
        return (
            0 if self.checks.verdict == "pass" else 1,
            0 if self.checks.verdict == "pass" else self.checks.violations(),
            -self.rank_score,
        )


@dataclass
class FormatResult:
    format: str
    verdict: str  # pass / needs_review
    best: Candidate
    candidates: list[Candidate]
    dropped: list[dict]
    png: Path
    layout_json: Path
    seconds: float
    reasons: list[str]
    revision: object = None

    def meta(self) -> dict:
        """CRM へ返す meta（format・seed・score・verdict・checks・dropped_items・revision）。"""
        extra = {} if self.revision is None else {"revision": self.revision}
        lay = self.best.layout
        return {
            **extra,
            "format": self.format,
            "palette_used": lay.palette,
            "palette_changed_reason": lay.palette_changed_reason,
            "seed": self.best.seed,
            "score": self.best.checks.score,
            "verdict": self.verdict,
            "checks": asdict(self.best.checks),
            "dropped_items": self.dropped,
        }


def fmt_slug(fmt: str) -> str:
    return fmt.replace(":", "x")


def compose_format(
    brief: Brief,
    sprites: list[Sprite],
    dropped: list[dict],
    fmt: str,
    out_dir: Path,
    fonts_dir: Path,
    *,
    chrome: str | None = None,
    render_top: int = 4,
    critic=None,
    log: Callable = print,
    render: bool = True,
) -> FormatResult:
    t0 = time.time()
    ms = Measurer(fonts_dir)
    # 地に沈む商品があれば、先に色を決める（色は候補ごとに変えない。同じ版の 3 形式で揃う）
    used, reason = choose_palette(
        brief.palette, [sp.light for sp in sprites], brief.palette_options or None
    )
    if used != brief.palette:
        log(f"  色を替えた: {reason}")
        brief = replace(
            brief, palette=used, palette_requested=brief.palette, palette_changed_reason=reason
        )
    cands = []
    for k in range(brief.candidates):
        lay = build_layout(brief, sprites, fmt, brief.seed + k, ms)
        cands.append(Candidate(brief.seed + k, lay, check_layout(lay, ms)))
    cands.sort(key=Candidate.sort_key)
    work = out_dir / "_work" / f"{brief.id}_{fmt_slug(fmt)}"
    work.mkdir(parents=True, exist_ok=True)
    if render:
        for c in cands[:render_top]:
            hp = work / f"cand_{c.seed}.html"
            hp.write_text(render_html(c.layout, fonts_dir))
            c.png = work / f"cand_{c.seed}.png"
            screenshot(hp, c.png, (c.layout.w, c.layout.h), chrome)
        if critic is not None:
            _apply_critique(critic, cands[:3], log)
            cands.sort(key=Candidate.sort_key)
    best = cands[0]
    verdict = "pass" if best.checks.verdict == "pass" else "needs_review"
    reasons = list(best.checks.problems)
    png = out_dir / f"{brief.id}_{fmt_slug(fmt)}.png"
    if render and best.png:
        shutil.copyfile(best.png, png)
    layout_json = out_dir / f"{brief.id}_{fmt_slug(fmt)}.layout.json"
    layout_json.write_text(
        json.dumps(
            {
                **best.layout.to_json(),
                "brief_id": brief.orig_id or brief.id,
                "verdict": verdict,
                "reasons": reasons,
                "checks": asdict(best.checks),
                "dropped_items": dropped,
                "critique": best.critique,
                "candidates": [
                    {
                        "seed": c.seed,
                        "verdict": c.checks.verdict,
                        "score": c.checks.score,
                        "critique": c.critique,
                        "problems": c.checks.problems,
                    }
                    for c in cands
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return FormatResult(
        fmt,
        verdict,
        best,
        cands,
        dropped,
        png,
        layout_json,
        time.time() - t0,
        reasons,
        revision=brief.revision,
    )


def _apply_critique(critic, cands: list[Candidate], log: Callable) -> None:
    for c in cands:
        if c.png is None:
            continue
        try:
            c.critique = critic.score(c.png)
        except Exception as e:  # 講評は任意。読めない・失敗したら規則だけで進める
            log(f"  講評を飛ばす: {e}")
            return


def thumb_data_uri(png: Path, width: int = 360) -> str:
    im = Image.open(png).convert("RGB")
    im.thumbnail((width, width * 2))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def index_html(results: list[tuple[Brief, list[FormatResult]]]) -> str:
    e = html.escape
    out = [
        '<!doctype html><meta charset="utf-8"><title>ad compose</title><style>'
        "body{font:14px/1.5 sans-serif;margin:16px;background:#f4f4f4}"
        "section{margin:0 0 28px}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}"
        "figure{margin:0;background:#fff;padding:8px;border-radius:6px;width:380px}"
        "figure.best{outline:3px solid #2a7}img{width:100%;display:block}"
        "figcaption{font-size:12px;color:#444;margin-top:4px}.fail{color:#b22}.pass{color:#262}"
        "</style>"
    ]
    for brief, fresults in results:
        out.append(f"<section><h2>{e(brief.orig_id or brief.id)}</h2>")
        for fr in fresults:
            out.append(
                f"<h3>{e(fr.format)} — <span class='{fr.verdict[:4]}'>{e(fr.verdict)}</span> "
                f"{fr.seconds:.1f}秒</h3>"
            )
            if fr.dropped:
                out.append(
                    "<p>外した商品: " + e(json.dumps(fr.dropped, ensure_ascii=False)) + "</p>"
                )
            out.append('<div class="row">')
            for c in fr.candidates:
                if c.png is None:
                    continue
                cls = "best" if c is fr.best else ""
                probs = "; ".join(c.checks.problems) or "問題なし"
                out.append(
                    f'<figure class="{cls}"><img src="{thumb_data_uri(c.png)}">'
                    f"<figcaption>seed {c.seed} / {c.checks.verdict} / 点 {c.checks.score}"
                    f"<br>{e(probs)}</figcaption></figure>"
                )
            out.append("</div>")
        out.append("</section>")
    return "".join(out)


def compose_brief(
    brief: Brief,
    formats: list[str],
    out_dir: Path,
    fonts_dir: Path,
    cache_dir: Path,
    *,
    chrome: str | None = None,
    critic=None,
    cutter_factory: Callable | None = None,
    fetch=default_fetch,
    base_dir: Path | None = None,
    render_top: int = 4,
    log: Callable = print,
) -> list[FormatResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sprites, dropped = prepare_sprites(
        brief, cache_dir, cutter_factory=cutter_factory, fetch=fetch, base_dir=base_dir, log=log
    )
    results = []
    for fmt in formats:
        fr = compose_format(
            brief,
            sprites,
            dropped,
            fmt,
            out_dir,
            fonts_dir,
            chrome=chrome,
            critic=critic,
            render_top=render_top,
            log=log,
        )
        log(
            f"  {brief.orig_id or brief.id} {fmt}: {fr.verdict} 点{fr.best.checks.score}"
            f" {fr.seconds:.1f}秒" + (f" {'; '.join(fr.reasons)}" if fr.reasons else "")
        )
        results.append(fr)
    return results
