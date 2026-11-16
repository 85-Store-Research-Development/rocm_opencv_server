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

from app import ad_lineup, ad_season, ad_single, ad_worn
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
WORN_NOTE_MAX = 20  # 着画に添える文の長さ（字）
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
CONTOUR_OFFSET_PAD = (
    12  # 輪郭からの離れの下限は、文字の内側の張り出し（BAND_DOWN × 文字の大きさ）+ これ
)
CONTOUR_STEP = 6  # 輪郭の点の間隔（px）
UPSIDE_MAX = 0.1  # 逆さになる区間（下辺を右から左へ流れる所）がこの割合を超える弧は使わない
MAX_WINDOW_TURN = 4.5  # 1本の弧の曲がりの合計（rad）の上限。角を回り込むと文字が詰まる
# 着画（人ごとの切り抜き）。縦長なのでほかの商品より大きく、端に据えてほかの商品をそのまわりに散らす
WORN_H_RATIO = {"4:5": 0.46, "1:1": 0.52, "9:16": 0.38}  # 画面の高さに対する着画の高さ
WORN_H_MAX_OF_REGION = 0.88  # ただし商品を置ける範囲の高さのこの割合まで
WORN_W_MAX_OF_REGION = 0.42  # 幅も同じく
WORN_ANGLE = (1.0, 4.0)  # 傾き（±度）。小さめ
WORN_GAP = 26  # 着画とほかの商品の領域の間
WORN_LINK_GAP = 16  # 点線の矢印と商品の間
WORN_LINK_MAX = 0.40  # 点線の矢印の長さの上限（画面の幅に対する割合）
WORN_LINK_CLEAR = (
    20  # 点線の矢印が、ほかの商品・文字から空ける距離（px）。これ以内を通るなら引かない
)
# 着画つきで気持ちよく入る商品の点数の目安（形式ごと。着画の商品を含む）。超えるぶんは後ろから外す
WORN_MAX_ITEMS = {"4:5": 4, "1:1": 3, "9:16": 3}
PROBE_CANDIDATES = 4  # 点数を減らして試すとき、まずこの数で1つも通らなければ次の点数へ
LINEUP_WORN_MAX_DROP = 1  # lineup で着画のために外してよい商品の数
WORN_MIN_OTHERS = 2  # 着画の商品のアイテム写真のほかに、最低これだけは残す
SHORT_LABEL = re.compile(r"^(?P<brand>.+?) の (?P<rest>.+ ¥\d[\d,]*)$")
WORN_LINK_THICK = 18  # 点線の矢印が空いていなければならない太さ（px）
STRAIGHT_FROM_ATTEMPT = 3  # 商品を縮めてもだめなら、この試行から先は商品の上下にまっすぐ置いてよい
STRAIGHT_GAP = 14  # 輪郭に沿わせられないとき、商品の上下にまっすぐ置く文字との間
STRAIGHT_TILTS = (-3.0, 0.0, 3.0)
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
    images: list[str] = field(default_factory=list)  # その商品の全写真の URL（着画を探す。任意）


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
    worn_note: str = ""  # 着画に添える短い手書きの文（無ければ添えない）
    layout: str = (
        "collage"  # collage（複数を並べる・既定）/ single（1点もの）/ lineup（新品のまとめ見せ）
    )
    single: ad_single.Single | None = None  # layout が single のときの 価格・シール・メモ
    lineup: ad_lineup.Lineup | None = None  # layout が lineup のときの 型・色・サイズ・価格


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
    worn_note = _text(problems, "worn_note", raw.get("worn_note"), maxlen=WORN_NOTE_MAX)
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
        if raw.get("layout") != ad_lineup.LAYOUT_NAME:  # lineup の商品は lineup.groups にある
            problems.append("items は配列にしてください")
        raw_items = []
    layout = ad_single.LAYOUT_NAME if raw.get("layout") == ad_single.LAYOUT_NAME else "collage"
    single = None
    lineup = None
    lineup_ok = False
    if (
        raw.get("layout") == ad_lineup.LAYOUT_NAME
    ):  # 新品のまとめ見せ: 商品は lineup.groups[].items[]
        lineup, lp, raw_items = ad_lineup.parse_lineup(raw.get("lineup"))
        problems.extend(lp)
        lineup_ok = lineup is not None and not lp
        if lineup_ok:
            labels = {
                li.id: ad_lineup.item_label(lineup, g, li) or li.color or g.name
                for g in lineup.groups
                for li in g.items
            }
            raw_items = [{**r, "label": labels.get(r["id"], r["id"])} for r in raw_items]
            if lineup.style:
                layout = ad_lineup.LAYOUT_NAME
            else:  # 知らない style は collage に落とす（商品を並べるだけ）
                layout, lineup = "collage", None
                raw_items = raw_items[:MAX_ITEMS]
    if layout == "single":  # 1点もの: 主役は先頭の1点（ほかは使わない）
        single, single_problems = ad_single.parse_single(raw.get("single"))
        problems.extend(single_problems)
        if not 1 <= len(raw_items) <= MAX_ITEMS:
            problems.append(f"items は 1 点以上にしてください（{len(raw_items)} 点）")
        raw_items = raw_items[:1]
    elif layout == ad_lineup.LAYOUT_NAME or (lineup_ok and layout == "collage"):
        if not MIN_ITEMS <= len(raw_items) <= ad_lineup.MAX_ITEMS:
            problems.append(
                f"lineup の商品は {MIN_ITEMS} 点以上にしてください（{len(raw_items)} 点）"
            )
    elif not MIN_ITEMS <= len(raw_items) <= MAX_ITEMS and not lineup_ok:
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
        images = ad_worn.normalize_images(image, it.get("images"))
        items.append(
            BriefItem(
                iid,
                image,
                label,
                itype if isinstance(itype, str) else None,
                cut or None,
                tone if tone in ("dark", "mid", "light") else None,
                images,
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
        worn_note=worn_note,
        layout=layout,
        single=single,
        lineup=lineup,
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


def offset_for(fs: int) -> int:
    """輪郭から文字の土台までの距離。文字が大きいほど離す（文字が商品の内側へ食い込まないように）。"""
    return round(max(CONTOUR_OFFSET, BAND_DOWN * fs + CONTOUR_OFFSET_PAD))


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
            # 傾けた左端が見出しに食い込むときだけ、食い込まなくなるまで下げる
            head_mask = head.mask((W, H))
            for _ in range(30):
                probe = Block(
                    "aside", [brief.aside], "hand", asz, W - mx - aw, ay, aw, asz * 1.3,
                    angle=ASIDE_ANGLE, align="right",
                )  # fmt: skip
                if not np.count_nonzero(probe.mask((W, H)) & head_mask):
                    break
                ay += 2
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
    fixed = Fixed(blocks, region, notes)
    if brief.layout == ad_lineup.LAYOUT_NAME:  # ブランド特集は、題字の分だけ範囲を下げる
        from app import ad_lineup_layout

        ad_lineup_layout.add_masthead(fixed, brief, geo, ms)
    if fixed.region[3] - fixed.region[1] < 0.25 * H:
        notes.append("商品を置ける範囲が狭い")
    return fixed


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
    worn: bool = False  # 着画（人ごとの切り抜き）。id は "<商品の id>#worn"、label は worn_note

    @property
    def light(self) -> float | None:
        """商品の明るさ（L*。0〜100）。測れなければ tone の目安、それも無ければ None。"""
        return item_lightness(self.hist, self.tone_hint)


ALPHA_MAX_H = 700
# 切り抜きの判定の規則を変えたら上げる（古い判定のキャッシュを使わない）
CUTOUT_CACHE_VERSION = 5


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
    min_items: int = MIN_ITEMS,
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
    if len(sprites) < min_items:
        raise CannotCompose(
            f"切り抜けた商品が{len(sprites)}点で、{min_items}点に足りません"
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
    straight: bool = False  # 輪郭に沿わせられず、商品の上か下にまっすぐ置いた文字
    contours: dict = field(default_factory=dict)  # 文字の大きさごとの輪郭（商品の座標）
    glow: bool = False  # 地に沈むので、縁にやわらかい光の縁取りを足す（画素は変えない）
    label_full: str = ""  # ラベルを短くしたとき、元のラベル（短くしていなければ空）

    def __post_init__(self):
        self.w = self.sprite.alpha.shape[1] * self.h / self.sprite.alpha.shape[0]

    def contour_for(self, fs: int) -> np.ndarray:
        if fs not in self.contours:
            self.contours[fs] = contour_local(self.sprite.alpha, round(self.h), offset_for(fs))[1]
        return self.contours[fs]

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


def total_turn(pts: np.ndarray) -> float:
    """弧の曲がりの合計（rad。向きを問わない）。"""
    t = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    return float(np.sum(np.abs(np.diff(ang))))


def arc_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def inside_distance(mask: np.ndarray, pad: int = 60) -> tuple[np.ndarray, int, int]:
    """商品のマスクの外の各画素から、商品までの距離（商品の中は 0）。商品の外接矩形 + pad の範囲だけ
    測る（範囲の外は pad より遠いので十分）。戻り値は (距離, 左上の x, y)。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.full((1, 1), 1e6, np.float32), 0, 0
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = (
        min(mask.shape[1], int(xs.max()) + pad + 1),
        min(mask.shape[0], int(ys.max()) + pad + 1),
    )
    crop = mask[y0:y1, x0:x1]
    return cv2.distanceTransform((~crop).astype(np.uint8), cv2.DIST_L2, 3), x0, y0


def window_ok(seg: np.ndarray, fs: int, dist: tuple[np.ndarray, int, int] | None) -> bool:
    """輪郭に沿わせてよい弧か。逆さ・急な曲がり・自分の商品への食い込みは使わない。"""
    if upside_fraction(seg) > UPSIDE_MAX or total_turn(seg) > MAX_WINDOW_TURN:
        return False
    if dist is not None:
        d, x0, y0 = dist
        h, w = d.shape
        ix, iy = np.rint(seg[:, 0]).astype(int) - x0, np.rint(seg[:, 1]).astype(int) - y0
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)  # 範囲の外は遠い
        vals = d[iy[inside], ix[inside]]
        if len(vals) and float(vals.min()) < BAND_DOWN * fs:
            return False
    return True


def best_window(
    it: ItemState, label_w: float, fs: int, occ: Occupancy, dist: tuple | None = None
) -> tuple[float, float, int, np.ndarray, Band] | None:
    """輪郭の上で、文字が流れる区間をいちばん良い位置に選ぶ。(bad, cost, start, arc 点, 帯)。

    区間は輪郭の始点・終点をまたがない連続した1本の弧。輪郭を半周ずらした並びからも切り出すので、
    どの場所も区間になれる。長さが文字の長さ（余裕込み）に足りない輪郭は使わない。"""
    pts = to_canvas(it.contour_for(fs), it.w, it.h, it.cx, it.cy, it.angle)
    n = len(pts)
    text_len = label_w * TEXT_LEN_SAFETY
    m = int(math.ceil(text_len * 1.04 / CONTOUR_STEP)) + 1
    if m > n - 2:
        return None
    best = None
    for shift in (0, n // 2):
        order = np.roll(pts, -shift, axis=0)
        for s in range(0, n - m + 1, 2):
            seg = order[s : s + m]
            if arc_length(seg) < text_len or not window_ok(seg, fs, dist):
                continue
            band = raster_band(seg, fs)
            bad, cost = window_cost(band, occ, seg)
            cost += 0.01 * ((s + shift) % n)  # 同じなら、上の左から始まる（読みやすい）ほうへ
            if best is None or cost < best[1]:
                best = (bad, cost, (s + shift) % n, seg, band)
    return best


def contour_feasible(it: ItemState, ms: Measurer, size: tuple[int, int], steps) -> bool:
    """輪郭に沿わせてよい弧（逆さ・急な曲がり・食い込みが無い）が、この商品にどこかにあるか。
    商品を少し動かしても変わらない（形だけで決まる）ので、先に調べて、無理なら探し直さない。"""
    dist = inside_distance(it.mask(size))
    for fs in steps:
        text_len = ms.width("hand", it.sprite.label, fs) * TEXT_LEN_SAFETY
        pts = to_canvas(it.contour_for(fs), it.w, it.h, it.cx, it.cy, it.angle)
        n = len(pts)
        m = int(math.ceil(text_len * 1.04 / CONTOUR_STEP)) + 1
        if m > n - 2:
            continue
        for shift in (0, n // 2):
            order = np.roll(pts, -shift, axis=0)
            for st in range(0, n - m + 1, 2):
                seg = order[st : st + m]
                if arc_length(seg) >= text_len and window_ok(seg, fs, dist):
                    return True
    return False


def _place_text(
    it: ItemState, ms: Measurer, occ: Occupancy, steps: tuple[int, ...] = LABEL_STEPS
) -> bool:
    """商品 it の文字の区間を選んで占有マップに書く。許容内に収まれば True。"""
    dist = inside_distance(it.mask(occ.size))
    for fs in steps:
        label_w = ms.width("hand", it.sprite.label, fs)
        best = best_window(it, label_w, fs, occ, dist)
        if best is None:
            continue
        bad, cost, start, seg, band = best
        if bad <= TOLERANCE_PX:
            pts = to_canvas(it.contour_for(fs), it.w, it.h, it.cx, it.cy, it.angle)
            n = len(pts)
            m_arc = min(
                n - 1, int(math.ceil(label_w * TEXT_LEN_SAFETY * ARC_SLACK / CONTOUR_STEP)) + 1
            )
            arc = pts[(start + np.arange(m_arc)) % n]
            # 描く弧も、始点・終点をまたがない連続した1本にする（またぐ分は切る）
            if start + m_arc > n:
                arc = seg
            it.arc = arc
            it.fs, it.text_len, it.cost, it.placed = fs, label_w * TEXT_LEN_SAFETY, cost, True
            it.straight = False
            band.paint(occ.text)
            return True
    return False


def _place_straight(it: ItemState, ms: Measurer, occ: Occupancy, geo: Geo) -> bool:
    """輪郭に沿わせられないとき、商品の上か下に、少し傾けてまっすぐ置く（左から右へ読む向き）。
    大きい段から試し、置けた最初の段を使う。"""
    return any(_place_straight_at(it, ms, occ, geo, fs) for fs in LABEL_STEPS)


def _place_straight_at(it: ItemState, ms: Measurer, occ: Occupancy, geo: Geo, fs: int) -> bool:
    label_w = ms.width("hand", it.sprite.label, fs)
    length = label_w * TEXT_LEN_SAFETY
    arc_len = length * ARC_SLACK
    k = int(math.ceil(arc_len / CONTOUR_STEP)) + 1
    sx0, sy0, sx1, sy1 = geo.safe_rect
    hx, hy = it.half_extent()
    best = None
    for above in (True, False):
        base_y = (
            it.cy - hy - BAND_DOWN * fs - STRAIGHT_GAP
            if above
            else it.cy + hy + BAND_UP * fs + STRAIGHT_GAP
        )
        for align in (0.0, -1.0, 1.0):  # 商品の中央・左寄せ・右寄せ
            x0 = it.cx - arc_len / 2 + align * max(0.0, hx - arc_len / 2)
            x0 = min(max(x0, sx0), max(sx0, sx1 - arc_len))
            for tilt in STRAIGHT_TILTS:
                d = np.array([math.cos(math.radians(tilt)), math.sin(math.radians(tilt))])
                start = np.array([x0, base_y])
                seg = start + np.arange(k)[:, None] * CONTOUR_STEP * d
                seg = seg[np.linalg.norm(seg - start, axis=1) <= arc_len + 1e-6]
                band = raster_band(seg, fs)
                bad, cost = window_cost(band, occ, seg)
                if best is None or cost < best[0]:
                    best = (cost, bad, seg, band)
    if best is None or best[1] > TOLERANCE_PX:
        return False
    cost, bad, seg, band = best
    it.arc, it.fs, it.text_len, it.cost = seg, fs, length, cost
    it.placed, it.straight = True, True
    band.paint(occ.text)
    return True


def item_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    small = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    return inter / small if small else 0.0


PRICE_TAIL = re.compile(r" (?P<price>¥\d[\d,]*)$")


def shorten_steps(label: str) -> list[str]:
    """「<ブランド> の <品名> ¥<価格>」の形のラベルを短くする段。
    先に品名を外して「<ブランド> ¥<価格>」
    （ブランドは商品の違いが分かる手がかりで、品名は見出しや絵で分かるので、ブランドを残す）、
    それでも入らなければ「<品名> ¥<価格>」。価格は必ず残す。その形でなければ空。"""
    m = SHORT_LABEL.match(label)
    if not m:
        return []
    rest = m.group("rest")
    price = PRICE_TAIL.search(rest)
    steps = [f"{m.group('brand')} {price.group('price')}"] if price else []
    steps.append(rest)
    return [s for s in steps if len(s) < len(label)]


def shorten_label(label: str) -> str | None:
    """短くする最初の段（「<ブランド> ¥<価格>」）。短くできなければ None。"""
    steps = shorten_steps(label)
    return steps[0] if steps else None


def _shorten(it: ItemState) -> bool:
    """輪郭に入らないラベルを短くする（文字を小さくするより先に）。短くできたら True。"""
    if it.sprite.worn:
        return False
    full = it.label_full or it.sprite.label
    steps = shorten_steps(full)
    # まだ使っていない次の段（いまのラベルより短い物）へ進む
    nxt = next((s for s in steps if len(s) < len(it.sprite.label) and s != it.sprite.label), None)
    if it.label_full and it.sprite.label in steps:
        later = steps[steps.index(it.sprite.label) + 1 :]
        nxt = later[0] if later else None
    if not nxt:
        return False
    it.label_full = full
    it.sprite = replace(it.sprite, label=nxt)
    return True


def place_texts(
    items: list[ItemState],
    fixed: Fixed,
    geo: Geo,
    ms: Measurer,
    rng: random.Random,
    steps: tuple[int, ...] = LABEL_STEPS,
    straight: bool = True,
) -> tuple[bool, float]:
    """商品ごとに、輪郭の文字の区間を選ぶ。置けない商品は、少し動かして探し直す。"""
    size = (geo.w, geo.h)
    text = np.zeros((geo.h, geo.w), bool)
    for b in fixed.blocks:
        text |= b.mask(size)
    ok_all = True
    placed_masks: dict[int, np.ndarray] = {}
    order = sorted(range(len(items)), key=lambda i: -items[i].h)
    occ = Occupancy(size, geo.safe_rect, np.zeros_like(text), text)
    for it in items:
        occ.items |= it.mask(size)
    for i in order:
        it = items[i]
        if not it.sprite.label:  # 添える文が無い（着画に worn_note が無いとき）。文字は置かない
            it.placed = True
            continue
        if straight and not contour_feasible(it, ms, size, steps):
            # 形のうえで輪郭に沿わせられない（長いラベル・曲がりが急）。まずラベルを短くして
            # 沿わせられるか見る。だめなら（縮めても動かしても同じなので）商品の上か下にまっすぐ置く
            if not (_shorten(it) and contour_feasible(it, ms, size, steps)):
                if _place_straight(it, ms, occ, geo):
                    continue
                ok_all = False
                continue
        if _place_text(it, ms, occ, steps):
            continue
        if _shorten(it) and _place_text(it, ms, occ, steps):  # 入らなければ、小さくする前に短く
            continue
        # 動かして探し直す（近い所から。着画は据えた場所から動かさない）
        done = False
        orig = (it.cx, it.cy)
        for r in () if it.sprite.worn else (30, 60, 100, 150):
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
            # 最後の手段（最後の試行だけ）: 輪郭に沿わせず、商品の上か下にまっすぐ置く
            if straight:
                occ.items = np.zeros_like(text)
                for o in items:
                    occ.items |= o.mask(size)
                if _place_straight(it, ms, occ, geo):
                    continue
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
    limit: int | None = None,
    size_scale: float = 1.0,
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
    if limit is not None:  # 1点もの: 少なめにし、何も指さない矢印は使わない
        want = min(want, limit)
        kinds = [k for k in kinds if k != "arrow"]
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
                rng.uniform(*size_range) * geo.w / 1080 * size_scale,
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
    worn_item_id: str | None = None  # 着画を入れた商品の id（入れなければ None）
    worn_candidates: int = 0  # 着画の候補が見つかった商品の数
    worn_fallback: str | None = None  # 着画を入れられず外したときの理由
    link: dict | None = None  # 着画からアイテム写真への点線の矢印（pts・mask）
    layout: str = "collage"
    single: ad_single.SinglePlan | None = None  # 1点ものの メモ・札・細部の写真
    single_target_h: float = 0.0  # 1点ものの主役の高さの目標（点数用）
    lineup: ad_lineup.LineupPlan | None = None  # 新品のまとめ見せの 文字・札・色見本

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
                    "worn": it.sprite.worn,
                    "label_full": it.label_full or None,
                    "label_size": it.fs,
                    "label_straight": it.straight,
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
            "worn_item_id": self.worn_item_id,
            "worn_candidates": self.worn_candidates,
            "worn_fallback": self.worn_fallback,
            "worn_link": None if self.link is None else self.link["pts"],
            "layout": self.layout,
            "single": None if self.single is None else self.single.to_json(),
            "lineup": None if self.lineup is None else self.lineup.to_json(),
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


def make_worn(
    sprite: Sprite, region: tuple, fmt: str, geo: Geo, rng: random.Random
) -> tuple[ItemState, tuple]:
    """着画の ItemState と、ほかの商品を散らす残りの範囲。左か右の端に据える（種で決める）。"""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    aspect = sprite.alpha.shape[1] / sprite.alpha.shape[0]
    h = min(
        WORN_H_RATIO[fmt] * geo.h, WORN_H_MAX_OF_REGION * rh, WORN_W_MAX_OF_REGION * rw / aspect
    )
    w = h * aspect
    left = rng.random() < 0.5
    cx = x0 + w / 2 + 14 if left else x1 - w / 2 - 14
    cy = (y0 + y1) / 2 + rng.uniform(-0.04, 0.04) * rh
    ang = rng.choice((-1, 1)) * rng.uniform(*WORN_ANGLE)
    it = ItemState(sprite, h, cx, cy, ang)
    clamp_item(it, region)
    hx, _ = it.half_extent()
    if left:
        rest = (it.cx + hx + WORN_GAP, y0, x1, y1)
    else:
        rest = (x0, y0, it.cx - hx - WORN_GAP, y1)
    return it, rest


def _quad_bezier(p0: np.ndarray, c: np.ndarray, p1: np.ndarray, n: int = 40) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t**2 * p1


def _boundary_points(mask: np.ndarray, n: int = 28) -> np.ndarray:
    """マスクの外周を等間隔にとった点（x, y）。"""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2))
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    idx = np.linspace(0, len(c), n, endpoint=False).astype(int)
    return c[idx]


def plan_link(
    worn: ItemState, target: ItemState, size: tuple[int, int], taken: np.ndarray
) -> dict | None:
    """着画から、その商品のアイテム写真へ、手描きの点線の矢印を1本。空きが無ければ None（省く）。
    taken は、ほかの商品・文字・固定の文字など、矢印が通ってはいけない場所。
    両者の外周の点どうしを、少し曲げた曲線で結び、空いている最短の物を選ぶ。"""
    W, H = size
    wm, tm = worn.mask(size), target.mask(size)
    wp, tp = _boundary_points(wm), _boundary_points(tm)
    if len(wp) == 0 or len(tp) == 0:
        return None
    wc = np.array(np.where(wm)[::-1]).mean(axis=1)
    tc = np.array(np.where(tm)[::-1]).mean(axis=1)

    def push(pts, center, gap):
        v = pts - center
        n = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        return pts + v / n * gap

    wp, tp = push(wp, wc, WORN_LINK_GAP), push(tp, tc, WORN_LINK_GAP)
    k = 2 * (WORN_LINK_THICK // 2) + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    kc = (
        2 * (WORN_LINK_CLEAR + WORN_LINK_THICK // 2) + 1
    )  # ほかの物から空ける距離（線の太さの半分も足す）
    blocked = (
        cv2.dilate(taken.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
        > 0
    )
    own = cv2.dilate((wm | tm).astype(np.uint8), kern) > 0
    # 着画・アイテム写真の縁のすぐ外までは通ってよい（始点・終点は縁から GAP だけ離してある）
    own_core = cv2.erode((wm | tm).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    best = None
    for a in wp:
        for b in tp:
            L = float(np.linalg.norm(b - a))
            if L < 30 or L > WORN_LINK_MAX * W or (best and L >= best[0]):
                continue
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            for bulge in (0.15, -0.15, 0.0):
                pts = _quad_bezier(a, (a + b) / 2 + nrm * bulge * L, b, 30)
                ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
                if ix.min() < 0 or iy.min() < 0 or ix.max() >= W or iy.max() >= H:
                    continue
                mid = slice(2, -2)
                if blocked[iy[mid], ix[mid]].any() or own_core[iy, ix].any():
                    continue
                if own[iy[3:-3], ix[3:-3]].any():  # 途中で着画・アイテム写真の縁に触れる
                    continue
                best = (L, pts)
                break
    if best is None:
        return None
    pts = best[1]
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, WORN_LINK_THICK)
    return {"pts": np.round(pts, 1).tolist(), "mask": m.astype(bool)}


def link_path_d(pts) -> str:
    """点線の矢印（本体の曲線と、先端の矢じり）のパス。"""
    p = np.asarray(pts, float)
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in p)
    tip, back = p[-1], p[-4]
    v = tip - back
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
        for ang in (150, -150):
            r = rot_matrix(ang) @ v
            e = tip + r * 24
            d += f" M{tip[0]:.1f} {tip[1]:.1f} L{e[0]:.1f} {e[1]:.1f}"
    return d


def build_layout(
    brief: Brief,
    sprites: list[Sprite],
    fmt: str,
    seed: int,
    ms: Measurer,
    worn: Sprite | None = None,
) -> Layout:
    """1つの種から、1つの候補を組む。文字が収まらなければ商品を少し縮めてやり直す。
    worn（着画の切り抜き）があれば、1枚だけ端に据え、ほかの商品をそのまわりに散らす。"""
    geo = geometry(fmt)
    fixed = layout_fixed(brief, fmt, ms)
    best: tuple[float, Layout] | None = None
    scale = 1.0
    for attempt in range(1, ATTEMPTS + 1):
        rng = random.Random(f"{seed}:{attempt}")
        wi, region = None, fixed.region
        if worn is not None:
            wi, region = make_worn(worn, fixed.region, fmt, geo, rng)
        items = make_items(sprites, region, rng, scale)
        relax(items, region)
        if region[3] - region[1] > 1.15 * (region[2] - region[0]):
            spread_vertically(items, region)
            relax(items, region)
        if wi is not None:
            items.append(wi)
        steps = LABEL_STEPS if attempt == ATTEMPTS else LABEL_STEPS[:1]
        ok, cost = place_texts(
            items, fixed, geo, ms, rng, steps, straight=attempt >= STRAIGHT_FROM_ATTEMPT
        )
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
    size = (geo.w, geo.h)
    occ = np.zeros((geo.h, geo.w), bool)
    for b in lay.blocks:
        occ |= b.mask(size)
    for it in lay.items:
        occ |= it.mask(size)
        if it.arc is not None:
            raster_band(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)], it.fs).paint(occ)
    worn_it = next((it for it in lay.items if it.sprite.worn), None)
    if worn_it is not None:
        lay.worn_item_id = worn_it.sprite.id.removesuffix("#worn")
        tgt = next((it for it in lay.items if it.sprite.id == lay.worn_item_id), None)
        if tgt is not None:
            taken = np.zeros((geo.h, geo.w), bool)
            for b in lay.blocks:
                taken |= b.mask(size)
            for it in lay.items:
                if it is worn_it or it is tgt:
                    continue
                taken |= it.mask(size)
            for it in lay.items:
                if it.arc is not None:
                    raster_band(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)], it.fs).paint(
                        taken
                    )
            # 着画とアイテム写真に付いた文字の帯も、矢印は避ける
            link = plan_link(worn_it, tgt, size, taken)
            if link is not None:
                lay.link = link
                occ |= link["mask"]
    lay.doodles = place_doodles(random.Random(f"{seed}:doodle"), geo, occ, lay.region, brief.season)
    return lay


# ---------------------------------------------------------------- 1点もの（layout: single）


def make_single_items(
    hero_sprite: Sprite,
    worn: Sprite | None,
    region: tuple,
    fmt: str,
    geo: Geo,
    rng: random.Random,
    scale: float,
) -> tuple[list[ItemState], float]:
    """主役（と着画）の ItemState。(商品の並び 主役が先頭, 主役の高さの目標)。
    着画があれば主役の横に置く（主役の外側に、メモを置く空きの列を残す）。縦長の形式では主役の下に置く。"""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    aspect = hero_sprite.alpha.shape[1] / hero_sprite.alpha.shape[0]
    tall = rh > 1.15 * rw
    target = ad_single.HERO_H_RATIO[fmt] * geo.h
    cap = ad_single.HERO_W_RATIO_WORN if worn else ad_single.HERO_W_RATIO
    if worn is not None and tall:
        target = min(target, 0.5 * rh)
    h = min(target * scale, cap * geo.w / aspect * scale**0.5, ad_single.HERO_H_OF_REGION * rh)
    h = max(h, MIN_ITEM_H_RATIO * geo.w + 1)
    ang = rng.choice((-1, 1)) * rng.uniform(*ad_single.HERO_ANGLE)
    hero = ItemState(hero_sprite, h, (x0 + x1) / 2, (y0 + y1) / 2, ang)
    items = [hero]
    if worn is None:
        hero.cx += rng.uniform(-0.04, 0.04) * geo.w
        hero.cy += rng.uniform(-0.03, 0.03) * rh
        clamp_item(hero, region)
        return items, target
    w_aspect = worn.alpha.shape[1] / worn.alpha.shape[0]
    on_left = rng.random() < 0.5  # 着画を置く側
    wang = rng.choice((-1, 1)) * rng.uniform(*ad_single.WORN_ANGLE)
    if tall:
        hx, hy = hero.half_extent()
        hero.cy = y0 + hy + 8
        hero.cx = (x0 + x1) / 2 + (0.08 if on_left else -0.08) * rw
        clamp_item(hero, region)
        room = y1 - (hero.cy + hy) - ad_single.HERO_GAP
        wh = min(ad_single.WORN_REL_H_STACK * room + 0.2 * room, 0.5 * rh, 0.62 * geo.w / w_aspect)
        wh = max(wh, 0.12 * geo.w)
        wi = ItemState(worn, wh, 0.0, y1 - wh / 2 - 6, wang)
        whx, _ = wi.half_extent()
        wi.cx = x0 + whx + 24 if on_left else x1 - whx - 24
        clamp_item(wi, region)
        items.append(wi)
        return items, target
    wh = min(ad_single.WORN_REL_H * h, 0.9 * rh)
    wi = ItemState(worn, wh, 0.0, (y0 + y1) / 2 + 0.03 * rh, wang)
    whx, _ = wi.half_extent()
    hhx, _ = hero.half_extent()
    wi.cx = x0 + whx + 10 if on_left else x1 - whx - 10
    hero.cx = wi.cx + (whx + ad_single.HERO_GAP + hhx) * (1 if on_left else -1)
    hero.cy = (y0 + y1) / 2 + rng.uniform(-0.03, 0.03) * rh
    clamp_item(hero, region)
    clamp_item(wi, region)
    items.append(wi)
    return items, target


def build_single_layout(
    brief: Brief,
    sprites: list[Sprite],
    fmt: str,
    seed: int,
    ms: Measurer,
    worn: Sprite | None = None,
    details: list | None = None,
) -> Layout:
    """1点ものの候補を1つ組む。主役を大きく置き、価格の札 → メモと引き出し線 → 細部の写真
    （着画が無いとき）
        の順に、空いている所へ置く。メモが2つ置けなければ主役を縮めてやり直す。置けないメモは後ろから省く。"""
    geo = geometry(fmt)
    size = (geo.w, geo.h)
    fixed = layout_fixed(brief, fmt, ms)
    single = brief.single or ad_single.Single()
    hero_sprite = replace(sprites[0], label="")
    wanted = len(single.callouts)
    best: tuple[float, Layout] | None = None
    scale = 1.0
    for attempt in range(1, ATTEMPTS + 1):
        rng = random.Random(f"{seed}:single{attempt}")
        items, target = make_single_items(hero_sprite, worn, fixed.region, fmt, geo, rng, scale)
        steps = LABEL_STEPS if attempt == ATTEMPTS else LABEL_STEPS[:1]
        text_ok, _ = place_texts(items, fixed, geo, ms, rng, steps, straight=True)
        bg = PALETTES[brief.palette].bg
        for it in items:
            it.glow = sinks(it.sprite.light, bg)
        blocked = np.zeros((geo.h, geo.w), bool)
        for b in fixed.blocks:
            blocked |= b.mask(size)
        for it in items[1:]:  # 着画（と、その文字の帯）
            blocked |= it.mask(size)
            if it.arc is not None:
                raster_band(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)], it.fs).paint(
                    blocked
                )
        hero_mask = items[0].mask(size)
        overlap = max(
            (item_overlap_ratio(hero_mask, it.mask(size)) for it in items[1:]), default=0.0
        )
        plan = ad_single.SinglePlan(
            wanted=wanted,
            price_text=single.price_text,
            stamp_text=single.stamp,
            notes=list(single.notes),
        )
        rng_p = random.Random(f"{seed}:plan{attempt}")
        safe = geo.safe_rect
        if single.price_text:
            plan.badge = ad_single.place_badge(
                single.price_text, hero_mask=hero_mask, blocked=blocked, safe=safe, ms=ms,
                rng=rng_p, width=geo.w,
            )  # fmt: skip
        if single.stamp:  # 札のすぐそばに置く（札を占有に入れる前に探す。札との重なりは自分で見る）
            plan.stamp = ad_single.place_stamp_tag(
                single.stamp, plan.badge, hero_mask=hero_mask, blocked=blocked, safe=safe, ms=ms,
                rng=rng_p, width=geo.w,
            )  # fmt: skip
        if plan.badge is not None:
            blocked |= ad_single.dilate(plan.badge.disc_mask((geo.h, geo.w)), 10)
        if plan.stamp is not None:
            ad_single.paint_poly(plan.stamp.polygon(8), blocked)
        memos, dropped, occ = ad_single.place_memos(
            single.callouts, hero_mask=hero_mask, blocked=blocked, safe=safe, ms=ms, rng=rng_p,
            scale=geo.w / 1080,
        )  # fmt: skip
        plan.memos, plan.dropped = memos, dropped
        if worn is None and details:
            plan.photos = ad_single.place_photos(
                details,
                hero_mask=hero_mask,
                blocked=occ,
                safe=safe,
                rng=rng_p,
                width=geo.w,
                limit=ad_single.PHOTO_MAX if fmt == "9:16" else 1,
            )
            for p in plan.photos:
                ad_single.paint_poly(p.polygon(10), occ)
        lay = Layout(
            fmt, geo.w, geo.h, seed, brief.palette, fixed.blocks, fixed.region, items, [],
            list(fixed.notes), attempt, scale, brief.palette_requested or brief.palette,
            brief.palette_changed_reason, brief.season,
        )  # fmt: skip
        lay.layout, lay.single, lay.single_target_h = "single", plan, target
        occ |= hero_mask
        lay.doodles = place_doodles(
            random.Random(f"{seed}:doodle"), geo, occ, lay.region, brief.season,
            limit=ad_single.DOODLE_LIMIT, size_scale=ad_single.DOODLE_SIZE_SCALE,
        )  # fmt: skip
        enough = len(memos) >= min(ad_single.MIN_CALLOUTS, wanted)
        good = enough and overlap <= 0.05 and text_ok
        rank = (wanted - len(memos)) * 1e4 + (0 if (plan.badge or not single.price_text) else 1e5)
        rank += overlap * 1e5 + (0 if text_ok else 1e5) + attempt
        if best is None or rank < best[0]:
            best = (rank, lay)
        if good:
            break
        scale *= SHRINK
    return best[1]


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
    truncated_labels: list[str] = field(default_factory=list)  # 弧が文字より短く、末尾が描かれない
    reversed_labels: list[str] = field(default_factory=list)  # 逆さに流れる区間が多い弧
    headline_lines: int = 0
    headline_fits: bool = True
    doodle_overlap_px: int = 0
    problems: list[str] = field(default_factory=list)
    verdict: str = "pass"
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)
    single: dict | None = None  # 1点ものの検査の数（メモ・線・札・写真）
    lineup: dict | None = None  # 新品のまとめ見せの検査の数（名前の大きさ・型の数・札）

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
    group_of = lay.lineup.group_of if lay.lineup is not None else {}
    within_max = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            ratio = item_overlap_ratio(masks[i], masks[j])
            gi, gj = group_of.get(lay.items[i].sprite.id), group_of.get(lay.items[j].sprite.id)
            if gi is not None and gi == gj:  # 同じ型の色ちがいは、重ねて見せてよい
                within_max = max(within_max, ratio)
            else:
                c.item_overlap_max = max(c.item_overlap_max, ratio)
    c.item_overlap_max = round(c.item_overlap_max, 3)
    if c.item_overlap_max > MAX_ITEM_OVERLAP:
        c.problems.append(f"商品どうしが重なりすぎ（{c.item_overlap_max:.0%}）")
    if within_max > ad_lineup.GROUP_OVERLAP_MAX:
        c.problems.append(f"同じ型の色ちがいが重なりすぎ（{within_max:.0%}）")
    min_h = (ad_lineup.MIN_ITEM_H_RATIO if lay.lineup is not None else MIN_ITEM_H_RATIO) * W
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
        if not it.sprite.label:
            continue  # 文字を添えない（着画で worn_note が無い）
        if it.arc is None:
            c.unplaced_text.append(it.sprite.id)
            continue
        need = ms.width("hand", it.sprite.label, it.fs) * TEXT_LEN_SAFETY if ms else it.text_len
        if arc_length(it.arc) < need - 1e-6:  # textPath は弧の終わりより先を描かない
            c.truncated_labels.append(it.sprite.id)
        if upside_fraction(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)]) > UPSIDE_MAX:
            c.reversed_labels.append(it.sprite.id)
        band = raster_band(it.arc[: max(2, int(it.text_len / CONTOUR_STEP) + 1)], it.fs)
        c.out_of_canvas_px += band.outside((0, 0, W, H))
        c.outside_safe_px += band.outside(geo.safe_rect)
        texts.append((f"label:{it.sprite.id}", band.full(W, H)))
    if lay.link is not None:  # 着画からの点線の矢印（商品・文字に重ならない）
        texts.append(("link", lay.link["mask"]))
    lineup_texts = np.zeros((H, W), bool)
    if lay.lineup is not None:  # 型の名前・色の名前・サイズ・札の文字（色見本の丸も文字の仲間）
        for t in lay.lineup.texts:
            m = polygon_mask(t.polygon(2), size)
            lineup_texts |= m
            texts.append((f"lt:{t.kind}", m))
        for sw in lay.lineup.swatches:
            m = polygon_mask(rect_polygon(sw.cx - sw.r, sw.cy - sw.r, 2 * sw.r, 2 * sw.r), size)
            lineup_texts |= m
            texts.append(("swatch", m))
    if c.truncated_labels:
        c.problems.append(f"ラベルが弧に収まらず末尾が欠ける（{', '.join(c.truncated_labels)}）")
    if c.reversed_labels:
        c.problems.append(f"ラベルが逆さに流れる（{', '.join(c.reversed_labels)}）")
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
    if lay.lineup is not None:
        lc, lp = ad_lineup.check_plan(
            lay.lineup, union=union, safe=geo.safe_rect, texts_mask=lineup_texts, width=W
        )
        c.lineup = asdict(lc)
        c.problems.extend(lp)
        c.verdict = "pass" if not c.problems else "fail"
        both = union | lineup_texts
        for b in lay.lineup.badges:
            both |= b.disc_mask((H, W))
        c.score, c.score_parts = ad_lineup.score_plan(union=both, region=lay.region)
        contrast = contrast_score(
            [it.sprite.light for it in lay.items],
            PALETTES[lay.palette].bg,
            [it.glow for it in lay.items],
        )
        if contrast is not None:
            c.score_parts["contrast"] = round(contrast, 3)
            c.score = round((1 - CONTRAST_WEIGHT) * c.score + 100 * CONTRAST_WEIGHT * contrast, 1)
        return c
    if lay.single is not None:
        blocked = np.zeros((H, W), bool)
        for _, m in texts:
            blocked |= m
        for m in masks[1:]:  # 着画
            blocked |= m
        sc, sp = ad_single.check_plan(
            lay.single,
            hero_mask=masks[0],
            blocked=blocked,
            safe=geo.safe_rect,
            width=W,
            expect_price=bool(lay.single.price_text),
            expect_stamp=bool(lay.single.stamp_text),
        )
        c.single = asdict(sc)
        c.problems.extend(sp)
        c.verdict = "pass" if not c.problems else "fail"
        c.score, c.score_parts = ad_single.score_plan(
            lay.single,
            hero_mask=masks[0],
            region=lay.region,
            width=W,
            height=H,
            hero_target_h=lay.single_target_h,
        )
        contrast = contrast_score(
            [it.sprite.light for it in lay.items],
            PALETTES[lay.palette].bg,
            [it.glow for it in lay.items],
        )
        if contrast is not None:
            c.score_parts["contrast"] = round(contrast, 3)
            c.score = round((1 - CONTRAST_WEIGHT) * c.score + 100 * CONTRAST_WEIGHT * contrast, 1)
        return c
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
svg.ov .link{{fill:none;stroke:{pal.doodle};stroke-width:{DOODLE_STROKE};stroke-linecap:round;
stroke-linejoin:round;stroke-dasharray:1 12;stroke-width:6;vector-effect:non-scaling-stroke}}
svg.ov .da{{fill:{pal.accent};stroke:{pal.doodle};stroke-width:{DOODLE_STROKE};
stroke-linejoin:round;vector-effect:non-scaling-stroke}}
svg.ov .stamp{{fill:none;stroke:{pal.accent};stroke-width:3;stroke-linecap:round;
vector-effect:non-scaling-stroke}}
svg.ov text.stampt{{fill:{pal.text};font-family:"DM Sans","Noto Sans CJK JP",sans-serif;
font-weight:700;letter-spacing:.1em;text-anchor:middle}}
svg.ov text{{fill:{pal.text};font-family:"Caveat","Yomogi";font-weight:700;letter-spacing:.03em}}
.acc{{color:{pal.accent}}}
{ad_single.css(pal) if lay.single is not None else ""}
{ad_lineup.css(pal) if lay.lineup is not None else ""}
.masthead{{font-weight:900;letter-spacing:-.02em;line-height:{HEAD_LINE_HEIGHT};color:{pal.accent}}}
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
            "masthead": "head masthead",
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
    if lay.single is not None:
        body.extend(ad_single.html_parts(lay.single, pal, e, W))
    if lay.lineup is not None:
        body.extend(ad_lineup.html_parts(lay.lineup, pal, e))
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
    if lay.link is not None:  # 着画からアイテム写真への手描きの点線の矢印
        svg.append(f'<path class="link" d="{link_path_d(lay.link["pts"])}"/>')
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
    if lay.single is not None:
        svg.extend(ad_single.svg_parts(lay.single, pal, e))
    if lay.lineup is not None:
        svg.extend(ad_lineup.svg_parts(lay.lineup, pal, e))
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
        single = {} if lay.single is None else single_meta(lay)
        if lay.layout == ad_lineup.LAYOUT_NAME:
            single = lineup_meta(lay)
        short = [
            {"id": it.sprite.id, "from": it.label_full, "to": it.sprite.label}
            for it in lay.items
            if it.label_full
        ]
        return {
            **extra,
            **single,
            **({"labels_shortened": short} if short else {}),
            "format": self.format,
            "palette_used": lay.palette,
            "palette_changed_reason": lay.palette_changed_reason,
            "worn_item_id": lay.worn_item_id,
            "worn_candidates": lay.worn_candidates,
            "worn_fallback": lay.worn_fallback,
            "seed": self.best.seed,
            "score": self.best.checks.score,
            "verdict": self.verdict,
            "checks": asdict(self.best.checks),
            "dropped_items": self.dropped,
        }


def single_meta(lay: Layout) -> dict:
    """1点ものの meta。置けたメモの数・省いたメモ・着画か細部の写真を使ったか。"""
    plan = lay.single
    worn_used = any(it.sprite.worn for it in lay.items)
    return {
        "layout": ad_single.LAYOUT_NAME,
        "callouts_wanted": plan.wanted,
        "callouts_placed": len(plan.memos),
        "callouts_placed_kinds": [m.kind for m in plan.memos],
        "callouts_dropped": plan.dropped,
        "worn_used": worn_used,
        "detail_photos_used": len(plan.photos),
        "price_badge": plan.badge is not None,
        "stamp_placed": plan.stamp is not None,
    }


def fmt_slug(fmt: str) -> str:
    return fmt.replace(":", "x")


def reduce_items(
    sprites: list[Sprite], n: int, keep_id: str | None
) -> tuple[list[Sprite], list[Sprite]]:
    """商品を n 点まで、ブリーフの並びの後ろから外す（keep_id の商品は外さない）。(残り, 外した)"""
    keep = list(sprites)
    removed: list[Sprite] = []
    for i in range(len(keep) - 1, -1, -1):
        if len(keep) <= n:
            break
        if keep[i].id != keep_id:
            removed.append(keep.pop(i))
    return keep, removed[::-1]


def reduce_lineup_items(
    sprites: list[Sprite],
    n: int,
    keep_id: str | None,
    group_of: dict[str, int],
    protect: bool = True,
) -> tuple[list[Sprite], list[Sprite]]:
    """lineup: 後ろから外すが、protect のあいだは型（グループ）の最後の1点は外さない
    （型が丸ごと消えないように）。それでも入らないときだけ protect=False で後ろの型を省く。"""
    keep = list(sprites)
    removed: list[Sprite] = []
    for i in range(len(keep) - 1, -1, -1):
        if len(keep) <= n:
            break
        sp = keep[i]
        if sp.id == keep_id:
            continue
        g = group_of.get(sp.id)
        if protect and g is not None and sum(1 for o in keep if group_of.get(o.id) == g) <= 1:
            continue
        removed.append(keep.pop(i))
    return keep, removed[::-1]


def lineup_meta(lay: Layout) -> dict:
    if lay.lineup is not None:
        return ad_lineup.lineup_meta(lay.lineup)
    return {"layout": ad_lineup.LAYOUT_NAME, "lineup_style": "brand"}  # 題字つきの collage


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
    worn: Sprite | None = None,
    worn_candidates: int = 0,
    details: list | None = None,
) -> FormatResult:
    t0 = time.time()
    ms = Measurer(fonts_dir)
    # 地に沈む商品があれば、先に色を決める（色は候補ごとに変えない。同じ版の 3 形式で揃う）。
    # 着画（人の肌や服）も同じ規則で測る
    lights = [sp.light for sp in sprites] + ([worn.light] if worn is not None else [])
    used, reason = choose_palette(brief.palette, lights, brief.palette_options or None)
    if used != brief.palette:
        log(f"  色を替えた: {reason}")
        brief = replace(
            brief, palette=used, palette_requested=brief.palette, palette_changed_reason=reason
        )
    fallback = None
    single = brief.layout == ad_single.LAYOUT_NAME
    is_lineup = brief.layout == ad_lineup.LAYOUT_NAME and brief.lineup is not None
    group_of = (
        {li.id: gi for gi, g in enumerate(brief.lineup.groups) for li in g.items}
        if is_lineup
        else {}
    )
    omitted: list[dict] = []  # 入りきらなくて外した商品（dropped_items に足す）

    def shrink(subset_n: int, keep_id: str | None, protect: bool = True):
        if is_lineup:
            return reduce_lineup_items(sprites, subset_n, keep_id, group_of, protect)
        return reduce_items(sprites, subset_n, keep_id)

    def build(k: int, subset: list[Sprite], with_worn: Sprite | None) -> Layout:
        if single:  # 1点もの。着画が無い（外した）ときだけ細部の写真を添える
            return build_single_layout(
                brief, subset, fmt, brief.seed + k, ms, with_worn, None if with_worn else details
            )
        if is_lineup and brief.lineup.style != "brand":  # 新品のまとめ見せ（型・色の段組み）
            from app import ad_lineup_layout

            return ad_lineup_layout.build_lineup_layout(
                brief, subset, fmt, brief.seed + k, ms, with_worn
            )
        lay = build_layout(brief, subset, fmt, brief.seed + k, ms, with_worn)
        if is_lineup:  # ブランド特集: 題字つきの collage
            lay.layout = ad_lineup.LAYOUT_NAME
        return lay

    def run(subset: list[Sprite], with_worn: Sprite | None) -> list[Candidate]:
        """この商品の並びで候補を作って並べる。先に PROBE_CANDIDATES 個で通る物が無ければ止める。"""
        out: list[Candidate] = []
        for k in range(brief.candidates):
            lay = build(k, subset, with_worn)
            out.append(Candidate(brief.seed + k, lay, check_layout(lay, ms)))
            if (
                k + 1 == PROBE_CANDIDATES
                and not single
                and not any(c.checks.verdict == "pass" for c in out)
            ):
                break
        out.sort(key=Candidate.sort_key)
        return out

    def omit(removed: list[Sprite], why: str) -> list[dict]:
        return [{"id": sp.id, "status": "omitted", "reasons": [why]} for sp in removed]

    cands: list[Candidate] = []
    if worn is not None:
        # 着画が優先: 入らないときは、着画を諦める前に商品を後ろから1点ずつ減らす
        # （着画の商品は外さない。下限は 着画の商品 + ほか2点）
        wid = worn.id.removesuffix("#worn")
        start = len(sprites) if is_lineup else min(len(sprites), WORN_MAX_ITEMS[fmt])
        # lineup は商品が主役: 着画のために外すのは 1 点まで（それで入らなければ着画を諦める）
        low = (
            max(len(sprites) - LINEUP_WORN_MAX_DROP, 1)
            if is_lineup
            else min(len(sprites), 1 + WORN_MIN_OTHERS)
        )
        last: Candidate | None = None
        for n in range(start, max(low, 1) - 1, -1):
            subset, removed = shrink(n, wid)
            cands = run(subset, worn)
            if cands[0].checks.verdict == "pass":
                omitted = omit(removed, "着画を入れるため")
                if removed:
                    log(f"  着画を入れるため {len(removed)} 点を外した")
                break
            last = cands[0]
            log(f"  着画つき {n} 点: 検査に通らない（{'; '.join(cands[0].checks.problems)}）")
            cands = []
        if not cands:
            fallback = "着画を入れると検査に通らなかった: " + "; ".join(
                last.checks.problems if last else []
            )
            log(f"  着画なしに戻す: {fallback}")
    if not cands:
        # 着画なしは、ブリーフの点数をなるべく保つ。検査に通らないときだけ後ろから減らす
        first: list[Candidate] | None = None
        for n in range(len(sprites), (1 if single else MIN_ITEMS) - 1, -1):
            subset, removed = shrink(n, None)
            got = run(subset, None)
            first = first or got
            if got[0].checks.verdict == "pass" or single:
                cands = got
                omitted = omit(removed, "この形式に入りきらず、検査に通らなかったため")
                if removed:
                    log(f"  検査に通るよう {len(removed)} 点を外した")
                break
        if is_lineup and (not cands or cands[0].checks.verdict != "pass"):
            # 型を残したままでは入らない: 後ろの型を丸ごと省いてよい（style の下限の型数までは）
            for n in range(len(sprites) - 1, 1, -1):
                subset, removed = shrink(n, None, protect=False)
                got = run(subset, None)
                if got[0].checks.verdict == "pass":
                    cands = got
                    omitted = omit(removed, "この形式に入りきらず、検査に通らなかったため")
                    log(f"  検査に通るよう {len(removed)} 点（後ろの型）を外した")
                    break
        if not cands or cands[0].checks.verdict != "pass":
            cands, omitted = first or cands, []  # どれも通らない: 全点で needs_review にする
    dropped = [*dropped, *omitted]
    for c in cands:
        c.layout.worn_candidates = worn_candidates
        c.layout.worn_fallback = fallback
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


def _lazy(factory: Callable) -> Callable:
    """モデルは最初に使うときに1回だけ作る（切り抜きと着画で同じ物を使う）。"""
    box: list = []

    def get():
        if not box:
            box.append(factory())
        return box[0]

    return get


def prepare_worn(
    brief: Brief,
    sprites: list[Sprite],
    cache_dir: Path,
    cutter_get: Callable,
    fetch: Callable[[str], bytes],
    log: Callable = print,
) -> tuple[Sprite | None, int, list[str]]:
    """着画を探して切り抜く。(着画のスプライト, 候補が見つかった商品の数, メモ)。
    ブリーフに images が無ければ探さない（今までどおり）。アイテム写真を置ける商品だけを探す
    （着画の商品のアイテム写真も一緒に置くため）。探す途中の失敗は、着画なしで続ける。"""
    placed = {sp.id for sp in sprites}
    searchable = [(it.id, it.images) for it in brief.items if it.images and it.id in placed]
    if not searchable:
        return None, 0, []
    try:
        res, ranked = ad_worn.find_worn(searchable, cache_dir, cutter_get, fetch, log)
        pick = ad_worn.cut_pick(ranked, cache_dir, cutter_get, fetch, log) if ranked else None
    except Exception as e:  # 着画は任意。失敗しても広告は作る
        log(f"  着画を探せなかった: {type(e).__name__}: {e}")
        return None, 0, [f"着画を探せなかった（{type(e).__name__}）"]
    if pick is None or pick.cut_path is None:
        return None, res.candidate_items, res.notes
    item = BriefItem(f"{pick.item_id}#worn", "", brief.worn_note, None, str(pick.cut_path))
    sp = sprite_from_png(item, pick.cut_path)
    sp.worn = True
    return sp, res.candidate_items, res.notes


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
    cutter_get = _lazy(cutter_factory or _default_cutter)
    is_single = brief.layout == ad_single.LAYOUT_NAME
    sprites, dropped = prepare_sprites(
        brief,
        cache_dir,
        cutter_factory=cutter_get,
        fetch=fetch,
        base_dir=base_dir,
        log=log,
        min_items=1 if is_single else MIN_ITEMS,
    )
    worn, worn_candidates, notes = prepare_worn(brief, sprites, cache_dir, cutter_get, fetch, log)
    for n in notes:
        log(f"  {n}")
    details = []
    if is_single and brief.items and len(brief.items[0].images) >= 2:
        try:  # 細部の写真は「あれば」。選べなくても1点ものは組める
            details = ad_single.prepare_details(
                brief.items[0].images, brief.items[0].image, cache_dir, cutter_get, fetch, log=log
            )
        except Exception as e:
            log(f"  細部の写真を選べなかった: {type(e).__name__}: {e}")
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
            worn=worn,
            worn_candidates=worn_candidates,
            details=details,
        )
        log(
            f"  {brief.orig_id or brief.id} {fmt}: {fr.verdict} 点{fr.best.checks.score}"
            f" {fr.seconds:.1f}秒" + (f" {'; '.join(fr.reasons)}" if fr.reasons else "")
        )
        results.append(fr)
    return results
