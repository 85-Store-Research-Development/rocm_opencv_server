"""新品のまとめ見せの広告（ブリーフの `layout: "lineup"`）。
同じシリーズの型ちがい・色ちがい・素材ちがいを雑誌の1ページのように並べる。出品中の新品
（サイズ・色の展開がある）のための組み方で、古着の「一点もの」とは言わない。

組み方（`lineup.style`）:
  compare  型くらべ: 2〜3 の型を段にして、型の名前を大きな手書きで。
           色ちがいは重ねてずらして扇のように見せる
  colors   色ならべ: 1つの型の色ちがいを同じ高さで並べ、色の名前と色見本の丸を添える
  catalog  シリーズ図鑑: 素材ちがい・型ちがいを番号つきの段に分け、
           段ごとに小さな見出しと色の名前
  brand    ブランド特集: ブランド名を大きな題字にして、代表の数点を散らす
           （組みは collage と同じ）
知らない style は collage に落とす（ブリーフの商品を並べるだけ）。

このモジュールはモデルも Chrome も使わず、ad_compose にも依存しない
（ブリーフの読み込み・計画の型・描画の部品・検査）。置き方は app/ad_lineup_layout.py。
**商品の画素は描き直さない**（切り抜き・拡大縮小・回転・影・重ねだけ。
色ちがいは必ず実物の写真）。文字は事実だけ（型の名前・色・サイズ・価格・説明文にある言葉）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import ad_single
from app.ad_palette import Palette

LAYOUT_NAME = "lineup"
STYLES = ("compare", "colors", "catalog", "brand")
GROUPS_RANGE = {"compare": (2, 3), "colors": (1, 1), "catalog": (2, 4), "brand": (2, 5)}
MAX_ITEMS = 10  # 1枚に載せる商品の上限（超える分は後ろから省く）
MAX_ITEMS_PER_GROUP = 5
NAME_MAX_LEN = 20
COLOR_MAX_LEN = 10
SIZES_MAX_LEN = 24
NOTE_MAX_LEN = 24
BADGE_MAX = 3
BADGE_MAX_LEN = 12
BRAND_MAX_LEN = 16
PRICE_MAX_LEN = ad_single.PRICE_MAX_LEN

# --- 文字の大きさ（幅 1080 のとき。大きいほうから試す） ---
NAME_STEPS = (112, 100, 88, 76, 66, 58, 52)
NAME_MIN = 52  # 型の名前はこれ未満にしない（読めること）
COLOR_STEPS = (44, 40, 36, 32)
COLOR_MIN = 32
SIZES_STEPS = (44, 40, 36)
SIZES_MIN = 34
NOTE_STEPS = (54, 50, 46, 42)
NOTE_MIN = 40
NUM_SIZE = 54
PRICE_PILL_STEPS = (52, 48, 44, 40)
BADGE_PILL_SIZE = 38
MASTHEAD_STEPS = (260, 230, 200, 170, 140, 120)
MASTHEAD_MIN = 120

# --- 配置 ---
FAN_OVERLAP = 0.42  # 扇に重ねる幅の割合（隣の商品の幅に対して）
FAN_ANGLES = {2: (-4.5, 4.0), 3: (-6.5, 0.5, 6.5), 4: (-7.5, -2.5, 2.5, 7.5), 5: (-8, -4, 0, 4, 8)}
GROUP_OVERLAP_MAX = 0.55  # 同じ型の色ちがいどうしの重なり（小さいほうの面積に対して）
ITEM_GAP = 26
LABEL_H = 52  # 商品の下の色の名前の行の高さ
ROW_GAP = 22
BAND_H = 76  # 帯（共通の売りの一言と事実の札）の行の高さ
WORN_W_RATIO = 0.21  # 着画の幅（画面の幅に対する割合）
WORN_GAP = 24
BADGE_R_RATIO = 0.076
BADGE_ITEM_OVERLAP = 0.12  # 価格の札が商品に重なってよい割合（札の面積に対して）
BADGE_ROOM_ROW_H = 372  # 段の高さがこれ以上なら価格を札（丸いシール）にする。足りなければ小さな札
MIN_ITEM_H_RATIO = 0.15  # 商品の高さの下限（幅に対する割合）
TOLERANCE_PX = 40
SWATCH_R = 13
DOODLE_LIMIT = 3
DOODLE_SIZE_SCALE = 0.8


# ---------------------------------------------------------------- ブリーフの lineup


@dataclass
class LItem:
    id: str
    color: str = ""


@dataclass
class Group:
    name: str
    price_text: str = ""
    sizes: str = ""
    items: list[LItem] = field(default_factory=list)


@dataclass
class Lineup:
    style: str = "compare"
    title_note: str = ""
    brand: str = ""
    groups: list[Group] = field(default_factory=list)
    badges: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # 読み込みで省いたこと


def _short_text(problems: list[str], name: str, value, maxlen: int, required=False) -> str:
    if value in (None, ""):
        if required:
            problems.append(f"{name} が必要です")
        return ""
    if not isinstance(value, str):
        problems.append(f"{name} は文字列にしてください")
        return ""
    text = value.strip()
    if len(text) > maxlen:
        problems.append(f"{name} は {maxlen} 字までです")
        return ""
    return text


def parse_lineup(raw) -> tuple[Lineup | None, list[str], list[dict]]:
    """ブリーフの lineup。(読めた値, 問題の一覧, 商品の素の dict の並び)。

    商品は groups[].items[] に置く（id・image・images・item_type・color）。ad_compose.parse_brief は
    この並びを items として読む（ブリーフの items は lineup では使わない）。知らない style は
    style = None を返し、呼び出し側が collage に落とす（商品の並びは返す）。"""
    problems: list[str] = []
    if not isinstance(raw, dict):
        return None, ["lineup はオブジェクトにしてください"], []
    style = raw.get("style")
    out = Lineup(style=style if style in STYLES else "")
    out.title_note = _short_text(problems, "lineup.title_note", raw.get("title_note"), NOTE_MAX_LEN)
    out.brand = _short_text(problems, "lineup.brand", raw.get("brand"), BRAND_MAX_LEN)
    badges = raw.get("badges")
    if badges is not None and not isinstance(badges, list):
        problems.append("lineup.badges は配列にしてください")
        badges = []
    for i, b in enumerate(badges or []):
        t = _short_text(problems, f"lineup.badges[{i}]", b, BADGE_MAX_LEN)
        if t:
            out.badges.append(t)
    if len(out.badges) > BADGE_MAX:
        for t in out.badges[BADGE_MAX:]:
            out.notes.append(f"事実の札が多すぎて省いた（{t}）")
        out.badges = out.badges[:BADGE_MAX]
    groups = raw.get("groups")
    if not isinstance(groups, list) or not groups:
        problems.append("lineup.groups は 1 つ以上の配列にしてください")
        return out, problems, []
    raw_items: list[dict] = []
    seen: set[str] = set()
    for gi, g in enumerate(groups):
        if not isinstance(g, dict):
            problems.append(f"lineup.groups[{gi}] はオブジェクトにしてください")
            continue
        name = _short_text(problems, f"lineup.groups[{gi}].name", g.get("name"), NAME_MAX_LEN, True)
        price = _short_text(
            problems, f"lineup.groups[{gi}].price_text", g.get("price_text"), PRICE_MAX_LEN
        )
        sizes = _short_text(problems, f"lineup.groups[{gi}].sizes", g.get("sizes"), SIZES_MAX_LEN)
        grp = Group(name, price, sizes)
        its = g.get("items")
        if not isinstance(its, list) or not its:
            problems.append(f"lineup.groups[{gi}].items は 1 つ以上の配列にしてください")
            continue
        for ii, it in enumerate(its):
            if not isinstance(it, dict):
                problems.append(f"lineup.groups[{gi}].items[{ii}] はオブジェクトにしてください")
                continue
            iid = it.get("id")
            if isinstance(iid, bool) or not isinstance(iid, (str, int)) or str(iid).strip() == "":
                problems.append(f"lineup.groups[{gi}].items[{ii}].id が必要です")
                continue
            iid = str(iid).strip()
            if iid in seen:
                problems.append(f"lineup の商品の id が重複しています（{iid}）")
                continue
            seen.add(iid)
            color = _short_text(
                problems, f"lineup.groups[{gi}].items[{ii}].color", it.get("color"), COLOR_MAX_LEN
            )
            grp.items.append(LItem(iid, color))
            raw_items.append({**it, "id": iid})
        if len(grp.items) > MAX_ITEMS_PER_GROUP:
            for li in grp.items[MAX_ITEMS_PER_GROUP:]:
                out.notes.append(
                    f"1つの型に商品が多すぎて省いた（{grp.name} / {li.color or li.id}）"
                )
            keep = {li.id for li in grp.items[:MAX_ITEMS_PER_GROUP]}
            raw_items = [r for r in raw_items if r["id"] in keep or r["id"] not in _ids(grp)]
            grp.items = grp.items[:MAX_ITEMS_PER_GROUP]
        out.groups.append(grp)
    if out.style == "brand":  # ブランド特集は、型ごとに先頭の1点だけを載せる
        for g in out.groups:
            for li in g.items[1:]:
                out.notes.append(
                    f"ブランド特集は型ごとに1点だけ使う（{g.name} / {li.color or li.id}）"
                )
            g.items = g.items[:1]
        keep = {li.id for g in out.groups for li in g.items}
        raw_items = [r for r in raw_items if r["id"] in keep]
    if sum(len(g.items) for g in out.groups) > MAX_ITEMS:  # 載せきれない分は後ろから省く
        total = 0
        keep = set()
        for g in out.groups:
            kept = []
            for li in g.items:
                if total < MAX_ITEMS:
                    kept.append(li)
                    keep.add(li.id)
                    total += 1
                else:
                    out.notes.append(f"商品が多すぎて省いた（{g.name} / {li.color or li.id}）")
            g.items = kept
        out.groups = [g for g in out.groups if g.items]
        raw_items = [r for r in raw_items if r["id"] in keep]
    if out.style:
        lo, hi = GROUPS_RANGE[out.style]
        if not lo <= len(out.groups) <= hi:
            problems.append(
                f"lineup.style が {out.style} のとき groups は {lo}〜{hi} 個にしてください"
                f"（{len(out.groups)} 個）"
            )
        if out.style == "brand" and not out.brand:
            problems.append("lineup.style が brand のとき lineup.brand（題字）が必要です")
        if out.style == "colors" and sum(len(g.items) for g in out.groups) < 2:
            problems.append("lineup.style が colors のとき商品は 2 つ以上にしてください")
    return out, problems, raw_items


def _ids(g: Group) -> set[str]:
    return {li.id for li in g.items}


def item_label(lineup: Lineup, group: Group, item: LItem) -> str:
    """collage に落とすとき・brand の輪郭の文字（商品の名前と価格）。事実だけ。"""
    parts = [group.name]
    if lineup.style != "brand" and item.color:
        parts.append(item.color)
    if group.price_text:
        parts.append(group.price_text)
    return " ".join(parts)[:60]


# ---------------------------------------------------------------- 計画の型


@dataclass
class LText:
    kind: str  # name / color / sizes / note / num / badge_pill / price_pill / worn_note
    text: str
    cx: float
    cy: float
    size: int
    angle: float
    w: float
    h: float
    accent: bool = False
    marker: bool = False  # 手書きの蛍光ペンの下線
    group: int = -1

    def polygon(self, pad: float = 0.0) -> np.ndarray:
        return ad_single.rect_poly(self.cx, self.cy, self.w, self.h, self.angle, pad)


@dataclass
class Swatch:
    cx: float
    cy: float
    r: float
    color: str
    item_id: str = ""


@dataclass
class LineupPlan:
    style: str
    texts: list[LText] = field(default_factory=list)
    badges: list[ad_single.Badge] = field(default_factory=list)
    swatches: list[Swatch] = field(default_factory=list)
    rules: list[list[float]] = field(default_factory=list)  # 段の区切りの波線 [x, y, w]
    group_of: dict[str, int] = field(default_factory=dict)  # 商品の id → 型の番号
    groups_wanted: int = 0
    groups_placed: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------- 色見本


def dominant_color(png: Path) -> str:
    """切り抜きの中の服の色（#RRGGBB）。縁（影・ふち）を避けて、中ほどの画素の中央値。"""
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    if rgba.shape[0] > 240:
        s = 240 / rgba.shape[0]
        rgba = cv2.resize(
            rgba, (max(1, round(rgba.shape[1] * s)), 240), interpolation=cv2.INTER_AREA
        )
    solid = (rgba[..., 3] > 250).astype(np.uint8)
    inner = cv2.erode(solid, np.ones((9, 9), np.uint8)).astype(bool)
    if np.count_nonzero(inner) < 50:
        inner = solid.astype(bool)
    px = rgba[..., :3][inner]
    if len(px) == 0:
        return "#888888"
    r, g, b = (int(v) for v in np.median(px, axis=0))
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------- 描画（HTML + SVG）


def css(pal: Palette) -> str:
    return f"""
.lt{{position:absolute;white-space:nowrap;text-align:center;font-family:"Caveat","Yomogi";
font-weight:700;line-height:1.2;letter-spacing:.03em;color:{pal.text}}}
.lt.acc{{color:{pal.accent}}}
.lt.num{{font-family:"DM Sans","Noto Sans CJK JP",sans-serif;font-weight:900;letter-spacing:0}}
.lt.pill{{background:{pal.text};color:{pal.bg};border-radius:999px;letter-spacing:.06em}}
.lt.pillacc{{background:{pal.accent};border-radius:999px;letter-spacing:.06em}}
body{{--acc:{pal.accent}}}
svg.ov text.price{{font-family:"DM Sans","Noto Sans CJK JP",sans-serif;font-weight:900;
text-anchor:middle;letter-spacing:-.01em}}
svg.ov path.rule{{fill:none;stroke:{pal.doodle};stroke-width:3.4;stroke-linecap:round;
stroke-linejoin:round;vector-effect:non-scaling-stroke;opacity:.7}}
svg.ov circle.sw{{stroke:{pal.doodle};stroke-width:2.6;vector-effect:non-scaling-stroke}}
"""


def _wave(x: float, y: float, w: float) -> str:
    n = max(2, int(round(w / 80)))
    dx = w / n
    d = f"M{x:.1f} {y:.1f}"
    for i in range(n):
        d += f" q{dx / 4:.1f} {-7 if i % 2 == 0 else 7} {dx / 2:.1f} 0 t{dx / 2:.1f} 0"
    return d


def html_parts(plan: LineupPlan, pal: Palette, e: Callable[[str], str]) -> list[str]:
    out: list[str] = []
    for t in plan.texts:
        left, top = t.cx - t.w / 2, t.cy - t.h / 2
        cls = "lt"
        style = ""
        if t.kind == "num":
            cls += " num"
        if t.accent and t.kind not in ("badge_pill", "price_pill"):
            cls += " acc"
        if t.kind == "badge_pill":
            cls += " pill"
        if t.kind == "price_pill":
            cls += " pillacc"
            style += f"color:{pill_text_color(pal)};"
        if t.marker:  # 文字の下半分に敷く蛍光ペン
            style += (
                "background:linear-gradient(transparent 56%,"
                "color-mix(in srgb,var(--acc) 40%,transparent) 56%,"
                "color-mix(in srgb,var(--acc) 40%,transparent) 92%,transparent 92%);"
            )
        out.append(
            f'<div class="{cls}" style="left:{left:.1f}px;top:{top:.1f}px;width:{t.w:.1f}px;'
            f"height:{t.h:.1f}px;font-size:{t.size}px;line-height:{t.h:.1f}px;"
            f'transform:rotate({t.angle:.2f}deg);{style}">{e(t.text)}</div>'
        )
    return out


def pill_text_color(pal: Palette) -> str:
    return ad_single.badge_text_color(pal)


def svg_parts(plan: LineupPlan, pal: Palette, e: Callable[[str], str]) -> list[str]:
    out: list[str] = []
    for x, y, w in plan.rules:
        out.append(f'<path class="rule" d="{_wave(x, y, w)}"/>')
    for s in plan.swatches:
        out.append(
            f'<circle class="sw" cx="{s.cx:.1f}" cy="{s.cy:.1f}" r="{s.r:.1f}" fill="{s.color}"/>'
        )
    tc = ad_single.badge_text_color(pal)
    for b in plan.badges:
        out.append(
            f'<g transform="translate({b.cx:.1f} {b.cy:.1f}) rotate({b.angle:.1f})">'
            f'<path d="{ad_single.badge_path(b)}" fill="{pal.accent}" stroke="{pal.doodle}"'
            ' stroke-width="3" stroke-linejoin="round" vector-effect="non-scaling-stroke"'
            ' opacity=".98"/>'
            f'<text class="price" y="{b.size * 0.34:.1f}" font-size="{b.size}" fill="{tc}">'
            f"{e(b.text)}</text></g>"
        )
    return out


# ---------------------------------------------------------------- 検査


@dataclass
class LineupChecks:
    name_min_size: int = 0
    color_min_size: int = 0
    groups_wanted: int = 0
    groups_placed: int = 0
    badge_item_overlap_max: float = 0.0
    badge_text_px: int = 0
    text_out_of_safe_px: int = 0


def check_plan(
    plan: LineupPlan,
    *,
    union: np.ndarray,
    safe: tuple[float, float, float, float],
    texts_mask: np.ndarray,
    width: int,
) -> tuple[LineupChecks, list[str]]:
    """lineup の検査（文字と商品の重なりは ad_compose.check_layout が数える）。
    型の名前が読める大きさ・型が全部載っている・札が商品や文字にほとんど重ならない・文字が安全域の中。"""
    c = LineupChecks()
    problems: list[str] = []
    names = [t for t in plan.texts if t.kind == "name"]
    colors = [t for t in plan.texts if t.kind == "color"]
    c.name_min_size = min((t.size for t in names), default=0)
    c.color_min_size = min((t.size for t in colors), default=0)
    c.groups_wanted = plan.groups_wanted
    c.groups_placed = len(plan.groups_placed)
    if plan.style != "brand" and plan.style:
        if names and c.name_min_size < NAME_MIN * width / 1080:
            problems.append(f"型の名前が小さすぎる（{c.name_min_size}px）")
        if plan.style != "colors" and len(names) < len(plan.groups_placed):
            problems.append("型の名前を置けていない")
        if colors and c.color_min_size < COLOR_MIN * width / 1080:
            problems.append(f"色の名前が小さすぎる（{c.color_min_size}px）")
        lo = GROUPS_RANGE[plan.style][0] if plan.style in GROUPS_RANGE else 1
        if c.groups_placed < lo:  # 入りきらない型を後ろから省くのは許す（meta に載せた数が残る）
            problems.append(f"型が {c.groups_placed} つしか載っていない（{lo} つ以上要る）")
    H, W = union.shape
    for b in plan.badges:
        d = b.disc_mask((H, W))
        area = max(1, int(np.count_nonzero(d)))
        ov = int(np.count_nonzero(d & union)) / area
        c.badge_item_overlap_max = max(c.badge_item_overlap_max, round(ov, 3))
        c.badge_text_px += int(np.count_nonzero(d & texts_mask))
        poly = b.polygon()
        if ad_single.poly_outside_rect(poly, safe, 2):
            problems.append("価格の札が安全域の外にある")
    if c.badge_item_overlap_max > BADGE_ITEM_OVERLAP + 0.03:
        problems.append(f"価格の札が商品に重なりすぎ（{c.badge_item_overlap_max:.0%}）")
    if c.badge_text_px > TOLERANCE_PX:
        problems.append(f"価格の札が文字に重なっている（{c.badge_text_px}px）")
    sx0, sy0, sx1, sy1 = safe
    for t in plan.texts:
        p = t.polygon()
        if p[:, 0].min() < sx0 - 8 or p[:, 0].max() > sx1 + 8:
            c.text_out_of_safe_px += 1000
        if p[:, 1].min() < sy0 - 8 or p[:, 1].max() > sy1 + 8:
            c.text_out_of_safe_px += 1000
    if c.text_out_of_safe_px:
        problems.append("lineup の文字が安全域の外にある")
    return c, problems


def score_plan(
    *,
    union: np.ndarray,
    region: tuple[float, float, float, float],
) -> tuple[float, dict]:
    """点数（0〜100）。商品と文字の塊が範囲をまんべんなく使っているか（左右・上下の空きのつり合い・
    埋まり具合・縦の使い切り）。lineup は商品の大きさをそろえるので、大きさのばらつきは見ない。"""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    ys, xs = np.where(union)
    if len(xs) == 0:
        return 0.0, {}
    ml, mr = xs.min() - x0, x1 - xs.max()
    mt, mb = ys.min() - y0, y1 - ys.max()
    balance = max(0.0, 1 - (abs(ml - mr) / rw + abs(mt - mb) / rh))
    cover = float(np.count_nonzero(union[int(y0) : int(y1), int(x0) : int(x1)])) / (rw * rh)
    fill = max(0.0, 1 - abs(cover - 0.45) / 0.45)
    use = min(1.0, float(ys.max() - ys.min()) / (rh * 0.9))
    parts = {
        "balance": round(balance, 3),
        "fill": round(fill, 3),
        "use": round(use, 3),
        "cover": round(cover, 3),
    }
    return round(100 * (0.3 * balance + 0.4 * fill + 0.3 * use), 1), parts


def lineup_meta(plan: LineupPlan, omitted: int = 0) -> dict:
    return {
        "layout": LAYOUT_NAME,
        "lineup_style": plan.style,
        "groups_wanted": plan.groups_wanted,
        "groups_placed": len(plan.groups_placed),
        "price_badges": len(plan.badges),
        "lineup_notes": plan.notes,
    }


__all__ = [
    "LAYOUT_NAME",
    "STYLES",
    "Group",
    "LItem",
    "LText",
    "Lineup",
    "LineupPlan",
    "Swatch",
    "check_plan",
    "css",
    "dominant_color",
    "html_parts",
    "item_label",
    "lineup_meta",
    "parse_lineup",
    "score_plan",
    "svg_parts",
]
