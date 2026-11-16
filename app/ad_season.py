"""広告画像の季節（ブリーフの season の読み取りと、季節の落書き）。

ブリーフ（CRM から来る JSON）の `season` を読む。無い・知らない motif・形がおかしいときは None
（季節なし。今までと同じ見た目）。季節の落書きは、白い線の手描き風の SVG パス
（単位の箱 −50〜50 の中心）。
塗りは基本なしで、アクセント色の小さな塗り（`"a"` の層）だけ使う。モデルと Chrome は使わない。
"""

# ruff: noqa: E501  （パスの数値の列は折らない）

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

MOTIFS = (
    "early_spring",
    "spring",
    "early_summer",
    "summer",
    "late_summer",
    "autumn",
    "late_autumn",
    "winter",
    "midwinter",
)
STAMP_MAX = 28
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class Season:
    motif: str
    label: str = ""
    stamp: str = ""  # 画像の隅に置く小さなスタンプの文字（無ければ置かない）
    weather_line: str = ""  # 文面側で使う。画像には入れない
    stage: str = ""
    tmax: float | None = None
    tmin: float | None = None
    trend: str = ""
    estimated: bool = False

    def to_json(self) -> dict:
        return {
            "motif": self.motif,
            "label": self.label,
            "stamp": self.stamp,
            "stage": self.stage,
            "tmax": self.tmax,
            "tmin": self.tmin,
            "trend": self.trend,
            "estimated": self.estimated,
        }


def _str(v, maxlen: int) -> str:
    if not isinstance(v, str) or _CTRL.search(v):
        return ""
    return v.strip()[:maxlen]


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def parse_season(raw) -> Season | None:
    """ブリーフの season。dict でない・motif が無い/知らない値なら None（季節なし）。
    ほかの項目はおかしければ空にして続ける（壊れたブリーフで落とさない）。"""
    if not isinstance(raw, dict):
        return None
    motif = raw.get("motif")
    if not isinstance(motif, str) or motif not in MOTIFS:
        return None
    stamp = raw.get("stamp")
    stamp = stamp.strip() if isinstance(stamp, str) and not _CTRL.search(stamp) else ""
    if len(stamp) > STAMP_MAX:  # 長すぎるスタンプは小さな枠に入らない。置かない
        stamp = ""
    return Season(
        motif,
        _str(raw.get("label"), 20),
        stamp,
        _str(raw.get("weather_line"), 80),
        _str(raw.get("stage"), 20),
        _num(raw.get("tmax")),
        _num(raw.get("tmin")),
        _str(raw.get("trend"), 20),
        raw.get("estimated") is True,
    )


# ---------------------------------------------------------------- 線を作る道具


def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if v % 1 else f"{int(v)}"


def poly(pts, closed: bool = False) -> str:
    d = "M" + " L".join(f"{_f(x)} {_f(y)}" for x, y in pts)
    return d + ("Z" if closed else "")


def smooth(pts, closed: bool = False) -> str:
    """点列を通る滑らかな曲線（Catmull-Rom → 3次ベジェ）。"""
    n = len(pts)
    get = (lambda i: pts[i % n]) if closed else (lambda i: pts[max(0, min(n - 1, i))])
    d = f"M{_f(pts[0][0])} {_f(pts[0][1])}"
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{_f(c1[0])} {_f(c1[1])} {_f(c2[0])} {_f(c2[1])} {_f(p2[0])} {_f(p2[1])}"
    return d + ("Z" if closed else "")


def circle(cx: float, cy: float, r: float) -> str:
    return f"M{_f(cx - r)} {_f(cy)} a{_f(r)} {_f(r)} 0 1 0 {_f(2 * r)} 0 a{_f(r)} {_f(r)} 0 1 0 {_f(-2 * r)} 0Z"


def mirror(pts):
    return [(-x, y) for x, y in pts]


def rot(pts, deg: float, ox: float = 0.0, oy: float = 0.0):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(ox + x * c - y * s, oy + x * s + y * c) for x, y in pts]


def shift(pts, dx: float, dy: float):
    return [(x + dx, y + dy) for x, y in pts]


def wobble(pts, amp: float, seed: int):
    """手描きらしい小さなゆれ（決まった種。毎回同じ形）。"""
    r = random.Random(seed)
    return [(x + r.uniform(-amp, amp), y + r.uniform(-amp, amp)) for x, y in pts]


# ---------------------------------------------------------------- 形（層・当たり判定の箱）


@dataclass(frozen=True)
class Shape:
    layers: tuple  # ((パス, "s"（線だけ）か "a"（アクセント色の小さな塗り）), ...)
    w: float  # 当たり判定の箱（−50〜50 の中での幅・高さ）
    h: float
    rot: float = 25.0  # 傾けてよい角度の上限（±度）


def _spokes(n: int, r0: float, r1: float, cx=0.0, cy=0.0, start=-90.0) -> str:
    segs = []
    for k in range(n):
        a = math.radians(start + 360 * k / n)
        segs.append(
            f"M{_f(cx + r0 * math.cos(a))} {_f(cy + r0 * math.sin(a))}"
            f" L{_f(cx + r1 * math.cos(a))} {_f(cy + r1 * math.sin(a))}"
        )
    return " ".join(segs)


def _snowflake_a() -> str:
    segs = []
    for k in range(6):
        a = math.radians(-90 + 60 * k)
        ux, uy = math.cos(a), math.sin(a)
        nx, ny = -uy, ux
        segs.append(f"M0 0 L{_f(44 * ux)} {_f(44 * uy)}")
        for r, ln in ((24, 11), (35, 8)):
            bx, by = r * ux, r * uy
            for sgn in (-1, 1):
                ex = bx + ln * (ux * 0.7 + sgn * nx * 0.7)
                ey = by + ln * (uy * 0.7 + sgn * ny * 0.7)
                segs.append(f"M{_f(bx)} {_f(by)} L{_f(ex)} {_f(ey)}")
    return " ".join(segs)


def _snowflake_b() -> str:
    segs = []
    for k in range(6):
        a = math.radians(-90 + 60 * k)
        ux, uy = math.cos(a), math.sin(a)
        nx, ny = -uy, ux
        segs.append(f"M{_f(13 * ux)} {_f(13 * uy)} L{_f(46 * ux)} {_f(46 * uy)}")
        for r, ln, wide in ((22, 15, 0.9), (34, 10, 0.9)):
            bx, by = r * ux, r * uy
            for sgn in (-1, 1):
                ex = bx + ln * (ux * 0.5 + sgn * nx * wide)
                ey = by + ln * (uy * 0.5 + sgn * ny * wide)
                segs.append(f"M{_f(bx)} {_f(by)} L{_f(ex)} {_f(ey)}")
    hexa = [
        (13 * math.cos(math.radians(-60 + 60 * k)), 13 * math.sin(math.radians(-60 + 60 * k)))
        for k in range(6)
    ]
    segs.append(poly(hexa, closed=True))
    return " ".join(segs)


def _maple() -> str:
    pts = [
        (0, -48), (8, -26), (20, -34), (17, -14), (38, -20), (28, -2), (40, 12),
        (18, 14), (14, 30), (4, 22), (3, 34),
    ]  # fmt: skip
    right = pts
    left = mirror(pts[::-1])[1:-1]
    outline = right + [(0, 34)] + left
    return poly(wobble(outline, 0.8, 11), closed=True)


def _ginkgo_outline() -> str:
    top = [
        (-42, -22),
        (-34, -37),
        (-18, -44),
        (-6, -40),
        (0, -28),
        (6, -40),
        (18, -44),
        (34, -37),
        (42, -22),
    ]
    return smooth(wobble(top, 0.6, 5)) + " L0 28 Z"


def _sun() -> tuple:
    rays = []
    for k in range(12):
        a = math.radians(-90 + 30 * k)
        r1 = 33 if k % 2 == 0 else 31
        r2 = 47 if k % 2 == 0 else 40
        rays.append(
            f"M{_f(r1 * math.cos(a))} {_f(r1 * math.sin(a))} L{_f(r2 * math.cos(a))} {_f(r2 * math.sin(a))}"
        )
    return ((" ".join(rays), "s"), (circle(0, 0, 22), "a"))


def _sunflower() -> tuple:
    petals = [
        smooth(rot([(0, -13), (-8, -26), (-3, -40), (0, -46), (3, -40), (8, -26)], 360 * k / 12, 0, -10), closed=True)
        for k in range(12)
    ]  # fmt: skip
    stem = "M0 24 C-2 36 2 44 0 52"
    leaf = "M0 40 C14 30 28 36 32 44 C20 50 6 48 0 40"
    return ((" ".join(petals), "s"), (circle(0, -10, 12), "a"), (stem + " " + leaf, "s"))


def _sakura() -> tuple:
    pts = [(0, -6), (-9, -12), (-18, -28), (-9, -42), (0, -35), (9, -42), (18, -28), (9, -12)]
    petals = " ".join(smooth(rot(pts, 72 * k), closed=True) for k in range(5))
    stamens = _spokes(5, 6, 14, start=-54)
    return ((petals, "s"), (circle(0, 0, 4.5), "a"), (stamens, "s"))


def _daisy() -> tuple:
    petals = " ".join(
        smooth(rot([(0, -14), (-6, -26), (-3, -41), (0, -45), (3, -41), (6, -26)], 360 * k / 10), closed=True)
        for k in range(10)
    )  # fmt: skip
    return ((petals, "s"), (circle(0, 0, 11), "a"))


def _ume() -> tuple:
    petals = " ".join(
        circle(
            24 * math.cos(math.radians(-90 + 72 * k)), 24 * math.sin(math.radians(-90 + 72 * k)), 15
        )
        for k in range(5)
    )
    stamens = _spokes(5, 4, 11, start=-54)
    return ((petals, "s"), (circle(0, 0, 4), "a"), (stamens, "s"))


def _tulip() -> tuple:
    cup = "M-25 -36 C-30 -2 -18 20 0 20 C18 20 30 -2 25 -36 L11 -20 L0 -40 L-11 -20 Z"
    stem = "M0 20 C1 32 -1 42 0 52"
    leaf = "M0 44 C-24 42 -32 24 -30 12 C-14 16 -3 28 0 44"
    return ((cup, "a"), (stem + " " + leaf, "s"))


def _bud() -> tuple:
    twig = smooth([(-46, 40), (-26, 24), (0, 18), (24, 4), (46, -22)])

    def drop(x, y, a, s=1.0):
        pts = [
            (0, 0),
            (-7 * s, -9 * s),
            (-4 * s, -20 * s),
            (0, -26 * s),
            (4 * s, -20 * s),
            (7 * s, -9 * s),
        ]
        return smooth(shift(rot(pts, a), x, y), closed=True)

    buds = " ".join(
        [
            drop(-24, 26, -40, 1.0),
            drop(2, 18, 30, 1.2),
            drop(26, 4, -25, 1.0),
            drop(-6, 20, 160, 0.8),
        ]
    )
    return ((twig + " " + buds, "s"), (circle(46, -24, 4), "a"))


def _butterfly() -> tuple:
    upper = [(1, -4), (-8, -30), (-26, -44), (-44, -38), (-42, -18), (-24, -4), (-8, 0)]
    lower = [(1, 2), (-12, 6), (-30, 12), (-34, 28), (-22, 38), (-8, 26)]
    wings = " ".join(smooth(p, closed=True) for p in (upper, lower, mirror(upper), mirror(lower)))
    body = "M0 -16 L0 30"
    ant = "M0 -16 Q-4 -30 -14 -34 M0 -16 Q4 -30 14 -34"
    spots = " ".join(circle(x, y, 4.5) for x, y in ((-28, -26), (28, -26)))
    return ((wings + " " + body + " " + ant, "s"), (spots, "a"))


def _sprout() -> tuple:
    stem = "M0 48 C-1 30 2 14 0 0"
    left = smooth([(0, 0), (-12, -30), (-30, -40), (-46, -30), (-40, -10), (-20, 2)], closed=True)
    right = smooth([(0, 4), (12, -22), (28, -32), (44, -24), (38, -6), (18, 4)], closed=True)
    soil = "M-26 48 L26 48"
    return ((stem + " " + left + " " + right + " " + soil, "s"),)


def _petals() -> tuple:
    pts = [(0, 0), (-12, -13), (-9, -31), (0, -26), (9, -31), (12, -13)]
    ps = []
    for x, y, a in ((-30, -4, -40), (6, -22, 30), (26, 14, 120), (-10, 28, 200)):
        ps.append(smooth(shift(rot(pts, a), x, y), closed=True))
    return ((" ".join(ps), "s"),)


def _cloud() -> str:
    return "M-34 22 a14 14 0 0 1 -2 -28 a22 22 0 0 1 40 -10 a18 18 0 0 1 30 10 a12 12 0 0 1 0 28 Z"


def _umbrella() -> tuple:
    canopy = (
        "M-44 6 C-44 -26 -22 -36 0 -36 C22 -36 44 -26 44 6 Q33 -5 22 6 Q11 -5 0 6"
        " Q-11 -5 -22 6 Q-33 -5 -44 6 Z"
    )
    ribs = "M0 -36 L-22 6 M0 -36 L22 6 M0 -36 L0 6"
    handle = "M0 6 L0 38 C0 52 -16 52 -16 40 M0 -36 L0 -44"
    return ((canopy, "a"), (ribs + " " + handle, "s"))


def _raindrops() -> tuple:
    def drop(x, y, s=1.0):
        pts = [(0, -12), (6 * s, 0), (6 * s, 6), (0, 11), (-6 * s, 6), (-6 * s, 0)]
        return smooth(shift([(px, py * s) for px, py in pts], x, y), closed=True)

    ds = " ".join(
        [drop(-26, -14), drop(2, -2, 1.2), drop(30, -18), drop(-8, 24, 0.9), drop(26, 22, 1.0)]
    )
    lines = "M-44 6 l-5 10 M42 4 l-4 8"
    return ((ds + " " + lines, "s"),)


def _icecream() -> tuple:
    cone = "M-20 0 L0 52 L20 0"
    hatch = "M-12 14 L8 46 M12 14 L-8 46"
    scoop = (
        "M-28 0 C-38 0 -38 -22 -20 -24 C-18 -44 18 -44 20 -24 C38 -22 38 0 28 0"
        " Q22 10 14 0 Q7 10 0 0 Q-7 10 -14 0 Q-22 10 -28 0 Z"
    )
    return ((scoop, "a"), (cone + " " + hatch, "s"))


def _watermelon() -> tuple:
    outer = "M-46 -24 A46 46 0 0 0 46 -24 Z"
    inner = "M-39 -24 A39 39 0 0 0 39 -24"
    seeds = " ".join(
        f"M{x} {y} q-2.5 5 0 8 q2.5 -3 0 -8Z"
        for x, y in ((-18, -8), (0, 2), (18, -8), (-4, -14), (10, -16))
    )
    return ((outer + " " + inner, "s"), (seeds, "a"))


def _uchiwa() -> tuple:
    face = circle(0, -12, 36)
    ribs = " ".join(
        f"M0 34 L{_f(36 * math.sin(math.radians(a)))} {_f(-12 - 36 * math.cos(math.radians(a)))}"
        for a in (-50, -25, 0, 25, 50)
    )
    handle = "M-6 34 L-5 52 L5 52 L6 34"
    band = "M-30 -26 Q0 -40 30 -26"
    return ((face + " " + ribs + " " + handle + " " + band, "s"), (circle(0, -14, 8), "a"))


def _dragonfly() -> tuple:
    body = "M0 -28 L0 44"
    head = circle(0, -33, 6)
    up = [(0, -18), (-20, -32), (-46, -30), (-44, -22), (-22, -14)]
    low = [(0, -10), (-18, -14), (-42, -4), (-38, 4), (-18, 0)]
    wings = " ".join(smooth(p, closed=True) for p in (up, low, mirror(up), mirror(low)))
    return ((body + " " + wings, "s"), (head, "a"))


def _acorn() -> tuple:
    cap = "M-27 -2 C-30 -34 30 -34 27 -2 Z"
    hatch = "M-16 -12 l8 8 M-2 -18 l10 12 M12 -12 l6 8"
    stem = "M0 -29 Q2 -40 8 -44"
    nut = "M-21 -2 C-24 26 -8 44 0 50 C8 44 24 26 21 -2"
    return ((cap, "a"), (hatch + " " + stem + " " + nut, "s"))


def _mushroom() -> tuple:
    cap = "M-44 6 C-46 -42 46 -42 44 6 Z"
    stem = "M-15 6 C-13 26 -18 38 -20 48 L20 48 C18 38 13 26 15 6"
    spots = " ".join(circle(x, y, r) for x, y, r in ((-22, -12, 6), (2, -22, 7), (24, -10, 5)))
    return ((cap, "a"), (stem + " " + spots, "s"))


def _chestnut() -> tuple:
    nut = "M0 -44 C28 -30 42 -4 32 16 L-32 16 C-42 -4 -28 -30 0 -44 Z"
    burr = []
    for k in range(13):
        a = math.radians(180 - 180 * k / 12)
        r = 46 if k % 2 == 0 else 35
        burr.append((r * math.cos(a), 8 + r * math.sin(a) * 0.95))
    shine = "M-14 -12 Q-8 -26 0 -30"
    return ((nut, "a"), (poly(burr) + " " + shine, "s"))


def _leaf() -> str:
    outline = "M-42 24 C-40 -26 8 -48 44 -40 C46 4 6 38 -42 24 Z"
    vein = "M-42 24 L30 -28 M-18 8 L-22 -10 M-2 -4 L-4 -26 M8 -14 L22 -4 M-26 16 L-10 26"
    return outline + " " + vein


def _wind() -> str:
    return (
        "M-48 -16 L12 -16 C34 -16 34 -42 14 -40 "
        "M-48 4 L30 4 C54 4 52 32 30 30 "
        "M-34 24 L-6 24 C8 24 8 40 -4 40"
    )


def _mug() -> tuple:
    body = "M-28 -6 L-28 30 Q-28 44 -14 44 L14 44 Q28 44 28 30 L28 -6 Z"
    handle = "M28 4 C50 2 50 32 27 30"
    steam = (
        "M-12 -14 C-22 -24 -6 -30 -14 -42 M6 -14 C-4 -24 12 -30 4 -42 M22 -14 C14 -22 26 -26 20 -36"
    )
    return ((body, "s"), (handle + " " + steam, "s"), ("M-28 12 L28 12", "a"))


def _mitten() -> tuple:
    body = "M-24 30 C-32 8 -32 -30 -6 -42 C18 -48 30 -22 24 30 Z"
    thumb = "M24 18 C42 14 46 -2 36 -10 C30 -14 26 -8 24 -2"
    cuff = "M-26 30 L26 30 L26 46 L-26 46 Z"
    zig = "M-24 38 l6 -5 l6 5 l6 -5 l6 5 l6 -5 l6 5 l6 -5"
    return ((cuff, "a"), (body + " " + thumb + " " + zig, "s"))


def _beanie() -> tuple:
    dome = "M-34 12 C-40 -34 40 -34 34 12 Z"
    ribs = "M-12 12 C-14 -8 -12 -22 -8 -28 M12 12 C14 -8 12 -22 8 -28 M0 12 L0 -30"
    brim = "M-40 12 L40 12 L40 32 L-40 32 Z"
    brim_lines = "M-22 14 L-22 30 M-6 14 L-6 30 M10 14 L10 30 M26 14 L26 30"
    return ((brim, "a"), (circle(0, -40, 9), "a"), (dome + " " + ribs + " " + brim_lines, "s"))


def _snowdots() -> tuple:
    dots = " ".join(
        circle(x, y, r)
        for x, y, r in (
            (-34, -18, 4),
            (-8, -30, 6),
            (22, -14, 4),
            (36, -32, 3),
            (-22, 10, 6),
            (6, 4, 3.5),
            (30, 16, 5),
            (-40, 30, 3),
            (-4, 30, 4),
        )
    )
    stars = "M16 -34 l0 -9 m-4.5 4.5 l9 0 M-30 28 l0 8 m-4 -4 l8 0"
    return ((dots + " " + stars, "s"),)


def _tree() -> tuple:
    body = "M0 -42 L-20 -14 H-10 L-28 12 H-14 L-36 38 H36 L14 12 H28 L10 -14 H20 Z"
    trunk = "M-8 38 L-8 50 L8 50 L8 38"
    return ((body + " " + trunk, "s"), (circle(0, -46, 4), "a"))


def _build() -> dict[str, Shape]:
    return {
        "maple": Shape(
            (
                (
                    _maple()
                    + " M0 20 L0 46 M0 14 L0 -34 M0 14 L26 -14 M0 14 L-26 -14 M0 14 L22 8 M0 14 L-22 8",
                    "s",
                ),
            ),
            96,
            96,
            50,
        ),
        "ginkgo": Shape(
            (
                (
                    _ginkgo_outline()
                    + " M0 28 L0 50 M0 28 L-26 -24 M0 28 L-12 -34 M0 28 L12 -34 M0 28 L26 -24",
                    "s",
                ),
            ),
            90,
            100,
            50,
        ),
        "leaf": Shape(((_leaf(), "s"),), 92, 80, 45),
        "acorn": Shape(_acorn(), 56, 96, 30),
        "mushroom": Shape(_mushroom(), 92, 96, 22),
        "chestnut": Shape(_chestnut(), 94, 92, 40),
        "wind": Shape(((_wind(), "s"),), 100, 84, 15),
        "mug": Shape(_mug(), 100, 100, 14),
        "snowflake_a": Shape(((_snowflake_a(), "s"),), 92, 92, 30),
        "snowflake_b": Shape(((_snowflake_b(), "s"),), 94, 94, 30),
        "mitten": Shape(_mitten(), 88, 100, 25),
        "beanie": Shape(_beanie(), 84, 100, 20),
        "snowdots": Shape(_snowdots(), 94, 80, 40),
        "tree": Shape(_tree(), 76, 100, 12),
        "sakura": Shape(_sakura(), 82, 84, 60),
        "ume": Shape(_ume(), 88, 88, 60),
        "tulip": Shape(_tulip(), 66, 100, 22),
        "bud": Shape(_bud(), 98, 80, 30),
        "butterfly": Shape(_butterfly(), 92, 86, 30),
        "sprout": Shape(_sprout(), 96, 96, 14),
        "petals": Shape(_petals(), 100, 98, 60),
        "daisy": Shape(_daisy(), 92, 92, 40),
        "sun": Shape(_sun(), 96, 96, 40),
        "cloud": Shape(((_cloud(), "s"),), 94, 52, 10),
        "umbrella": Shape(_umbrella(), 92, 100, 28),
        "raindrops": Shape(_raindrops(), 100, 70, 15),
        "icecream": Shape(_icecream(), 76, 100, 28),
        "sunflower": Shape(_sunflower(), 92, 100, 25),
        "watermelon": Shape(_watermelon(), 96, 72, 22),
        "uchiwa": Shape(_uchiwa(), 80, 100, 30),
        "dragonfly": Shape(_dragonfly(), 96, 80, 30),
    }


SHAPES: dict[str, Shape] = _build()

# motif ごとの落書き（6 種類以上。汎用の星・波・矢印と混ぜて置く）
MOTIF_DOODLES: dict[str, tuple[str, ...]] = {
    "early_spring": ("ume", "bud", "sprout", "butterfly", "tulip", "petals"),
    "spring": ("sakura", "tulip", "bud", "butterfly", "sprout", "petals", "daisy"),
    "early_summer": ("umbrella", "raindrops", "cloud", "sun", "icecream", "sunflower"),
    "summer": ("sun", "cloud", "icecream", "sunflower", "watermelon", "uchiwa"),
    "late_summer": ("sunflower", "watermelon", "uchiwa", "dragonfly", "cloud", "sun"),
    "autumn": ("maple", "ginkgo", "leaf", "acorn", "mushroom", "chestnut", "wind"),
    "late_autumn": ("maple", "ginkgo", "leaf", "acorn", "mushroom", "chestnut", "wind", "mug"),
    "winter": ("snowflake_a", "mitten", "mug", "beanie", "snowdots", "tree", "wind"),
    "midwinter": ("snowflake_a", "snowflake_b", "mitten", "mug", "beanie", "snowdots", "tree"),
}
SEASON_SHARE = 0.72  # 落書きのうち季節の物の割合のめやす


def doodle_kinds(season: Season | None) -> tuple[str, ...]:
    return MOTIF_DOODLES[season.motif] if season else ()
