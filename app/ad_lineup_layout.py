"""新品のまとめ見せ（layout: lineup）の置き方。
app/ad_lineup.py の計画の型を、ad_compose の部品（商品の状態・マスク・落書き・着画の文字）で組む。
モデルも Chrome も使わない。

どの style も同じ手順: 商品を置ける範囲（ad_compose.layout_fixed）→ 着画を横に1枚（あれば）→
帯（共通の売りの一言と事実の札）→ style ごとの置き方（段に分けて、型の名前・サイズ・価格・
色の名前を添える）→ 落書き。置き方は規則だけで決まる（乱数の種は傾きと左右の入れ替えだけ）。
収まらないときは検査（check_layout）が fail にし、呼び出し側が商品を後ろから減らす。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np

from app import ad_compose as ac
from app import ad_lineup as lu
from app import ad_single as sg
from app.ad_palette import PALETTES, sinks


@dataclass
class Ctx:
    geo: ac.Geo
    ms: ac.Measurer
    rng: random.Random
    brief: ac.Brief
    plan: lu.LineupPlan
    region: tuple[float, float, float, float]
    groups: list[tuple[int, lu.Group, list[ac.Sprite]]]  # (型の番号, 型, 載せる商品)
    items: list[ac.ItemState]
    colors: dict[str, str]  # 商品の id → 色見本

    @property
    def W(self) -> int:
        return self.geo.w

    @property
    def tx(self) -> tuple[float, float]:
        """文字を置ける左右（安全域）。商品は画面の端に近いところまで置ける。"""
        return max(float(self.geo.mx), self.region[0]), min(
            float(self.geo.w - self.geo.mx), self.region[2]
        )


# ---------------------------------------------------------------- 部品


def _asp(sp: ac.Sprite) -> float:
    return sp.alpha.shape[1] / sp.alpha.shape[0]


def _item(sp: ac.Sprite, h: float, cx: float, cy: float, ang: float) -> ac.ItemState:
    return ac.ItemState(replace(sp, label=""), h, cx, cy, ang)


def fit_size(ms: ac.Measurer, text: str, max_w: float, steps) -> tuple[int, bool]:
    """大きい段から、幅が max_w に収まる最初の段。(大きさ, 収まったか)"""
    for s in steps:
        if ms.width("hand", text, s) * 1.04 + 6 <= max_w:
            return s, True
    return steps[-1], False


def text_at(ms, kind, text, size, cx, cy, angle=0.0, *, accent=False, marker=False, group=-1):
    w = ms.width("hand", text, size) * 1.04 + 6
    return lu.LText(kind, text, cx, cy, size, angle, w, size * 1.2, accent, marker, group)


def pill_at(ms, kind, text, size, cx, cy, angle, group=-1):
    w = ms.width("hand", text, size) * 1.06 + 34
    return lu.LText(kind, text, cx, cy, size, angle, w, size * 1.2 + 14, True, False, group)


def _extent(items: list[ac.ItemState]) -> tuple[float, float, float, float]:
    lo = min(it.cx - it.half_extent()[0] for it in items)
    hi = max(it.cx + it.half_extent()[0] for it in items)
    top = min(it.cy - it.half_extent()[1] for it in items)
    bot = max(it.cy + it.half_extent()[1] for it in items)
    return lo, hi, top, bot


def _shift(items: list[ac.ItemState], dx: float, dy: float) -> None:
    for it in items:
        it.cx += dx
        it.cy += dy


def _bisect(build, bw: float, bh: float, hmax: float) -> list[ac.ItemState]:
    """商品の高さを二分探索して、箱 (bw × bh) に収まる最大のものを選ぶ。"""
    lo, hi = 30.0, max(31.0, hmax)
    best = build(lo)
    for _ in range(16):
        mid = (lo + hi) / 2
        its = build(mid)
        a, b, c, d = _extent(its)
        if b - a <= bw and d - c <= bh:
            best, lo = its, mid
        else:
            hi = mid
    return best


def place_fan(sps, box, hmax, rng, *, flip=False) -> list[ac.ItemState]:
    """同じ型の色ちがいを、重ねてずらして扇のように（後ろの物は少し高く、前の物に重なる）。"""
    x0, y0, x1, y1 = box
    n = len(sps)
    base = list(lu.FAN_ANGLES.get(n, (0.0,) * n)) if n > 1 else [rng.choice((-1, 1)) * 2.5]
    if flip:
        base = base[::-1]
    angs = [a + rng.uniform(-1.0, 1.0) for a in base]

    def build(h: float):
        ws = [_asp(s) * h for s in sps]
        xs = [0.0]
        for i in range(1, n):
            xs.append(xs[-1] + (ws[i - 1] + ws[i]) / 2 * (1 - lu.FAN_OVERLAP))
        its = []
        for i, s in enumerate(sps):
            dy = -0.045 * h * (n - 1 - i) / max(1, n - 1)
            its.append(_item(s, h, xs[i], dy, angs[i]))
        return its

    its = _bisect(build, x1 - x0, y1 - y0, hmax)
    a, b, c, d = _extent(its)
    _shift(its, (x0 + x1) / 2 - (a + b) / 2, (y0 + y1) / 2 - (c + d) / 2)
    return its


def place_line(sps, box, hmax, rng, *, wobble=0.0) -> list[ac.ItemState]:
    """同じ高さで横に並べる。少しずつ傾け、上下を互い違いにずらす（wobble は高さに対する割合）。"""
    x0, y0, x1, y1 = box
    n = len(sps)
    tilts = [(-1) ** (i + rng.randint(0, 1)) * rng.uniform(1.5, 3.5) for i in range(n)]

    def build(h: float):
        ws = [_asp(s) * h for s in sps]
        total = sum(ws) + lu.ITEM_GAP * (n - 1)
        x = -total / 2
        its = []
        for i, s in enumerate(sps):
            dy = wobble * h * (0.5 if i % 2 == 0 else -0.5)
            its.append(_item(s, h, x + ws[i] / 2, dy, tilts[i]))
            x += ws[i] + lu.ITEM_GAP
        return its

    its = _bisect(build, x1 - x0, y1 - y0, hmax)
    a, b, c, d = _extent(its)
    _shift(its, (x0 + x1) / 2 - (a + b) / 2, (y0 + y1) / 2 - (c + d) / 2)
    # 横は箱いっぱいに広げる（余りを間に配る）
    if n > 1:
        spare = (x1 - x0) - (b - a)
        mean_w = sum(it.w for it in its) / n
        step = min(spare / (n - 1), 0.9 * mean_w)
        for i, it in enumerate(its):
            it.cx += (i - (n - 1) / 2) * step
    return its


# ---------------------------------------------------------------- 帯と着画


def place_band(ctx: Ctx, lineup: lu.Lineup) -> float:
    """範囲の上に、共通の売りの一言（蛍光ペン）と事実の札を置く。使った高さを返す。"""
    x0, y0, _, _ = ctx.region
    x0, x1 = ctx.tx
    rw = x1 - x0
    if not lineup.title_note and not lineup.badges:
        return 0.0
    cy = y0 + lu.BAND_H / 2
    note_w = 0.0
    if lineup.title_note:
        size, _ = fit_size(ctx.ms, lineup.title_note, rw * 0.66, lu.NOTE_STEPS)
        t = text_at(ctx.ms, "note", lineup.title_note, size, 0, cy, -1.2, marker=True)
        t.cx = x0 + 8 + t.w / 2
        ctx.plan.texts.append(t)
        note_w = t.w + 8
    x = x1
    for b in reversed(lineup.badges):
        p = pill_at(ctx.ms, "badge_pill", b, lu.BADGE_PILL_SIZE, 0, cy, ctx.rng.uniform(-3, 3))
        if x - p.w < x0 + note_w + 20:
            ctx.plan.notes.append(f"事実の札が入りきらず省いた（{b}）")
            continue
        p.cx = x - p.w / 2
        ctx.plan.texts.append(p)
        x = p.cx - p.w / 2 - 12
    return lu.BAND_H + 10


def place_worn(ctx: Ctx, worn: ac.Sprite, region):
    """着画を横の端に1枚。(着画の商品, 残りの範囲)。幅が足りなければ置かない（None）。"""
    x0, y0, x1, y1 = region
    rh = y1 - y0
    asp = _asp(worn)
    w = lu.WORN_W_RATIO * ctx.W
    h = min(w / asp, 0.86 * rh)
    w = h * asp
    left = ctx.rng.random() < 0.5
    ang = ctx.rng.choice((-1, 1)) * ctx.rng.uniform(1.0, 3.0)
    it = ac.ItemState(worn, h, 0.0, (y0 + y1) / 2 + 0.03 * rh, ang)
    hx, _ = it.half_extent()
    if left:
        it.cx = x0 + hx + 6
        rest = (it.cx + hx + lu.WORN_GAP, y0, x1, y1)
    else:
        it.cx = x1 - hx - 6
        rest = (x0, y0, it.cx - hx - lu.WORN_GAP, y1)
    ac.clamp_item(it, region)
    return it, rest


# ---------------------------------------------------------------- 色の名前・名前の束


def add_color_labels(ctx: Ctx, group_items: list[list[ac.ItemState]], *, swatch: bool) -> None:
    """商品の下に色の名前（と色見本）。後ろの物は、前の物に隠れていない部分の下へ。"""
    size = (ctx.geo.w, ctx.geo.h)
    allitems = [it for g in group_items for it in g]
    masks = {id(it): it.mask(size) for it in allitems}
    union = np.zeros((ctx.geo.h, ctx.geo.w), bool)
    for m in masks.values():
        union |= m
    taken = union.copy()  # 色の名前を置くときに避ける物: 商品に加え、すでにある文字と価格の札
    for t in ctx.plan.texts:
        sg.paint_poly(t.polygon(4), taken)
    for b in ctx.plan.badges:
        taken |= sg.dilate(b.disc_mask((ctx.geo.h, ctx.geo.w)), 6)
    entries = []  # (item, 色の名前, 型の番号, 見えている部分の下の x・y)
    for gi_pos, its in enumerate(group_items):
        gi = ctx.groups[gi_pos][0]
        lis = ctx.groups[gi_pos][1].items
        ids = {sp.id: li for li in lis for sp in ctx.groups[gi_pos][2] if sp.id == li.id}
        for it in its:
            li = ids.get(it.sprite.id)
            if li is None or not li.color:
                continue
            later = np.zeros_like(union)
            for o in allitems[allitems.index(it) + 1 :]:
                later |= masks[id(o)]
            vis = masks[id(it)] & ~later
            ys, xs = np.where(vis)
            if len(xs) == 0:
                continue
            low = ys > ys.max() - max(10, 0.22 * it.h)
            entries.append((it, li.color, gi, float(np.median(xs[low])), float(ys.max())))
    if not entries:
        return
    steps = lu.COLOR_STEPS
    sw_extra = 2 * lu.SWATCH_R + 8 if swatch else 0
    chosen = None
    for fs in steps:
        chosen = _label_row(ctx, entries, fs, sw_extra, taken)
        if chosen is not None:
            break
    if chosen is None:
        chosen = _label_row(ctx, entries, steps[-1], sw_extra, taken, force=True) or []
    ctx.plan.texts.extend(chosen)


def _label_row(ctx, entries, fs, sw_extra, union, force=False):
    """色の名前を商品の下に並べる。横にぶつかるものは左右へ離す（縦にはずらさない）。"""
    tx0, tx1 = ctx.tx
    gap = 8
    ws = [ctx.ms.width("hand", color, fs) * 1.04 + 6 for _, color, *_ in entries]
    tot = [w + sw_extra for w in ws]
    cs = [e[3] for e in entries]
    ys = [e[4] + 10 + fs * 0.62 for e in entries]
    order = sorted(range(len(entries)), key=lambda i: cs[i])
    near = [  # 同じ行（縦に近い）の組。段をまたぐ組は比べない
        (a, b)
        for ia, a in enumerate(order)
        for b in order[ia + 1 :]
        if abs(ys[a] - ys[b]) <= 1.2 * fs
    ]
    lims = []  # 商品・ほかの文字にぶつからない左右の範囲（その名前の行の高さで測る）
    for i in range(len(cs)):
        band = union[max(0, int(ys[i] - 0.6 * fs)) : int(ys[i] + 0.6 * fs) + 1]
        blocked = np.where(band.any(axis=0))[0]
        x = int(min(max(cs[i], 0), union.shape[1] - 1))
        left = blocked[blocked < x]
        right = blocked[blocked > x]
        lo = max(tx0, float(left.max()) + 4 if len(left) else tx0)
        hi = min(tx1, float(right.min()) - 4 if len(right) else tx1)
        lims.append((lo, hi))
        if hi - lo < tot[i] + 8 and not force:
            return None

    def clamp_all():
        for i in range(len(cs)):
            lo, hi = lims[i]
            if hi - lo >= tot[i] + 8:
                cs[i] = min(max(cs[i], lo + tot[i] / 2 + 4), hi - tot[i] / 2 - 4)
            else:
                cs[i] = min(max(cs[i], tx0 + tot[i] / 2 + 4), tx1 - tot[i] / 2 - 4)

    clamp_all()  # 先に範囲へ収める（収めたあとに離す）
    for _ in range(60):
        moved = False
        for a, b in near:
            if cs[a] > cs[b]:
                a, b = b, a
            over = (tot[a] + tot[b]) / 2 + gap - (cs[b] - cs[a])
            if over > 0:
                cs[a] -= over / 2
                cs[b] += over / 2
                moved = True
        clamp_all()
        if not moved:
            break
    if not force:  # 離しきれなかった（幅が足りない）なら、もっと小さい文字で
        for a, b in near:
            lo, hi = (a, b) if cs[a] <= cs[b] else (b, a)
            if cs[hi] - cs[lo] < (tot[a] + tot[b]) / 2 + gap - 1:
                return None
    out: list[lu.LText] = []
    sws: list[lu.Swatch] = []
    polys = []
    for k, (it, color, gi, _x, _y) in enumerate(entries):
        t = lu.LText(
            "color", color, cs[k] + sw_extra / 2, ys[k], fs, (-1) ** k * 1.2, ws[k], fs * 1.2,
            False, False, gi,
        )  # fmt: skip
        base_cx = t.cx
        for dx in (0, 12, -12, 24, -24, 40, -40, 60, -60, 90, -90, 130, -130):
            t.cx = base_cx + dx  # 商品・ほかの文字に重なるなら、まず横へずらす
            lo = t.cx - t.w / 2 - sw_extra
            if lo < tx0 + 2 or t.cx + t.w / 2 > tx1 - 2:
                continue
            if sg.poly_overlap(t.polygon(2), union) <= lu.TOLERANCE_PX:
                break
        else:
            t.cx = base_cx
            for _ in range(6):  # 横で解けなければ下へ（36px まで）
                if sg.poly_overlap(t.polygon(2), union) <= lu.TOLERANCE_PX:
                    break
                t.cy += 6
            else:
                if not force:
                    return None
        p = sg.rect_poly(
            t.cx - sw_extra / 2, t.cy, t.w + sw_extra + 4, t.h, t.angle, 4
        )  # 色見本こみ
        if not force and any(_poly_hit(p, q) for q in polys):
            return None
        polys.append(p)
        out.append(t)
        if sw_extra:
            r = lu.SWATCH_R
            sw = lu.Swatch(
                t.cx - t.w / 2 - 8 - r, t.cy, r, ctx.colors.get(it.sprite.id, "#888"), it.sprite.id
            )
            if sg.poly_overlap(sg.rect_poly(sw.cx, sw.cy, 2 * r, 2 * r), union) > lu.TOLERANCE_PX:
                if not force:
                    return None
            sws.append(sw)
    ctx.plan.swatches = [s for s in ctx.plan.swatches if s.item_id not in {x.item_id for x in sws}]
    ctx.plan.swatches.extend(sws)
    return out


def _poly_hit(a: np.ndarray, b: np.ndarray) -> bool:
    """凸四角形どうしの重なり（分離軸）。"""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def name_stack(
    ctx: Ctx,
    gi: int,
    g: lu.Group,
    name_size: int,
    box,
    *,
    num: str = "",
    angle: float,
    badge_ok: bool,
) -> None:
    """型の名前・サイズ・価格（と番号）を、箱の中に縦に積む。上から 番号 → 名前 → サイズ → 価格。"""
    x0, y0, x1, y1 = box
    cx, cw, ch = (x0 + x1) / 2, x1 - x0, y1 - y0
    ms = ctx.ms
    for compact in (False, True):  # 段が低くて入らないときは、サイズと価格を小さめにして詰める
        gap = 14 if compact else 15
        parts: list[tuple[str, float]] = []
        if num:
            parts.append(("num", lu.NUM_SIZE * 1.1))
        parts.append(("name", name_size * 1.2))
        sizes_fs = 0
        if g.sizes:
            steps = (lu.SIZES_MIN,) if compact else lu.SIZES_STEPS
            sizes_fs, _ = fit_size(ms, g.sizes, cw, steps)
            parts.append(("sizes", sizes_fs * 1.2))
        badge = None
        price_fs = 0
        if g.price_text:
            r = lu.BADGE_R_RATIO * ctx.W
            need_badge = sum(h for _, h in parts) + 2 * r + gap * len(parts)
            if badge_ok and need_badge <= ch and not compact:
                fs = sg.fit_price(g.price_text, r, ms)
                badge = sg.Badge(
                    g.price_text, cx, 0.0, r, fs, ctx.rng.uniform(-7, 7),
                    ctx.rng.choice(("scallop", "star")), g.price_text,
                )  # fmt: skip
                parts.append(("badge", 2 * r))
            else:
                steps = (lu.PRICE_PILL_STEPS[-1],) if compact else lu.PRICE_PILL_STEPS
                price_fs = next(
                    (s for s in steps if ms.width("hand", g.price_text, s) + 34 <= cw), steps[-1]
                )
                parts.append(("price", price_fs * 1.2 + 14))
        total = sum(h for _, h in parts) + gap * (len(parts) - 1)
        if total <= ch:
            break
    y = y0 + max(0.0, (ch - total) / 2)
    for kind, h in parts:
        yc = y + h / 2
        if kind == "num":
            t = text_at(ms, "num", num, lu.NUM_SIZE, cx, yc, angle, accent=True, group=gi)
            t.w = ms.width("head", num, lu.NUM_SIZE) + 6
            ctx.plan.texts.append(t)
        elif kind == "name":
            ctx.plan.texts.append(
                text_at(ms, "name", g.name, name_size, cx, yc, angle, accent=False, group=gi)
            )
        elif kind == "sizes":
            ctx.plan.texts.append(
                text_at(ms, "sizes", g.sizes, sizes_fs, cx, yc, angle * 0.6, group=gi)
            )
        elif kind == "badge":
            badge.cy = yc
            ctx.plan.badges.append(badge)
        else:
            ctx.plan.texts.append(
                pill_at(ms, "price_pill", g.price_text, price_fs, cx, yc, -angle, gi)
            )
        y += h + gap


def common_name_size(ctx: Ctx, max_w: float, steps=lu.NAME_STEPS) -> tuple[int, float]:
    """すべての型で同じ大きさにそろえる。いちばん長い名前が max_w に収まる大きい段。(大きさ, 幅)"""
    longest = max((g.name for _, g, _ in ctx.groups), key=lambda s: ctx.ms.width("hand", s, 100))
    size, _ = fit_size(ctx.ms, longest, max_w, steps)
    return size, ctx.ms.width("hand", longest, size) * 1.04 + 6


# ---------------------------------------------------------------- style ごとの置き方


def rows_split(region, n: int, gap: float):
    x0, y0, x1, y1 = region
    rh = (y1 - y0 - gap * (n - 1)) / n
    return [(x0, y0 + i * (rh + gap), x1, y0 + i * (rh + gap) + rh) for i in range(n)]


def build_compare(ctx: Ctx, y0: float) -> list[list[ac.ItemState]]:
    x0, _, x1, y1 = ctx.region
    tx0, tx1 = ctx.tx
    rows = rows_split((x0, y0, x1, y1), len(ctx.groups), lu.ROW_GAP)
    name_size, name_w = common_name_size(ctx, 0.36 * (tx1 - tx0), lu.NAME_STEPS[1:])
    col_w = name_w + 24
    out = []
    first_left = ctx.rng.random() < 0.5
    for k, ((gi, g, sps), row) in enumerate(zip(ctx.groups, rows, strict=True)):
        left = first_left if k % 2 == 0 else not first_left
        rx0, ry0, rx1, ry1 = row
        if left:
            col = (tx0, ry0, tx0 + col_w, ry1 - lu.LABEL_H)
            fan_box = (tx0 + col_w + 10, ry0, rx1, ry1 - lu.LABEL_H)
        else:
            col = (tx1 - col_w, ry0, tx1, ry1 - lu.LABEL_H)
            fan_box = (rx0, ry0, tx1 - col_w - 10, ry1 - lu.LABEL_H)
        its = place_fan(sps, fan_box, ry1 - ry0 - lu.LABEL_H, ctx.rng, flip=k % 2 == 1)
        out.append(its)
        row_h = ry1 - ry0
        name_stack(
            ctx, gi, g, name_size, (col[0], col[1], col[2], ry1 - 14),
            angle=(-2.4 if left else 2.4), badge_ok=row_h >= lu.BADGE_ROOM_ROW_H,
        )  # fmt: skip
        ctx.items.extend(its)
    return out


def build_catalog(ctx: Ctx, y0: float) -> list[list[ac.ItemState]]:
    x0, _, x1, y1 = ctx.region
    rw = x1 - x0
    rows = rows_split((x0, y0, x1, y1), len(ctx.groups), 34)
    name_size, name_w = common_name_size(ctx, 0.31 * (ctx.tx[1] - ctx.tx[0]), (66, 60, 56, 52))
    col_w = max(name_w + 16, 0.2 * rw)
    out = []
    for k, ((gi, g, sps), row) in enumerate(zip(ctx.groups, rows, strict=True)):
        rx0, ry0, rx1, ry1 = row
        col = (ctx.tx[0], ry0, ctx.tx[0] + col_w, ry1 - 8)
        box = (ctx.tx[0] + col_w + 14, ry0, rx1, ry1 - lu.LABEL_H)
        its = place_line(sps, box, ry1 - ry0 - lu.LABEL_H, ctx.rng, wobble=0.02)
        out.append(its)
        name_stack(
            ctx, gi, g, name_size, col, num=f"{k + 1:02d}", angle=-2.0, badge_ok=False,
        )  # fmt: skip
        if k < len(rows) - 1:
            ctx.plan.rules.append([rx0 + 6, ry1 + 17, rw - 12])
        ctx.items.extend(its)
    return out


def build_colors(ctx: Ctx, y0: float) -> list[list[ac.ItemState]]:
    x0, _, x1, y1 = ctx.region
    tx0, tx1 = ctx.tx
    rw = x1 - x0
    gi, g, sps = ctx.groups[0]
    size, _ = fit_size(ctx.ms, g.name, 0.62 * (tx1 - tx0), lu.NAME_STEPS)
    # 見出し: 左に型の名前とサイズ、右に価格
    name_h = size * 1.2
    sizes_fs = fit_size(ctx.ms, g.sizes, 0.5 * rw, lu.SIZES_STEPS)[0] if g.sizes else 0
    head_h = name_h + (sizes_fs * 1.2 + 22 if g.sizes else 0)
    r = lu.BADGE_R_RATIO * ctx.W
    head_h = max(head_h, 2 * r + 6 if g.price_text else 0)
    nw = ctx.ms.width("hand", g.name, size) * 1.04 + 6
    ctx.plan.texts.append(
        text_at(ctx.ms, "name", g.name, size, tx0 + 8 + nw / 2, y0 + name_h / 2, -2.2, group=gi)
    )
    if g.sizes:
        sw = ctx.ms.width("hand", g.sizes, sizes_fs) * 1.04 + 6
        ctx.plan.texts.append(
            text_at(
                ctx.ms, "sizes", g.sizes, sizes_fs, tx0 + 8 + sw / 2,
                y0 + name_h + sizes_fs * 0.6 + 18, -1.4, group=gi,
            )  # fmt: skip
        )  # fmt: skip
    if g.price_text:
        fs = sg.fit_price(g.price_text, r, ctx.ms)
        ctx.plan.badges.append(
            sg.Badge(g.price_text, tx1 - r - 6, y0 + head_h / 2, r, fs, ctx.rng.uniform(-7, 7),
                     ctx.rng.choice(("scallop", "star")), g.price_text)
        )  # fmt: skip
    top = y0 + head_h + 24
    n = len(sps)
    best = None
    for rows_n in (1, 2, 3):
        if rows_n > n:
            break
        per = [n // rows_n + (1 if i < n % rows_n else 0) for i in range(rows_n)]
        idx, chunks = 0, []
        for c in per:
            chunks.append(sps[idx : idx + c])
            idx += c
        asp_sum = max(sum(_asp(s) for s in ch) for ch in chunks)
        widest = max(len(ch) for ch in chunks)
        h_w = (rw * 0.94 - lu.ITEM_GAP * (widest - 1)) / asp_sum
        h_h = (y1 - top - rows_n * lu.LABEL_H - (rows_n - 1) * lu.ROW_GAP) / rows_n
        h = min(h_w, h_h, 0.62 * ctx.geo.h)
        if best is None or h > best[0] * 1.08:
            best = (h, chunks)
    h, chunks = best
    rows = rows_split((x0, top, x1, y1), len(chunks), lu.ROW_GAP)
    out = []
    its_all = []
    for k, (ch, row) in enumerate(zip(chunks, rows, strict=True)):
        rx0, ry0, rx1, ry1 = row
        inset = 0.035 * rw * (1 if k % 2 else -1) if len(chunks) > 1 else 0
        its = place_line(
            ch, (rx0 + max(0, inset), ry0, rx1 + min(0, inset), ry1 - lu.LABEL_H), h * 1.02,
            ctx.rng, wobble=0.05,
        )  # fmt: skip
        its_all.extend(its)
    out.append(its_all)
    ctx.items.extend(its_all)
    return out


STYLE_BUILDERS = {"compare": build_compare, "catalog": build_catalog, "colors": build_colors}


# ---------------------------------------------------------------- ブランド特集（collage + 題字）


def add_masthead(fixed: ac.Fixed, brief: ac.Brief, geo: ac.Geo, ms: ac.Measurer) -> None:
    """ブランドの題字を、見出しの下に大きく置く（範囲はその分だけ下げる）。"""
    lineup = brief.lineup
    if lineup is None or lineup.style != "brand" or not lineup.brand:
        return
    x0, y0, x1, y1 = fixed.region
    content_w = geo.w - 2 * geo.mx
    size = lu.MASTHEAD_STEPS[-1]
    for s in lu.MASTHEAD_STEPS:
        if ms.width("head", lineup.brand, s) <= content_w * 0.94 and s * 1.1 <= 0.2 * geo.h:
            size = s
            break
    w = ms.width("head", lineup.brand, size)
    h = size * ac.HEAD_LINE_HEIGHT
    blk = ac.Block(
        "masthead", [lineup.brand], "head", size, geo.mx, y0 - 6, w, h, angle=-1.5,
        line_widths=[w],
    )  # fmt: skip
    fixed.blocks.append(blk)
    fixed.region = (x0, y0 + h + 4, x1, y1)
    if size == lu.MASTHEAD_STEPS[-1] and w > content_w * 0.94:
        fixed.notes.append("題字が幅に収まらない")


# ---------------------------------------------------------------- 全体


def block_for_text(t: lu.LText) -> ac.Block:
    """文字の占有を、着画の文字の置き場所探し（ad_compose.place_texts）に教える。"""
    return ac.Block(
        "lt", [t.text], "hand", t.size, t.cx - t.w / 2, t.cy - t.h / 2, t.w, t.h, angle=t.angle
    )


def block_for_badge(b: sg.Badge) -> ac.Block:
    return ac.Block("lt", [b.text], "hand", b.size, b.cx - b.r, b.cy - b.r, 2 * b.r, 2 * b.r)


def build_lineup_layout(
    brief: ac.Brief,
    sprites: list[ac.Sprite],
    fmt: str,
    seed: int,
    ms: ac.Measurer,
    worn: ac.Sprite | None = None,
) -> ac.Layout:
    geo = ac.geometry(fmt)
    size = (geo.w, geo.h)
    fixed = ac.layout_fixed(brief, fmt, ms)
    lineup = brief.lineup
    smap = {sp.id: sp for sp in sprites}
    rng = random.Random(f"{seed}:lineup")
    plan = lu.LineupPlan(lineup.style, groups_wanted=len(lineup.groups), notes=list(lineup.notes))
    groups = []
    for gi, g in enumerate(lineup.groups):
        sps = [smap[li.id] for li in g.items if li.id in smap]
        if sps:
            groups.append((gi, g, sps))
            plan.groups_placed.append(gi)
            for sp in sps:
                plan.group_of[sp.id] = gi
    region = fixed.region
    ctx = Ctx(geo, ms, rng, brief, plan, region, groups, [], {})
    for _, _, sps in groups:
        for sp in sps:
            ctx.colors[sp.id] = lu.dominant_color(sp.png_path)
    worn_item = None
    if worn is not None:
        worn_item, region = place_worn(ctx, worn, region)
    ctx.region = region
    y0 = region[1]
    y0 += place_band(ctx, lineup) if (lineup.title_note or lineup.badges) else 0.0
    built = STYLE_BUILDERS[lineup.style](ctx, y0)
    add_color_labels(ctx, built, swatch=lineup.style in ("colors", "compare"))
    items = list(ctx.items)
    bg = PALETTES[brief.palette].bg
    if worn_item is not None:
        items.append(worn_item)
    for it in items:
        it.glow = sinks(it.sprite.light, bg)
    lay = ac.Layout(
        fmt, geo.w, geo.h, seed, brief.palette, fixed.blocks, fixed.region, items, [],
        list(fixed.notes), 1, 1.0, brief.palette_requested or brief.palette,
        brief.palette_changed_reason, brief.season,
    )  # fmt: skip
    lay.layout, lay.lineup = lu.LAYOUT_NAME, plan
    if worn_item is not None and worn_item.sprite.label:  # 着画の文字（輪郭に沿わせる）
        extra = [block_for_text(t) for t in plan.texts] + [block_for_badge(b) for b in plan.badges]
        tmp = ac.Fixed([*fixed.blocks, *extra], fixed.region, [])
        ac.place_texts(items, tmp, geo, ms, rng, ac.LABEL_STEPS, straight=True)
    if worn_item is not None:
        lay.worn_item_id = worn_item.sprite.id.removesuffix("#worn")
    occ = np.zeros((geo.h, geo.w), bool)
    for b in lay.blocks:
        occ |= b.mask(size)
    for it in items:
        occ |= it.mask(size)
        if it.arc is not None:
            ac.raster_band(it.arc[: max(2, int(it.text_len / ac.CONTOUR_STEP) + 1)], it.fs).paint(
                occ
            )
    for t in plan.texts:
        sg.paint_poly(t.polygon(6), occ)
    for b in plan.badges:
        occ |= sg.dilate(b.disc_mask((geo.h, geo.w)), 8)
    for s in plan.swatches:
        sg.paint_poly(sg.rect_poly(s.cx, s.cy, 2 * s.r, 2 * s.r, 0, 6), occ)
    lay.doodles = ac.place_doodles(
        random.Random(f"{seed}:doodle"), geo, occ, lay.region, brief.season,
        limit=lu.DOODLE_LIMIT, size_scale=lu.DOODLE_SIZE_SCALE,
    )  # fmt: skip
    return lay
