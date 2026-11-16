"""広告画像の色（パレット）と、商品が地に沈まないようにする規則。

- パレットは 10 色（名前は CRM と共通。変えない）。地・文字・アクセント・落書きの線・影を持つ。
  明るい地（cream・sky）は文字と線を濃くする。どの色も 文字・アクセント・線と地の明暗比は 4.5:1 以上
- 商品の明るさは切り抜きの画素から測る（アルファで重み付けした L* の分布）。地との差が小さい商品は
  「沈む」。まず同じ系統の別の色に替え、替えても沈む商品は縁に地より少し明るい（明るい地では暗い）
  やわらかい光の縁取りを足す。**商品の画素そのものは変えない**
モデルと Chrome は使わない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- 色の計算


def _rgb(hexcolor: str) -> tuple[float, float, float]:
    h = hexcolor.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def luminance(hexcolor: str) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in _rgb(hexcolor))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def lstar_from_luminance(y: float) -> float:
    """相対輝度 Y（0〜1）から CIE L*（0〜100）。"""
    return 116 * (y ** (1 / 3) if y > 216 / 24389 else (24389 / 27 * y + 16) / 116) - 16


def lstar(hexcolor: str) -> float:
    return lstar_from_luminance(luminance(hexcolor))


def mix(a: str, b: str, t: float) -> str:
    """a から b へ t（0〜1）だけ寄せた色。"""
    ra, rb = _rgb(a), _rgb(b)
    return "#" + "".join(
        f"{round(255 * (x + (y - x) * t)):02X}" for x, y in zip(ra, rb, strict=True)
    )


# ---------------------------------------------------------------- パレット


@dataclass(frozen=True)
class Palette:
    bg: str
    text: str
    accent: str
    doodle: str
    shadow: str

    @property
    def light(self) -> bool:
        """明るい地か（L* が 60 を超える）。"""
        return lstar(self.bg) > 60


MIN_CONTRAST = 4.5
DEFAULT_PALETTE = "navy"

# 色はパレットの中だけ。少しくすんだ調子（古着の写真と並べて落ち着くように）。
# 濃い地は白に近い文字、明るい地（cream・sky・paper）は濃い文字と線。
PALETTES = {
    "navy": Palette("#16335A", "#FFFFFF", "#F5D547", "#FFFFFF", "rgba(0,0,0,.35)"),
    "brown": Palette("#3B2A1F", "#FFF6E8", "#F2B84B", "#FFF6E8", "rgba(0,0,0,.38)"),
    "burgundy": Palette("#5C1A2E", "#FFF3EE", "#F7C85A", "#FFF3EE", "rgba(0,0,0,.38)"),
    "green": Palette("#1D4A38", "#FFFFFF", "#F5D547", "#FFFFFF", "rgba(0,0,0,.35)"),
    "mustard": Palette("#7B5C12", "#FFF4D8", "#FFDC66", "#FFF4D8", "rgba(30,15,0,.38)"),
    "olive": Palette("#4A4E24", "#F7F2DC", "#EBCB5E", "#F7F2DC", "rgba(10,12,0,.38)"),
    "charcoal": Palette("#2F3235", "#F4F2EE", "#EDC25C", "#F4F2EE", "rgba(0,0,0,.40)"),
    "terracotta": Palette("#A04528", "#FFF4EA", "#FFDA8A", "#FFF4EA", "rgba(40,10,0,.34)"),
    "cream": Palette("#F2E8D0", "#1F1A14", "#A93F1B", "#1F1A14", "rgba(60,40,20,.24)"),
    "sky": Palette("#BBD5E6", "#14283A", "#9A3517", "#14283A", "rgba(20,45,70,.26)"),
    # 旧名（以前のブリーフ用。cream と同じ系統の明るい地）
    "paper": Palette("#F3EEDF", "#1B1A18", "#C0391B", "#1B1A18", "rgba(40,30,20,.24)"),
}
# CRM と共通の 10 色（paper は旧名で数えない）
PALETTE_NAMES = (
    "navy",
    "brown",
    "burgundy",
    "green",
    "mustard",
    "olive",
    "charcoal",
    "cream",
    "sky",
    "terracotta",
)


def palette_family(name: str) -> tuple[str, ...]:
    """同じ明暗の系統（濃い地どうし・明るい地どうし）の色の名前。差し替えの候補の既定。"""
    light = PALETTES[name].light
    return tuple(n for n in PALETTE_NAMES if PALETTES[n].light == light)


# ---------------------------------------------------------------- 商品の明るさ

SINK_DELTA = 24.0  # 商品の明るさ（L* の中央値）と地の L* の差がこれ未満なら「沈む」
CONTRAST_FULL = 40.0  # 明暗の差（L*）がこれ以上なら満点（検査の点数）
TONE_BINS = 101
GLOW_MIX = 0.34  # 縁取りの色: 地から（濃い地は白へ・明るい地は黒へ）これだけ寄せる
GLOW_BLUR = (3, 9)  # ぼかしの半径（px）。内側を細く・外側を広く重ねる
TONE_L = {"dark": 12.0, "mid": 50.0, "light": 88.0}  # ブリーフの tone だけが手がかりのときの目安


def tone_hist(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """切り抜き（RGB とアルファ）から、アルファで重み付けした L*（0〜100）の分布（割合）。
    重みの合計が 0（透明）なら全部 0。"""
    c = rgb.astype(np.float64) / 255
    lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    ls = 116 * np.where(y > 216 / 24389, np.cbrt(y), (24389 / 27 * y + 16) / 116) - 16
    w = alpha.astype(np.float64) / 255
    idx = np.clip(np.rint(ls), 0, TONE_BINS - 1).astype(int)
    hist = np.bincount(idx.ravel(), weights=w.ravel(), minlength=TONE_BINS)
    total = hist.sum()
    return hist / total if total > 0 else hist


def hist_median(hist: np.ndarray) -> float | None:
    if hist is None or hist.sum() <= 0:
        return None
    cum = np.cumsum(hist)
    return float(np.searchsorted(cum, 0.5 * cum[-1]))


def item_lightness(hist: np.ndarray | None, hint: str | None = None) -> float | None:
    """商品の明るさ（L*）。画素から測れればそれ、無ければ tone の目安、どちらも無ければ None。"""
    m = hist_median(hist) if hist is not None else None
    if m is not None:
        return m
    return TONE_L.get(hint) if hint else None


def sink_gap(item_l: float, bg: str) -> float:
    """商品と地の明るさの差（L*、絶対値）。"""
    return abs(item_l - lstar(bg))


def sinks(item_l: float | None, bg: str) -> bool:
    return item_l is not None and sink_gap(item_l, bg) < SINK_DELTA


def contrast_score(
    lights: list[float | None], bg: str, glow: list[bool] | None = None
) -> float | None:
    """検査の「商品と地の明暗の差」（0〜1）。差が CONTRAST_FULL 以上で 1。縁取りで救った商品は
    最低でも 0.6 とみなす。測れた商品が無ければ None（点数に混ぜない）。"""
    vals = []
    for i, ll in enumerate(lights):
        if ll is None:
            continue
        v = min(1.0, sink_gap(ll, bg) / CONTRAST_FULL)
        if glow and glow[i]:
            v = max(v, 0.6)
        vals.append(v)
    return float(np.mean(vals)) if vals else None


def choose_palette(
    requested: str,
    lights: list[float | None],
    options: list[str] | None = None,
) -> tuple[str, str | None]:
    """(使う色, 替えた理由)。理由は替えなかったら None。

    候補 = ブリーフの palette_options（知っている名前だけ。無ければ同じ明暗の系統）。
    要求の色で沈む商品が無ければ替えない。沈む商品があるとき、候補のうち沈む商品がより少ない色
    （同数なら地との明暗差の合計が大きい色）があれば、その色に替える。"""
    known = [n for n in (options or []) if n in PALETTES]
    cands = known if known else list(palette_family(requested))
    if requested not in cands:
        cands = [requested, *cands]

    def sunk(n: str) -> int:
        return sum(1 for ll in lights if sinks(ll, PALETTES[n].bg))

    def gap_sum(n: str) -> float:
        return sum(sink_gap(ll, PALETTES[n].bg) for ll in lights if ll is not None)

    base = sunk(requested)
    if base == 0:
        return requested, None
    best = min(cands, key=lambda n: (sunk(n), -gap_sum(n), cands.index(n)))
    if sunk(best) >= base:
        return requested, None
    n_items = sum(1 for ll in lights if ll is not None)
    reason = (
        f"{requested} の地では {base} 点（{n_items} 点中）が沈むため {best} に替えた"
        f"（沈む商品 {sunk(best)} 点）"
    )
    return best, reason


def glow_color(p: Palette) -> str:
    """沈む商品の縁取りの色。濃い地は地より少し明るく、明るい地は少し暗く。"""
    return mix(p.bg, "#FFFFFF" if not p.light else "#000000", GLOW_MIX)
