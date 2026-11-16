"""着画（人が着ている商品写真）を探して切り抜き、広告のコラージュに1枚だけ入れる部品。

ブリーフの `items[].images`（その商品の全写真の URL）から、人が着ている写真を探す。判定は規則
（しきい値は下の名前付きの定数。本番の商品写真 92 枚で見た暫定値）:
  (a) 人が1人だけ  (b) 人の箱が画像の十分な割合  (c) 顔が写っていて、上半身〜腰以上が入っている
  (d) 人の確信度が十分
服だけを手で持っている・手だけの写真は、顔が無いので着画にしない（顔が写らない着画は見逃す）。
1商品につき最も良い1枚、案の中で最も良い1点を選ぶ（同点は案の中で先頭の商品）。

ここは検出の整理・選び方・取得のキャッシュ（モデルを使わない）と、それをつなぐ処理だけ。人の切り抜きは
`app/cutout.py` の `Cutter.cut_person`。**画素は描き直さない**。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# --- 着画の候補にする規則（暫定） ---
WORN_PERSON_MIN = 0.50  # (d) 人の確信度
WORN_OTHER_PERSON = 0.50  # (a) 最良の人のほかに、この確信度以上の人がいたら「1人だけ」ではない
WORN_BOX_AREA_MIN = 0.25  # (b) 人の箱が画像に占める割合
WORN_BOX_HEIGHT_MIN = 0.50  # (c) 人の箱の高さ ÷ 画像の高さ
WORN_BOX_ASPECT_MIN = 1.0  # (c) 人の箱の 高さ ÷ 幅（px）。上半身〜腰以上は縦長
WORN_FACE_MIN = 0.45  # 顔の確信度（顔が無い写真＝服だけ・手だけは着画にしない）
WORN_FACE_ASPECT_MIN = (
    1.0  # 顔の箱の 高さ ÷ 幅（正面・横顔は 1.2 以上）。端で切れた顔（0.8 以下）を除く
)
WORN_FACE_WIDTH = (0.08, 0.60)  # 顔の幅 ÷ 人の箱の幅（小さすぎ・大きすぎは顔でない）
WORN_FACE_UPPER = 0.60  # 顔の中心が人の箱の上からこの割合より上にある
WORN_FULL_BODY_BOTTOM = 0.85  # 箱の下端が画像の高さのここより下で、箱の高さが 0.7 以上なら「全身」
WORN_MAX_IMAGES = 10
WORN_CUT_TRIES = 3  # 切り抜きが判定に通らないとき、次の候補を試す回数


@dataclass
class WornCandidate:
    item_id: str
    image_index: int  # images の何枚目か
    url: str
    score: float  # 良さの点数（高いほど良い）
    box: list[float]  # 人の箱（px）
    person_score: float
    area_ratio: float  # 人の箱 ÷ 画像
    full_body: bool
    size: tuple[int, int] = (0, 0)
    cut_status: str = ""
    cut_path: Path | None = None
    reasons: list[str] = field(default_factory=list)


def judge(dets: list[dict], size: tuple[int, int]) -> dict | None:
    """検出（{kind, score, box}。px）から、着画の候補か。候補なら {score, box, person_score,
    area_ratio, full_body}、そうでなければ None。モデルを使わない。"""
    W, H = size
    people = sorted((d for d in dets if d["kind"] == "person"), key=lambda d: -d["score"])
    if not people or people[0]["score"] < WORN_PERSON_MIN:
        return None
    best = people[0]
    if any(p["score"] >= WORN_OTHER_PERSON for p in people[1:]):  # (a) 1人だけ
        return None
    x0, y0, x1, y1 = best["box"]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    area = bw * bh / (W * H)
    if area < WORN_BOX_AREA_MIN or bh / H < WORN_BOX_HEIGHT_MIN or bh / bw < WORN_BOX_ASPECT_MIN:
        return None  # (b)(c)
    face = None
    for d in dets:
        if d["kind"] != "face" or d["score"] < WORN_FACE_MIN:
            continue
        fx0, fy0, fx1, fy1 = d["box"]
        fw, fh = max(1.0, fx1 - fx0), max(1.0, fy1 - fy0)
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y0 + WORN_FACE_UPPER * bh):
            continue
        if (
            not WORN_FACE_WIDTH[0] <= fw / bw <= WORN_FACE_WIDTH[1]
            or fh / fw < WORN_FACE_ASPECT_MIN
        ):
            continue
        if face is None or d["score"] > face["score"]:
            face = d
    if face is None:
        return None  # 服だけ・手だけ・顔の無い写真
    full = y1 >= WORN_FULL_BODY_BOTTOM * H and bh / H >= 0.7
    score = (
        best["score"] + 0.3 * face["score"] + 0.3 * min(1.0, area / 0.5) + (0.15 if full else 0.0)
    )
    return {
        "score": round(score, 4),
        "box": [round(v, 1) for v in best["box"]],
        "person_score": round(best["score"], 3),
        "area_ratio": round(area, 3),
        "full_body": bool(full),
    }


def best_per_item(cands: list[WornCandidate]) -> dict[str, WornCandidate]:
    """商品ごとに最も良い1枚（同点は images の前のほう）。"""
    out: dict[str, WornCandidate] = {}
    for c in cands:
        cur = out.get(c.item_id)
        if cur is None or (c.score, -c.image_index) > (cur.score, -cur.image_index):
            out[c.item_id] = c
    return out


def pick_worn_item(per_item: dict[str, WornCandidate], order: list[str]) -> WornCandidate | None:
    """案の中で最も良い1点。点数が高い物（人が大きく写っている物ほど高い）。同点は案の中で先頭の商品。"""
    ranked = [(per_item[i], k) for k, i in enumerate(order) if i in per_item]
    if not ranked:
        return None
    return max(ranked, key=lambda t: (t[0].score, -t[1]))[0]


# ------------------------------------------------------------ 取得・検出・切り抜き（キャッシュ）


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:20]


def fetch_cached(url: str, cache_dir: Path, fetch: Callable[[str], bytes]) -> bytes:
    """写真の取得（URL ごとにキャッシュ。必要なときだけ取る）。"""
    p = cache_dir / f"img-{url_key(url)}.bin"
    if p.exists():
        return p.read_bytes()
    data = fetch(url)
    Image.open(io.BytesIO(data)).verify()
    p.write_bytes(data)
    return data


def detect_cached(data: bytes, cache_dir: Path, cutter_get: Callable) -> tuple[list[dict], tuple]:
    """人・顔の検出（写真の中身ごとにキャッシュ）。(検出, 画像の大きさ)。"""
    key = hashlib.sha1(data).hexdigest()[:20]
    p = cache_dir / f"worn-det-{key}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return j["dets"], tuple(j["size"])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    dets = cutter_get().detect_people(img)
    p.write_text(json.dumps({"size": list(img.size), "dets": dets}))
    return dets, img.size


_IMG_URL = re.compile(r"^https?://")


def normalize_images(item_image: str, images) -> list[str]:
    """ブリーフの images（URL の配列。最大10）。無い・おかしいときは空（探さない）。
    1枚目が image と違っても、そのまま使う。"""
    if not isinstance(images, list):
        return []
    out: list[str] = []
    for u in images:
        if isinstance(u, str) and _IMG_URL.match(u.strip()) and u.strip() not in out:
            out.append(u.strip())
    return out[:WORN_MAX_IMAGES]


@dataclass
class WornResult:
    pick: WornCandidate | None = None
    candidate_items: int = 0  # 着画の候補が見つかった商品の数
    candidate_photos: int = 0  # 候補になった写真の数
    searched_photos: int = 0
    notes: list[str] = field(default_factory=list)


def find_worn(
    items: list[tuple[str, list[str]]],
    cache_dir: Path,
    cutter_get: Callable,
    fetch: Callable[[str], bytes],
    log: Callable = print,
) -> tuple[WornResult, list[WornCandidate]]:
    """items=(商品の id, images)。候補を探し、案の中で最良の1点の候補を pick に入れる。
    切り抜きはここでは行わない（cut_pick で）。返す一覧は 商品ごとの最良の候補の、点数の高い順。"""
    res = WornResult()
    cands: list[WornCandidate] = []
    for item_id, urls in items:
        for idx, url in enumerate(urls):
            try:
                data = fetch_cached(url, cache_dir, fetch)
                dets, size = detect_cached(data, cache_dir, cutter_get)
            except Exception as e:  # 取れない・読めない写真は飛ばす
                res.notes.append(f"{item_id} の写真 {idx + 1} を使えなかった（{type(e).__name__}）")
                continue
            res.searched_photos += 1
            j = judge(dets, size)
            if j:
                cands.append(
                    WornCandidate(
                        item_id,
                        idx,
                        url,
                        j["score"],
                        j["box"],
                        j["person_score"],
                        j["area_ratio"],
                        j["full_body"],
                        size,
                    )
                )
    res.candidate_photos = len(cands)
    per = best_per_item(cands)
    res.candidate_items = len(per)
    order = [i for i, _ in items]
    ranked = sorted(
        per.values(), key=lambda c: (-c.score, order.index(c.item_id))
    )  # 同点は案の中で先頭の商品
    log(
        f"  着画の候補: {res.candidate_items} 商品 / {res.candidate_photos} 枚"
        f"（{res.searched_photos} 枚を調べた）"
    )
    return res, ranked


def cut_pick(
    ranked: list[WornCandidate],
    cache_dir: Path,
    cutter_get: Callable,
    fetch: Callable[[str], bytes],
    log: Callable = print,
) -> WornCandidate | None:
    """点数の高い順に人を切り抜き、判定が ok の最初の1枚を返す（切り抜きはキャッシュ）。
    check / fail の切り抜きは使わない（次の候補へ）。無ければ None。"""
    tried = 0
    for c in ranked:
        if tried >= WORN_CUT_TRIES:
            break
        tried += 1
        data = fetch_cached(c.url, cache_dir, fetch)
        key = hashlib.sha1(data).hexdigest()[:20]
        meta_p, png_p = cache_dir / f"worn-cut-{key}.json", cache_dir / f"worn-cut-{key}.png"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
        else:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            r = cutter_get().cut_person(img, c.box, c.person_score)
            meta = {"status": r.status, "reasons": r.reasons, "quality": r.summary()["quality"]}
            if r.rgba is not None:
                r.rgba.save(png_p)
            meta_p.write_text(json.dumps(meta, ensure_ascii=False))
        c.cut_status, c.reasons = meta["status"], meta["reasons"]
        log(f"  着画の切り抜き {c.item_id}: {c.cut_status} {'; '.join(c.reasons)}")
        if meta["status"] == "ok" and png_p.exists():
            c.cut_path = png_p
            return c
    return None
