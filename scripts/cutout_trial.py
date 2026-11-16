"""商品写真の切り抜きの試し。画像の一覧を受け取り、data/cutout/ に結果と一覧の HTML を書く。

  # 対象の一覧（JSON: [{"id", "url" か "path", "product_type"(任意), "title"(任意)}]）を指定する
  uv run --group clip --group clef-local python -m scripts.cutout_trial --targets targets.json
  # item_subtype の判定の結果から、品目がばらけるように N 点を選ぶ
  # （選んだ一覧は data/cutout/targets.json）
  uv run --group clip --group clef-local python -m scripts.cutout_trial \\
      --predictions data/item-subtype/predictions.json --pick 20
  # 画像の URL かパスを直接
  ... python -m scripts.cutout_trial https://.../a.jpg ./b.png

出力: data/cutout/<id>.png（切り抜き）・<id>_src.jpg（元の写真）・report.json・
index.html（1ファイル・画像は相対パス・外部の読み込みなし）。判定が fail の写真は切り抜かず、写真のままにする。
モデルは手元の ~/models をオフラインで読む（app/cutout.py）。商品の画素は描き直さない。
"""

import argparse
import html
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx2 as httpx
from PIL import Image

from app import cutout

DATA = Path("data/cutout")
WIDTH = 1200  # Shopify の CDN に頼む幅
STATUS_NAMES = {"ok": "ok", "check": "check（人が見る）", "fail": "fail（写真のまま使う）"}


def pick_targets(predictions: dict, n: int) -> list[dict]:
    """品目がばらけるように n 点を選ぶ。品目ごとに、細分類もばらけるよう順に取る（決まった順）。"""
    groups: dict[str, dict[str, list[tuple[str, dict]]]] = defaultdict(lambda: defaultdict(list))
    for gid, p in sorted(predictions.items()):
        if not p.get("images"):
            continue
        sub = ((p.get("photo") or {}).get("answers") or {}).get("subtype", {}).get("choice", "")
        groups[p["productType"]][sub].append((gid, p))
    # 品目の中で細分類を順に回した並び
    per_type: dict[str, list[tuple[str, dict]]] = {}
    for t, subs in groups.items():
        order, queues = [], [list(v) for _, v in sorted(subs.items())]
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop(0))
        per_type[t] = order
    # 品目を順に回して n 点
    out, types = [], sorted(per_type)
    while len(out) < n and any(per_type[t] for t in types):
        for t in types:
            if per_type[t] and len(out) < n:
                out.append(per_type[t].pop(0))
    return [
        {
            "id": gid.rsplit("/", 1)[-1],
            "url": p["images"][0],
            "product_type": p["productType"],
            "title": p.get("title", ""),
        }
        for gid, p in out
    ]


def load_image(target: dict, http: httpx.Client) -> Image.Image:
    if target.get("path"):
        return Image.open(target["path"]).convert("RGB")
    url = target["url"]
    sep = "&" if "?" in url else "?"
    res = http.get(f"{url}{sep}width={WIDTH}")
    res.raise_for_status()
    return Image.open(io.BytesIO(res.content)).convert("RGB")


def page(rows: list[dict]) -> str:
    """一覧の HTML（1ファイル・相対パス・外部の読み込みなし）。"""
    e = html.escape
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_NAMES}
    cards = []
    for r in rows:
        cut = f'<img src="{e(r["cutout"])}" alt="">' if r.get("cutout") else "<em>切り抜かない</em>"
        fallback = f'<img src="{e(r["source"])}" alt="">' if not r.get("cutout") else cut
        reasons = "".join(f"<li>{e(x)}</li>" for x in r["reasons"]) or "<li>問題なし</li>"
        q = r.get("quality") or {}
        meta = (
            f"面積 {q['area_ratio']:.0%} ・確信度 {q['box_score']:.2f} ・塊 {q['components']}"
            if q
            else ""
        )
        cards.append(
            f'<article class="{e(r["status"])}">'
            f"<header><b>{e(r['id'])}</b> {e(r.get('product_type', ''))} "
            f'<span class="tag {e(r["status"])}">{e(STATUS_NAMES[r["status"]])}</span></header>'
            f'<div class="row">'
            f'<figure><img src="{e(r["source"])}" alt=""><figcaption>元の写真</figcaption></figure>'
            f'<figure class="bg light">{fallback}<figcaption>明るい背景</figcaption></figure>'
            f'<figure class="bg dark">{fallback}<figcaption>暗い背景</figcaption></figure>'
            f"</div><ul>{reasons}</ul><small>{e(meta)} ・{r['seconds']:.1f} 秒</small></article>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>切り抜きの試し</title>
<style>{STYLE}</style></head><body>
<h1>切り抜きの試し（{len(rows)}点）</h1>
<p>ok {counts["ok"]} ・ check {counts["check"]} ・ fail {counts["fail"]}</p>
{"".join(cards)}
</body></html>
"""


STYLE = """
body{font-family:system-ui,sans-serif;margin:12px;color:#222;background:#f4f4f2}
h1{font-size:18px}
article{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0}
article.check{border-color:#e0a030} article.fail{border-color:#d04040}
header{font-size:14px;margin-bottom:6px}
.tag{font-size:12px;padding:1px 8px;border-radius:10px;margin-left:6px;color:#fff;
  background:#2a8a4a}
.tag.check{background:#c78000} .tag.fail{background:#c03030}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
figure{margin:0;text-align:center} figcaption{font-size:11px;color:#777;margin-top:2px}
figure img{width:100%;aspect-ratio:3/4;object-fit:contain;display:block}
.bg{border-radius:6px;padding:6px;box-sizing:border-box}
.bg.light{background:linear-gradient(#f3efe8,#e6e0d6)}
.bg.dark{background:linear-gradient(#2a2d33,#14161a)}
.bg img{filter:drop-shadow(0 10px 8px rgba(0,0,0,.35))}
.bg em{display:block;padding:40% 0;color:#999;font-size:12px}
ul{margin:6px 0;padding-left:20px;font-size:13px} small{color:#888}
@media(max-width:560px){.row{grid-template-columns:1fr 1fr}.row figure:first-child{grid-column:1/3}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="画像の URL かパス")
    parser.add_argument("--targets", type=Path, help="対象の一覧の JSON")
    parser.add_argument("--predictions", type=Path, help="item_subtype の判定の結果の JSON")
    parser.add_argument("--pick", type=int, default=20, help="--predictions から選ぶ点数")
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()

    targets = []
    if args.predictions:
        targets = pick_targets(json.loads(args.predictions.read_text()), args.pick)
    if args.targets:
        targets += json.loads(args.targets.read_text())
    for i, s in enumerate(args.sources, 1):
        targets.append({"id": f"s{i:02d}", **({"url": s} if s.startswith("http") else {"path": s})})
    if not targets:
        parser.error("対象がありません")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "targets.json").write_text(json.dumps(targets, ensure_ascii=False, indent=1))

    http = httpx.Client(timeout=60, follow_redirects=True)
    t0 = time.time()
    cutter = cutout.Cutter()
    print(f"モデルの読み込み {time.time() - t0:.1f} 秒（{cutter.device}）", flush=True)

    rows = []
    for t in targets:
        src = args.out / f"{t['id']}_src.jpg"
        image = load_image(t, http)
        image.save(src, quality=92)
        res = cutter.cut(image, t.get("product_type"))
        row = {
            "id": t["id"],
            "product_type": t.get("product_type", ""),
            "source": src.name,
            **res.summary(),
        }
        if res.rgba is not None:
            out = args.out / f"{t['id']}.png"
            res.rgba.save(out)
            row["cutout"] = out.name
        rows.append(row)
        print(f"{t['id']} {res.status} {res.seconds:.1f}s {'; '.join(res.reasons)}", flush=True)

    times = [r["seconds"] for r in rows]
    summary = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_NAMES}
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "mean_seconds": round(sum(times) / len(times), 2),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    (args.out / "index.html").write_text(page(rows))
    print(summary, f"平均 {sum(times) / len(times):.2f} 秒/点")
    print((args.out / "index.html").resolve())


if __name__ == "__main__":
    main()
