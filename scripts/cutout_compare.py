"""切り抜きの方式の比較（SAM2 だけ / SAM2 + BiRefNet）。1商品1行の HTML を data/cutout-compare/ へ。

  python -m scripts.cutout_compare --targets data/cutout/targets.json --src-dir data/cutout \\
      --extra-dir <元の写真のフォルダ>:<品目> ...

対象: --src-dir の `<id>_src.jpg`（品目は --targets の product_type）と、--extra-dir の `<id>.jpg`。
行: 元の写真 / 今の切り抜き / 新しい切り抜き（暗い地）/ 襟元の拡大 / 判定。
DINO の箱と SAM2 のマスクは1回だけ取り、同じものから両方の方式で仕上げる。"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app import cutout

OUT = Path("data/cutout-compare")
DARK = (38, 38, 42)
THUMB = 420


def on_dark(rgba: Image.Image, height: int = THUMB) -> Image.Image:
    bg = Image.new("RGB", rgba.size, DARK)
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg.resize((max(1, round(bg.width * height / bg.height)), height), Image.LANCZOS)


def neck(rgba: Image.Image, size: int = 300) -> Image.Image:
    """切り抜きの上の部分（襟・肩）を拡大した小さな切り出し（暗い地）。"""
    a = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(a > 8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w = x1 - x0
    box = (x0 + w // 4, y0, x0 + w * 3 // 4, y0 + max(1, (y1 - y0) // 3))
    crop = rgba.crop(box)
    bg = Image.new("RGB", crop.size, DARK)
    bg.paste(crop, mask=crop.getchannel("A"))
    scale = size / max(bg.size)
    return bg.resize(
        (max(1, round(bg.width * scale)), max(1, round(bg.height * scale))), Image.LANCZOS
    )


def collect(args) -> list[dict]:
    types = {}
    if args.targets and Path(args.targets).exists():
        for t in json.loads(Path(args.targets).read_text()):
            types[t["id"]] = t.get("product_type")
    items = []
    for p in sorted(Path(args.src_dir).glob("*_src.jpg")):
        pid = p.name.removesuffix("_src.jpg")
        items.append({"id": pid, "path": p, "type": types.get(pid), "group": "cutout"})
    for spec in args.extra_dir or []:
        d, _, kind = spec.rpartition(":")
        for p in sorted(Path(d).glob("*.jpg")):
            same = any(i["id"] == p.stem for i in items)  # 同じ商品が両方にあれば名前を分ける
            items.append(
                {
                    "id": p.stem + ("_crm" if same else ""),
                    "path": p,
                    "type": types.get(p.stem) or kind or None,
                    "group": "crm",
                }
            )
    return items


def row(r: dict) -> str:
    def cell(res: dict) -> str:
        reasons = "".join(f"<li>{html.escape(x)}</li>" for x in res["reasons"])
        q = res["quality"] or {}
        return (
            f'<div class="g {res["status"]}">{res["status"]}</div>'
            f'<div class="q">縁の色差 {q.get("edge_delta")} / ギザギザ {q.get("roughness")}</div>'
            f"<ul>{reasons}</ul>"
        )

    imgs = "".join(
        f'<td><img src="{r["id"]}_{k}.jpg"></td>'
        if k != "src"
        else f'<td><img src="{r["id"]}_src.jpg"></td>'
        for k in ("src", "old", "new", "neck_old", "neck_new")
    )
    return (
        f"<tr><th>{r['id']}<br><small>{html.escape(r['type'] or '')}</small></th>{imgs}"
        f"<td>今: {cell(r['old'])}</td><td>新: {cell(r['new'])}</td></tr>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets")
    ap.add_argument("--src-dir", default="data/cutout")
    ap.add_argument("--extra-dir", action="append", help="<フォルダ>:<品目>")
    ap.add_argument("--only", help="id をカンマで区切って絞る")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cutter = cutout.Cutter()
    if not cutter.load_birefnet():
        raise SystemExit(f"BiRefNet を読めない: {cutter.birefnet_error}")
    rows = []
    for it in collect(args):
        if args.only and it["id"] not in args.only.split(","):
            continue
        image = Image.open(it["path"]).convert("RGB")
        picked = cutter.detect(image, it["type"])
        if picked is None:
            print(it["id"], "検出なし")
            continue
        box, score = picked
        mask = cutter.segment(image, box)
        fg = cutter.foreground(image, box)
        old = cutout.finish(image, mask, score)
        new = cutout.finish_alpha(image, mask, fg, score)
        row_ = {"id": it["id"], "type": it["type"], "old": old.summary(), "new": new.summary()}
        # fail は写真のまま（切り抜き無し）なので、写真を暗い地に出す
        for key, res in (("old", old), ("new", new)):
            rgba = res.rgba if res.rgba is not None else image.convert("RGBA")
            on_dark(rgba).save(OUT / f"{it['id']}_{key}.jpg", quality=88)
            neck(rgba).save(OUT / f"{it['id']}_neck_{key}.jpg", quality=88)
            if res.rgba is not None:
                res.rgba.save(OUT / f"{it['id']}_{key}.png")
        image.resize((round(image.width * THUMB / image.height), THUMB)).save(
            OUT / f"{it['id']}_src.jpg", quality=88
        )
        print(it["id"], it["group"], old.status, "->", new.status, f"{new.seconds:.2f}s")
        row_["group"] = it["group"]
        rows.append(row_)
    (OUT / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    head = (
        "<meta charset=utf-8><style>body{font:13px sans-serif;background:#eee}"
        "table{border-collapse:collapse}"
        "td,th{padding:4px;vertical-align:top;border-bottom:1px solid #bbb}img{display:block}"
        ".g{font-weight:bold;padding:1px 6px;display:inline-block}"
        ".ok{background:#8c8}.check{background:#fc6}"
        ".fail{background:#f88}.q{color:#555}ul{margin:2px;padding-left:16px;max-width:240px}</style>"
        "<h3>切り抜きの比較: 元 / 今（SAM2）/ 新（SAM2+BiRefNet）/ 襟元 今・新</h3><table>"
    )
    (OUT / "index.html").write_text(head + "".join(row(r) for r in rows) + "</table>")
    print("->", OUT / "index.html")


if __name__ == "__main__":
    main()
