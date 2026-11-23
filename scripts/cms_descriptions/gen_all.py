"""対象の全商品の案を作る（書き込まない）。drafts.json に残す。"""

import json
import re
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import describe as d

KEEP = re.compile(
    r"ブログ|身幅|着丈|肩幅|袖丈|裄丈|実寸|Size\s*[:：]|サイズ\s*[:：]|表記サイズ|model\s*[:：]|身長|\d\s*cm|素材表記|material\s*[:：]",
    re.I,
)
TEMPLATE = re.compile(
    r"素材を使用した一着です|雰囲気の一着です|^[^。]{0,40}の(半袖|長袖)?[^。]{0,30}(シャツ|スウェット|ニット|カーディガン|Tシャツ)。$"
)


def split_old(html):
    """今の説明を、残す行（リンク・採寸だけの短い行）と、文章に分ける。"""
    html = re.sub(r"<meta[^>]*>", "", html or "")
    blocks = re.findall(
        r"<(?:p|ul|ol|h\d|div|table)[^>]*>.*?</(?:p|ul|ol|h\d|div|table)>", html, re.S
    ) or ([html] if html.strip() else [])
    keep, prose = [], []
    for b in blocks:
        t = d.text_of(b)
        if not t:
            continue
        (keep if len(t) < 60 and KEEP.search(t) else prose).append(b)
    return keep, prose


prods = json.load(open("products.json"))
import os

out = (
    json.load(open("drafts.json"))
    if os.path.exists("drafts.json") and "--resume" in sys.argv
    else []
)
done = {r["id"] for r in out}
fresh = {}
for p in prods:
    if p["id"] in done:
        continue
    try:
        p = d.get(f"{d.CMS}/api/products/{p['id']}?depth=1")
    except Exception as e:
        print("読めない", p["id"], e)
        continue
    keep, prose = split_old(p.get("descriptionHtml"))
    core = " ".join(d.text_of(b) for b in prose)
    r = {
        "id": p["id"],
        "title": p.get("title"),
        "status": p.get("status"),
        "old_html": p.get("descriptionHtml") or "",
        "core": core,
        "keep_html": "\n".join(keep),
    }
    pick = next(
        (
            i
            for i in p.get("images") or []
            if i.get("autoAltLabel") == "正面" and i.get("shopifyUrl")
        ),
        None,
    ) or next((i for i in p.get("images") or [] if i.get("shopifyUrl")), None)
    r["image"] = pick["shopifyUrl"] if pick else None
    conflict = d.material_conflict(p)
    if p.get("status") != "active":
        r["mode"] = "skip"
        r["reason"] = "売り場に出ていない"
    elif conflict:
        r["mode"] = "skip"
        r["reason"] = "素材を確かめてください: " + conflict
    elif len(core) >= 30 and not (len(core) <= 90 and TEMPLATE.search(core)):
        r["mode"] = "skip"
        r["reason"] = "店が書いた説明がある"
    elif not pick:
        r["mode"] = "skip"
        r["reason"] = "写真が無い"
    else:
        r["mode"] = "template" if core else "new"
        if not core:
            p = {**p, "descriptionHtml": ""}
        for _attempt in range(3):
            try:
                g = d.describe(p)
                break
            except Exception as e:
                print("やり直し", p["id"], e, flush=True)
                time.sleep(20)
        else:
            print("作れない", p["id"])
            continue
        r.update(
            html=g["html"],
            look=g["look"],
            facts=g["facts"],
            problems=g["problems"],
            chars=g["chars"],
        )
        print(
            f"{r['id']} {r['title'][:40]} {r['chars']}字 {'⚠ ' + '、'.join(r['problems']) if r['problems'] else ''}",
            flush=True,
        )
    out.append(r)
    json.dump(out, open("drafts.json", "w"), ensure_ascii=False, indent=1)
import collections

print(
    collections.Counter(
        r["mode"] + (":" + r.get("reason", "").split(":")[0] if r["mode"] == "skip" else "")
        for r in out
    )
)
print("確認に掛かった", sum(bool(r.get("problems")) for r in out))
