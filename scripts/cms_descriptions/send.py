"""確かめた案を CMS の受け口に送る。send.py <開始> <件数>。送った結果は sent.json に残す。"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import describe as d

D = [r for r in json.load(open("drafts.json")) if r["mode"] != "skip" and not r["problems"]]
sent = json.load(open("sent.json")) if os.path.exists("sent.json") else {}
start, count = int(sys.argv[1]), int(sys.argv[2])
batch = [r for r in D if str(r["id"]) not in sent][start : start + count]
import re


def clean_keep(h):
    """残す行（リンク・採寸）から、使えないタグを外す（太字などは中身を残し、画像は消す）。"""
    h = re.sub(r"<img\b[^>]*>", "", h or "")
    h = re.sub(r"</?(?!(?:p|ul|ol|li|br|a)\b)[a-zA-Z][^>]*>", "", h)
    h = re.sub(r"<(p|ul|ol|li)\b[^>]*>", r"<\1>", h)
    h = re.sub(r'<a\b[^>]*?href="(https?://[^"\s]+)"[^>]*>', r'<a href="\1">', h)
    h = re.sub(r"<br\b[^>]*>", "<br>", h)
    return "\n".join(
        b for b in re.findall(r"<(?:p|ul|ol)>.*?</(?:p|ul|ol)>", h, re.S) if d.text_of(b)
    )


items = [
    {
        "id": r["id"],
        "html": r["html"]
        + ("\n" + clean_keep(r["keep_html"]) if clean_keep(r.get("keep_html")) else ""),
        "overwrite": r["mode"] == "template",
    }
    for r in batch
]
req = urllib.request.Request(
    f"{d.CMS}/api/products/descriptions",
    method="POST",
    data=json.dumps({"items": items}).encode(),
    headers={"Content-Type": "application/json"},
)
res = json.load(urllib.request.urlopen(req, timeout=600))
skipped = {s["ref"]: s["reason"] for s in res["skipped"]}
for r in batch:
    sent[str(r["id"])] = skipped.get(r["id"], "ok")
    print(
        ("済 " if r["id"] not in skipped else "見送り ")
        + f"{r['id']} {r['title'][:44]}"
        + (f"：{skipped[r['id']]}" if r["id"] in skipped else "")
    )
json.dump(sent, open("sent.json", "w"), ensure_ascii=False, indent=1)
print(
    "入れた",
    res["updated"],
    "/ 見送り",
    len(res["skipped"]),
    "/ 残り",
    len([r for r in D if str(r["id"]) not in sent]),
)
