"""drafts.json から、店主が確かめる一覧ページ（page/index.html と写真）を作る。"""

import html
import json
import os
import re
import subprocess

os.makedirs("page/img", exist_ok=True)
D = json.load(open("drafts.json"))


def thumb(r):
    if not r.get("image"):
        return ""
    f = f"page/img/{r['id']}.jpg"
    if not os.path.exists(f):
        u = r["image"]
        subprocess.run(["curl", "-s", "-o", f, u + ("&" if "?" in u else "?") + "width=220"])
    return f'<img src="img/{r["id"]}.jpg" alt="" loading="lazy">'


def final_html(r):
    return r["html"] + ("\n" + r["keep_html"] if r.get("keep_html") else "")


todo = [r for r in D if r["mode"] in ("new", "template")]
ok = [r for r in todo if not r["problems"]]
warn = [r for r in todo if r["problems"]]
skip = [r for r in D if r["mode"] == "skip"]


def card(r):
    old = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r["old_html"])).strip()
    facts = r.get("facts") or {}
    meta = " ／ ".join(
        f"{k}: {v}"
        for k, v in (
            ("素材", facts.get("素材")),
            ("写真の色", (r.get("look") or {}).get("base_color")),
            ("柄", (r.get("look") or {}).get("pattern")),
        )
        if v
    )
    return f"""<article id="p{r["id"]}"><div class="ph">{thumb(r)}</div><div class="body">
<h3>{html.escape(r["title"] or "")} <a href="https://cms.85-store.com/admin/collections/products/{r["id"]}">CMS</a></h3>
<p class="meta">{html.escape(meta)}{" ／ 定型文を置き換え" if r["mode"] == "template" else ""}</p>
{"".join(f'<p class="warn">⚠ {html.escape(p)}</p>' for p in r["problems"])}
<div class="new">{final_html(r)}</div>
<p class="old">今の説明: {html.escape(old) if old else "（空）"}</p></div></article>"""


groups = {}
for r in skip:
    groups.setdefault(r["reason"].split(":")[0], []).append(r)
page = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>トップスの商品説明の案（{len(todo)} 商品）</title><style>
body{{font-family:system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.7;color:#1c1c1c;background:#fafaf8;margin:0}}
main{{max-width:900px;margin:0 auto;padding:24px 16px 80px}}h1{{font-size:1.4rem;margin:0}}h2{{font-size:1.1rem;margin:36px 0 10px;border-bottom:1px solid #ddd;padding-bottom:4px}}
article{{display:flex;gap:14px;background:#fff;border:1px solid #e2e2dc;border-radius:8px;padding:12px;margin:10px 0}}
.ph{{flex:0 0 110px}}.ph img{{width:110px;border-radius:4px;display:block}}.body{{min-width:0;flex:1}}
h3{{font-size:.95rem;margin:0 0 2px}}h3 a{{font-size:.75rem;font-weight:400;margin-left:6px}}.meta{{font-size:.78rem;color:#777;margin:0 0 6px}}
.new{{border-left:3px solid #6a8f6a;padding-left:10px}}.new p{{margin:.3em 0}}.new ul{{margin:.3em 0;padding-left:1.3em}}
.old{{font-size:.8rem;color:#888;margin:8px 0 0}}.warn{{color:#b3541e;font-size:.85rem;margin:2px 0}}
.lead{{background:#fff;border:1px solid #e2e2dc;border-radius:8px;padding:12px 16px}}li{{margin:.15em 0}}
@media (prefers-color-scheme:dark){{body{{background:#181816;color:#e8e8e2}}article,.lead{{background:#22221f;border-color:#444}}.meta,.old{{color:#aaa}}h2{{border-color:#444}}}}
</style></head><body><main>
<h1>トップスの商品説明の案</h1>
<p class="meta">手元のモデル（gemma4）が、CMS の項目と正面の写真から書いた案です。まだどの商品にも書き込んでいません。</p>
<div class="lead"><ul>
<li>入れる案: <strong>{len(todo)} 商品</strong>（そのまま入れられる {len(ok)}、自動の確認に掛かった {len(warn)}）</li>
<li>入れない: {len(skip)} 商品（{"、".join(f"{k} {len(v)}" for k, v in groups.items())}）</li>
<li>緑の線の中が、ショップに出る説明文です。今の説明にあるリンク・採寸の行は、うしろに残しています。</li>
<li>直したい商品は、商品名（か番号）と直し方を教えてください。全体の調子の直しも受けます。</li></ul></div>
<h2>自動の確認に掛かった案（{len(warn)}）</h2>{"".join(card(r) for r in warn) or "<p>ありません。</p>"}
<h2>入れる案（{len(ok)}）</h2>{"".join(card(r) for r in ok)}
<h2>入れない商品（{len(skip)}）</h2>
{"".join(f"<h3>{html.escape(k)}（{len(v)}）</h3><ul>" + "".join(f"<li>{html.escape(r['title'] or '')}{'：' + html.escape(r['reason'].split(':', 1)[1].strip()) if ':' in r['reason'] else ''}</li>" for r in v) + "</ul>" for k, v in groups.items())}
</main></body></html>"""
open("page/index.html", "w").write(page)
print(len(todo), len(ok), len(warn), len(skip), {k: len(v) for k, v in groups.items()})
