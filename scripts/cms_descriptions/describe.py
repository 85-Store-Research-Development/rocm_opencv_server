"""商品の短い説明文を、手元のモデル（Ollama の gemma4）で作る。

1. CMS の項目（確定の情報）を読む  2. 正面の写真から、色・柄・見える作りを読む（JSON）
3. 2 つを渡して、短い説明文（HTML）を書かせる  4. 禁止の言葉・長さ・色柄素材が入っているかを確かめる
書き込みはしない（--out の JSON に残すだけ）。
"""

import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# 作業用のフォルダ（CSV・taxonomy.ts の写し・案の JSON を置く）。既定は今いるフォルダ
HERE = Path(os.getenv("DESCRIPTIONS_DIR") or Path.cwd())
CMS = "https://cms.85-store.com"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:26b"

tax = (HERE / "taxonomy.ts").read_text()
NAMES = dict(re.findall(r"\['(\d+)', '([^']+)'\]", tax))


def get(url, raw=False):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "describe"}), timeout=120
    ) as r:
        data = r.read()
    return data if raw else json.loads(data)


def chat(system, user, images=(), temperature=0.3, as_json=False):
    body = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"temperature": temperature, "repeat_penalty": 1.1, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user, **({"images": list(images)} if images else {})},
        ],
    }
    if as_json:
        body["format"] = "json"
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"].strip()


LOOK_SYSTEM = """あなたは古着店の検品係です。写真に写っている服を見て、はっきり見えることだけを JSON で答えます。
少しでも迷うものは「分からない」を選びます（当て推量をしない。間違いは店の信用を落とします）。
素材・年代・ブランドは答えません（写真では決められないため）。背景の壁・ハンガー・カラーカード・マーカーは無視します。
出力は次のキーだけの JSON（選択肢があるものは、その中から 1 つ）:
{"base_color": "服の地の色（いちばん広い色）を日本語で 1 つ。例: ネイビー、オフホワイト、ブラウン、チャコールグレー",
 "accent_colors": ["柄・プリント・切り替えの色を 0〜2 個。無ければ空"],
 "pattern": "無地 / チェック / ストライプ / 花柄 / ペイズリー / 幾何学柄 / 総柄 / プリント / 刺繍 / ボーダー / 迷彩 / その他 から 1 つ。チェックやストライプは、細かさが分かれば「細かいチェック」のように添える",
 "graphic": "プリント・刺繍・ロゴがあるときだけ、位置と内容を短く（例: 胸に筆記体の刺繍、前面に大きなスキーヤーのプリント）。読める文字はそのまま書く。無ければ null",
 "collar": "レギュラーカラー / ボタンダウン / オープンカラー / スタンドカラー / クルーネック / Vネック / モックネック / フード / 襟なし / 分からない",
 "front": "ボタン / スナップボタン / ジッパー / 前開きなし / 分からない",
 "sleeve": "長袖 / 半袖 / 袖なし / 分からない",
 "chest_pockets": "なし / 1 つ / 2 つ / 分からない",
 "pocket_flap": "あり / なし / 分からない",
 "cuffs": "リブ / ボタン留め / そのまま / 分からない",
 "hem": "リブ / まっすぐ / ラウンド / 分からない"}
ボタンダウンは、襟の先がボタンで留まっているのがはっきり見えるときだけ選びます。"""

WRITE_SYSTEM = """あなたは、落ち着いた空気感のセレクト古着店で、商品の説明を書く係です。
この店が扱うのは、日常で着やすい、状態と作りのよいレギュラー古着です。新品の洋服と合わせて着られることを大事にしています。
渡されたデータだけをもとに、商品の事実を静かに伝える、短い商品説明文を書きます。

# 姿勢
- 売り込みや購買の催促はしません。色・柄・素材・作りという事実を誠実に言葉にし、読み手が自分で選べる余白を残します。
- 「古いこと・珍しいこと・ブランドの格」を価値として語りません。いまの服と同じように、素直に着られる一枚の服として書きます。

# 厳守すること
1. 短く書く。全体で 100〜200 字。前置き・まとめ・あいさつは書かない。
2. 種別は「品目」と「商品名」に従う（Knit / Sweater はニット、Sweat はスウェット、Cardigan はカーディガン。ニットをスウェットと書かない）。
   最初の文に、アイテムの種別・色・柄・素材を必ず入れる（検索する人が探す情報）。色と柄は「写真から読んだこと」を使う。素材は「確定の情報」の「素材」の表記をそのまま使う（「コットン 100%」など。割合を変えない）。素材が渡されていなければ、素材には触れない（写真から推測しない）。
3. 事実だけを書く。
   - 情報の信頼度は【確定の情報 ＞ 写真から読んだこと】。食い違うときは確定の情報を使う。
   - 渡されていない年代・生産国・ブランドの歴史・モデル名の由来は書かない。知っているつもりのことも書かない。
   - 寸法の数字・値段・在庫には触れない（別の欄に出る）。
4. <li> は「写真から読んだこと」と「確定の情報」にある作りだけを、1〜2 個書く。「分からない」の項目・渡されていない作り（スリット、裏地、ステッチ、形など）は書かない。同じ内容を言い換えて並べない。
5. 最後の <p> に、合わせ方を 1 文だけ書く。
   - この服の色・柄・襟の形から言えることにする。新品の服や手持ちの服と自然に合うことを、具体的な相手（無地のスラックス、デニム、ニット、ジャケットの中、白い T シャツの上 など）を 1〜2 個挙げて伝える。
   - 断定・指示はしない。「〜とも合わせやすい色です。」「〜の上に羽織っても収まります。」「〜と合わせると落ち着いたバランスになります。」のように、読み手に委ねる。
   - 柄や色が強い服は、無地のものと合わせる方向で書く。無地の服は、合わせる相手を選ばないことを書く。
   - その服自身と同じ種類の服を相手にしない（T シャツに「T シャツの上に」、スウェットに「スウェットの上に」と書かない）。T シャツは、シャツやジャケットの中に着ること、パンツとの組み合わせで書く。
   - 文の形を毎回変える。「落ち着いた〜なので」で始めない。「自然に馴染みます」「自然なバランス」は使わない。デニムやチノに「無地の」は付けない。
   - 言い方の例（そのまま写さず、服に合わせて選び、言い換える）:
     「白い T シャツの上に羽織ると、柄がすっきり見えます。」
     「グレーのスラックスやネイビーのニットと、色がぶつからずに合います。」
     「ジャケットの中に着ても、襟元が収まります。」
     「新しく買ったスラックスにも、手持ちのデニムにも合わせられる色です。」
     「上に重ねるものを選ばない、無地のグレーです。」
6. 状態は「状態メモ」が渡されたときだけ、合わせ方の文の前に 1 文で客観的に書く（美化しない）。渡されていなければ、状態・新品かどうかには一切触れない。
7. 感嘆符・疑問符（！ ？ ! ?）は使わない。文はすべて「。」で終える。
8. 次の言葉と、似た言い回しは使わない:
   - 煽り: 激レア、入手困難、希少、レア、早い者勝ち、必見、おすすめ、マスト、一生モノ、絶対、相当、お買い得、ぜひ、見逃せない、大人気、人気、定番、極上、最高、魅力、存在感、主役、こなれ、旬、今季、トレンド、抜け感、一着、逸品、重宝、万能、着回し。
   - 古さ・格を売りにする言葉: ヴィンテージ、ビンテージ、オールド、年代、当時、名作、名品、アーカイブ、歴史、伝統、老舗、本格、通好み、マニア、味、風格、雰囲気のある。

# 出力
HTML の <p> と <ul><li> だけを使う。見出し・太字・リンク・絵文字は使わない。次の形:
<p>種別・色・柄・素材を入れた 1〜2 文。</p>
<ul><li>作り（1〜2 個。1 個は 30 字以内）</li></ul>
<p>（状態メモがあるときだけ状態の 1 文。）合わせ方の 1 文。</p>

# 例 1（柄もの・素材あり・状態メモあり）
<p>ネイビーにグリーンのチェック柄が入った、コットン 100% の長袖シャツです。洗いを重ねた生地は柔らかく、肌に沿います。</p>
<ul><li>襟はボタンダウンで、左胸にポケットが 1 つ付きます。</li></ul>
<p>袖口に薄い擦れがあります。無地のスラックスやデニムと合わせると、柄が落ち着いて収まります。</p>

# 例 2（無地にプリント・素材なし・状態メモなし）
<p>グレーの地に、胸元へ白のロゴプリントが入ったクルーネックのスウェットです。</p>
<ul><li>袖口と裾はリブ仕様です。</li></ul>
<p>色を選ばないグレーなので、シャツの上に重ねても、コートの中に着ても収まります。</p>

# 例 3（無地・素材あり・状態メモなし）
<p>ネイビーの無地の、ポリエステル 65%、コットン 35% の長袖シャツです。張りのある生地で、形が崩れにくい作りです。</p>
<ul><li>レギュラーカラーで、胸にフラップ付きのポケットが 2 つ付きます。</li></ul>
<p>無地のネイビーは、手持ちのスラックスやニットとも色がぶつからずに合わせられます。</p>"""

BANNED = (
    "激レア 入手困難 希少 レア 早い者勝ち 必見 おすすめ オススメ マスト 一生モノ 絶対 お買い得 ぜひ 是非 見逃せない 大人気 人気 定番 極上 最高 魅力 存在感 主役 こなれ 旬 今季 トレンド 抜け感 一着 逸品 円相当 重宝 万能 着回し "
    "ヴィンテージ ビンテージ オールド 年代 当時 名作 名品 アーカイブ 歴史 伝統 老舗 本格 通好み マニア 風格"
).split()


def text_of(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def facts(p):
    """CMS の項目から、確定の情報だけを取り出す。"""
    brand = (p.get("brand") or {}).get("name") if isinstance(p.get("brand"), dict) else None
    old = text_of(p.get("descriptionHtml")).replace("はやまるブログ", "").strip()
    f = {
        "商品名": p.get("title"),
        "ブランド": brand,
        "品目": p.get("productType"),
        "区分": {"used": "古着", "new": "新品", "consignment": "委託（新品）"}.get(
            p.get("kind"), p.get("kind")
        ),
        "素材": None
        if re.search(r"不明|unknown|possibly|\?|？|かも", p.get("material") or "", re.I)
        else (p.get("material") or None),
        "生地（登録）": [NAMES[i] for i in p.get("fabrics") or [] if i in NAMES] or None,
        "色・柄（登録）": [NAMES[i] for i in p.get("colors") or [] if i in NAMES] or None,
        "原産国": p.get("originLabel") or p.get("originCountry") or None,
        "状態メモ": p.get("conditionNote") or None,
        "特徴（登録）": p.get("features") or None,
        "今の説明文（店が書いたもの。事実として使ってよい）": old if len(old) >= 12 else None,
    }
    return {k: v for k, v in f.items() if v}


MATERIAL_WORDS = {
    "cotton": "コットン|綿",
    "wool": "ウール|毛",
    "silk": "シルク|絹",
    "linen": "リネン|麻",
    "rayon": "レーヨン",
    "nylon": "ナイロン",
    "polyester": "ポリエステル",
    "cashmere": "カシミ",
    "acrylic": "アクリル",
    "corduroy": None,
    "suede": None,
}


def material_conflict(p):
    """商品名に書いてある素材が、素材の欄に無い（どちらかが間違い。人が確かめる）。"""
    m = p.get("material") or ""
    if not m:
        return None
    for en, ja in MATERIAL_WORDS.items():
        if (
            ja
            and re.search(rf"\b{en}\b", p.get("title") or "", re.I)
            and not re.search(ja + "|" + en, m, re.I)
        ):
            return f"商品名の {en} が素材の欄（{m}）に無い"
    return None


def photo(p):
    ims = p.get("images") or []
    pick = next(
        (i for i in ims if i.get("autoAltLabel") == "正面" and i.get("shopifyUrl")), None
    ) or next((i for i in ims if i.get("shopifyUrl")), None)
    if not pick:
        return None
    url = pick["shopifyUrl"]
    return base64.b64encode(
        get(url + ("&" if "?" in url else "?") + "width=900", raw=True)
    ).decode()


def check(html, look, f):
    t = text_of(html)
    problems = []
    if re.search(r"[!?！？]", t):
        problems.append("感嘆符・疑問符")
    hit = [w for w in BANNED if w in t]
    if hit:
        problems.append("禁止の言葉: " + "・".join(hit))
    if re.search(r"<(?!/?(p|ul|li)\b)[^>]*>", html):
        problems.append("使えないタグ")
    if (
        re.sub(r"</?(p|ul|li)>", "", html).count("<")
        or html.count("<p>") != html.count("</p>")
        or html.count("<li>") != html.count("</li>")
        or html.count("<ul>") != html.count("</ul>")
    ):
        problems.append("HTML が崩れている")
    if not (90 <= len(t) <= 240):
        problems.append(f"長さ {len(t)} 字")
    paras = re.findall(r"<p>(.*?)</p>", html, re.S)
    if len(paras) < 2:
        problems.append("合わせ方の文が無い")
    elif len([x for x in re.split(r"。", text_of(paras[-1])) if x.strip()]) > (
        2 if f.get("状態メモ") else 1
    ):
        problems.append("最後の段落が長い（合わせ方は 1 文）")
    if re.search(r"べき|しましょう|してください|に最適|がおすすめ|間違いな", t):
        problems.append("断定・指示の言い方")
    if re.search(r"自然に馴染|自然なバランス|無地のデニム|無地のチノ|前開きのない", t):
        problems.append("決まり文句（自然に馴染む・無地のデニム・前開きのない）")
    if re.search(r"T-?\s?Shirt|\bTee\b", f.get("商品名") or "", re.I) and re.search(
        r"T\s?シャツの上", t
    ):
        problems.append("T シャツに「T シャツの上に」と書いている")
    if re.search(r"落ち着いた[^。]{0,12}なので|一枚です|一枚。", t):
        problems.append("決まり文句（落ち着いた〜なので・一枚です）")
    if look and look.get("base_color") and look["base_color"][:2] not in t:
        problems.append("地の色が入っていない")
    if not f.get("状態メモ") and re.search(
        r"状態|コンディション|未使用|新品|ダメージ|傷み|汚れ", t
    ):
        problems.append("状態メモが無いのに状態に触れている")
    if re.search(r"スリット|裏地|シルエット", t) and not re.search(
        r"スリット|裏地|シルエット", json.dumps(f, ensure_ascii=False)
    ):
        problems.append("渡していない作りを書いている")
    if f.get("素材") and not any(
        w in t for w in re.findall(r"[ァ-ヶー]{2,}|[A-Za-z]{4,}|[一-龥]{1,2}", f["素材"])
    ):
        problems.append("素材が入っていない")
    if (
        not f.get("素材")
        and not f.get("生地（登録）")
        and re.search(
            r"コットン|ウール|ポリエステル|シルク|レーヨン|リネン|ナイロン|アクリル|綿|麻|絹", t
        )
        and not re.search(
            r"コットン|ウール|ポリエステル|シルク|レーヨン|リネン|ナイロン|アクリル|cotton|wool|silk|rayon|linen|nylon",
            json.dumps(f, ensure_ascii=False),
            re.I,
        )
    ):
        problems.append("渡していない素材を書いている")
    if re.search(r"19\d\d|20\d\d|\d0年代", t) and not re.search(
        r"19\d\d|20\d\d|\d0年代|\d0s", json.dumps(f, ensure_ascii=False)
    ):
        problems.append("渡していない年代を書いている")
    return problems


def describe(p):
    f = facts(p)
    img = photo(p)
    look = None
    conflict = material_conflict(p)
    if img:
        raw = chat(
            LOOK_SYSTEM,
            "この写真の服について答えてください。商品名: " + str(p.get("title")),
            [img],
            0.1,
            True,
        )
        try:
            look = {
                k: v for k, v in json.loads(raw).items() if v not in ("分からない", None, "", [])
            }
        except Exception:
            look = None
    if look:
        f.pop("色・柄（登録）", None)  # 登録の色は選択肢から近いものを選んだだけ。写真を優先
        if look.get("front") == "前開きなし":
            look.pop("front")  # かぶりの服では書くことではない
    if not f.get("素材"):
        f.pop("生地（登録）", None)  # 素材の表示が無い商品の生地は推測で入っている。書かせない
    user = (
        "次のデータから、ルールどおりの短い商品説明文（HTML）だけを出力してください。\n\n【確定の情報（CMS）】\n"
        + json.dumps(f, ensure_ascii=False, indent=1)
        + "\n\n【写真から読んだこと】\n"
        + json.dumps(look, ensure_ascii=False, indent=1)
    )
    html, problems = "", []
    for _attempt in range(3):
        html = chat(
            WRITE_SYSTEM,
            user
            + (
                ""
                if not problems
                else "\n\n前の出力は次の点がルール違反でした。直して書き直してください: "
                + "、".join(problems)
            ),
            temperature=0.35,
        )
        html = re.sub(r"^```(?:html)?|```$", "", html.strip(), flags=re.M).strip()
        problems = check(html, look, f)
        if not problems:
            break
    if conflict:
        problems = problems + [conflict]
    return {
        "id": p["id"],
        "title": p.get("title"),
        "facts": f,
        "look": look,
        "html": html,
        "chars": len(text_of(html)),
        "problems": problems,
        "old": text_of(p.get("descriptionHtml")),
    }


def products_from_csv(path):
    ids = []
    for r in list(csv.reader(open(path, encoding="utf-8-sig")))[3:]:
        m = re.match(r"shopify_JP_(\d+)_", r[1])
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "merchant-tops.csv"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "out.json"))
    a = ap.parse_args()
    out = []
    for sid in products_from_csv(a.csv)[a.offset : a.offset + a.limit]:
        q = urllib.parse.urlencode(
            {
                "where[shopify.productId][equals]": f"gid://shopify/Product/{sid}",
                "depth": 1,
                "limit": 1,
            }
        )
        docs = get(f"{CMS}/api/products?{q}")["docs"]
        if not docs:
            print("CMS に無い", sid)
            continue
        t = time.time()
        r = describe(docs[0])
        r["seconds"] = round(time.time() - t, 1)
        out.append(r)
        print(
            f"\n■ {r['title']}（{r['chars']} 字・{r['seconds']} 秒）{' ⚠ ' + '、'.join(r['problems']) if r['problems'] else ''}\n  写真: {json.dumps(r['look'], ensure_ascii=False)}\n{r['html']}",
            flush=True,
        )
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
