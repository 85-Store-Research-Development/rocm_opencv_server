# 商品説明を手元のモデルで書く（85store-cms）

Google Merchant Center が「説明文に色・柄・素材が足りない」と指摘した商品に、短い説明文を入れる。
文章は Ollama の `gemma4:26b` が書き、人が一覧で確かめてから CMS に送る。翻訳は CMS の今の仕組みが行う。

方針（店主。2026-11-23）: 上質で、新品の服とも合わせやすいレギュラー古着として、事実を静かに伝える。
古さ・珍しさ・ブランドの格を売りにしない。売り込まない。短く（100〜200 字）。

## 流れ

作業用のフォルダ（例 `~/data/85store-descriptions`）に、Merchant Center の CSV（`merchant-tops.csv`）と、
85store-cms の `src/shopify/taxonomy.ts` の写しを置き、そのフォルダで動かす。

```sh
cd ~/data/85store-descriptions
S=~/orca/rocm_opencv_server/scripts/cms_descriptions
python3 $S/describe.py --limit 5        # 数件を試す（書き込まない）
python3 $S/gen_all.py [--resume]        # 対象の全商品の案 → drafts.json（products.json に商品の一覧が要る）
python3 $S/build_page.py                # 確かめる一覧 → page/index.html
python3 $S/send.py 0 10                 # 確かめた案を 10 件送る（POST /api/products/descriptions）
```

- `describe.py`: 2 段階。(1) 正面の写真から、色・柄・襟・ポケットなどを選択肢で読む（迷えば「分からない」）。
  (2) CMS の項目（確定の情報）と合わせて、説明文を書く。出力は機械で確かめ（禁止の言葉・感嘆符・長さ・
  渡していない素材や年代・HTML の崩れ・決まり文句）、掛かれば 2 回まで書き直す。
- 対象は、売り場に出ていて、今の説明に店が書いた文章（30 字以上）が無い商品。短い定型文
  （「◯◯のシャツ。コットン素材を使用した一着です。」）は置き換える。リンクと採寸の行はうしろに残す。
- 素材が空の商品は、素材に触れない（写真から推測しない）。商品名と素材の欄が食い違う商品は作らない。
- CMS へは PC の端末として tailnet の whois で通る（トークンは要らない）。
