"""切り抜いた商品の写真から、SNS 広告の画像を組む（app/ad_compose.py）。

  # ブリーフ（JSON）から作る。形式は 4:5 / 1:1 / 9:16
  uv run --group clip --group clef-local python -m scripts.ad_compose \\
      --brief brief.json --out out/ --fonts <fonts dir> [--formats 4:5,1:1,9:16] [--critique]
  # CRM の未処理のブリーフを受け取り、作った PNG を返す（ホストは引数で渡す）
  ... python -m scripts.ad_compose --crm-base https://<host> --out out/ --fonts <dir>

出力: <out>/<id>_<形式>.png（4:5 は 4x5）・<id>_<形式>.layout.json・index.html（確認用）。
フォントはリポジトリに入れない（OFL。--fonts か環境変数 AD_FONTS で置き場所を指す）。
Chrome は --chrome か AD_CHROME。--critique は手元の Qwen3-VL（~/models）で上位の候補を採点する
（読めない環境では黙って飛ばす）。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx2 as httpx

from app import ad_compose as ac

CACHE = Path("data/ad-cutout-cache")


# ---------------------------------------------------------------- CRM との受け渡し（小さな関数）


def fetch_pending_briefs(base: str, client: httpx.Client | None = None) -> list[dict]:
    """GET <base>/api/ads/briefs?status=pending → {"briefs": [...]}。"""
    c = client or httpx.Client(timeout=60)
    r = c.get(f"{base.rstrip('/')}/api/ads/briefs", params={"status": "pending"})
    r.raise_for_status()
    return r.json().get("briefs", [])


def post_creative(
    base: str, brief_id: str, png: Path, meta: dict, client: httpx.Client | None = None
) -> dict:
    """POST <base>/api/ads/briefs/<元の id>/creatives（multipart: image と meta の JSON）。"""
    c = client or httpx.Client(timeout=120)
    url = f"{base.rstrip('/')}/api/ads/briefs/{quote(str(brief_id), safe='')}/creatives"
    with png.open("rb") as f:
        r = c.post(
            url,
            files={"image": (png.name, f, "image/png")},
            data={"meta": json.dumps(meta, ensure_ascii=False)},
        )
    r.raise_for_status()
    return r.json() if r.content else {}


def meta_for_post(fr: ac.FormatResult) -> dict:
    meta = fr.meta()
    # 画素数など細かい検査は残し、CRM が JSON として読めるようにする
    return json.loads(json.dumps(meta, ensure_ascii=False, default=float))


def run_briefs(
    raw_briefs: list[dict],
    formats: list[str] | None,
    out: Path,
    fonts: Path,
    *,
    chrome=None,
    critic=None,
    base_dir: Path | None = None,
    crm_base=None,
    client=None,
    **kw,
) -> tuple[list, list[dict]]:
    """ブリーフごとに作って返す。1つが失敗しても止めず、(結果, 失敗の一覧) を返す。"""
    results, failures = [], []
    for raw in raw_briefs:
        raw_id = raw.get("id") if isinstance(raw, dict) else None
        try:
            brief = ac.parse_brief(raw)
            raw_id = brief.orig_id
            fmts = formats or brief.formats or [brief.format]
            print(f"{raw_id}: {', '.join(fmts)}", flush=True)
            frs = ac.compose_brief(
                brief,
                fmts,
                out,
                fonts,
                CACHE,
                chrome=chrome,
                critic=critic,
                base_dir=base_dir,
                **kw,
            )
            if crm_base:
                for fr in frs:
                    post_creative(crm_base, brief.orig_id, fr.png, meta_for_post(fr), client)
            results.append((brief, frs))
        except Exception as e:  # 1件の失敗で全体を止めない
            reason = "; ".join(e.problems) if isinstance(e, ac.BriefError) else f"{e}"
            reason = f"{type(e).__name__}: {reason}"[:400]
            print(f"  失敗 {raw_id}: {reason}", flush=True)
            failures.append({"id": None if raw_id is None else str(raw_id), "reason": reason})
    return results, failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--brief", type=Path, action="append", help="ブリーフの JSON（複数可）")
    p.add_argument(
        "--crm-base",
        default=os.getenv("AD_CRM_BASE"),
        help="CRM のベース URL。未処理のブリーフを受け取り、結果を返す（AD_CRM_BASE でも渡せる）",
    )
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--formats", help="例: 4:5,1:1,9:16（無ければブリーフの format）")
    p.add_argument("--fonts", type=Path, default=os.getenv("AD_FONTS"))
    p.add_argument("--chrome", default=None)
    p.add_argument("--critique", action="store_true")
    p.add_argument("--render-top", type=int, default=4, help="描画して見せる候補の数")
    args = p.parse_args()

    try:
        fonts = ac.check_fonts(args.fonts)
    except ac.FontsError as e:
        sys.exit(str(e))
    formats = args.formats.split(",") if args.formats else None
    for f in formats or []:
        if f not in ac.FORMATS:
            sys.exit(f"形式は {' / '.join(ac.FORMATS)} のどれか: {f}")
    raws, base_dir = [], None
    for path in args.brief or []:
        raws.append(json.loads(path.read_text()))
        base_dir = path.parent
    if args.crm_base:
        raws += fetch_pending_briefs(args.crm_base)
    if not raws:
        if args.crm_base:
            # 定期実行（deploy/ad-compose.timer）では、これがふつう。モデルを読み込まずに終わる
            print("未処理の案はありません")
            return
        p.error("--brief か --crm-base が要ります")
    critic = None
    if args.critique:
        from app.ad_critic import load_critic

        critic = load_critic()
    t0 = time.time()
    results, failures = run_briefs(
        raws,
        formats,
        args.out,
        fonts,
        chrome=args.chrome,
        critic=critic,
        base_dir=base_dir,
        crm_base=args.crm_base,
        render_top=args.render_top,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(ac.index_html(results))
    n = sum(len(f) for _, f in results)
    print(f"{n} 枚 / {time.time() - t0:.1f} 秒 → {(args.out / 'index.html').resolve()}")
    if failures:
        print(f"失敗 {len(failures)} 件:")
        for f in failures:
            print(f"  - {f['id']}: {f['reason']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
