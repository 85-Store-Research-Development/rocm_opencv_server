"""服のタグの写真から、ブランド名・素材・原産国・サイズを読む（Qwen3-VL を手元の GPU で）。

Clef は選択肢から選ぶだけで文字を書けないので、タグの文字を読むのはこちらで行う
（scripts/cms_photo_batches.py が、まとまりを商品ごとに分けたあとに呼ぶ）。

- モデルは Qwen3-VL-8B-Instruct（BF16 で約 17GB）。置き場所は ~/models/qwen3-vl-8b-instruct
  （TAG_READER_MODEL_PATH）。
  GPU は 1 つなので、Clef と同時には載せない（Clef を下ろしてから読み込む）
- 読めないものは null にさせる（推測で埋めない）。値は人が CMS の画面で確かめる
- 素材はタグの表記のまま（例: "Cotton 100%"、"65% Polyester / 35% Cotton"）
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

DEFAULT_MODEL_PATH = "~/models/qwen3-vl-8b-instruct"
MAX_SIDE = 1280  # タグの小さな文字が読める大きさ（長辺）

PROMPT = "\n".join(
    [
        "These photos show the labels/tags of one second-hand garment"
        " (brand label, size label, care label).",
        "Read only what is actually printed. Do not guess."
        " If something is not visible or not readable, use null.",
        "",
        "Answer with one JSON object and nothing else:",
        "{",
        '  "brand": "brand name as printed on the brand label'
        ' (not the retailer, not the manufacturer code)",',
        '  "material": "fiber composition as printed, in English,'
        ' e.g. \\"Cotton 100%\\" or \\"Polyester 65% / Cotton 35%\\"",',
        '  "origin_country": "country after MADE IN / Hecho en / 製, as printed",',
        '  "origin_code": "ISO 3166-1 alpha-2 code of that country'
        ' (e.g. US, CN, JP, GB), or null",',
        '  "size": "size as printed, e.g. \\"L\\", \\"XL\\", \\"42\\", \\"M/M\\""',
        "}",
    ]
)


class TagReaderError(Exception):
    pass


def parse_answer(text: str) -> dict:
    """モデルの答えから JSON を取り出し、値を整える（空文字・"null"・"unknown" は None）。"""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise TagReaderError(f"JSON がありません: {text[:200]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise TagReaderError(f"JSON を読めません: {e}: {text[:200]}") from e
    out = {}
    for key in ("brand", "material", "origin_country", "origin_code", "size"):
        value = data.get(key)
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in ("null", "none", "unknown", "n/a", "not visible"):
                value = None
        elif value is not None:
            value = str(value)
        out[key] = value
    if out["origin_code"]:
        code = out["origin_code"].upper()
        out["origin_code"] = code if re.fullmatch(r"[A-Z]{2}", code) else None
    return out


class TagReader:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        path = Path(model_path).expanduser()
        if not (path / "config.json").exists():
            raise TagReaderError(f"Qwen3-VL の重みがありません: {path}")
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise TagReaderError("uv run --group clip --group clef-local で起動してください") from e
        self._torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda"
        )
        self.processor = AutoProcessor.from_pretrained(path)

    def read(self, images: list[bytes]) -> dict:
        """タグの写真（同じ服の 1〜3 枚）→ brand・material・origin_country・origin_code・size。"""
        from PIL import Image

        pics = []
        for data in images:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            pics.append(image)
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": p} for p in pics),
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        result = parse_answer(text)
        result["raw"] = text.strip()
        return result

    def close(self) -> None:
        """GPU から下ろす（Clef と順番に使うため）。"""
        del self.model
        self._torch.cuda.empty_cache()


# 素材の表記をそろえる（「100% COTTON/COTON」→「Cotton 100%」。表地・裏地・中綿は分けて書く）
SECTIONS = {
    "SHELL": "表地",
    "EXTERIOR": "表地",
    "OUTER": "表地",
    "BODY": "身頃",
    "LINING": "裏地",
    "FILLER": "中綿",
    "FILL": "中綿",
    "PADDING": "中綿",
    "TRIM": "付属",
    "RIB": "リブ",
}
# 英語でない言い方（タグに並べて書かれる）は英語にそろえる
FIBER_ALIASES = {
    "COTON": "COTTON",
    "ALGODON": "COTTON",
    "ALGODÓN": "COTTON",
    "ACRILICO": "ACRYLIC",
    "ACRÍLICO": "ACRYLIC",
    "ACRILICA": "ACRYLIC",
    "ACRÍLICA": "ACRYLIC",
    "POLIESTER": "POLYESTER",
    "LAINE": "WOOL",
    "LANA": "WOOL",
    "SOIE": "SILK",
    "SEDA": "SILK",
    "OTHERS": "その他",
    "OTHER": "その他",
}
_SECTION = re.compile(r"\b(" + "|".join(SECTIONS) + r")\b\s*(?:/\s*[A-ZÀ-Ý]+\s*)?:", re.I)
_PAIR = re.compile(
    r"(\d{1,3})\s*%\s*([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)?)|([A-Za-zÀ-ÿ]+)\s*(\d{1,3})\s*%"
)


def _fiber(word: str) -> str:
    word = word.strip().upper()
    word = FIBER_ALIASES.get(word, word)
    return word if word == "その他" else word.capitalize()


def _pairs(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for m in _PAIR.finditer(text):
        pct, word = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        name = _fiber(word.split()[0])
        if not any(n == name for n, _ in out):
            out.append((name, int(pct)))
    return out


def normalize_material(text: str | None) -> str | None:
    """タグの素材の表記 → 「Cotton 100%」の形。% の無いもの（読めていない）は None。"""
    if not text or "%" not in text:
        return None
    marks = list(_SECTION.finditer(text))
    if not marks:
        pairs = _pairs(text)
        return " / ".join(f"{n} {p}%" for n, p in pairs) or None
    parts = []
    for i, m in enumerate(marks):
        body = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        pairs = _pairs(body)
        if pairs:
            label = SECTIONS[m.group(1).upper()]
            parts.append(f"{label}: " + " / ".join(f"{n} {p}%" for n, p in pairs))
    return "、".join(parts) or None
