"""広告画像の講評（任意）。手元の Qwen3-VL（タグ読みと同じ重み）に見せて 1〜5 で採点させる。

読めない環境（重み・GPU・ライブラリが無い）では load_critic が None を返し、規則の検査だけで進む。
新しいモデルはダウンロードしない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.tag_reader import DEFAULT_MODEL_PATH

PROMPT = (
    "This is an Instagram ad image for a clothing shop, made as a collage of product photos with "
    "handwritten text running along the clothes and a bold headline. Rate it strictly from 1 "
    "(bad) to 5 (excellent) on three points:\n"
    '- "readable": can every piece of text be read clearly (no text covered by or lying on top '
    "of a garment, nothing cut off)?\n"
    '- "no_overlap": are there no unwanted overlaps between garments, text and drawings?\n'
    '- "cohesion": does it look like one tidy, well-balanced magazine-style layout?\n'
    'Answer with one JSON object only, e.g. {"readable":4,"no_overlap":3,"cohesion":4}'
)
MAX_SIDE = 1024


def parse_scores(text: str) -> dict | None:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("readable", "no_overlap", "cohesion"):
        v = raw.get(k)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return None
        out[k] = float(v)
    out["mean"] = round(sum(out.values()) / 3, 2)
    return out


class Critic:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = Path(model_path).expanduser()
        self._torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)

    def score(self, png: Path) -> dict:
        from PIL import Image

        image = Image.open(png).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
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
            out = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        scores = parse_scores(text)
        if scores is None:
            raise ValueError(f"採点を読めなかった: {text[:80]}")
        scores["raw"] = text.strip()
        return scores


def load_critic():
    """読めなければ None（黙って飛ばす）。"""
    try:
        return Critic()
    except Exception as e:  # 重み・GPU・ライブラリのどれが無くても規則だけで進める
        print(f"講評のモデルを読めないので飛ばします: {type(e).__name__}")
        return None
