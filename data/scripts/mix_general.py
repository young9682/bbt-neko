#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用指令数据混料：下载/读取中文 Alpaca 数据 → ShareGPT 单轮格式 → data/sft/general_zh.jsonl

防灾难性遗忘用：微调时按 85% 猫娘 + 15% 通用 interleave。

用法（三选一）：
  1) 自动下载（走 hf-mirror，集群可直连）：
       HF_ENDPOINT=https://hf-mirror.com python mix_general.py --dataset BelleGroup/alpaca-gpt4-data-zh --sample 400
  2) 本地已有 Alpaca 格式 jsonl（字段 instruction/input/output）：
       python mix_general.py --src /path/to/alpaca-zh.jsonl --sample 400
  3) 换数据集同理，只要是 instruction/input/output 三字段
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "data" / "sft" / "general_zh.jsonl"


def load_alpaca(src: str | None, dataset: str | None):
    if src:
        rows = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset(dataset, split="train")
    return list(ds)


def to_sharegpt(row: dict) -> dict | None:
    q = (row.get("instruction") or "").strip()
    extra = (row.get("input") or "").strip()
    a = (row.get("output") or "").strip()
    if not q or not a:
        return None
    if extra:
        q = f"{q}\n{extra}"
    return {"conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": a}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="本地 alpaca 格式 jsonl")
    ap.add_argument("--dataset", default=None, help="HF/镜像上的数据集名")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.src and not args.dataset:
        raise SystemExit("需要 --src 或 --dataset 之一")

    rows = load_alpaca(args.src, args.dataset)
    convs = [c for c in (to_sharegpt(r) for r in rows) if c]
    rng = random.Random(args.seed)
    rng.shuffle(convs)
    convs = convs[: args.sample]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(convs)} general samples -> {OUT_FILE}")


if __name__ == "__main__":
    main()
