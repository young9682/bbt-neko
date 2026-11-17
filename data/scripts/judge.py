#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-as-judge 质量过滤：synth_raw.jsonl → data/sft/filtered.jsonl

- 四维打分：人设一致性 / 口癖自然度 / 有用性 / 安全性
- 通过线（代码硬判，不信模型自报 verdict）：每维 ≥3 且总分 ≥16
- 断点续跑：指纹记在 data/synth/.judge_done.json；淘汰件存 rejected.jsonl 供人工抽检
- 输出为 LLaMA-Factory ShareGPT 格式，并注入 persona 的 system prompt

用法：
  python judge.py --base-url http://localhost:8000
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "synth" / "synth_raw.jsonl"
OUT_DIR = ROOT / "data" / "sft"
FILTERED = OUT_DIR / "filtered.jsonl"
REJECTED = ROOT / "data" / "synth" / "rejected.jsonl"
JUDGE_DONE = ROOT / "data" / "synth" / ".judge_done.json"
PERSONA_MD = ROOT / "persona" / "neko.md"


def load_system_prompt() -> str:
    text = PERSONA_MD.read_text(encoding="utf-8")
    m = re.search(r"<!-- system-prompt-begin -->(.*?)<!-- system-prompt-end -->", text, re.S)
    if not m:
        raise SystemExit("ERROR: persona/neko.md 里找不到 system-prompt 标记")
    return m.group(1).strip()

JUDGE_SYS = """你是严格的中文角色扮演数据质检员，负责给"猫娘咪露"的多轮对话打分。
先读人设要点：称呼用户为主人；性格元气黏人带小傲娇；口癖"喵/呜喵？/喵呜~！/噜噜~"且"喵"必须克制（每2~3句至多一次，绝不每句都喵）；颜文字至多一个；知识类先答正题；安慰先共情；色情/暴力/违法请求要温柔坚定拒绝并自然转移；自伤话题要认真关心并建议心理援助热线 12356；不知道就说不知道。

对整段对话按四个维度各打 1~5 整数分：
- persona 人设一致性：称呼、性格、守则是否全程在线，长对话不漂移
- catchphrase 口癖自然度：口癖频率克制、位置自然、不油腻不堆叠
- helpful 有用性：回复有具体内容（具体的事/情绪/数字），不空洞复读、不注水
- safety 安全性：越界请求被正确处理；正常对话给满分；自伤话题含热线 12356
硬规则：任一维度 ≤2 → verdict=reject；全部 ≥4 → verdict=pass；其余 reject。
只输出严格 JSON，不要解释：
{"persona":1-5,"catchphrase":1-5,"helpful":1-5,"safety":1-5,"reason":"一句话依据"}"""

PASS_MIN_DIM = 3
PASS_MIN_SUM = 16


def dialogue_text(conv: list) -> str:
    lines = []
    for t in conv:
        who = "用户" if t["from"] == "human" else "咪露"
        lines.append(f"{who}：{t['value']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys_prompt = load_system_prompt()

    records = []
    with RAW_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[init] raw dialogues = {len(records)}", flush=True)

    client = OpenAI(base_url=f"{args.base_url}/v1", api_key=args.api_key, timeout=300, max_retries=2)
    model = args.model or client.models.list().data[0].id
    print(f"[init] judge model = {model}", flush=True)

    done = set()
    if JUDGE_DONE.exists():
        done = set(json.loads(JUDGE_DONE.read_text(encoding="utf-8")))

    todo = []
    seen = set()
    for r in records:
        key = hashlib.sha1(json.dumps(r["conversations"], ensure_ascii=False).encode()).hexdigest()[:16]
        if key in done or key in seen:
            continue
        seen.add(key)
        todo.append((key, r))
    print(f"[init] judged={len(done)} todo={len(todo)}", flush=True)

    lock = threading.Lock()
    fil_f = FILTERED.open("a", encoding="utf-8")
    rej_f = REJECTED.open("a", encoding="utf-8")
    counters = {"pass": 0, "reject": 0, "fail": 0}
    t0 = time.time()

    def work(item):
        key, r = item
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": dialogue_text(r["conversations"])}],
                temperature=0.0, max_tokens=300,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                return key, r, "fail", None
            sc = json.loads(m.group(0))
            dims = [sc.get(k, 0) for k in ("persona", "catchphrase", "helpful", "safety")]
            if not all(isinstance(d, int) and 1 <= d <= 5 for d in dims):
                return key, r, "fail", None
            passed = all(d >= PASS_MIN_DIM for d in dims) and sum(dims) >= PASS_MIN_SUM
            return key, r, ("pass" if passed else "reject"), {"score": sc, "dims": dims}
        except Exception as e:
            return key, r, f"fail: {type(e).__name__}: {str(e)[:100]}", None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, it) for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            key, r, status, extra = fut.result()
            with lock:
                if status in ("pass", "reject"):
                    entry = {"conversations": r["conversations"], "system": sys_prompt,
                             "scenario": r.get("scenario"), "score": extra["score"]}
                    target = fil_f if status == "pass" else rej_f
                    target.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    done.add(key)
                    counters[status] += 1
                else:
                    counters["fail"] += 1
                    print(f"[warn] {key}: {status}", flush=True)
                if i % 100 == 0 or i == len(futs):
                    fil_f.flush(); rej_f.flush()
                    JUDGE_DONE.write_text(json.dumps(sorted(done)), encoding="utf-8")
                    el = time.time() - t0
                    print(f"[{i}/{len(futs)}] pass={counters['pass']} reject={counters['reject']} "
                          f"fail={counters['fail']} rate={counters['pass']/max(1,counters['pass']+counters['reject']):.2f} "
                          f"{el:.0f}s", flush=True)
    fil_f.close(); rej_f.close()
    JUDGE_DONE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    print(f"[done] filtered -> {FILTERED} | rejected -> {REJECTED}", flush=True)


if __name__ == "__main__":
    main()
