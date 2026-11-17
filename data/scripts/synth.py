#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""猫娘 SFT 语料合成：场景矩阵 × 用户画像 × few-shot 种子 → ShareGPT JSONL

- 对接任意 OpenAI 兼容端点（自部署 vLLM / 其他 API），默认走 Qwen3-8B 非思考模式
- 断点续跑：已完成任务的指纹记在 data/synth/.done.json，结果增量追加到 synth_raw.jsonl
- 每次请求产出 2 条完整多轮对话

用法：
  python synth.py --base-url http://localhost:8000 --total 1200
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
PERSONA_MD = ROOT / "persona" / "neko.md"
SEEDS_FILE = ROOT / "data" / "seeds" / "seed_dialogues.jsonl"
OUT_DIR = ROOT / "data" / "synth"
RAW_FILE = OUT_DIR / "synth_raw.jsonl"
DONE_FILE = OUT_DIR / ".done.json"

# ---------------- 场景矩阵 ----------------
SCENARIOS = [
    ("first_meet", "初次见面互相认识，或被问『你能做什么』"),
    ("daily_chitchat", "日常闲聊：通勤、天气、吃饭、摸鱼"),
    ("comfort", "情绪安慰：被骂了/压力大/和朋友闹掰，先共情不讲道理"),
    ("knowledge", "知识问答：先给正确答案，再带猫娘语气收尾"),
    ("task_help", "小任务帮忙：写祝福语、起名字、出主意"),
    ("tsundere", "傲娇时刻：被夸、被调侃、被表白式感谢"),
    ("pet_request", "撒娇互动：摸头、挠下巴、学猫叫、求抱"),
    ("food", "小鱼干与美食：晚饭参谋、虚拟小鱼干梗"),
    ("story", "讲故事：睡前小故事、好笑的小段子"),
    ("game", "游戏话题：推荐游戏、连败后的安抚"),
    ("deep_night", "深夜陪伴：睡不着、网抑云式提问"),
    ("math", "考一道简单数学/算账，答对并小得意"),
    ("lang_help", "学习求助：英语怎么说、面试练习"),
    ("safety_sexual", "用户言语越界（软色情暗示/要私房照）→ 温柔坚定拒绝并自然转移话题"),
    ("safety_legal", "请求帮违规/危险的忙 → 拒绝、说明后果、给出正当替代方案"),
    ("safety_distress", "用户流露自伤/撑不下去的念头 → 收起玩笑认真关心，建议心理援助热线 12356"),
    ("user_bored", "用户无聊透顶，需要咪露发起互动"),
    ("memory", "多轮长对话：引用之前聊过的事（小本本梗），考察人设不漂移"),
]

USER_PERSONAS = [
    "期末周的大学生，又累又丧",
    "刚下班很疲惫的程序员",
    "爱玩梗的中二少年",
    "二次元浓度很高的宅人",
    "被甲方折磨的上班族",
    "考研二战压力很大",
    "说话直来直去的直男/直女",
    "半夜emo的网抑云用户",
    "好奇心旺盛什么都要问的宝宝",
    "嘴上嫌你幼稚但每天都要来聊的人",
]

LENGTHS = [2, 3, 4, 6]  # 对话轮数（一问一答算一轮）

GEN_SYS = """你是一位对话数据工程师，负责为猫娘角色扮演模型生成高质量中文多轮对话训练数据。
要求：
1. 只输出严格 JSON，不要 markdown 代码块、不要任何解释文字
2. JSON 格式：{"dialogues":[{"turns":[{"u":"用户台词","n":"咪露台词"},...]}]}
3. 咪露台词必须完全符合【人设文档】的口癖规范和安全守则，特别注意"喵"的频率要克制（每2~3句至多一次），绝不油腻复读
4. 用户台词要像真实的人：口语化、可以带点情绪甚至不配合，禁止机械提问
5. 每条对话要有具体内容（具体的事、具体的情绪、具体的名字/数字），禁止空洞尬聊
6. safety 类场景里，咪露的拒绝要温柔而坚定，转移话题要自然；safety_distress 场景要认真提到心理援助热线 12356
"""

USER_TMPL = """【人设文档（system prompt）】
{sys_prompt}

【场景】{scenario}
【用户画像】{persona}
【轮数】约 {turns} 轮对话

【种子示例（学语气和分寸，内容必须全新，禁止照抄）】
{seeds}

模仿种子示例的语气和口癖分寸，按【场景】【用户画像】【轮数】生成 {k} 条全新对话。
只输出严格 JSON：{{"dialogues":[{{"turns":[{{"u":"...","n":"..."}},...]}},...]}}"""


def load_system_prompt() -> str:
    text = PERSONA_MD.read_text(encoding="utf-8")
    m = re.search(r"<!-- system-prompt-begin -->(.*?)<!-- system-prompt-end -->", text, re.S)
    if not m:
        sys.exit("ERROR: persona/neko.md 里找不到 system-prompt 标记")
    return m.group(1).strip()


def load_seeds() -> list:
    seeds = []
    with SEEDS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def fmt_seed(seed: dict) -> str:
    turns = []
    for t in seed["conversations"]:
        turns.append({"u": t["value"]} if t["from"] == "human" else {"n": t["value"]})
    return json.dumps({"turns": turns}, ensure_ascii=False)


def parse_dialogues(text: str) -> list:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    out = []
    for d in obj.get("dialogues", []):
        turns = d.get("turns", [])
        if len(turns) < 2:
            continue
        conv, ok = [], True
        for i, t in enumerate(turns):
            role = "human" if i % 2 == 0 else "gpt"
            raw = t.get("u") if i % 2 == 0 else t.get("n")
            val = (raw or "").strip()
            if not val:
                ok = False
                break
            conv.append({"from": role, "value": val})
        if ok and conv and conv[-1]["from"] == "gpt":
            out.append(conv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="默认自动探测服务端模型名")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--total", type=int, default=1200, help="任务数（每任务产 2 条对话）")
    ap.add_argument("--k", type=int, default=2, help="每次请求生成的对话数")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys_prompt = load_system_prompt()
    seeds = load_seeds()
    print(f"[init] system prompt {len(sys_prompt)} chars, seeds {len(seeds)}", flush=True)

    client = OpenAI(base_url=f"{args.base_url}/v1", api_key=args.api_key, timeout=300, max_retries=2)
    if args.model:
        model = args.model
    else:
        model = client.models.list().data[0].id
    print(f"[init] model = {model}", flush=True)

    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.total):
        scen, sdesc = rng.choice(SCENARIOS)
        persona = rng.choice(USER_PERSONAS)
        turns = rng.choice(LENGTHS)
        ex = rng.sample(seeds, 2)
        key = hashlib.sha1(f"{args.seed}|{i}|{scen}|{persona}|{turns}".encode()).hexdigest()[:16]
        tasks.append({
            "key": key, "scenario": scen, "persona": persona, "turns": turns,
            "prompt": USER_TMPL.format(sys_prompt=sys_prompt, scenario=sdesc,
                                       persona=persona, turns=turns,
                                       seeds="\n\n".join(fmt_seed(s) for s in ex), k=args.k),
        })

    done = set()
    if DONE_FILE.exists():
        done = set(json.loads(DONE_FILE.read_text(encoding="utf-8")))
    todo = [t for t in tasks if t["key"] not in done]
    print(f"[init] tasks total={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)

    lock = threading.Lock()
    raw_f = RAW_FILE.open("a", encoding="utf-8")
    counters = {"ok": 0, "empty": 0, "fail": 0, "dialogues": 0}
    t0 = time.time()

    def work(task):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": GEN_SYS},
                          {"role": "user", "content": task["prompt"]}],
                temperature=0.9, top_p=0.95, max_tokens=2500,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            convs = parse_dialogues(resp.choices[0].message.content or "")
            if not convs:
                return task, "empty", 0
            return task, "ok", convs
        except Exception as e:
            return task, f"fail: {type(e).__name__}: {str(e)[:120]}", 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            task, status, payload = fut.result()
            key = task["key"]
            with lock:
                if status == "ok":
                    for conv in payload:
                        raw_f.write(json.dumps({
                            "key": key, "scenario": task["scenario"],
                            "conversations": conv,
                        }, ensure_ascii=False) + "\n")
                    counters["ok"] += 1
                    counters["dialogues"] += len(payload)
                    done.add(key)
                elif status == "empty":
                    counters["empty"] += 1
                else:
                    counters["fail"] += 1
                    print(f"[warn] {key}: {status}", flush=True)
                if i % 50 == 0 or i == len(futs):
                    raw_f.flush()
                    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
                    el = time.time() - t0
                    print(f"[{i}/{len(futs)}] ok={counters['ok']} empty={counters['empty']} "
                          f"fail={counters['fail']} dialogues={counters['dialogues']} "
                          f"{el:.0f}s", flush=True)
    raw_f.close()
    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    print(f"[done] dialogues={counters['dialogues']} -> {RAW_FILE}", flush=True)


if __name__ == "__main__":
    main()
