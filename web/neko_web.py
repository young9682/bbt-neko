#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""咪露 提示词版 web 测试台（纯标准库，零依赖）

- 浏览器打开 http://127.0.0.1:8787 即可与"咪露"（persona/neko.md 的 system prompt）对话
- 后端可插拔：任何 OpenAI 兼容 /chat/completions 端点（GLM、DeepSeek、集群 vLLM）
- 聊天记录自动追加到 data/raw/web_chat_log.jsonl —— 这就是将来 v2 蒸馏的语料库
- persona/neko.md 每次请求实时重读，改完人设点"清空对话"立刻生效
- --mock 模式不接任何后端，纯验证 UI/链路

用法：
  # 1) 先看 UI 长什么样（占位回复）
  python web/neko_web.py --port 8787 --mock

  # 2) 接真实后端（三选一，key 也可以用环境变量 NEKO_API_KEY）
  python web/neko_web.py --port 8787 --base-url https://open.bigmodel.cn/api/paas/v4 --model glm-4-flash --api-key <KEY>
  python web/neko_web.py --port 8787 --base-url https://api.deepseek.com --model deepseek-chat --api-key <KEY>
  python web/neko_web.py --port 8787 --base-url http://<集群vLLM地址>:8000/v1 --model /workspace/datasets/Qwen3-8B
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONA_MD = ROOT / "persona" / "neko.md"
LOG_FILE = ROOT / "data" / "raw" / "web_chat_log.jsonl"

CFG = {"base_url": "", "model": "", "api_key": "", "mock": False}

MOCK_REPLIES = [
    "喵？主人在叫我~（我是 MOCK 占位回复：用 --base-url / --api-key 接上真实后端，咪露才会真正开嗓喵）",
    "(=^･ω･^=) 咪露在这里喵~ 这条也是占位回复哦主人，配置好端点后记得重启服务喵！",
    "呜喵……现在还没接真模型呢。主人把 --base-url 和 --api-key 给我，我就能真的说话了喵！",
]

MOCK_STATE = {"i": 0}


def system_prompt() -> str:
    text = PERSONA_MD.read_text(encoding="utf-8")
    m = re.search(r"<!-- system-prompt-begin -->(.*?)<!-- system-prompt-end -->", text, re.S)
    return m.group(1).strip() if m else "你是咪露（Miru），一只猫娘陪伴 AI，称呼用户为主人。"


def call_upstream(messages: list) -> str:
    url = CFG["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": CFG["model"],
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if CFG["api_key"]:
        req.add_header("Authorization", f"Bearer {CFG['api_key']}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def mock_reply() -> str:
    time.sleep(0.4)
    r = MOCK_REPLIES[MOCK_STATE["i"] % len(MOCK_REPLIES)]
    MOCK_STATE["i"] += 1
    return r


def log_turn(session: str, user: str, assistant: str, model: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session": session,
        "user": user,
        "assistant": assistant,
        "model": model,
        "source": "web_test",
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>咪露 · 提示词版测试台</title>
<style>
  :root { --bg:#12141a; --panel:#1b1e27; --neko:#f6a5c0; --user:#6ea8fe; --text:#e8eaf0; --dim:#8b93a7; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 15px/1.6 "Microsoft YaHei", system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  header { padding: 12px 18px; background: var(--panel); display: flex; align-items: center; gap: 10px;
           border-bottom: 1px solid #2a2e3c; }
  header b { font-size: 16px; }
  .badge { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #2a2e3c; color: var(--dim); }
  .badge.mock { background: #5a3b1e; color: #ffb864; }
  header .spacer { flex: 1; }
  button#clear { background: none; border: 1px solid #3a3f52; color: var(--dim); border-radius: 8px;
                 padding: 4px 12px; cursor: pointer; font-size: 13px; }
  button#clear:hover { color: var(--text); border-color: var(--neko); }
  #chat { flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 72%; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; word-break: break-word; }
  .neko { align-self: flex-start; background: var(--panel); border: 1px solid #3a2f3c; border-left: 3px solid var(--neko); }
  .user { align-self: flex-end; background: #24304a; border: 1px solid #31406b; border-right: 3px solid var(--user); }
  .msg.err { border-left-color: #ff6b6b; color: #ffb0b0; font-size: 13px; }
  .typing { color: var(--dim); font-size: 13px; align-self: flex-start; padding: 6px 14px; }
  footer { padding: 14px 18px; background: var(--panel); border-top: 1px solid #2a2e3c; display: flex; gap: 10px; }
  #input { flex: 1; background: #12141a; border: 1px solid #3a3f52; border-radius: 10px; color: var(--text);
           padding: 10px 14px; font-size: 15px; resize: none; height: 46px; outline: none; }
  #input:focus { border-color: var(--neko); }
  #send { background: var(--neko); border: none; color: #1b1e27; font-weight: bold; border-radius: 10px;
          padding: 0 22px; cursor: pointer; font-size: 15px; }
  #send:disabled { opacity: .5; cursor: wait; }
</style>
</head>
<body>
<header>
  <b>咪露（Miru）</b><span style="color:var(--dim);font-size:13px">提示词版 · persona/neko.md</span>
  <span id="badge" class="badge">…</span>
  <div class="spacer"></div>
  <button id="clear">清空对话</button>
</header>
<div id="chat"></div>
<footer>
  <textarea id="input" placeholder="跟咪露说点什么…（Enter 发送，Shift+Enter 换行）"></textarea>
  <button id="send">发送</button>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');
const badge = document.getElementById('badge');
let history = [];   // {role:'user'|'assistant', content}

fetch('/api/config').then(r=>r.json()).then(c=>{
  badge.textContent = c.mock ? 'MOCK 占位模式' : (c.model || '已配置');
  badge.className = 'badge' + (c.mock ? ' mock' : '');
});

function bubble(role, text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + (cls || role);
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function ask() {
  const text = input.value.trim();
  if (!text || send.disabled) return;
  input.value = '';
  history.push({role:'user', content:text});
  bubble('user', text);
  send.disabled = true;
  const t = bubble('typing', '咪露思考中…');
  try {
    const resp = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages: history.slice(-40)})
    });
    const data = await resp.json();
    t.remove();
    if (data.error) { bubble('err', '⚠ ' + data.error); }
    else {
      history.push({role:'assistant', content:data.reply});
      bubble('neko', data.reply);
    }
  } catch (e) {
    t.remove();
    bubble('err', '⚠ 请求失败: ' + e.message);
  }
  send.disabled = false;
  input.focus();
}

send.onclick = ask;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
document.getElementById('clear').onclick = () => { history = []; chat.innerHTML = ''; input.focus(); };
bubble('neko', '喵呜~！咪露上线啦！主人今天想做点什么喵？');
input.focus();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/config":
            shown = "off" if CFG["mock"] else (CFG["model"] or "NOT-SET")
            self._json(200, {"mock": CFG["mock"], "model": shown})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except UnicodeDecodeError:
                return self._json(400, {"error": "request body must be UTF-8"})
            session = str(body.get("session", ""))[:40]
            history = [m for m in body.get("messages", [])
                       if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
            if not history or history[-1]["role"] != "user":
                return self._json(400, {"error": "bad messages"})
            messages = [{"role": "system", "content": system_prompt()}] + history[-40:]
            if CFG["mock"]:
                reply = mock_reply()
            else:
                try:
                    reply = call_upstream(messages)
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", "replace")[:300]
                    return self._json(502, {"error": f"upstream HTTP {e.code}: {detail}"})
                except Exception as e:
                    return self._json(502, {"error": f"upstream {type(e).__name__}: {e}"})
            user_text = history[-1]["content"]
            log_turn(session, user_text, reply, "mock" if CFG["mock"] else CFG["model"])
            self._json(200, {"reply": reply})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):  # 静默访问日志
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--base-url", default="", help="OpenAI 兼容端点，如 https://open.bigmodel.cn/api/paas/v4")
    ap.add_argument("--model", default="", help="模型名，如 glm-4-flash / deepseek-chat")
    ap.add_argument("--api-key", default=os.environ.get("NEKO_API_KEY", ""))
    ap.add_argument("--mock", action="store_true", help="不接后端，用占位回复验证 UI")
    args = ap.parse_args()

    CFG.update(base_url=args.base_url, model=args.model, api_key=args.api_key, mock=args.mock)
    if not args.mock and (not args.base_url or not args.model):
        print("[cfg] 未配置后端！用 --mock 先看 UI，或加上：\n"
              "      --base-url https://open.bigmodel.cn/api/paas/v4 --model glm-4-flash --api-key <KEY>\n"
              "      （key 也可放环境变量 NEKO_API_KEY）")
        return

    print(f"[neko-web] http://{args.host}:{args.port}  mock={args.mock}  model={CFG['model'] or 'mock'}")
    print(f"[neko-web] 聊天记录 -> {LOG_FILE}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
