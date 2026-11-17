# bbt-neko — 猫娘模型微调项目

把 Qwen3-8B 微调成猫娘「咪露」：自部署 Qwen3-8B 合成 SFT 语料 → LoRA 微调（LLaMA-Factory）→ 评估 → vLLM 部署。全部训练跑在 BBT 集群（Volcano + RTX 5090）。

## 目录

```
persona/neko.md          人设唯一真源（system prompt 从这里生成，改人设只改这里）
data/seeds/              41 条人工种子对话（few-shot 锚点 + 合成时的语气标尺）
data/scripts/synth.py    场景矩阵合成（断点续跑）
data/scripts/judge.py    LLM-as-judge 四维打分过滤（断点续跑）
data/scripts/mix_general.py  通用指令混料（防灾难性遗忘）
serve/synth-job.yaml     合成 VolcanoJob（vLLM + synth + judge 一条龙）
train/neko_lora.yaml     LoRA 微调配置
train/dataset_info.json  LLaMA-Factory 数据集注册
train/volcano-sft.yaml   训练 VolcanoJob
train/run_sft.sh         训练入口（自动从最新 checkpoint 续跑）
train/merge.yaml         LoRA 合并导出
eval/testset.md          100 条测试集规范 + 通过线
```

## 0.5 提示词版 web 测试台（今天就能玩）

微调开始前，先用提示词版验证人设设计。零依赖纯标准库：

```bash
python web/neko_web.py --port 8787 --mock          # 占位回复，只看 UI
# 接 Qwen3-8B 本尊：ModelScope API-Inference（免费额度每日 2000 次，OpenAI 兼容）
python web/neko_web.py --port 8787 \
  --base-url https://api-inference.modelscope.cn/v1 --model Qwen/Qwen3-8B \
  --api-key <MODELSCOPE_TOKEN>    # modelscope.cn 账号设置 → API Token
```

浏览器开 `http://127.0.0.1:8787`。聊天记录自动落 `data/raw/web_chat_log.jsonl`（v2 蒸馏的语料来源之一）；`persona/neko.md` 每次请求实时重读，改人设 → 页面点"清空对话"即生效。集群 vLLM 起来后接：`--base-url http://<vLLM地址>:8000/v1 --model /workspace/datasets/Qwen3-8B`。

## 0. 前置

- kubectl 已配好你的导师 NS（接入流程见集群 skill `bbt-onboard`）
- 一个 NFS PVC（下文统一叫 `workspace-pvc`），且 PVC 下已有目录结构：
  `repos/`（代码）、`datasets/`（模型）、`venvs/`、`checkpoints/`
- 两个 YAML 里的占位符：`REPLACE` → 你的 NS uid（队列 `advisor-<uid>-gpu`）；`workspace-pvc` → 实际 PVC 名

## 1. 一次性环境准备（在 DevPod 或任意挂 PVC 的 Pod 里做一次）

```bash
# 代码上 PVC
mkdir -p /workspace/repos && cd /workspace/repos
git clone <你的仓库地址> bbt-neko   # 或直接 rsync 本地目录上去

# 两个独立 venv（合成用 vLLM，训练用 LLaMA-Factory，避免 torch 版本打架）
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv venv /workspace/venvs/neko-synth --python 3.12 && source /workspace/venvs/neko-synth/bin/activate
uv pip install vllm openai datasets && deactivate

uv venv /workspace/venvs/neko-train --python 3.12 && source /workspace/venvs/neko-train/bin/activate
uv pip install llamafactory && deactivate

# 模型下载（ModelScope 国内直连）
uvx --from modelscope modelscope download --model Qwen/Qwen3-8B --local_dir /workspace/datasets/Qwen3-8B
```

> 若 vLLM 在 5090（Blackwell sm_120）上报 `no kernel image is available`，
> 重装：`uv pip install vllm --torch-backend=cu128`

## 2. 数据合成（~2400 条原始 → 过滤后 ~1500）

```bash
kubectl create -f serve/synth-job.yaml
kubectl logs -f <pod>          # PYTHONUNBUFFERED=1 已开，能实时看到进度
```

- 单卡 Qwen3-8B vLLM：合成 + 判分大约 40~70 分钟，超 1h 时限被杀也会自动续跑（maxRetry + 断点）
- 产物：`data/synth/synth_raw.jsonl`（原始）、`data/sft/filtered.jsonl`（过审）、`data/synth/rejected.jsonl`（淘汰，记得抽检 50 条看 judge 是不是太严/太松）
- 改合成量：`synth.py --total 1200`（每任务产 2 条对话）

本地小规模调试（不起集群，任何 OpenAI 兼容端点都行）：

```bash
python data/scripts/synth.py --base-url <endpoint> --total 20
python data/scripts/judge.py --base-url <endpoint>
```

## 3. 混通用数据（防遗忘，15%）

```bash
HF_ENDPOINT=https://hf-mirror.com python data/scripts/mix_general.py \
  --dataset BelleGroup/alpaca-gpt4-data-zh --sample 400
# 产物 data/sft/general_zh.jsonl
```

## 4. 训练（LoRA，单卡）

```bash
kubectl create -f train/volcano-sft.yaml
kubectl logs -f <pod>
```

- ~2000 条 × 3 epoch，5090 单卡约 30~60 分钟，1h 内大概率一次跑完；被杀自动续
- loss 曲线看 `output_dir/` 里的 `training_loss.png`；eval loss 连续上升 = 过拟合，停
- 训完合并：`llamafactory-cli export train/merge.yaml` → `/workspace/checkpoints/neko-merged-v1`

## 5. 评估

按 `eval/testset.md`：100 条固定测试集 + judge 打分 + 人工盲评 50 条。
微调版 vs 提示词版（直接拿 `persona/neko.md` 的 system prompt 套 Qwen3-8B）对比着看——提示词版是免费基线，别输给它。

## 6. 部署

```bash
vllm serve /workspace/checkpoints/neko-merged-v1 --port 8000 --gpu-memory-utilization 0.9
```

配合 nginx Ingress 暴露成 `neko.bbt.sspu.edu.cn`（流程见集群 skill `bbt-web-bootstrap`）。
注意：物理 GPU 常驻服务会被 1h 时限杀，常驻部署要么用 mirage 池（慢，不适合 LLM），要么找管理员豁免/接受分段拉起，v1 先建议按需起 + 用完停。

## 7. 上传 ModelScope

微调合并模型（`/workspace/checkpoints/neko-merged-v1`）验收后发布到魔搭：

```bash
pip install modelscope
modelscope login --token <TOKEN>
# 先在 modelscope.cn 网页端新建模型仓库 <你的用户名>/neko-miru-v1，然后：
modelscope upload <你的用户名>/neko-miru-v1 /workspace/checkpoints/neko-merged-v1
```

注意：基座 Qwen/Qwen3-8B 不需要上传，ModelScope 已有（`Qwen/Qwen3-8B`），集群直连下载。

## 常见坑速查

| 症状 | 处理 |
|---|---|
| `no kernel image is available` | vLLM/torch 需 cu128+，见 §1 注 |
| Pod 卡住但 Running、无日志 | 检查 `PYTHONUNBUFFERED=1`（模板已带） |
| checkpoint 丢了 | 只能存 `/workspace/checkpoints/`（PVC），别存容器 /tmp |
| 合成跑不完 | 正常，1h 时限 → maxRetry 自动续跑，看 `.done.json` 进度 |
| 微调完只会喵喵喵 | 混料比例没生效（检查 `interleave_probs`），或猫娘数据过拟合（epoch 降 2） |
| 口癖油腻每句都喵 | 去 rejected.jsonl 看这类样本是否过了 judge，收紧 `JUDGE_SYS` 或提高通过线 |
