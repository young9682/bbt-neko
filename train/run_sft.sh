#!/usr/bin/env bash
# LoRA 训练入口：自动从 PVC 上最新 checkpoint 续跑（配合 Volcano 1h 时限 + maxRetry）
set -euo pipefail

source /workspace/venvs/neko-train/bin/activate
cd /workspace/repos/bbt-neko

OUT=/workspace/checkpoints/neko-lora-v1
LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || true)

if [ -n "${LAST:-}" ]; then
  echo "[train] resuming from checkpoint-${LAST}"
  llamafactory-cli train train/neko_lora.yaml "resume_from_checkpoint=${OUT}/checkpoint-${LAST}"
else
  echo "[train] fresh start"
  llamafactory-cli train train/neko_lora.yaml
fi
