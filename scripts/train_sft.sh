#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore::UserWarning"

# ============================================================
# Video LVC - Stage 1: SFT Training
#
# Recurrent latent cache + InfoNCE contrastive loss
# Trains the model to reason about key visual moments in video
# ============================================================

# ==============================
# Configuration (EDIT THESE)
# ==============================

# Model: path to a Qwen3.5 multimodal model
MODEL_NAME="${MODEL_NAME:-models/Qwen3.5-9B-Base}"

# Data: path to SFT training data JSON
DATA_PATH="${DATA_PATH:-datasets/lvc_sft.json}"

# Output
OUTPUT_DIR="checkpoints/video_lvc_sft_all_image/"
export WANDB_PROJECT="Video-LVC-SFT"

# Training params
MAX_STEPS=1200
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=4
LR=1e-5

# LVC contrastive loss weight
LAMBDA_LVC=0.1

# Visual token limits
MAX_TOKEN=5120
MIN_TOKEN=128

# Video settings (DO NOT change resolution / frames)
VIDEO_MIN_PIXELS=100352   # 128 * 28 * 28
VIDEO_MAX_PIXELS=602112   # 768 * 28 * 28

RUN_NAME="VideoLVC_SFT_Lambda${LAMBDA_LVC}"

# ==============================
# Launch training
# ==============================
deepspeed "$PROJECT_ROOT/src/train/train_sft.py" \
    --run_name "$RUN_NAME" \
    --coconut True \
    --loss_lvc_fct cosine \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --lvc_head False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvc_lambda $LAMBDA_LVC \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    --fps 1.0 \
    --max_video_frames 16 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 5 \
    --dataloader_num_workers 4
