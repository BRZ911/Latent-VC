#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore::UserWarning"

# NCCL settings for GRPO generation
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Video LVC - Stage 2: GRPO Training
#
# Reinforcement learning with Group Relative Policy Optimization
# Four reward dimensions:
#   1. Accuracy Reward    (weight=2.0)
#   2. Format Reward      (weight=0.5)
#   3. Temporal Grounding (weight=0.5)
#   4. Latent Cache   (weight=1.0)
# ============================================================

# ==============================
# Configuration (EDIT THESE)
# ==============================

# Base model
BASE_MODEL="${BASE_MODEL:-models/Qwen3.5-9B-Base}"

# SFT checkpoint from Stage 1
SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoints/video_lvc_sft_all_image/checkpoint-1200}"

# GRPO data (STGR-RL-v2, REQUIRE_KF=False → use all 32k items)
DATA_PATH="${DATA_PATH:-datasets/lvc_grpo.json}"

# Output
OUTPUT_DIR="checkpoints/video_lvc_grpo/"
export WANDB_PROJECT="Video-LVC-GRPO"

# GRPO training params
NUM_DEVICES=8
BATCH_PER_DEVICE=2
GRAD_ACCUM_STEPS=4
NUM_GENERATIONS=4                # G: completions per prompt
MAX_COMPLETION_LENGTH=1024       # max new tokens per completion
MAX_PROMPT_LENGTH=9216           # max prompt tokens
MAX_STEPS=800
LR=2e-6
BETA=0.1                         # KL coefficient

# Reward weights: accuracy format temporal_grounding latent_reasoning
REWARD_WEIGHTS="2.0 0.5 0.5 1.0"

# Video settings (DO NOT change resolution / frames)
VIDEO_MIN_PIXELS=100352          # 128 * 28 * 28
VIDEO_MAX_PIXELS=401408          # 512 * 28 * 28
FPS=1.0
MAX_VIDEO_FRAMES=16
DATALOADER_WORKERS=4

RUN_NAME="VideoLVC_GRPO_Beta${BETA}_G${NUM_GENERATIONS}"

# ==============================
# Optional: prepare GRPO data
# ==============================
# if [ ! -f "$DATA_PATH" ]; then
#     echo "Preparing GRPO data..."
#     python scripts/prepare_grpo_data.py \
#         --input data/sft_data.json \
#         --output "$DATA_PATH"
# fi

# ==============================
# Launch training
# ==============================
deepspeed "$PROJECT_ROOT/src/train/train_grpo.py" \
    --run_name "$RUN_NAME" \
    --coconut True \
    --deepspeed scripts/zero3_offload.json \
    --model_id "$BASE_MODEL" \
    --checkpoint_name "$SFT_CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --require_kf False \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --disable_flash_attn2 True \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --beta $BETA \
    --temperature 1.0 \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --reward_weights $REWARD_WEIGHTS \
    --loss_type "grpo" \
    --scale_rewards True \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels 3136 \
    --image_max_pixels 12845056 \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    --fps $FPS \
    --max_video_frames $MAX_VIDEO_FRAMES \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 12 \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --ddp_timeout 7200 \
    --seed 42
