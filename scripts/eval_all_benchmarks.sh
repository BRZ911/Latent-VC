#!/bin/bash
# eval_all_benchmarks.sh
# Evaluate Video LVC (onlion) model on all Video-R1 benchmarks.
#
# Supports both SFT and GRPO checkpoints.
# Set MODEL_TYPE="sft" or MODEL_TYPE="grpo" (default: grpo).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DECORD_EOF_RETRY_MAX=20480
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ==============================
# Configuration (EDIT THESE)
# ==============================

# Model type: "sft" or "grpo"
MODEL_TYPE="${MODEL_TYPE:-grpo}"

# Model checkpoint path & RL-forward flag
if [ "$MODEL_TYPE" = "sft" ]; then
    MODEL_PATH="${MODEL_PATH:-checkpoints/video_lvc_sft/checkpoint-1200}"
    USE_RL_FORWARD=""
    DEFAULT_FILE_NAME="lvc_sft1200"
else
    MODEL_PATH="${MODEL_PATH:-checkpoints/video_lvc_grpo/checkpoint-800}"
    USE_RL_FORWARD="--use_rl_forward"
    DEFAULT_FILE_NAME="lvc_new_grpo_800"
fi

# Output identifier (used in output filenames)
FILE_NAME="${FILE_NAME:-$DEFAULT_FILE_NAME}"

# Evaluation data & output dirs
EVAL_DIR="${EVAL_DIR:-src/r1-v/Evaluation}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results_final}"

# Datasets to evaluate (comma-separated)
# Available: videomme,mvbench,tempcompass,videommmu,vsibench,mmvu
DATASETS="${DATASETS:-vsibench}"

# LVC decoding config
LVC_STEPS="${LVC_STEPS:-8}"
DECODING_STRATEGY="${DECODING_STRATEGY:-lvc_reasoning}"

# Generation config
MAX_FRAMES="${MAX_FRAMES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Video LVC - Multi-Benchmark Evaluation"
echo "============================================"
echo "  Model Type:       $MODEL_TYPE"
echo "  Model:            $MODEL_PATH"
echo "  File Name:        $FILE_NAME"
echo "  Eval Dir:         $EVAL_DIR"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Datasets:         $DATASETS"
echo "  LVC Steps:        $LVC_STEPS"
echo "  Decoding:         $DECODING_STRATEGY"
echo "  RL Forward:       $([ -n "$USE_RL_FORWARD" ] && echo 'True' || echo 'False')"
echo "  Max Frames:       $MAX_FRAMES"
echo "  Max New Tokens:   $MAX_NEW_TOKENS"
echo "============================================"

CMD="python eval_all_benchmarks.py \
    --model_path $MODEL_PATH \
    --file_name $FILE_NAME \
    --eval_dir $EVAL_DIR \
    --output_dir $OUTPUT_DIR \
    --datasets $DATASETS \
    --lvc_steps $LVC_STEPS \
    --decoding_strategy $DECODING_STRATEGY \
    --max_frames $MAX_FRAMES \
    --max_new_tokens $MAX_NEW_TOKENS \
    $USE_RL_FORWARD"

eval $CMD
