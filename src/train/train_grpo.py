"""
Video LVC - GRPO (Stage 2) Training Entry.

Usage:
    deepspeed src/train/train_grpo.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --checkpoint_name checkpoints/sft/checkpoint-500 \
        --data_path data/grpo_data.json \
        ...
"""

import sys
import os
import types
import importlib.machinery
import torch


def _install_flash_attn_stub(import_error):
    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash_attn is unavailable. Use --disable_flash_attn2 True.") from import_error

    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn_mod.flash_attn_func = _unavailable
    flash_attn_mod.flash_attn_varlen_func = _unavailable

    bert_padding_mod = types.ModuleType("flash_attn.bert_padding")
    bert_padding_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding_mod.index_first_axis = _unavailable
    bert_padding_mod.pad_input = _unavailable
    bert_padding_mod.unpad_input = _unavailable

    layers_mod = types.ModuleType("flash_attn.layers")
    layers_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers", loader=None)
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers.rotary", loader=None)
    rotary_mod.apply_rotary_emb = _unavailable

    flash_attn_mod.bert_padding = bert_padding_mod
    flash_attn_mod.layers = layers_mod
    layers_mod.rotary = rotary_mod

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding_mod
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod


BROKEN_FLASH_ATTN_ERROR = None
try:
    import flash_attn  # noqa: F401
except Exception as flash_attn_error:
    BROKEN_FLASH_ATTN_ERROR = flash_attn_error
    _install_flash_attn_stub(flash_attn_error)


from transformers import AutoProcessor, AutoConfig, HfArgumentParser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.params_grpo import DataArguments, ModelArguments, GRPOArguments
from src.trainer.grpo_trainer import VideoLVCV2GRPOTrainer as VideoLVCGRPOTrainer
from src.dataset.grpo_dataset import make_grpo_data_module
from src.model.forward_grpo import replace_with_grpo_forward
from src.train.reward_funcs import get_reward_funcs

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    visual_module = getattr(model, "visual", None)
    if visual_module is None and hasattr(model, "model"):
        visual_module = getattr(model.model, "visual", None)
    if visual_module is not None:
        visual_module.to(dtype=compute_dtype, device=device)
        set_requires_grad(visual_module.parameters(), not training_args.freeze_vision_tower)
        merger = getattr(visual_module, "merger", None)
        if merger is not None:
            set_requires_grad(merger.parameters(), not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    llm_backbone = getattr(model, "model", None)
    if llm_backbone is not None:
        set_requires_grad(llm_backbone.parameters(), not training_args.freeze_llm)


def safe_save_model_for_hf_trainer(trainer, output_dir):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("=" * 60)
    rank0_print("Video LVC - GRPO Training (Stage 2)")
    rank0_print("  Rewards: accuracy + format + temporal_grounding + latent_reasoning")
    rank0_print("=" * 60)

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Model path
    if training_args.checkpoint_name:
        model_pth = training_args.checkpoint_name
        rank0_print(f"Loading from SFT checkpoint: {model_pth}")
    else:
        model_pth = model_args.model_id
        rank0_print(f"Loading base model: {model_pth}")

    if BROKEN_FLASH_ATTN_ERROR is not None and not training_args.disable_flash_attn2:
        rank0_print(f"Detected incompatible flash_attn, forcing SDPA fallback.")
        training_args.disable_flash_attn2 = True

    # Load model
    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    SUPPORTED_MODEL_TYPES = ["qwen2_5_vl", "qwen3_5"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type}. Supported: {SUPPORTED_MODEL_TYPES}")

    rank0_print(f"Loading model (model_type={model_type})...")

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_pth, config=config, torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )
    elif model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_pth, config=config, torch_dtype=compute_dtype, trust_remote_code=True,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_pth, config=config, torch_dtype=compute_dtype, trust_remote_code=True,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

    # Apply GRPO forward patch
    replace_with_grpo_forward(model=model, model_type=model_type)
    model.config.use_cache = False

    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    # Processor and special tokens
    processor = AutoProcessor.from_pretrained(
        model_pth,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )

    processor.tokenizer.add_tokens("<|lvc_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvc|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvc_latent_end|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvc_end|>", special_tokens=True)

    model.config.lvc_id = processor.tokenizer.convert_tokens_to_ids("<|lvc|>")
    model.config.lvc_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_latent_end|>")
    model.config.lvc_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_start|>")
    model.config.lvc_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_end|>")

    model_vocab_size = getattr(model.config, "vocab_size", None)
    if model_vocab_size is None and hasattr(model.config, "text_config"):
        model_vocab_size = getattr(model.config.text_config, "vocab_size", None)
    if model_vocab_size is not None and model_vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    def _param_numel(p):
        if hasattr(p, "ds_numel"):
            return p.ds_numel
        return p.numel()

    total_params = sum(_param_numel(p) for p in model.parameters())
    trainable_params = sum(_param_numel(p) for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

    # Dataset
    rank0_print(f"Loading GRPO dataset from {data_args.data_path}...")
    dataset_module = make_grpo_data_module(model_id=model_args.model_id, processor=processor, data_args=data_args)
    rank0_print(f"  Train dataset size: {len(dataset_module['train_dataset'])}")

    # Reward functions
    reward_funcs = get_reward_funcs()
    rank0_print(f"Reward functions: {[f.__name__ for f in reward_funcs]}")

    if training_args.reward_weights is None:
        training_args.reward_weights = [1.0, 0.5, 0.5, 1.0]

    # Train
    rank0_print("Initializing GRPO Trainer...")
    trainer = VideoLVCGRPOTrainer(
        model=model, ref_model_pth=model_pth, reward_funcs=reward_funcs,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        processing_class=processor, args=training_args,
    )

    rank0_print("Starting GRPO training...")
    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        rank0_print(f"Resuming GRPO training from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
    rank0_print("GRPO Training complete!")


if __name__ == "__main__":
    train()
