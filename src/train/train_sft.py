"""
Video LVC - SFT (Stage 1) Training Entry.

Usage:
    deepspeed src/train/train_sft.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --data_path data/sft_data.json \
        ...
"""

import sys
import os
import types
import importlib.machinery
import torch


def _install_flash_attn_stub(import_error):
    """Install minimal flash_attn stubs so transformers import can proceed."""
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

from src.params import DataArguments, ModelArguments, TrainingArguments
from src.trainer.sft_trainer import SFTTrainer
from src.dataset import make_sft_data_module
from src.model.forward_sft import replace_with_sft_forward, LVCProjectionHead

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

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("=" * 60)
    rank0_print("Video LVC - SFT Training (Stage 1)")
    rank0_print("  Mode: Recurrent latent cache + InfoNCE contrastive loss")
    rank0_print("=" * 60)

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if BROKEN_FLASH_ATTN_ERROR is not None and not training_args.disable_flash_attn2:
        rank0_print(f"Detected incompatible flash_attn, forcing SDPA fallback.")
        training_args.disable_flash_attn2 = True

    # Load model
    model_pth = model_args.model_id
    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    SUPPORTED_MODEL_TYPES = ["qwen2_5_vl", "qwen3_5"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type}. Supported: {SUPPORTED_MODEL_TYPES}")

    rank0_print(f"Loading model from {model_pth} (model_type={model_type})...")

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

    # Create LVC Projection Head
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(config, "text_config"):
        hidden_size = getattr(config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Cannot find hidden_size in model config")

    lvc_proj_head = LVCProjectionHead(hidden_size=hidden_size, proj_size=hidden_size, dropout=0.1).to(dtype=compute_dtype)
    rank0_print(f"Created LVC Projection Head: {hidden_size} -> {hidden_size}")

    replace_with_sft_forward(model=model, model_type=model_type, lvc_proj_head=lvc_proj_head)
    model.config.use_cache = False
    model.config.lvc_temperature = 0.07

    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    for p in model.lvc_proj_head.parameters():
        p.requires_grad = True

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Processor and special tokens
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )

    processor.tokenizer.add_tokens("<|lvc_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvc|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvc_latent_end|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvc_end|>", special_tokens=True)

    lvc_id = processor.tokenizer.convert_tokens_to_ids("<|lvc|>")
    model.config.lvc_id = lvc_id
    model.config.lvc_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_latent_end|>")
    model.config.lvc_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_start|>")
    model.config.lvc_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_end|>")

    model_vocab_size = getattr(model.config, "vocab_size", None)
    if model_vocab_size is None and hasattr(model.config, "text_config"):
        model_vocab_size = getattr(model.config.text_config, "vocab_size", None)
    if model_vocab_size is not None and model_vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    model.config.loss_lvc_fct = training_args.loss_lvc_fct

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total: {total_params:,}, Trainable: {trainable_params:,} ({100.0 * trainable_params / total_params:.1f}%)")

    # Dataset
    rank0_print(f"Loading dataset from {data_args.data_path}...")
    data_module = make_sft_data_module(model_id=model_args.model_id, processor=processor, data_args=data_args)

    # Train
    trainer = SFTTrainer(model=model, processing_class=processor, args=training_args, **data_module)
    rank0_print("Starting SFT training...")
    trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
    rank0_print("SFT Training complete!")


if __name__ == "__main__":
    train()
