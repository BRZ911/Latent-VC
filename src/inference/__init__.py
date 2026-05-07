"""
Video LVC - Inference Script.

Supports both SFT and GRPO trained models.

Decoding strategies:
- "lvc_reasoning": Insert <|lvc_start|><|lvc|>...<|lvc|><|lvc_end|> tokens before
  generation. The LVC tokens act as "thinking slots" that attend back to video
  context. Recommended for both SFT and GRPO models.
- "forced_lvc": Hidden-state feedback loop (V1 compat).
- "greedy": Normal auto-regressive decoding without LVC.
"""

import os
import sys
import types
import importlib.machinery

BROKEN_FLASH_ATTN_ERROR = None
try:
    import flash_attn  # noqa: F401
except Exception as flash_attn_error:
    BROKEN_FLASH_ATTN_ERROR = flash_attn_error

    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash_attn unavailable") from flash_attn_error

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


import torch
import torch.nn.functional as F

from transformers import AutoProcessor, AutoConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

from src.constants import VIDEO_LVC_SYSTEM_MESSAGE
from src.model.forward_sft import (
    replace_with_sft_forward,
    LVCProjectionHead,
)
from src.model.forward_grpo import replace_with_grpo_forward


# ======================================================================
# Custom generate with LVC latent cache
# ======================================================================

def lvc_generate(
    model,
    inputs,
    processor,
    max_new_tokens: int = 512,
    lvc_steps: int = 8,
    decoding_strategy: str = "lvc_reasoning",
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_rl_forward: bool = False,
):
    """
    Generate text with LVC latent visual cache.

    Decoding strategies:
    - "lvc_reasoning": Insert LVC tokens into prompt, single prefill, then auto-regressive.
    - "forced_lvc": Hidden-state feedback loop.
    - "greedy": No LVC, standard decoding.
    """
    device = next(model.parameters()).device

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    extra_kwargs = {}
    for key in [
        "pixel_values", "pixel_values_videos",
        "image_grid_thw", "video_grid_thw",
        "second_per_grid_ts", "mm_token_type_ids",
    ]:
        if key in inputs and inputs[key] is not None:
            extra_kwargs[key] = inputs[key].to(device)

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Inference currently supports batch_size=1 only."

    lvc_start_id = getattr(model.config, "lvc_start_id", -1)
    lvc_end_id = getattr(model.config, "lvc_end_id", -1)
    lvc_id = getattr(model.config, "lvc_id", -1)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = processor.tokenizer.eos_token_id

    # V2 Forced: Insert LVC tokens into prompt
    if decoding_strategy == "lvc_reasoning" and lvc_start_id > 0 and lvc_end_id > 0 and lvc_id > 0:
        print(f"  [lvc_reasoning] Inserting {lvc_steps} LVC thinking tokens into prompt...")

        lvc_token_sequence = torch.tensor(
            [[lvc_start_id] + [lvc_id] * lvc_steps + [lvc_end_id]],
            dtype=torch.long, device=device
        )

        num_lvc_tokens = lvc_steps + 2
        input_ids = torch.cat([input_ids, lvc_token_sequence], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, num_lvc_tokens), dtype=attention_mask.dtype, device=device)
        ], dim=1)

        if "mm_token_type_ids" in extra_kwargs and extra_kwargs["mm_token_type_ids"] is not None:
            extra_kwargs["mm_token_type_ids"] = torch.cat([
                extra_kwargs["mm_token_type_ids"],
                torch.zeros((1, num_lvc_tokens), dtype=extra_kwargs["mm_token_type_ids"].dtype, device=device)
            ], dim=1)

        print(f"  [lvc_reasoning] Prompt extended. Total length: {input_ids.shape[1]}")

    # Prefill
    need_hidden = (decoding_strategy == "forced_lvc")
    with torch.no_grad():
        fwd_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=need_hidden,
            lvc_mode_switch=None,
            last_position_hidden_state=None,
            **extra_kwargs,
        )
        # logits_to_keep is only supported by the GRPO forward
        if use_rl_forward:
            fwd_kwargs["logits_to_keep"] = 1
        outputs = model(**fwd_kwargs)

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    generated_ids = input_ids.clone()

    # V1-compatible forced LVC (hidden-state feedback)
    if decoding_strategy == "forced_lvc" and lvc_start_id > 0 and lvc_end_id > 0:
        print(f"  [forced_lvc] Running {lvc_steps} latent reasoning steps...")
        last_hidden = getattr(outputs, "last_position_hidden_state", None)
        if last_hidden is None:
            hs = getattr(outputs, "hidden_states", None)
            if hs is not None and len(hs) > 0:
                last_hidden = hs[-1][:, -1, :]

        lvc_start_token = torch.tensor([[lvc_start_id]], device=device)
        generated_ids = torch.cat([generated_ids, lvc_start_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=-1)

        with torch.no_grad():
            out = model(
                input_ids=lvc_start_token, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True, output_hidden_states=True,
                lvc_mode_switch=torch.tensor([False], device=device),
            )
        past_key_values = out.past_key_values
        last_hidden = getattr(out, "last_position_hidden_state", None)
        if last_hidden is None:
            hs = getattr(out, "hidden_states", None)
            if hs is not None and len(hs) > 0:
                last_hidden = hs[-1][:, -1, :]

        for _ in range(lvc_steps):
            dummy_token = torch.tensor([[lvc_id]], device=device)
            generated_ids = torch.cat([generated_ids, dummy_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=-1)

            with torch.no_grad():
                out = model(
                    input_ids=dummy_token, attention_mask=attention_mask,
                    past_key_values=past_key_values, use_cache=True, output_hidden_states=True,
                    lvc_mode_switch=torch.tensor([True], device=device),
                    last_position_hidden_state=last_hidden,
                )
            past_key_values = out.past_key_values
            last_hidden = getattr(out, "last_position_hidden_state", None)
            if last_hidden is None:
                hs = getattr(out, "hidden_states", None)
                if hs is not None and len(hs) > 0:
                    last_hidden = hs[-1][:, -1, :]

        lvc_end_token = torch.tensor([[lvc_end_id]], device=device)
        generated_ids = torch.cat([generated_ids, lvc_end_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=-1)

        with torch.no_grad():
            out = model(
                input_ids=lvc_end_token, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True, output_hidden_states=True,
                lvc_mode_switch=torch.tensor([False], device=device),
            )
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

    # Auto-regressive generation
    prompt_len = input_ids.shape[1]

    def _sample_next_token(logits_in):
        if do_sample and temperature > 0:
            scaled = logits_in / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled[indices_to_remove] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(logits_in, dim=-1, keepdim=True)

    for step in range(max_new_tokens):
        next_token = _sample_next_token(next_token_logits)
        next_token_id = next_token.item()

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token_id == eos_token_id:
            break

        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
        ], dim=-1)

        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
            )
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

    new_ids = generated_ids[0, prompt_len:]
    output_text = processor.tokenizer.decode(
        new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    return output_text


# ======================================================================
# Video preprocessing
# ======================================================================

def prepare_video_input(
    video_path, question, processor,
    fps=2.0, video_min_pixels=100352, video_max_pixels=602112,
):
    """Prepare video input for inference."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": VIDEO_LVC_SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path,
             "min_pixels": video_min_pixels, "max_pixels": video_max_pixels, "fps": fps},
            {"type": "text", "text": question},
        ]},
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    )

    if videos and isinstance(videos[0], tuple):
        video_tensors, video_metadatas = zip(*videos)
        video_tensors = list(video_tensors)
        video_metadatas = list(video_metadatas)
    else:
        video_tensors = videos
        video_metadatas = None

    inputs = processor(
        text=[prompt_text],
        images=images if images else None,
        videos=video_tensors if video_tensors else None,
        video_metadata=video_metadatas,
        do_resize=False,
        padding=True,
        return_tensors="pt",
        **(video_kwargs or {}),
    )

    return inputs


# ======================================================================
# Model loading
# ======================================================================

def load_model(model_path, device="auto", use_flash_attn=False, use_rl_forward=False):
    """Load a trained Video LVC model.

    Args:
        model_path: Path to the model checkpoint.
        device: Device to load the model on.
        use_flash_attn: Whether to use flash attention 2.
        use_rl_forward: If True, apply the GRPO forward monkey-patch.
                        Use this when loading GRPO-trained checkpoints.
    """
    print(f"Loading model from {model_path}...")
    print(f"  use_rl_forward={use_rl_forward}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    attn_impl = "flash_attention_2" if use_flash_attn and BROKEN_FLASH_ATTN_ERROR is None else "sdpa"

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, config=config, torch_dtype="auto",
            attn_implementation=attn_impl, device_map=device,
        )
    elif model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, config=config, torch_dtype="auto",
            trust_remote_code=True, attn_implementation=attn_impl, device_map=device,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype="auto",
            trust_remote_code=True, attn_implementation=attn_impl, device_map=device,
        )

    # Apply the appropriate forward monkey-patch
    if use_rl_forward:
        replace_with_grpo_forward(model=model, model_type=model_type)
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            import types as _types
            from src.model.forward_grpo import video_lvc_grpo_forward
            model.forward = _types.MethodType(video_lvc_grpo_forward, model)
    else:
        proj_head = None
        if hasattr(model, "lvc_proj_head"):
            proj_head = model.lvc_proj_head
        else:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None and hasattr(config, "text_config"):
                hidden_size = getattr(config.text_config, "hidden_size", None)
            proj_head = LVCProjectionHead(hidden_size, hidden_size).to(
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        replace_with_sft_forward(model=model, model_type=model_type, lvc_proj_head=proj_head)
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            import types as _types
            from src.model.forward_sft import video_lvc_sft_forward
            model.forward = _types.MethodType(video_lvc_sft_forward, model)

    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    # Add LVC special tokens
    special_tokens = ["<|lvc_start|>", "<|lvc|>", "<|lvc_latent_end|>", "<|lvc_end|>"]
    for tok in special_tokens:
        if tok not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens(tok, special_tokens=True)

    if not hasattr(config, "lvc_id") or config.lvc_id is None:
        config.lvc_id = processor.tokenizer.convert_tokens_to_ids("<|lvc|>")
        config.lvc_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_start|>")
        config.lvc_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_end|>")
        config.lvc_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvc_latent_end|>")

    print(f"Model loaded: {model_type}")
    print(f"  LVC token IDs: lvc={config.lvc_id}, start={config.lvc_start_id}, end={config.lvc_end_id}")

    return model, processor
