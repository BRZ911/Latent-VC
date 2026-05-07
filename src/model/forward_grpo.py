"""
Monkey-patched forward for Video LVC GRPO (Stage 2).

Key differences from SFT forward:
1. No contrastive LVC loss — rewards handle the learning signal
2. Supports lvc_mask + lvc_states for teacher-forced LVC replay
3. Supports prompt_length for separating prompt/completion embeddings
4. Vision encoder is called but frozen (no grad)
"""

import torch
import math
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from src.constants import IGNORE_INDEX

from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class VideoLVCGRPOOutput(ModelOutput):
    """Output class for GRPO forward."""
    loss: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


def replace_with_grpo_forward(model=None, model_type="qwen3_5"):
    """Replace the default forward with Video LVC GRPO forward."""
    print("#" * 42)
    print("Activated Video LVC GRPO forward")
    print(f"  model_type={model_type}")

    if model_type == "qwen2_5_vl":
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
        target_cls = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    elif model_type == "qwen3_5" and model is not None:
        target_cls = type(model)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    target_cls.forward = video_lvc_grpo_forward

    print(f"  Patched class: {target_cls.__name__}")
    print("#" * 42)


def video_lvc_grpo_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    mm_token_type_ids=None,
    logits_to_keep=0,
    lvc_mode_switch=None,
    last_position_hidden_state=None,
    lvc_mask=None,
    lvc_states=None,
    prompt_length=None,
) -> Union[Tuple, VideoLVCGRPOOutput]:
    """Forward pass for Video LVC GRPO stage."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    def _get_vision_features(pixel_tensor, grid_thw):
        visual_module = getattr(self, "visual", None)
        if visual_module is None and hasattr(self, "model"):
            visual_module = getattr(self.model, "visual", None)
        if visual_module is not None:
            feats = visual_module(pixel_tensor, grid_thw=grid_thw)
        elif hasattr(self, "get_image_features"):
            feats = self.get_image_features(pixel_tensor, grid_thw)
        else:
            raise AttributeError("Cannot find visual module.")
        if hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            feats = feats.last_hidden_state
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        if isinstance(feats, (list, tuple)):
            feats = torch.cat(feats, dim=0)
        return feats

    def _find_merger():
        visual_module = getattr(self, "visual", None)
        if visual_module is None and hasattr(self, "model"):
            visual_module = getattr(self.model, "visual", None)
        for parent in [visual_module, self, getattr(self, "model", None)]:
            if parent is not None:
                m = getattr(parent, "merger", None)
                if m is not None and callable(m):
                    return m
        return None

    def _maybe_apply_merger(feats, grid_thw, expected_n_tokens):
        n_feats = feats.shape[0]
        if n_feats == expected_n_tokens:
            return feats
        merger_mod = _find_merger()
        if merger_mod is not None:
            try:
                feats = merger_mod(feats, grid_thw=grid_thw)
            except TypeError:
                feats = merger_mod(feats)
            if hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
                feats = feats.last_hidden_state
            if isinstance(feats, (list, tuple)):
                feats = torch.cat(feats, dim=0)
        return feats

    # Get input embeddings
    if inputs_embeds is None:
        embed_fn = None
        if hasattr(self, "model") and hasattr(self.model, "get_input_embeddings"):
            embed_fn = self.model.get_input_embeddings()
        elif hasattr(self, "get_input_embeddings"):
            embed_fn = self.get_input_embeddings()
        else:
            raise AttributeError("Cannot find input embedding layer.")
        inputs_embeds = embed_fn(input_ids)

    # Generation: LVC mode switch
    if last_position_hidden_state is not None and lvc_mode_switch is not None:
        inputs_embeds[lvc_mode_switch, -1, :] = last_position_hidden_state[lvc_mode_switch]

    # Teacher-forced LVC replay
    if lvc_states is not None and lvc_mask is not None and prompt_length is not None:
        comp_embeds = inputs_embeds[:, prompt_length:, :]
        comp_embeds = torch.where(lvc_mask.unsqueeze(-1), lvc_states, comp_embeds)
        inputs_embeds = torch.cat([inputs_embeds[:, :prompt_length, :], comp_embeds], dim=1)

    # Process video visual tokens
    if pixel_values_videos is not None and video_grid_thw is not None:
        with torch.no_grad():
            video_embeds = _get_vision_features(pixel_values_videos, video_grid_thw)
        video_embeds = video_embeds.detach()
        video_mask = input_ids == self.config.video_token_id
        n_video_tokens = video_mask.sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            video_embeds = _maybe_apply_merger(video_embeds, video_grid_thw, n_video_tokens)
            n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(f"Video tokens mismatch: tokens={n_video_tokens}, features={n_video_features}")
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        video_mask_unsqueeze = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_unsqueeze, video_embeds)
        del video_embeds, video_mask_unsqueeze
    elif pixel_values_videos is not None and video_grid_thw is None:
        pixel_values_videos = None

    # Process image visual tokens
    if pixel_values is not None and image_grid_thw is not None:
        with torch.no_grad():
            image_embeds = _get_vision_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.detach()
        image_mask = input_ids == self.config.image_token_id
        n_image_tokens = image_mask.sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            image_embeds = _maybe_apply_merger(image_embeds, image_grid_thw, n_image_tokens)
            n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(f"Image tokens mismatch: tokens={n_image_tokens}, features={n_image_features}")
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
        del image_embeds, image_mask_unsqueeze

    # Dummy visual for DeepSpeed ZeRO-3
    is_decode_step = past_key_values is not None and inputs_embeds.shape[1] == 1
    if pixel_values is None and pixel_values_videos is None and not is_decode_step:
        visual_module = getattr(self, "visual", None)
        if visual_module is None and hasattr(self, "model"):
            visual_module = getattr(self.model, "visual", None)
        if visual_module is not None:
            vision_param = next(visual_module.parameters(), None)
            if vision_param is not None:
                vision_device = vision_param.device
                vision_dtype = vision_param.dtype
            else:
                vision_device = inputs_embeds.device
                vision_dtype = inputs_embeds.dtype
            patch_embed = getattr(visual_module, "patch_embed", None)
            if patch_embed is not None:
                in_channels = int(getattr(patch_embed, "in_channels", 3))
                temporal_patch = int(getattr(patch_embed, "temporal_patch_size", 2))
                patch_size = int(getattr(patch_embed, "patch_size", 14))
            else:
                vision_cfg = getattr(self.config, "vision_config", None)
                in_channels = int(getattr(vision_cfg, "in_channels", 3))
                temporal_patch = int(getattr(vision_cfg, "temporal_patch_size", 2))
                patch_size = int(getattr(vision_cfg, "patch_size", 14))
            dummy_t, dummy_h, dummy_w = 1, 28, 28
            patch_dim = in_channels * temporal_patch * patch_size * patch_size
            dummy_pixel = torch.zeros((dummy_t * dummy_h * dummy_w, patch_dim), device=vision_device, dtype=vision_dtype)
            dummy_grid = torch.tensor([[dummy_t, dummy_h, dummy_w]], device=vision_device, dtype=torch.long)
            dummy_embeds = visual_module(dummy_pixel, grid_thw=dummy_grid)
            if hasattr(dummy_embeds, "last_hidden_state") and dummy_embeds.last_hidden_state is not None:
                dummy_embeds = dummy_embeds.last_hidden_state
            elif isinstance(dummy_embeds, (list, tuple)):
                dummy_embeds = dummy_embeds[0]
            merger_mod = getattr(visual_module, "merger", None)
            if merger_mod is None:
                merger_mod = getattr(self, "merger", getattr(getattr(self, "model", None), "merger", None))
            if merger_mod is not None:
                try:
                    dummy_merged = merger_mod(dummy_embeds, grid_thw=dummy_grid)
                    if hasattr(dummy_merged, "last_hidden_state"):
                        dummy_merged = dummy_merged.last_hidden_state
                    inputs_embeds += dummy_merged.mean() * 0
                except Exception:
                    inputs_embeds += dummy_embeds.mean() * 0
            else:
                inputs_embeds += dummy_embeds.mean() * 0

    # Compute position IDs
    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    model_core = getattr(self, "model", None)
    if model_core is None:
        raise AttributeError("Cannot find model core.")

    if position_ids is None:
        is_decode = past_key_values is not None and inputs_embeds.shape[1] == 1
        batch_size_pos = inputs_embeds.shape[0]
        seq_length_pos = inputs_embeds.shape[1]

        if is_decode:
            if attention_mask is not None:
                pos_val = attention_mask.long().sum(dim=-1, keepdim=True) - 1
            else:
                kv_len = past_key_values[0][0].shape[2] if isinstance(past_key_values, (list, tuple)) else past_key_values.get_seq_length()
                pos_val = torch.full((batch_size_pos, 1), kv_len, device=inputs_embeds.device, dtype=torch.long)
            position_ids = pos_val.unsqueeze(0).expand(3, -1, -1)
        else:
            position_ids = None
            compute_3d_fn = getattr(model_core, "compute_3d_position_ids", None)
            if callable(compute_3d_fn):
                try:
                    position_ids = compute_3d_fn(
                        input_ids=input_ids, inputs_embeds=inputs_embeds,
                        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask, past_key_values=past_key_values,
                    )
                except (ValueError, IndexError, RuntimeError):
                    position_ids = None
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                    position_ids = position_ids.view(1, batch_size_pos, -1).expand(3, -1, -1)
                else:
                    position_ids = torch.arange(seq_length_pos, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size_pos, -1)

    # Language model forward
    language_model = getattr(model_core, "language_model", None)
    if language_model is None:
        raise AttributeError("Cannot find language model backbone.")

    outputs = language_model(
        input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
        cache_position=cache_position, output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    del inputs_embeds

    last_layer_hidden = outputs[0]
    lhs = getattr(outputs, "last_hidden_state", last_layer_hidden)
    last_position_hidden_state = lhs[:, -1, :]

    effective_logits_to_keep = logits_to_keep
    override = getattr(self, "_logits_to_keep_override", None)
    if override is not None and isinstance(override, int) and override > 0:
        effective_logits_to_keep = override
    if isinstance(effective_logits_to_keep, int) and effective_logits_to_keep > 0:
        logits = self.lm_head(last_layer_hidden[:, -effective_logits_to_keep:, :])
    else:
        logits = self.lm_head(last_layer_hidden)
    del last_layer_hidden

    # CE loss (optional)
    loss_ce = None
    if labels is not None:
        logits_float = logits.float()
        shift_logits = logits_float[..., :-1, :].contiguous()
        del logits_float
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None and hasattr(self.config, "text_config"):
            vocab_size = getattr(self.config.text_config, "vocab_size", None)
        if vocab_size is None:
            vocab_size = shift_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        lvc_id = getattr(self.config, "lvc_id", -1)
        shift_labels = shift_labels.masked_fill(shift_labels == lvc_id, IGNORE_INDEX)
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)
        del shift_logits, shift_labels

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss_ce,) + output if loss_ce is not None else output

    _model_ref = getattr(self, "model", self)
    return VideoLVCGRPOOutput(
        loss_ce=loss_ce, logits=logits,
        past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self, "rope_deltas", getattr(_model_ref, "rope_deltas", None)),
        last_position_hidden_state=last_position_hidden_state,
    )
