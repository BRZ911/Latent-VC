"""
Monkey-patched forward for Video LVC SFT (Stage 1).

Key design:
1. Recurrent latent cache: <|lvc|> positions use learned token embeddings
   (no teacher forcing). Hidden state feedback happens through attention.
2. Contrastive LVC Loss (InfoNCE): Hidden states at <|lvc|> positions are trained
   via contrastive learning to encode key-frame information.
3. LVC Projection Head: MLP projects LLM hidden states to visual embedding space.

Loss = CE(text) + lambda * InfoNCE(hidden[lvc], kf_embeds)
"""

import torch
import math
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.utils import is_torchdynamo_compiling
from src.constants import IGNORE_INDEX

from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


@dataclass
class VideoLVCSFTOutput(ModelOutput):
    """Output class for SFT forward."""
    loss: Optional[torch.FloatTensor] = None
    loss_lvc: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


class LVCProjectionHead(nn.Module):
    """MLP that projects LLM hidden states to the visual embedding space for contrastive loss."""

    def __init__(self, hidden_size, proj_size=None, dropout=0.1):
        super().__init__()
        if proj_size is None:
            proj_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, proj_size),
        )

    def forward(self, x):
        weight_dtype = next(self.parameters()).dtype
        x = x.to(dtype=weight_dtype)
        out = self.proj(x)
        return out.float()


def info_nce_loss(query, positive_keys, temperature=0.07, all_keys=None):
    """Compute InfoNCE contrastive loss.

    Args:
        query: [N, D] projected hidden states at <|lvc|> positions
        positive_keys: [N, D] corresponding KF visual embeddings
        temperature: softmax temperature
        all_keys: [M, D] optional larger set of negatives
    """
    query = F.normalize(query, dim=-1)
    positive_keys = F.normalize(positive_keys, dim=-1)

    if all_keys is not None:
        all_keys = F.normalize(all_keys, dim=-1)
    else:
        all_keys = positive_keys

    logits = torch.matmul(query, all_keys.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)

    if logits.shape[0] != logits.shape[1]:
        loss = F.cross_entropy(logits, labels.clamp(max=logits.shape[1] - 1))
    else:
        loss = F.cross_entropy(logits, labels)

    return loss


def replace_with_sft_forward(model=None, model_type="qwen2_5_vl", lvc_proj_head=None):
    """Replace the default forward with Video LVC SFT forward."""
    print("#" * 42)
    print("Activated Video LVC SFT forward")
    print(f"  model_type={model_type}")
    print(f"  Mode: Recurrent latent cache + InfoNCE contrastive loss")

    if model_type == "qwen2_5_vl":
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
        target_cls = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    elif model_type == "qwen3_5" and model is not None:
        target_cls = type(model)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    target_cls.forward = video_lvc_sft_forward

    if lvc_proj_head is not None:
        model.lvc_proj_head = lvc_proj_head
    elif not hasattr(model, "lvc_proj_head"):
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None and hasattr(model.config, "text_config"):
            hidden_size = getattr(model.config.text_config, "hidden_size", None)
        model.lvc_proj_head = LVCProjectionHead(hidden_size, hidden_size).to(
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )

    print(f"  Patched class: {target_cls.__name__}")
    print("#" * 42)


def video_lvc_sft_forward(
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
    lvc_tokens=None,
    kf_pixel_values=None,
    kf_image_grid_thw=None,
    lvc_mode_switch=None,
    last_position_hidden_state=None,
) -> Union[Tuple, VideoLVCSFTOutput]:
    """Forward pass for Video LVC SFT."""
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
        ratio = n_feats / expected_n_tokens if expected_n_tokens > 0 else 0
        sms = int(math.sqrt(ratio) + 0.5)
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
        if sms > 1 and sms * sms == int(ratio + 0.5):
            hidden_dim = feats.shape[-1]
            merged_parts = []
            offset = 0
            for i in range(grid_thw.shape[0]):
                t_i, h_i, w_i = grid_thw[i].tolist()
                n_tokens_i = t_i * h_i * w_i
                chunk = feats[offset:offset + n_tokens_i]
                h_new = h_i // sms
                w_new = w_i // sms
                chunk = chunk.view(t_i, h_i, w_i, hidden_dim)
                chunk = chunk[:, :h_new * sms, :w_new * sms, :]
                chunk = chunk.view(t_i, h_new, sms, w_new, sms, hidden_dim)
                chunk = chunk.mean(dim=(2, 4))
                merged_parts.append(chunk.reshape(-1, hidden_dim))
                offset += n_tokens_i
            feats = torch.cat(merged_parts, dim=0)
        return feats

    if inputs_embeds is None:
        embed_fn = None
        if hasattr(self, "model") and hasattr(self.model, "get_input_embeddings"):
            embed_fn = self.model.get_input_embeddings()
        elif hasattr(self, "get_input_embeddings"):
            embed_fn = self.get_input_embeddings()
        else:
            raise AttributeError("Cannot find input embedding layer.")
        inputs_embeds = embed_fn(input_ids)

    if last_position_hidden_state is not None and lvc_mode_switch is not None:
        inputs_embeds[lvc_mode_switch, -1, :] = last_position_hidden_state[lvc_mode_switch]

    # Process video visual tokens
    if pixel_values_videos is not None:
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
            raise ValueError(f"Video features mismatch: tokens={n_video_tokens}, features={n_video_features}")
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        video_mask_unsqueeze = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_unsqueeze, video_embeds)
        del video_embeds, video_mask_unsqueeze

    # Process image visual tokens
    if pixel_values is not None:
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
            raise ValueError(f"Image features mismatch: tokens={n_image_tokens}, features={n_image_features}")
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
        del image_embeds, image_mask_unsqueeze

    # Process key frame images -> KF embeddings (contrastive targets ONLY)
    kf_embeds_for_loss = None
    lvc_positions_info = None

    if kf_pixel_values is not None and lvc_tokens is not None and len(lvc_tokens) > 0:
        with torch.no_grad():
            kf_embeds = _get_vision_features(kf_pixel_values, kf_image_grid_thw)

        llm_hidden_size = inputs_embeds.shape[-1]
        kf_hidden_size = kf_embeds.shape[-1]

        if kf_hidden_size != llm_hidden_size:
            merger_mod = _find_merger()
            if merger_mod is not None:
                with torch.no_grad():
                    try:
                        kf_embeds = merger_mod(kf_embeds, grid_thw=kf_image_grid_thw)
                    except TypeError:
                        kf_embeds = merger_mod(kf_embeds)
                    if hasattr(kf_embeds, "last_hidden_state") and kf_embeds.last_hidden_state is not None:
                        kf_embeds = kf_embeds.last_hidden_state
                    if isinstance(kf_embeds, (list, tuple)):
                        kf_embeds = torch.cat(kf_embeds, dim=0)
            else:
                raise RuntimeError(f"KF dim ({kf_hidden_size}) != LLM dim ({llm_hidden_size}), no merger found.")

        kf_embeds = kf_embeds.detach()

        # Compute per-KF mean embedding as contrastive target
        # (average pooling over all tokens of each key frame)
        n_kf_total = kf_embeds.shape[0]
        n_kf_images = kf_image_grid_thw.shape[0]

        pre_merger_counts = kf_image_grid_thw.prod(dim=-1)
        pre_merger_total = pre_merger_counts.sum().item()

        if pre_merger_total == n_kf_total:
            kf_token_counts = pre_merger_counts
        else:
            sms_sq = pre_merger_total / n_kf_total if n_kf_total > 0 else 4
            kf_token_counts = (pre_merger_counts.float() / sms_sq).long()
            kf_token_counts[-1] = n_kf_total - kf_token_counts[:-1].sum()

        # Compute mean embedding per key frame (for contrastive loss)
        kf_mean_embeds = []
        offset = 0
        for i in range(n_kf_images):
            count = kf_token_counts[i].item()
            kf_chunk = kf_embeds[offset:offset + count]
            kf_mean_embeds.append(kf_chunk.mean(dim=0))
            offset += count
        kf_embeds_for_loss = torch.stack(kf_mean_embeds, dim=0)  # [num_kf, hidden_size]

        lvc_id = getattr(self.config, "lvc_id", -1)
        lvc_mask = input_ids == lvc_id
        batch_indices, seq_positions = torch.nonzero(lvc_mask, as_tuple=True)
        lvc_positions_info = (batch_indices, seq_positions, n_kf_images, kf_token_counts)

    # Dummy visual for DeepSpeed ZeRO-3
    # Skip during decode steps (past_key_values present, seq_len=1) — the
    # parameters are already gathered so the dummy forward is unnecessary.
    is_decode_step = past_key_values is not None and inputs_embeds.shape[1] == 1
    if pixel_values is None and pixel_values_videos is None and kf_pixel_values is None and not is_decode_step:
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

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        rope_deltas_cache = getattr(self, "rope_deltas", None)
        if rope_deltas_cache is None and hasattr(self, "model"):
            rope_deltas_cache = getattr(self.model, "rope_deltas", None)

        if (prefill_compiled_stage or prefill_noncompiled_stage) or rope_deltas_cache is None:
            _mm_token_type_ids = mm_token_type_ids
            if _mm_token_type_ids is None and input_ids is not None:
                _mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
                image_token_id = getattr(self.config, "image_token_id", None)
                if image_token_id is not None:
                    _mm_token_type_ids[input_ids == image_token_id] = 1
                video_token_id = getattr(self.config, "video_token_id", None)
                if video_token_id is not None:
                    _mm_token_type_ids[input_ids == video_token_id] = 2

            _rope_fn = getattr(self, "get_rope_index", None)
            if _rope_fn is None:
                _rope_fn = getattr(self.model, "get_rope_index", None)

            import inspect
            _rope_sig = inspect.signature(_rope_fn)
            _rope_kwargs = dict(image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=attention_mask)
            if "mm_token_type_ids" in _rope_sig.parameters:
                _rope_kwargs["mm_token_type_ids"] = _mm_token_type_ids
            if "second_per_grid_ts" in _rope_sig.parameters:
                _rope_kwargs["second_per_grid_ts"] = second_per_grid_ts

            position_ids, rope_deltas = _rope_fn(input_ids, **_rope_kwargs)

            if hasattr(self, "rope_deltas"):
                self.rope_deltas = rope_deltas
            elif hasattr(self, "model"):
                self.model.rope_deltas = rope_deltas
            rope_deltas_cache = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + rope_deltas_cache).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    # Language model forward
    language_model = getattr(self, "model", None)
    if language_model is None:
        raise AttributeError("Cannot find language model backbone.")

    outputs = language_model(
        input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, cache_position=cache_position,
    )
    del inputs_embeds

    hidden_states = outputs[0]
    lhs = getattr(outputs, "last_hidden_state", hidden_states)
    last_position_hidden_state = lhs[:, -1, :]
    logits = self.lm_head(hidden_states)

    # Compute losses
    loss_ce = None
    loss_lvc = None

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
        valid_label_mask = shift_labels != IGNORE_INDEX
        if valid_label_mask.any():
            loss_ce = loss_fct(shift_logits, shift_labels)
        else:
            loss_ce = shift_logits.sum() * 0.0
        del shift_logits, shift_labels, valid_label_mask

        # Contrastive LVC loss (InfoNCE)
        # IMPORTANT: Under DeepSpeed ZeRO-3, every trainable parameter must
        # participate in the forward/backward graph every step. We use a dummy
        # forward through lvc_proj_head (with zero-valued output) to ensure its
        # parameters get proper gradients even when no key frames are present.
        def _zero_lvc_loss_via_dummy_forward():
            """Produce a zero loss that still runs lvc_proj_head forward."""
            ph = getattr(self, "lvc_proj_head", None)
            if ph is not None:
                hidden_size = hidden_states.shape[-1]
                dummy_input = torch.zeros(1, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
                dummy_out = ph(dummy_input)
                return (dummy_out * 0.0).sum()
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32, requires_grad=True)

        if kf_embeds_for_loss is not None and lvc_positions_info is not None:
            batch_indices, seq_positions, n_kf_images, kf_token_counts = lvc_positions_info

            if len(batch_indices) > 0:
                lvc_hidden_all = hidden_states[batch_indices, seq_positions]
                lvc_mean_hidden = []
                offset = 0
                for i in range(n_kf_images):
                    n_lvc_for_kf = kf_token_counts[i].item()
                    if offset + n_lvc_for_kf <= lvc_hidden_all.shape[0]:
                        kf_hidden = lvc_hidden_all[offset:offset + n_lvc_for_kf]
                        lvc_mean_hidden.append(kf_hidden.mean(dim=0))
                        offset += n_lvc_for_kf
                    elif offset < lvc_hidden_all.shape[0]:
                        kf_hidden = lvc_hidden_all[offset:]
                        lvc_mean_hidden.append(kf_hidden.mean(dim=0))
                        offset = lvc_hidden_all.shape[0]
                        break

                if len(lvc_mean_hidden) > 0:
                    lvc_mean_hidden = torch.stack(lvc_mean_hidden, dim=0)
                    n_matched = min(lvc_mean_hidden.shape[0], kf_embeds_for_loss.shape[0])
                    proj_head = getattr(self, "lvc_proj_head", None)
                    if proj_head is not None:
                        projected = proj_head(lvc_mean_hidden[:n_matched])
                        targets = kf_embeds_for_loss[:n_matched].float()
                        if n_matched >= 2:
                            loss_lvc = info_nce_loss(projected, targets, temperature=getattr(self.config, "lvc_temperature", 0.07))
                        else:
                            loss_lvc = 1.0 - F.cosine_similarity(projected, targets, dim=-1).mean()
                    else:
                        loss_lvc = 1.0 - F.cosine_similarity(lvc_mean_hidden[:n_matched].float(), kf_embeds_for_loss[:n_matched].float(), dim=-1).mean()
                else:
                    loss_lvc = _zero_lvc_loss_via_dummy_forward()
            else:
                loss_lvc = _zero_lvc_loss_via_dummy_forward()
        else:
            loss_lvc = _zero_lvc_loss_via_dummy_forward()

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (None,) + output

    _model_ref = getattr(self, "model", self)
    return VideoLVCSFTOutput(
        loss_ce=loss_ce, loss_lvc=loss_lvc, logits=logits,
        past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self, "rope_deltas", getattr(_model_ref, "rope_deltas", None)),
        last_position_hidden_state=last_position_hidden_state,
    )
