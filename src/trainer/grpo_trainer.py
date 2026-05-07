"""
Video LVC V2 GRPO Trainer.

Extends HuggingFace Trainer to implement Group Relative Policy Optimization (GRPO)
for Video Latent Visual Cache V2.

Key features:
1. Generates multiple completions per prompt
2. Computes four reward dimensions (accuracy, format, temporal grounding, latent cache)
3. Computes group-relative advantages
4. Optimizes policy with clipped surrogate loss
5. Handles multimodal inputs (video + key frames)
6. LVC teacher-forced replay for scoring
7. Extracts completion-region hidden states for latent cache reward
   (falls back to mean-pooled completion when <|lvc|> tokens are absent)

Based on TRL's GRPOTrainer architecture but adapted for:
- Qwen3.5 multimodal model
- Video inputs via qwen_vl_utils
- LVC special token handling
- Latent reasoning reward computation
"""

import os
import sys
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
from collections import defaultdict, deque
from collections.abc import Sized

import datasets
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader, Sampler

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSequenceClassification,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import (
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
)
from functools import partial
from transformers.trainer_utils import seed_worker

from trl.trainer.utils import selective_log_softmax
from trl import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.import_utils import is_deepspeed_available
from trl.trainer.callbacks import SyncRefModelCallback
from trl.extras.profiling import profiling_decorator, profiling_context
from trl.data_utils import maybe_apply_chat_template, is_conversational, apply_chat_template
from trl.trainer.utils import pad

from accelerate.utils import set_seed, gather, gather_object

from src.constants import MULTIMODAL_KEYWORDS

# Try qwen_vl_utils
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# ============================================================
# Utility functions
# ============================================================

def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """Compute std ignoring NaNs (1D tensor)."""
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1)
    return torch.sqrt(variance)


class RepeatSampler(Sampler):
    """Sampler that repeats indices for GRPO multi-generation."""

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# ============================================================
# Main Trainer
# ============================================================

class VideoLVCV2GRPOTrainer(Trainer):
    """
    GRPO Trainer for Video Latent Visual Cache V2.

    Implements:
    1. Multi-completion generation per prompt
    2. Four-dimensional reward computation
    3. Group-relative advantage normalization
    4. PPO-style clipped surrogate loss
    5. LVC teacher-forced replay for per-token logprob computation
    6. Latent reasoning reward via hidden state extraction
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model_pth: Optional[str],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
    ):
        if args is None:
            model_name = model.config._name_or_path.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_id = model.config._name_or_path

        # Reference model
        self.beta = args.beta
        self.ref_model_pth = ref_model_pth
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            from transformers import AutoConfig
            ref_model_config = AutoConfig.from_pretrained(self.ref_model_pth, trust_remote_code=True)
            compute_dtype = (
                torch.float16 if args.fp16
                else (torch.bfloat16 if args.bf16 else torch.float32)
            )
            model_type = getattr(ref_model_config, "model_type", None)
            if model_type == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration
                self.ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    self.ref_model_pth,
                    config=ref_model_config,
                    torch_dtype=compute_dtype,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
            elif model_type == "qwen2_5_vl":
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.ref_model_pth,
                    config=ref_model_config,
                    torch_dtype=compute_dtype,
                    attn_implementation="sdpa",
                )
            else:
                from transformers import AutoModelForCausalLM
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    self.ref_model_pth,
                    config=ref_model_config,
                    torch_dtype=compute_dtype,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                )
            # Apply RL forward patch to ref model too
            from src.model.forward_grpo import replace_with_grpo_forward
            replace_with_grpo_forward(model=self.ref_model, model_type=model_type or "qwen3_5")
        else:
            self.ref_model = create_reference_model(model)

        if processing_class is None:
            raise ValueError("Please pass the processor to VideoLVCV2GRPOTrainer")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Reward weights ({len(args.reward_weights)}) != reward functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Data collator (no-op for GRPO)
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = getattr(args, "min_p", None)
        self.repetition_penalty = args.repetition_penalty
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.shuffle_dataset = args.shuffle_dataset

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        maxlen = (
            self.accelerator.num_processes
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Validate batch size vs num_generations
        if self.num_generations < 2:
            raise ValueError(f"GRPO requires >= 2 generations, got {self.num_generations}")
        num_processes = self.accelerator.num_processes
        effective_bs = args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        if effective_bs % self.num_generations != 0:
            raise ValueError(
                f"Effective batch size ({effective_bs}) must be divisible by num_generations ({self.num_generations})"
            )

        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.tokenizer.pad_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _flush_metrics_to_logs(self, logs: dict[str, float], mode: str) -> dict[str, float]:
        """Merge buffered metric averages into the visible trainer log dict."""
        if mode not in self._metrics:
            return logs

        metric_store = self._metrics[mode]
        if not metric_store:
            return logs

        prefix = "eval_" if mode == "eval" else ""
        logs = dict(logs)

        for key, values in metric_store.items():
            if not values:
                continue
            mean_value = sum(values) / len(values)
            if key == "reward":
                logs[f"{prefix}reward/mean"] = mean_value
            elif key == "reward_std":
                logs[f"{prefix}reward/std"] = mean_value
            else:
                logs[f"{prefix}{key}"] = mean_value

        metric_store.clear()
        return logs

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        logs = self._flush_metrics_to_logs(logs, mode)
        super().log(logs, start_time=start_time)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1,  # generation now handles num_generations internally
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep,
                             batch_size=None, **multimodal_inputs) -> torch.Tensor:
        """Compute per-token log probabilities.

        To avoid materialising a huge ``(batch, full_seq, vocab_size)`` logits
        tensor (~100+ GiB), we communicate the desired ``logits_to_keep`` to
        the model forward via **two** channels:
        1. The ``logits_to_keep`` keyword argument (standard path).
        2. A model-instance attribute ``_logits_to_keep_override`` (fallback).

        The monkey-patched forward checks both and uses whichever is set,
        guaranteeing that lm_head only projects the last *N* hidden states
        regardless of how the call is dispatched (DeepSpeed, hooks, etc.).
        """
        # We need logits for completion tokens.  Because of the shift-by-1 in
        # next-token prediction we need (logits_to_keep + 1) positions.
        keep = logits_to_keep + 1 if logits_to_keep is not None else 0

        # Set the override attribute on the underlying model so our
        # monkey-patched forward can read it even if the kwarg is lost.
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped._logits_to_keep_override = keep

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=keep,
                **multimodal_inputs,
            )
            logits = outputs.logits  # (batch, keep_or_full, vocab)
            del outputs
        finally:
            # Always clean up the override
            unwrapped._logits_to_keep_override = None

        # Shift: predict next token
        logits = logits[:, :-1, :]       # (batch, keep-1, vocab)
        input_ids = input_ids[:, 1:]
        if logits_to_keep is not None:
            input_ids = input_ids[:, -logits_to_keep:]

        # In-place temperature scaling to avoid allocating a copy
        if self.temperature != 1.0:
            logits = logits.div_(self.temperature)
        logps = selective_log_softmax(logits, input_ids)
        del logits
        return logps

    def _replay_lvc_to_collect_states(
        self, model, prompt_ids, prompt_mask, prompt_completion_ids,
        multimodal_inputs, lvc_steps=10
    ):
        """
        Replay LVC generation under no_grad to collect hidden states
        at <|lvc|> positions for teacher-forced scoring.

        Returns:
            lvc_states: (B, C, H) hidden states at LVC positions
            lvc_mask: (B, C) boolean mask for LVC positions in completion
            had_lvc: list[bool] whether each sample had LVC tokens
            model_kwargs_after_lvc: dict of any extra info
        """
        device = prompt_completion_ids.device
        B = prompt_completion_ids.size(0)
        prompt_len = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_len:]
        comp_len = completion_ids.size(1)

        lvc_id = getattr(model.config, "lvc_id", -1)
        lvc_start_id = getattr(model.config, "lvc_start_id", -1)
        lvc_end_id = getattr(model.config, "lvc_end_id", -1)

        # Build LVC mask over completion tokens
        lvc_mask = torch.zeros(B, comp_len, dtype=torch.bool, device=device)
        had_lvc = []
        for b in range(B):
            active = False
            found_any = False
            for t in range(comp_len):
                tok = completion_ids[b, t].item()
                if tok == lvc_start_id:
                    active = True
                    found_any = True
                elif tok == lvc_end_id:
                    active = False
                if active and tok == lvc_id:
                    lvc_mask[b, t] = True
            had_lvc.append(found_any)

        # Run full forward to get hidden states
        with torch.no_grad():
            full_attention_mask = torch.cat([
                prompt_mask,
                torch.ones(B, comp_len, dtype=prompt_mask.dtype, device=device)
            ], dim=1)

            outputs = model(
                input_ids=prompt_completion_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                logits_to_keep=1,  # minimize logits memory; we only need hidden_states
                **multimodal_inputs,
            )
            # Get last hidden states
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]
            else:
                last_hidden = outputs.logits  # fallback

            # Extract completion part hidden states
            comp_hidden = last_hidden[:, prompt_len:, :]  # (B, C, H)

            # Extract LVC states where mask is True
            hidden_dim = comp_hidden.size(-1)
            lvc_states = torch.zeros(B, comp_len, hidden_dim, dtype=comp_hidden.dtype, device=device)
            lvc_states[lvc_mask] = comp_hidden[lvc_mask]

        return lvc_states, lvc_mask, had_lvc, {}

    def _extract_lvc_hidden_states(
        self, model, prompt_completion_ids, prompt_mask, multimodal_inputs, prompt_length
    ):
        """
        Extract hidden states from the completion region for latent cache reward.

        Instead of searching for <|lvc|> token positions (which the model may not
        generate during free generation), this method extracts the **mean-pooled
        hidden states over the completion region**. The completion hidden states
        represent the model's reasoning about the video content and can be
        meaningfully compared with key frame visual embeddings via cosine similarity.

        This provides a learning signal that encourages the model's reasoning
        representation to align with the visual content of key frames.

        Returns:
            list of tensors, one per batch item: [1, hidden_dim] or None
        """
        device = prompt_completion_ids.device
        B = prompt_completion_ids.size(0)
        comp_len = prompt_completion_ids.size(1) - prompt_length

        if comp_len <= 0:
            return [None] * B

        lvc_id = getattr(model.config, "lvc_id", -1)

        with torch.no_grad():
            full_attention_mask = torch.cat([
                prompt_mask,
                torch.ones(B, comp_len, dtype=prompt_mask.dtype, device=device)
            ], dim=1)

            outputs = model(
                input_ids=prompt_completion_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                logits_to_keep=1,  # minimize logits memory; we only need hidden_states
                **multimodal_inputs,
            )

            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
            else:
                # Can't extract hidden states — return None
                return [None] * B

        # Extract completion region hidden states
        completion_ids = prompt_completion_ids[:, prompt_length:]  # (B, comp_len)
        pad_token_id = getattr(self.processing_class.tokenizer, "pad_token_id", 0) or 0
        eos_token_id = self.processing_class.tokenizer.eos_token_id

        lvc_hidden_list = []
        for b in range(B):
            # First, try the original approach: look for <|lvc|> tokens
            if lvc_id >= 0:
                lvc_positions = (prompt_completion_ids[b] == lvc_id).nonzero(as_tuple=True)[0]
                if len(lvc_positions) > 0:
                    lvc_h = hidden[b, lvc_positions, :]  # (num_lvc, hidden_dim)
                    lvc_hidden_list.append(lvc_h)
                    continue

            # Fallback: mean-pool over non-padding completion tokens
            comp_hidden = hidden[b, prompt_length:, :]  # (comp_len, hidden_dim)
            comp_tok = completion_ids[b]  # (comp_len,)

            # Build mask: exclude padding and EOS tokens
            valid_mask = (comp_tok != pad_token_id)
            if eos_token_id is not None:
                valid_mask = valid_mask & (comp_tok != eos_token_id)

            num_valid = valid_mask.sum().item()
            if num_valid == 0:
                lvc_hidden_list.append(None)
                continue

            # Mean pool over valid completion tokens
            comp_mean = comp_hidden[valid_mask].mean(dim=0, keepdim=True)  # (1, hidden_dim)
            lvc_hidden_list.append(comp_mean)

        return lvc_hidden_list

    def _encode_key_frames(self, model, key_frames_list, kf_folder=None):
        """
        Encode key frame images through the vision encoder to get embeddings
        for the latent cache reward.

        IMPORTANT (ZeRO-3 safety): Under ZeRO-3, visual_module() triggers
        allgather to reconstruct parameters, which is a collective operation
        requiring ALL ranks to participate in lockstep.  The old per-sample
        loop called visual_module() a variable number of times per rank
        (depending on how many samples in each rank's local batch had valid
        key_frames), causing an allgather mismatch → NCCL timeout / deadlock.

        Fix: collect ALL valid key-frame images across the local batch first,
        then make exactly ONE batched call to visual_module() + merger so that
        every rank performs the same number of collective operations.  Ranks
        that have zero valid images still participate with a tiny dummy input
        so the allgather stays in sync.

        Args:
            model: the model with visual encoder
            key_frames_list: list of list of dict with "path" key
            kf_folder: optional root folder for key frame images

        Returns:
            list of tensors [mean_hidden_dim] or None per batch item
        """
        from PIL import Image

        unwrapped = self.accelerator.unwrap_model(model)
        visual_module = getattr(unwrapped, "visual", None)
        if visual_module is None and hasattr(unwrapped, "model"):
            visual_module = getattr(unwrapped.model, "visual", None)

        if visual_module is None:
            return [None] * len(key_frames_list)

        # ---- Phase 1: collect images for each batch item (CPU only) ----
        per_item_images = []  # list of list[PIL.Image]
        for kfs in key_frames_list:
            kf_images = []
            if kfs:
                for kf in kfs:
                    if isinstance(kf, dict):
                        kf_path = kf.get("path", "")
                    else:
                        kf_path = str(kf)

                    # Resolve path
                    if not os.path.isabs(kf_path) and kf_folder and kf_folder.strip():
                        kf_path = os.path.join(kf_folder, kf_path)
                    if not os.path.exists(kf_path) and not os.path.isabs(kf_path):
                        project_root = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "..", ".."))
                        kf_path = os.path.join(project_root, kf_path)

                    if os.path.exists(kf_path):
                        try:
                            kf_images.append(Image.open(kf_path).convert("RGB"))
                        except Exception:
                            pass
            per_item_images.append(kf_images)

        # Flatten all valid images into a single list for batched processing
        all_images = []
        # item_ranges[i] = (start_idx, end_idx) into all_images, or None
        item_ranges = []
        for imgs in per_item_images:
            if imgs:
                start = len(all_images)
                all_images.extend(imgs)
                item_ranges.append((start, len(all_images)))
            else:
                item_ranges.append(None)

        has_valid_images = len(all_images) > 0

        # ---- Phase 2: ONE batched call to visual_module (ZeRO-3 safe) ----
        # Under ZeRO-3 ALL ranks must call visual_module() the same number of
        # times.  The caller (_generate_and_score_completions) already did an
        # all_reduce to ensure ALL ranks enter this function, so we must make
        # exactly one visual_module() call.  If a rank has no valid images we
        # use a tiny 1-pixel dummy so the allgather still matches.

        try:
            if has_valid_images:
                kf_inputs = self.processing_class(
                    text=[""] * len(all_images),
                    images=all_images,
                    videos=None,
                    padding=False,
                    return_tensors="pt",
                )
            else:
                # Dummy single-pixel image so the collective stays in sync
                dummy_img = Image.new("RGB", (28, 28), color=(0, 0, 0))
                kf_inputs = self.processing_class(
                    text=[""],
                    images=[dummy_img],
                    videos=None,
                    padding=False,
                    return_tensors="pt",
                )

            if "pixel_values" not in kf_inputs:
                return [None] * len(key_frames_list)

            kf_pixel = kf_inputs["pixel_values"].to(
                device=next(visual_module.parameters()).device,
                dtype=next(visual_module.parameters()).dtype,
            )
            kf_grid = kf_inputs["image_grid_thw"].to(
                device=next(visual_module.parameters()).device,
            )

            with torch.no_grad():
                kf_embeds = visual_module(kf_pixel, grid_thw=kf_grid)
                if hasattr(kf_embeds, "last_hidden_state"):
                    kf_embeds = kf_embeds.last_hidden_state
                elif isinstance(kf_embeds, (list, tuple)):
                    kf_embeds = torch.cat(kf_embeds, dim=0)

                # Apply merger if available
                merger = getattr(visual_module, "merger", None)
                if merger is None:
                    merger = getattr(unwrapped, "merger",
                                    getattr(getattr(unwrapped, "model", None), "merger", None))
                if merger is not None:
                    try:
                        kf_embeds = merger(kf_embeds, grid_thw=kf_grid)
                    except TypeError:
                        kf_embeds = merger(kf_embeds)
                    if hasattr(kf_embeds, "last_hidden_state"):
                        kf_embeds = kf_embeds.last_hidden_state
                    if isinstance(kf_embeds, (list, tuple)):
                        kf_embeds = torch.cat(kf_embeds, dim=0)

        except Exception as e:
            print(f"Warning: Failed to encode key frames: {e}")
            return [None] * len(key_frames_list)

        # If we used a dummy image, discard the result
        if not has_valid_images:
            return [None] * len(key_frames_list)

        # ---- Phase 3: split embeddings back to per-item results ----
        # kf_embeds shape: (total_tokens, hidden_dim) — tokens from all images
        # We need to split by image using kf_grid (each row = one image's grid)
        # Each image contributes grid[0]*grid[1]*grid[2] tokens before merger,
        # but after merger the count may differ.  Use kf_grid to compute
        # per-image token counts.

        # Compute per-image token counts from grid_thw
        # After ViT: tokens_per_image = t * h * w
        # After merger (spatial_merge_size=s): tokens = t * (h/s) * (w/s)
        # We can infer from total tokens: total_tokens = kf_embeds.size(0)
        # and number of images = kf_grid.size(0)
        n_images = kf_grid.size(0)
        total_tokens = kf_embeds.size(0)

        if n_images == 1:
            tokens_per_image = [total_tokens]
        else:
            # Estimate tokens per image from grid_thw
            # After merger with spatial_merge_size, each image has
            # t * ceil(h/s) * ceil(w/s) tokens.  Since we don't know s exactly,
            # compute proportional to t*h*w and distribute total_tokens.
            raw_counts = []
            for gi in range(n_images):
                t, h, w = kf_grid[gi].tolist()
                raw_counts.append(int(t * h * w))
            raw_total = sum(raw_counts)
            if raw_total > 0:
                tokens_per_image = []
                assigned = 0
                for i, rc in enumerate(raw_counts):
                    if i == n_images - 1:
                        tokens_per_image.append(total_tokens - assigned)
                    else:
                        n_tok = round(total_tokens * rc / raw_total)
                        tokens_per_image.append(n_tok)
                        assigned += n_tok
            else:
                # Fallback: equal split
                tokens_per_image = [total_tokens // n_images] * n_images
                tokens_per_image[-1] = total_tokens - sum(tokens_per_image[:-1])

        # Map image embeddings back to batch items
        embeddings_list = []
        img_idx = 0
        token_offset = 0
        for i, rng in enumerate(item_ranges):
            if rng is None:
                embeddings_list.append(None)
            else:
                start_img, end_img = rng
                n_imgs_for_item = end_img - start_img
                # Gather all tokens for this item's images
                item_token_start = token_offset
                for j in range(n_imgs_for_item):
                    token_offset += tokens_per_image[img_idx]
                    img_idx += 1
                item_token_end = token_offset

                if item_token_end > item_token_start:
                    item_embeds = kf_embeds[item_token_start:item_token_end]
                    kf_mean = item_embeds.mean(dim=0)  # (hidden_dim,)
                    embeddings_list.append(kf_mean.detach())
                else:
                    embeddings_list.append(None)

        return embeddings_list

    @profiling_decorator
    def _generate_and_score_completions(self, inputs):
        """Generate completions and compute rewards."""
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # ============================================================
        # Repeat each prompt `num_generations` times BEFORE processing
        # so that the processor correctly generates pixel_values and
        # video_grid_thw for G copies of each video.
        # E.g., if batch has prompts [A, B] and G=4:
        #   prompts becomes [A, A, A, A, B, B, B, B]
        # Each copy will get a different sample from generate() (do_sample=True).
        # ============================================================
        num_unique_prompts = len(prompts)
        prompts_repeated = [p for p in prompts for _ in range(self.num_generations)]
        prompts_text_repeated = [p for p in prompts_text for _ in range(self.num_generations)]

        # Process vision info from the repeated prompts
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                prompts_repeated, return_video_kwargs=True
            )
        except Exception as e:
            print(f"Warning: process_vision_info failed: {e}")
            image_inputs, video_inputs, video_kwargs = None, None, {}

        # Validate that vision inputs are not empty/malformed
        if video_inputs is not None:
            try:
                if isinstance(video_inputs, (list, tuple)) and len(video_inputs) == 0:
                    video_inputs = None
            except Exception:
                video_inputs = None
        if image_inputs is not None:
            try:
                if isinstance(image_inputs, (list, tuple)) and len(image_inputs) == 0:
                    image_inputs = None
            except Exception:
                image_inputs = None

        # Build proc_kwargs for the processing_class call.
        proc_kwargs = {}
        if video_kwargs:
            fps_list = video_kwargs.pop("fps", None)
            proc_kwargs.update(video_kwargs)

            if video_inputs is not None and fps_list and isinstance(fps_list, list):
                video_metadata = []
                for i, vid in enumerate(video_inputs):
                    nframes = vid.shape[0] if hasattr(vid, "shape") else len(vid)
                    sample_fps = fps_list[i] if i < len(fps_list) else fps_list[-1]
                    video_metadata.append({
                        "total_num_frames": nframes,
                        "fps": sample_fps,
                        "frames_indices": list(range(nframes)),
                    })
                proc_kwargs["video_metadata"] = video_metadata

        prompt_inputs = self.processing_class(
            text=prompts_text_repeated,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            **proc_kwargs,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Hard sequence-length cap to avoid conv1d 32-bit index overflow:
        # Qwen3.5 linear_attn conv1d processes (batch, 8192, seq_len) tensors.
        # With batch=8 and bf16, the tensor byte count is 8*8192*seq*2.
        # int32 cap ≈ 2.1B → max seq per sample ≈ 16384.  Use conservative
        # limit of 8192 to leave headroom.
        HARD_SEQ_LIMIT = 8192
        trunc_len = min(
            self.max_prompt_length if self.max_prompt_length is not None else HARD_SEQ_LIMIT,
            HARD_SEQ_LIMIT,
        )
        cur_len = prompt_ids.size(1)
        if cur_len > trunc_len:
            # Count vision tokens BEFORE truncation
            vid_token_id = getattr(self.model.config, "video_token_id", None)
            img_token_id = getattr(self.model.config, "image_token_id", None)
            n_vid_before = int((prompt_ids == vid_token_id).sum()) if vid_token_id is not None else 0
            n_img_before = int((prompt_ids == img_token_id).sum()) if img_token_id is not None else 0

            # Truncate from left (keep most-recent tokens)
            prompt_ids = prompt_ids[:, -trunc_len:]
            prompt_mask = prompt_mask[:, -trunc_len:]

            # Check if any vision tokens were removed by truncation
            n_vid_after = int((prompt_ids == vid_token_id).sum()) if vid_token_id is not None else 0
            n_img_after = int((prompt_ids == img_token_id).sum()) if img_token_id is not None else 0
            if n_vid_before != n_vid_after:
                # Remove video inputs — forward will skip video embedding
                prompt_inputs.pop("pixel_values_videos", None)
                prompt_inputs.pop("video_grid_thw", None)
                # Also replace residual video-token-ids with pad token so
                # forward doesn't try to masked_scatter with zero features
                pad_id = getattr(self.processing_class.tokenizer, "pad_token_id", 0) or 0
                if n_vid_after > 0:
                    prompt_ids = prompt_ids.clone()
                    prompt_ids[prompt_ids == vid_token_id] = pad_id
            if n_img_before != n_img_after:
                prompt_inputs.pop("pixel_values", None)
                prompt_inputs.pop("image_grid_thw", None)
                pad_id = getattr(self.processing_class.tokenizer, "pad_token_id", 0) or 0
                if n_img_after > 0:
                    prompt_ids = prompt_ids.clone()
                    prompt_ids[prompt_ids == img_token_id] = pad_id

            # Update dict entries for generate()
            prompt_inputs["input_ids"] = prompt_ids
            prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions (each copy of the same prompt → different sample)
        # IMPORTANT:
        # 1. Switch to eval() so GradientCheckpointingLayer doesn't nullify
        #    past_key_values during decode.
        # 2. Explicitly disable gradient_checkpointing so that use_cache=True
        #    actually works (otherwise transformers forces use_cache=False and
        #    every decode step re-processes the full sequence — extremely slow).
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            unwrapped_model.eval()
            # Disable gradient checkpointing for generation — must also
            # force-set the attribute on ALL sub-modules because the
            # @can_return_tuple decorator in transformers checks
            # `self.gradient_checkpointing and self.training` on each
            # sub-module individually.  If any sub-module still has
            # gradient_checkpointing=True the decorator sets use_cache=False,
            # disabling KV-cache and making generation ~100x slower.
            if hasattr(unwrapped_model, "gradient_checkpointing_disable"):
                unwrapped_model.gradient_checkpointing_disable()
            for module in unwrapped_model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = False
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config
            )
            # Re-enable gradient checkpointing and training mode
            if self.args.gradient_checkpointing:
                if hasattr(unwrapped_model, "gradient_checkpointing_enable"):
                    unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.train()

        # Extract prompt and completion — use the (possibly truncated)
        # prompt_inputs["input_ids"] length.
        prompt_length = prompt_inputs["input_ids"].size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        prompt_mask = prompt_inputs["attention_mask"]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Sanitize completion: the model may freely generate vision placeholder
        # tokens (video_token_id / image_token_id) during auto-regressive
        # decoding.  These spurious tokens would cause a mismatch between the
        # number of vision tokens in input_ids and the number of features from
        # the vision encoder during the scoring (log-prob) pass, leading to:
        #   ValueError: Video tokens mismatch: tokens=N+k, features=N
        # Fix: replace any vision placeholder tokens in the *completion* region
        # with pad_token_id so they are treated as normal text tokens.
        vid_token_id = getattr(self.model.config, "video_token_id", None)
        img_token_id = getattr(self.model.config, "image_token_id", None)
        pad_id = getattr(self.processing_class.tokenizer, "pad_token_id", 0) or 0
        _completion_dirty = False
        if vid_token_id is not None and (completion_ids == vid_token_id).any():
            completion_ids = completion_ids.clone()
            completion_ids[completion_ids == vid_token_id] = pad_id
            _completion_dirty = True
        if img_token_id is not None and (completion_ids == img_token_id).any():
            if not _completion_dirty:
                completion_ids = completion_ids.clone()
            completion_ids[completion_ids == img_token_id] = pad_id
            _completion_dirty = True
        if _completion_dirty:
            # Rebuild prompt_completion_ids with sanitized completions
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # EOS masking
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()

        # Build LVC mask (exclude LVC tokens from policy loss)
        # Safety: if <|lvc_start|> appears without matching <|lvc_end|>, only mask
        # tokens that are truly between a matched start/end pair. An unmatched
        # <|lvc_start|> without <|lvc_end|> should NOT mask all subsequent tokens,
        # otherwise final_mask becomes all-zero and loss=0 / grad_norm=0.
        lvc_start_id = getattr(self.model.config, "lvc_start_id", -1)
        lvc_end_id = getattr(self.model.config, "lvc_end_id", -1)
        lvc_mask_full = torch.ones_like(prompt_completion_ids, dtype=torch.bool)
        for b in range(prompt_completion_ids.size(0)):
            # First pass: find matched (start, end) pairs
            seq = prompt_completion_ids[b]
            starts = []
            pairs = []
            for t in range(seq.size(0)):
                tok = seq[t].item()
                if tok == lvc_start_id:
                    starts.append(t)
                elif tok == lvc_end_id and starts:
                    s = starts.pop(-1)
                    pairs.append((s, t))
            # Only mask tokens within matched pairs (inclusive of start, exclusive of end)
            for s, e in pairs:
                for t in range(s, e):
                    lvc_mask_full[b, t] = False

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        final_mask = attention_mask.bool() & lvc_mask_full

        # Safety check: if final_mask has no True tokens in the completion region
        # for any sample, fall back to completion_mask to avoid loss=0
        logits_to_keep_check = completion_ids.size(1)
        comp_final_mask = final_mask[:, -logits_to_keep_check:]
        empty_samples = comp_final_mask.sum(-1) == 0
        if empty_samples.any():
            # For samples with all-zero final_mask, use completion_mask instead
            for b in range(empty_samples.size(0)):
                if empty_samples[b]:
                    final_mask[b, -logits_to_keep_check:] = completion_mask[b].bool()

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        # Only include multimodal keys that actually have non-None values.
        # Passing None explicitly can shadow forward() default parameters.
        multimodal_inputs = {
            k: prompt_inputs[k]
            for k in MULTIMODAL_KEYWORDS
            if k in prompt_inputs and prompt_inputs[k] is not None
        }

        # Compute per-token logps
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps * final_mask[:, -logits_to_keep:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size, **multimodal_inputs
                )
                ref_per_token_logps = ref_per_token_logps * final_mask[:, -logits_to_keep:]
            else:
                ref_per_token_logps = None

        # Decode completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts_repeated, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = [[{"role": "assistant", "content": ct}] for ct in completions_text]

        # ============================================================
        # Compute rewards
        # ============================================================
        num_samples = prompt_completion_ids.size(0)  # = num_unique_prompts * num_generations
        rewards_per_func = torch.zeros(num_samples, len(self.reward_funcs), device=device)

        # Prepare reward kwargs — repeat to match num_generations
        assistants = [x.get("assistant", {"content": ""}) for x in inputs for _ in range(self.num_generations)]
        tasks = [x.get("task", "") for x in inputs for _ in range(self.num_generations)]
        key_frames_all = [x.get("key_frames", None) for x in inputs for _ in range(self.num_generations)]

        # Latent reasoning reward: extract hidden states and KF embeddings
        lvc_hidden_states = None
        kf_embeddings = None

        # CRITICAL: Under ZeRO-3, ALL ranks must participate in model forward calls
        # because allgather is used to reconstruct parameters. If only some ranks
        # call model(), the allgather operations will mismatch and cause NCCL timeout.
        # Therefore we use an all_reduce to check if ANY rank has key frames,
        # and if so, ALL ranks must enter _extract_lvc_hidden_states (which calls model()).
        has_any_kf_local = any(kf is not None and len(kf) > 0 for kf in key_frames_all)
        has_any_kf_tensor = torch.tensor([1 if has_any_kf_local else 0], dtype=torch.long, device=device)
        if self.accelerator.num_processes > 1:
            torch.distributed.all_reduce(has_any_kf_tensor, op=torch.distributed.ReduceOp.MAX)
        has_any_kf_global = has_any_kf_tensor.item() > 0

        if has_any_kf_global:
            try:
                # All ranks participate in this forward pass to keep ZeRO-3 allgather in sync
                lvc_hidden_states = self._extract_lvc_hidden_states(
                    self.model, prompt_completion_ids, prompt_mask,
                    multimodal_inputs, prompt_length
                )
                kf_folder = getattr(self.args, "kf_folder", None) or os.environ.get("VIDEO_LVC_KF_ROOT", "")
                kf_embeddings = self._encode_key_frames(
                    self.model, key_frames_all, kf_folder=kf_folder
                )
            except Exception as e:
                print(f"Warning: LVC hidden state extraction failed: {e}")
                lvc_hidden_states = None
                kf_embeddings = None

        for i, (reward_func, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    # Neural reward model
                    texts = [p + c for p, c in zip(prompts_text_repeated, completions_text)]
                    reward_inputs = self.processing_class.tokenizer(
                        text=texts, return_tensors="pt", padding=True,
                        padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    # Function-based reward
                    reward_kwargs = {
                        "assistant": assistants,
                        "task": tasks,
                        "key_frames": key_frames_all,
                        "lvc_hidden_states": lvc_hidden_states,
                        "kf_embeddings": kf_embeddings,
                    }
                    try:
                        output = reward_func(completions=completions, **reward_kwargs)
                    except TypeError:
                        # Some reward funcs don't accept all kwargs
                        try:
                            output = reward_func(completions=completions, assistant=assistants)
                        except TypeError:
                            output = reward_func(completions=completions)
                    output = [r if r is not None else torch.nan for r in output]
                    rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = gather(rewards_per_func)

        # Weighted sum of rewards
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Group-relative normalization
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice for local process
        # Each process has num_samples = num_unique_prompts * num_generations completions
        process_slice = slice(
            self.accelerator.process_index * num_samples,
            (self.accelerator.process_index + 1) * num_samples,
        )
        advantages = advantages[process_slice]

        # Log metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(
                attention_mask.sum()
            ).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())

        for i, name in enumerate(self.reward_func_names):
            mean_r = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{name}/mean"].append(mean_r)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text_repeated))
        self._textual_logs["completion"].extend(gather_object(completions_text))

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "final_mask": final_mask[:, -logits_to_keep:],
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "multimodal_inputs": multimodal_inputs,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        """Compute the GRPO clipped surrogate loss."""
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        final_mask = inputs["final_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        old_per_token_logps = inputs["old_per_token_logps"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        device = prompt_ids.device
        logits_to_keep = completion_ids.size(1)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        per_token_logps = self._get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            self.args.per_device_train_batch_size,
            **multimodal_inputs,
        )

        per_token_logps = per_token_logps * final_mask

        if self.num_iterations > 1:
            old_logps = old_per_token_logps * final_mask
        else:
            old_logps = per_token_logps.detach()

        adv = advantages.to(device).unsqueeze(1)

        coef_1 = torch.exp(per_token_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * adv
        per_token_loss2 = coef_2 * adv
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # KL penalty
        if self.beta != 0.0 and ref_per_token_logps is not None:
            ref_logps = ref_per_token_logps * final_mask
            per_token_kl = (
                torch.exp(ref_logps - per_token_logps)
                - (ref_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Aggregate
        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * final_mask).sum(-1)
                / final_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * final_mask).sum() / final_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * final_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _prepare_inputs(self, inputs):
        return inputs
