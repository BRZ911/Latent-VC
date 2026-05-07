"""
Reward functions for Video LVC GRPO training.

Four reward dimensions:
1. Accuracy Reward: Compare model output with ground truth
2. Format Reward: Check <think>/<answer> tag compliance
3. Temporal Grounding Reward: Timestamp accuracy vs key frame times
4. Latent Cache Reward: Cosine similarity between latent hidden states and KF embeddings
"""

import os
import re
from datetime import datetime
from typing import List, Optional

try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

try:
    from rouge_score import rouge_scorer
    _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    _rouge_scorer = None


# ============================================================
# 1. Accuracy Reward
# ============================================================

def accuracy_reward(completions, assistant, task=None, **kwargs):
    """Check if the completion is correct (MCQ or free-form)."""
    contents = [completion[0]["content"] for completion in completions]
    solutions = [a["content"] if isinstance(a, dict) else a for a in assistant]
    tasks = task if task is not None else [None] * len(contents)

    rewards = []
    for content, sol, t in zip(contents, solutions, tasks):
        reward = 0.0

        content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else content.strip()

        sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        student_option = _extract_mcq_option(student_answer)
        gt_option = _extract_mcq_option(ground_truth)

        if student_option and gt_option:
            if student_option == gt_option:
                reward = 1.0
        elif gt_option is None:
            if student_answer.lower().strip() == ground_truth.lower().strip():
                reward = 1.0
            elif HAS_MATH_VERIFY and reward == 0.0:
                try:
                    parsed_answer = parse(student_answer)
                    if float(verify(parsed_answer, parse(ground_truth))) > 0:
                        reward = 1.0
                except Exception:
                    pass
            if reward == 0.0 and ground_truth.lower() in student_answer.lower():
                reward = 0.5
            if reward == 0.0 and HAS_ROUGE and _rouge_scorer is not None:
                try:
                    scores = _rouge_scorer.score(ground_truth, student_answer)
                    rouge_l = scores["rougeL"].fmeasure
                    if rouge_l > 0.5:
                        reward = rouge_l
                except Exception:
                    pass

        rewards.append(reward)
    return rewards


def _extract_mcq_option(text):
    """Extract MCQ option letter (A/B/C/D/E) from text."""
    text = text.strip()
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    match = re.search(r"(?:answer|option)\s*[:：]\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:answer|option)\s+(?:is|=)\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"^\(?([A-E])\)?[\.\):\s]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


# ============================================================
# 2. Format Reward
# ============================================================

def format_reward(completions, **kwargs):
    """Check if completion follows <think>/<answer> format."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in contents:
        has_think = bool(re.search(r"<think>.*?</think>", content, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", content, re.DOTALL))
        if has_think and has_answer:
            reward = 1.0
        elif has_answer:
            reward = 0.5
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


# ============================================================
# 3. Temporal Grounding Reward
# ============================================================

def temporal_grounding_reward(completions, key_frames=None, **kwargs):
    """Check if timestamps in reasoning match ground-truth key frame times."""
    contents = [completion[0]["content"] for completion in completions]

    if key_frames is None:
        return [0.0] * len(contents)

    rewards = []
    for content, kfs in zip(contents, key_frames):
        if not kfs or kfs is None:
            rewards.append(0.0)
            continue

        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue

        think_text = think_match.group(1)
        mentioned_times = _extract_timestamps(think_text)

        if not mentioned_times:
            rewards.append(0.0)
            continue

        gt_times = []
        for kf in kfs:
            if isinstance(kf, dict) and "time" in kf:
                gt_times.append(float(kf["time"]))

        if not gt_times:
            rewards.append(0.0)
            continue

        tolerance = 3.0
        matched = 0
        for gt_t in gt_times:
            for mt in mentioned_times:
                if abs(mt - gt_t) <= tolerance:
                    matched += 1
                    break

        reward = min(1.0, matched / len(gt_times))
        rewards.append(reward)
    return rewards


def _extract_timestamps(text):
    """Extract timestamp values (in seconds) from text."""
    timestamps = []
    patterns = [
        r"(?:at|around|approximately|~)\s*(\d+\.?\d*)\s*(?:s(?:ec(?:ond)?s?)?|seconds?)",
        r"(\d+\.?\d*)\s*(?:s(?:ec(?:ond)?s?)?)\b",
        r"(?:time|timestamp|t)\s*[:=]\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*(?:sec(?:ond)?s?)",
        r"\[(\d+\.?\d*)\]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                t = float(m)
                if 0.0 <= t <= 7200.0:
                    timestamps.append(t)
            except ValueError:
                pass
    return list(set(timestamps))


# ============================================================
# 4. Latent Cache Reward
# ============================================================

def latent_reasoning_reward(completions, lvc_hidden_states=None, kf_embeddings=None,
                            temperature=0.1, threshold=0.3, **kwargs):
    """Cosine similarity between latent token hidden states and KF visual embeddings."""
    import torch
    import torch.nn.functional as F

    n = len(completions)

    if lvc_hidden_states is None or kf_embeddings is None:
        return [0.0] * n

    rewards = []
    for i in range(n):
        lvc_h = lvc_hidden_states[i] if i < len(lvc_hidden_states) else None
        kf_e = kf_embeddings[i] if i < len(kf_embeddings) else None

        if lvc_h is None or kf_e is None:
            rewards.append(0.0)
            continue
        if not isinstance(lvc_h, torch.Tensor) or not isinstance(kf_e, torch.Tensor):
            rewards.append(0.0)
            continue
        if lvc_h.numel() == 0 or kf_e.numel() == 0:
            rewards.append(0.0)
            continue

        if lvc_h.dim() == 1:
            lvc_h = lvc_h.unsqueeze(0)
        if kf_e.dim() == 1:
            kf_e = kf_e.unsqueeze(0)

        lvc_mean = lvc_h.mean(dim=0, keepdim=True).float()
        kf_mean = kf_e.mean(dim=0, keepdim=True).float()

        cos_sim = F.cosine_similarity(lvc_mean, kf_mean, dim=-1).item()

        if cos_sim >= threshold:
            reward = (cos_sim - threshold) / (1.0 - threshold)
        else:
            reward = (cos_sim - threshold) / threshold

        rewards.append(float(reward))
    return rewards


# ============================================================
# Utility
# ============================================================

def get_reward_funcs():
    """Return the list of reward functions for GRPO training."""
    return [accuracy_reward, format_reward, temporal_grounding_reward, latent_reasoning_reward]


def get_reward_func_names():
    return ["accuracy_reward", "format_reward", "temporal_grounding_reward", "latent_reasoning_reward"]
