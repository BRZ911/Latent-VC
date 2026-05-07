"""
Data utility functions for Video Latent Visual Cache.
Handles token replacement, image/video loading, and sequence operations.
"""

import os
import re
import numpy as np
import torch
from PIL import Image

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    LVC_START_TOKEN,
    LVC_END_TOKEN,
    LVC_TOKEN,
    LVC_PLACEHOLDER,
)


def replace_image_tokens(input_string, is_video=False):
    """Replace LLAVA-style image/video tokens with Qwen-style tokens."""
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_video_tokens(input_string):
    """Replace LLAVA-style video tokens with Qwen-style tokens."""
    pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
    replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_lvc_tokens(input_string, lvc_token_idxs_list):
    """Replace <lvc> (or legacy <lvr>) placeholders with LVC special tokens."""
    # Support both new <lvc> and legacy <lvr> placeholder
    LEGACY_PLACEHOLDER = "<lvr>"
    placeholder = LVC_PLACEHOLDER
    if LVC_PLACEHOLDER not in input_string and LEGACY_PLACEHOLDER in input_string:
        placeholder = LEGACY_PLACEHOLDER

    pattern = r'\n?' + re.escape(placeholder) + r'\n?'
    if re.search(pattern, input_string):
        parts = input_string.split(placeholder)
        output_parts = [parts[0]] if parts[0] else []
        remaining_parts = parts[1:]

        for seg, idxs in zip(remaining_parts, lvc_token_idxs_list):
            num_tokens = len(idxs)
            replacement = LVC_START_TOKEN + LVC_TOKEN * num_tokens + LVC_END_TOKEN
            output_parts.append(replacement + seg)

        extra_start = len(lvc_token_idxs_list)
        for seg in remaining_parts[extra_start:]:
            replacement = LVC_START_TOKEN + LVC_TOKEN * 4 + LVC_END_TOKEN
            output_parts.append(replacement + seg)

        return "".join(output_parts)
    return input_string


def llava_to_openai_video_lvc(conversations, is_video=False, lvc_token_idxs_list=None):
    """Convert LLaVA-format conversations to OpenAI-format with LVC tokens."""
    if lvc_token_idxs_list is None:
        lvc_token_idxs_list = []

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        content = replace_image_tokens(conversation["value"], is_video=is_video)
        if lvc_token_idxs_list:
            content = replace_lvc_tokens(content, lvc_token_idxs_list)
        transformed_data.append({
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": content,
        })

    return transformed_data


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """Pad a list of sequences to the same length."""
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel, width, height):
    """Load and process a single image using qwen_vl_utils."""
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(messages)
    return image_input[0]


def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    """Load and process a video using qwen_vl_utils.

    Returns:
        video_tensor: [T, C, H, W] sampled & resized frames
        video_metadata: dict with per-video metadata
        video_kwargs: dict with fps info
    """
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps,
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    video_tensor, video_metadata = videos[0]
    return video_tensor, video_metadata, video_kwargs
