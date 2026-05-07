"""
SFT Dataset for Video Latent Visual Cache (Stage 1).

Each sample contains:
- A video input (passed as videos to processor with video_metadata)
- Key frame images (as LVC latent targets)
- Text conversations with <lvc> placeholders for key frames
"""

import copy
import os
import math
from typing import Dict, List, Tuple
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    LVC_START_TOKEN,
    LVC_END_TOKEN,
    LVC_TOKEN,
    LVC_PLACEHOLDER,
    VIDEO_LVC_SYSTEM_MESSAGE,
)

from src.dataset.data_utils import (
    get_video_info,
    get_image_info,
    replace_video_tokens,
    replace_lvc_tokens,
    llava_to_openai_video_lvc,
    pad_sequence,
)


class SFTDataset(Dataset):
    """Dataset for Video LVC SFT training."""

    def __init__(self, data_path, processor, data_args, model_id, padding=True):
        super(SFTDataset, self).__init__()
        if isinstance(data_path, str):
            self.list_data_dict = json.load(open(data_path, "r"))
            self.data_dir = os.path.dirname(os.path.abspath(data_path))
        else:
            self.list_data_dict = data_path
            self.data_dir = os.path.abspath(os.getcwd())

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.padding = padding

        # Media path resolution roots
        extra_media_roots = os.environ.get("VIDEO_LVC_MEDIA_ROOT", "")
        media_roots = [
            self.project_root,
            os.path.join(self.project_root, "data"),
            self.data_dir,
            os.path.join(self.data_dir, "data"),
        ]
        if extra_media_roots:
            media_roots.extend([p for p in extra_media_roots.split(":") if p])

        self.media_roots = []
        seen = set()
        for root in media_roots:
            root_abs = os.path.abspath(root)
            if root_abs not in seen:
                self.media_roots.append(root_abs)
                seen.add(root_abs)

        # Settings
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.max_video_frames = data_args.max_video_frames

    def __len__(self):
        return len(self.list_data_dict)

    def _resolve_media_path(self, raw_path):
        """Resolve dataset media paths against common local roots."""
        if not raw_path:
            return raw_path
        if os.path.isabs(raw_path) and os.path.exists(raw_path):
            return raw_path
        if os.path.exists(raw_path):
            return raw_path

        normalized = raw_path[2:] if raw_path.startswith("./") else raw_path
        normalized = normalized.lstrip("/")

        for root in self.media_roots:
            candidate = os.path.join(root, normalized)
            if os.path.exists(candidate):
                return candidate

        return raw_path

    def _load_key_frame_images(self, kf_paths):
        """Load key frame images from paths."""
        images = []
        for kf_path in kf_paths:
            resolved = self._resolve_media_path(kf_path)
            if os.path.exists(resolved):
                img = Image.open(resolved).convert("RGB")
                images.append(img)
            else:
                print(f"Warning: Key frame not found: {kf_path}")
                img = Image.new("RGB", (224, 224), (128, 128, 128))
                images.append(img)
        return images

    def bbox_to_token_idxs_manual(self, images, bboxes):
        """Convert bounding box coordinates to visual token indices."""
        token_idx_list = []
        for img, bbox in zip(images, bboxes):
            patch_size = self.processor.image_processor.patch_size
            image_width, image_height = img.width, img.height

            grid_height = image_height // patch_size
            grid_width = image_width // patch_size

            temporal_patch = getattr(self.processor.image_processor, "temporal_patch_size", 2)
            token_grid_height = grid_height // temporal_patch
            token_grid_width = grid_width // temporal_patch

            x1, y1, x2, y2 = bbox
            if max(x1, y1, x2, y2) > 1.0:
                x1 /= image_width
                y1 /= image_height
                x2 /= image_width
                y2 /= image_height

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(1, x2), min(1, y2)

            tx1 = int(x1 * token_grid_width)
            ty1 = int(y1 * token_grid_height)
            tx2 = min(int(math.ceil(x2 * token_grid_width)), token_grid_width)
            ty2 = min(int(math.ceil(y2 * token_grid_height)), token_grid_height)

            if tx2 <= tx1:
                tx2 = tx1 + 1
            if ty2 <= ty1:
                ty2 = ty1 + 1

            indices = []
            for y in range(ty1, ty2):
                for x in range(tx1, tx2):
                    indices.append(y * token_grid_width + x)
            token_idx_list.append(np.array(indices))

        return token_idx_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        processor = self.processor

        is_video = bool(sources.get("video", ""))
        has_kf = bool(sources.get("key_frame_images", []))
        has_kf_meta = bool(sources.get("key_frames", []))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_pixel_values_videos = []
        all_video_grid_thw = []

        # 1) Load video
        video_tensor = None
        video_metadata = None
        video_kwargs = {}

        if is_video and sources.get("video"):
            video_path = self._resolve_media_path(sources["video"])
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {sources['video']}")
                return self._get_dummy_sample()
            try:
                video_tensor, video_metadata, video_kwargs = get_video_info(
                    video_path, self.video_min_pixel, self.video_max_pixel,
                    self.video_resized_w, self.video_resized_h, self.fps,
                )
            except Exception as e:
                print(f"Warning: Failed to load video {video_path}: {e}")
                return self._get_dummy_sample()

        # 2) Handle key frames
        kf_images_for_lvc = None
        lvc_token_idxs_list = []

        kf_root = sources.get("kf_root", "") or os.environ.get("VIDEO_LVC_KF_ROOT", "")

        if has_kf_meta and video_tensor is not None:
            key_frames_meta = sources["key_frames"]
            kf_pil_images = []
            for kf_meta in key_frames_meta:
                kf_path = kf_meta["path"]
                if kf_root and not os.path.isabs(kf_path):
                    kf_path = os.path.join(kf_root, kf_path)
                kf_path = self._resolve_media_path(kf_path)
                if os.path.exists(kf_path):
                    kf_pil_images.append(Image.open(kf_path).convert("RGB"))
                else:
                    kf_pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
            kf_images_for_lvc = kf_pil_images
            bboxes = sources.get("bboxes", [[0.0, 0.0, 1.0, 1.0]] * len(kf_pil_images))
            lvc_token_idxs_list = self.bbox_to_token_idxs_manual(kf_pil_images, bboxes)
        elif has_kf and video_tensor is not None:
            kf_paths = sources["key_frame_images"]
            kf_images_for_lvc = self._load_key_frame_images(kf_paths)
            bboxes = sources.get("bboxes", [[0.0, 0.0, 1.0, 1.0]] * len(kf_paths))
            lvc_token_idxs_list = self.bbox_to_token_idxs_manual(kf_images_for_lvc, bboxes)

        # 3) Build conversation text
        conversations = copy.deepcopy(
            llava_to_openai_video_lvc(
                sources["conversations"], is_video=is_video,
                lvc_token_idxs_list=lvc_token_idxs_list,
            )
        )

        # 4) Tokenize
        system_message = (
            f"{DEFAULT_IM_START_TOKEN}system\n"
            f"{VIDEO_LVC_SYSTEM_MESSAGE}"
            f"{DEFAULT_IM_END_TOKEN}\n"
        )
        system_ids = processor.tokenizer(
            system_message, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        system_labels = torch.full_like(system_ids, IGNORE_INDEX)
        all_input_ids.append(system_ids.squeeze(0))
        all_labels.append(system_labels.squeeze(0))

        for j in range(0, len(conversations), 2):
            user_input = conversations[j]
            gpt_response = conversations[j + 1]

            user_text = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
                f"{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            )
            gpt_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            has_video_token = DEFAULT_VIDEO_TOKEN in user_text

            if has_video_token and video_tensor is not None:
                inputs = processor(
                    text=[user_text], images=None, videos=[video_tensor],
                    video_metadata=[video_metadata] if video_metadata is not None else None,
                    padding=False, do_resize=False, return_tensors="pt",
                    **(video_kwargs or {}),
                )
                prompt_input_ids = inputs["input_ids"]
                if "pixel_values_videos" in inputs:
                    all_pixel_values_videos.append(inputs["pixel_values_videos"])
                    all_video_grid_thw.append(inputs["video_grid_thw"])
            else:
                prompt_input_ids = processor.tokenizer(
                    user_text, add_special_tokens=False, padding=False, return_tensors="pt"
                )["input_ids"]

            response_input_ids = processor.tokenizer(
                gpt_text, add_special_tokens=False, padding=False, return_tensors="pt"
            )["input_ids"]

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [torch.tensor([IGNORE_INDEX] * prompt_input_ids.size(1)),
                 response_input_ids.squeeze(0)], dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # 5) Assemble output
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        lvc_tokens = [torch.tensor(arr) for arr in lvc_token_idxs_list]

        data_dict = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, lvc_tokens=lvc_tokens,
        )

        if kf_images_for_lvc:
            kf_pixel_list = []
            kf_grid_list = []
            for kf_img in kf_images_for_lvc:
                kf_input = processor(
                    text=[""], images=[kf_img], videos=None,
                    padding=False, return_tensors="pt",
                )
                if "pixel_values" in kf_input:
                    kf_pixel_list.append(kf_input["pixel_values"])
                    kf_grid_list.append(kf_input["image_grid_thw"])
            if kf_pixel_list:
                data_dict["kf_pixel_values"] = torch.cat(kf_pixel_list, dim=0)
                data_dict["kf_image_grid_thw"] = torch.cat(kf_grid_list, dim=0)

        if all_pixel_values_videos:
            data_dict["pixel_values_videos"] = torch.cat(all_pixel_values_videos, dim=0)
            data_dict["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)

        if all_pixel_values:
            data_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        return data_dict

    def _get_dummy_sample(self):
        """Return a minimal dummy sample for error recovery."""
        dummy_ids = torch.zeros(10, dtype=torch.long)
        return {
            "input_ids": dummy_ids,
            "attention_mask": torch.ones(10, dtype=torch.long),
            "labels": torch.full((10,), IGNORE_INDEX, dtype=torch.long),
            "lvc_tokens": [],
        }


class DataCollatorForSFT(object):
    """Collate examples for SFT training."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_thw = []
        batch_video_pixel_values = []
        batch_video_thw = []
        batch_kf_pixel_values = []
        batch_kf_image_thw = []

        for example in examples:
            keys = example.keys()
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            if "pixel_values_videos" in keys:
                batch_video_pixel_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            if "kf_pixel_values" in keys:
                batch_kf_pixel_values.append(example["kf_pixel_values"])
                batch_kf_image_thw.append(example["kf_image_grid_thw"])

        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX)

        lvc_tokens_all = []
        for example in examples:
            for idx_tensor in example.get("lvc_tokens", []):
                if isinstance(idx_tensor, torch.Tensor):
                    lvc_tokens_all.append(idx_tensor)
                else:
                    lvc_tokens_all.append(torch.tensor(idx_tensor))

        data_dict = {
            "input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask, "lvc_tokens": lvc_tokens_all,
        }

        if batch_pixel_values:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)
        if batch_video_pixel_values:
            data_dict["pixel_values_videos"] = torch.cat(batch_video_pixel_values, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)
        if batch_kf_pixel_values:
            data_dict["kf_pixel_values"] = torch.cat(batch_kf_pixel_values, dim=0)
            data_dict["kf_image_grid_thw"] = torch.cat(batch_kf_image_thw, dim=0)

        return data_dict


def make_sft_data_module(model_id, processor, data_args):
    """Make dataset and collator for SFT training."""
    dataset = SFTDataset(
        data_path=data_args.data_path, processor=processor,
        data_args=data_args, model_id=model_id,
    )
    collator = DataCollatorForSFT(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=dataset, eval_dataset=None, data_collator=collator)
