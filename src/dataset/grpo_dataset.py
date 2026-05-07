"""
GRPO Dataset for Video Latent Visual Cache (Stage 2).

Each __getitem__ returns a prompt dict for the GRPO trainer to generate completions.
"""

import os
from typing import Dict
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.constants import VIDEO_LVC_SYSTEM_MESSAGE


def get_video_content(video_path, min_pixels, max_pixels, width, height, fps, max_frames=None):
    """Create video content dict for qwen_vl_utils processing."""
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps,
    }
    if max_frames is not None and max_frames > 0:
        content["max_frames"] = int(max_frames)
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    return content


class GRPODataset(Dataset):
    """Dataset for Video LVC GRPO training."""

    def __init__(self, data_path, processor, data_args, model_id, padding=True):
        super(GRPODataset, self).__init__()
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

        # Media path resolution — mirrors the V2 dataset paths so that
        # STGR-RL-v2.json bare filenames can be located.
        extra_media_roots = os.environ.get("VIDEO_LVC_MEDIA_ROOT", "")
        # Also check the original project root for data that lives outside onlion/
        original_project_root = os.path.abspath(os.path.join(self.project_root, ".."))
        data_dir_in_project = os.path.join(original_project_root, "data")
        videos_dir = os.path.join(data_dir_in_project, "videos")
        stgr_plm_videos = os.path.join(videos_dir, "stgr", "plm", "videos")
        stgr_tg_videos = os.path.join(videos_dir, "stgr", "temporal_grounding", "videos")
        video_r1_dir = os.path.join(videos_dir, "video_r1")
        videoespresso_kfs = os.path.join(videos_dir, "videoespresso", "kfs")
        videoespresso_train = "VideoEspresso_train_video"
        timerft_videos = os.path.join(videos_dir, "timerft_data")
        media_roots = [
            self.project_root,
            os.path.join(self.project_root, "data"),
            original_project_root,
            data_dir_in_project,
            videos_dir,
            video_r1_dir,
            stgr_plm_videos,
            stgr_tg_videos,
            videoespresso_train,
            videoespresso_kfs,
            timerft_videos,
            self.data_dir,
            os.path.join(self.data_dir, "data"),
        ]
        if data_args.video_folder:
            media_roots.insert(0, data_args.video_folder)
        if extra_media_roots:
            media_roots.extend([p for p in extra_media_roots.split(":") if p])

        self.media_roots = []
        seen = set()
        for root in media_roots:
            root_abs = os.path.abspath(root)
            if root_abs not in seen:
                self.media_roots.append(root_abs)
                seen.add(root_abs)

        # Prefix stripping rules (from V2 dataset)
        self._strip_prefixes = [
            "GroundedVLLM/",
            "LLaVA-Video-178K/academic_source/",
            "LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/",
            "LLaVA-Video-178K/liwei_youtube_videos/videos/",
            "LLaVA-Video-178K/liwei_youtube_videos/",
            "LLaVA-Video-178K/",
        ]

        # Pre-filter: remove items whose video cannot be found
        valid_items = []
        skipped = 0
        for item in self.list_data_dict:
            vp = item.get("video_path", "")
            resolved = self._resolve_media_path(vp)
            if resolved and os.path.exists(resolved):
                item["video_path"] = resolved
                valid_items.append(item)
            else:
                skipped += 1
        self.list_data_dict = valid_items
        if skipped > 0:
            print(f"[GRPODataset] Skipped {skipped} items with missing videos. "
                  f"Remaining: {len(self.list_data_dict)}")

        # Resolve key_frames paths
        kf_resolved = 0
        for item in self.list_data_dict:
            kf_list = item.get("key_frames")
            if not kf_list:
                continue
            for kf in kf_list:
                kf_path = kf.get("path", "")
                if not kf_path or os.path.isabs(kf_path):
                    continue
                # Try current project root first, then original project root
                for root in [self.project_root, original_project_root]:
                    abs_kf = os.path.join(root, kf_path)
                    if os.path.exists(abs_kf):
                        kf["path"] = abs_kf
                        kf_resolved += 1
                        break
        if kf_resolved > 0:
            print(f"[GRPODataset] Resolved {kf_resolved} key_frame paths to absolute.")

        # Optional: only keep items with valid key_frames
        require_kf = getattr(data_args, "require_kf", False)
        if require_kf:
            before = len(self.list_data_dict)
            self.list_data_dict = [
                item for item in self.list_data_dict
                if item.get("key_frames") and len(item["key_frames"]) > 0
                and all(os.path.exists(kf.get("path", "")) for kf in item["key_frames"])
            ]
            dropped = before - len(self.list_data_dict)
            print(f"[GRPODataset] require_kf=True: kept {len(self.list_data_dict)}, dropped {dropped}.")

        # Settings
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.max_video_frames = getattr(data_args, "max_video_frames", None)

    def __len__(self):
        return len(self.list_data_dict)

    def _resolve_media_path(self, raw_path):
        if not raw_path:
            return raw_path
        if os.path.isabs(raw_path) and os.path.exists(raw_path):
            return raw_path
        if os.path.exists(raw_path):
            return raw_path

        normalized = raw_path[2:] if raw_path.startswith("./") else raw_path
        normalized = normalized.lstrip("/")

        # Direct search in media roots
        for root in self.media_roots:
            candidate = os.path.join(root, normalized)
            if os.path.exists(candidate):
                return candidate

        # Try stripping known prefixes and re-searching
        for prefix in self._strip_prefixes:
            if normalized.startswith(prefix):
                stripped = normalized[len(prefix):]
                if stripped:
                    for root in self.media_roots:
                        candidate = os.path.join(root, stripped)
                        if os.path.exists(candidate):
                            return candidate

        # For bare filenames, also try basename in each root
        basename = os.path.basename(normalized)
        if basename != normalized:
            for root in self.media_roots:
                candidate = os.path.join(root, basename)
                if os.path.exists(candidate):
                    return candidate

        return raw_path

    def __getitem__(self, i) -> Dict:
        sources = self.list_data_dict[i]
        video_path = sources.get("video_path", "")

        video_content = get_video_content(
            video_path, self.video_min_pixel, self.video_max_pixel,
            self.video_resized_w, self.video_resized_h, self.fps,
            self.max_video_frames,
        )

        question = sources.get("question", "")
        answer = sources.get("answer", "")

        user_content = [video_content, {"type": "text", "text": question}]

        prompt = [
            {"role": "system", "content": VIDEO_LVC_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]

        assistant = {"role": "assistant", "content": answer}

        data_dict = {
            "prompt": prompt,
            "assistant": assistant,
            "id": sources.get("id", f"item_{i}"),
            "task": sources.get("task", ""),
            "video_path": video_path,
        }

        key_frames = sources.get("key_frames", [])
        if key_frames:
            data_dict["key_frames"] = key_frames

        return data_dict


def make_grpo_data_module(model_id, processor, data_args):
    """Create dataset for GRPO training."""
    dataset = GRPODataset(
        data_path=data_args.data_path, processor=processor,
        data_args=data_args, model_id=model_id,
    )
    return dict(train_dataset=dataset, eval_dataset=None)
