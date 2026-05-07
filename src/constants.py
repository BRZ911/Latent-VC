"""
Constants for Video Latent Visual Cache.
"""

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

# LVC special tokens
LVC_START_TOKEN = "<|lvc_start|>"
LVC_END_TOKEN = "<|lvc_end|>"
LVC_TOKEN = "<|lvc|>"
LVC_LATENT_END_TOKEN = "<|lvc_latent_end|>"
LVC_PLACEHOLDER = "<lvc>"

SYSTEM_MESSAGE = "You are a helpful assistant."

VIDEO_LVC_SYSTEM_MESSAGE = (
    "A conversation between user and assistant. The user provides a video and asks a question. "
    "The assistant reasons about key visual moments in the video using latent visual cache, "
    "then provides the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively."
)

MULTIMODAL_KEYWORDS = [
    "pixel_values", "image_grid_thw", "video_grid_thw",
    "pixel_values_videos", "second_per_grid_ts"
]
