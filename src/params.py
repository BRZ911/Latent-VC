"""
Parameters for Video LVC SFT (Stage 1) training.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    coconut: bool = field(default=True)
    lvc_head: bool = field(default=False)
    lvc_head_type: str = field(default="simple")
    latent_end_token: bool = field(default=False)
    max_lvc_tokens: int = field(default=None)


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    # LVC loss
    loss_lvc_fct: str = field(default="mse")
    loss_lvc_lambda: float = field(default=1e-1)

    # Freeze settings
    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(default=32768)

    # Quantization
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)

    # LoRA (optional)
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    # Learning rates
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lvc_head_lr: Optional[float] = None

    run_name: Optional[str] = field(default="video_lvc_sft")

    # Mode switch loss
    mode_switch_loss: Optional[bool] = False
    loss_mode_switch_fct: Optional[str] = field(default="mse")
    loss_mode_switch_lambda: Optional[float] = field(default=1e-1)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data JSON."})
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    kf_folder: Optional[str] = field(default=None, metadata={"help": "Root folder for key frame images."})

    # Image resolution
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)

    # Resizing (optional)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)

    # Video settings
    fps: float = field(default=2.0)
    max_video_frames: int = field(default=16)

    random_seed: Optional[int] = field(default=None)
