"""
Parameters for Video LVC GRPO (Stage 2) training.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    coconut: bool = field(default=True)
    lvc_head: bool = field(default=False)
    lvc_head_type: str = field(default="simple")
    latent_end_token: bool = field(default=False)
    max_lvc_tokens: int = field(default=None)


@dataclass
class GRPOArguments(GRPOConfigTRL):
    """GRPO training arguments for Video LVC."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    # LVC loss
    loss_lvc_fct: str = field(default="infonce")
    loss_lvc_lambda: float = field(default=0.1)

    # Freeze settings
    freeze_vision_tower: bool = field(default=True)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=True)
    disable_flash_attn2: bool = field(default=True)

    max_seq_length: int = field(default=32768)

    # Quantization
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)

    # LoRA
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

    run_name: Optional[str] = field(default="video_lvc_grpo")

    # Checkpoint
    checkpoint_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to SFT stage1 checkpoint to start from."}
    )

    # GRPO specific
    beta: float = field(default=0.04, metadata={"help": "KL coefficient."})
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = field(default=512, metadata={"help": "Max new tokens per generation."})
    max_prompt_length: int = field(default=4096, metadata={"help": "Max prompt length (tokens)."})

    # LVC decoding
    decoding_strategy: str = field(default="lvc_reasoning", metadata={"help": "'lvc_reasoning' or 'steps'"})
    lvc_steps: int = field(default=10, metadata={"help": "Number of LVC tokens per key frame."})

    # Reward weights [accuracy, format, temporal_grounding, latent_reasoning]
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights for reward functions: [accuracy, format, temporal_grounding, latent_reasoning]"}
    )

    # Latent reasoning reward
    lvc_reward_temperature: float = field(default=0.1)
    lvc_reward_threshold: float = field(default=0.3)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data JSON."})
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    kf_folder: Optional[str] = field(default=None, metadata={"help": "Root folder for key frame images."})
    require_kf: bool = field(
        default=False,
        metadata={"help": "If True, only use items with valid key_frames."}
    )

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
    fps: float = field(default=1.0)
    max_video_frames: int = field(default=16)

    random_seed: Optional[int] = field(default=None)
