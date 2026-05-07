"""
Video LVC SFT Trainer.
Computes combined CE + contrastive LVC losses with separate LR for projection head.
"""

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


class SFTTrainer(Trainer):
    """Trainer for Video LVC SFT with CE + InfoNCE contrastive loss."""

    def __init__(self, *args, **kwargs):
        super(SFTTrainer, self).__init__(*args, **kwargs)

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            proj_head_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "merger" in name
                ]

            lvc_head_lr = getattr(self.args, "lvc_head_lr", None)
            if lvc_head_lr is None:
                lvc_head_lr = self.args.learning_rate * 10.0

            proj_head_parameters = [
                name for name, _ in opt_model.named_parameters()
                if "lvc_proj_head" in name
            ]
            if proj_head_parameters:
                lr_mapper["lvc_proj_head"] = lvc_head_lr

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters + proj_head_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_lr,
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_lr,
                        },
                    ])

                if merger_parameters:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.merger_lr,
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                            "weight_decay": 0.0,
                            "lr": self.args.merger_lr,
                        },
                    ])

                if proj_head_parameters:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if (n in decay_parameters and n in proj_head_parameters and p.requires_grad)],
                            "weight_decay": self.args.weight_decay,
                            "lr": lvc_head_lr,
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if (n not in decay_parameters and n in proj_head_parameters and p.requires_grad)],
                            "weight_decay": 0.0,
                            "lr": lvc_head_lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """Compute combined CE + Contrastive LVC loss."""
        outputs = model(**inputs)

        loss_ce = outputs.loss_ce
        loss_lvc = outputs.loss_lvc

        if loss_lvc is not None and self.args.loss_lvc_lambda > 0:
            loss = loss_ce + self.args.loss_lvc_lambda * loss_lvc
        else:
            loss = loss_ce

        # DeepSpeed requires a differentiable loss tensor for backward.
        if loss is not None and not loss.requires_grad:
            params = [p for p in model.parameters() if p.requires_grad]
            if len(params) > 0:
                loss = sum(p.float().sum() * 0.0 for p in params[:1])
            else:
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        self.log({
            "loss_total": loss.detach().item(),
            "loss_ce": loss_ce.detach().item() if loss_ce is not None else 0.0,
            "loss_lvc_contrastive": loss_lvc.detach().item() if loss_lvc is not None else 0.0,
        })

        return (loss, outputs) if return_outputs else loss
