import timm
import torch.nn as nn
from omegaconf import DictConfig

from .base import freeze_backbone


# Available ViT variants in timm
VIT_VARIANTS = [
    "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
    "vit_large_patch16_224", "vit_huge_patch14_224",
    "vit_base_patch32_224", "vit_large_patch32_224",
    "deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224",
    "deit3_small_patch16_224", "deit3_base_patch16_224", "deit3_large_patch16_224",
    "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224", "swin_large_patch4_window7_224",
]


def create_vit(cfg: DictConfig) -> nn.Module:
    """
    Create a Vision Transformer model using timm.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        ViT model
    """
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.get("drop_rate", 0.0),
        drop_path_rate=cfg.model.get("drop_path_rate", 0.1),
    )

    if cfg.model.get("freeze_backbone", False):
        freeze_backbone(model)

    return model


def is_vit(model_name: str) -> bool:
    """Check if the model name is a ViT variant."""
    name_lower = model_name.lower()
    return any(variant in name_lower for variant in ["vit_", "deit", "swin"])
