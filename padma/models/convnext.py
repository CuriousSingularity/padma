import timm
import torch.nn as nn
from omegaconf import DictConfig

from .base import freeze_backbone


# Available ConvNeXt variants in timm
CONVNEXT_VARIANTS = [
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xlarge",
    "convnextv2_tiny", "convnextv2_base", "convnextv2_large", "convnextv2_huge",
]


def create_convnext(cfg: DictConfig) -> nn.Module:
    """
    Create a ConvNeXt model using timm.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        ConvNeXt model
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


def is_convnext(model_name: str) -> bool:
    """Check if the model name is a ConvNeXt variant."""
    return "convnext" in model_name.lower()
