import timm
import torch.nn as nn
from omegaconf import DictConfig

from .base import freeze_backbone


# Available ResNet variants in timm
RESNET_VARIANTS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnet18d", "resnet34d", "resnet50d", "resnet101d", "resnet152d",
    "resnetv2_50", "resnetv2_101", "resnetv2_152",
    "wide_resnet50_2", "wide_resnet101_2",
    "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
]


def create_resnet(cfg: DictConfig) -> nn.Module:
    """
    Create a ResNet model using timm.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        ResNet model
    """
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.get("drop_rate", 0.0),
    )

    if cfg.model.get("freeze_backbone", False):
        freeze_backbone(model)

    return model


def is_resnet(model_name: str) -> bool:
    """Check if the model name is a ResNet variant."""
    name_lower = model_name.lower()
    return any(variant in name_lower for variant in ["resnet", "resnext", "wide_resnet"])
