import timm
import torch.nn as nn
from omegaconf import DictConfig

from .base import freeze_backbone


# Available MobileNet variants in timm (sorted by size/speed)
MOBILENET_VARIANTS = [
    # MobileNetV3 Small variants (fastest)
    "mobilenetv3_small_050",  # Smallest - ~1M params
    "mobilenetv3_small_075",
    "mobilenetv3_small_100",
    # MobileNetV3 Large variants
    "mobilenetv3_large_075",
    "mobilenetv3_large_100",
    # MobileNetV2 variants
    "mobilenetv2_050",
    "mobilenetv2_100",
    # LCNet (very fast, designed for CPU)
    "lcnet_050",
    "lcnet_075",
    "lcnet_100",
]


def create_mobilenet(cfg: DictConfig) -> nn.Module:
    """
    Create a MobileNet model using timm.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        MobileNet model
    """
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.get("drop_rate", 0.2),
    )

    if cfg.model.get("freeze_backbone", False):
        freeze_backbone(model)

    return model


def is_mobilenet(model_name: str) -> bool:
    """Check if the model name is a MobileNet variant."""
    name_lower = model_name.lower()
    return any(variant in name_lower for variant in ["mobilenet", "lcnet"])
