import timm
import torch.nn as nn
from omegaconf import DictConfig

from .base import freeze_backbone


# Available EfficientNet variants in timm
EFFICIENTNET_VARIANTS = [
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_xl",
    "tf_efficientnet_b0", "tf_efficientnet_b1", "tf_efficientnet_b2",
    "tf_efficientnetv2_s", "tf_efficientnetv2_m", "tf_efficientnetv2_l",
]


def create_efficientnet(cfg: DictConfig) -> nn.Module:
    """
    Create an EfficientNet model using timm.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        EfficientNet model
    """
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.get("drop_rate", 0.2),
        drop_path_rate=cfg.model.get("drop_path_rate", 0.2),
    )

    if cfg.model.get("freeze_backbone", False):
        freeze_backbone(model)

    return model


def is_efficientnet(model_name: str) -> bool:
    """Check if the model name is an EfficientNet variant."""
    return "efficientnet" in model_name.lower()
