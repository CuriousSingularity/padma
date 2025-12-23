import timm
import torch.nn as nn
from omegaconf import DictConfig

from .base import freeze_backbone, get_model_info, load_checkpoint
from .resnet import create_resnet, is_resnet
from .vit import create_vit, is_vit
from .efficientnet import create_efficientnet, is_efficientnet
from .convnext import create_convnext, is_convnext
from .mobilenet import create_mobilenet, is_mobilenet


def create_model(cfg: DictConfig) -> nn.Module:
    """
    Create a model based on configuration.

    Routes to architecture-specific factory based on model name.
    Falls back to generic timm creation for unsupported architectures.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        PyTorch model
    """
    model_name = cfg.model.name

    # Route to architecture-specific factory
    if is_resnet(model_name):
        return create_resnet(cfg)
    elif is_vit(model_name):
        return create_vit(cfg)
    elif is_efficientnet(model_name):
        return create_efficientnet(cfg)
    elif is_convnext(model_name):
        return create_convnext(cfg)
    elif is_mobilenet(model_name):
        return create_mobilenet(cfg)
    else:
        # Fallback to generic timm model creation
        return create_generic_model(cfg)


def create_generic_model(cfg: DictConfig) -> nn.Module:
    """
    Create a generic model using timm for architectures not explicitly supported.

    Args:
        cfg: Model configuration from Hydra

    Returns:
        PyTorch model
    """
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.get("drop_rate", 0.0),
        drop_path_rate=cfg.model.get("drop_path_rate", 0.0),
    )

    if cfg.model.get("freeze_backbone", False):
        freeze_backbone(model)

    return model


__all__ = [
    "create_model",
    "create_generic_model",
    "freeze_backbone",
    "get_model_info",
    "load_checkpoint",
]
