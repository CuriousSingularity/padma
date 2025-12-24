"""Model factory for creating timm models with architecture-specific defaults."""

import logging
from typing import Optional

import timm
import torch.nn as nn

logger = logging.getLogger(__name__)


class TimmModelFactory:
    """
    Factory for creating timm models with architecture-specific defaults.

    This factory encapsulates model creation logic and applies architecture-specific
    default values for regularization parameters (drop_rate, drop_path_rate).
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        drop_rate: Optional[float] = None,
        drop_path_rate: Optional[float] = None,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the model factory.

        Args:
            model_name: Name of the timm model to create
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            drop_rate: Dropout rate (None = use architecture-specific default)
            drop_path_rate: Drop path rate (None = use architecture-specific default)
            freeze_backbone: Whether to freeze backbone layers
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.freeze_backbone = freeze_backbone

    def create(self) -> nn.Module:
        """
        Create the model with architecture-specific defaults.

        Returns:
            PyTorch model
        """
        # Get architecture-specific defaults
        defaults = self._get_architecture_defaults()

        # Build kwargs, preferring explicit values over defaults
        kwargs = {
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
        }

        # Apply drop_rate
        if self.drop_rate is not None:
            kwargs['drop_rate'] = self.drop_rate
        elif defaults.get('drop_rate') is not None:
            kwargs['drop_rate'] = defaults['drop_rate']

        # Apply drop_path_rate
        if self.drop_path_rate is not None:
            kwargs['drop_path_rate'] = self.drop_path_rate
        elif defaults.get('drop_path_rate') is not None:
            kwargs['drop_path_rate'] = defaults['drop_path_rate']

        logger.info(f"Creating model: {self.model_name}")
        logger.info(f"Parameters: num_classes={self.num_classes}, pretrained={self.pretrained}, "
                   f"drop_rate={kwargs.get('drop_rate', 'N/A')}, "
                   f"drop_path_rate={kwargs.get('drop_path_rate', 'N/A')}")

        # Create model
        model = timm.create_model(self.model_name, **kwargs)

        # Freeze backbone if requested
        if self.freeze_backbone:
            from .base import freeze_backbone
            freeze_backbone(model)
            logger.info("Backbone layers frozen")

        return model

    def _get_architecture_defaults(self) -> dict:
        """
        Get architecture-specific default parameters.

        Returns:
            Dictionary of default parameters for the architecture
        """
        name_lower = self.model_name.lower()

        # ResNet family
        if any(v in name_lower for v in ['resnet', 'resnext', 'wide_resnet']):
            return {'drop_rate': 0.0}

        # Vision Transformer family
        if any(v in name_lower for v in ['vit_', 'deit', 'swin']):
            return {'drop_rate': 0.0, 'drop_path_rate': 0.1}

        # EfficientNet family
        if 'efficientnet' in name_lower:
            return {'drop_rate': 0.2, 'drop_path_rate': 0.2}

        # ConvNeXt family
        if 'convnext' in name_lower:
            return {'drop_rate': 0.0, 'drop_path_rate': 0.1}

        # MobileNet family
        if any(v in name_lower for v in ['mobilenet', 'lcnet']):
            return {'drop_rate': 0.2}

        # Generic/unknown architecture
        return {'drop_rate': 0.0, 'drop_path_rate': 0.0}
