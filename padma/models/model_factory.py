"""Unified model factory for creating both timm and time series models."""

import logging
from typing import Optional, List

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Unified factory for creating models with architecture-specific defaults.

    Supports both timm models (ResNet, ViT, EfficientNet, etc.) and custom
    time series models (CNN1D, LSTM, GRU, Transformer1D).
    """

    # Time series model names
    TIMESERIES_MODELS = {'cnn1d', 'lstm', 'gru', 'transformer1d'}

    @staticmethod
    def create(
        model_name: str,
        num_classes: int,
        # Timm-specific parameters
        pretrained: bool = True,
        drop_rate: Optional[float] = None,
        drop_path_rate: Optional[float] = None,
        freeze_backbone: bool = False,
        # Time series-specific parameters
        input_channels: Optional[int] = None,
        sequence_length: Optional[int] = None,
        # Model-specific kwargs
        **kwargs
    ) -> nn.Module:
        """
        Create a model based on the model name.

        Args:
            model_name: Name of the model ('resnet50', 'cnn1d', 'lstm', etc.)
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights (timm models only)
            drop_rate: Dropout rate (None = use architecture-specific default)
            drop_path_rate: Drop path rate (None = use architecture-specific default)
            freeze_backbone: Whether to freeze backbone layers (timm models only)
            input_channels: Number of input channels (time series models)
            sequence_length: Length of time series sequence (required for Transformer)
            **kwargs: Model-specific parameters

        Returns:
            PyTorch model

        Raises:
            ValueError: If required parameters are missing or model is unknown
        """
        model_name_lower = model_name.lower()

        # Route to appropriate factory based on model name
        if model_name_lower in ModelFactory.TIMESERIES_MODELS:
            return ModelFactory._create_timeseries_model(
                model_name=model_name_lower,
                num_classes=num_classes,
                input_channels=input_channels,
                sequence_length=sequence_length,
                **kwargs
            )
        else:
            return ModelFactory._create_timm_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                freeze_backbone=freeze_backbone,
            )

    @staticmethod
    def _create_timm_model(
        model_name: str,
        num_classes: int,
        pretrained: bool,
        drop_rate: Optional[float],
        drop_path_rate: Optional[float],
        freeze_backbone: bool,
    ) -> nn.Module:
        """Create a timm model with architecture-specific defaults."""
        import timm

        # Get architecture-specific defaults
        defaults = ModelFactory._get_architecture_defaults(model_name)

        # Build kwargs, preferring explicit values over defaults
        kwargs = {
            'pretrained': pretrained,
            'num_classes': num_classes,
        }

        # Apply drop_rate
        if drop_rate is not None:
            kwargs['drop_rate'] = drop_rate
        elif defaults.get('drop_rate') is not None:
            kwargs['drop_rate'] = defaults['drop_rate']

        # Apply drop_path_rate
        if drop_path_rate is not None:
            kwargs['drop_path_rate'] = drop_path_rate
        elif defaults.get('drop_path_rate') is not None:
            kwargs['drop_path_rate'] = defaults['drop_path_rate']

        logger.info(f"Creating timm model: {model_name}")
        logger.info(f"Parameters: num_classes={num_classes}, pretrained={pretrained}, "
                   f"drop_rate={kwargs.get('drop_rate', 'N/A')}, "
                   f"drop_path_rate={kwargs.get('drop_path_rate', 'N/A')}")

        # Create model
        model = timm.create_model(model_name, **kwargs)

        # Freeze backbone if requested
        if freeze_backbone:
            from .base import freeze_backbone as freeze_fn
            freeze_fn(model)
            logger.info("Backbone layers frozen")

        return model

    @staticmethod
    def _create_timeseries_model(
        model_name: str,
        num_classes: int,
        input_channels: Optional[int],
        sequence_length: Optional[int],
        **kwargs
    ) -> nn.Module:
        """Create a time series model."""
        if input_channels is None:
            raise ValueError("input_channels is required for time series models")

        logger.info(f"Creating time series model: {model_name}")
        logger.info(f"Parameters: input_channels={input_channels}, "
                   f"num_classes={num_classes}, "
                   f"sequence_length={sequence_length}")

        if model_name == 'cnn1d':
            return ModelFactory._create_cnn1d(
                input_channels=input_channels,
                num_classes=num_classes,
                **kwargs
            )
        elif model_name == 'lstm':
            return ModelFactory._create_rnn(
                rnn_type='lstm',
                input_channels=input_channels,
                num_classes=num_classes,
                **kwargs
            )
        elif model_name == 'gru':
            return ModelFactory._create_rnn(
                rnn_type='gru',
                input_channels=input_channels,
                num_classes=num_classes,
                **kwargs
            )
        elif model_name == 'transformer1d':
            return ModelFactory._create_transformer1d(
                input_channels=input_channels,
                num_classes=num_classes,
                sequence_length=sequence_length,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown time series model: {model_name}. "
                f"Supported models: {', '.join(ModelFactory.TIMESERIES_MODELS)}"
            )

    @staticmethod
    def _create_cnn1d(
        input_channels: int,
        num_classes: int,
        **kwargs
    ) -> nn.Module:
        """Create 1D CNN model."""
        from .timeseries.cnn1d import CNN1D

        # Get model-specific parameters with defaults
        filters = kwargs.get('filters', [64, 128, 256])
        kernel_size = kwargs.get('kernel_size', 3)
        dropout = kwargs.get('dropout', 0.5)

        logger.info(f"CNN1D config: filters={filters}, kernel_size={kernel_size}, dropout={dropout}")

        return CNN1D(
            input_channels=input_channels,
            num_classes=num_classes,
            filters=filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    @staticmethod
    def _create_rnn(
        rnn_type: str,
        input_channels: int,
        num_classes: int,
        **kwargs
    ) -> nn.Module:
        """Create LSTM or GRU model."""
        from .timeseries.rnn import RNNModel

        # Get model-specific parameters with defaults
        hidden_size = kwargs.get('hidden_size', 128)
        num_layers = kwargs.get('num_layers', 2)
        bidirectional = kwargs.get('bidirectional', True)
        dropout = kwargs.get('dropout', 0.5)

        logger.info(f"{rnn_type.upper()} config: hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, bidirectional={bidirectional}, "
                   f"dropout={dropout}")

        return RNNModel(
            input_channels=input_channels,
            num_classes=num_classes,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    @staticmethod
    def _create_transformer1d(
        input_channels: int,
        num_classes: int,
        sequence_length: Optional[int],
        **kwargs
    ) -> nn.Module:
        """Create 1D Transformer model."""
        from .timeseries.transformer1d import Transformer1D

        if sequence_length is None:
            raise ValueError("sequence_length is required for Transformer1D model")

        # Get model-specific parameters with defaults
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 4)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        dropout = kwargs.get('dropout', 0.1)

        logger.info(f"Transformer1D config: d_model={d_model}, nhead={nhead}, "
                   f"num_layers={num_layers}, dim_feedforward={dim_feedforward}, "
                   f"dropout={dropout}")

        return Transformer1D(
            input_channels=input_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    @staticmethod
    def _get_architecture_defaults(model_name: str) -> dict:
        """
        Get architecture-specific default parameters for timm models.

        Returns:
            Dictionary of default parameters for the architecture
        """
        name_lower = model_name.lower()

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
