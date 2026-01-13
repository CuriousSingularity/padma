"""ONNX model export utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_dummy_input(
    cfg: DictConfig,
    is_timeseries: bool,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Create a dummy input tensor for ONNX export.

    Args:
        cfg: Hydra configuration
        is_timeseries: Whether the model is a time series model
        batch_size: Batch size for dummy input

    Returns:
        Dummy input tensor with appropriate shape
    """
    if is_timeseries:
        # Time series models: (batch_size, input_channels, sequence_length)
        input_channels = cfg.model.get("input_channels", 1)
        sequence_length = cfg.model.get("sequence_length", 140)  # Default for ECG5000
        input_shape = (batch_size, input_channels, sequence_length)
        logger.info(f"Time series input shape: {input_shape}")
    else:
        # Image models: (batch_size, channels, height, width)
        image_size = cfg.dataset.get("image_size", 224)
        # Most models expect 3 channels (RGB), even if dataset is grayscale (converted in transforms)
        channels = 3
        input_shape = (batch_size, channels, image_size, image_size)
        logger.info(f"Image input shape: {input_shape}")

    return torch.randn(input_shape)


def export_model_to_onnx(
    model: nn.Module,
    checkpoint_path: str,
    cfg: DictConfig,
    is_timeseries: bool = False,
    opset_version: int = 18,
    dynamic_axes: Optional[dict] = None,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        checkpoint_path: Path to the model checkpoint (used to determine save directory)
        cfg: Hydra configuration
        is_timeseries: Whether the model is a time series model
        opset_version: ONNX opset version to use
        dynamic_axes: Optional dynamic axes configuration for variable input sizes

    Returns:
        Path to the exported ONNX model

    Raises:
        RuntimeError: If ONNX export fails
    """
    logger.info("=" * 60)
    logger.info("EXPORTING MODEL TO ONNX")
    logger.info("=" * 60)

    # Determine output directory (same as checkpoint directory)
    checkpoint_path = Path(checkpoint_path)
    onnx_dir = checkpoint_path.parent
    model_name = cfg.model.model_name.replace("/", "_")  # Handle model names with slashes
    onnx_path = onnx_dir / f"{model_name}.onnx"

    logger.info(f"Model: {cfg.model.model_name}")
    logger.info(f"Output path: {onnx_path}")

    # Move model to CPU for ONNX export (required for compatibility)
    logger.info("Moving model to CPU for ONNX export...")
    model = model.cpu()

    # Set model to evaluation mode
    model.eval()

    # Create dummy input on CPU
    dummy_input = get_dummy_input(cfg, is_timeseries)

    # Default dynamic axes if not provided
    if dynamic_axes is None:
        # Allow dynamic batch size
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    try:
        # Export to ONNX using legacy exporter (more stable for various model architectures)
        logger.info(f"Exporting with ONNX opset version {opset_version} (using legacy exporter)...")

        # Disable gradient computation
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                dynamo=False,  # Use legacy exporter for better compatibility
            )

        logger.info(f"✓ Successfully exported model to ONNX format")
        logger.info(f"✓ ONNX model saved at: {onnx_path}")

        # Verify the exported model
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model verification passed")
        except ImportError:
            logger.warning("onnx package not installed, skipping verification")
        except Exception as e:
            logger.warning(f"ONNX model verification failed: {e}")

        logger.info("=" * 60)

        return onnx_path

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise RuntimeError(f"ONNX export failed: {e}")


def export_lightning_model_to_onnx(
    lightning_module,
    checkpoint_path: str,
    cfg: DictConfig,
    is_timeseries: bool = False,
    opset_version: int = 18,
) -> Path:
    """
    Export a PyTorch Lightning model to ONNX format.

    Args:
        lightning_module: PyTorch Lightning module containing the model
        checkpoint_path: Path to the model checkpoint
        cfg: Hydra configuration
        is_timeseries: Whether the model is a time series model
        opset_version: ONNX opset version to use

    Returns:
        Path to the exported ONNX model
    """
    # Extract the underlying PyTorch model from the Lightning module
    model = lightning_module.model

    return export_model_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        is_timeseries=is_timeseries,
        opset_version=opset_version,
    )
