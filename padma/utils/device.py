import logging

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_device(cfg: DictConfig) -> str:
    """
    Get the device to use for training/evaluation.

    Auto-detection order: CUDA (GPU) > MPS (Apple Silicon) > CPU

    Args:
        cfg: Configuration with device field (auto, cuda, mps, or cpu)

    Returns:
        Device string for PyTorch
    """
    device = cfg.device

    if device == "auto":
        if torch.cuda.is_available():
            logger.info("Using CUDA (GPU)")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon)")
            return "mps"
        else:
            logger.info("Using CPU")
            return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        return "cpu"

    return device


def get_accelerator(cfg: DictConfig) -> str:
    """
    Get accelerator type for PyTorch Lightning based on configuration.

    Auto-detection order: CUDA > MPS > CPU

    Args:
        cfg: Configuration with device field (auto, cuda, mps, or cpu)

    Returns:
        Accelerator string for PyTorch Lightning (cuda, mps, or cpu)
    """
    device = cfg.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def get_precision(cfg: DictConfig, accelerator: str) -> str:
    """
    Get precision setting for PyTorch Lightning based on configuration and accelerator.

    Args:
        cfg: Configuration with mixed_precision field
        accelerator: Accelerator type (cuda, mps, or cpu)

    Returns:
        Precision string for PyTorch Lightning (e.g., '16-mixed', '32-true')
    """
    if cfg.get("mixed_precision", False):
        if accelerator in ["cuda", "mps"]:
            return "16-mixed"
    return "32-true"
