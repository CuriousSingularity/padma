import torch
from omegaconf import DictConfig


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
            print("Using CUDA (GPU)")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon)")
            return "mps"
        else:
            print("Using CPU")
            return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        return "cpu"

    return device
