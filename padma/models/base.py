import torch
import torch.nn as nn


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head.

    Args:
        model: PyTorch model to freeze
    """
    for name, param in model.named_parameters():
        if "head" not in name and "fc" not in name and "classifier" not in name:
            param.requires_grad = False


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load model weights from a checkpoint.

    Args:
        model: PyTorch model
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model
