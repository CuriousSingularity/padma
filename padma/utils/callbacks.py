"""Callback utilities for PyTorch Lightning training."""

import logging
from typing import List

from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def create_callbacks(cfg: DictConfig) -> List[Callback]:
    """
    Create PyTorch Lightning callbacks from configuration using hydra.utils.instantiate().

    Args:
        cfg: Hydra configuration object containing callbacks configuration

    Returns:
        List of configured callbacks
    """
    callbacks = []

    for callback_name, callback_cfg in cfg.callbacks.items():
        # Skip if not a callback config (e.g., if it's just a string)
        if not isinstance(callback_cfg, DictConfig):
            continue

        # Check if enabled flag exists and is False
        if callback_cfg.get('enabled', True) is False:
            logger.info(f"Skipping {callback_name} callback (disabled)")
            continue

        # Remove 'enabled' from config before instantiate
        cfg_copy = OmegaConf.to_container(callback_cfg, resolve=True)
        cfg_copy.pop('enabled', None)

        # Instantiate callback
        try:
            callback = instantiate(cfg_copy)
            if callback is not None:
                callbacks.append(callback)
                logger.info(f"Added {callback.__class__.__name__} callback")
        except Exception as e:
            logger.error(f"Failed to instantiate {callback_name} callback: {e}")
            raise

    logger.info(f"Total callbacks configured: {len(callbacks)}")
    return callbacks
