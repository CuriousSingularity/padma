"""1D CNN for time series classification."""

from typing import List

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time series classification.

    Architecture:
        - Multiple Conv1d layers with increasing filters
        - BatchNorm1d and ReLU after each conv
        - MaxPool1d for downsampling
        - Dropout for regularization
        - Global average pooling
        - Fully connected classifier
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        filters: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        """
        Initialize the 1D CNN model.

        Args:
            input_channels: Number of input channels (1 for univariate, >1 for multivariate)
            num_classes: Number of output classes
            filters: List of filter sizes for each conv layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels

        for out_channels in filters:
            conv_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Linear(filters[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, timesteps)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Ensure input has 3 dimensions (batch, channels, timesteps)
        if x.ndim == 2:
            # Add channel dimension for univariate time series
            x = x.unsqueeze(1)

        # Convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = x.squeeze(-1)

        # Classifier
        x = self.classifier(x)

        return x
