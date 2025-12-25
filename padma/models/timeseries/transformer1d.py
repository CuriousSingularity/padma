"""1D Transformer for time series classification."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer1D(nn.Module):
    """
    1D Transformer for time series classification.

    Architecture:
        - Linear projection to embedding dimension
        - Positional encoding
        - Multi-head self-attention layers (TransformerEncoder)
        - Layer normalization
        - Global average pooling over sequence
        - Fully connected classifier
    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize the Transformer1D model.

        Args:
            input_channels: Number of input channels
            sequence_length: Length of time series sequence
            num_classes: Number of output classes
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, timesteps)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Ensure input has 3 dimensions
        if x.ndim == 2:
            # Add channel dimension for univariate time series
            x = x.unsqueeze(1)

        # Transformer expects (batch, seq_len, features)
        # Our input is (batch, channels, timesteps)
        # Transpose to (batch, timesteps, channels)
        x = x.transpose(1, 2)

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)

        # Classifier
        logits = self.classifier(x)

        return logits
