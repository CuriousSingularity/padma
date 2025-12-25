"""LSTM and GRU models for time series classification."""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    LSTM/GRU model for time series classification.

    Architecture:
        - Multi-layer bidirectional LSTM or GRU
        - Dropout between layers
        - Take final hidden state or mean pooling
        - Fully connected classifier
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        rnn_type: str = 'lstm',
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        """
        Initialize the RNN model.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            rnn_type: Type of RNN ('lstm' or 'gru')
            hidden_size: Hidden dimension size
            num_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # RNN layer
        rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU

        self.rnn = rnn_class(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classifier
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, num_classes),
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

        # RNN expects (batch, seq_len, features)
        # Our input is (batch, channels, timesteps)
        # Transpose to (batch, timesteps, channels)
        x = x.transpose(1, 2)

        # RNN forward
        # output shape: (batch, seq_len, hidden_size * num_directions)
        # hidden/cell shape: (num_layers * num_directions, batch, hidden_size)
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(x)
        else:
            output, hidden = self.rnn(x)

        # Use mean pooling over sequence dimension
        # This is often more robust than just using the last hidden state
        pooled = torch.mean(output, dim=1)

        # Classifier
        logits = self.classifier(pooled)

        return logits
