import torch
import torch.nn as nn
from typing import Tuple

class NeuralCognitiveEncoder(nn.Module):
    """Encodes input sequences into a rich cognitive representation using LSTMs
    and self-attention to capture temporal dependencies and focus on relevant information."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out, attn_weights