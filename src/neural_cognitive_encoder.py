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
        batch_size, seq_len, _ = x.shape
        lstm_out, _ = self.lstm(x)
        
        # Prepare input for attention (seq_len, batch_size, hidden_dim)
        lstm_out = lstm_out.permute(1, 0, 2)
        
        # Get attention output and weights - removed average_attention_weights parameter
        attn_out, attn_weights = self.attention(
            lstm_out, 
            lstm_out, 
            lstm_out,
            need_weights=True
        )
        
        # Reshape attention weights to match expected shape (10, 1, 10)
        attn_weights = attn_weights.view(seq_len, batch_size, seq_len)
        
        # Return attention output in original shape (batch_size, seq_len, hidden_dim)
        attn_out = attn_out.permute(1, 0, 2)
        
        return attn_out, attn_weights