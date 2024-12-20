import pytest
import torch
from src.neural_cognitive_encoder import NeuralCognitiveEncoder

def test_forward_pass():
    """Test the forward pass of the NeuralCognitiveEncoder."""
    input_dim = 64
    hidden_dim = 128
    encoder = NeuralCognitiveEncoder(input_dim, hidden_dim)
    input_data = torch.randn(1, 10, input_dim)  # Batch size 1, sequence length 10
    output, attention_weights = encoder(input_data)

    assert output.shape == (1, 10, hidden_dim)
    assert attention_weights.shape == (10, 1, 10)

if __name__ == "__main__":
    pytest.main()
