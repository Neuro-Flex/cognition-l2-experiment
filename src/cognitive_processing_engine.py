import torch
from collections import deque
from typing import Optional, List
from src.cognitive_state import CognitiveState
from src.safety_config import SafetyConfig
from src.neural_cognitive_encoder import NeuralCognitiveEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitiveProcessingEngine:
    """The central processing unit, managing information flow, maintaining cognitive
    state, enforcing safety and ethical guidelines, and interacting with memory."""
    def __init__(self, input_dim, hidden_dim, safety_config):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.safety_config = safety_config
        self.encoder = NeuralCognitiveEncoder(input_dim, hidden_dim)
        self.state = CognitiveState()
        self.memory_bank = deque(maxlen=safety_config.memory_capacity)
        self.knowledge_base = {"important_constant": torch.tensor([3.14159])}

    def check_safety_constraints(self) -> List[str]:
        violations = []
        if self.state.ethical_compliance < self.safety_config.min_ethical_compliance:
            violations.append(f"Ethical compliance low: {self.state.ethical_compliance:.2f}")
        if self.state.uncertainty > self.safety_config.max_uncertainty:
            violations.append(f"Uncertainty high: {self.state.uncertainty:.2f}")
        return violations

    def evaluate_ethical_rules(self, data: torch.Tensor) -> float:
        mean_output = torch.mean(data).item()
        return max(0.0, 1.0 - (mean_output - 0.5)) if mean_output > 0.5 else 1.0

    def process_input(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            # Process the input using the encoder
            output, _ = self.encoder(input_data)
            return output
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.state.uncertainty += 0.1  # Increase uncertainty on error
            return torch.zeros_like(input_data)  # Return safe default instead of None

    def _generate_safe_output(self, data: torch.Tensor) -> torch.Tensor:
        lower_bound = -self.state.uncertainty
        upper_bound = self.state.uncertainty
        return torch.clamp(data, lower_bound, upper_bound)