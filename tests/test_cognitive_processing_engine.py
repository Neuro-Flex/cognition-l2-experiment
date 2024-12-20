import unittest
import torch
from src.cognitive_processing_engine import CognitiveProcessingEngine
from src.safety_config import SafetyConfig

class TestCognitiveProcessingEngine(unittest.TestCase):
    def setUp(self):
        self.input_dim = 64
        self.hidden_dim = 128
        self.safety_config = SafetyConfig()
        self.engine = CognitiveProcessingEngine(self.input_dim, self.hidden_dim, self.safety_config)

    def test_process_input(self):
        """Test the process_input method."""
        input_data = torch.randn(1, 10, self.input_dim)
        output = self.engine.process_input(input_data)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 10, self.hidden_dim))

    def test_safety_constraints(self):
        """Test the check_safety_constraints method."""
        self.engine.state.ethical_compliance = 0.9
        self.engine.state.uncertainty = 0.2
        violations = self.engine.check_safety_constraints()
        self.assertEqual(violations, [])

        self.engine.state.ethical_compliance = 0.7
        violations = self.engine.check_safety_constraints()
        self.assertEqual(violations, ["Ethical compliance low: 0.70"])

        self.engine.state.uncertainty = 0.4
        violations = self.engine.check_safety_constraints()
        self.assertEqual(violations, ["Ethical compliance low: 0.70", "Uncertainty high: 0.40"])

if __name__ == "__main__":
    unittest.main()