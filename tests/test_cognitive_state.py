import unittest
import torch
from src.cognitive_state import CognitiveState

class TestCognitiveState(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of the CognitiveState dataclass."""
        state = CognitiveState()
        self.assertTrue(torch.equal(state.attention_focus, torch.tensor([])))
        self.assertEqual(state.uncertainty, 0.0)
        self.assertTrue(state.safety_status)
        self.assertEqual(state.ethical_compliance, 1.0)
        self.assertIsNone(state.current_task)
        self.assertEqual(state.working_memory, {})
        self.assertIsNone(state.last_processed_input_shape)

    def test_update_state(self):
        """Test updating the CognitiveState attributes."""
        state = CognitiveState()
        state.attention_focus = torch.tensor([1.0, 2.0, 3.0])
        state.uncertainty = 0.5
        state.safety_status = False
        state.ethical_compliance = 0.9
        state.current_task = "test_task"
        state.working_memory = {"key": torch.tensor([1.0])}
        state.last_processed_input_shape = (1, 10, 64)

        self.assertTrue(torch.equal(state.attention_focus, torch.tensor([1.0, 2.0, 3.0])))
        self.assertEqual(state.uncertainty, 0.5)
        self.assertFalse(state.safety_status)
        self.assertEqual(state.ethical_compliance, 0.9)
        self.assertEqual(state.current_task, "test_task")
        self.assertEqual(state.working_memory, {"key": torch.tensor([1.0])})
        self.assertEqual(state.last_processed_input_shape, (1, 10, 64))

if __name__ == "__main__":
    unittest.main()