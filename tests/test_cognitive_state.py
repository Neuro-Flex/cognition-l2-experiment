import pytest
import torch
from src.cognitive_state import CognitiveState

def test_initialization():
    """Test the initialization of the CognitiveState dataclass."""
    state = CognitiveState()
    assert torch.equal(state.attention_focus, torch.tensor([]))
    assert state.uncertainty == 0.0
    assert state.safety_status
    assert state.ethical_compliance == 1.0
    assert state.current_task is None
    assert state.working_memory == {}
    assert state.last_processed_input_shape is None

def test_update_state():
    """Test updating the CognitiveState attributes."""
    state = CognitiveState()
    state.attention_focus = torch.tensor([1.0, 2.0, 3.0])
    state.uncertainty = 0.5
    state.safety_status = False
    state.ethical_compliance = 0.9
    state.current_task = "test_task"
    state.working_memory = {"key": torch.tensor([1.0])}
    state.last_processed_input_shape = (1, 10, 64)

    assert torch.equal(state.attention_focus, torch.tensor([1.0, 2.0, 3.0]))
    assert state.uncertainty == 0.5
    assert not state.safety_status
    assert state.ethical_compliance == 0.9
    assert state.current_task == "test_task"
    assert state.working_memory == {"key": torch.tensor([1.0])}
    assert state.last_processed_input_shape == (1, 10, 64)
