import pytest
import torch
from src.cognitive_processing_engine import CognitiveProcessingEngine
from src.safety_config import SafetyConfig

@pytest.fixture
def setup_engine():
    input_dim = 64
    hidden_dim = 128
    safety_config = SafetyConfig()
    engine = CognitiveProcessingEngine(input_dim, hidden_dim, safety_config)
    return engine

def test_process_input(setup_engine):
    """Test the process_input method."""
    input_data = torch.randn(1, 10, setup_engine.input_dim)
    output = setup_engine.process_input(input_data)
    assert output is not None
    assert output.shape == (1, 10, setup_engine.hidden_dim)

def test_safety_constraints(setup_engine):
    """Test the check_safety_constraints method."""
    setup_engine.state.ethical_compliance = 0.9
    setup_engine.state.uncertainty = 0.2
    violations = setup_engine.check_safety_constraints()
    assert violations == []

    setup_engine.state.ethical_compliance = 0.7
    violations = setup_engine.check_safety_constraints()
    assert violations == ["Ethical compliance low: 0.70"]

    setup_engine.state.uncertainty = 0.4
    violations = setup_engine.check_safety_constraints()
    assert violations == ["Ethical compliance low: 0.70", "Uncertainty high: 0.40"]
