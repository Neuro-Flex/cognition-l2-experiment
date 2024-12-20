import pytest
from src.safety_monitor import SafetyMonitor
from src.cognitive_state import CognitiveState

@pytest.fixture
def setup_monitor():
    monitor = SafetyMonitor()
    state = CognitiveState()
    return monitor, state

def test_monitor_safe(setup_monitor):
    """Test the monitor method when the state is safe."""
    monitor, state = setup_monitor
    state.safety_status = True
    state.ethical_compliance = 0.9
    is_safe = monitor.monitor(state)
    assert is_safe
    assert monitor.safety_violations == []
    assert monitor.ethical_violations == []

def test_monitor_unsafe(setup_monitor):
    """Test the monitor method when the state is unsafe."""
    monitor, state = setup_monitor
    state.safety_status = False
    state.ethical_compliance = 0.6
    is_safe = monitor.monitor(state)
    assert not is_safe
    assert monitor.safety_violations == [{'type': 'safety', 'details': 'General failure'}]
    assert monitor.ethical_violations == [{'type': 'ethics', 'value': 0.6}]