import pytest
from src.safety_config import SafetyConfig

def test_initialization():
    """Test the initialization of the SafetyConfig dataclass."""
    config = SafetyConfig()
    assert config.min_ethical_compliance == 0.8
    assert config.max_uncertainty == 0.3
    assert config.memory_capacity == 1000
    assert config.max_processing_time == 1.0

def test_custom_config():
    """Test custom initialization of the SafetyConfig dataclass."""
    config = SafetyConfig(min_ethical_compliance=0.7, max_uncertainty=0.4, memory_capacity=500, max_processing_time=2.0)
    assert config.min_ethical_compliance == 0.7
    assert config.max_uncertainty == 0.4
    assert config.memory_capacity == 500
    assert config.max_processing_time == 2.0
