import unittest
from src.safety_config import SafetyConfig

class TestSafetyConfig(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of the SafetyConfig dataclass."""
        config = SafetyConfig()
        self.assertEqual(config.min_ethical_compliance, 0.8)
        self.assertEqual(config.max_uncertainty, 0.3)
        self.assertEqual(config.memory_capacity, 1000)
        self.assertEqual(config.max_processing_time, 1.0)

    def test_custom_config(self):
        """Test custom initialization of the SafetyConfig dataclass."""
        config = SafetyConfig(min_ethical_compliance=0.7, max_uncertainty=0.4, memory_capacity=500, max_processing_time=2.0)
        self.assertEqual(config.min_ethical_compliance, 0.7)
        self.assertEqual(config.max_uncertainty, 0.4)
        self.assertEqual(config.memory_capacity, 500)
        self.assertEqual(config.max_processing_time, 2.0)

if __name__ == "__main__":
    unittest.main()