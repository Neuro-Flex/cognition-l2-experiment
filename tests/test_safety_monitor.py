import unittest
from src.safety_monitor import SafetyMonitor
from src.cognitive_state import CognitiveState

class TestSafetyMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = SafetyMonitor()
        self.state = CognitiveState()

    def test_monitor_safe(self):
        """Test the monitor method when the state is safe."""
        self.state.safety_status = True
        self.state.ethical_compliance = 0.9
        is_safe = self.monitor.monitor(self.state)
        self.assertTrue(is_safe)
        self.assertEqual(self.monitor.safety_violations, [])
        self.assertEqual(self.monitor.ethical_violations, [])

    def test_monitor_unsafe(self):
        """Test the monitor method when the state is unsafe."""
        self.state.safety_status = False
        self.state.ethical_compliance = 0.6
        is_safe = self.monitor.monitor(self.state)
        self.assertFalse(is_safe)
        self.assertEqual(self.monitor.safety_violations, [{'type': 'safety', 'details': 'General failure'}])
        self.assertEqual(self.monitor.ethical_violations, [{'type': 'ethics', 'value': 0.6}])

if __name__ == "__main__":
    unittest.main()