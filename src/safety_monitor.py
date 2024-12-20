from src.cognitive_state import CognitiveState

class SafetyMonitor:
    """Independently monitors the cognitive state for any deviations from safety
    and ethical standards, logging violations for review and potential intervention."""
    def __init__(self):
        self.safety_violations = []
        self.ethical_violations = []

    def monitor(self, state: CognitiveState) -> bool:
        is_safe = True
        if not state.safety_status:
            self.safety_violations.append({'type': 'safety', 'details': 'General failure'})
        if state.ethical_compliance < 0.7:
            self.ethical_violations.append({'type': 'ethics', 'value': state.ethical_compliance})
        return is_safe