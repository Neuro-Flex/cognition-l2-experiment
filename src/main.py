import torch
from src.cognitive_processing_engine import CognitiveProcessingEngine
from src.safety_config import SafetyConfig
from src.safety_monitor import SafetyMonitor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    safety_config = SafetyConfig()
    input_dim = 64
    hidden_dim = 128
    engine = CognitiveProcessingEngine(input_dim, hidden_dim, safety_config)
    safety_monitor = SafetyMonitor()
    example_input = torch.randn(1, 10, input_dim).cuda()
    if torch.cuda.is_available():
        engine.encoder.cuda()
    output = engine.process_input(example_input)
    is_safe = safety_monitor.monitor(engine.state)
    if output is not None and is_safe:
        logger.info(f"Processed. Ethics: {engine.state.ethical_compliance:.2f}, Uncertainty: {engine.state.uncertainty:.2f}")
    else:
        logger.warning(f"Failed/Unsafe. Safety: {safety_monitor.safety_violations}, Ethics: {safety_monitor.ethical_violations}")

if __name__ == "__main__":
    main()