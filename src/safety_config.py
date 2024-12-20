from dataclasses import dataclass

@dataclass
class SafetyConfig:
    """Configuration for safety parameters, defining acceptable operational boundaries."""
    min_ethical_compliance: float = 0.8
    max_uncertainty: float = 0.3
    memory_capacity: int = 1000
    max_processing_time: float = 1.0