from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

@dataclass
class CognitiveState:
    """Represents the system's internal awareness, tracking attention, uncertainty,
    safety, ethical stance, current focus, and short-term memory."""
    attention_focus: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    uncertainty: float = 0.0
    safety_status: bool = True
    ethical_compliance: float = 1.0
    current_task: Optional[str] = None
    working_memory: Dict[str, torch.Tensor] = field(default_factory=dict)
    last_processed_input_shape: Optional[Tuple[int, ...]] = None