from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class ActionInfo:
    message: str
    agent_id: str
    time: datetime
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)
