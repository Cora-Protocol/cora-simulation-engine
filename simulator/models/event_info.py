from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class EventInfo:
    message: str
    time: datetime
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)
