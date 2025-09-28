from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RunSummary:
    run_id: str
    status: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags: Dict[str, Any] = None
