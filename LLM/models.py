from pydantic import BaseModel
from typing import List, Optional


class Event(BaseModel):
    event_type: str
    zone: Optional[str] = None
    timestamp: str
    duration: Optional[int] = None
    severity: str

class Statistics(BaseModel):
    persons: int
    forklifts: int
    pallets: int
    alerts: int
    
class RuleEngineData(BaseModel):
    time: str
    events: List[Event]
    statistics: Statistics
    total_count: Optional[int] = None

    
    
    