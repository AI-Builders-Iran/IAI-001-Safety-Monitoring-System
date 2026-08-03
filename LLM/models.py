"""Pydantic data models for the Video Analytics -> LLM Safety Report pipeline.

This module replaces the previous `models.py` (RuleEngineData / Event /
Statistics). It mirrors the JSON structure actually produced by the
video-tracking / rule-engine pipeline (video_info, tracking_summary,
alerts_report), so it can be used directly as:

    report = PipelineReport.model_validate(json_data)

and then passed into `HSEPromptGenerator` (prompt.py) for LLM prompt
building, after updating that module's type hints to `PipelineReport`.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# video_info
# ---------------------------------------------------------------------------
class VideoInfo(BaseModel):
    source: str
    fps: int
    total_frames: int
    processed_at: datetime


# ---------------------------------------------------------------------------
# tracking_summary
# ---------------------------------------------------------------------------
class ObjectDuration(BaseModel):
    class_: str = Field(alias="class")
    duration_seconds: float

    class Config:
        populate_by_name = True


class ObjectMovement(BaseModel):
    total_distance: float
    avg_speed: float


class TrackingSummary(BaseModel):
    # keys are stringified track ids, e.g. "12"
    object_durations: Dict[str, ObjectDuration]
    object_movement: Dict[str, ObjectMovement]
    # keys are stringified class ids, e.g. "0" -> "Hardhat"
    classes: Dict[str, str]


# ---------------------------------------------------------------------------
# alerts_report
# ---------------------------------------------------------------------------
class Alert(BaseModel):
    type: str
    frame_id: int
    track_id: int
    severity: str
    message: str
    # Present only for machinery/person proximity-type alerts; optional so
    # other alert types (e.g. person_no_hardhat) validate without them.
    machinery_id: Optional[int] = None
    distance_m: Optional[float] = None


class AlertsReport(BaseModel):
    total_alerts: int
    alerts_by_type: Dict[str, int]
    severity_distribution: Dict[str, int]
    alerts: List[Alert]


# ---------------------------------------------------------------------------
# top-level document
# ---------------------------------------------------------------------------
class PipelineReport(BaseModel):
    """Top-level object matching the pipeline's output JSON exactly."""
    video_info: VideoInfo
    tracking_summary: TrackingSummary
    alerts_report: AlertsReport
