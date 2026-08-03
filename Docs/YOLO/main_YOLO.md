# Work Report — Video Processing & LLM Integration Layer
## Warehouse Safety Monitoring System

**Team:** AI Builders Iran — LLM Sub-team
**Scope:** Video ingestion, detection/tracking, rule-based safety analysis, and the interface layer feeding the LLM report generator
**Prepared for:** Team Lead / Project Manager

---

## 1. Objective

The LLM sub-team is responsible for turning raw warehouse video footage into a structured, natural-language HSE (Health, Safety, Environment) report. This requires a reliable data pipeline in front of the LLM: video goes in, a clean JSON object describing every safety violation comes out, and that JSON is what the LLM's prompt generator consumes.

This report summarizes the design and implementation of that pipeline: a single, self-contained module that a caller can hand a video file to and receive a ready-to-use JSON report from, with no manual steps in between.

---

## 2. Pipeline Overview

```
Video file
    │
    ▼
Stage 1 — Detection & Tracking (YOLO)
    Runs the trained YOLO model frame-by-frame with object tracking enabled.
    Produces bounding boxes, class labels, track IDs, and PPE (hardhat/vest)
    association per person.
    │
    ▼
Stage 2 — Rule Engine
    Applies the team's safety rules (proximity to vehicles/machinery, missing
    PPE, idle workers, crowding, abnormal speed) to the frame-by-frame data
    and produces a structured list of alerts with severity levels.
    │
    ▼
Stage 3 — JSON Assembly
    Combines video metadata, per-object tracking summary, and the alerts
    report into a single JSON-serializable dictionary.
    │
    ▼
Output JSON → handed directly to the LLM prompt generator (HSEPromptGenerator)
```

All three stages run in a single Python process, in memory, with no intermediate files required. This was an intentional design choice: it removes I/O overhead, avoids file-path coordination issues between stages, and keeps the interface to the LLM layer to a single function call.

---

## 3. Components

### 3.1 Detection & Tracking (`YOLODetector`)

- Loads the trained YOLO model once (not per video), so repeated calls in a running service don't pay model-loading cost every time.
- Runs detection + multi-object tracking on every frame using Ultralytics' `.track()` API with configurable confidence and IoU thresholds.
- For every detected person, checks bounding-box overlap against hardhat and safety-vest detections to determine PPE compliance (`has_hardhat`, `has_vest`, `has_ppe`).
- Tracks each object's first/last-seen frame and trajectory across the video, to compute how long objects were present and how far/fast they moved — this feeds duration- and speed-based rules downstream.
- Adds frame-relative position/size to every detection (0–1 range) so rules can reason about zones independent of video resolution.
- Output schema (per frame): frame ID, timestamp, frame size, and a list of detections with bounding box, center, class, confidence, track ID, and PPE flags.

### 3.2 Rule Engine

The rule engine (existing team component) evaluates the following safety conditions per frame and accumulates alerts with a severity level (`low` / `medium` / `high` / `critical`):

| Rule | Trigger | Severity |
|---|---|---|
| Vehicle–person proximity | Person within unsafe distance of a vehicle | high |
| Machinery–person proximity | Machinery within unsafe distance of a person | critical |
| Missing PPE (general) | Person with no PPE for N consecutive frames | high |
| Missing hardhat | Person with no hardhat for N consecutive frames | medium |
| Missing vest | Person with no safety vest for N consecutive frames | medium |
| Idle worker | Person stationary in place beyond a time threshold | low |
| Crowd detection | Too many people in one frame/zone | medium |
| Abnormal speed | Object moving faster than the configured threshold | medium |

Distance thresholds are calibrated in real-world meters and converted to pixels via a configurable pixels-per-meter factor, so the same rule set can be reused across cameras with different framing.

### 3.3 Pipeline Wrapper (`SafetyVideoPipeline`)

This is the component the LLM sub-team built specifically to close the gap between the detection module and the rule engine, and to give the LLM layer one clean entry point:

- Wires the detector and rule engine together behind a single method, `process_video(video_path)`.
- Resets rule-engine state at the start of every video, so alerts from a previous run never leak into a new one.
- Returns one dictionary with three top-level sections: `video_info`, `tracking_summary`, and `alerts_report` — exactly what the LLM prompt generator needs to build the final HSE report.
- Optionally persists intermediate tracking data and the alerts report to disk for debugging/auditing, without requiring it for normal operation.
- Designed to be instantiated once per service lifetime (so the model loads once) and reused for every incoming video.

### 3.4 Output Format

```json
{
  "video_info": {
    "source": "warehouse.mp4",
    "fps": 30,
    "total_frames": 900,
    "processed_at": "2026-08-03T12:00:00"
  },
  "tracking_summary": {
    "object_durations": { "12": {"class": "Person", "duration_seconds": 24.3} },
    "object_movement": { "12": {"total_distance": 812.4, "avg_speed": 3.1} },
    "classes": { "0": "Hardhat", "5": "Person", "9": "vehicle" }
  },
  "alerts_report": {
    "total_alerts": 12,
    "alerts_by_type": {"machinery_person_proximity": 3, "person_no_hardhat": 9},
    "severity_distribution": {"critical": 3, "medium": 9},
    "alerts": [
      {
        "type": "machinery_person_proximity",
        "frame_id": 145,
        "track_id": 12,
        "machinery_id": 7,
        "distance_m": 0.62,
        "severity": "critical",
        "message": "Machinery 7 approached person 12! Distance: 0.62m"
      }
    ]
  }
}
```

This structure is stable and self-describing, which keeps prompt-template design on the LLM side decoupled from changes in the detection/rule-engine internals.

---

## 4. Usage

Single-call usage (simplest integration point):

```python
from llm_pipeline_wrapper import run_pipeline_for_llm

result = run_pipeline_for_llm(
    video_path="warehouse.mp4",
    model_path="models/best.pt",
)
```

Service-style usage (recommended for production, avoids reloading the model per request):

```python
from llm_pipeline_wrapper import SafetyVideoPipeline

pipeline = SafetyVideoPipeline(detector_config={"model_path": "models/best.pt"})

result_1 = pipeline.process_video("video1.mp4")
result_2 = pipeline.process_video("video2.mp4")
```

The returned `result` dictionary is passed directly into the LLM prompt-generation step (`HSEPromptGenerator`) to produce the final Persian-language HSE report.

---

## 5. Quality & Testing

| Check | Method | Result |
|---|---|---|
| Module syntax | `py_compile` | Passed |
| Component isolation | Instantiated `YOLODetector` and `SafetyVideoPipeline` with mocked YOLO/OpenCV dependencies | Passed — classes build and wire together correctly |
| End-to-end logic | Fed synthetic frame data (a person placed next to machinery) through the rule engine | Passed — correctly produced a `machinery_person_proximity` alert with `critical` severity |

During implementation, a configuration bug was found and fixed in the existing rule-engine module: a stray docstring inside the default configuration dictionary was silently merging with the first setting's key (due to Python's automatic string concatenation), which caused the rule engine to fail on startup with a `KeyError`. This has been corrected; the rule engine now initializes and applies distance thresholds correctly. Details are documented separately for the code review.

Full end-to-end validation with a real video and the trained model weights is the remaining step — it wasn't possible in the review environment due to missing `ultralytics`/`opencv-python`, but the detection logic was ported directly from the team's existing, already-validated detection script, so no behavioral changes are expected.

---

## 6. Status & Next Steps

**Status:** Detection, tracking, rule engine, and the unifying wrapper are implemented, wired together, and unit-verified. Ready for integration testing with real footage.

**Next steps:**
1. Run the pipeline against real warehouse footage with the production model weights to validate detection quality and alert accuracy end-to-end.
2. Connect `SafetyVideoPipeline`'s output directly into `HSEPromptGenerator` and the Jinja2 report templates.
3. Decide on deployment shape — whether the pipeline runs as a long-lived service (model loaded once) or a per-job script — and confirm resource/VRAM budget accordingly.
4. Add basic logging/metrics around processing time per video and alert counts, for monitoring once this goes into the broader CV/backend pipeline.

---

## 7. Deliverables

- `llm_pipeline_wrapper.py` — the detection + rule-engine + JSON-assembly module described above.
- Corrected rule-engine configuration file, ready to replace the existing one in the repository.