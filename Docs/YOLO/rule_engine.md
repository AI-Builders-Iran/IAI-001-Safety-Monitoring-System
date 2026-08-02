# 🏭 Warehouse Safety Rule Engine

A modular, production-ready rule-based system for detecting safety violations in warehouse/industrial environments through video analysis.

## 📋 Overview

The `RuleEngine` class processes frame-by-frame detection data (from YOLO + tracking pipelines) and applies intelligent safety rules to generate structured alerts. It's designed to integrate seamlessly with downstream modules like LLM report generators, database systems, and visualization tools.

### Key Features

✅ **8 Safety Rules**
- Vehicle-Person Proximity Detection
- Machinery-Person Proximity Detection  
- PPE (Personal Protective Equipment) Violations
- Safety Hardhat Detection
- Safety Vest Detection
- Worker Idle Time Detection
- Crowd Density Monitoring
- Abnormal Movement Detection

✅ **Production-Ready**
- Full English docstrings (Google style)
- Type hints for all functions
- Comprehensive error handling
- State management for temporal analysis
- Configurable thresholds
- Duplicate alert prevention

✅ **Modular Architecture**
- Clean API for upstream/downstream integration
- Stateless frame processing option
- Batch processing support
- Real-time streaming ready

---

## 🚀 Quick Start

### Basic Usage

```python
from rules_engine import RuleEngine

# Initialize engine
engine = RuleEngine()

# Process video tracking data
engine.process_video('tracking_complete.json')

# Get structured report
report = engine.get_report()

# Display formatted output
engine.print_report()

# Save to JSON file
engine.save_report('alerts_report.json')
```

### Expected Output

```
======================================================================
📊 SAFETY MONITORING REPORT
======================================================================

📈 Total Alerts: 147

📊 Alerts by Type:
   • vehicle_person_proximity: 23
   • person_no_ppe: 18
   • machinery_person_proximity: 12
   • ...

⚠️ Severity Distribution:
   • critical: 12
   • high: 34
   • medium: 65
   • low: 36

📋 Sample Alerts (first 10 of 147):
   [1] [critical] Machinery 5 approached person 12! Distance: 0.85m
   [2] [high] Person 12 approached vehicle 3! Distance: 2.75m
   ...
======================================================================
```

---

## 📚 API Reference

### Initialization

#### `RuleEngine(config: Optional[Dict[str, Any]] = None)`

Initialize the rule engine with optional custom configuration.

**Parameters:**
- `config` (dict, optional): Custom configuration dictionary with thresholds. If None, uses DEFAULT_CONFIG.

**Example:**
```python
# Using default configuration
engine = RuleEngine()

# Using custom configuration (stricter thresholds)
custom_config = {
    "distance_vehicle_person_m": 2.0,  # 2m instead of 3m
    "distance_machinery_person_m": 0.5,  # 0.5m instead of 1m
    "no_ppe_frames": 3,                # 3 frames instead of 5
    "crowd_threshold": 2,               # 2 people instead of 3
}
engine = RuleEngine(config=custom_config)
```

---

### Main Methods

#### `process_video(video_data_path: str) -> None`

Load and process complete video tracking data from JSON file.

**Parameters:**
- `video_data_path`: Path to JSON file containing frame detection data

**Expected JSON Format:**
```json
{
    "metadata": {
        "fps": 30,
        "classes": {...}
    },
    "frames": [
        {
            "frame_id": 0,
            "detections": [
                {
                    "track_id": 1,
                    "class_name": "Person",
                    "center": {"x": 100, "y": 200},
                    "has_ppe": true,
                    "has_hardhat": true,
                    "has_vest": true
                },
                ...
            ]
        },
        ...
    ]
}
```

**Example:**
```python
engine = RuleEngine()
engine.process_video('warehouse_tracking.json')
```

---

#### `process_frame(frame_data: Dict[str, Any]) -> List[Dict[str, Any]]`

Process a single frame and apply all safety rules.

**Parameters:**
- `frame_data`: Dictionary containing `frame_id` and `detections`

**Returns:**
- List of new alerts generated for this frame

**Use Case:** Real-time/streaming video processing

**Example:**
```python
engine = RuleEngine()

# Process frames as they arrive from camera/stream
frame = {
    "frame_id": 0,
    "detections": [...]
}

alerts = engine.process_frame(frame)
if alerts:
    print(f"⚠️ {len(alerts)} violations detected")
```

---

#### `get_report() -> Dict[str, Any]`

Generate comprehensive alert report.

**Returns:**
```python
{
    "total_alerts": int,
    "alerts_by_type": {
        "vehicle_person_proximity": int,
        "person_no_ppe": int,
        ...
    },
    "severity_distribution": {
        "critical": int,
        "high": int,
        "medium": int,
        "low": int
    },
    "alerts": [
        {
            "type": str,
            "frame_id": int,
            "track_id": int,
            "severity": str,
            "message": str,
            ...  # Type-specific fields
        },
        ...
    ]
}
```

**Example:**
```python
report = engine.get_report()
print(f"Total alerts: {report['total_alerts']}")

# Access specific violation type
ppe_violations = report['alerts_by_type'].get('person_no_ppe', 0)
print(f"PPE violations: {ppe_violations}")

# Iterate all alerts
for alert in report['alerts']:
    print(alert['message'])
```

---

#### `print_report(max_alerts: int = 10) -> None`

Print formatted report to console.

**Parameters:**
- `max_alerts`: Maximum individual alerts to display (default: 10)

**Example:**
```python
engine.print_report(max_alerts=20)
```

---

#### `save_report(output_path: str) -> None`

Save alert report to JSON file.

**Parameters:**
- `output_path`: Path where JSON should be saved

**Example:**
```python
engine.save_report('alerts_report.json')
# Output file can be used by downstream modules (LLM, visualization, etc.)
```

---

#### `load_config(config_path: str) -> None`

Load configuration from JSON file.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Example:**
```python
engine = RuleEngine()
engine.load_config('custom_config.json')  # Update thresholds
engine.process_video('tracking.json')
```

---

#### `reset() -> None`

Clear all accumulated alerts and internal state.

**Use Case:** Processing multiple videos sequentially

**Example:**
```python
engine = RuleEngine()

# Process first video
engine.process_video('warehouse1.json')
report1 = engine.get_report()

# Reset for second video
engine.reset()

# Process second video
engine.process_video('warehouse2.json')
report2 = engine.get_report()
```

---

## ⚙️ Configuration

### Default Configuration

```python
DEFAULT_CONFIG = {
    # Distance thresholds (meters)
    "distance_vehicle_person_m": 3.0,
    "distance_machinery_person_m": 1.0,
    "pixels_per_meter": 50,
    
    # Time thresholds (frames)
    "no_ppe_frames": 5,
    "no_hardhat_frames": 3,
    "no_vest_frames": 3,
    "idle_frames": 150,
    "crowd_threshold": 3,
    
    # Speed thresholds (pixels/frame)
    "speed_threshold": 10,
}
```

### Custom Configuration File

Create `config.json`:
```json
{
    "distance_vehicle_person_m": 2.5,
    "distance_machinery_person_m": 0.8,
    "pixels_per_meter": 50,
    "no_ppe_frames": 4,
    "no_hardhat_frames": 2,
    "no_vest_frames": 2,
    "idle_frames": 120,
    "crowd_threshold": 2,
    "speed_threshold": 8
}
```

Load it:
```python
engine = RuleEngine()
engine.load_config('config.json')
```

---

## 🧠 Alert Types and Fields

Each alert contains:
- `type`: Violation type name
- `frame_id`: Frame where violation occurred
- `severity`: One of "critical", "high", "medium", "low"
- `message`: Human-readable description
- Type-specific fields (see below)

### 1. Vehicle-Person Proximity
```python
{
    "type": "vehicle_person_proximity",
    "track_id": 12,              # Person track ID
    "vehicle_id": 5,             # Vehicle track ID
    "distance_m": 2.75,
    "distance_px": 137.5,
    "severity": "high",
    "message": "Person 12 approached vehicle 5! Distance: 2.75m"
}
```

### 2. Machinery-Person Proximity
```python
{
    "type": "machinery_person_proximity",
    "track_id": 12,
    "machinery_id": 8,
    "distance_m": 0.85,
    "distance_px": 42.5,
    "severity": "critical",
    "message": "Machinery 8 approached person 12! Distance: 0.85m"
}
```

### 3. PPE Violation
```python
{
    "type": "person_no_ppe",
    "track_id": 12,
    "duration_frames": 5,
    "severity": "high",
    "message": "Person 12 detected without PPE for 5 frames!"
}
```

### 4. No Hardhat
```python
{
    "type": "person_no_hardhat",
    "track_id": 12,
    "duration_frames": 3,
    "severity": "medium",
    "message": "Person 12 detected without hardhat for 3 frames!"
}
```

### 5. No Safety Vest
```python
{
    "type": "person_no_vest",
    "track_id": 12,
    "duration_frames": 3,
    "severity": "medium",
    "message": "Person 12 detected without safety vest for 3 frames!"
}
```

### 6. Idle Worker
```python
{
    "type": "person_idle",
    "track_id": 12,
    "idle_seconds": 5.3,
    "severity": "low",
    "message": "Person 12 idle for 5.3 seconds"
}
```

### 7. Crowd Detection
```python
{
    "type": "crowd_detection",
    "person_count": 5,
    "threshold": 3,
    "severity": "medium",
    "message": "Crowd detected! 5 people (threshold: 3)"
}
```

### 8. High Speed Movement
```python
{
    "type": "high_speed",
    "track_id": 8,
    "class_name": "machinery",
    "speed": 12.5,
    "severity": "medium",
    "message": "Object 8 (machinery) moving at high speed: 12.5 px/frame"
}
```

---

## 🔗 Integration with Downstream Modules

### Pattern 1: LLM Report Generation

```python
from rules_engine import RuleEngine
from llm_report_generator import ReportGenerator

# Process video
engine = RuleEngine()
engine.process_video('tracking.json')

# Pass alerts to LLM generator
report_gen = ReportGenerator(engine.get_report())
natural_language_report = report_gen.generate()
```

### Pattern 2: Database Storage

```python
from rules_engine import RuleEngine
from database import AlertsDB

engine = RuleEngine()
engine.process_video('tracking.json')

db = AlertsDB()
for alert in engine.alerts:
    db.insert_alert(alert)
```

### Pattern 3: Real-time Alerting

```python
from rules_engine import RuleEngine
from alerting_service import AlertService

engine = RuleEngine()
alert_service = AlertService()

# Process frame stream
for frame in video_stream:
    alerts = engine.process_frame(frame)
    
    for alert in alerts:
        if alert['severity'] in ['critical', 'high']:
            alert_service.send_alert(alert)
```

### Pattern 4: Batch Processing

```python
from rules_engine import RuleEngine
import glob

all_reports = {}

for video_file in glob.glob('warehouse_videos/*.json'):
    engine = RuleEngine()
    engine.process_video(video_file)
    all_reports[video_file] = engine.get_report()
```

---

## 📊 Usage Examples

### Example 1: Simple Pipeline

```python
from rules_engine import RuleEngine

engine = RuleEngine()
engine.process_video('tracking_complete.json')
engine.print_report()
engine.save_report('alerts.json')
```

### Example 2: Custom Thresholds

```python
from rules_engine import RuleEngine

config = {
    "distance_vehicle_person_m": 2.0,
    "distance_machinery_person_m": 0.5,
    "crowd_threshold": 2,
}

engine = RuleEngine(config=config)
engine.process_video('tracking.json')
report = engine.get_report()
```

### Example 3: Filter Alerts

```python
from rules_engine import RuleEngine

engine = RuleEngine()
engine.process_video('tracking.json')
report = engine.get_report()

# Get critical alerts
critical_alerts = [
    a for a in report['alerts'] 
    if a['severity'] == 'critical'
]

print(f"Critical violations: {len(critical_alerts)}")
for alert in critical_alerts:
    print(f"  • {alert['message']}")
```

### Example 4: Real-time Processing

```python
from rules_engine import RuleEngine

engine = RuleEngine()

# Process frames as they arrive
for frame in video_stream:
    alerts = engine.process_frame(frame)
    
    for alert in alerts:
        print(f"⚠️ {alert['message']}")
```

### Example 5: Multi-video Comparison

```python
from rules_engine import RuleEngine

videos = ['warehouse1.json', 'warehouse2.json', 'warehouse3.json']
reports = {}

for video in videos:
    engine = RuleEngine()
    engine.process_video(video)
    reports[video] = engine.get_report()
    engine.reset()

# Compare results
for video, report in reports.items():
    print(f"{video}: {report['total_alerts']} alerts")
```

---

## 🛠️ Development Notes

### Architecture

The RuleEngine follows a modular design:

1. **Initialization Phase**: Set up configuration and register rules
2. **Processing Phase**: Apply rules to frames sequentially
3. **State Management**: Track temporal information (counters, positions)
4. **Duplicate Prevention**: Avoid repeating identical alerts
5. **Report Generation**: Aggregate and structure results

### Extending with New Rules

To add a new safety rule:

```python
def _check_new_rule(self, frame_data):
    """
    New safety rule: [description]
    
    Args:
        frame_data: Current frame detection data
    
    Returns:
        List of alerts if violations detected
    """
    alerts = []
    # Your rule logic here
    return alerts

# Register the rule in __init__:
self.rules.append(("new_rule_name", self._check_new_rule))
```

### Performance Optimization

- **Batch Processing**: Process multiple videos in separate engine instances
- **Frame Sampling**: Skip frames for lower-precision requirements
- **Config Caching**: Load config once, reuse engine instance
- **State Reset**: Call `reset()` between videos to free memory

---

## 📝 Change Log

### v2.0 (Refactored)
- ✅ Comprehensive English docstrings (Google style)
- ✅ Type hints for all functions
- ✅ Modular API for downstream integration
- ✅ Improved configuration management
- ✅ Better error handling
- ✅ Severity levels for alerts
- ✅ Frame ID tracking in alerts
- ✅ Example integration patterns

### v1.0 (Original)
- Initial implementation with 8 safety rules
- Basic alert generation and reporting

---

## 📄 License & Support

This module is part of the Warehouse Safety Monitoring System.

For integration examples and troubleshooting, see `example_integration.py`

---

## 💡 Best Practices

1. **Initialize once, reuse many times**
   ```python
   engine = RuleEngine()
   for video in videos:
       engine.process_video(video)
       save_report(engine.get_report())
       engine.reset()
   ```

2. **Handle file errors gracefully**
   ```python
   try:
       engine.process_video('tracking.json')
   except FileNotFoundError:
       print("Video file not found!")
   ```

3. **Use appropriate configuration for your use case**
   - Stricter settings → Higher false positives, better safety
   - Looser settings → Fewer alerts, may miss violations

4. **Monitor report size** - Large videos generate many alerts
   - Consider filtering before saving to database
   - Batch process by time windows for large datasets

5. **Integrate with monitoring dashboards**
   - Stream alerts to visualization tools
   - Feed into LLM for natural language reports
   - Archive to database for compliance

---

## 🎯 Next Steps

1. **Prepare tracking data**: Use YOLO + tracking pipeline
2. **Configure thresholds**: Adjust for your warehouse
3. **Process videos**: Run `engine.process_video()`
4. **Integrate downstream**: Connect to LLM/database/alerting
5. **Monitor results**: Set up dashboards and reports

Enjoy! 🚀