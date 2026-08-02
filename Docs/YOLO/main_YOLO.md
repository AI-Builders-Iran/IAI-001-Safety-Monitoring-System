# 🏭 Pipeline Orchestrator - Complete Usage Guide

## 📋 Overview

The `pipeline_orchestrator.py` script manages the complete safety monitoring pipeline:

```
Video File
    ↓
Detection Module (YOLO + Tracking)
    ↓
tracking_complete.json (intermediate)
    ↓
Rule Engine (Safety Rules Analysis)
    ↓
alerts_report.json (final output)
    ↓
LLM Report Generator or Database (optional)
```

---

## 🚀 Quick Start

### Installation

1. Place these files in your project directory:
```
project/
├── pipeline_orchestrator.py      ← Main orchestrator
├── info_detect.py                ← Detection module
├── rules_engine.py               ← Rule engine module
├── models/
│   └── best.pt                   ← YOLO weights
└── videos/
    └── warehouse_video.mp4       ← Input video
```

2. Install dependencies:
```bash
pip install opencv-python ultralytics torch torchvision pyyaml
```

### Basic Usage

```bash
# Process video with defaults
python pipeline_orchestrator.py

# Process specific video
python pipeline_orchestrator.py --video warehouse_video.mp4

# Use webcam (live feed)
python pipeline_orchestrator.py --video 0

# Custom model path
python pipeline_orchestrator.py --model models/best.pt
```

---

## 📊 Command-Line Arguments

### Required Arguments
None - all have defaults in CONFIG dictionary

### Optional Arguments

#### `--video` 
Input video file path or device ID

**Type:** `str`  
**Default:** `warehouse_video.mp4`  
**Values:**
- Path to MP4/AVI file: `warehouse.mp4`
- Webcam: `0`
- Other device: `1`, `2`, etc.

**Examples:**
```bash
# From file
--video /path/to/warehouse.mp4

# From webcam
--video 0

# Relative path
--video videos/sample.mp4
```

#### `--model`
Path to YOLO model weights

**Type:** `str`  
**Default:** `models/best.pt`  
**Note:** File must exist; validation will fail otherwise

**Examples:**
```bash
--model models/best.pt
--model /absolute/path/to/weights.pt
--model yolov8n.pt  # YOLOv8 Nano
```

#### `--output-tracking`
Path to output tracking JSON (intermediate file)

**Type:** `str`  
**Default:** `tracking_complete.json`  
**Note:** This is the detection module's output; can be reused

**Examples:**
```bash
--output-tracking tracking_data.json
--output-tracking outputs/tracking_2024_01_15.json
```

#### `--output-alerts`
Path to final alerts JSON report

**Type:** `str`  
**Default:** `alerts_report.json`  
**Note:** Main output file; typically used by LLM or database

**Examples:**
```bash
--output-alerts alerts.json
--output-alerts outputs/safety_alerts_2024_01_15.json
```

#### `--conf`
Detection confidence threshold

**Type:** `float`  
**Range:** `0.0` to `1.0`  
**Default:** `0.3`  
**Typical Values:**
- `0.3`: Aggressive (more detections, more false positives)
- `0.5`: Balanced
- `0.7`: Conservative (fewer detections, higher accuracy)

**Examples:**
```bash
--conf 0.3   # Aggressive detection
--conf 0.5   # Balanced
--conf 0.7   # Conservative
```

#### `--iou`
Non-Max Suppression IOU threshold

**Type:** `float`  
**Range:** `0.0` to `1.0`  
**Default:** `0.45`  
**Purpose:** Removes overlapping boxes during detection

**Examples:**
```bash
--iou 0.3    # Aggressive suppression
--iou 0.45   # Standard
--iou 0.6    # Lenient suppression
```

---

## 💡 Usage Examples

### Example 1: Basic Processing
```bash
python pipeline_orchestrator.py
```
**What it does:**
- Uses default video path
- Uses default model
- Generates tracking_complete.json
- Generates alerts_report.json

**Output:**
```
======================================================================
  🏭 Warehouse Safety Monitoring Pipeline
======================================================================

ℹ️  Input video: warehouse_video.mp4
ℹ️  Output: tracking_complete.json

✅ Detection & Tracking stage completed
✅ Valid tracking JSON (1500 frames)

ℹ️  Input: tracking_complete.json
ℹ️  Output: alerts_report.json

✅ Rule Engine stage completed
✅ Valid alerts JSON (147 alerts across 8 types)

======================================================================
  📊 Pipeline Results Summary
======================================================================

📈 Total Violations Detected: 147

📋 Violations by Type:
   • vehicle_person_proximity: 23
   • person_no_ppe: 18
   • machinery_person_proximity: 12
   ...

✅ Pipeline completed successfully!
```

### Example 2: Custom Video and Output
```bash
python pipeline_orchestrator.py \
  --video warehouse_2024_01_15.mp4 \
  --output-tracking data/tracking_jan15.json \
  --output-alerts data/alerts_jan15.json
```

### Example 3: Webcam Live Processing
```bash
python pipeline_orchestrator.py \
  --video 0 \
  --output-tracking live_tracking.json \
  --output-alerts live_alerts.json
```

**Note:** For webcam, pipeline will process video until stopped (Ctrl+C)

### Example 4: Aggressive Safety Settings
```bash
python pipeline_orchestrator.py \
  --video warehouse.mp4 \
  --conf 0.4 \
  --iou 0.35
```

**Effect:** More detections, potentially more false positives, but catches more violations

### Example 5: Conservative Settings
```bash
python pipeline_orchestrator.py \
  --video warehouse.mp4 \
  --conf 0.7 \
  --iou 0.55
```

**Effect:** Fewer but more confident detections, fewer false positives

### Example 6: Batch Processing Multiple Videos
```bash
#!/bin/bash
# process_all.sh

for video in videos/*.mp4; do
  echo "Processing: $video"
  python pipeline_orchestrator.py \
    --video "$video" \
    --output-tracking "tracking_${video%.*}.json" \
    --output-alerts "alerts_${video%.*}.json"
done
```

**Run with:**
```bash
bash process_all.sh
```

### Example 7: Python Integration
```python
from pipeline_orchestrator import SafetyMonitoringPipeline

# Initialize
pipeline = SafetyMonitoringPipeline()

# Execute
success = pipeline.execute(
    video_path="warehouse.mp4",
    model_path="models/best.pt",
    output_tracking="tracking.json",
    output_alerts="alerts.json",
    conf_threshold=0.5,
    iou_threshold=0.45
)

if success:
    # Load results
    report = pipeline.load_alerts_report("alerts.json")
    print(f"Total alerts: {report['total_alerts']}")
    
    # Process further
    for alert in report['alerts'][:10]:
        print(f"• {alert['message']}")
```

---

## 📤 Output Files

### 1. tracking_complete.json
**Intermediate output from Detection Module**

```json
{
    "metadata": {
        "fps": 30,
        "total_frames": 1500,
        "classes": {
            "0": "Person",
            "1": "vehicle",
            "2": "machinery"
        }
    },
    "frames": [
        {
            "frame_id": 0,
            "detections": [
                {
                    "track_id": 1,
                    "class_id": 0,
                    "class_name": "Person",
                    "confidence": 0.95,
                    "bbox": [100, 150, 200, 400],
                    "center": {"x": 150, "y": 275},
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

**Usage:** 
- Can be reused with different rule configurations
- Can be visualized for debugging
- Can be analyzed separately

### 2. alerts_report.json
**Final output from Rule Engine**

```json
{
    "total_alerts": 147,
    "alerts_by_type": {
        "vehicle_person_proximity": 23,
        "person_no_ppe": 18,
        "machinery_person_proximity": 12,
        "person_no_hardhat": 15,
        "person_no_vest": 20,
        "person_idle": 31,
        "crowd_detection": 18,
        "high_speed": 14
    },
    "severity_distribution": {
        "critical": 12,
        "high": 34,
        "medium": 65,
        "low": 36
    },
    "alerts": [
        {
            "type": "vehicle_person_proximity",
            "frame_id": 120,
            "track_id": 5,
            "vehicle_id": 3,
            "distance_m": 2.5,
            "severity": "high",
            "message": "Person 5 approached vehicle 3! Distance: 2.5m"
        },
        ...
    ]
}
```

**Usage:**
- Input to LLM Report Generator
- Store in database
- Create visualizations
- Generate compliance reports

---

## 🔧 Configuration

### Method 1: Edit DEFAULT_CONFIG (Permanent)

Edit `pipeline_orchestrator.py`:
```python
DEFAULT_CONFIG = {
    "DETECT_SCRIPT": "info_detect.py",
    "RULE_ENGINE_SCRIPT": "rules_engine.py",
    "MODEL_PATH": "models/best.pt",
    "VIDEO_PATH": "warehouse_video.mp4",
    "TRACKING_JSON": "tracking_complete.json",
    "ALERTS_JSON": "alerts_report.json",
    "CONF_THRESHOLD": 0.3,
    "IOU_THRESHOLD": 0.45,
}
```

### Method 2: Use Command-Line Arguments (Temporary)

```bash
python pipeline_orchestrator.py \
  --video video.mp4 \
  --model best.pt \
  --conf 0.4 \
  --iou 0.5
```

### Method 3: Python Configuration Object (Programmatic)

```python
from pipeline_orchestrator import SafetyMonitoringPipeline

custom_config = {
    "DETECT_SCRIPT": "detection.py",
    "RULE_ENGINE_SCRIPT": "rules.py",
    "MODEL_PATH": "models/best.pt",
    "VIDEO_PATH": "video.mp4",
    "TRACKING_JSON": "tracking.json",
    "ALERTS_JSON": "alerts.json",
    "CONF_THRESHOLD": 0.4,
    "IOU_THRESHOLD": 0.5,
}

pipeline = SafetyMonitoringPipeline(config=custom_config)
pipeline.execute(video_path="test.mp4", model_path="best.pt")
```

---

## 🐛 Troubleshooting

### Issue: "File not found"
```
❌ Video file not found: /path/to/warehouse.mp4
```

**Solution:**
```bash
# Check file exists
ls -la warehouse.mp4

# Use absolute path
python pipeline_orchestrator.py --video /absolute/path/video.mp4

# Use relative path from current directory
python pipeline_orchestrator.py --video ./videos/warehouse.mp4
```

### Issue: "YOLO model not found"
```
❌ YOLO model not found: models/best.pt
```

**Solution:**
```bash
# Create models directory
mkdir -p models

# Place model there
cp /path/to/best.pt models/

# Verify
ls -la models/best.pt
```

### Issue: "Script not found"
```
❌ Detection & Tracking script not found: info_detect.py
```

**Solution:**
```bash
# Check files in current directory
ls -la *.py

# Ensure these exist:
# - pipeline_orchestrator.py
# - info_detect.py
# - rules_engine.py

# Run from correct directory
cd /path/to/project
python pipeline_orchestrator.py
```

### Issue: "Validation failed - missing keys"
```
❌ Missing required keys in tracking JSON: ['metadata', 'frames']
```

**Solution:**
- Verify detection module is generating correct format
- Check intermediate JSON file manually
- Ensure detection module completed successfully

### Issue: Webcam not working
```
❌ Cannot open video source (device 0)
```

**Solution:**
```bash
# Check if webcam is accessible
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different device ID
python pipeline_orchestrator.py --video 1  # Try device 1

# On Linux, check permissions
ls -la /dev/video*
```

---

## 📊 Integration with Other Modules

### With LLM Report Generator
```python
from pipeline_orchestrator import SafetyMonitoringPipeline
from llm_report_generator import HSEReportGenerator

# 1. Run pipeline
pipeline = SafetyMonitoringPipeline()
pipeline.execute(video_path="warehouse.mp4", model_path="best.pt")

# 2. Load alerts
alerts_report = pipeline.load_alerts_report("alerts_report.json")

# 3. Generate LLM report
llm_gen = HSEReportGenerator()
persian_report = llm_gen.generate_report(alerts_report)

# 4. Save
with open("hse_report.md", "w", encoding='utf-8') as f:
    f.write(persian_report)
```

### With Database
```python
from pipeline_orchestrator import SafetyMonitoringPipeline
from database import SafetyAlertsDB

pipeline = SafetyMonitoringPipeline()
pipeline.execute(video_path="warehouse.mp4", model_path="best.pt")

db = SafetyAlertsDB()
report = pipeline.load_alerts_report("alerts_report.json")

for alert in report['alerts']:
    db.insert_alert(alert)

print(f"Inserted {len(report['alerts'])} alerts into database")
```

### With Visualization
```python
from pipeline_orchestrator import SafetyMonitoringPipeline
from visualization import AlertVisualizer

pipeline = SafetyMonitoringPipeline()
pipeline.execute(video_path="warehouse.mp4", model_path="best.pt")

viz = AlertVisualizer()
viz.create_report("alerts_report.json", output_html="report.html")
```

---

## 🎯 Best Practices

### 1. Directory Structure
```
project/
├── pipeline_orchestrator.py
├── info_detect.py
├── rules_engine.py
├── models/
│   └── best.pt
├── videos/
│   ├── warehouse_2024_01_15.mp4
│   └── warehouse_2024_01_20.mp4
├── outputs/
│   ├── tracking_2024_01_15.json
│   ├── alerts_2024_01_15.json
│   ├── tracking_2024_01_20.json
│   └── alerts_2024_01_20.json
└── README.md
```

### 2. Configuration Management
```bash
# Create config file
cat > pipeline_config.txt << EOF
VIDEO_PATH=videos/warehouse.mp4
MODEL_PATH=models/best.pt
CONF_THRESHOLD=0.4
IOU_THRESHOLD=0.45
EOF

# Use in script
source pipeline_config.txt
python pipeline_orchestrator.py --video $VIDEO_PATH --model $MODEL_PATH
```

### 3. Error Handling
```bash
#!/bin/bash

if python pipeline_orchestrator.py --video warehouse.mp4; then
    echo "✅ Pipeline succeeded"
    # Process alerts
else
    echo "❌ Pipeline failed"
    exit 1
fi
```

### 4. Logging to File
```bash
python pipeline_orchestrator.py \
  --video warehouse.mp4 \
  2>&1 | tee pipeline_log.txt
```

### 5. Performance Monitoring
```python
import time
from pipeline_orchestrator import SafetyMonitoringPipeline

start = time.time()
pipeline = SafetyMonitoringPipeline()
pipeline.execute(video_path="warehouse.mp4", model_path="best.pt")
elapsed = time.time() - start

print(f"Total execution time: {elapsed:.2f} seconds")
```

---

## 📝 Notes

- **Output files are overwritten**: Each run overwrites previous outputs with same names
- **Intermediate files**: tracking_complete.json is kept after pipeline for debugging
- **Error recovery**: If detection fails, rule engine won't run (fail-fast)
- **Validation**: Both outputs are validated before pipeline completes
- **Performance**: Larger videos take longer; consider resolution/FPS tradeoff

---

## 🔗 Next Steps

1. ✅ Run pipeline: `python pipeline_orchestrator.py`
2. ✅ Check outputs: `alerts_report.json`
3. ✅ Integrate with LLM: Pass alerts to report generator
4. ✅ Archive results: Store tracking & alerts JSON
5. ✅ Monitor database: Track violations over time

---

## 📞 Support

For issues with:
- **Detection module**: See `info_detect.py` documentation
- **Rule engine**: See `rules_engine.py` documentation
- **Orchestrator**: Check paths and command-line arguments

Enjoy! 🚀