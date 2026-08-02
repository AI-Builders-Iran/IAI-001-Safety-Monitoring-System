#!/usr/bin/env python
"""
Warehouse Safety Monitoring Pipeline Orchestrator.

This module orchestrates the complete safety monitoring pipeline:
    1. Detection & Tracking (YOLO + Tracking) → tracking JSON
    2. Rule Engine Analysis → alerts JSON
    3. Report Generation (optional LLM integration)

Pipeline Flow:
    Video Input
        ↓
    Detection Module (info_detect.py)
        ↓
    tracking_complete.json
        ↓
    Rule Engine (rules_engine.py)
        ↓
    alerts_report.json
        ↓
    LLM Report Generator (optional)
        ↓
    Final HSE Report

Configuration:
    Set file paths and parameters in the CONFIG dictionary below.
    Or pass them via command-line arguments.

Example Usage:
    # Using defaults from CONFIG
    python pipeline_orchestrator.py

    # Using custom video and model
    python pipeline_orchestrator.py \
        --video warehouse_video.mp4 \
        --model models/best.pt \
        --output-tracking tracking.json \
        --output-alerts alerts.json

    # Using webcam
    python pipeline_orchestrator.py --video 0
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Any, Optional

# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    """
    Pipeline Configuration:

    Paths:
        - DETECT_SCRIPT: Path to detection/tracking module (info_detect.py)
        - RULE_ENGINE_SCRIPT: Path to rule engine module (rules_engine.py)
        - MODEL_PATH: Path to YOLO model weights
        - VIDEO_PATH: Path to input video or device ID (0 for webcam)

    Output Files:
        - TRACKING_JSON: Intermediate output from detection module
        - ALERTS_JSON: Final output from rule engine

    Detection Parameters:
        - CONF_THRESHOLD: Detection confidence threshold (0.0-1.0)
        - IOU_THRESHOLD: Non-max suppression IOU threshold
    """

    # Script paths
    "DETECT_SCRIPT": "info_detect.py",
    "RULE_ENGINE_SCRIPT": "rules_engine.py",

    # Model path
    "MODEL_PATH": "models/best.pt",

    # Input video
    "VIDEO_PATH": "warehouse_video.mp4",

    # Output files
    "TRACKING_JSON": "tracking_complete.json",
    "ALERTS_JSON": "alerts_report.json",

    # Detection thresholds
    "CONF_THRESHOLD": 0.3,
    "IOU_THRESHOLD": 0.45,
}


# ============================================================
# Logger Class
# ============================================================

class PipelineLogger:
    """
    Unified logging for pipeline orchestration.

    Provides consistent formatting for status messages, errors, and progress.
    """

    @staticmethod
    def header(title: str, width: int = 70) -> None:
        """
        Print section header with formatting.

        Args:
            title: Header text
            width: Width of separator line
        """
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)

    @staticmethod
    def info(message: str) -> None:
        """Print informational message."""
        print(f"ℹ️  {message}")

    @staticmethod
    def success(message: str) -> None:
        """Print success message."""
        print(f"✅ {message}")

    @staticmethod
    def error(message: str) -> None:
        """Print error message."""
        print(f"❌ {message}")

    @staticmethod
    def warning(message: str) -> None:
        """Print warning message."""
        print(f"⚠️  {message}")

    @staticmethod
    def task(stage: int, title: str) -> None:
        """Print task progress."""
        print(f"\n📋 Stage {stage}: {title}")

    @staticmethod
    def step(message: str) -> None:
        """Print execution step."""
        print(f"   → {message}")


log = PipelineLogger()


# ============================================================
# Path Management
# ============================================================

class PathManager:
    """
    Centralized path management and validation.

    Handles absolute path resolution, existence checking, and
    directory creation for the pipeline.
    """

    @staticmethod
    def resolve(path: str) -> str:
        """
        Convert path to absolute path.

        Args:
            path: Relative or absolute path

        Returns:
            Absolute path as string
        """
        return os.path.abspath(path)

    @staticmethod
    def exists(path: str) -> bool:
        """
        Check if file exists.

        Args:
            path: File path to check

        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(PathManager.resolve(path))

    @staticmethod
    def validate_file(path: str, file_type: str = "file") -> bool:
        """
        Validate file existence and log result.

        Args:
            path: File path to validate
            file_type: Description of file type (for logging)

        Returns:
            True if validation passed, False otherwise
        """
        abs_path = PathManager.resolve(path)

        if not os.path.exists(abs_path):
            log.error(f"{file_type} not found: {abs_path}")
            return False

        if not os.path.isfile(abs_path):
            log.error(f"Path is not a file: {abs_path}")
            return False

        log.success(f"{file_type} found: {abs_path}")
        return True

    @staticmethod
    def ensure_directory(path: str) -> bool:
        """
        Create directory if it doesn't exist.

        Args:
            path: Directory path

        Returns:
            True if directory exists or was created
        """
        abs_path = PathManager.resolve(path)

        try:
            os.makedirs(abs_path, exist_ok=True)
            return True
        except Exception as e:
            log.error(f"Failed to create directory: {e}")
            return False

    @staticmethod
    def get_directory(file_path: str) -> str:
        """
        Get directory containing file.

        Args:
            file_path: Path to file

        Returns:
            Directory path
        """
        return os.path.dirname(PathManager.resolve(file_path))


# ============================================================
# Module Execution
# ============================================================

class PipelineStage:
    """
    Base class for pipeline stages (Detection, RuleEngine, etc).

    Handles subprocess execution, argument building, and error handling
    for individual pipeline components.
    """

    def __init__(self, script_path: str, stage_name: str):
        """
        Initialize pipeline stage.

        Args:
            script_path: Path to Python script for this stage
            stage_name: Human-readable name for logging
        """
        self.script_path = script_path
        self.stage_name = stage_name
        self.last_return_code = None

    def validate(self) -> bool:
        """
        Validate that script file exists.

        Returns:
            True if validation passed
        """
        return PathManager.validate_file(
            self.script_path,
            f"{self.stage_name} script"
        )

    def build_command(self, arguments: List[str]) -> List[str]:
        """
        Build subprocess command.

        Args:
            arguments: List of command-line arguments

        Returns:
            Full command list for subprocess.run()
        """
        abs_path = PathManager.resolve(self.script_path)
        return [sys.executable, abs_path] + arguments

    def execute(self, arguments: List[str]) -> bool:
        """
        Execute stage script with arguments.

        Args:
            arguments: List of command-line arguments to pass

        Returns:
            True if execution succeeded, False otherwise

        Raises:
            Various subprocess exceptions if execution fails
        """
        script_abs = PathManager.resolve(self.script_path)

        if not os.path.exists(script_abs):
            log.error(f"Script not found: {script_abs}")
            return False

        cmd = self.build_command(arguments)
        log.step(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                shell=False
            )

            # Capture return code
            self.last_return_code = result.returncode

            # Log output
            if result.stdout:
                print(result.stdout)

            # Check for errors
            if result.returncode != 0:
                log.error(f"Execution failed (exit code {result.returncode})")
                if result.stderr:
                    log.error(f"Error output: {result.stderr}")
                return False

            return True

        except FileNotFoundError as e:
            log.error(f"Python executable not found: {e}")
            return False
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            return False


# ============================================================
# Pipeline Stages
# ============================================================

class DetectionStage(PipelineStage):
    """
    Detection & Tracking Stage.

    Executes YOLO detection and object tracking to generate
    tracking JSON file with frame-by-frame detections.
    """

    def __init__(self, script_path: str = DEFAULT_CONFIG["DETECT_SCRIPT"]):
        """Initialize detection stage."""
        super().__init__(script_path, "Detection & Tracking")

    def run(
            self,
            video_path: str,
            model_path: str,
            output_json: str,
            conf_threshold: float = 0.3,
            iou_threshold: float = 0.45
    ) -> bool:
        """
        Execute detection stage.

        Args:
            video_path: Path to input video or device ID (0 for webcam)
            model_path: Path to YOLO model weights
            output_json: Path to output tracking JSON
            conf_threshold: Detection confidence threshold (0.0-1.0)
            iou_threshold: NMS IOU threshold

        Returns:
            True if detection succeeded
        """
        log.task(1, "Detection & Tracking Stage")

        # Validate inputs
        if not PathManager.validate_file(model_path, "YOLO model"):
            return False

        # Handle webcam case
        if video_path != "0":
            if not PathManager.validate_file(video_path, "Video file"):
                return False

        # Build arguments
        arguments = [
            "--video", video_path,
            "--model", model_path,
            "--output", output_json,
            "--conf", str(conf_threshold),
            "--iou", str(iou_threshold)
        ]

        log.info(f"Input video: {video_path}")
        log.info(f"Output: {output_json}")

        return self.execute(arguments)


class RuleEngineStage(PipelineStage):
    """
    Rule Engine Analysis Stage.

    Processes tracking JSON and applies safety rules to generate
    structured alerts JSON.
    """

    def __init__(self, script_path: str = DEFAULT_CONFIG["RULE_ENGINE_SCRIPT"]):
        """Initialize rule engine stage."""
        super().__init__(script_path, "Rule Engine")

    def run(
            self,
            input_json: str,
            output_json: str
    ) -> bool:
        """
        Execute rule engine stage.

        Args:
            input_json: Path to tracking JSON from detection stage
            output_json: Path to output alerts JSON

        Returns:
            True if analysis succeeded
        """
        log.task(2, "Rule Engine Analysis Stage")

        # Validate input
        if not PathManager.validate_file(input_json, "Tracking JSON"):
            return False

        # Build arguments
        arguments = [
            "--input", input_json,
            "--output", output_json
        ]

        log.info(f"Input: {input_json}")
        log.info(f"Output: {output_json}")

        return self.execute(arguments)


# ============================================================
# Report Validation
# ============================================================

class ReportValidator:
    """
    Validate pipeline output files.

    Ensures generated JSON files are valid and contain expected structure.
    """

    @staticmethod
    def validate_tracking_json(file_path: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate tracking JSON structure.

        Expected structure:
        {
            "metadata": {"fps": int, "classes": {...}},
            "frames": [{"frame_id": int, "detections": [...]}]
        }

        Args:
            file_path: Path to tracking JSON

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            required_keys = ["metadata", "frames"]
            if not all(key in data for key in required_keys):
                log.error(f"Missing required keys in tracking JSON: {required_keys}")
                return False, None

            # Validate frames
            frames = data.get("frames", [])
            if not isinstance(frames, list):
                log.error("'frames' must be a list")
                return False, None

            log.success(f"Valid tracking JSON ({len(frames)} frames)")
            return True, data

        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON format: {e}")
            return False, None
        except FileNotFoundError:
            log.error(f"File not found: {file_path}")
            return False, None

    @staticmethod
    def validate_alerts_json(file_path: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate alerts JSON structure.

        Expected structure:
        {
            "total_alerts": int,
            "alerts_by_type": {...},
            "severity_distribution": {...},
            "alerts": [...]
        }

        Args:
            file_path: Path to alerts JSON

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            required_keys = ["total_alerts", "alerts"]
            if not all(key in data for key in required_keys):
                log.error(f"Missing required keys in alerts JSON: {required_keys}")
                return False, None

            total = data.get("total_alerts", 0)
            alerts = data.get("alerts", [])

            log.success(
                f"Valid alerts JSON ({total} alerts across "
                f"{len(data.get('alerts_by_type', {}))} types)"
            )
            return True, data

        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON format: {e}")
            return False, None
        except FileNotFoundError:
            log.error(f"File not found: {file_path}")
            return False, None


# ============================================================
# Pipeline Orchestrator
# ============================================================

class SafetyMonitoringPipeline:
    """
    Main pipeline orchestrator.

    Coordinates execution of all pipeline stages from detection
    through rule engine analysis, with validation and error handling.

    Example:
        >>> pipeline = SafetyMonitoringPipeline()
        >>> success = pipeline.execute(
        ...     video_path="warehouse.mp4",
        ...     model_path="models/best.pt",
        ...     output_tracking="tracking.json",
        ...     output_alerts="alerts.json"
        ... )
        >>> if success:
        ...     report = pipeline.load_alerts_report()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: Optional configuration dictionary. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.detection_stage = DetectionStage(
            self.config.get("DETECT_SCRIPT")
        )
        self.rule_engine_stage = RuleEngineStage(
            self.config.get("RULE_ENGINE_SCRIPT")
        )
        self.validator = ReportValidator()

    def validate_prerequisites(self) -> bool:
        """
        Validate all required scripts exist.

        Returns:
            True if all prerequisites are satisfied
        """
        log.header("🔍 Validating Prerequisites", width=70)

        all_ok = True

        # Check scripts
        if not self.detection_stage.validate():
            all_ok = False

        if not self.rule_engine_stage.validate():
            all_ok = False

        if not all_ok:
            log.error("Some required files are missing. Check paths and try again.")
            return False

        log.success("All prerequisites satisfied")
        return True

    def execute(
            self,
            video_path: str,
            model_path: str,
            output_tracking: Optional[str] = None,
            output_alerts: Optional[str] = None,
            conf_threshold: float = 0.3,
            iou_threshold: float = 0.45
    ) -> bool:
        """
        Execute complete pipeline.

        Pipeline Stages:
            1. Detection & Tracking: Process video → tracking JSON
            2. Rule Engine: Analyze tracking data → alerts JSON

        Args:
            video_path: Input video file or device ID (0 for webcam)
            model_path: Path to YOLO model weights
            output_tracking: Path to output tracking JSON (default from config)
            output_alerts: Path to output alerts JSON (default from config)
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IOU threshold

        Returns:
            True if all stages succeeded, False if any stage failed
        """
        # Use config defaults if not specified
        output_tracking = output_tracking or self.config["TRACKING_JSON"]
        output_alerts = output_alerts or self.config["ALERTS_JSON"]

        log.header("🏭 Warehouse Safety Monitoring Pipeline", width=70)

        # ===== Stage 1: Detection & Tracking =====
        if not self.detection_stage.run(
                video_path=video_path,
                model_path=model_path,
                output_json=output_tracking,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
        ):
            log.error("Pipeline aborted: Detection stage failed")
            return False

        # Validate tracking JSON
        log.info("Validating detection output...")
        valid, data = self.validator.validate_tracking_json(output_tracking)
        if not valid:
            return False

        # ===== Stage 2: Rule Engine =====
        if not self.rule_engine_stage.run(
                input_json=output_tracking,
                output_json=output_alerts
        ):
            log.error("Pipeline aborted: Rule Engine stage failed")
            return False

        # Validate alerts JSON
        log.info("Validating rule engine output...")
        valid, data = self.validator.validate_alerts_json(output_alerts)
        if not valid:
            return False

        return True

    def load_alerts_report(self, alerts_json_path: str) -> Optional[Dict]:
        """
        Load and return alerts report from JSON file.

        Args:
            alerts_json_path: Path to alerts JSON file

        Returns:
            Dictionary containing alerts report, or None if loading failed
        """
        try:
            with open(alerts_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load alerts report: {e}")
            return None

    def print_summary(self, alerts_json_path: str) -> None:
        """
        Print summary of pipeline results.

        Args:
            alerts_json_path: Path to alerts JSON file
        """
        report = self.load_alerts_report(alerts_json_path)

        if not report:
            return

        log.header("📊 Pipeline Results Summary", width=70)

        print(f"\n📈 Total Violations Detected: {report.get('total_alerts', 0)}")

        print("\n📋 Violations by Type:")
        for vtype, count in report.get('alerts_by_type', {}).items():
            print(f"   • {vtype}: {count}")

        print("\n⚠️ Severity Distribution:")
        for severity, count in report.get('severity_distribution', {}).items():
            print(f"   • {severity}: {count}")

        print("\n✅ Pipeline completed successfully!")
        print(f"📄 Tracking data: tracking_complete.json")
        print(f"📄 Alerts report: {alerts_json_path}")


# ============================================================
# CLI Interface
# ============================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Warehouse Safety Monitoring Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python pipeline_orchestrator.py

  # Process with custom video and model
  python pipeline_orchestrator.py \\
    --video warehouse.mp4 \\
    --model models/best.pt \\
    --output-tracking tracking.json \\
    --output-alerts alerts.json

  # Use webcam (device 0)
  python pipeline_orchestrator.py --video 0

  # Custom detection thresholds
  python pipeline_orchestrator.py \\
    --video video.mp4 \\
    --conf 0.4 \\
    --iou 0.5
        """
    )

    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_CONFIG["VIDEO_PATH"],
        help="Input video file or device ID (0 for webcam)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["MODEL_PATH"],
        help="Path to YOLO model weights"
    )

    parser.add_argument(
        "--output-tracking",
        type=str,
        default=DEFAULT_CONFIG["TRACKING_JSON"],
        help="Output path for tracking JSON"
    )

    parser.add_argument(
        "--output-alerts",
        type=str,
        default=DEFAULT_CONFIG["ALERTS_JSON"],
        help="Output path for alerts JSON"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIG["CONF_THRESHOLD"],
        help="Detection confidence threshold (0.0-1.0)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_CONFIG["IOU_THRESHOLD"],
        help="NMS IOU threshold (0.0-1.0)"
    )

    return parser


def main():
    """
    Main entry point for CLI.

    Parses arguments, initializes pipeline, and executes
    the complete safety monitoring workflow.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SafetyMonitoringPipeline()

    # Validate prerequisites
    if not pipeline.validate_prerequisites():
        sys.exit(1)

    # Execute pipeline
    success = pipeline.execute(
        video_path=args.video,
        model_path=args.model,
        output_tracking=args.output_tracking,
        output_alerts=args.output_alerts,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    if not success:
        log.header("❌ Pipeline Failed", width=70)
        sys.exit(1)

    # Print results
    pipeline.print_summary(args.output_alerts)

    log.header("✅ Pipeline Completed Successfully", width=70)
    log.info(f"Tracking data: {args.output_tracking}")
    log.info(f"Alerts report: {args.output_alerts}")
    log.info("Ready for downstream processing (LLM, database, etc.)")
