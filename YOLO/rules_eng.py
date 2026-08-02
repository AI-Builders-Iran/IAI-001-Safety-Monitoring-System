"""
Rule Engine Module for Warehouse Safety Monitoring System.

This module provides a comprehensive rule-based engine for detecting and reporting
safety violations in warehouse/industrial environments. It processes frame data 
from video analysis pipelines and generates structured alert reports based on 
predefined safety rules.

Key Rules:
    - Vehicle-Person Proximity: Detect when workers approach vehicles
    - Machinery-Person Proximity: Detect when machinery gets close to workers
    - PPE Violations: Monitor proper use of safety equipment
    - Idle Workers: Detect prolonged inactivity in specified zones
    - Crowd Detection: Alert when worker density exceeds safe levels
    - Abnormal Movement: Detect unusually fast movements

Typical Usage:
    >>> engine = RuleEngine(config_path="config.json")
    >>> engine.process_video("warehouse_video.json")
    >>> report = engine.get_report()
    >>> engine.save_report("alerts_report.json")
"""

from __future__ import annotations

import io
import sys
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# UTF-8 encoding support
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================
# Default Configuration
# ============================================================

DEFAULT_CONFIG = {
    """
    Distance thresholds (in meters):
        - distance_vehicle_person_m: Safe distance from vehicles (default: 3m)
        - distance_machinery_person_m: Safe distance from machinery (default: 1m)
        - pixels_per_meter: Calibration factor for distance calculation
    
    Time-based thresholds (in frames):
        - no_ppe_frames: Frames required to trigger PPE violation alert
        - no_hardhat_frames: Frames required to trigger hardhat violation alert
        - no_vest_frames: Frames required to trigger safety vest violation alert
        - idle_frames: Frames to track before classifying as idle
        - crowd_threshold: Maximum persons per frame before crowd alert
    
    Speed thresholds (in pixels per frame):
        - speed_threshold: Maximum allowed speed for normal movement
    """
    
    # Distance thresholds (in meters) - Calibration: 1 meter = 50 pixels
    "distance_vehicle_person_m": 3.0,
    "distance_machinery_person_m": 1.0,
    "pixels_per_meter": 50,

    # Time-based thresholds (in frames)
    "no_ppe_frames": 5,
    "no_hardhat_frames": 3,
    "no_vest_frames": 3,
    "idle_frames": 150,
    "crowd_threshold": 3,

    # Speed thresholds (in pixels per frame)
    "speed_threshold": 10,
}


# ============================================================
# Utility Functions
# ============================================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        p1: First point as (x, y) tuple
        p2: Second point as (x, y) tuple

    Returns:
        float: Distance between the two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_center(detection: Dict[str, Any]) -> Tuple[float, float]:
    """
    Extract center coordinates from detection object.

    Args:
        detection: Detection dictionary containing 'center' key with 'x', 'y' fields

    Returns:
        tuple: (x, y) center coordinates
    """
    return (detection["center"]["x"], detection["center"]["y"])


def meters_to_pixels(meters: float, config: Dict[str, Any]) -> float:
    """
    Convert distance from meters to pixels using calibration factor.

    Args:
        meters: Distance in meters
        config: Configuration dictionary containing 'pixels_per_meter'

    Returns:
        float: Distance in pixels
    """
    return meters * config["pixels_per_meter"]


# ============================================================
# Rule Engine Class
# ============================================================

class RuleEngine:
    """
    Smart rule-based system for analyzing safety violations in warehouse/industrial videos.

    The RuleEngine processes frame-by-frame detection data and applies a set of
    pre-defined safety rules to generate alerts. It maintains internal state for
    temporal analysis (e.g., duration-based rules) and supports customizable
    configuration for thresholds and distances.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with thresholds
        alerts (List[Dict]): Accumulated alerts from all processed frames
        fps (int): Frames per second of processed video

    Example:
        >>> engine = RuleEngine()
        >>> engine.process_video("tracking_data.json")
        >>> report = engine.get_report()
        >>> engine.save_report("output.json")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RuleEngine with configuration and state tracking.

        Args:
            config: Optional configuration dictionary. If None, uses DEFAULT_CONFIG.
                   Config should include distance thresholds, frame counts, etc.
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.alerts = []
        self.fps = 30  # Default FPS, will be updated when processing video

        # Temporal state tracking
        self.person_no_ppe_frames = defaultdict(int)
        self.person_no_hardhat_frames = defaultdict(int)
        self.person_no_vest_frames = defaultdict(int)
        self.person_idle_start = {}
        self.person_positions = defaultdict(list)
        self.alerted_conditions = set()

        # Previous positions for speed calculation
        self.prev_positions = {}

        # Register all safety rules
        self.rules = [
            ("vehicle_person_proximity", self._check_vehicle_person_proximity),
            ("person_no_ppe", self._check_person_no_ppe),
            ("machinery_person_proximity", self._check_machinery_person_proximity),
            ("person_no_hardhat", self._check_person_no_hardhat),
            ("person_no_vest", self._check_person_no_vest),
            ("person_idle", self._check_person_idle),
            ("crowd_detection", self._check_crowd),
            ("high_speed", self._check_high_speed),
        ]

        # Calculate pixel-based distance thresholds
        self._update_pixel_thresholds()

    def _update_pixel_thresholds(self) -> None:
        """
        Calculate and cache pixel-based distance thresholds from config.

        This converts meter-based thresholds to pixel equivalents using
        the pixels_per_meter calibration factor from configuration.
        """
        self.pixel_dist_vehicle_person = (
            self.config["distance_vehicle_person_m"] * self.config["pixels_per_meter"]
        )
        self.pixel_dist_machinery_person = (
            self.config["distance_machinery_person_m"] * self.config["pixels_per_meter"]
        )

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Raises:
            FileNotFoundError: If config file does not exist
            json.JSONDecodeError: If file is not valid JSON
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self._update_pixel_thresholds()

    def process_video(self, video_data_path: str) -> None:
        """
        Process video tracking data from JSON file.

        Reads frame-by-frame detection data from JSON file (typically output from
        YOLO + tracking pipeline) and applies all safety rules to generate alerts.

        Args:
            video_data_path: Path to JSON file containing frame detection data

        JSON Format Expected:
            {
                "metadata": {
                    "fps": 30,
                    "classes": {...}
                },
                "frames": [
                    {
                        "frame_id": 0,
                        "detections": [...]
                    },
                    ...
                ]
            }

        Raises:
            FileNotFoundError: If video data file does not exist
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required keys are missing from JSON structure
        """
        print(f"📂 Loading video data from: {video_data_path}")

        with open(video_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract metadata
        self.fps = data.get("metadata", {}).get("fps", 30)
        frames = data.get("frames", [])

        print(f"⏱️  FPS: {self.fps}")
        print(f"📊 Processing {len(frames)} frames...")

        # Process each frame
        for frame in frames:
            self.process_frame(frame)

        print(f"✅ Processing complete! Generated {len(self.alerts)} alerts.")

    def process_frame(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single frame and apply all safety rules.

        Args:
            frame_data: Dictionary containing frame_id and detections
                       {
                           "frame_id": int,
                           "detections": [
                               {
                                   "track_id": int,
                                   "class_name": str,
                                   "center": {"x": float, "y": float},
                                   "has_ppe": bool,
                                   "has_hardhat": bool,
                                   "has_vest": bool
                               },
                               ...
                           ]
                       }

        Returns:
            List[Dict]: List of new alerts generated for this frame
        """
        frame_alerts = []

        # Apply each safety rule
        for rule_name, rule_func in self.rules:
            try:
                result = rule_func(frame_data)
                if result:
                    for alert in result:
                        # Generate unique alert ID to prevent duplicates
                        alert_id = (
                            f"{alert['type']}_"
                            f"{alert.get('track_id', '')}_"
                            f"{frame_data['frame_id']}"
                        )

                        if alert_id not in self.alerted_conditions:
                            self.alerted_conditions.add(alert_id)
                            self.alerts.append(alert)
                            frame_alerts.append(alert)
            except Exception as e:
                print(f"⚠️  Error in rule '{rule_name}': {str(e)}")
                continue

        return frame_alerts

    # ============================================================
    # Safety Rules Implementation
    # ============================================================

    def _check_vehicle_person_proximity(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 1: Alert if a person approaches a vehicle closer than safe distance.

        Safe distance threshold: 3 meters (configurable)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]
        vehicles = [d for d in detections if d.get("class_name") == "vehicle"]

        alerts = []
        for person in persons:
            p_center = get_center(person)
            for vehicle in vehicles:
                v_center = get_center(vehicle)
                dist = distance(p_center, v_center)

                if dist < self.pixel_dist_vehicle_person:
                    distance_m = dist / self.config["pixels_per_meter"]
                    alerts.append({
                        "type": "vehicle_person_proximity",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": person.get("track_id"),
                        "vehicle_id": vehicle.get("track_id"),
                        "distance_px": round(dist, 2),
                        "distance_m": round(distance_m, 2),
                        "severity": "high",
                        "message": (
                            f"⚠️ Person {person.get('track_id')} approached "
                            f"vehicle {vehicle.get('track_id')}! "
                            f"Distance: {distance_m:.2f}m"
                        )
                    })
        return alerts

    def _check_machinery_person_proximity(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 2: Alert if machinery approaches a person closer than safe distance.

        Safe distance threshold: 1 meter (configurable)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]
        machineries = [d for d in detections if d.get("class_name") == "machinery"]

        alerts = []
        for person in persons:
            p_center = get_center(person)
            for machinery in machineries:
                m_center = get_center(machinery)
                dist = distance(p_center, m_center)

                if dist < self.pixel_dist_machinery_person:
                    distance_m = dist / self.config["pixels_per_meter"]
                    alerts.append({
                        "type": "machinery_person_proximity",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": person.get("track_id"),
                        "machinery_id": machinery.get("track_id"),
                        "distance_px": round(dist, 2),
                        "distance_m": round(distance_m, 2),
                        "severity": "critical",
                        "message": (
                            f"⚠️ Machinery {machinery.get('track_id')} approached "
                            f"person {person.get('track_id')}! "
                            f"Distance: {distance_m:.2f}m"
                        )
                    })
        return alerts

    def _check_person_no_ppe(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 3: Alert if person detected without any PPE for consecutive frames.

        Threshold: Configured number of consecutive frames (default: 5)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]

        alerts = []
        for person in persons:
            track_id = person.get("track_id")
            has_ppe = person.get("has_ppe", False)

            if not has_ppe:
                self.person_no_ppe_frames[track_id] += 1
            else:
                self.person_no_ppe_frames[track_id] = 0

            if self.person_no_ppe_frames[track_id] >= self.config["no_ppe_frames"]:
                alerts.append({
                    "type": "person_no_ppe",
                    "frame_id": frame_data.get("frame_id"),
                    "track_id": track_id,
                    "duration_frames": self.person_no_ppe_frames[track_id],
                    "severity": "high",
                    "message": (
                        f"⚠️ Person {track_id} detected without PPE for "
                        f"{self.person_no_ppe_frames[track_id]} frames!"
                    )
                })
                self.person_no_ppe_frames[track_id] = 0

        return alerts

    def _check_person_no_hardhat(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 4: Alert if person detected without safety hardhat for consecutive frames.

        Threshold: Configured number of consecutive frames (default: 3)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]

        alerts = []
        for person in persons:
            track_id = person.get("track_id")
            has_hardhat = person.get("has_hardhat", False)

            if not has_hardhat:
                self.person_no_hardhat_frames[track_id] += 1
            else:
                self.person_no_hardhat_frames[track_id] = 0

            if self.person_no_hardhat_frames[track_id] >= self.config["no_hardhat_frames"]:
                alerts.append({
                    "type": "person_no_hardhat",
                    "frame_id": frame_data.get("frame_id"),
                    "track_id": track_id,
                    "duration_frames": self.person_no_hardhat_frames[track_id],
                    "severity": "medium",
                    "message": (
                        f"⚠️ Person {track_id} detected without hardhat for "
                        f"{self.person_no_hardhat_frames[track_id]} frames!"
                    )
                })
                self.person_no_hardhat_frames[track_id] = 0

        return alerts

    def _check_person_no_vest(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 5: Alert if person detected without safety vest for consecutive frames.

        Threshold: Configured number of consecutive frames (default: 3)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]

        alerts = []
        for person in persons:
            track_id = person.get("track_id")
            has_vest = person.get("has_vest", False)

            if not has_vest:
                self.person_no_vest_frames[track_id] += 1
            else:
                self.person_no_vest_frames[track_id] = 0

            if self.person_no_vest_frames[track_id] >= self.config["no_vest_frames"]:
                alerts.append({
                    "type": "person_no_vest",
                    "frame_id": frame_data.get("frame_id"),
                    "track_id": track_id,
                    "duration_frames": self.person_no_vest_frames[track_id],
                    "severity": "medium",
                    "message": (
                        f"⚠️ Person {track_id} detected without safety vest for "
                        f"{self.person_no_vest_frames[track_id]} frames!"
                    )
                })
                self.person_no_vest_frames[track_id] = 0

        return alerts

    def _check_person_idle(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 6: Alert if person remains idle (stationary) for extended duration.

        Detects lack of movement over consecutive frames using position history.
        Threshold: 5+ seconds (configurable via idle_frames)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]

        alerts = []
        for person in persons:
            track_id = person.get("track_id")
            center = get_center(person)

            # Track position history
            self.person_positions[track_id].append(center)

            # Keep only last 20 positions for analysis
            if len(self.person_positions[track_id]) > 20:
                self.person_positions[track_id].pop(0)

            # Need sufficient history for analysis
            if len(self.person_positions[track_id]) < 10:
                continue

            # Calculate total movement across position history
            positions = self.person_positions[track_id]
            total_movement = sum(
                distance(positions[i], positions[i + 1])
                for i in range(len(positions) - 1)
            )

            # Check if person is stationary
            if total_movement < 20 and len(positions) > self.config["idle_frames"]:
                if track_id not in self.person_idle_start:
                    self.person_idle_start[track_id] = frame_data.get("frame_id", 0)
                else:
                    idle_duration = (
                        frame_data.get("frame_id", 0) - self.person_idle_start[track_id]
                    ) / self.fps

                    if idle_duration > 5:  # 5 seconds threshold
                        alerts.append({
                            "type": "person_idle",
                            "frame_id": frame_data.get("frame_id"),
                            "track_id": track_id,
                            "idle_seconds": round(idle_duration, 1),
                            "severity": "low",
                            "message": (
                                f"⚠️ Person {track_id} idle for "
                                f"{idle_duration:.1f} seconds"
                            )
                        })
                        self.person_idle_start[track_id] = frame_data.get("frame_id", 0)
            else:
                # Reset idle timer if person moves
                if track_id in self.person_idle_start:
                    del self.person_idle_start[track_id]

        return alerts

    def _check_crowd(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 7: Alert if number of people in frame exceeds safe density threshold.

        Threshold: Configured maximum persons (default: 3)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])
        person_count = sum(1 for d in detections if d.get("class_name") == "Person")

        alerts = []
        if person_count > self.config["crowd_threshold"]:
            alerts.append({
                "type": "crowd_detection",
                "frame_id": frame_data.get("frame_id"),
                "person_count": person_count,
                "threshold": self.config["crowd_threshold"],
                "severity": "medium",
                "message": (
                    f"👥 Crowd detected! {person_count} people "
                    f"(threshold: {self.config['crowd_threshold']})"
                )
            })
        return alerts

    def _check_high_speed(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule 8: Alert if object moves with unusually high speed.

        Detects abnormal acceleration which may indicate unsafe handling.
        Threshold: Configured speed in pixels/frame (default: 10)

        Args:
            frame_data: Current frame detection data

        Returns:
            List of alerts if violations detected
        """
        detections = frame_data.get("detections", [])

        alerts = []
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            center = get_center(det)

            # Calculate speed from position history
            if track_id in self.prev_positions:
                prev_center = self.prev_positions[track_id]
                speed = distance(center, prev_center)

                if speed > self.config["speed_threshold"]:
                    alerts.append({
                        "type": "high_speed",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": track_id,
                        "class_name": det.get("class_name"),
                        "speed": round(speed, 1),
                        "severity": "medium",
                        "message": (
                            f"⚡ Object {track_id} ({det.get('class_name')}) "
                            f"moving at high speed: {speed:.1f} px/frame"
                        )
                    })

            self.prev_positions[track_id] = center

        return alerts

    # ============================================================
    # Report Generation and Output
    # ============================================================

    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of all detected alerts.

        Returns:
            Dict containing:
                - total_alerts: Total number of alerts generated
                - alerts_by_type: Count of alerts grouped by violation type
                - severity_distribution: Count of alerts by severity level
                - alerts: Full list of alert dictionaries
        """
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity = alert.get("severity", "unknown")
            severity_counts[severity] += 1

        return {
            "total_alerts": len(self.alerts),
            "alerts_by_type": self._count_by_type(),
            "severity_distribution": dict(severity_counts),
            "alerts": self.alerts
        }

    def _count_by_type(self) -> Dict[str, int]:
        """
        Count alerts grouped by violation type.

        Returns:
            Dict mapping alert type names to counts
        """
        counts = defaultdict(int)
        for alert in self.alerts:
            counts[alert["type"]] += 1
        return dict(counts)

    def save_report(self, output_path: str) -> None:
        """
        Save alert report to JSON file.

        Args:
            output_path: Path where JSON report should be saved
        """
        report = self.get_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ Report saved to: {output_path}")

    def print_report(self, max_alerts: int = 10) -> None:
        """
        Print formatted report to console.

        Args:
            max_alerts: Maximum number of individual alerts to print (default: 10)
        """
        report = self.get_report()

        print("\n" + "=" * 70)
        print("📊 SAFETY MONITORING REPORT")
        print("=" * 70)

        print(f"\n📈 Total Alerts: {report['total_alerts']}")

        print("\n📊 Alerts by Type:")
        for alert_type, count in sorted(
            report["alerts_by_type"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"   • {alert_type}: {count}")

        print("\n⚠️ Severity Distribution:")
        for severity, count in sorted(
            report["severity_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"   • {severity}: {count}")

        print(f"\n📋 Sample Alerts (first {min(max_alerts, len(report['alerts']))} of {len(report['alerts'])}):")
        for i, alert in enumerate(report["alerts"][:max_alerts], 1):
            print(
                f"   [{i}] [{alert.get('severity', 'unknown')}] "
                f"{alert.get('message', 'No message')}"
            )

        print("\n" + "=" * 70)

    def reset(self) -> None:
        """
        Reset engine state to process new video.

        Clears all accumulated alerts and internal state tracking.
        Configuration is preserved.
        """
        self.alerts = []
        self.person_no_ppe_frames.clear()
        self.person_no_hardhat_frames.clear()
        self.person_no_vest_frames.clear()
        self.person_idle_start.clear()
        self.person_positions.clear()
        self.alerted_conditions.clear()
        self.prev_positions.clear()
        print("🔄 Engine state reset. Ready for new video.")
