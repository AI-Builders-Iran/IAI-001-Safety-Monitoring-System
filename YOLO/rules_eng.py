"""
Rule Engine Module for Warehouse Safety Monitoring System - Optimized for Small Models.

This module provides a lightweight rule-based engine for detecting safety violations
with simplified outputs suitable for small language models (1.5B parameters).
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Simplified default configuration for small models
DEFAULT_CONFIG = {
    "distance_vehicle_person_m": 3.0,
    "distance_machinery_person_m": 1.0,
    "pixels_per_meter": 50,
    "no_ppe_frames": 5,
    "no_hardhat_frames": 3,
    "no_vest_frames": 3,
    "idle_frames": 150,
    "crowd_threshold": 3,
    "speed_threshold": 10,
    "max_alerts_per_frame": 3,  # Limit alerts per frame for small models
}


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_center(detection: Dict[str, Any]) -> Tuple[float, float]:
    """Extract center coordinates from detection object."""
    return (detection["center"]["x"], detection["center"]["y"])


def meters_to_pixels(meters: float, config: Dict[str, Any]) -> float:
    """Convert distance from meters to pixels using calibration factor."""
    return meters * config["pixels_per_meter"]


class RuleEngine:
    """
    Lightweight Rule Engine optimized for small models (1.5B parameters).

    Generates simplified, concise outputs with limited alerts per frame.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RuleEngine with simplified state tracking."""
        self.config = config or DEFAULT_CONFIG.copy()
        self.alerts = []
        self.fps = 30

        # Simplified state tracking
        self.person_no_ppe_frames = defaultdict(int)
        self.person_no_hardhat_frames = defaultdict(int)
        self.person_no_vest_frames = defaultdict(int)
        self.prev_positions = {}
        self.alerted_conditions = set()

        # Register priority rules (most important first)
        self.rules = [
            ("critical_proximity", self._check_critical_proximity),
            ("ppe_violation", self._check_ppe_violation),
            ("crowd_detection", self._check_crowd),
            ("high_speed", self._check_high_speed),
        ]

        self._update_pixel_thresholds()

    def _update_pixel_thresholds(self) -> None:
        """Calculate and cache pixel-based distance thresholds."""
        self.pixel_dist_vehicle = (
            self.config["distance_vehicle_person_m"] * self.config["pixels_per_meter"]
        )
        self.pixel_dist_machinery = (
            self.config["distance_machinery_person_m"] * self.config["pixels_per_meter"]
        )

    def process_video(self, video_data_path: str) -> None:
        """Process video tracking data from JSON file with simplified output."""
        logger.info(f"📂 Loading video data: {video_data_path}")

        with open(video_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.fps = data.get("metadata", {}).get("fps", 30)
        frames = data.get("frames", [])

        logger.info(f"⏱️ FPS: {self.fps}, Processing {len(frames)} frames")

        for frame in frames:
            self.process_frame(frame)

        logger.info(f"✅ Processing complete! Generated {len(self.alerts)} alerts.")

    def process_frame(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single frame with limit on alerts per frame.
        Returns simplified alert list for small models.
        """
        frame_alerts = []
        frame_id = frame_data.get("frame_id", 0)
        detections = frame_data.get("detections", [])

        # Quick person and object counts
        person_count = sum(1 for d in detections if d.get("class_name") == "Person")

        # Apply only priority rules first
        for rule_name, rule_func in self.rules:
            if len(frame_alerts) >= self.config["max_alerts_per_frame"]:
                break
            try:
                result = rule_func(frame_data)
                if result:
                    for alert in result:
                        alert_id = f"{alert['type']}_{frame_id}"
                        if alert_id not in self.alerted_conditions:
                            self.alerted_conditions.add(alert_id)
                            # Simplify alert for small models
                            simplified_alert = self._simplify_alert(alert)
                            self.alerts.append(simplified_alert)
                            frame_alerts.append(simplified_alert)
            except Exception as e:
                logger.error(f"⚠️ Error in rule '{rule_name}': {str(e)}")
                continue

        return frame_alerts

    def _simplify_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify alert structure for small models - keep only essential fields."""
        simplified = {
            "type": alert.get("type"),
            "frame": alert.get("frame_id"),
            "severity": alert.get("severity"),
            "msg": alert.get("message", "").replace("⚠️ ", "").replace("👥 ", "").replace("⚡ ", "")
        }

        # Add only critical IDs if present
        if "track_id" in alert:
            simplified["id"] = alert["track_id"]
        if "vehicle_id" in alert:
            simplified["veh_id"] = alert["vehicle_id"]
        if "machinery_id" in alert:
            simplified["mach_id"] = alert["machinery_id"]

        # Add numeric values when relevant
        if "distance_m" in alert:
            simplified["dist_m"] = round(alert["distance_m"], 1)
        if "speed" in alert:
            simplified["speed"] = round(alert["speed"], 1)
        if "person_count" in alert:
            simplified["count"] = alert["person_count"]

        return simplified

    def _check_critical_proximity(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check both vehicle and machinery proximity in one pass."""
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]
        vehicles = [d for d in detections if d.get("class_name") == "vehicle"]
        machineries = [d for d in detections if d.get("class_name") == "machinery"]

        alerts = []

        # Check vehicle-person proximity
        for person in persons:
            p_center = get_center(person)
            for vehicle in vehicles:
                dist = distance(p_center, get_center(vehicle))
                if dist < self.pixel_dist_vehicle:
                    distance_m = dist / self.config["pixels_per_meter"]
                    alerts.append({
                        "type": "vehicle_proximity",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": person.get("track_id"),
                        "vehicle_id": vehicle.get("track_id"),
                        "distance_m": distance_m,
                        "severity": "high",
                        "message": f"Person {person.get('track_id')} near vehicle {vehicle.get('track_id')} ({distance_m:.1f}m)"
                    })
                    break  # One alert per person per frame

        # Check machinery-person proximity
        for person in persons:
            p_center = get_center(person)
            for machinery in machineries:
                dist = distance(p_center, get_center(machinery))
                if dist < self.pixel_dist_machinery:
                    distance_m = dist / self.config["pixels_per_meter"]
                    alerts.append({
                        "type": "machinery_proximity",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": person.get("track_id"),
                        "machinery_id": machinery.get("track_id"),
                        "distance_m": distance_m,
                        "severity": "critical",
                        "message": f"Person {person.get('track_id')} near machinery {machinery.get('track_id')} ({distance_m:.1f}m)"
                    })
                    break

        return alerts

    def _check_ppe_violation(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check PPE violations in one pass."""
        detections = frame_data.get("detections", [])
        persons = [d for d in detections if d.get("class_name") == "Person"]

        alerts = []
        for person in persons:
            track_id = person.get("track_id")
            has_ppe = person.get("has_ppe", False)
            has_hardhat = person.get("has_hardhat", False)
            has_vest = person.get("has_vest", False)

            # Check overall PPE
            if not has_ppe:
                self.person_no_ppe_frames[track_id] += 1
                if self.person_no_ppe_frames[track_id] >= self.config["no_ppe_frames"]:
                    alerts.append({
                        "type": "no_ppe",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": track_id,
                        "severity": "high",
                        "message": f"Person {track_id} without PPE"
                    })
                    self.person_no_ppe_frames[track_id] = 0
            else:
                self.person_no_ppe_frames[track_id] = 0

            # Check hardhat
            if not has_hardhat:
                self.person_no_hardhat_frames[track_id] += 1
                if self.person_no_hardhat_frames[track_id] >= self.config["no_hardhat_frames"]:
                    alerts.append({
                        "type": "no_hardhat",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": track_id,
                        "severity": "medium",
                        "message": f"Person {track_id} missing hardhat"
                    })
                    self.person_no_hardhat_frames[track_id] = 0
            else:
                self.person_no_hardhat_frames[track_id] = 0

            # Check vest
            if not has_vest:
                self.person_no_vest_frames[track_id] += 1
                if self.person_no_vest_frames[track_id] >= self.config["no_vest_frames"]:
                    alerts.append({
                        "type": "no_vest",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": track_id,
                        "severity": "medium",
                        "message": f"Person {track_id} missing vest"
                    })
                    self.person_no_vest_frames[track_id] = 0
            else:
                self.person_no_vest_frames[track_id] = 0

        return alerts

    def _check_crowd(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple crowd detection."""
        detections = frame_data.get("detections", [])
        person_count = sum(1 for d in detections if d.get("class_name") == "Person")

        if person_count > self.config["crowd_threshold"]:
            return [{
                "type": "crowd",
                "frame_id": frame_data.get("frame_id"),
                "person_count": person_count,
                "severity": "medium",
                "message": f"Crowd: {person_count} people"
            }]
        return []

    def _check_high_speed(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple speed check."""
        detections = frame_data.get("detections", [])
        alerts = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            center = get_center(det)

            if track_id in self.prev_positions:
                speed = distance(center, self.prev_positions[track_id])
                if speed > self.config["speed_threshold"]:
                    alerts.append({
                        "type": "high_speed",
                        "frame_id": frame_data.get("frame_id"),
                        "track_id": track_id,
                        "speed": speed,
                        "severity": "medium",
                        "message": f"Object {track_id} high speed ({speed:.1f} px/frame)"
                    })

            self.prev_positions[track_id] = center

        return alerts

    def get_simple_report(self) -> Dict[str, Any]:
        """Generate extremely simplified report for small models."""
        if not self.alerts:
            return {
                "status": "safe",
                "total": 0,
                "alerts": []
            }

        # Count by severity
        severity_counts = defaultdict(int)
        alert_types = defaultdict(int)

        for alert in self.alerts:
            severity_counts[alert.get("severity", "unknown")] += 1
            alert_types[alert.get("type", "unknown")] += 1

        # Determine overall status
        if severity_counts.get("critical", 0) > 0:
            status = "critical"
        elif severity_counts.get("high", 0) > 0:
            status = "warning"
        elif severity_counts.get("medium", 0) > 0:
            status = "caution"
        else:
            status = "safe"

        return {
            "status": status,
            "total": len(self.alerts),
            "by_type": dict(alert_types),
            "by_severity": dict(severity_counts),
            "alerts": self.alerts[:10]  # Limit alerts for small models
        }

    def get_report(self) -> Dict[str, Any]:
        """Get report - simplified version."""
        return self.get_simple_report()

    def save_report(self, output_path: str) -> None:
        """Save simplified report to JSON file."""
        report = self.get_simple_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Report saved: {output_path}")

    def print_simple_report(self) -> None:
        """Print extremely simplified report for small model consumption."""
        report = self.get_simple_report()

        print(f"Status: {report['status']}")
        print(f"Total Alerts: {report['total']}")

        if report['alerts']:
            print("\nAlerts:")
            for i, alert in enumerate(report['alerts'][:5], 1):
                print(f"  {i}. [{alert['severity']}] {alert['msg']}")

    def reset(self) -> None:
        """Reset engine state."""
        self.alerts = []
        self.person_no_ppe_frames.clear()
        self.person_no_hardhat_frames.clear()
        self.person_no_vest_frames.clear()
        self.prev_positions.clear()
        self.alerted_conditions.clear()
        logger.info("🔄 Engine reset")