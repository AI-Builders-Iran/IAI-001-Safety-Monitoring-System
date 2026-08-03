#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video-to-JSON Wrapper for the LLM Report Generation Layer.

This module is the single entry point that the LLM sub-system should call.
It hides the two upstream stages (YOLO detection/tracking and the rule
engine) behind one simple function/class, so the LLM layer only ever has
to do:

    from llm_pipeline_wrapper import run_pipeline_for_llm
    result = run_pipeline_for_llm("warehouse.mp4", model_path="models/best.pt")
    # result is a plain dict, ready to json.dumps() or feed into a prompt

Pipeline Flow:
    video file
        │
        ▼
    YOLODetector.run()          -> در حافظه، بدون فایل واسط: تشخیص + ردیابی هر فریم
        │
        ▼
    RuleEngine.process_frame()  -> اعمال قوانین ایمنی روی هر فریم (از فایل rules_eng.py)
        │
        ▼
    SafetyVideoPipeline.process_video() -> ترکیب دو خروجی در یک دیکشنری واحد
        │
        ▼
    JSON output (dict)          -> ورودی نهایی برای HSEPromptGenerator / LLM

Notes:
    - No intermediate files are required; everything runs in memory by
      default. Saving tracking/alerts JSON to disk is optional (useful for
      debugging or auditing) via the `save_tracking_json` / `save_alerts_json`
      arguments.
    - This module deliberately does NOT import/execute `info_DETECT.py`
      directly (that file is a top-level script with hard-coded Windows
      paths). Instead, the detection logic has been refactored into the
      reusable `YOLODetector` class below, using the same detection/PPE
      matching logic.
    - `RuleEngine` is imported as-is from `rules_eng.py` and used
      programmatically (no subprocess call), which is faster and avoids
      writing tracking_complete.json to disk on every run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# کتابخانه‌های پردازش ویدیو و مدل یولو
import cv2
from ultralytics import YOLO

# موتور قوانین ایمنی (همان فایل rules_eng.py موجود در پروژه)
from rules_eng import RuleEngine

# ============================================================
# پیکربندی پیش‌فرض تشخیص (برگرفته از info_DETECT.py)
# ============================================================

# نگاشت کلاس‌ها بر اساس data.yaml پروژه؛ در صورت نیاز عوض شود
DEFAULT_CLASS_MAP: Dict[int, str] = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
    10: "mask",
    11: "no-mask",
    12: "vehicle",
}

DEFAULT_DETECTOR_CONFIG: Dict[str, Any] = {
    # مسیر وزن‌های مدل یولو؛ حتماً هنگام ساخت YOLODetector مقداردهی شود
    "model_path": "models/best.pt",

    # آستانه‌های تشخیص و ردیابی
    "conf_threshold": 0.3,
    "iou_threshold": 0.45,

    # آستانه هم‌پوشانی برای اتصال تجهیزات ایمنی (کلاه/جلیقه) به شخص
    "iou_ppe_threshold": 0.15,

    # نگاشت کلاس‌ها
    "class_map": DEFAULT_CLASS_MAP,

    # نام کلاس‌های کلیدی (برای منطق اتصال PPE به شخص)
    "person_class_name": "Person",
    "hardhat_class_name": "Hardhat",
    "vest_class_name": "Safety Vest",

    # هر چند فریم یک بار پیشرفت در کنسول چاپ شود
    "log_every_n_frames": 30,
}


# ============================================================
# توابع کمکی هندسی
# ============================================================

def _calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        box1: Dict with keys x1, y1, x2, y2.
        box2: Dict with keys x1, y1, x2, y2.

    Returns:
        IoU value in range [0.0, 1.0].
    """
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _euclidean_distance(p1: tuple, p2: tuple) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Args:
        p1: (x, y) tuple.
        p2: (x, y) tuple.

    Returns:
        Distance between p1 and p2.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ============================================================
# مرحله ۱: تشخیص و ردیابی با YOLO
# ============================================================

class YOLODetector:
    """
    Wraps an Ultralytics YOLO model to run detection + multi-object tracking
    on a video and produce a structured, frame-by-frame tracking report
    entirely in memory (no intermediate JSON file is written to disk).

    The output schema matches what `RuleEngine` (rules_eng.py) expects:
        {
            "metadata": {...},
            "object_durations": {...},
            "object_movement": {...},
            "frames": [
                {"frame_id": int, "detections": [...], ...}, ...
            ]
        }

    Example:
        >>> detector = YOLODetector({"model_path": "models/best.pt"})
        >>> tracking_data = detector.run("warehouse.mp4")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the detector and load the YOLO model weights.

        Args:
            config: Optional overrides merged on top of
                DEFAULT_DETECTOR_CONFIG. Must include a valid "model_path"
                pointing to a .pt weights file.
        """
        self.config: Dict[str, Any] = {**DEFAULT_DETECTOR_CONFIG, **(config or {})}

        # مدل فقط یک بار در سازنده بارگذاری می‌شود تا برای هر ویدیو دوباره لود نشود
        self.model = YOLO(self.config["model_path"])

    def run(self, video_path: str) -> Dict[str, Any]:
        """
        Run detection + tracking over every frame of a video.

        Args:
            video_path: Path to a video file, or "0" for the default webcam.

        Returns:
            A dict with "metadata", "object_durations", "object_movement",
            and "frames" keys — the same structure previously produced by
            info_DETECT.py's tracking_complete.json, but returned in memory.

        Raises:
            FileNotFoundError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(0 if video_path == "0" else video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video could not be opened: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        all_frames_data: List[Dict[str, Any]] = []
        # اولین/آخرین فریمی که هر track_id در آن دیده شده (برای محاسبه مدت‌زمان حضور)
        object_first_last: Dict[int, Dict[str, Any]] = {}
        # تاریخچه مسیر حرکت هر شیء (برای محاسبه مسافت/سرعت)
        object_trajectory: Dict[int, List[tuple]] = {}

        frame_id = 0
        log_every = self.config["log_every_n_frames"]

        print(f"🎬 شروع پردازش ویدیو (FPS: {fps})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # تشخیص + ردیابی روی فریم جاری
            results = self.model.track(
                frame,
                conf=self.config["conf_threshold"],
                iou=self.config["iou_threshold"],
                persist=True,
                verbose=False,
            )

            detections = self._extract_detections(
                results, frame_id, object_first_last, object_trajectory
            )
            self._attach_ppe_flags(detections)

            frame_height, frame_width = frame.shape[:2]
            self._attach_relative_geometry(detections, frame_width, frame_height)

            all_frames_data.append({
                "frame_id": frame_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "frame_size": {"width": frame_width, "height": frame_height},
                "total_objects": len(detections),
                "detections": detections,
            })

            if frame_id % log_every == 0:
                print(f"   → پردازش فریم {frame_id}...")

        cap.release()
        print(f"✅ پردازش ویدیو کامل شد. تعداد کل فریم‌ها: {frame_id}")

        object_durations, object_movement = self._summarize_objects(
            object_first_last, object_trajectory, fps
        )

        return {
            "metadata": {
                "model": self.config["model_path"],
                "fps": fps,
                "total_frames": frame_id,
                "conf_threshold": self.config["conf_threshold"],
                "iou_threshold": self.config["iou_threshold"],
                "iou_ppe_threshold": self.config["iou_ppe_threshold"],
                "export_time": datetime.datetime.now().isoformat(),
                "classes": self.config["class_map"],
            },
            "object_durations": object_durations,
            "object_movement": object_movement,
            "frames": all_frames_data,
        }

    def _extract_detections(
            self,
            results,
            frame_id: int,
            object_first_last: Dict[int, Dict[str, Any]],
            object_trajectory: Dict[int, List[tuple]],
    ) -> List[Dict[str, Any]]:
        """
        Convert raw YOLO tracking results for one frame into the project's
        detection dict schema, updating per-object duration/trajectory state.

        Args:
            results: Output of `self.model.track(...)` for a single frame.
            frame_id: 1-based index of the current frame.
            object_first_last: Mutable state tracking first/last-seen frame
                per track_id (updated in place).
            object_trajectory: Mutable state tracking center-point history
                per track_id (updated in place).

        Returns:
            List of detection dicts for this frame.
        """
        detections: List[Dict[str, Any]] = []

        boxes = results[0].boxes if results and results[0].boxes is not None else None
        if boxes is None:
            return detections

        class_map = self.config["class_map"]

        for box in boxes:
            # اشیایی که track_id ندارند (هنوز ردیابی نشده‌اند) نادیده گرفته می‌شوند
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_map.get(cls, f"class_{cls}")
            track_id = int(box.id[0])

            # به‌روزرسانی اولین/آخرین فریم مشاهده این شیء
            if track_id not in object_first_last:
                object_first_last[track_id] = {
                    "first": frame_id, "last": frame_id, "class": class_name
                }
            else:
                object_first_last[track_id]["last"] = frame_id

            center_x = round((x1 + x2) / 2, 2)
            center_y = round((y1 + y2) / 2, 2)
            object_trajectory.setdefault(track_id, []).append(
                (center_x, center_y, frame_id)
            )

            detections.append({
                "track_id": track_id,
                "class_id": cls,
                "class_name": class_name,
                "bbox": {
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                },
                "center": {"x": center_x, "y": center_y},
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
                "confidence": round(conf, 3),
            })

        return detections

    def _attach_ppe_flags(self, detections: List[Dict[str, Any]]) -> None:
        """
        Attach PPE (hardhat/vest) presence flags to each "Person" detection
        by matching bounding-box overlap with Hardhat/Safety Vest detections.

        Mutates the detection dicts in place, adding: has_hardhat, has_vest,
        has_ppe, ppe_list.

        Args:
            detections: Detections for a single frame.
        """
        person_cls = self.config["person_class_name"]
        hardhat_cls = self.config["hardhat_class_name"]
        vest_cls = self.config["vest_class_name"]
        ppe_iou_threshold = self.config["iou_ppe_threshold"]

        persons = [d for d in detections if d["class_name"] == person_cls]
        hardhats = [d for d in detections if d["class_name"] == hardhat_cls]
        vests = [d for d in detections if d["class_name"] == vest_cls]

        for person in persons:
            person["has_hardhat"] = False
            person["has_vest"] = False
            person["ppe_list"] = []

            for hardhat in hardhats:
                if _calculate_iou(person["bbox"], hardhat["bbox"]) > ppe_iou_threshold:
                    person["has_hardhat"] = True
                    person["ppe_list"].append(hardhat_cls)
                    break

            for vest in vests:
                if _calculate_iou(person["bbox"], vest["bbox"]) > ppe_iou_threshold:
                    person["has_vest"] = True
                    person["ppe_list"].append(vest_cls)
                    break

            person["has_ppe"] = person["has_hardhat"] or person["has_vest"]

    @staticmethod
    def _attach_relative_geometry(
            detections: List[Dict[str, Any]], frame_width: int, frame_height: int
    ) -> None:
        """
        Attach frame-relative position/size (0..1 range) to each detection,
        useful for zone-based rules regardless of video resolution.

        Args:
            detections: Detections for a single frame (mutated in place).
            frame_width: Width of the frame in pixels.
            frame_height: Height of the frame in pixels.
        """
        for det in detections:
            det["relative_position"] = {
                "x": round(det["center"]["x"] / frame_width, 3),
                "y": round(det["center"]["y"] / frame_height, 3),
            }
            det["relative_size"] = {
                "width": round(det["width"] / frame_width, 3),
                "height": round(det["height"] / frame_height, 3),
            }

    @staticmethod
    def _summarize_objects(
            object_first_last: Dict[int, Dict[str, Any]],
            object_trajectory: Dict[int, List[tuple]],
            fps: float,
    ) -> tuple:
        """
        Build the object_durations / object_movement summary blocks from
        the raw per-track state collected during `run()`.

        Args:
            object_first_last: first/last frame + class name per track_id.
            object_trajectory: list of (x, y, frame_id) points per track_id.
            fps: Video frame rate, used to convert frame counts to seconds.

        Returns:
            Tuple of (object_durations dict, object_movement dict).
        """
        object_durations: Dict[str, Any] = {}
        object_movement: Dict[str, Any] = {}

        for track_id, info in object_first_last.items():
            first, last = info["first"], info["last"]
            duration_frames = last - first + 1
            object_durations[str(track_id)] = {
                "class": info["class"],
                "first_frame": first,
                "last_frame": last,
                "duration_frames": duration_frames,
                "duration_seconds": round(duration_frames / fps, 2) if fps else None,
            }

            trajectory = object_trajectory.get(track_id, [])
            if len(trajectory) > 1:
                total_distance = sum(
                    _euclidean_distance(
                        (trajectory[i - 1][0], trajectory[i - 1][1]),
                        (trajectory[i][0], trajectory[i][1]),
                    )
                    for i in range(1, len(trajectory))
                )
                object_movement[str(track_id)] = {
                    "total_distance": round(total_distance, 2),
                    "avg_speed": round(total_distance / len(trajectory), 3),
                    "trajectory_points": len(trajectory),
                }

        return object_durations, object_movement


# ============================================================
# مرحله ۲ + ۳: ترکیب تشخیص + Rule Engine در یک Wrapper واحد
# ============================================================

class SafetyVideoPipeline:
    """
    High-level wrapper that ties YOLO detection/tracking and the safety
    rule engine together behind a single method call.

    This is the object the LLM layer should instantiate once (e.g. at
    service startup, so the YOLO model is loaded a single time) and then
    reuse for every uploaded video via `process_video()`.

    Example:
        >>> pipeline = SafetyVideoPipeline(
        ...     detector_config={"model_path": "models/best.pt"}
        ... )
        >>> result = pipeline.process_video("warehouse.mp4")
        >>> result["alerts_report"]["total_alerts"]
    """

    def __init__(
            self,
            detector_config: Optional[Dict[str, Any]] = None,
            rule_engine_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Build the detector and rule engine once and keep them warm.

        Args:
            detector_config: Overrides for DEFAULT_DETECTOR_CONFIG
                (must include a valid "model_path").
            rule_engine_config: Overrides for the rule engine's
                RULE_ENGINE_DEFAULT_CONFIG (distance/time thresholds, etc).
        """
        self.detector = YOLODetector(detector_config)
        self.rule_engine = RuleEngine(rule_engine_config)

    def process_video(
            self,
            video_path: str,
            save_tracking_json: Optional[str] = None,
            save_alerts_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full video -> detection -> rule-engine -> JSON pipeline.

        This is the single function the LLM report-generation layer is
        expected to call for every incoming video.

        Args:
            video_path: Path to the input video file (or "0" for webcam).
            save_tracking_json: If given, also persist raw tracking data
                to this path (useful for debugging/auditing).
            save_alerts_json: If given, also persist the alerts report to
                this path.

        Returns:
            A JSON-serializable dict with three top-level keys:
                - "video_info": basic metadata about the processed video
                - "tracking_summary": per-object duration/movement stats
                - "alerts_report": full output of RuleEngine.get_report()
                  (total_alerts, alerts_by_type, severity_distribution,
                  alerts)
            This dict is exactly what should be passed to the LLM prompt
            generator (e.g. HSEPromptGenerator) to produce the HSE report.
        """
        # ری‌ست موتور قوانین تا آلارم‌های ویدیوی قبلی باقی نمانند
        self.rule_engine.reset()

        # مرحله ۱: تشخیص + ردیابی
        tracking_data = self.detector.run(video_path)

        if save_tracking_json:
            self._save_json(tracking_data, save_tracking_json)

        # مرحله ۲: اعمال قوانین ایمنی روی هر فریم
        self.rule_engine.fps = tracking_data["metadata"]["fps"]
        for frame in tracking_data["frames"]:
            self.rule_engine.process_frame(frame)

        alerts_report = self.rule_engine.get_report()

        if save_alerts_json:
            self._save_json(alerts_report, save_alerts_json)

        # مرحله ۳: ترکیب همه چیز در یک خروجی JSON واحد
        final_output = {
            "video_info": {
                "source": video_path,
                "fps": tracking_data["metadata"]["fps"],
                "total_frames": tracking_data["metadata"]["total_frames"],
                "processed_at": datetime.datetime.now().isoformat(),
            },
            "tracking_summary": {
                "object_durations": tracking_data["object_durations"],
                "object_movement": tracking_data["object_movement"],
                "classes": tracking_data["metadata"]["classes"],
            },
            "alerts_report": alerts_report,
        }
        return final_output

    @staticmethod
    def _save_json(data: Dict[str, Any], path: str) -> None:
        """
        Persist a dict to disk as UTF-8 JSON (Persian text preserved as-is).

        Args:
            data: JSON-serializable dict.
            path: Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 ذخیره شد: {path}")


# ============================================================
# تابع سرراست برای فراخوانی مستقیم توسط لایه LLM
# ============================================================

def run_pipeline_for_llm(
        video_path: str,
        model_path: str,
        detector_overrides: Optional[Dict[str, Any]] = None,
        rule_engine_overrides: Optional[Dict[str, Any]] = None,
        save_tracking_json: Optional[str] = None,
        save_alerts_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-shot convenience function: build a pipeline with sensible defaults,
    process a single video, and return the final JSON dict.

    Prefer instantiating `SafetyVideoPipeline` directly (and reusing it
    across multiple videos) in a long-running service, since this function
    reloads the YOLO model on every call.

    Args:
        video_path: Path to the input video file.
        model_path: Path to the YOLO .pt weights file.
        detector_overrides: Optional extra overrides for the detector config.
        rule_engine_overrides: Optional extra overrides for the rule engine
            config (distance/time thresholds).
        save_tracking_json: Optional path to also persist tracking data.
        save_alerts_json: Optional path to also persist the alerts report.

    Returns:
        Final JSON dict (video_info / tracking_summary / alerts_report),
        ready to be handed to the LLM report generator.
    """
    detector_config = {"model_path": model_path, **(detector_overrides or {})}
    pipeline = SafetyVideoPipeline(
        detector_config=detector_config,
        rule_engine_config=rule_engine_overrides,
    )
    return pipeline.process_video(
        video_path,
        save_tracking_json=save_tracking_json,
        save_alerts_json=save_alerts_json,
    )


# ============================================================
# اجرای مستقل از خط فرمان (برای تست دستی)
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for standalone/manual testing.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Run the YOLO + Rule Engine pipeline on a video and print/save the JSON result."
    )
    parser.add_argument("--video", required=True, help="مسیر فایل ویدیو یا 0 برای وبکم")
    parser.add_argument("--model", required=True, help="مسیر وزن مدل یولو (.pt)")
    parser.add_argument("--output", default=None, help="مسیر ذخیره خروجی نهایی JSON (اختیاری)")
    parser.add_argument("--save-tracking", default=None, help="مسیر ذخیره tracking خام (اختیاری)")
    parser.add_argument("--save-alerts", default=None, help="مسیر ذخیره گزارش آلارم‌ها (اختیاری)")
    return parser


def main() -> None:
    """CLI entry point for manual/local testing of the wrapper."""
    args = _build_arg_parser().parse_args()

    result = run_pipeline_for_llm(
        video_path=args.video,
        model_path=args.model,
        save_tracking_json=args.save_tracking,
        save_alerts_json=args.save_alerts,
    )

    if args.output:
        SafetyVideoPipeline._save_json(result, args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
