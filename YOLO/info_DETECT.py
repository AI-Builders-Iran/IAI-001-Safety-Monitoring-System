import cv2
import json
import datetime
import math
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# ============================================================
# ۱. تنظیمات (آستانه‌ها رو خودت عوض کن)
# ============================================================
MODEL_PATH = r"C:\Users\E-PART.iR\Desktop\safey_detect_project\runs\detect\end_train_als\yolo_train_ss-3\weights\best.pt"
VIDEO_PATH = r"C:\Users\E-PART.iR\Desktop\mydatas\source_files\source_files\JapanPPE.mp4"
OUTPUT_JSON = "tracking_complete.json"

CONF_THRESHOLD = 0.3              # آستانه تشخیص
IOU_THRESHOLD = 0.45
IOU_PPE_THRESHOLD = 0.15            # آستانه اتصال تجهیزات به شخص

# کلاس‌ها (بر اساس data.yaml خودت)
CLASS_MAP = {
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
    12: "vehicle"

}

PERSON_CLASS = 5
HARDHAT_CLASS = 0
VEST_CLASS = 7

# ============================================================
# ۲. توابع کمکی
# ============================================================
def calculate_iou(box1, box2):
    """محاسبه درصد همپوشانی بین دو باکس"""
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

def calculate_distance(p1, p2):
    """فاصله اقلیدسی بین دو نقطه"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_in_zone(center, zone):
    """بررسی اینکه نقطه داخل منطقه مشخص هست یا نه"""
    x, y = center
    return zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]

# ============================================================
# ۳. بارگذاری مدل و باز کردن ویدیو
# ============================================================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

all_frames_data = []
frame_id = 0

# دیکشنری برای ذخیره اولین و آخرین فریم هر شیء
object_first_last = defaultdict(lambda: {"first": None, "last": None, "class": None})

# دیکشنری برای ذخیره تاریخچه مسیر هر شیء (برای تحلیل حرکت)
object_trajectory = defaultdict(list)

print(f"🎬 شروع پردازش ویدیو (FPS: {fps})")
print(f"🎯 آستانه اطمینان: {CONF_THRESHOLD}")

# ============================================================
# ۴. حلقه اصلی: فریم‌به‌فریم
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    # تشخیص + ردیابی
    results = model.track(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        persist=True,
        verbose=False
    )
    
    detections = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        
        # ===== ۱. استخراج همه اشیاء =====
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = CLASS_MAP.get(cls, f"class_{cls}")
            
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])
                if object_first_last[track_id]["first"] is None:
                    object_first_last[track_id]["first"] = frame_id
                    object_first_last[track_id]["class"] = class_name
                object_first_last[track_id]["last"] = frame_id
                
                # ذخیره تاریخچه مسیر
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                object_trajectory[track_id].append((center_x, center_y, frame_id))
                detections.append({
                "track_id": track_id,
                "class_id": cls,
                "class_name": class_name,
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                },
                "center": {
                    "x": round((x1 + x2) / 2, 2),
                    "y": round((y1 + y2) / 2, 2)
                },
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
                "confidence": round(conf, 3)
            })
        
        # ===== ۲. اتصال تجهیزات به Person =====
        persons = [d for d in detections if d["class_name"] == "Person"]
        hardhats = [d for d in detections if d["class_name"] == "Hardhat"]
        vests = [d for d in detections if d["class_name"] == "Safety Vest"]
        
        for person in persons:
            person["has_hardhat"] = False
            person["has_vest"] = False
            person["has_ppe"] = False
            person["ppe_list"] = []
            
            # بررسی کلاه‌ها
            for hardhat in hardhats:
                iou = calculate_iou(person["bbox"], hardhat["bbox"])
                if iou > IOU_PPE_THRESHOLD:
                    person["has_hardhat"] = True
                    person["ppe_list"].append("Hardhat")
                    break
            
            # بررسی جلیقه‌ها
            for vest in vests:
                iou = calculate_iou(person["bbox"], vest["bbox"])
                if iou > IOU_PPE_THRESHOLD:
                    person["has_vest"] = True
                    person["ppe_list"].append("Safety Vest")
                    break
            
            person["has_ppe"] = person["has_hardhat"] or person["has_vest"]
        
        # ===== ۳. اضافه کردن اطلاعات موقعیت نسبی =====
        frame_height, frame_width = frame.shape[:2]
        for det in detections:
            det["relative_position"] = {
                "x": round(det["center"]["x"] / frame_width, 3),
                "y": round(det["center"]["y"] / frame_height, 3)
            }
            det["relative_size"] = {
                "width": round(det["width"] / frame_width, 3),
                "height": round(det["height"] / frame_height, 3)
            }
    
    # ذخیره اطلاعات فریم
    all_frames_data.append({
        "frame_id": frame_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "frame_size": {
            "width": frame.shape[1],
            "height": frame.shape[0]
        },
        "total_objects": len(detections),
        "detections": detections
    })
    
    if frame_id % 30 == 0:
        print(f"   پردازش فریم {frame_id}...")

cap.release()
print(f"✅ پردازش کامل شد! تعداد فریم‌ها: {frame_id}")

# ============================================================
# ۵. محاسبه مدت زمان و تحلیل حرکت
# ============================================================
object_durations = {}
object_movement = {}

for track_id, data in object_first_last.items():
    if data["first"] is not None and data["last"] is not None:
        first = data["first"]
        last = data["last"]
        class_name = data["class"]
        duration_frames = last - first + 1
        duration_seconds = round(duration_frames / fps, 2)
        
        object_durations[str(track_id)] = {
            "class": class_name,
            "first_frame": first,
            "last_frame": last,
            "duration_frames": duration_frames,
            "duration_seconds": duration_seconds
        }
        
        # تحلیل حرکت (مسافت کل طی شده)
        trajectory = object_trajectory.get(track_id, [])
        if len(trajectory) > 1:
            total_distance = 0
            for i in range(1, len(trajectory)):
                p1 = (trajectory[i-1][0], trajectory[i-1][1])
                p2 = (trajectory[i][0], trajectory[i][1])
                total_distance += calculate_distance(p1, p2)
            
            # سرعت میانگین (پیکسل بر فریم)
            avg_speed = total_distance / len(trajectory) if trajectory else 0
            
            object_movement[str(track_id)] = {
                "total_distance": round(total_distance, 2),
                "avg_speed": round(avg_speed, 3),
                "trajectory_points": len(trajectory)
            }

# ============================================================
# ۶. ساخت خروجی نهایی JSON (همه چیز)
# ============================================================
output_data = {
    "metadata": {
        "model": MODEL_PATH,
        "fps": fps,
        "total_frames": frame_id,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "iou_ppe_threshold": IOU_PPE_THRESHOLD,
        "export_time": datetime.datetime.now().isoformat(),
        "classes": CLASS_MAP
    },
    "object_durations": object_durations,
    "object_movement": object_movement,
    "frames": all_frames_data
}

# ============================================================
# ۷. ذخیره JSON
# ============================================================
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON کامل ذخیره شد: {OUTPUT_JSON}")

# ============================================================
# ۸. نمایش خلاصه
# ============================================================
print("\n📊 خلاصه اطلاعات:")
print(f"   تعداد کل فریم‌ها: {frame_id}")
print(f"   تعداد کل اشیای یکتا: {len(object_durations)}")
print(f"   آستانه اطمینان: {CONF_THRESHOLD}")

# آمار تجهیزات
total_hardhat = 0
total_vest = 0
persons_without_ppe = 0

for frame in all_frames_data:
    for det in frame["detections"]:
        if det["class_name"] == "Hardhat":
            total_hardhat += 1
        elif det["class_name"] == "Safety Vest":
            total_vest += 1
        elif det["class_name"] == "Person" and not det.get("has_ppe", False):
            persons_without_ppe += 1

print(f"   تعداد کل کلاه‌ها: {total_hardhat}")
print(f"   تعداد کل جلیقه‌ها: {total_vest}")
print(f"   افراد بدون تجهیزات: {persons_without_ppe}")

print("\n✅ JSON برای همه شرط‌های ممکن آماده است!")

# ============================================================
# 🆕 ۹. بخش جدید: گزارش آماری کامل (شمارش اشیا)
# ============================================================
print("\n" + "="*60)
print("📊 گزارش آماری کامل (شمارش اشیا)")
print("="*60)

# ۱. تعداد کل اشیای یکتا بر اساس کلاس
class_counts = {}
for track_id, info in object_durations.items():
    cls = info["class"]
    class_counts[cls] = class_counts.get(cls, 0) + 1

print("\n📈 تعداد کل اشیای یکتا:")
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"   {cls}: {count}")

# ۲. آمار دقیق افراد (Person)
persons_with_hardhat = 0
persons_with_vest = 0
persons_with_both = 0
persons_without_ppe = 0
total_person_detections = 0

for frame in all_frames_data:
    for det in frame["detections"]:
        if det["class_name"] == "Person":
            total_person_detections += 1
            has_hardhat = det.get("has_hardhat", False)
            has_vest = det.get("has_vest", False)
            
            if has_hardhat:
                persons_with_hardhat += 1
            if has_vest:
                persons_with_vest += 1
            if has_hardhat and has_vest:
                persons_with_both += 1
            if not has_hardhat and not has_vest:
                persons_without_ppe += 1

print(f"\n👤 آمار دقیق افراد (تشخیص‌ها در فریم‌ها):")
print(f"   تعداد کل تشخیص افراد: {total_person_detections}")
print(f"   افراد با کلاه: {persons_with_hardhat}")
print(f"   افراد با جلیقه: {persons_with_vest}")
print(f"   افراد با هر دو: {persons_with_both}")
print(f"   افراد بدون تجهیزات: {persons_without_ppe}")
# ۳. آمار تجهیزات (کلاه و جلیقه)
total_hardhat_detections = 0
total_vest_detections = 0
for frame in all_frames_data:
    for det in frame["detections"]:
        if det["class_name"] == "Hardhat":
            total_hardhat_detections += 1
        elif det["class_name"] == "Safety Vest":
            total_vest_detections += 1

print(f"\n🪖 آمار تجهیزات:")
print(f"   تعداد کل تشخیص کلاه: {total_hardhat_detections}")
print(f"   تعداد کل تشخیص جلیقه: {total_vest_detections}")

# ۴. آمار فریم‌ها
frame_counts = [frame["total_objects"] for frame in all_frames_data]
avg_objects = sum(frame_counts) / len(frame_counts) if frame_counts else 0
max_objects = max(frame_counts) if frame_counts else 0
min_objects = min(frame_counts) if frame_counts else 0

print(f"\n📊 آمار فریم‌ها:")
print(f"   میانگین اشیاء در هر فریم: {avg_objects:.2f}")
print(f"   حداکثر اشیاء در یک فریم: {max_objects}")
print(f"   حداقل اشیاء در یک فریم: {min_objects}")

# ۵. ذخیره گزارش در فایل
report_data = {
    "class_counts": class_counts,
    "persons": {
        "total_detections": total_person_detections,
        "with_hardhat": persons_with_hardhat,
        "with_vest": persons_with_vest,
        "with_both": persons_with_both,
        "without_ppe": persons_without_ppe
    },
    "equipment": {
        "hardhat_detections": total_hardhat_detections,
        "vest_detections": total_vest_detections
    },
    "frames": {
        "avg_objects": avg_objects,
        "max_objects": max_objects,
        "min_objects": min_objects,
        "total_frames": frame_id
    }
}

with open("statistics_report.json", "w", encoding='utf-8') as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

print("\n✅ گزارش آماری در statistics_report.json ذخیره شد.")