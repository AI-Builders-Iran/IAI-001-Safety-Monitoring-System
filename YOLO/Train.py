from ultralytics import YOLO
from pathlib import Path
import time
import os


def build_model (model_path : str):
    model = YOLO(model_path)
    return model


def train_model (model):
    model.train(
        data=r"C:\Users\E-PART.iR\Desktop\mydatas\data_amir_backup\data.yaml",           # (str, optional) path to data file, i.e. coco8.yaml
        epochs=40,                  # (int) number of epochs to train for
        time=None,                   # (float, optional) number of hours to train for, overrides epochs if supplied
        patience=50,                 # (int) epochs to wait for no observable improvement for early stopping of training
        batch=6,                    # (int) number of images per batch (-1 for AutoBatch)
        imgsz=640,                   # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
        save=True,                   # (bool) save train checkpoints and predict results
        save_period=-1,              # (int) Save checkpoint every x epochs (disabled if < 1)
        cache='disk',                                  # (bool) True/ram, disk or False. Use cache for data loading
        device=0,                 # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers=2,                   # (int) number of worker threads for data loading (per RANK if DDP)
        project="end_train_als",                # (str, optional) project name
        name="yolo_train_ss",                   # (str, optional) experiment name, results saved to 'project/name' directory
        exist_ok=False,              # (bool) whether to overwrite existing experiment
        val=True,                    # (bool) validate/test during training
        pretrained=True,             # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
        optimizer="auto",             # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        verbose=True,                # (bool) whether to print verbose output
        seed=42,                      # (int) random seed for reproducibility
        deterministic=True,          # (bool) whether to enable deterministic mode
        single_cls=False,            # (bool) train multi-class data as single-class
        rect=False,                  # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
        cos_lr=False,                # (bool) use cosine learning rate scheduler
        close_mosaic=10,             # (int) disable mosaic augmentation for final epochs (0 to disable)
        resume=False,                # (bool) resume training from last checkpoint
        amp=True,                    # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
        fraction=1.0,                # (float) dataset fraction to train on (default is 1.0, all images in train set)
        profile=False,               # (bool) profile ONNX and TensorRT speeds during training for loggers
        freeze=None,                 # (int | list, optional) freeze first n layers, or freeze list of layer indices during training
        multi_scale=False,           # (bool) Whether to use multiscale during training
        plots=True,                  # (bool) save plots and images during train/val
        # Segmentation
        overlap_mask=True,           # (bool) masks should overlap during training (segment train only)
        mask_ratio=4,                # (int) mask downsample ratio (segment train only)
        # Classification
        dropout=0.0,                 # (float) use dropout regularization (classify train only)
        # Hyperparameters
        lr0=0.005,                    # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf=0.01,                    # (float) final learning rate (lr0 * lrf)
        momentum=0.937,              # (float) SGD momentum/Adam beta1
        weight_decay=0.0005,         # (float) optimizer weight decay 5e-4
        warmup_epochs=3.0,           # (float) warmup epochs (fractions ok)
        warmup_momentum=0.8,         # (float) warmup initial momentum
        warmup_bias_lr=0.1,          # (float) warmup initial bias lr
        box=8.0,                     # (float) box loss gain
        cls=0.6,                     # (float) cls loss gain (scale with pixels)
        dfl=1.5,                     # (float) dfl loss gain
        pose=12.0,                   # (float) pose loss gain
        kobj=1.0,                    # (float) keypoint obj loss gain
        label_smoothing=0.05,         # (float) label smoothing (fraction)
        nbs=64,                      # (int) nominal batch size
        hsv_h=0.015,                 # (float) image HSV-Hue augmentation (fraction)
        hsv_s=0.5,                   # (float) image HSV-Saturation augmentation (fraction)
        hsv_v=0.2,                   # (float) image HSV-Value augmentation (fraction)
        degrees=5.0,                # (float) image rotation (+/- deg)
        translate=0.1,               # (float) image translation (+/- fraction)
        scale=0.7,                   # (float) image scale (+/- gain)
        shear=0.0,                   # (float) image shear (+/- deg)
        perspective=0.0,             # (float) image perspective (+/- fraction), range 0-0.001
        flipud=0.0,                  # (float) image flip up-down (probability)
        fliplr=0.5,                  # (float) image flip left-right (probability)
        bgr=0.001,                     # (float) image channel BGR (probability)
        mosaic=0.9,                  # (float) image mosaic (probability)
        mixup=0.1,                   # (float) image mixup (probability)
        copy_paste=0.2,              # (float) segment copy-paste (probability)
        auto_augment="randaugment",  # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
        erasing=0.4,                 # (float) probability of random erasing during classify training [0-0.9], 0 is no erasing, must be < 1.0.
        crop_fraction=1.0,           # (float) image crop fraction for classification [0.1-1], 1.0 is no crop, must be > 0.
    )

# if __name__ == '__main__':
#     model = build_model(r"C:\Users\E-PART.iR\Desktop\safey_detect_project\runs\detect\end_train\yolo_train-5\weights\best.pt")
#     train_model(model)

#____________________________________________________________________________________________

# def evals_model (model_path):
#     model = YOLO(model_path)
#     results = model.val(
#         data=r"C:\Users\E-PART.iR\Desktop\mydatas\data_amir_backup\data.yaml",  # (str) Specifies the path to the dataset configuration file (e.g., coco8.yaml).
#         imgsz=640,          # (int) Defines the size of input images. All images are resized to this dimension before processing.
#         batch=16,           # (int) Sets the number of images per batch. Use -1 for AutoBatch, which automatically adjusts based on GPU memory availability.
#         save_json=True,    # (bool) If True, saves the results to a JSON file for further analysis or integration with other tools.
#         save_hybrid=False,  # (bool) If True, saves a hybrid version of labels that combines original annotations with additional model predictions.
#         conf=0.25,         # (float) Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.
#         iou=0.6,            # (float) Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.
#         max_det=300,        # (int) Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections.
#         half=True,          # (bool) Enables half-precision (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on accuracy.
#         device=0,        # (str | int) Specifies the device for validation (cpu, cuda:0, etc.). Allows flexibility in utilizing CPU or GPU resources.
#         dnn=False,          # (bool) If True, uses the OpenCV DNN module for ONNX model inference, offering an alternative to PyTorch inference methods.
#         plots=True,        # (bool) When set to True, generates and saves plots of predictions versus ground truth for visual evaluation of the model's performance.
#         rect=False,         # (bool) If True, uses rectangular inference for batching, reducing padding and potentially increasing speed and efficiency.
#         split="val",        # (str) Determines the dataset split to use for validation (val, test, or train).
#     )
#     return results


# def print_metrics (result) -> None :
#     print(f'mAP50 : { result.box.map50:.4f}')
#     print(f'mAP50-95 : {result.box.map:.4f}')
#     print(f'Precision :{result.box.mp:.4f}')
#     print(f'Recall :{result.box.mr}')


# def print_f1 (result):

#     p = result.box.mp 
#     r = result.box.mr

#     f1 = (2 * p * r) / (p + r + 1e-8)

#     print(f'F1 :{f1:.4f}')


# def print_speed(results):

#     speed = results.speed

#     print("\n========== Speed ==========\n")

#     print(f"Preprocess : {speed['preprocess']:.3f} ms")
#     print(f"Inference  : {speed['inference']:.3f} ms")
#     print(f"Postprocess: {speed['postprocess']:.3f} ms")


# from pathlib import Path

# images = len(list(Path("datasets/val/images").glob("*.*")))

# print(images)


# if __name__ == "__main__":

#     results = evals_model(r"C:\Users\E-PART.iR\Desktop\safey_detect_project\runs\detect\end_train_als\yolo_train_ss-3\weights\best.pt")

#     print_metrics(results)
#     print_f1(results)
#     print_speed(results)
    


# model = build_model(r"C:\Users\E-PART.iR\Desktop\safey_detect_project\runs\detect\end_train_als\yolo_train_ss-3\weights\best.pt")

# # __________________________
# model.predict(
#     source=r"C:\Users\E-PART.iR\Downloads\videoplayback.mp4",            # (str, optional) source directory for images or videos
#     imgsz=720,            # (int | list) input images size as int or list[w,h] for predict
#     conf=0.35,            # (float) minimum confidence threshold
#     iou=0.7,              # (float) intersection over union (IoU) threshold for NMS
#     device=None,          # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
#     batch=1,              # (int) batch size
#     half=False,           # (bool) use FP16 half-precision inference
#     max_det=300,          # (int) Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections.
#     vid_stride=1,         # (int) video frame-rate stride
#     stream_buffer=False,  # (bool) buffer all streaming frames (True) or return the most recent frame (False)
#     visualize=False,      # (bool) visualize model features
#     augment=False,        # (bool) apply image augmentation to prediction sources
#     agnostic_nms=False,   # (bool) class-agnostic NMS
#     classes=None,         # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
#     retina_masks=False,   # (bool) use high-resolution segmentation masks
#     embed=None,           # (list[int], optional) return feature vectors/embeddings from given layers
#     show=True,           # (bool) show predicted images and videos if environment allows
#     save=True,            # (bool) save prediction results
#     save_frames=False,    # (bool) save predicted individual video frames
#     save_txt=False,       # (bool) save results as .txt file
#     save_conf=False,      # (bool) save results with confidence scores
#     save_crop=False,      # (bool) save cropped images with results
#     stream=False,         # (bool) for processing long videos or numerous images with reduced memory usage by returning a generator
#     verbose=True,         # (bool) enable/disable verbose inference logging in the terminal
# )



#_________________________ vals model :
from ultralytics import YOLO
import pandas as pd
import json

def get_all_metrics(model_path, data_yaml, split='val'):
    """
    دریافت همه متریک‌های مدل YOLO
    
    Args:
        model_path: مسیر مدل (best.pt)
        data_yaml: مسیر فایل data.yaml
        split: 'val' یا 'test'
    
    Returns:
        dict: همه متریک‌ها
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split=split)
    
    # ===== متریک‌های کلی =====
    metrics = {
        # متریک‌های اصلی
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'F1': (2 * results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0,
        
        # متریک‌های تکمیلی
        'mAP75': results.box.map75 if hasattr(results.box, 'map75') else None,
        'AP_per_class': results.box.ap_class_index if hasattr(results.box, 'ap_class_index') else None,
    }
    
    # ===== متریک‌های هر کلاس =====
    class_metrics = {}
    if hasattr(results.box, 'ap_per_class'):
        for i, (ap50, ap, p, r) in enumerate(zip(
            results.box.ap50_per_class if hasattr(results.box, 'ap50_per_class') else [],
            results.box.ap_per_class if hasattr(results.box, 'ap_per_class') else [],
            results.box.p_per_class if hasattr(results.box, 'p_per_class') else [],
            results.box.r_per_class if hasattr(results.box, 'r_per_class') else []
        )):
            class_metrics[f'class_{i}'] = {
                'AP50': ap50,
                'AP50-95': ap,
                'Precision': p,
                'Recall': r
            }
    
    metrics['per_class'] = class_metrics
    
    # ===== سرعت =====
    metrics['speed'] = {
        'preprocess_ms': results.speed.get('preprocess', 0),
        'inference_ms': results.speed.get('inference', 0),
        'postprocess_ms': results.speed.get('postprocess', 0),
    }
    
    # ===== متریک‌های زمان =====
    total_time = results.speed.get('preprocess', 0) + results.speed.get('inference', 0) + results.speed.get('postprocess', 0)
    metrics['total_time_ms'] = total_time
    metrics['fps'] = 1000 / total_time if total_time > 0 else 0
    
    return metrics

# ============================================================
# استفاده
# ============================================================
if __name__ == "__main__":
    # ۱. همه متریک‌ها رو بگیر
    all_metrics = get_all_metrics(
        model_path=r"C:\Users\E-PART.iR\Desktop\safey_detect_project\runs\detect\end_train_als\yolo_train_ss-3\weights\best.pt",
        data_yaml=r"C:\Users\E-PART.iR\Desktop\mydatas\data_amir_backup\data.yaml"
    )
    
    # ۲. نمایش
    print("="*60)
    print("📊 همه متریک‌های مدل")
    print("="*60)
    print(f"🎯 mAP50: {all_metrics['mAP50']:.4f}")
    print(f"🎯 mAP50-95: {all_metrics['mAP50-95']:.4f}")
    print(f"🎯 Precision: {all_metrics['Precision']:.4f}")
    print(f"🎯 Recall: {all_metrics['Recall']:.4f}")
    print(f"🎯 F1: {all_metrics['F1']:.4f}")
    print(f"⚡ سرعت: {all_metrics['fps']:.1f} FPS")
    print(f"⏱️ زمان کل: {all_metrics['total_time_ms']:.2f} ms")
    
    print("\n📈 متریک‌های هر کلاس:")
    for cls, m in all_metrics['per_class'].items():
        print(f"   {cls}: AP50={m['AP50']:.4f}, AP={m['AP50-95']:.4f}")
    
    # ۳. ذخیره در فایل
    import json
    with open("all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n✅ متریک‌ها در all_metrics.json ذخیره شد.")
    
    # ۴. ذخیره در CSV
    import pandas as pd
    df = pd.DataFrame({
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1', 'FPS'],
        'Value': [
            all_metrics['mAP50'],
            all_metrics['mAP50-95'],
            all_metrics['Precision'],
            all_metrics['Recall'],
            all_metrics['F1'],
            all_metrics['fps']
        ]
    })
    df.to_csv("all_metrics.csv", index=False)
    print("✅ متریک‌ها در all_metrics.csv ذخیره شد.")