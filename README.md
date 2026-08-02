<div align="center">

# 🦺 Safety Monitoring System

### Intelligent Workplace Safety Monitoring using Computer Vision & LLMs

Detect real-time safety violations on construction sites, factories, and warehouses using YOLO, a Rule Engine, and an LLM-powered reporting layer.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=yolo&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?logo=postgresql&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-LLM-yellow)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 📑 Table of Contents

- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [🏗 Project Structure](#-project-structure)
- [📊 Dataset](#-dataset)
- [🛠 Tech Stack](#-tech-stack)
- [⚙ Pipeline](#-pipeline)
- [🤖 Detected Classes & Safety Rules](#-detected-classes--safety-rules)
- [📈 Model Performance](#-model-performance)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🚀 Installation](#-installation)
- [▶ Running the System](#-running-the-system)
- [📊 Example Alert & Report](#-example-alert--report)
- [👥 Team](#-team)
- [🤝 Contributing](#-contributing)
- [⭐ Support](#-support)
- [📄 License](#-license)

---

# 📖 Overview

Workplace safety violations — missing PPE, machinery operating too close to workers, unauthorized access to hazardous zones — cause serious injuries and financial losses every year across construction sites, factories, and warehouses.

This project builds an **end-to-end AI safety monitoring pipeline** that turns any existing site camera into an automated safety observer, capable of detecting violations in real time and generating management-ready reports.

The project is designed as a complete, production-oriented application including:

- Object detection & multi-object tracking
- Rule-based violation analysis
- Event storage & REST API
- Real-time monitoring dashboard
- LLM-powered daily/weekly HSE reporting

---

# ✨ Features

- Real-time multi-class object detection (PPE, personnel, vehicles, machinery)
- Persistent object tracking across video frames
- PPE-to-worker association (hardhat / vest compliance)
- Configurable rule engine (8 industrial safety rules)
- Proximity, crowd-density, idleness, and abnormal-speed detection
- Structured event storage in PostgreSQL via FastAPI
- Live monitoring dashboard with real-time stats and alerts
- LLM-generated daily/weekly HSE reports (Persian / English)
- Fully configurable thresholds for every safety rule

---

# 🏗 Project Structure

```text
safety-monitoring-system/

│
├── cv/
│   ├── info_detect.py
│   ├── train.py
│   └── models/
│       ├── best.pt
│
├── rules_engine.py
│
├── pipeline_orchestrator.py
│
├── LLM/
│   ├── models.py
│   ├── llm_client.py
│   ├── prompt.py
│   ├── main.py
│
├── backend/
│   ├── app.py
│
├── dashboard/
│
├── data/
│   ├── data.yaml
│
├── reports/
│   ├── figures/
│
├── Team_Report/
│
├── docs/
│
├── .dockerignore
│
├── .gitignore
│
├── Dockerfile
│
├── SYSTEM_PROMPT.md
│
├── README.md
│
└── requirements.txt
```

---

# 📊 Dataset

This project uses the **Construction Site Safety Image Dataset**, a Roboflow-exported, YOLO-format benchmark for PPE and workplace-hazard detection.

> **Source:** Construction Site Safety Image Dataset (Kaggle / Roboflow)

📂 **View Dataset:**
[Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)

### Dataset Statistics

| Property | Value |
|-----------|--------|
| Images | 2,801 |
| Classes | 10 |
| Format | YOLOv8 (Ultralytics-ready) |
| Size | ~1.2 GB |
| Split | Train / Valid / Test (pre-split) |
| Target | Multi-class Object Detection |

The dataset ships pre-labeled and pre-split, allowing the team to move directly into model training and system integration without manual annotation.

---

# 🛠 Tech Stack

## Computer Vision

- Ultralytics YOLOv8
- OpenCV
- ByteTrack

## Rule Engine

- Python
- Pydantic

## Backend

- FastAPI
- PostgreSQL

## LLM / Reporting

- Hugging Face Transformers
- Qwen2.5-Instruct (local inference)
- Jinja2

## Dashboard

- React / Streamlit
- Recharts / Plotly

## Deployment

- Docker

## Version Control

- Git
- GitHub

---

# ⚙ Pipeline

```
Site Camera
     │
     ▼
YOLOv8 Detection
     │
     ▼
ByteTrack Tracking
     │
     ▼
PPE ↔ Person Association
     │
     ▼
Rule Engine (8 Safety Rules)
     │
     ▼
Alert Storage (PostgreSQL)
     │
     ├──────────────► Dashboard (real-time)
     │
     ▼
LLM Report Generator
     │
     ▼
Daily / Weekly HSE Report
```

---

# 🤖 Detected Classes & Safety Rules

**Detected classes:** `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Mask`, `Person`, `machinery`, `vehicle`, `Safety Cone`, and dataset-specific extensions.

**Implemented safety rules:**

- Vehicle–person proximity
- Machinery–person proximity
- Missing PPE (general)
- Missing hardhat
- Missing safety vest
- Prolonged worker idleness
- Crowd density
- Abnormal / high-speed movement

---

# 📈 Model Performance

The current production model is a fine-tuned **YOLOv8** trained on the Construction Site Safety dataset. Final benchmark numbers are being finalized by the Model Training & Evaluation teams and will be published here once validated.

| Metric | Score |
|:--------|------:|
| Model | YOLOv8 (fine-tuned) |
| Confidence Threshold | 0.30 |
| IoU Threshold (NMS) | 0.45 |
| mAP50 | *pending evaluation* |
| mAP50-95 | *pending evaluation* |
| Precision | *pending evaluation* |
| Recall | *pending evaluation* |
| F1-Score | *pending evaluation* |
| Inference Speed (FPS) | *pending evaluation* |

> **Note:** As with any real-world detection task, model selection prioritizes **Precision, Recall, F1-Score, and mAP** per class over raw accuracy, since safety-critical classes (e.g. `NO-Hardhat`, `machinery`) matter far more than overall correctness.

---

# 📈 Evaluation Metrics

Since safety violation detection is a multi-class, safety-critical problem, multiple metrics are tracked:

- mAP50
- mAP50-95
- Precision
- Recall
- F1-Score
- Per-class Average Precision
- Inference Speed (FPS / latency)
- Confusion Matrix

---

# 🚀 Installation

Clone repository

```bash
git clone https://github.com/AI-Builders-Iran/safety-monitoring-system.git
```

Enter project

```bash
cd safety-monitoring-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

(Optional) Download the local LLM used for report generation

```bash
python -c "from LLM.llm_client import download_model; download_model()"
```

---

# ▶ Running the System

### Run detection + tracking on a video

```bash
python cv/info_detect.py --video path/to/video.mp4 --model cv/models/best.pt --output tracking_complete.json
```

### Run the full pipeline (detection → rule engine)

```bash
python pipeline_orchestrator.py --video path/to/video.mp4 --model cv/models/best.pt
```

### Run the Backend API

```bash
uvicorn backend.app:app --reload
```

Once the server is running, open:

### Swagger UI

```
http://127.0.0.1:8000/docs
```

### ReDoc

```
http://127.0.0.1:8000/redoc
```

---

# 📊 Example Alert & Report

### Example Alert (Rule Engine Output)

```json
{
  "type": "machinery_person_proximity",
  "track_id": 4,
  "distance_m": 0.72,
  "severity": "critical",
  "message": "⚠️ Machinery 4 approached person 4! Distance: 0.72m"
}
```

### Example Daily HSE Report (LLM Output)

```
📋 گزارش روزانه — ۲۴ تیر ۱۴۰۳

امروز ۲۴ هشدار در کارگاه ثبت شده که از این تعداد:
  • ۱۵ مورد: کارگر بدون کلاه ایمنی
  • ۵ مورد: نزدیکی بیش از حد لیفتراک به کارگران
  • ۳ مورد: کارگر بدون جلیقه‌ی ایمنی
  • ۱ مورد: ورود به منطقه‌ی خطرناک

پیشنهادات:
  ۱. آموزش مجدد کارگران در مورد استفاده از کلاه ایمنی
  ۲. نصب هشدار صوتی در نزدیکی لیفتراک‌ها
  ۳. بازبینی تجهیزات ایمنی شیفت شب
```

---

# 👥 Team

This project is developed by the **AI Builders Iran** team.

| Member | Contact | Role |
|--------|---------|------|
| Amirdevlp | [@Amirdevlp](https://t.me/Amirdevlp) | Lead — Dataset · Lead — Rule Engine · Core Member — Model Training & Tracking |
| Ryhnne7 | [@Ryhnne7](https://t.me/Ryhnne7) | Lead — Model Training (YOLO) & Tracking |
| SonayHajirezaei | [@SonayHajirezaei](https://t.me/SonayHajirezaei) | Lead — Evaluation · Core Member — Rule Engine |
| Hossein_h8304 | [@Hossein_h8304](https://t.me/Hossein_h8304) | Lead — LLM |
| FarshadZ1997 | [@FarshadZ1997](https://t.me/FarshadZ1997) | Lead — Dashboard & Backend |
| a_taherkho86 | [@a_taherkho86](https://t.me/a_taherkho86) | Core Member — Dataset |
| AJzahed | [@AJzahed](https://t.me/AJzahed) | Core Member — Model Training & Tracking |
| HiTech_Manage | [@HiTech_Manage](https://t.me/HiTech_Manage) | Core Member — Evaluation |
| thisisatestforid | [@thisisatestforid](https://t.me/thisisatestforid) | Core Member — Rule Engine |
| Mohii_Mhmdi | [@Mohii_Mhmdi](https://t.me/Mohii_Mhmdi) | Core Member — Rule Engine |
| Beeehrraaad | [@Beeehrraaad](https://t.me/Beeehrraaad) | Core Member — LLM |
| whatever0_00 | [@whatever0_00](https://t.me/whatever0_00) | Core Member — LLM |
| Erfanjenab86 | [@Erfanjenab86](https://t.me/Erfanjenab86) | Core Member — LLM |
| ArianaTheClown | [@ArianaTheClown](https://t.me/ArianaTheClown) | Core Member — Dashboard & Backend |

We collaborate to build an open-source, industrial-grade Computer Vision and AI safety system while learning and growing together.

---

# 🤝 Contributing

Contributions are always welcome.

1. Fork repository

2. Create new branch

```bash
git checkout -b feature/new-feature
```

3. Commit

```bash
git commit -m "Add new feature"
```

4. Push

```bash
git push origin feature/new-feature
```

5. Open Pull Request

> All code is reviewed via Pull Request before merging — no direct commits to the main branch. Please coordinate with your workstream's Lead before major changes.

---

# ⭐ Support

If you found this project useful,

please consider giving it a ⭐ on GitHub.

It helps the project grow.

---

# 📄 License

This project is licensed under the MIT License.
