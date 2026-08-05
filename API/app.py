"""
FastAPI backend for the Safety Monitoring System.

This service is the single integration point between:
    - YOLO/mains.py        (SafetyVideoPipeline: detection + tracking + rule engine)
    - LLM/prompt.py         (HSEPromptGenerator / AlertsSummarizer: ready-made prompts)
    - LLM/llm_client.py     (LLMModel: local Qwen2.5 inference, requires an NVIDIA GPU)

None of those modules are modified here -- they are only imported and called
programmatically. This file wires them together behind a small HTTP API that
the Gradio UI (UI/app.py) talks to.

Endpoints:
    GET  /health   -> liveness + whether the YOLO pipeline / LLM loaded successfully
    POST /analyze  -> upload a video, get back alerts + an LLM-generated HSE report

Custom prompt support:
    The /analyze endpoint accepts an optional `custom_prompt` form field. When
    provided, it is combined with a compact summary of that video's detected
    alerts and sent to the LLM instead of the built-in Persian/English
    templates in HSEPromptGenerator.
"""

from __future__ import annotations

import json
import os
import logging
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Make the existing project modules importable WITHOUT modifying them.
#
# YOLO/mains.py itself does `from rules_eng import RuleEngine` (a bare,
# top-level import), so YOLO/ has to be on sys.path directly -- not just
# the project root -- for that import inside mains.py to resolve.
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = PROJECT_ROOT / "YOLO"

for _path in (str(PROJECT_ROOT), str(YOLO_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from YOLO.mains import SafetyVideoPipeline
from LLM.llm_client import LLMModel
from LLM.prompt import AlertsSummarizer, HSEPromptGenerator

# ============================================================
# Configuration (overridable via environment variables, e.g. from
# docker-compose.yml)
# ============================================================
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", str(YOLO_DIR / "models" / "best.onnx"))
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", str(PROJECT_ROOT / "LLM" / "model"))
YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.3"))
YOLO_IOU_THRESHOLD = float(os.environ.get("YOLO_IOU_THRESHOLD", "0.45"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))

# Bounds for the user-selectable max_new_tokens passed to LLMModel.generate().
# Kept configurable via env vars since the right ceiling depends on available
# VRAM; DEFAULT matches LLMModel.generate()'s own default (see llm_client.py).
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "300"))
MIN_MAX_NEW_TOKENS = int(os.environ.get("MIN_MAX_NEW_TOKENS", "16"))
MAX_MAX_NEW_TOKENS = int(os.environ.get("MAX_MAX_NEW_TOKENS", "2048"))

ALLOWED_LANGUAGES = {"fa", "en"}
ALLOWED_MODES = {"summary", "detailed", "json"}


# ============================================================
# App state -- the YOLO pipeline and the LLM are heavy objects, loaded once
# at startup and reused for every request.
# ============================================================
class PipelineState:
    pipeline: Optional[SafetyVideoPipeline] = None
    llm: Optional[LLMModel] = None
    pipeline_error: Optional[str] = None
    llm_error: Optional[str] = None


state = PipelineState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # --- YOLO detector + rule engine ---
    try:
        state.pipeline = SafetyVideoPipeline(
            detector_config={
                "model_path": YOLO_MODEL_PATH,
                "conf_threshold": YOLO_CONF_THRESHOLD,
                "iou_threshold": YOLO_IOU_THRESHOLD,
            }
        )
        logger.info(f"[startup] YOLO pipeline loaded from {YOLO_MODEL_PATH}")
    except Exception as exc:
        state.pipeline_error = str(exc)
        logger.error(f"[startup] Failed to load YOLO pipeline: {exc}", exc_info=True)

    # --- Local LLM (requires CUDA; LLMModel raises RuntimeError otherwise) ---
    try:
        state.llm = LLMModel(model_path=LLM_MODEL_PATH)
        logger.info(f"[startup] LLM loaded from {LLM_MODEL_PATH}")
    except Exception as exc:
        state.llm_error = str(exc)
        # warning, not error: a missing GPU is a degraded-but-expected mode
        # (e.g. running on CPU-only hardware) -- the API stays up and every
        # non-LLM feature keeps working; see the "GPU handling" note in
        # docker-compose.yml.
        logger.warning(f"[startup] LLM not available: {exc}", exc_info=True)

    yield

    state.pipeline = None
    state.llm = None


app = FastAPI(
    title="Safety Monitoring System API",
    description="Video -> YOLO detection/tracking -> rule engine -> LLM HSE report",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Response models
# ============================================================
class HealthResponse(BaseModel):
    status: str
    yolo_pipeline_loaded: bool
    llm_loaded: bool
    pipeline_error: Optional[str] = None
    llm_error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    video_info: Dict[str, Any]
    tracking_summary: Dict[str, Any]
    alerts_report: Dict[str, Any]
    report_text: str
    prompt_used: str
    mode: str
    language: str
    custom_prompt_used: bool
    max_new_tokens: int


# ============================================================
# Helpers
# ============================================================
def _normalize_alerts_for_summarizer(alerts_report: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in fields AlertsSummarizer (LLM/prompt.py) assumes exist.

    AlertsSummarizer._compute_track_summary() unconditionally reads
    alert['track_id'], alert['class_name'], and alert['speed'] for every
    alert. In practice, RuleEngine (YOLO/rules_eng.py) only attaches
    class_name/speed to "high_speed" alerts, and crowd_detection alerts
    don't have a track_id at all. Without this normalization step,
    HSEPromptGenerator crashes with a KeyError on any realistic alert set.

    This builds a defensive copy with sane defaults filled in -- it does
    NOT modify prompt.py or rules_eng.py.
    """
    normalized = dict(alerts_report)
    fixed_alerts = []
    for alert in alerts_report.get("alerts", []):
        fixed = dict(alert)
        fixed.setdefault("class_name", fixed.get("type", "unknown"))
        fixed.setdefault("speed", 0.0)
        fixed.setdefault("track_id", "n/a")
        fixed_alerts.append(fixed)
    normalized["alerts"] = fixed_alerts
    return normalized


def _build_ready_made_prompt(alerts_report: Dict[str, Any], mode: str, language: str) -> str:
    normalized = _normalize_alerts_for_summarizer(alerts_report)
    generator = HSEPromptGenerator(alerts_data=normalized)
    if mode == "detailed":
        return generator.generate_detailed_prompt(language=language)
    if mode == "json":
        return generator.generate_json_prompt()
    return generator.generate_summary_prompt(language=language)


def _build_custom_prompt(alerts_report: Dict[str, Any], custom_prompt: str, language: str) -> str:
    normalized = _normalize_alerts_for_summarizer(alerts_report)
    summarizer = AlertsSummarizer(normalized)
    if language == "fa":
        alerts_context = summarizer.get_text_summary()
    else:
        alerts_context = json.dumps(summarizer.get_json_summary(), ensure_ascii=False, indent=2)

    return (
        f"{custom_prompt.strip()}\n\n"
        f"--- Detected safety alerts context (from the uploaded video) ---\n"
        f"{alerts_context}"
    )


# ============================================================
# Routes
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        yolo_pipeline_loaded=state.pipeline is not None,
        llm_loaded=state.llm is not None,
        pipeline_error=state.pipeline_error,
        llm_error=state.llm_error,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    language: str = Form("fa", description="Report language: 'fa' or 'en'"),
    mode: str = Form("summary", description="Ready-made prompt style: 'summary' | 'detailed' | 'json'"),
    custom_prompt: Optional[str] = Form(
        None,
        description=(
            "If set, this replaces the ready-made prompt templates. It is "
            "combined with a summary of this video's alerts before being "
            "sent to the LLM."
        ),
    ),
    max_new_tokens: int = Form(
        DEFAULT_MAX_NEW_TOKENS,
        description=(
            "Maximum number of new tokens the LLM should generate "
            f"(passed through to LLMModel.generate()); must be between "
            f"{MIN_MAX_NEW_TOKENS} and {MAX_MAX_NEW_TOKENS}."
        ),
    ),
) -> AnalyzeResponse:
    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(422, f"language must be one of {sorted(ALLOWED_LANGUAGES)}")
    if mode not in ALLOWED_MODES:
        raise HTTPException(422, f"mode must be one of {sorted(ALLOWED_MODES)}")
    if not (MIN_MAX_NEW_TOKENS <= max_new_tokens <= MAX_MAX_NEW_TOKENS):
        raise HTTPException(
            422,
            f"max_new_tokens must be between {MIN_MAX_NEW_TOKENS} and {MAX_MAX_NEW_TOKENS}",
        )

    if state.pipeline is None:
        raise HTTPException(
            503,
            f"Video pipeline is not available (failed to load YOLO model): {state.pipeline_error}",
        )

    # --- Persist the upload to a temp file; SafetyVideoPipeline expects a path ---
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    content = await video.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Video exceeds the {MAX_UPLOAD_MB}MB upload limit.")

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = state.pipeline.process_video(tmp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Video processing failed: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    alerts_report = result["alerts_report"]

    use_custom = bool(custom_prompt and custom_prompt.strip())
    if use_custom:
        final_prompt = _build_custom_prompt(alerts_report, custom_prompt, language)
    else:
        final_prompt = _build_ready_made_prompt(alerts_report, mode, language)

    if state.llm is None:
        raise HTTPException(
            503,
            f"LLM model is not available (requires an NVIDIA GPU): {state.llm_error}",
        )

    try:
        report_text = state.llm.generate(final_prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"LLM generation failed: {exc}") from exc

    return AnalyzeResponse(
        max_new_tokens=max_new_tokens,
        video_info=result["video_info"],
        tracking_summary=result["tracking_summary"],
        alerts_report=alerts_report,
        report_text=report_text,
        prompt_used=final_prompt,
        mode=mode,
        language=language,
        custom_prompt_used=use_custom,
    )
