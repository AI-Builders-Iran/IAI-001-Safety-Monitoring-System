"""
Gradio UI for the Safety Monitoring System.

This is a thin client: it never imports the YOLO/rule-engine/LLM code
directly. It only talks to the FastAPI backend (API/app.py) over HTTP, so
the UI and the API can run in separate containers (see docker-compose.yml).

Flow:
    1. User uploads a video.
    2. User picks a language + one of the ready-made report styles
       (summary / detailed / json) ...
    3. ... OR flips the "use my own prompt" flag and types a free-text
       prompt instead. When that flag is on, the ready-made style picker
       is hidden and ignored -- the custom prompt (plus that video's
       detected alerts as context) is what gets sent to the LLM.
    4. The generated report, the raw alerts, and the tracking summary are
       shown back.
"""

from __future__ import annotations

import os

import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "1800"))

# Matches API/app.py's DEFAULT/MIN/MAX_MAX_NEW_TOKENS env-driven bounds. Kept
# as separate constants here (rather than fetched from the API) so the slider
# renders immediately without an extra round trip; if you change the bounds
# in the API's environment, update these to match.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "300"))
MIN_MAX_NEW_TOKENS = int(os.environ.get("MIN_MAX_NEW_TOKENS", "16"))
MAX_MAX_NEW_TOKENS = int(os.environ.get("MAX_MAX_NEW_TOKENS", "2048"))


def analyze(
    video_path: str | None,
    language: str,
    mode: str,
    use_custom_prompt: bool,
    custom_prompt_text: str,
    max_new_tokens: float,
):
    if not video_path:
        return "⚠️ Please upload a video first.", None, None

    if use_custom_prompt and not (custom_prompt_text or "").strip():
        return (
            "⚠️ 'Use my own prompt' is on but the prompt box is empty. "
            "Type a prompt, or turn the flag off to use a ready-made style.",
            None,
            None,
        )

    try:
        with open(video_path, "rb") as f:
            files = {"video": (os.path.basename(video_path), f, "video/mp4")}
            data = {
                "language": language,
                "mode": mode,
                # gr.Slider always yields a float -- coerce to int for the
                # API's `max_new_tokens: int` form field.
                "max_new_tokens": str(int(max_new_tokens)),
            }
            if use_custom_prompt:
                data["custom_prompt"] = custom_prompt_text

            response = requests.post(
                f"{API_URL}/analyze",
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
    except requests.exceptions.ConnectionError:
        return f"❌ Could not reach the backend API at {API_URL}. Is it running?", None, None
    except requests.exceptions.Timeout:
        return (
            "❌ The request timed out. Try a shorter video, or raise "
            "REQUEST_TIMEOUT_SECONDS.",
            None,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error while calling the API: {exc}", None, None

    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        return f"❌ Backend error ({response.status_code}): {detail}", None, None

    result = response.json()
    return result["report_text"], result["alerts_report"], result["tracking_summary"]


def toggle_custom_prompt(use_custom: bool):
    # Show the free-text box and hide the ready-made style picker (and vice
    # versa) -- this is the "flag" that lets the user send whatever prompt
    # they want instead of the built-in templates.
    return gr.update(visible=use_custom), gr.update(visible=not use_custom)


def check_backend_health() -> str:
    try:
        
        resp = requests.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        pieces = [
            f"API: {API_URL}",
            f"YOLO pipeline loaded: {info.get('yolo_pipeline_loaded')}",
            f"LLM loaded: {info.get('llm_loaded')}",
        ]
        if info.get("pipeline_error"):
            pieces.append(f"pipeline error: {info['pipeline_error']}")
        if info.get("llm_error"):
            pieces.append(f"LLM error: {info['llm_error']}")
        return " | ".join(pieces)
    except Exception as exc:  # noqa: BLE001
        return f"Could not reach backend at {API_URL}: {exc}"


with gr.Blocks(title="Safety Monitoring System") as demo:
    gr.Markdown(
        "# 🦺 Safety Monitoring System\n"
        "Upload a warehouse / construction-site video to detect safety "
        "violations (PPE, proximity, crowding, idleness, speed) and "
        "generate an HSE report."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload video")

            language = gr.Radio(
                choices=[("Persian (fa)", "fa"), ("English (en)", "en")],
                value="fa",
                label="Report language",
            )

            mode = gr.Radio(
                choices=[("Summary", "summary"), ("Detailed", "detailed"), ("Raw JSON", "json")],
                value="summary",
                label="Ready-made report style",
            )

            use_custom_prompt = gr.Checkbox(
                label="Use my own prompt instead of the ready-made styles",
                value=False,
            )

            custom_prompt = gr.Textbox(
                label="Custom prompt",
                placeholder=(
                    "e.g. Summarize today's violations for a night-shift "
                    "supervisor in 3 bullet points, in English."
                ),
                lines=6,
                visible=False,
            )

            max_new_tokens = gr.Slider(
                minimum=MIN_MAX_NEW_TOKENS,
                maximum=MAX_MAX_NEW_TOKENS,
                value=DEFAULT_MAX_NEW_TOKENS,
                step=1,
                label="Max new tokens (LLM response length)",
                info="Passed to LLMModel.generate(max_new_tokens=...). Higher = longer report, slower generation.",
            )

            analyze_btn = gr.Button("Analyze video", variant="primary")

            gr.Markdown("---")
            health_btn = gr.Button("Check backend status", size="sm")
            health_output = gr.Markdown()

        with gr.Column():
            report_output = gr.Textbox(label="Generated report", lines=18)
            alerts_output = gr.JSON(label="Alerts report")
            tracking_output = gr.JSON(label="Tracking summary")

    use_custom_prompt.change(
        fn=toggle_custom_prompt,
        inputs=use_custom_prompt,
        outputs=[custom_prompt, mode],
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[video_input, language, mode, use_custom_prompt, custom_prompt, max_new_tokens],
        outputs=[report_output, alerts_output, tracking_output],
    )

    health_btn.click(fn=check_backend_health, outputs=health_output)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
