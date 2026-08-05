# Safety Monitoring System -- Gradio UI service
#
# Talks to the API service over HTTP (API_URL env var); does not need the
# YOLO/LLM dependencies at all, so this image stays small.

FROM python:3.14-slim

WORKDIR /app

COPY UI/requirements.txt /app/UI/requirements.txt
RUN pip install --no-cache-dir -r /app/UI/requirements.txt

COPY UI/ /app/UI/

ENV API_URL=http://api:8000 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "UI/app.py"]
