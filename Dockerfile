FROM runpod/worker-comfyui:5.8.7-base


# ==========================================================
# VIDEO WORKER DEPENDENCIES
# ==========================================================

COPY requirements.txt /tmp/video-requirements.txt

RUN uv pip install \
    -r /tmp/video-requirements.txt


# ==========================================================
# NETWORK-VOLUME MODEL CONFIGURATION
# ==========================================================

COPY extra_model_paths.yaml \
    /comfyui/extra_model_paths.yaml


# ==========================================================
# WORKFLOW
# ==========================================================

RUN mkdir -p /app/workflows

COPY workflows/wan_video_api.json \
    /app/workflows/wan_video_api.json


# ==========================================================
# CUSTOM RUNPOD HANDLER
# ==========================================================

COPY handler.py \
    /handler.py