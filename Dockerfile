FROM runpod/worker-comfyui:5.8.6-base


# ==========================================================
# VIDEO WORKER DEPENDENCIES
# ==========================================================

RUN uv pip install "boto3>=1.34,<2"


# ==========================================================
# WORKFLOW
# ==========================================================

RUN mkdir -p /app/workflows

COPY workflows/wan_video_api.json \
    /app/workflows/wan_video_api.json


# ==========================================================
# CUSTOM RUNPOD HANDLER
#
# The base worker's /start.sh starts ComfyUI and /handler.py.
# We replace the standard image handler with our video handler.
# ==========================================================

COPY handler.py \
    /handler.py