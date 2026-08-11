FROM runpod/worker-comfyui:5.8.6-base


# ==========================================================
# VIDEO WORKER DEPENDENCIES
# ==========================================================

RUN uv pip install "boto3>=1.34,<2"


# ==========================================================
# NETWORK VOLUME MODEL PATHS
#
# Tell ComfyUI explicitly that our models live on the
# attached RunPod network volume.
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
# CUSTOM VIDEO HANDLER
# ==========================================================

COPY handler.py \
    /handler.py