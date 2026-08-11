FROM runpod/worker-comfyui:5.8.6-base


# ==========================================================
# SYSTEM DEPENDENCIES
#
# ffmpeg is required because Wan2.2 produces WEBM in our
# workflow and handler.py converts it to H.264 MP4 for R2.
# ==========================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# ==========================================================
# PYTHON DEPENDENCIES
# ==========================================================

RUN uv pip install "boto3>=1.34,<2"


# ==========================================================
# NETWORK VOLUME MODEL PATHS
#
# Serverless network volume:
# /runpod-volume/models
# ==========================================================

COPY extra_model_paths.yaml \
    /comfyui/extra_model_paths.yaml


# ==========================================================
# WORKFLOWS
# ==========================================================

RUN mkdir -p /app/workflows


# Old Wan2.1 workflow - keep for rollback.
COPY workflows/wan_video_api.json \
    /app/workflows/wan_video_api.json


# New Wan2.2 TI2V-5B workflow.
COPY workflows/wan22_ti2v_5b_api.json \
    /app/workflows/wan22_ti2v_5b_api.json


# ==========================================================
# CUSTOM RUNPOD HANDLER
# ==========================================================

COPY handler.py \
    /handler.py