import copy
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import requests
import runpod


# ==========================================================
# BASIC CONFIGURATION
# ==========================================================

COMFY_URL = os.environ.get(
    "COMFY_URL",
    "http://127.0.0.1:8188",
)


WORKFLOW_FILE = Path(
    "/app/workflows/wan22_ti2v_5b_api.json"
)


COMFY_OUTPUT_DIR = Path(
    "/comfyui/output"
)


# ==========================================================
# WAN2.2 MODEL FILES
# ==========================================================

WAN_MODEL_NAME = (
    "wan2.2_ti2v_5B_fp16.safetensors"
)


WAN_TEXT_ENCODER_NAME = (
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)


WAN_VAE_NAME = (
    "wan2.2_vae.safetensors"
)


# ==========================================================
# NETWORK VOLUME PATHS
# ==========================================================

WAN_MODEL_PATH = Path(
    "/runpod-volume/models/"
    "diffusion_models/"
    "wan2.2_ti2v_5B_fp16.safetensors"
)


WAN_TEXT_ENCODER_PATH = Path(
    "/runpod-volume/models/"
    "text_encoders/"
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)


WAN_VAE_PATH = Path(
    "/runpod-volume/models/"
    "vae/"
    "wan2.2_vae.safetensors"
)


# ==========================================================
# DEFAULT VIDEO SETTINGS
# ==========================================================

#
# 41 frames is deliberately used for the initial smoke test.
#
# After everything works:
#
# 1280 x 704
# 121 frames
# 24 fps
#
# is the higher-quality reference configuration.
#

DEFAULT_WIDTH = 1280

DEFAULT_HEIGHT = 704

DEFAULT_FRAMES = 41

DEFAULT_FPS = 24.0


# ==========================================================
# DEFAULT SAMPLING SETTINGS
# ==========================================================

DEFAULT_STEPS = 30

DEFAULT_CFG = 5.0

DEFAULT_SAMPLER_NAME = "uni_pc"

DEFAULT_SCHEDULER = "simple"

DEFAULT_DENOISE = 1.0

DEFAULT_MODEL_SHIFT = 8.0


# ==========================================================
# WEBM SETTINGS
# ==========================================================

DEFAULT_WEBM_CODEC = "vp9"

DEFAULT_WEBM_CRF = 16.1


# ==========================================================
# MP4 SETTINGS
# ==========================================================

DEFAULT_MP4_CRF = 18

DEFAULT_MP4_PRESET = "medium"


# ==========================================================
# DEFAULT NEGATIVE PROMPT
# ==========================================================

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, CGI, "
    "video game graphics, obvious 3D render, synthetic appearance, "
    "plastic materials, fake reflections, unrealistic physics, "
    "warped geometry, deformed objects, duplicated objects, "
    "morphing objects, changing object shape, flickering, jitter, "
    "ghosting, frame blending artifacts, temporal inconsistency, "
    "unstable background, unstable camera, camera shake, "
    "sudden zoom, abrupt camera movement, scene transitions, "
    "cuts, excessive motion blur, blurry subject, low detail, "
    "low quality, low resolution, compression artifacts, "
    "text, subtitles, watermark, logo"
)


# ==========================================================
# SUPPORTED COMFY VIDEO FILE TYPES
# ==========================================================

VIDEO_EXTENSIONS = {
    ".webm",
    ".mp4",
    ".mkv",
    ".mov",
}


# ==========================================================
# R2 CONFIGURATION
# ==========================================================

R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL"
)


R2_ACCESS_KEY_ID = os.environ.get(
    "R2_ACCESS_KEY_ID"
)


R2_SECRET_ACCESS_KEY = os.environ.get(
    "R2_SECRET_ACCESS_KEY"
)


R2_BUCKET = os.environ.get(
    "R2_BUCKET"
)


# ==========================================================
# VALIDATE R2 CONFIG
# ==========================================================

required_r2_values = {

    "R2_ENDPOINT_URL":
        R2_ENDPOINT_URL,

    "R2_ACCESS_KEY_ID":
        R2_ACCESS_KEY_ID,

    "R2_SECRET_ACCESS_KEY":
        R2_SECRET_ACCESS_KEY,

    "R2_BUCKET":
        R2_BUCKET,

}


missing_r2_values = [

    name

    for name, value
    in required_r2_values.items()

    if not value

]


if missing_r2_values:

    raise RuntimeError(
        "Missing required R2 environment variables: "
        + ", ".join(
            missing_r2_values
        )
    )


# ==========================================================
# R2 CLIENT
# ==========================================================

s3 = boto3.client(

    "s3",

    endpoint_url=
        R2_ENDPOINT_URL,

    aws_access_key_id=
        R2_ACCESS_KEY_ID,

    aws_secret_access_key=
        R2_SECRET_ACCESS_KEY,

    region_name=
        "auto",

)


# ==========================================================
# RUNTIME VALIDATION
# ==========================================================

def verify_runtime():

    print(
        "Checking Wan2.2 runtime...",
        flush=True,
    )


    # ------------------------------------------------------
    # Workflow
    # ------------------------------------------------------

    if not WORKFLOW_FILE.exists():

        raise RuntimeError(
            f"Wan2.2 workflow not found: "
            f"{WORKFLOW_FILE}"
        )


    print(
        f"Workflow found: {WORKFLOW_FILE}",
        flush=True,
    )


    # ------------------------------------------------------
    # Model files
    # ------------------------------------------------------

    required_models = {

        "Wan2.2 diffusion model":
            WAN_MODEL_PATH,

        "UMT5 text encoder":
            WAN_TEXT_ENCODER_PATH,

        "Wan2.2 VAE":
            WAN_VAE_PATH,

    }


    for name, path in (
        required_models.items()
    ):

        if not path.exists():

            raise RuntimeError(
                f"{name} not found at: "
                f"{path}"
            )


        size_gb = (
            path.stat().st_size
            / (
                1024 ** 3
            )
        )


        print(
            f"{name}: OK "
            f"({size_gb:.2f} GB) "
            f"{path}",
            flush=True,
        )


    # ------------------------------------------------------
    # FFmpeg
    # ------------------------------------------------------

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )


    if not ffmpeg_path:

        raise RuntimeError(
            "FFmpeg is not installed in the worker. "
            "The Dockerfile must install ffmpeg "
            "before this handler can convert WEBM to MP4."
        )


    print(
        f"FFmpeg found: {ffmpeg_path}",
        flush=True,
    )


    print(
        "Wan2.2 runtime validation passed.",
        flush=True,
    )


# ==========================================================
# WAIT FOR COMFYUI
# ==========================================================

def wait_for_comfyui(
    timeout_seconds=600
):

    print(
        "Waiting for ComfyUI...",
        flush=True,
    )


    deadline = (
        time.time()
        + timeout_seconds
    )


    while (
        time.time()
        < deadline
    ):

        try:

            response = requests.get(

                f"{COMFY_URL}/system_stats",

                timeout=5,

            )


            if response.ok:

                print(
                    "ComfyUI is ready.",
                    flush=True,
                )

                return


        except requests.RequestException:

            pass


        time.sleep(
            2
        )


    raise RuntimeError(
        "Timed out waiting for ComfyUI."
    )


# ==========================================================
# LOAD WORKFLOW
# ==========================================================

def load_workflow():

    with WORKFLOW_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        workflow = json.load(
            file
        )


    if not isinstance(
        workflow,
        dict
    ):

        raise RuntimeError(
            "Wan2.2 workflow JSON must be "
            "a ComfyUI API workflow object."
        )


    return workflow


# Load once at worker startup.

WORKFLOW_TEMPLATE = (
    load_workflow()
)


# ==========================================================
# FIND NODE BY CLASS
# ==========================================================

def find_first_node(
    workflow,
    class_type,
):

    for node_id, node in (
        workflow.items()
    ):

        if not isinstance(
            node,
            dict
        ):

            continue


        if (
            node.get(
                "class_type"
            )
            == class_type
        ):

            return (
                str(node_id),
                node,
            )


    return None


# ==========================================================
# FIND CONNECTED NODE
# ==========================================================

def connected_node(
    workflow,
    source_node,
    input_name,
):

    source_inputs = (
        source_node.get(
            "inputs",
            {}
        )
    )


    connection = (
        source_inputs.get(
            input_name
        )
    )


    if (
        not isinstance(
            connection,
            list
        )
        or
        len(connection) == 0
    ):

        return None


    connected_id = str(
        connection[0]
    )


    connected = workflow.get(
        connected_id
    )


    if not isinstance(
        connected,
        dict
    ):

        return None


    return (
        connected_id,
        connected,
    )


# ==========================================================
# PREPARE WAN2.2 WORKFLOW
# ==========================================================

def prepare_workflow(
    *,
    prompt,
    negative_prompt,
    width,
    height,
    frames,
    fps,
    seed,
    steps,
    cfg,
    webm_crf,
    filename_prefix,
):

    workflow = copy.deepcopy(
        WORKFLOW_TEMPLATE
    )


    # ======================================================
    # UNET / DIFFUSION MODEL
    # ======================================================

    unet_result = find_first_node(
        workflow,
        "UNETLoader",
    )


    if unet_result is None:

        raise RuntimeError(
            "UNETLoader was not found in "
            "wan22_ti2v_5b_api.json."
        )


    _, unet_node = (
        unet_result
    )


    unet_inputs = (
        unet_node.setdefault(
            "inputs",
            {}
        )
    )


    unet_inputs[
        "unet_name"
    ] = WAN_MODEL_NAME


    # ======================================================
    # TEXT ENCODER
    # ======================================================

    clip_result = find_first_node(
        workflow,
        "CLIPLoader",
    )


    if clip_result is None:

        raise RuntimeError(
            "CLIPLoader was not found in "
            "wan22_ti2v_5b_api.json."
        )


    _, clip_node = (
        clip_result
    )


    clip_inputs = (
        clip_node.setdefault(
            "inputs",
            {}
        )
    )


    clip_inputs[
        "clip_name"
    ] = WAN_TEXT_ENCODER_NAME


    #
    # Preserve these if the workflow already has them.
    #

    if (
        "type"
        in clip_inputs
    ):

        clip_inputs[
            "type"
        ] = "wan"


    # ======================================================
    # VAE
    # ======================================================

    vae_result = find_first_node(
        workflow,
        "VAELoader",
    )


    if vae_result is None:

        raise RuntimeError(
            "VAELoader was not found in "
            "wan22_ti2v_5b_api.json."
        )


    _, vae_node = (
        vae_result
    )


    vae_inputs = (
        vae_node.setdefault(
            "inputs",
            {}
        )
    )


    vae_inputs[
        "vae_name"
    ] = WAN_VAE_NAME


    # ======================================================
    # MODEL SAMPLING SD3
    # ======================================================

    sampling_result = find_first_node(
        workflow,
        "ModelSamplingSD3",
    )


    if sampling_result is None:

        raise RuntimeError(
            "ModelSamplingSD3 was not found."
        )


    _, sampling_node = (
        sampling_result
    )


    sampling_inputs = (
        sampling_node.setdefault(
            "inputs",
            {}
        )
    )


    sampling_inputs[
        "shift"
    ] = DEFAULT_MODEL_SHIFT


    # ======================================================
    # KSAMPLER
    # ======================================================

    sampler_result = find_first_node(
        workflow,
        "KSampler",
    )


    if sampler_result is None:

        raise RuntimeError(
            "KSampler was not found."
        )


    _, sampler = (
        sampler_result
    )


    sampler_inputs = (
        sampler.setdefault(
            "inputs",
            {}
        )
    )


    sampler_inputs[
        "steps"
    ] = steps


    sampler_inputs[
        "cfg"
    ] = cfg


    sampler_inputs[
        "sampler_name"
    ] = DEFAULT_SAMPLER_NAME


    sampler_inputs[
        "scheduler"
    ] = DEFAULT_SCHEDULER


    sampler_inputs[
        "denoise"
    ] = DEFAULT_DENOISE


    # ------------------------------------------------------
    # Seed
    # ------------------------------------------------------

    if (
        "seed"
        in sampler_inputs
    ):

        sampler_inputs[
            "seed"
        ] = seed


    elif (
        "noise_seed"
        in sampler_inputs
    ):

        sampler_inputs[
            "noise_seed"
        ] = seed


    else:

        sampler_inputs[
            "seed"
        ] = seed


    # ======================================================
    # POSITIVE PROMPT
    # ======================================================

    positive_result = connected_node(

        workflow,

        sampler,

        "positive",

    )


    if positive_result is None:

        raise RuntimeError(
            "Positive prompt node connected "
            "to KSampler was not found."
        )


    _, positive_node = (
        positive_result
    )


    positive_inputs = (
        positive_node.setdefault(
            "inputs",
            {}
        )
    )


    if (
        "text"
        not in positive_inputs
    ):

        raise RuntimeError(
            "Positive CLIPTextEncode node "
            "has no text input."
        )


    positive_inputs[
        "text"
    ] = prompt


    # ======================================================
    # NEGATIVE PROMPT
    # ======================================================

    negative_result = connected_node(

        workflow,

        sampler,

        "negative",

    )


    if negative_result is None:

        raise RuntimeError(
            "Negative prompt node connected "
            "to KSampler was not found."
        )


    _, negative_node = (
        negative_result
    )


    negative_inputs = (
        negative_node.setdefault(
            "inputs",
            {}
        )
    )


    if (
        "text"
        in negative_inputs
    ):

        negative_inputs[
            "text"
        ] = negative_prompt


    # ======================================================
    # WAN2.2 LATENT
    # ======================================================

    latent_result = find_first_node(
        workflow,
        "Wan22ImageToVideoLatent",
    )


    if latent_result is None:

        raise RuntimeError(
            "Wan22ImageToVideoLatent was not found. "
            "Check that the workflow is the Wan2.2 "
            "TI2V-5B workflow and that the ComfyUI "
            "version supports Wan2.2."
        )


    _, latent_node = (
        latent_result
    )


    latent_inputs = (
        latent_node.setdefault(
            "inputs",
            {}
        )
    )


    latent_inputs[
        "width"
    ] = width


    latent_inputs[
        "height"
    ] = height


    latent_inputs[
        "length"
    ] = frames


    latent_inputs[
        "batch_size"
    ] = 1


    # ------------------------------------------------------
    # PURE TEXT-TO-VIDEO
    #
    # Wan2.2 TI2V can also accept a start image.
    # For the current /api/video endpoint we deliberately
    # remove the image inputs so this remains T2V.
    # ------------------------------------------------------

    latent_inputs.pop(
        "start_image",
        None,
    )


    latent_inputs.pop(
        "end_image",
        None,
    )


    # ======================================================
    # SAVE WEBM
    # ======================================================

    save_result = find_first_node(
        workflow,
        "SaveWEBM",
    )


    if save_result is None:

        raise RuntimeError(
            "SaveWEBM was not found in "
            "wan22_ti2v_5b_api.json."
        )


    _, save_node = (
        save_result
    )


    save_inputs = (
        save_node.setdefault(
            "inputs",
            {}
        )
    )


    save_inputs[
        "filename_prefix"
    ] = filename_prefix


    save_inputs[
        "codec"
    ] = DEFAULT_WEBM_CODEC


    save_inputs[
        "fps"
    ] = fps


    save_inputs[
        "crf"
    ] = webm_crf


    # ======================================================
    # DIAGNOSTIC LOG
    # ======================================================

    print(
        "WAN2.2 WORKFLOW SETTINGS:",
        {
            "model":
                WAN_MODEL_NAME,

            "vae":
                WAN_VAE_NAME,

            "text_encoder":
                WAN_TEXT_ENCODER_NAME,

            "width":
                width,

            "height":
                height,

            "frames":
                frames,

            "fps":
                fps,

            "duration_seconds":
                round(
                    frames / fps,
                    3
                ),

            "seed":
                seed,

            "steps":
                steps,

            "cfg":
                cfg,

            "model_shift":
                DEFAULT_MODEL_SHIFT,

            "sampler":
                DEFAULT_SAMPLER_NAME,

            "scheduler":
                DEFAULT_SCHEDULER,
        },
        flush=True,
    )


    return workflow


# ==========================================================
# QUEUE COMFYUI WORKFLOW
# ==========================================================

def queue_workflow(
    workflow
):

    response = requests.post(

        f"{COMFY_URL}/prompt",

        json={
            "prompt":
                workflow
        },

        timeout=60,

    )


    if not response.ok:

        raise RuntimeError(
            "ComfyUI rejected workflow:\n"
            + response.text
        )


    result = response.json()


    prompt_id = result.get(
        "prompt_id"
    )


    if not prompt_id:

        raise RuntimeError(
            "ComfyUI did not return a prompt_id."
        )


    print(
        f"ComfyUI prompt queued: {prompt_id}",
        flush=True,
    )


    return prompt_id


# ==========================================================
# WAIT FOR COMFYUI COMPLETION
# ==========================================================

def wait_for_completion(
    prompt_id,
    timeout_seconds=3600,
):

    deadline = (
        time.time()
        + timeout_seconds
    )


    while (
        time.time()
        < deadline
    ):

        try:

            response = requests.get(

                (
                    f"{COMFY_URL}"
                    f"/history/"
                    f"{prompt_id}"
                ),

                timeout=30,

            )


            if not response.ok:

                time.sleep(
                    2
                )

                continue


            history = response.json()


            result = history.get(
                prompt_id
            )


            if not isinstance(
                result,
                dict
            ):

                time.sleep(
                    2
                )

                continue


            status = result.get(
                "status",
                {}
            )


            if not isinstance(
                status,
                dict
            ):

                status = {}


            status_string = (
                status.get(
                    "status_str"
                )
            )


            completed = (
                status.get(
                    "completed"
                )
            )


            # --------------------------------------------------
            # ERROR
            # --------------------------------------------------

            if (
                status_string
                == "error"
            ):

                raise RuntimeError(
                    "ComfyUI execution failed:\n"
                    + json.dumps(
                        status,
                        indent=2,
                    )
                )


            # --------------------------------------------------
            # SUCCESS
            # --------------------------------------------------

            if (
                completed is True
                or
                status_string
                == "success"
            ):

                print(
                    "ComfyUI generation completed.",
                    flush=True,
                )

                return result


            #
            # Fallback for ComfyUI versions whose history
            # result lacks an explicit completed flag.
            #

            outputs = result.get(
                "outputs"
            )


            if (
                outputs
                and
                status_string
                not in (
                    "running",
                    "processing",
                )
            ):

                return result


        except requests.RequestException:

            pass


        time.sleep(
            2
        )


    raise RuntimeError(
        "Timed out waiting for Wan2.2 "
        "video generation."
    )


# ==========================================================
# RECURSIVELY COLLECT VIDEO DESCRIPTORS
# ==========================================================

def collect_video_files(
    value
):

    found = []


    if isinstance(
        value,
        dict
    ):

        filename = value.get(
            "filename"
        )


        if isinstance(
            filename,
            str
        ):

            extension = (
                Path(
                    filename
                )
                .suffix
                .lower()
            )


            if (
                extension
                in VIDEO_EXTENSIONS
            ):

                found.append({

                    "filename":
                        filename,

                    "subfolder":
                        value.get(
                            "subfolder",
                            ""
                        ),

                    "type":
                        value.get(
                            "type",
                            "output"
                        ),

                })


        for child in (
            value.values()
        ):

            found.extend(
                collect_video_files(
                    child
                )
            )


    elif isinstance(
        value,
        list
    ):

        for child in value:

            found.extend(
                collect_video_files(
                    child
                )
            )


    return found


# ==========================================================
# DOWNLOAD OUTPUT FROM COMFYUI
# ==========================================================

def download_comfy_video(
    descriptor
):

    source_filename = descriptor[
        "filename"
    ]


    suffix = (
        Path(
            source_filename
        ).suffix.lower()
        or ".webm"
    )


    response = requests.get(

        f"{COMFY_URL}/view",

        params={

            "filename":
                source_filename,

            "subfolder":
                descriptor.get(
                    "subfolder",
                    ""
                ),

            "type":
                descriptor.get(
                    "type",
                    "output"
                ),

        },

        stream=True,

        timeout=600,

    )


    response.raise_for_status()


    temporary_file = (
        tempfile.NamedTemporaryFile(

            suffix=
                suffix,

            delete=
                False,

        )
    )


    try:

        for chunk in (
            response.iter_content(
                chunk_size=
                    1024 * 1024
            )
        ):

            if chunk:

                temporary_file.write(
                    chunk
                )


    finally:

        temporary_file.close()


    result = Path(
        temporary_file.name
    )


    print(
        f"Downloaded ComfyUI output: {result}",
        flush=True,
    )


    return result


# ==========================================================
# FALLBACK: FIND FILE DIRECTLY
# ==========================================================

def find_video_on_disk(
    job_id
):

    if not COMFY_OUTPUT_DIR.exists():

        return None


    candidates = []


    for path in (
        COMFY_OUTPUT_DIR.rglob(
            "*"
        )
    ):

        if not path.is_file():

            continue


        if (
            path.suffix.lower()
            not in VIDEO_EXTENSIONS
        ):

            continue


        if (
            job_id
            not in str(
                path
            )
        ):

            continue


        candidates.append(
            path
        )


    if not candidates:

        return None


    newest = max(

        candidates,

        key=lambda path:
            path.stat().st_mtime,

    )


    print(
        f"Found ComfyUI output directly: {newest}",
        flush=True,
    )


    return newest


# ==========================================================
# CONVERT TO H.264 MP4
# ==========================================================

def convert_to_mp4(
    source_file
):

    if not source_file.exists():

        raise RuntimeError(
            f"Video source does not exist: "
            f"{source_file}"
        )


    temporary_output = (
        tempfile.NamedTemporaryFile(

            suffix=
                ".mp4",

            delete=
                False,

        )
    )


    temporary_output.close()


    output_path = Path(
        temporary_output.name
    )


    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(
            source_file
        ),

        # Use first video stream.
        "-map",
        "0:v:0",

        # Instagram/browser compatible H.264.
        "-c:v",
        "libx264",

        "-preset",
        DEFAULT_MP4_PRESET,

        "-crf",
        str(
            DEFAULT_MP4_CRF
        ),

        # Broad device/browser compatibility.
        "-pix_fmt",
        "yuv420p",

        # Put metadata at the start of MP4
        # for faster web playback.
        "-movflags",
        "+faststart",

        # Current Wan workflow has no audio.
        "-an",

        str(
            output_path
        ),

    ]


    print(
        "Converting WEBM to H.264 MP4...",
        flush=True,
    )


    result = subprocess.run(

        command,

        stdout=
            subprocess.PIPE,

        stderr=
            subprocess.PIPE,

        text=
            True,

        timeout=
            900,

    )


    if (
        result.returncode
        != 0
    ):

        try:

            output_path.unlink(
                missing_ok=True
            )

        except OSError:

            pass


        error_text = (
            result.stderr
            or
            "Unknown FFmpeg error"
        )


        raise RuntimeError(
            "FFmpeg MP4 conversion failed:\n"
            + error_text[
                -6000:
            ]
        )


    if (
        not output_path.exists()
        or
        output_path.stat().st_size
        == 0
    ):

        raise RuntimeError(
            "FFmpeg finished but produced "
            "an empty MP4 file."
        )


    size_mb = (
        output_path.stat().st_size
        / (
            1024 ** 2
        )
    )


    print(
        f"MP4 conversion complete: "
        f"{size_mb:.2f} MB",
        flush=True,
    )


    return output_path


# ==========================================================
# UPLOAD MP4 TO CLOUDFLARE R2
# ==========================================================

def upload_video_to_r2(
    mp4_file
):

    storage_key = (

        "generated-videos/"

        + str(
            uuid.uuid4()
        )

        + ".mp4"

    )


    print(
        f"Uploading MP4 to R2: {storage_key}",
        flush=True,
    )


    s3.upload_file(

        str(
            mp4_file
        ),

        R2_BUCKET,

        storage_key,

        ExtraArgs={

            "ContentType":
                "video/mp4",

        },

    )


    # 24-hour signed URL.
    #
    # The object itself remains in R2.

    video_url = (
        s3.generate_presigned_url(

            "get_object",

            Params={

                "Bucket":
                    R2_BUCKET,

                "Key":
                    storage_key,

            },

            ExpiresIn=
                86400,

        )
    )


    print(
        "R2 upload complete.",
        flush=True,
    )


    return (
        storage_key,
        video_url,
    )


# ==========================================================
# SAFE TEMP FILE DELETE
# ==========================================================

def remove_temp_file(
    path
):

    if (
        path is None
    ):

        return


    try:

        path = Path(
            path
        )


        if path.exists():

            path.unlink()


    except OSError:

        pass


# ==========================================================
# RUNPOD HANDLER
# ==========================================================

def handler(
    job
):

    started_at = (
        time.time()
    )


    print(
        "=" * 70,
        flush=True,
    )


    print(
        f"Starting Wan2.2 job: "
        f"{job.get('id')}",
        flush=True,
    )


    # ======================================================
    # INPUT OBJECT
    # ======================================================

    job_input = (
        job.get(
            "input"
        )
        or
        {}
    )


    if not isinstance(
        job_input,
        dict
    ):

        raise ValueError(
            "job.input must be a JSON object."
        )


    # ======================================================
    # PROMPT
    # ======================================================

    prompt = str(
        job_input.get(
            "prompt",
            ""
        )
    ).strip()


    if not prompt:

        raise ValueError(
            "prompt is required."
        )


    if (
        len(prompt)
        > 10000
    ):

        raise ValueError(
            "prompt is too long."
        )


    # ======================================================
    # NEGATIVE PROMPT
    # ======================================================

    negative_prompt = str(
        job_input.get(

            "negative_prompt",

            DEFAULT_NEGATIVE_PROMPT,

        )
    ).strip()


    # ======================================================
    # DIMENSIONS
    # ======================================================

    width = int(
        job_input.get(

            "width",

            DEFAULT_WIDTH,

        )
    )


    height = int(
        job_input.get(

            "height",

            DEFAULT_HEIGHT,

        )
    )


    # ======================================================
    # FRAMES / FPS
    # ======================================================

    frames = int(
        job_input.get(

            "frames",

            DEFAULT_FRAMES,

        )
    )


    fps = float(
        job_input.get(

            "fps",

            DEFAULT_FPS,

        )
    )


    # ======================================================
    # QUALITY SETTINGS
    # ======================================================

    steps = int(
        job_input.get(

            "steps",

            DEFAULT_STEPS,

        )
    )


    cfg = float(
        job_input.get(

            "cfg",

            DEFAULT_CFG,

        )
    )


    webm_crf = float(
        job_input.get(

            "webm_crf",

            DEFAULT_WEBM_CRF,

        )
    )


    # ======================================================
    # SEED
    # ======================================================

    seed_value = (
        job_input.get(
            "seed"
        )
    )


    if (
        seed_value
        is None
    ):

        seed = random.randint(

            0,

            2**31 - 1,

        )


    else:

        seed = int(
            seed_value
        )


    # ======================================================
    # VALIDATION
    # ======================================================

    if not (
        256
        <= width
        <= 1280
    ):

        raise ValueError(
            "width must be between "
            "256 and 1280."
        )


    if not (
        256
        <= height
        <= 1280
    ):

        raise ValueError(
            "height must be between "
            "256 and 1280."
        )


    if (
        width % 16 != 0
        or
        height % 16 != 0
    ):

        raise ValueError(
            "width and height must "
            "be divisible by 16."
        )


    if not (
        5
        <= frames
        <= 161
    ):

        raise ValueError(
            "frames must be between "
            "5 and 161."
        )


    if not (
        1
        <= fps
        <= 30
    ):

        raise ValueError(
            "fps must be between "
            "1 and 30."
        )


    if not (
        1
        <= steps
        <= 60
    ):

        raise ValueError(
            "steps must be between "
            "1 and 60."
        )


    if not (
        0
        < cfg
        <= 20
    ):

        raise ValueError(
            "cfg must be greater than "
            "0 and at most 20."
        )


    if not (
        0
        <= webm_crf
        <= 63
    ):

        raise ValueError(
            "webm_crf must be between "
            "0 and 63."
        )


    # ======================================================
    # UNIQUE JOB OUTPUT PREFIX
    # ======================================================

    runpod_job_id = str(
        job.get(
            "id"
        )
        or
        uuid.uuid4()
    )


    filename_prefix = (

        "wan22/"

        + runpod_job_id

    )


    # ======================================================
    # BUILD WORKFLOW
    # ======================================================

    workflow = prepare_workflow(

        prompt=
            prompt,

        negative_prompt=
            negative_prompt,

        width=
            width,

        height=
            height,

        frames=
            frames,

        fps=
            fps,

        seed=
            seed,

        steps=
            steps,

        cfg=
            cfg,

        webm_crf=
            webm_crf,

        filename_prefix=
            filename_prefix,

    )


    # ======================================================
    # SEND TO COMFYUI
    # ======================================================

    comfy_prompt_id = (
        queue_workflow(
            workflow
        )
    )


    # ======================================================
    # WAIT FOR GENERATION
    # ======================================================

    generation_result = (
        wait_for_completion(
            comfy_prompt_id
        )
    )


    # ======================================================
    # LOCATE GENERATED WEBM
    # ======================================================

    outputs = (
        generation_result.get(
            "outputs",
            {}
        )
    )


    video_descriptors = (
        collect_video_files(
            outputs
        )
    )


    downloaded_file = None

    source_video = None

    mp4_file = None


    try:

        # --------------------------------------------------
        # Preferred method: ComfyUI /view
        # --------------------------------------------------

        if video_descriptors:

            source_video = (
                download_comfy_video(
                    video_descriptors[0]
                )
            )


            downloaded_file = (
                source_video
            )


        # --------------------------------------------------
        # Fallback: direct file system
        # --------------------------------------------------

        else:

            source_video = (
                find_video_on_disk(
                    runpod_job_id
                )
            )


        if (
            source_video is None
            or
            not source_video.exists()
        ):

            raise RuntimeError(
                "Wan2.2 generation completed "
                "but no WEBM/video output "
                "could be located."
            )


        # ==================================================
        # CONVERT TO MP4
        # ==================================================

        mp4_file = (
            convert_to_mp4(
                source_video
            )
        )


        # ==================================================
        # UPLOAD TO R2
        # ==================================================

        (
            storage_key,
            video_url,
        ) = upload_video_to_r2(
            mp4_file
        )


    finally:

        #
        # Do not delete ComfyUI's original file when
        # find_video_on_disk() was used.
        #
        # Only delete temporary files created by handler.
        #

        remove_temp_file(
            downloaded_file
        )


        remove_temp_file(
            mp4_file
        )


    # ======================================================
    # FINAL METADATA
    # ======================================================

    execution_seconds = (
        time.time()
        - started_at
    )


    result = {

        "type":
            "video",

        "model":
            "Wan2.2-TI2V-5B",

        "video_url":
            video_url,

        "storage_key":
            storage_key,

        "seed":
            seed,

        "width":
            width,

        "height":
            height,

        "frames":
            frames,

        "fps":
            fps,

        "duration_seconds":
            round(
                frames / fps,
                3
            ),

        "steps":
            steps,

        "cfg":
            cfg,

        "execution_seconds":
            round(
                execution_seconds,
                2
            ),

        "comfy_prompt_id":
            comfy_prompt_id,

    }


    print(
        "WAN2.2 JOB COMPLETED:",
        result,
        flush=True,
    )


    print(
        "=" * 70,
        flush=True,
    )


    return result


# ==========================================================
# STARTUP
# ==========================================================

verify_runtime()


wait_for_comfyui()


print(
    "Starting Wan2.2 TI2V-5B "
    "RunPod Serverless handler...",
    flush=True,
)


runpod.serverless.start({

    "handler":
        handler

})