import copy
import json
import os
import random
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import requests
import runpod


# ==========================================================
# CONFIGURATION
# ==========================================================

COMFY_URL = "http://127.0.0.1:8188"

WORKFLOW_FILE = Path(
    "/app/workflows/wan_video_api.json"
)


DEFAULT_WIDTH = 384
DEFAULT_HEIGHT = 224
DEFAULT_FRAMES = 17
DEFAULT_FPS = 8.0


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, "
    "deformed, artifacts, jitter, "
    "flickering, bad motion, "
    "inconsistent motion"
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
}


# ==========================================================
# ENVIRONMENT VARIABLES
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


required_variables = {
    "R2_ENDPOINT_URL":
        R2_ENDPOINT_URL,

    "R2_ACCESS_KEY_ID":
        R2_ACCESS_KEY_ID,

    "R2_SECRET_ACCESS_KEY":
        R2_SECRET_ACCESS_KEY,

    "R2_BUCKET":
        R2_BUCKET,
}


missing_variables = [
    name
    for name, value
    in required_variables.items()
    if not value
]


if missing_variables:

    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(
            missing_variables
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
# WAIT FOR COMFYUI
# ==========================================================

def wait_for_comfyui(
    timeout_seconds=600
):

    print(
        "Waiting for ComfyUI..."
    )


    started = time.time()


    while True:

        try:

            response = requests.get(
                f"{COMFY_URL}/system_stats",
                timeout=5
            )


            if response.ok:

                print(
                    "ComfyUI is ready."
                )

                return


        except requests.RequestException:

            pass


        if (
            time.time()
            - started
            > timeout_seconds
        ):

            raise TimeoutError(
                "ComfyUI startup timed out."
            )


        time.sleep(2)


# ==========================================================
# LOAD WORKFLOW
# ==========================================================

def load_workflow():

    if not WORKFLOW_FILE.exists():

        raise FileNotFoundError(
            f"Workflow missing: "
            f"{WORKFLOW_FILE}"
        )


    with open(
        WORKFLOW_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


WORKFLOW_TEMPLATE = (
    load_workflow()
)


# ==========================================================
# FIND NODE
# ==========================================================

def find_first_node(
    workflow,
    class_type
):

    for node_id, node in (
        workflow.items()
    ):

        if (
            node.get(
                "class_type"
            )
            ==
            class_type
        ):

            return (
                str(node_id),
                node
            )


    return None


# ==========================================================
# BUILD WAN WORKFLOW
# ==========================================================

def prepare_workflow(
    prompt,
    negative_prompt,
    width,
    height,
    frames,
    fps,
    seed,
    filename_prefix,
):

    workflow = copy.deepcopy(
        WORKFLOW_TEMPLATE
    )


    # ------------------------------------------------------
    # KSampler
    # ------------------------------------------------------

    sampler_result = (
        find_first_node(
            workflow,
            "KSampler"
        )
    )


    if sampler_result is None:

        raise RuntimeError(
            "KSampler was not found "
            "in workflow."
        )


    _, sampler = sampler_result


    sampler_inputs = (
        sampler["inputs"]
    )


    # ------------------------------------------------------
    # POSITIVE PROMPT
    # ------------------------------------------------------

    positive_connection = (
        sampler_inputs.get(
            "positive"
        )
    )


    if (
        not isinstance(
            positive_connection,
            list
        )
        or
        not positive_connection
    ):

        raise RuntimeError(
            "Positive prompt connection "
            "was not found."
        )


    positive_id = str(
        positive_connection[0]
    )


    positive_node = (
        workflow.get(
            positive_id
        )
    )


    if not positive_node:

        raise RuntimeError(
            "Positive prompt node "
            "was not found."
        )


    positive_node[
        "inputs"
    ][
        "text"
    ] = prompt


    # ------------------------------------------------------
    # NEGATIVE PROMPT
    # ------------------------------------------------------

    negative_connection = (
        sampler_inputs.get(
            "negative"
        )
    )


    if (
        isinstance(
            negative_connection,
            list
        )
        and
        negative_connection
    ):

        negative_id = str(
            negative_connection[0]
        )


        negative_node = (
            workflow.get(
                negative_id
            )
        )


        if (
            negative_node
            and
            "text"
            in negative_node.get(
                "inputs",
                {}
            )
        ):

            negative_node[
                "inputs"
            ][
                "text"
            ] = (
                negative_prompt
            )


    # ------------------------------------------------------
    # SEED
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


    # ------------------------------------------------------
    # VIDEO LATENT
    # ------------------------------------------------------

    latent_result = (
        find_first_node(
            workflow,
            "EmptyHunyuanLatentVideo"
        )
    )


    if latent_result is None:

        raise RuntimeError(
            "EmptyHunyuanLatentVideo "
            "was not found."
        )


    _, latent = latent_result


    latent_inputs = (
        latent["inputs"]
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


    if (
        "batch_size"
        in latent_inputs
    ):

        latent_inputs[
            "batch_size"
        ] = 1


    # ------------------------------------------------------
    # FPS
    # ------------------------------------------------------

    create_video_result = (
        find_first_node(
            workflow,
            "CreateVideo"
        )
    )


    if (
        create_video_result
        is not None
    ):

        _, create_video = (
            create_video_result
        )


        create_inputs = (
            create_video[
                "inputs"
            ]
        )


        if (
            "fps"
            in create_inputs
        ):

            create_inputs[
                "fps"
            ] = fps


    # ------------------------------------------------------
    # SAVE VIDEO
    # ------------------------------------------------------

    save_video_result = (
        find_first_node(
            workflow,
            "SaveVideo"
        )
    )


    if (
        save_video_result
        is None
    ):

        raise RuntimeError(
            "SaveVideo node was not found."
        )


    _, save_video = (
        save_video_result
    )


    save_inputs = (
        save_video[
            "inputs"
        ]
    )


    save_inputs[
        "filename_prefix"
    ] = filename_prefix


    if (
        "format"
        in save_inputs
    ):

        save_inputs[
            "format"
        ] = "mp4"


    if (
        "codec"
        in save_inputs
    ):

        save_inputs[
            "codec"
        ] = "h264"


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


    prompt_id = (
        result.get(
            "prompt_id"
        )
    )


    if not prompt_id:

        raise RuntimeError(
            "ComfyUI did not return "
            "a prompt_id."
        )


    return prompt_id


# ==========================================================
# WAIT FOR GENERATION
# ==========================================================

def wait_for_completion(
    prompt_id,
    timeout_seconds=1800
):

    started = time.time()


    while True:

        response = requests.get(
            (
                f"{COMFY_URL}"
                f"/history/"
                f"{prompt_id}"
            ),

            timeout=30,
        )


        response.raise_for_status()


        history = response.json()


        if (
            prompt_id
            in history
        ):

            result = (
                history[
                    prompt_id
                ]
            )


            status = (
                result.get(
                    "status",
                    {}
                )
            )


            if (
                status.get(
                    "completed"
                )
            ):

                return result


            if (
                result.get(
                    "outputs"
                )
            ):

                return result


        if (
            time.time()
            - started
            >
            timeout_seconds
        ):

            raise TimeoutError(
                "Video generation timed out."
            )


        time.sleep(2)


# ==========================================================
# FIND VIDEO FILES
# ==========================================================

def collect_video_files(
    value
):

    videos = []


    if isinstance(
        value,
        dict
    ):

        filename = (
            value.get(
                "filename"
            )
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

                videos.append({
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

            videos.extend(
                collect_video_files(
                    child
                )
            )


    elif isinstance(
        value,
        list
    ):

        for child in value:

            videos.extend(
                collect_video_files(
                    child
                )
            )


    return videos


# ==========================================================
# DOWNLOAD VIDEO FROM COMFYUI
# ==========================================================

def download_video_to_temp(
    descriptor
):

    filename = (
        descriptor[
            "filename"
        ]
    )


    extension = (
        Path(
            filename
        ).suffix
        or ".mp4"
    )


    response = requests.get(
        f"{COMFY_URL}/view",

        params={
            "filename":
                filename,

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


    temp_file = (
        tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        )
    )


    temp_path = Path(
        temp_file.name
    )


    try:

        for chunk in (
            response.iter_content(
                chunk_size=
                    1024 * 1024
            )
        ):

            if chunk:

                temp_file.write(
                    chunk
                )


    finally:

        temp_file.close()


    return temp_path


# ==========================================================
# UPLOAD VIDEO TO CLOUDFLARE R2
# ==========================================================

def upload_video_to_r2(
    local_file
):

    extension = (
        local_file.suffix.lower()
        or ".mp4"
    )


    key = (
        "generated-videos/"
        + str(
            uuid.uuid4()
        )
        + extension
    )


    if (
        extension == ".mp4"
    ):

        content_type = (
            "video/mp4"
        )

    elif (
        extension == ".webm"
    ):

        content_type = (
            "video/webm"
        )

    else:

        content_type = (
            "application/octet-stream"
        )


    s3.upload_file(
        str(
            local_file
        ),

        R2_BUCKET,

        key,

        ExtraArgs={
            "ContentType":
                content_type
        },
    )


    # 24-hour temporary GET URL

    video_url = (
        s3.generate_presigned_url(
            "get_object",

            Params={
                "Bucket":
                    R2_BUCKET,

                "Key":
                    key,
            },

            ExpiresIn=
                86400,
        )
    )


    return (
        key,
        video_url
    )


# ==========================================================
# RUNPOD HANDLER
# ==========================================================

def handler(
    job
):

    started = time.time()


    job_input = (
        job.get(
            "input",
            {}
        )
    )


    if not isinstance(
        job_input,
        dict
    ):

        raise ValueError(
            "input must be an object"
        )


    # ------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------

    prompt = str(
        job_input.get(
            "prompt",
            ""
        )
    ).strip()


    if not prompt:

        raise ValueError(
            "prompt is required"
        )


    # ------------------------------------------------------
    # SETTINGS
    # ------------------------------------------------------

    negative_prompt = str(
        job_input.get(
            "negative_prompt",
            DEFAULT_NEGATIVE_PROMPT
        )
    )


    width = int(
        job_input.get(
            "width",
            DEFAULT_WIDTH
        )
    )


    height = int(
        job_input.get(
            "height",
            DEFAULT_HEIGHT
        )
    )


    frames = int(
        job_input.get(
            "frames",
            DEFAULT_FRAMES
        )
    )


    fps = float(
        job_input.get(
            "fps",
            DEFAULT_FPS
        )
    )


    seed = int(
        job_input.get(
            "seed",
            random.randint(
                0,
                2**31 - 1
            )
        )
    )


    # ------------------------------------------------------
    # SAFETY LIMITS
    # ------------------------------------------------------

    if not (
        128 <= width <= 1280
    ):

        raise ValueError(
            "width must be between "
            "128 and 1280"
        )


    if not (
        128 <= height <= 1280
    ):

        raise ValueError(
            "height must be between "
            "128 and 1280"
        )


    if not (
        5 <= frames <= 161
    ):

        raise ValueError(
            "frames must be between "
            "5 and 161"
        )


    if not (
        1 <= fps <= 30
    ):

        raise ValueError(
            "fps must be between "
            "1 and 30"
        )


    # ------------------------------------------------------
    # UNIQUE OUTPUT PREFIX
    # ------------------------------------------------------

    job_id = str(
        job.get(
            "id",
            uuid.uuid4()
        )
    )


    filename_prefix = (
        "video/"
        + job_id
    )


    # ------------------------------------------------------
    # PREPARE WORKFLOW
    # ------------------------------------------------------

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

        filename_prefix=
            filename_prefix,
    )


    # ------------------------------------------------------
    # SUBMIT TO COMFYUI
    # ------------------------------------------------------

    comfy_prompt_id = (
        queue_workflow(
            workflow
        )
    )


    print(
        "ComfyUI prompt ID:",
        comfy_prompt_id
    )


    # ------------------------------------------------------
    # WAIT
    # ------------------------------------------------------

    result = (
        wait_for_completion(
            comfy_prompt_id
        )
    )


    # ------------------------------------------------------
    # LOCATE MP4
    # ------------------------------------------------------

    videos = (
        collect_video_files(
            result.get(
                "outputs",
                {}
            )
        )
    )


    if not videos:

        raise RuntimeError(
            "Generation completed but "
            "no video output was found."
        )


    descriptor = videos[0]


    print(
        "Generated video:",
        descriptor
    )


    # ------------------------------------------------------
    # DOWNLOAD FROM COMFYUI
    # ------------------------------------------------------

    temp_file = (
        download_video_to_temp(
            descriptor
        )
    )


    try:

        # --------------------------------------------------
        # R2
        # --------------------------------------------------

        (
            storage_key,
            video_url
        ) = upload_video_to_r2(
            temp_file
        )


    finally:

        try:

            temp_file.unlink(
                missing_ok=True
            )

        except Exception:

            pass


    elapsed = (
        time.time()
        - started
    )


    # ------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------

    return {

        "type":
            "video",

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

        "execution_seconds":
            round(
                elapsed,
                2
            ),

        "comfy_prompt_id":
            comfy_prompt_id,
    }


# ==========================================================
# START
# ==========================================================

wait_for_comfyui()


print(
    "Starting Wan RunPod "
    "Serverless handler..."
)


runpod.serverless.start({
    "handler":
        handler
})