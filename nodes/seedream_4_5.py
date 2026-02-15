"""
Jimbo's Seedream 4.5 - ComfyUI custom node for the bytedance/seedream-4.5
model on Replicate.

Sends a text prompt (and optional reference images) to the Replicate API,
polls for the result, downloads the generated image(s), and returns them
as a ComfyUI IMAGE tensor.
"""

import base64
import io
import json
import os
import time
import urllib.request
import urllib.error

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPLICATE_MODEL_URL = (
    "https://api.replicate.com/v1/models/bytedance/seedream-4.5/predictions"
)

_POLL_INTERVAL_SECONDS = 2.0
_POLL_TIMEOUT_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "REPLICATE_API_TOKEN environment variable is not set. "
            "Please set it to your Replicate API token before using this node."
        )
    return token


def _comfy_image_to_data_uri(image_tensor: torch.Tensor, index: int = 0) -> str:
    img_np = image_tensor[index].cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np, mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _download_image_as_tensor(url: str) -> torch.Tensor:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = response.read()

    pil_image = Image.open(io.BytesIO(data))
    pil_image = pil_image.convert("RGB")

    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)[None,]


def _replicate_request(url: str, token: str, payload: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    else:
        req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"Replicate API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to connect to Replicate API: {exc.reason}"
        ) from exc


def _poll_prediction(prediction_url: str, token: str) -> dict:
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > _POLL_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"Replicate prediction timed out after {_POLL_TIMEOUT_SECONDS:.0f} "
                f"seconds. You can retry the node execution."
            )

        result = _replicate_request(prediction_url, token)
        status = result.get("status", "unknown")

        if status == "succeeded":
            return result
        elif status in ("failed", "canceled"):
            error_msg = result.get("error", "No error message provided.")
            raise RuntimeError(
                f"Replicate prediction {status}: {error_msg}"
            )
        time.sleep(_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class Seedream4_5:
    """ComfyUI node that generates images via the bytedance/seedream-4.5
    model on Replicate.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Jimbo Comfy Nodes/replicate_api"
    DESCRIPTION = (
        "Generate images using ByteDance's Seedream 4.5 model via the "
        "Replicate API. Supports text-to-image and multi-reference "
        "image-to-image generation with up to 14 input images."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "A text description of the image you want to generate",
                }),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "size": (["2K", "4K", "custom"], {
                    "default": "2K",
                }),
                "aspect_ratio": (
                    [
                        "match_input_image",
                        "1:1",
                        "4:3",
                        "3:4",
                        "16:9",
                        "9:16",
                        "3:2",
                        "2:3",
                        "21:9",
                    ],
                    {"default": "match_input_image"},
                ),
                "width": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                    "tooltip": "Only used when size is 'custom'",
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                    "tooltip": "Only used when size is 'custom'",
                }),
                "sequential_image_generation": (["disabled", "auto"], {
                    "default": "disabled",
                }),
                "max_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Max images when sequential_image_generation is 'auto'",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate(
        self,
        prompt: str,
        image_input: torch.Tensor | None = None,
        size: str = "2K",
        aspect_ratio: str = "match_input_image",
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
    ) -> tuple[torch.Tensor]:
        token = _get_api_token()

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        # Build the API payload
        api_input: dict = {
            "prompt": prompt,
            "size": size,
            "aspect_ratio": aspect_ratio,
            "sequential_image_generation": sequential_image_generation,
            "max_images": max_images,
        }

        if size == "custom":
            api_input["width"] = width
            api_input["height"] = height

        # Convert optional ComfyUI IMAGE tensor(s) to data URIs
        if image_input is not None:
            batch_size = image_input.shape[0]
            max_ref = min(batch_size, 14)
            data_uris = []
            for i in range(max_ref):
                data_uris.append(_comfy_image_to_data_uri(image_input, i))
            api_input["image_input"] = data_uris
        else:
            api_input["image_input"] = []

        payload = {"input": api_input}

        # Create prediction
        prediction = _replicate_request(_REPLICATE_MODEL_URL, token, payload)

        # Poll until complete
        status = prediction.get("status", "unknown")

        if status == "succeeded":
            final = prediction
        elif status in ("failed", "canceled"):
            error_msg = prediction.get("error", "No error message provided.")
            raise RuntimeError(
                f"Replicate prediction {status}: {error_msg}"
            )
        else:
            poll_url = prediction.get("urls", {}).get("get")
            if not poll_url:
                raise RuntimeError(
                    "Replicate API response did not include a polling URL. "
                    f"Full response: {json.dumps(prediction, indent=2)}"
                )
            final = _poll_prediction(poll_url, token)

        # Extract output and download image(s)
        output = final.get("output")

        if output is None:
            raise RuntimeError(
                "Replicate prediction succeeded but returned no output. "
                f"Full response: {json.dumps(final, indent=2)}"
            )

        # Output can be a single URL or a list of URLs
        if isinstance(output, str):
            urls = [output]
        elif isinstance(output, list):
            urls = [u for u in output if isinstance(u, str)]
        else:
            raise RuntimeError(
                f"Unexpected output format from Replicate: {type(output).__name__}. "
                f"Value: {output}"
            )

        if not urls:
            raise RuntimeError(
                "Replicate prediction succeeded but output list is empty."
            )

        # Download all images and batch them into a single tensor
        tensors = []
        for url in urls:
            tensors.append(_download_image_as_tensor(url))

        image_tensor = torch.cat(tensors, dim=0)

        return (image_tensor,)
