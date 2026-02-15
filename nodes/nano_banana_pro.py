"""
Jimbo's Nano Banana Pro - ComfyUI custom node for the google/nano-banana-pro
model on Replicate.

Sends a text prompt (and optional reference images) to the Replicate API,
polls for the result, downloads the generated image, and returns it as a
ComfyUI IMAGE tensor.
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
    "https://api.replicate.com/v1/models/google/nano-banana-pro/predictions"
)

# Polling behaviour
_POLL_INTERVAL_SECONDS = 2.0
_POLL_TIMEOUT_SECONDS = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_token() -> str:
    """Return the Replicate API token from the environment.

    Raises:
        EnvironmentError: If the token is not set.
    """
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "REPLICATE_API_TOKEN environment variable is not set. "
            "Please set it to your Replicate API token before using this node."
        )
    return token


def _comfy_image_to_data_uri(image_tensor: torch.Tensor, index: int = 0) -> str:
    """Convert a single image from a ComfyUI IMAGE batch tensor to a
    base64-encoded PNG data URI suitable for the Replicate API.

    Args:
        image_tensor: Tensor of shape (B, H, W, C) with float values in [0, 1].
        index: Which image in the batch to convert.

    Returns:
        A ``data:image/png;base64,...`` string.
    """
    # Extract single image: (H, W, C)
    img_np = image_tensor[index].cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np, mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _download_image_as_tensor(url: str) -> torch.Tensor:
    """Download an image from a URL and return it as a ComfyUI IMAGE tensor.

    Args:
        url: The URL of the image to download.

    Returns:
        A torch.Tensor of shape (1, H, W, 3) with float32 values in [0, 1].
    """
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = response.read()

    pil_image = Image.open(io.BytesIO(data))
    pil_image = pil_image.convert("RGB")

    img_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np)[None,]  # Add batch dimension -> (1, H, W, C)
    return tensor


def _replicate_request(url: str, token: str, payload: dict | None = None) -> dict:
    """Make an authenticated request to the Replicate API.

    Args:
        url: The API endpoint URL.
        token: The Replicate API bearer token.
        payload: Optional JSON body for POST requests. If None, a GET is used.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        RuntimeError: On HTTP errors or malformed responses.
    """
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
    """Poll a Replicate prediction until it reaches a terminal state.

    Args:
        prediction_url: The URL to poll (from the prediction's ``urls.get`` field).
        token: The Replicate API bearer token.

    Returns:
        The final prediction dict.

    Raises:
        RuntimeError: If the prediction fails, is canceled, or times out.
    """
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
        # status is "starting" or "processing" -- keep waiting
        time.sleep(_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class NanoBananaPro:
    """ComfyUI node that generates images via the google/nano-banana-pro model
    on Replicate.

    Accepts a text prompt and optional reference images, sends them to the
    Replicate API, and returns the generated image as a ComfyUI IMAGE tensor.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Jimbo Comfy Nodes/replicate_api"
    DESCRIPTION = (
        "Generate images using Google's Nano Banana Pro model via the "
        "Replicate API. Provide a text prompt and optionally reference images."
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
                "resolution": (["1K", "2K", "4K"], {
                    "default": "2K",
                }),
                "aspect_ratio": (
                    [
                        "match_input_image",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    {"default": "match_input_image"},
                ),
                "output_format": (["jpg", "png"], {
                    "default": "jpg",
                }),
                "safety_filter_level": (
                    [
                        "block_low_and_above",
                        "block_medium_and_above",
                        "block_only_high",
                    ],
                    {"default": "block_only_high"},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute this node because the API may return different
        results even with the same inputs."""
        return float("NaN")

    def generate(
        self,
        prompt: str,
        image_input: torch.Tensor | None = None,
        resolution: str = "2K",
        aspect_ratio: str = "match_input_image",
        output_format: str = "jpg",
        safety_filter_level: str = "block_only_high",
    ) -> tuple[torch.Tensor]:
        """Execute the Replicate API call and return the generated image.

        Returns:
            A single-element tuple containing an IMAGE tensor of shape
            (1, H, W, 3).
        """
        # -------------------------------------------------------------------
        # 1. Validate
        # -------------------------------------------------------------------
        token = _get_api_token()

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        # -------------------------------------------------------------------
        # 2. Build the API payload
        # -------------------------------------------------------------------
        api_input: dict = {
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "safety_filter_level": safety_filter_level,
        }

        # Convert optional ComfyUI IMAGE tensor(s) to data URIs
        if image_input is not None:
            batch_size = image_input.shape[0]
            max_images = min(batch_size, 14)  # API supports up to 14 images
            data_uris = []
            for i in range(max_images):
                data_uris.append(_comfy_image_to_data_uri(image_input, i))
            api_input["image_input"] = data_uris
        else:
            api_input["image_input"] = []

        payload = {"input": api_input}

        # -------------------------------------------------------------------
        # 3. Create prediction
        # -------------------------------------------------------------------
        prediction = _replicate_request(_REPLICATE_MODEL_URL, token, payload)

        # -------------------------------------------------------------------
        # 4. Poll until complete
        # -------------------------------------------------------------------
        status = prediction.get("status", "unknown")

        if status == "succeeded":
            # Some predictions complete immediately (with Prefer: wait)
            final = prediction
        elif status in ("failed", "canceled"):
            error_msg = prediction.get("error", "No error message provided.")
            raise RuntimeError(
                f"Replicate prediction {status}: {error_msg}"
            )
        else:
            # Need to poll
            poll_url = prediction.get("urls", {}).get("get")
            if not poll_url:
                raise RuntimeError(
                    "Replicate API response did not include a polling URL. "
                    f"Full response: {json.dumps(prediction, indent=2)}"
                )
            final = _poll_prediction(poll_url, token)

        # -------------------------------------------------------------------
        # 5. Extract output URL and download the image
        # -------------------------------------------------------------------
        output = final.get("output")

        if output is None:
            raise RuntimeError(
                "Replicate prediction succeeded but returned no output. "
                f"Full response: {json.dumps(final, indent=2)}"
            )

        # Output can be a single URL string or a list of URL strings
        if isinstance(output, list):
            if len(output) == 0:
                raise RuntimeError(
                    "Replicate prediction succeeded but output list is empty."
                )
            image_url = output[0]
        elif isinstance(output, str):
            image_url = output
        else:
            raise RuntimeError(
                f"Unexpected output format from Replicate: {type(output).__name__}. "
                f"Value: {output}"
            )

        image_tensor = _download_image_as_tensor(image_url)

        return (image_tensor,)
