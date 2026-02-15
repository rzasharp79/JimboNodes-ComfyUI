"""
Jimbo's OpenRouter Tunnel - ComfyUI custom node that sends text (and
optional image/video/audio) to any LLM via the OpenRouter chat completions
API and returns the response as a STRING.

Model list is stored in a JSON file and can be managed (add/edit/delete)
directly from the node.
"""

import base64
import io
import json
import mimetypes
import os
import urllib.request
import urllib.error

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")
_MODELS_FILE = os.path.join(_PRESETS_DIR, "openrouter_models.json")


# ---------------------------------------------------------------------------
# Model list helpers
# ---------------------------------------------------------------------------

def _load_models() -> list[str]:
    if not os.path.isfile(_MODELS_FILE):
        return []
    with open(_MODELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_models(models: list[str]) -> None:
    os.makedirs(_PRESETS_DIR, exist_ok=True)
    with open(_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(set(models)), f, indent=4)


def _model_names() -> list[str]:
    models = _load_models()
    if not models:
        return ["(no models)"]
    return sorted(models)


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------

def _comfy_image_to_data_uri(image_tensor: torch.Tensor, index: int = 0) -> str:
    img_np = image_tensor[index].cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np, mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _file_to_data_uri(file_path: str) -> str:
    """Read a file from disk and return it as a base64 data URI."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class OpenRouterTunnel:
    """ComfyUI node that sends prompts to LLMs via the OpenRouter API."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "Jimbo Comfy Nodes/openrouter"
    DESCRIPTION = (
        "Send a prompt (with optional image/video/audio) to any LLM via "
        "OpenRouter and return the response as a STRING."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_names(), {
                    "default": _model_names()[0],
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in comprehensive detail. Include: the main subject and its appearance, positioning and composition, lighting direction and quality, color palette, background elements, artistic style, overall atmosphere and mood, and camera perspective/angle.",
                    "multiline": True,
                    "placeholder": "Your message to the model",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "You are a helpful AI assistant. You only output exactly what is asked, do not include any preamble or follow-up or thoughts.",
                    "multiline": True,
                    "placeholder": "Optional system prompt",
                }),
                "image": ("IMAGE",),
                "video_path": ("STRING", {
                    "default": "",
                    "placeholder": "Path to video file (optional)",
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "placeholder": "Path to audio file (optional)",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 128000,
                    "step": 256,
                }),
                "model_action": (["none", "add", "edit", "delete"], {
                    "default": "none",
                }),
                "new_model_name": ("STRING", {
                    "default": "",
                    "placeholder": "Model ID for add/edit (e.g. openai/gpt-4o)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def execute(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        image: torch.Tensor | None = None,
        video_path: str = "",
        audio_path: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model_action: str = "none",
        new_model_name: str = "",
    ) -> dict:
        # ---- Model management actions ----
        if model_action == "add":
            name = new_model_name.strip()
            if not name:
                raise ValueError("Enter a model ID in new_model_name to add.")
            models = _load_models()
            if name not in models:
                models.append(name)
                _save_models(models)
                print(f"[OpenRouterTunnel] Added model: {name}")
            return {
                "ui": {"text": [f"Added model: {name}"]},
                "result": ("",),
            }

        if model_action == "edit":
            name = new_model_name.strip()
            if not name:
                raise ValueError("Enter the new model ID in new_model_name.")
            models = _load_models()
            if model in models:
                models.remove(model)
            models.append(name)
            _save_models(models)
            print(f"[OpenRouterTunnel] Replaced {model} with {name}")
            return {
                "ui": {"text": [f"Replaced {model} with {name}"]},
                "result": ("",),
            }

        if model_action == "delete":
            models = _load_models()
            if model in models:
                models.remove(model)
                _save_models(models)
                print(f"[OpenRouterTunnel] Deleted model: {model}")
            return {
                "ui": {"text": [f"Deleted model: {model}"]},
                "result": ("",),
            }

        # ---- API call ----
        token = os.environ.get("OPENROUTER_API", "").strip()
        if not token:
            raise EnvironmentError(
                "OPENROUTER_API environment variable is not set. "
                "Please set it to your OpenRouter API key."
            )

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        # Build message content (multimodal if attachments provided)
        content_parts = []

        # Image
        if image is not None:
            batch_size = image.shape[0]
            for i in range(batch_size):
                data_uri = _comfy_image_to_data_uri(image, i)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                })

        # Video
        if video_path.strip() and os.path.isfile(video_path.strip()):
            data_uri = _file_to_data_uri(video_path.strip())
            content_parts.append({
                "type": "video_url",
                "video_url": {"url": data_uri},
            })

        # Audio
        if audio_path.strip() and os.path.isfile(audio_path.strip()):
            data_uri = _file_to_data_uri(audio_path.strip())
            content_parts.append({
                "type": "input_audio",
                "input_audio": {
                    "data": data_uri.split(",", 1)[1] if "," in data_uri else data_uri,
                    "format": os.path.splitext(audio_path)[1].lstrip(".") or "wav",
                },
            })

        # Text prompt
        content_parts.append({"type": "text", "text": prompt})

        # If only text, use simple string content
        if len(content_parts) == 1:
            user_content = prompt
        else:
            user_content = content_parts

        # Build messages
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Make the API request
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _OPENROUTER_URL, data=data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                body = response.read().decode("utf-8")
                result = json.loads(body)
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"OpenRouter API returned HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to connect to OpenRouter API: {exc.reason}"
            ) from exc

        # Extract response text
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"OpenRouter returned no choices. Full response: "
                f"{json.dumps(result, indent=2)}"
            )

        response_text = choices[0].get("message", {}).get("content", "")

        return {
            "ui": {"text": [response_text[:200]]},
            "result": (response_text,),
        }
