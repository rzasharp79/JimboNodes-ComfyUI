"""
Jimbo Image Edit Presets - ComfyUI custom node for managing text presets
used for image editing prompts.

Presets are stored as simple name->text pairs in a JSON file and can be
added, edited, or deleted directly from the node. The selected preset's
text is output as a STRING.
"""

import json
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")
_PRESETS_FILE = os.path.join(_PRESETS_DIR, "image_edit_presets.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_presets() -> dict:
    """Load presets from the JSON file. Returns an empty dict on error."""
    if not os.path.isfile(_PRESETS_FILE):
        return {}
    with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_presets(presets: dict) -> None:
    """Write presets dict to the JSON file."""
    os.makedirs(_PRESETS_DIR, exist_ok=True)
    with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=4)


def _preset_names() -> list[str]:
    """Return sorted preset names, with a fallback if the file is empty."""
    presets = _load_presets()
    if not presets:
        return ["(no presets)"]
    return sorted(presets.keys())


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class ImageEditPresets:
    """ComfyUI node for managing text presets for image editing.

    Select a preset from the dropdown to output its text, or use the
    action input to save new presets, overwrite existing ones, or delete them.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Jimbo Comfy Nodes/preset_managers"
    DESCRIPTION = (
        "Manage text presets for image editing prompts. Select a preset to "
        "output its text, or save/delete presets using the action input."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (_preset_names(), {
                    "default": _preset_names()[0],
                }),
                "action": (["load", "save", "delete"], {
                    "default": "load",
                }),
            },
            "optional": {
                "save_name": ("STRING", {
                    "default": "",
                    "placeholder": "Name for new/updated preset",
                }),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Preset text content",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Re-read presets each execution so the dropdown stays current."""
        return float("NaN")

    def execute(
        self,
        preset: str,
        action: str,
        save_name: str = "",
        text: str = "",
    ) -> dict:
        presets = _load_presets()

        # ---- Save action ----
        if action == "save":
            name = save_name.strip()
            if not name:
                raise ValueError(
                    "Please enter a name in save_name to save a preset."
                )
            presets[name] = text
            _save_presets(presets)
            print(f"[ImageEditPresets] Saved preset: {name}")
            return {
                "ui": {"text": [f"Saved preset: {name}"]},
                "result": (text,),
            }

        # ---- Delete action ----
        if action == "delete":
            if preset in presets:
                del presets[preset]
                _save_presets(presets)
                print(f"[ImageEditPresets] Deleted preset: {preset}")
            else:
                print(f"[ImageEditPresets] Preset not found: {preset}")
            return {
                "ui": {"text": [f"Deleted preset: {preset}"]},
                "result": ("",),
            }

        # ---- Load action (default) ----
        # Text input overrides preset when non-empty
        if text.strip():
            return {
                "ui": {"text": ["Using text override"]},
                "result": (text,),
            }

        if preset in presets:
            loaded_text = presets[preset]
            return {
                "ui": {"text": [f"Loaded preset: {preset}"]},
                "result": (loaded_text,),
            }

        return {
            "ui": {"text": [f"Preset not found: {preset}"]},
            "result": ("",),
        }
