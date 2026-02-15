# JimboNodes-ComfyUI

A collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/JimboNodes-ComfyUI.git
```

Restart ComfyUI to load the nodes.

## Prerequisites

Some nodes in this pack require API tokens. Set the following environment variables before launching ComfyUI:

| Variable | Required By | Description |
|---|---|---|
| `REPLICATE_API_TOKEN` | Nano Banana Pro, Seedream 4.5, Qwen Image Edit+, Flux Kontext Max | Your API token from [replicate.com](https://replicate.com/account/api-tokens) |
| `OPENROUTER_API` | OpenRouter Tunnel | Your API key from [openrouter.ai](https://openrouter.ai/keys) |

## Nodes

---

### Jimbo's Nano Banana Pro

**Category:** `Jimbo Comfy Nodes/replicate_api`

Generate images using Google's Nano Banana Pro model via the Replicate API. Provide a text prompt and optionally one or more reference images.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | STRING | Yes | `""` | Text description of the image to generate. |
| `image_input` | IMAGE | No | — | Reference images. Batch up to 14 via the Batch Images node. |
| `resolution` | `1K` / `2K` / `4K` | No | `2K` | Output resolution. |
| `aspect_ratio` | Combo (11 options) | No | `match_input_image` | Output aspect ratio. |
| `output_format` | `jpg` / `png` | No | `jpg` | Output image format. |
| `safety_filter_level` | Combo (3 levels) | No | `block_only_high` | Content safety filter strictness. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Generated image tensor. |

---

### Jimbo Seedream 4.5

**Category:** `Jimbo Comfy Nodes/replicate_api`

Generate images using ByteDance's Seedream 4.5 model via the Replicate API. Supports text-to-image and multi-reference image-to-image with up to 14 input images.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | STRING | Yes | `""` | Text description of the image to generate. |
| `image_input` | IMAGE | No | — | Reference images. Batch up to 14 via the Batch Images node. |
| `size` | `2K` / `4K` / `custom` | No | `2K` | Output resolution. Use `custom` with width/height. |
| `aspect_ratio` | Combo (9 options) | No | `match_input_image` | Output aspect ratio. Ignored when size is `custom`. |
| `width` | INT (1024–4096) | No | `2048` | Custom width. Only used when size is `custom`. |
| `height` | INT (1024–4096) | No | `2048` | Custom height. Only used when size is `custom`. |
| `sequential_image_generation` | `disabled` / `auto` | No | `disabled` | `auto` lets the model generate multiple related images. |
| `max_images` | INT (1–15) | No | `1` | Max images when sequential generation is `auto`. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Generated image(s) batched into a single tensor. |

---

### Jimbo Qwen Image Edit+

**Category:** `Jimbo Comfy Nodes/replicate_api`

Edit images using Qwen Image Edit+ via the Replicate API. Provide one or more reference images and a text instruction describing the desired edit.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | IMAGE | Yes | — | Reference image(s). Batch for multiple. |
| `prompt` | STRING | Yes | `""` | Text instruction describing the edit. |
| `seed` | INT | No | `-1` | Random seed. `-1` for random. |
| `go_fast` | BOOLEAN | No | `True` | Faster predictions with optimizations. |
| `aspect_ratio` | Combo (6 options) | No | `match_input_image` | Output aspect ratio. |
| `output_format` | `webp` / `jpg` / `png` | No | `webp` | Output format. |
| `output_quality` | INT (0–100) | No | `95` | Quality for jpg/webp. Ignored for png. |
| `disable_safety_checker` | BOOLEAN | No | `False` | Disable content safety filter. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Edited image tensor. |

---

### Jimbo Flux Kontext Max

**Category:** `Jimbo Comfy Nodes/replicate_api`

Generate or edit images using Black Forest Labs' Flux Kontext Max via the Replicate API. Provide a text prompt and optionally a single reference image.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | STRING | Yes | `""` | Text description or edit instruction. |
| `input_image` | IMAGE | No | — | Single reference image. |
| `seed` | INT | No | `-1` | Random seed. `-1` for random. |
| `aspect_ratio` | Combo (14 options) | No | `match_input_image` | Output aspect ratio. |
| `output_format` | `png` / `jpg` | No | `png` | Output format. |
| `safety_tolerance` | INT (0–6) | No | `2` | 0 = most strict, 6 = most permissive. Max 2 when using input images. |
| `prompt_upsampling` | BOOLEAN | No | `False` | Automatic prompt improvement. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Generated/edited image tensor. |

---

### Jimbo Image Edit Presets

**Category:** `Jimbo Comfy Nodes/preset_managers`

Manage reusable text presets for image editing prompts. Select a preset from the dropdown to output its text, or save/delete presets using the action input. If the text field has content, it overrides the selected preset.

Presets are stored in `presets/image_edit_presets.json`.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `preset` | Dropdown | Yes | *(first preset)* | Select a saved preset. |
| `action` | `load` / `save` / `delete` | Yes | `load` | `load` outputs the preset text, `save` saves a new preset, `delete` removes the selected preset. |
| `save_name` | STRING | No | `""` | Name for a new or updated preset (used with `save`). |
| `text` | STRING | No | `""` | Text content. Overrides the preset when non-empty. Used as the value when saving. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `text` | STRING | The preset text (or override text). |

#### Preset Management

- **Add a preset:** Set action to `save`, enter a name in `save_name`, type the text, and queue.
- **Edit a preset:** Same as add — saving with an existing name overwrites it.
- **Delete a preset:** Select the preset in the dropdown, set action to `delete`, and queue.
- **Refresh dropdown:** Re-queue after any save/delete to see the updated list.

---

### Jimbo OpenRouter Tunnel

**Category:** `Jimbo Comfy Nodes/openrouter`

Send a prompt (with optional image, video, or audio) to any LLM via the OpenRouter chat completions API and return the response as a STRING. The model dropdown is managed the same way as presets — add, edit, or delete models directly from the node.

Models are stored in `presets/openrouter_models.json`.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | Dropdown | Yes | *(first model)* | LLM model to use. |
| `prompt` | STRING | Yes | *(image description prompt)* | Your message to the model. |
| `system_prompt` | STRING | No | *(helpful assistant prompt)* | System message sent before the user prompt. |
| `image` | IMAGE | No | — | Image(s) sent as base64 content to the model. |
| `video_path` | STRING | No | `""` | File path to a video to include. |
| `audio_path` | STRING | No | `""` | File path to an audio file to include. |
| `temperature` | FLOAT (0.0–2.0) | No | `0.7` | Sampling temperature. |
| `max_tokens` | INT (1–128000) | No | `1024` | Maximum response length. |
| `model_action` | `none` / `add` / `edit` / `delete` | No | `none` | Manage the model list. |
| `new_model_name` | STRING | No | `""` | Model ID for add/edit (e.g. `openai/gpt-4o`). |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `response` | STRING | The LLM's response text. |

#### Model Management

- **Add a model:** Set `model_action` to `add`, type the model ID in `new_model_name`, and queue.
- **Edit a model:** Select the model to replace, set `model_action` to `edit`, enter the new ID in `new_model_name`, and queue.
- **Delete a model:** Select the model, set `model_action` to `delete`, and queue.
- **Refresh dropdown:** Re-queue after changes to see the updated list.

---

## Notes

- **API costs apply.** Replicate and OpenRouter nodes call external APIs that may incur costs.
- **Timeout.** Replicate nodes poll for up to 5 minutes (300 seconds) before timing out.
- **No extra packages required.** All nodes use only Python standard library plus torch, numpy, and Pillow (already in ComfyUI).
- **Always re-executes.** All API nodes use `IS_CHANGED = NaN` so they re-run on every queue.

## License

MIT
