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
| `REPLICATE_API_TOKEN` | Jimbo's Nano Banana Pro | Your API token from [replicate.com](https://replicate.com/account/api-tokens) |

## Nodes

---

### Jimbo's Nano Banana Pro

**Category:** `Jimbo Comfy Nodes/replicate_api`

**Description:** Generates images using Google's Nano Banana Pro model via the Replicate API. Provide a text prompt and optionally one or more reference images. The node sends the request to Replicate, waits for the result, and returns the generated image as a standard ComfyUI IMAGE tensor.

#### Inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | STRING (multiline) | Yes | `""` | A text description of the image you want to generate. |
| `image_input` | IMAGE | No | *(none)* | Optional input images to use as reference. Supports a batch of up to 14 images. Each image is converted to a PNG data URI and sent to the API. |
| `resolution` | Combo: `1K`, `2K`, `4K` | No | `2K` | Resolution of the generated image. |
| `aspect_ratio` | Combo: `match_input_image`, `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` | No | `match_input_image` | Aspect ratio of the generated image. |
| `output_format` | Combo: `jpg`, `png` | No | `jpg` | Format of the output image. |
| `safety_filter_level` | Combo: `block_low_and_above`, `block_medium_and_above`, `block_only_high` | No | `block_only_high` | Content safety filter strictness. `block_low_and_above` is the strictest; `block_only_high` is the most permissive. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | The generated image as a ComfyUI IMAGE tensor (B, H, W, 3) with float32 values in [0, 1]. |

#### Usage Instructions

1. **Set your API token.** Before launching ComfyUI, set the `REPLICATE_API_TOKEN` environment variable to your Replicate API token.
2. **Add the node.** In ComfyUI, right-click the canvas and navigate to **Jimbo Comfy Nodes > replicate_api > Jimbo's Nano Banana Pro**.
3. **Enter a prompt.** Type a description of the image you want to generate in the `prompt` text area.
4. **Optionally connect reference images.** Connect an IMAGE output from another node (such as Load Image) to the `image_input` input. Batched images (up to 14) are all sent as references.
5. **Configure options.** Adjust resolution, aspect ratio, output format, and safety filter level as desired.
6. **Queue the prompt.** The node will send the request to Replicate, poll until the image is ready, download it, and output it as a standard IMAGE tensor.
7. **Connect the output.** Connect the `image` output to a Preview Image or Save Image node to see the result.

#### Example Workflow

```
[Load Image] --> image_input --> [Jimbo's Nano Banana Pro] --> image --> [Preview Image]
                                        ^
                                        |
                                   prompt: "A futuristic city at sunset"
                                   resolution: 2K
                                   aspect_ratio: 16:9
```

A minimal workflow:
- Add a **Jimbo's Nano Banana Pro** node.
- Type your prompt (e.g., `"A cat wearing a space helmet"`).
- Connect the `image` output to a **Preview Image** node.
- Click **Queue Prompt**.

#### Notes and Tips

- **API costs apply.** Each execution calls the Replicate API, which may incur costs on your Replicate account.
- **Timeout.** The node polls for up to 5 minutes (300 seconds) before timing out. If the model is cold-starting, it may take longer on the first run.
- **No external packages required.** This node uses only Python standard library modules plus torch, numpy, and Pillow, all of which are already available in ComfyUI.
- **Always re-executes.** Because the API may return different results for the same inputs, this node always re-executes when the workflow is queued (it uses `IS_CHANGED = NaN`).
- **Batch input support.** If you connect a batched IMAGE tensor, up to 14 images from the batch will be sent as reference images. Images beyond the 14th are silently ignored.
- **Error messages.** If the API token is missing, the prompt is empty, or the API returns an error, the node raises a clear error message describing the problem.

---

## License

MIT
