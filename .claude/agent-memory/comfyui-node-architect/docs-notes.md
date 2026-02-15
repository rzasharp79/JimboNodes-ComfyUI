# ComfyUI Documentation Notes

## Doc URLs (as of Feb 2026)
- Overview: https://docs.comfy.org/custom-nodes/overview
- Server overview: https://docs.comfy.org/custom-nodes/backend/server_overview
- Datatypes: https://docs.comfy.org/custom-nodes/backend/datatypes
- Images/masks: https://docs.comfy.org/custom-nodes/backend/images_and_masks
- Snippets: https://docs.comfy.org/custom-nodes/backend/snippets

## Class Attributes (from docs)
- `INPUT_TYPES`: classmethod returning dict with `required`, `optional`, `hidden` keys
- `RETURN_TYPES`: tuple of type strings (trailing comma for single!)
- `RETURN_NAMES`: optional tuple of human-readable output names
- `FUNCTION`: string name of execution method
- `CATEGORY`: slash-separated menu path
- `OUTPUT_NODE`: True for terminal nodes (Save, Preview, etc.)
- `DESCRIPTION`: human-readable string
- `IS_CHANGED`: classmethod, return comparable value; `float("NaN")` = always re-execute
- `VALIDATE_INPUTS`: classmethod, returns True or error string
- `SEARCH_ALIASES`: list of alternative search terms

## IMAGE Tensor Convention
- Shape: [B, H, W, C] where C=3 (RGB)
- dtype: float32
- Range: [0.0, 1.0]
- Channel-last format (unlike PyTorch default)

## PIL <-> Tensor Conversion
```python
# PIL to tensor:
img_np = np.array(pil_image).astype(np.float32) / 255.0
tensor = torch.from_numpy(img_np)[None,]  # (1, H, W, C)

# Tensor to PIL:
img_np = (tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
pil_image = Image.fromarray(img_np, mode="RGB")
```

## Widget Configs
- INT: `{"default": N, "min": N, "max": N, "step": N}`
- FLOAT: `{"default": F, "min": F, "max": F, "step": F, "round": F}`
- STRING: `{"default": "", "multiline": bool, "placeholder": str}`
- BOOLEAN: `{"default": bool, "label_on": str, "label_off": str}`
- Combo: `(["opt1", "opt2"], {"default": "opt1"})`
