# Agent Memory - ComfyUI Node Architect

## Project: JimboNodes-ComfyUI
- Location: `D:\ComfyUI\custom_nodes\JimboNodes-ComfyUI`
- Structure: `__init__.py` (root) + `nodes/` package with individual node files
- Naming convention: Internal class key = `Jimbo<NodeName>`, display = `Jimbo's <Node Name>`
- Category root: `Jimbo Comfy Nodes/`

## ComfyUI Docs
- Custom nodes docs moved from `/essentials/` to `/custom-nodes/backend/` path
- Key pages: `server_overview`, `datatypes`, `images_and_masks`, `snippets`
- See [docs-notes.md](docs-notes.md) for detailed patterns

## Replicate API Node Pattern
- See [replicate-pattern.md](replicate-pattern.md) for the full pattern used
- Uses stdlib `urllib.request` instead of `requests` (no extra deps)
- Poll with `Prefer: wait` header then fallback to GET polling
- IMAGE tensor: (B,H,W,C) float32 [0,1], convert via PIL+numpy

## Key Conventions Discovered
- `IS_CHANGED` returning `float("NaN")` forces re-execution every queue
- Combo/dropdown: first element of tuple is list of strings, second is config dict
- Optional IMAGE input: just `("IMAGE",)` with no config dict needed
- `__all__` in root `__init__.py` should export `NODE_CLASS_MAPPINGS` (and optionally `NODE_DISPLAY_NAME_MAPPINGS`)
