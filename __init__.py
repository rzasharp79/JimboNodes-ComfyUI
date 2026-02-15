"""
JimboNodes-ComfyUI
==================
A collection of custom nodes for ComfyUI.

This package exports NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
so that ComfyUI can discover and register the nodes at startup.
"""

from .nodes import NanoBananaPro, ImageEditPresets, Seedream4_5, QwenImageEditPlus, FluxKontextMax, OpenRouterTunnel

NODE_CLASS_MAPPINGS = {
    "JimboNanoBananaPro": NanoBananaPro,
    "JimboImageEditPresets": ImageEditPresets,
    "JimboSeedream4_5": Seedream4_5,
    "JimboQwenImageEditPlus": QwenImageEditPlus,
    "JimboFluxKontextMax": FluxKontextMax,
    "JimboOpenRouterTunnel": OpenRouterTunnel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimboNanoBananaPro": "Jimbo's Nano Banana Pro",
    "JimboImageEditPresets": "Jimbo Image Edit Presets",
    "JimboSeedream4_5": "Jimbo Seedream 4.5",
    "JimboQwenImageEditPlus": "Jimbo Qwen Image Edit+",
    "JimboFluxKontextMax": "Jimbo Flux Kontext Max",
    "JimboOpenRouterTunnel": "Jimbo OpenRouter Tunnel",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
