"""
Gemma4PromptGen — ComfyUI Custom Node (LM Studio Edition)
Multi-model prompt engineer powered by LM Studio API.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .gemma4_prompt_gen import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    print(f"[Gemma4] Could not load Gemma4 PromptGen nodes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
