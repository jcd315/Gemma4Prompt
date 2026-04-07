"""
Gemma4PromptGen — ComfyUI Node (LM Studio Edition)
====================================================
Multi-model prompt engineer powered by LM Studio API.

Replaces local llama-server subprocess management with LM Studio REST API
for model loading, unloading, and chat completions. Supports smart caching
to skip LLM calls when inputs are unchanged.

  IMAGE MODELS: Flux.1, SDXL, Pony XL, SD 1.5
  VIDEO MODELS: LTX 2.3, Wan 2.2

All models support: NSFW content, image grounding (I2V/I2I), character lock,
environment presets, animation presets, dialogue injection, POV modes.
"""

import re
import time

from . import lm_studio_client as lms
from .image_utils import prepare_image_for_api
from .prompt_cache import PromptCache
from .system_prompts import (
    TARGET_MODELS,
    ENVIRONMENT_PRESETS,
    ANIMATION_PRESETS,
    get_system_prompt,
    is_video_model,
    has_audio,
    build_user_message,
    clean_llm_output,
)


class Gemma4PromptGen:
    """
    Multi-model prompt engineer using LM Studio API backend.

    Supports: LTX 2.3, Wan 2.2, Flux.1, SDXL 1.0, Pony XL, SD 1.5.
    All models support NSFW content, image grounding, character lock, environments.

    Single execution mode with smart caching:
    - If inputs unchanged from last run, returns cached prompt (no LLM call).
    - If inputs changed, flushes ComfyUI VRAM, loads model in LM Studio,
      generates prompt, unloads model, caches result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        env_keys = list(ENVIRONMENT_PRESETS.keys())
        anim_keys = list(ANIMATION_PRESETS.keys())
        return {
            "required": {
                "target_model": (
                    TARGET_MODELS,
                    {
                        "default": TARGET_MODELS[0],
                        "tooltip": (
                            "Which model to generate the prompt FOR. "
                            "Each model has a completely different prompt style."
                        ),
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Describe the scene — characters, action, mood, position, clothing...",
                    },
                ),
                "lm_studio_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234",
                        "tooltip": "LM Studio server URL. Default: http://localhost:1234",
                    },
                ),
                "environment": (
                    env_keys,
                    {
                        "default": "None — LLM decides",
                        "tooltip": "Location preset — injects rich location, lighting and sound context.",
                    },
                ),
                "animation_preset": (
                    anim_keys,
                    {
                        "default": "None",
                        "tooltip": (
                            "Animation preset — pre-loads character names, locations, and tone "
                            "from iconic cartoons. Select a show then describe your scene using character names."
                        ),
                    },
                ),
                "dialogue": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force spoken dialogue into the prompt — video models only.",
                    },
                ),
                "use_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable image grounding — sends your wired IMAGE to the model as context. "
                            "For video: I2V grounding. For image: I2I reference."
                        ),
                    },
                ),
                "screenplay_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "LTX 2.3 only. Generates structured screenplay-style prompt.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Reference or start frame image. Read when use_image is ON."},
                ),
                "character": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Character lock — paste your LoRA trigger or character description.",
                    },
                ),
                "frame_count": (
                    "INT",
                    {
                        "default": 257,
                        "min": 1,
                        "max": 2000,
                        "step": 1,
                        "tooltip": "LTX/Wan frame count @ 25fps. 257 = ~10s. Video models only.",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "LM Studio model identifier. "
                            "Leave empty to use whatever is loaded."
                        ),
                    },
                ),
                "skip_load_unload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Skip model load/unload — just send to whatever is already loaded in LM Studio. "
                            "Use this when you want to manually manage which model is loaded."
                        ),
                    },
                ),
                "unload_model_after": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Unload the model from LM Studio after generation completes. "
                            "Only applies when skip_load_unload is ON. Frees VRAM so ComfyUI "
                            "can use it for the rest of the workflow."
                        ),
                    },
                ),
                "context_length": (
                    "INT",
                    {
                        "default": 14336,
                        "min": 2048,
                        "max": 131072,
                        "step": 1024,
                        "tooltip": "Context window size when loading the model. Default 14k.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 8192,
                        "step": 256,
                        "tooltip": "Maximum response tokens from the LLM.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Generation temperature. Higher = more creative, lower = more focused.",
                    },
                ),
                "pov_mode": (
                    ["Off", "POV Female", "POV Male"],
                    {
                        "default": "Off",
                        "tooltip": "First-person POV mode. Camera IS the viewer's eyes.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Seed for Random environment pick. 0 = different every run.",
                    },
                ),
                "flush_comfyui_vram": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Clear ComfyUI models from VRAM before calling LM Studio.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    CATEGORY = "LoRa-Daddy/Gemma4"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — our internal PromptCache handles optimization
        return float("nan")

    def execute(
        self,
        target_model,
        instruction,
        lm_studio_url,
        environment,
        animation_preset,
        dialogue,
        use_image,
        screenplay_mode=False,
        image=None,
        character="",
        frame_count=257,
        model_name="",
        skip_load_unload=False,
        unload_model_after=False,
        context_length=14336,
        max_tokens=2048,
        temperature=0.7,
        pov_mode="Off",
        seed=0,
        flush_comfyui_vram=True,
    ):
        t0 = time.time()

        # ── 1. Build system prompt ───────────────────────────────────────
        system_prompt = get_system_prompt(target_model, screenplay_mode, animation_preset)

        # ── 2. Prepare image ─────────────────────────────────────────────
        image_base64 = None
        if use_image and image is not None:
            try:
                image_base64 = prepare_image_for_api(image, target_mp=0.75, jpeg_quality=75)
            except Exception as e:
                print(f"[Gemma4] Image conversion failed: {e}")

        # ── 3. Check cache ───────────────────────────────────────────────
        cache = PromptCache.get()
        input_hash = cache.compute_hash(
            system_prompt=system_prompt,
            instruction=instruction,
            image_hash=image_base64[:64] if image_base64 else "",
            model_name=model_name,
            temperature=str(temperature),
            max_tokens=str(max_tokens),
            environment=environment,
            animation_preset=animation_preset,
            dialogue=str(dialogue),
            pov_mode=pov_mode,
            character=character,
            frame_count=str(frame_count),
            screenplay_mode=str(screenplay_mode),
        )

        cached = cache.check(input_hash)
        if cached is not None:
            elapsed = time.time() - t0
            print(f"\n{'='*60}")
            print(f"GEMMA4 PROMPT GEN — {target_model}")
            print(f"Cache hit — prompt unchanged, skipping LLM ({elapsed:.1f}s)")
            print(f"{'='*60}")
            print(cached[:600] + ("..." if len(cached) > 600 else ""))
            print(f"{'='*60}\n")
            return {"ui": {"text": [cached]}, "result": (cached,)}

        # ── 4. Check LM Studio is running ────────────────────────────────
        if not lms.is_server_running(lm_studio_url):
            return (
                f"Error: Cannot connect to LM Studio at {lm_studio_url}. "
                "Make sure LM Studio is running with the server enabled.",
            )

        # ── 5. Flush ComfyUI VRAM ────────────────────────────────────────
        if flush_comfyui_vram:
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            print("[Gemma4] ComfyUI VRAM flushed")

        # ── 6. Load model in LM Studio ───────────────────────────────────
        loaded_by_us = False
        effective_model = model_name.strip() if model_name else ""

        if not skip_load_unload:
            if not effective_model:
                return (
                    "Error: model_name is required when skip_load_unload is off. "
                    "Set the model identifier (e.g. google/gemma-4-27b-it-GGUF) or enable skip_load_unload.",
                )
            try:
                # Check if already loaded
                loaded_models = lms.get_loaded_models(lm_studio_url)
                if effective_model not in loaded_models:
                    print(f"[Gemma4] Loading model: {effective_model}")
                    lms.load_model(
                        effective_model, lm_studio_url,
                        context_length=context_length, flash_attention=True,
                    )
                    loaded_by_us = True
                    print(f"[Gemma4] Model loaded: {effective_model}")
                else:
                    print(f"[Gemma4] Model already loaded: {effective_model}")
            except Exception as e:
                return (f"Error loading model: {e}",)
        else:
            # Bypass mode — detect whatever is loaded
            try:
                loaded_models = lms.get_loaded_models(lm_studio_url)
                if loaded_models:
                    effective_model = loaded_models[0]
                    print(f"[Gemma4] Bypass mode — using loaded model: {effective_model}")
                else:
                    return (
                        "Error: skip_load_unload is enabled but no model is loaded in LM Studio. "
                        "Load a model in LM Studio first, or disable skip_load_unload.",
                    )
            except Exception as e:
                return (f"Error detecting loaded models: {e}",)

        # ── 7. Build messages ────────────────────────────────────────────
        has_image = image_base64 is not None
        user_text = build_user_message(
            instruction=instruction,
            system_prompt=system_prompt,
            target_model=target_model,
            environment=environment,
            frame_count=frame_count,
            dialogue=dialogue,
            character=character,
            seed=seed,
            has_image=has_image,
            screenplay_mode=screenplay_mode,
            pov_mode=pov_mode,
            animation_preset=animation_preset,
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Build user content — multimodal if image provided
        if image_base64 is not None and instruction.strip():
            # Image + text: image first, then text as supplemental
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": user_text},
            ]
        elif image_base64 is not None:
            # Image only
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": user_text},
            ]
        else:
            # Text only
            user_content = user_text

        messages.append({"role": "user", "content": user_content})

        # ── 8. Call LLM ──────────────────────────────────────────────────
        try:
            print(f"[Gemma4] Sending chat completion to {effective_model}...")
            raw = lms.chat_completion(
                base_url=lm_studio_url,
                model=effective_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            # Try to unload even on failure
            should_unload_on_error = (not skip_load_unload and loaded_by_us) or (skip_load_unload and unload_model_after)
            if should_unload_on_error:
                try:
                    lms.unload_model(effective_model, lm_studio_url)
                except Exception:
                    pass
            return (f"Error during generation: {e}",)

        # Strip residual thinking tags
        raw = re.sub(r'<\|channel>thought\n.*?<channel\|>', '', raw, flags=re.DOTALL)

        # ── 9. Clean output ──────────────────────────────────────────────
        is_screenplay = screenplay_mode and "LTX" in target_model
        cleaned = clean_llm_output(raw.strip(), screenplay_mode=is_screenplay)

        # ── 10. Unload model ─────────────────────────────────────────────
        # Unload if: (a) we loaded it ourselves, or (b) user wants unload in bypass mode
        should_unload = (not skip_load_unload and loaded_by_us) or (skip_load_unload and unload_model_after)
        if should_unload:
            try:
                lms.unload_model(effective_model, lm_studio_url)
                print(f"[Gemma4] Model unloaded: {effective_model} — VRAM freed")
            except Exception as e:
                print(f"[Gemma4] Warning: failed to unload model: {e}")

        # ── 11. Cache result ─────────────────────────────────────────────
        if not cleaned.startswith("Error") and not cleaned.startswith("⚠️"):
            cache.store(input_hash, cleaned)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"GEMMA4 PROMPT GEN — {target_model}")
        print(f"Generated in {elapsed:.1f}s via {effective_model}")
        print(f"{'='*60}")
        print(cleaned[:600] + ("..." if len(cleaned) > 600 else ""))
        print(f"{'='*60}\n")

        return {"ui": {"text": [cleaned]}, "result": (cleaned,)}


# ── ComfyUI Registration ────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "Gemma4PromptGen": Gemma4PromptGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma4PromptGen": "Gemma 4 Prompt Generator (LM Studio)",
}
