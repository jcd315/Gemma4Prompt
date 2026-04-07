# Gemma4PromptGen — LM Studio Edition

ComfyUI custom node that generates model-specific prompts using Gemma 4 (or any LLM) via LM Studio's API.

Give it a text description, an image, or both — and it generates a properly formatted prompt for your target model.

## Supported Target Models

| Model | Format |
|---|---|
| LTX 2.3 | Cinematic video prompt with audio layer |
| Wan 2.2 | Motion-first video prompt, 80-120 words |
| Flux.1 | Natural language image prompt |
| SDXL 1.0 | Comma-separated booru tags + negative |
| Pony XL | Score/rating prefix + booru tags |
| SD 1.5 | Weighted 75-token classic prompt |

## How It Works

1. You describe a scene (text, image, or both)
2. The node sends it to your LLM running in LM Studio
3. The LLM generates a prompt formatted specifically for your chosen target model
4. The prompt is output as a STRING to connect to your generation workflow

### VRAM Management

The node handles VRAM automatically:
- **Flushes ComfyUI models** before calling the LLM (configurable)
- **Loads the LLM** in LM Studio via API
- **Generates the prompt**
- **Unloads the LLM** so ComfyUI can reclaim VRAM for image/video generation

### Smart Caching

If you queue the same prompt twice without changing any inputs, the node skips the LLM entirely and returns the cached result instantly. No VRAM shuffling, no waiting.

## Installation

1. Clone or download this repo into your ComfyUI `custom_nodes/` folder:
   ```
   cd ComfyUI/custom_nodes/
   git clone https://github.com/jcd315/Gemma4Prompt.git
   ```
2. Install dependencies (if not already present):
   ```
   pip install httpx Pillow numpy
   ```
3. Restart ComfyUI
4. Find the node: **Add Node > LoRa-Daddy/Gemma4 > Gemma 4 Prompt Generator (LM Studio)**

## Requirements

- [LM Studio](https://lmstudio.ai/) running with the server enabled (default port 1234)
- A downloaded model in LM Studio (e.g. Gemma 4 27B/31B, or any model you prefer)
- Python packages: `httpx`, `Pillow`, `numpy` (plus `torch` from ComfyUI)

## Node Inputs

### Required
| Input | Description |
|---|---|
| **target_model** | Which model format to generate the prompt for |
| **instruction** | Your scene description — characters, action, mood, etc. |
| **lm_studio_url** | LM Studio server URL (default: `http://localhost:1234`) |
| **environment** | Location preset with lighting and sound (80+ options) |
| **animation_preset** | Cartoon universe preset (SpongeBob, Bluey, Rick and Morty, etc.) |
| **dialogue** | Force spoken dialogue into video prompts |
| **use_image** | Send a connected image to the LLM for grounding |
| **screenplay_mode** | LTX 2.3 structured screenplay format |

### Optional
| Input | Description |
|---|---|
| **image** | Reference image for I2V/I2I grounding |
| **character** | Character lock — LoRA trigger or physical description |
| **frame_count** | Video frame count for duration scaling |
| **model_name** | LM Studio model identifier for auto-loading |
| **skip_load_unload** | Use whatever model is already loaded in LM Studio |
| **unload_model_after** | Unload model after generation (only with skip_load_unload) |
| **context_length** | Context window when loading (default: 14k) |
| **max_tokens** | Max response tokens (default: 2048) |
| **temperature** | Generation temperature (default: 0.7) |
| **pov_mode** | First-person POV (Off / Female / Male) |
| **seed** | Seed for random environment selection |
| **flush_comfyui_vram** | Clear ComfyUI models before LLM call |

## Bypass Mode

Enable **skip_load_unload** to skip model management entirely. The node just sends the prompt to whatever model you've already loaded in LM Studio. Useful when:
- You want to use a different model (not Gemma)
- You're running multiple generations and want to keep the model loaded
- You manage models manually through LM Studio's GUI

Optionally enable **unload_model_after** to free VRAM when the generation is done, even in bypass mode.

## File Structure

```
Gemma4Prompt/
  __init__.py              — ComfyUI entry point
  gemma4_prompt_gen.py     — Main node class
  lm_studio_client.py      — LM Studio REST API client
  system_prompts.py        — Target model prompts, environment & animation presets
  image_utils.py           — Image tensor to base64 pipeline
  prompt_cache.py          — Smart input-hash caching
```

## Credits

- Original project by [Brojakhoeman](https://github.com/Brojakhoeman/Gemma4Prompt)
- LM Studio edition by [jcd315](https://github.com/jcd315)
- Built with [Claude Code](https://claude.com/claude-code)

## License

MIT
