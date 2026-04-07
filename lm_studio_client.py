"""
LM Studio REST API client for model management and chat completions.

Endpoints used:
  - GET  /api/v1/models       — list all models and their load status
  - POST /api/v1/models/load   — load a model into VRAM
  - POST /api/v1/models/unload — unload a model from VRAM
  - POST /v1/chat/completions  — OpenAI-compatible chat (supports multimodal)
"""

import os
import httpx
from urllib.parse import urlsplit, urlunsplit

LM_STUDIO_API_TOKEN = os.getenv("LM_STUDIO_API_TOKEN", os.getenv("LM_API_TOKEN", "")).strip()


def get_headers() -> dict:
    if not LM_STUDIO_API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {LM_STUDIO_API_TOKEN}"}


def get_base_url(url: str) -> str:
    if not url:
        url = "http://localhost:1234"
    raw = url.strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    for suffix in ("/api/v1", "/v1", "/api"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def is_server_running(base_url: str) -> bool:
    try:
        url = f"{get_base_url(base_url)}/api/v1/models"
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url, headers=get_headers())
            return resp.status_code == 200
    except Exception:
        return False


def list_models(base_url: str, timeout: float = 30.0) -> dict:
    base = get_base_url(base_url)
    headers = get_headers()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base}/api/v1/models", headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError:
        # Fallback to OpenAI-compatible endpoint
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{base}/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "models": [
                        {
                            "type": "llm",
                            "key": m.get("id", "unknown"),
                            "display_name": m.get("id", "Unknown"),
                            "loaded_instances": [],
                        }
                        for m in data.get("data", [])
                    ]
                }
        except Exception as e:
            print(f"[Gemma4] Failed to list models: {e}")
            return {"models": []}
    except Exception as e:
        print(f"[Gemma4] Error listing models: {e}")
        return {"models": []}


def get_loaded_models(base_url: str) -> list:
    result = list_models(base_url)
    loaded = []
    for m in result.get("models", []):
        if len(m.get("loaded_instances", [])) > 0:
            loaded.append(m.get("key", ""))
    return loaded


def _extract_instance_id(instance) -> str:
    if isinstance(instance, str):
        return instance
    if isinstance(instance, dict):
        return instance.get("id") or instance.get("instance_id") or ""
    return ""


def resolve_instance_id(model_key: str, base_url: str) -> str:
    result = list_models(base_url)
    for m in result.get("models", []):
        if m.get("key") != model_key:
            continue
        instances = m.get("loaded_instances", [])
        if instances:
            return _extract_instance_id(instances[0])
    return ""


def load_model(
    model_key: str,
    base_url: str,
    context_length: int = 8192,
    flash_attention: bool = True,
    timeout: float = 120.0,
) -> dict:
    base = get_base_url(base_url)
    endpoint = f"{base}/api/v1/models/load"
    payload = {
        "model": model_key,
        "context_length": context_length,
        "flash_attention": flash_attention,
        "echo_load_config": True,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=payload, headers=get_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Model loading timed out after {timeout}s. "
            "The model may still be loading in LM Studio."
        )
    except httpx.HTTPStatusError as e:
        detail = f"HTTP {e.response.status_code}"
        try:
            body = e.response.json()
            if "error" in body:
                detail = body["error"].get("message", str(body))
        except Exception:
            detail += f": {e.response.text[:200]}"
        raise RuntimeError(f"Failed to load model '{model_key}': {detail}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_key}': {e}")


def unload_model(model_key: str, base_url: str, timeout: float = 30.0) -> dict:
    instance_id = resolve_instance_id(model_key, base_url)
    if not instance_id:
        raise RuntimeError(
            f"No loaded instance found for model '{model_key}'. "
            "Is the model currently loaded in LM Studio?"
        )
    base = get_base_url(base_url)
    endpoint = f"{base}/api/v1/models/unload"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                endpoint, json={"instance_id": instance_id}, headers=get_headers()
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = f"HTTP {e.response.status_code}"
        try:
            detail = e.response.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise RuntimeError(f"Failed to unload model '{model_key}': {detail}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error unloading model '{model_key}': {e}")


def chat_completion(
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 64,
    timeout: float = 180.0,
) -> str:
    base = get_base_url(base_url)
    endpoint = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=payload, headers=get_headers())
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("LM Studio returned no choices in response.")
            return choices[0].get("message", {}).get("content", "")
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Chat completion timed out after {timeout}s. "
            "Try increasing timeout or reducing max_tokens."
        )
    except httpx.HTTPStatusError as e:
        detail = f"HTTP {e.response.status_code}"
        try:
            detail = e.response.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise RuntimeError(f"Chat completion failed: {detail}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Chat completion error: {e}")
