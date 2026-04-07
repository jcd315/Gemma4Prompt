"""
Image processing utilities for ComfyUI tensor → LM Studio API.

Pipeline: ComfyUI IMAGE tensor → PIL → resize → JPEG → base64 string
"""

import io
import base64
import numpy as np
from PIL import Image


def comfyui_tensor_to_pil(image_tensor) -> Image.Image:
    if hasattr(image_tensor, "cpu"):
        image_tensor = image_tensor.cpu().numpy()
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    if not isinstance(image_tensor, np.ndarray):
        image_tensor = np.array(image_tensor)
    if len(image_tensor.shape) == 3:
        channels = image_tensor.shape[2]
        if channels == 1:
            image_tensor = image_tensor[:, :, 0]
        elif channels == 2:
            image_tensor = image_tensor[:, :, 0]
        elif channels > 4:
            image_tensor = image_tensor[:, :, :3]
    image_array = (image_tensor * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_array)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def resize_to_megapixels(image: Image.Image, target_mp: float = 0.5) -> Image.Image:
    current = image.width * image.height
    target = target_mp * 1_000_000
    if current <= target:
        return image
    scale = (target / current) ** 0.5
    w = max(1, int(image.width * scale))
    h = max(1, int(image.height * scale))
    return image.resize((w, h), Image.Resampling.LANCZOS)


def prepare_image_for_api(
    image_tensor, target_mp: float = 0.5, jpeg_quality: int = 75
) -> str:
    pil = comfyui_tensor_to_pil(image_tensor)
    pil = resize_to_megapixels(pil, target_mp)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode()
