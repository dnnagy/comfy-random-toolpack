"""Experimental Ideogram 4 image-prompt / conditioning-append nodes.

Ideogram 4's text encoder is a Qwen3-VL-8B model (see
``comfy/text_encoders/ideogram4.py``). The Qwen3-VL path can accept images via
``clip.tokenize(prompt, images=[...])``, inserting the image into the token
sequence at a ``<|vision_start|><|image_pad|><|vision_end|>`` placeholder, the
same mechanism the built-in Qwen image-edit nodes use.

These nodes let you:

* encode a single image (plus optional text) into Ideogram 4 ``CONDITIONING``;
* encode text into Ideogram 4 ``CONDITIONING``;
* optionally APPEND the result onto an incoming ``CONDITIONING`` (concatenated
  along the sequence dimension), so you can chain prompts:

      CLIP text prompt -> Image prompt -> Image prompt 2 -> Sampling
      Image prompt -> CLIP text prompt -> Image prompt 2 -> Sampling

EXPERIMENTAL: whether Ideogram 4 makes meaningful use of the image signal
depends on how the model was trained; this only wires up the encoder plumbing.
"""

from __future__ import annotations

import logging
import math

import torch

from comfy.utils import common_upscale

# Qwen3-VL vision placeholder. The image-capable tokenizer swaps the image_pad
# token for the embedded image when an image is supplied.
_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _prep_vl_image(image: torch.Tensor, max_megapixels: float = 0.0) -> torch.Tensor:
    """Return a ComfyUI IMAGE ([B,H,W,C], 0..1), RGB only.

    The Qwen-VL image processor (``process_qwen2vl_images``) already smart-resizes
    to its own ``[min_pixels, max_pixels=~12.8MP]`` range and snaps to the patch
    grid, so no downscaling is required for correctness. We only optionally cap
    the resolution to bound VRAM/compute, and we never upscale.

    ``max_megapixels <= 0`` passes the image through at native resolution.
    """
    image = image[:, :, :, :3]
    if max_megapixels and max_megapixels > 0:
        budget = int(max_megapixels * 1_000_000)
        samples = image.movedim(-1, 1)
        h, w = samples.shape[2], samples.shape[3]
        if w * h > budget:
            scale_by = math.sqrt(budget / float(w * h))
            new_w = max(1, round(w * scale_by))
            new_h = max(1, round(h * scale_by))
            samples = common_upscale(samples, new_w, new_h, "area", "disabled")
            image = samples.movedim(1, -1)
    return image


def _encode(clip, prompt: str, images: list):
    """Tokenize + encode, tolerating CLIPs that do not accept an ``images`` kwarg."""
    try:
        tokens = clip.tokenize(prompt, images=images)
    except TypeError:
        if images:
            logging.warning(
                "CRTP Ideogram4: this CLIP does not accept images; encoding text only."
            )
        tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def _append(base, addition):
    """Append ``addition`` conditioning onto ``base`` along the sequence dim.

    Mirrors the built-in ``ConditioningConcat`` semantics: the first cond tensor
    of ``addition`` is concatenated onto every cond tensor in ``base``. When
    ``base`` is ``None``/empty, ``addition`` is returned unchanged.
    """
    if not base:
        return addition
    if not addition:
        return base

    add_tensor = addition[0][0]
    if len(addition) > 1:
        logging.warning(
            "CRTP Ideogram4: addition conditioning has >1 entry; only the first is appended."
        )

    out = []
    for entry in base:
        base_tensor = entry[0]
        if base_tensor.shape[-1] != add_tensor.shape[-1]:
            raise ValueError(
                "CRTP Ideogram4: cannot append conditioning with feature dim "
                f"{add_tensor.shape[-1]} onto {base_tensor.shape[-1]}. Both prompts "
                "must come from the same Ideogram 4 CLIP."
            )
        merged = torch.cat((base_tensor, add_tensor), dim=1)
        out.append([merged, entry[1].copy()])
    return out


class CRTP_Ideogram4ImagePrompt:
    """Encode a single image (+ optional text) as Ideogram 4 CONDITIONING.

    Optionally appends the result to an incoming CONDITIONING so image prompts
    can be chained after text or other image prompts.
    """

    CATEGORY = "conditioning/ideogram4"
    FUNCTION = "encode"

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
            },
            "optional": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "max_megapixels": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 12.8,
                        "step": 0.1,
                        "tooltip": (
                            "Cap the image resolution (MP) before encoding to bound "
                            "VRAM/compute. 0 = native resolution; the Qwen-VL processor "
                            "still clamps to ~12.8 MP internally. Never upscales."
                        ),
                    },
                ),
                "conditioning": ("CONDITIONING",),
            },
        }

    def encode(self, clip, image, text="", max_megapixels=0.0, conditioning=None):
        vl_image = _prep_vl_image(image, max_megapixels)
        prompt = _VISION_PLACEHOLDER + (text or "")
        encoded = _encode(clip, prompt, [vl_image])
        return (_append(conditioning, encoded),)


class CRTP_Ideogram4TextPrompt:
    """Encode text as Ideogram 4 CONDITIONING, optionally appending to an input.

    Use this to insert a CLIP text prompt anywhere in an Ideogram 4 prompt chain.
    """

    CATEGORY = "conditioning/ideogram4"
    FUNCTION = "encode"

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "conditioning": ("CONDITIONING",),
            },
        }

    def encode(self, clip, text, conditioning=None):
        encoded = _encode(clip, text or "", [])
        return (_append(conditioning, encoded),)


class CRTP_ConditioningAppend:
    """Append ``conditioning_from`` onto ``conditioning_to`` (sequence concat).

    Generic helper to chain any two same-CLIP conditionings. Equivalent to the
    built-in ``ConditioningConcat`` but named for prompt-chaining workflows.
    """

    CATEGORY = "conditioning/ideogram4"
    FUNCTION = "append"

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
            },
        }

    def append(self, conditioning_to, conditioning_from):
        return (_append(conditioning_to, conditioning_from),)


NODE_CLASS_MAPPINGS = {
    "CRTP_Ideogram4ImagePrompt": CRTP_Ideogram4ImagePrompt,
    "CRTP_Ideogram4TextPrompt": CRTP_Ideogram4TextPrompt,
    "CRTP_ConditioningAppend": CRTP_ConditioningAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_Ideogram4ImagePrompt": "CRTP Ideogram4 Image Prompt (experimental)",
    "CRTP_Ideogram4TextPrompt": "CRTP Ideogram4 Text Prompt (experimental)",
    "CRTP_ConditioningAppend": "CRTP Conditioning Append",
}
