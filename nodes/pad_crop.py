"""Pad / crop helpers for image and video tensors.

These nodes are intended for models that require spatial dimensions to be a
multiple of some divisor (e.g. LTX-Video 2.3 requires both width and height to
be divisible by 32). The typical pattern is:

    IMAGE/VIDEO --> PadImageToDivisible --> model --> CropImageByPadding --> result

ComfyUI ``IMAGE`` tensors have shape ``[B, H, W, C]`` with float values in
``[0, 1]``. This same layout is used for video frame batches, so the same
nodes work for both single images and video sequences.
"""

from __future__ import annotations

import torch


def _parse_hex_color(value: str) -> tuple[float, float, float]:
    """Parse ``#RRGGBB`` or ``RRGGBB`` into ``(r, g, b)`` floats in [0, 1]."""
    s = value.strip().lstrip("#")
    if len(s) == 3:  # short form like "fff"
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {value!r}. Expected '#RRGGBB'.")
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError as e:
        raise ValueError(f"Invalid hex color: {value!r}.") from e
    return r, g, b


def _split_pad(total: int, mode: str, side: str) -> tuple[int, int]:
    """Return ``(before, after)`` padding amounts.

    For ``symmetric`` the extra pixel (when ``total`` is odd) goes to the
    ``after`` side. For ``asymmetric`` everything goes to one side, controlled
    by ``side`` which must be ``"end"`` (bottom/right) or ``"start"``
    (top/left).
    """
    if total <= 0:
        return 0, 0
    if mode == "symmetric":
        before = total // 2
        after = total - before
        return before, after
    if mode == "asymmetric":
        if side == "end":
            return 0, total
        if side == "start":
            return total, 0
        raise ValueError(f"Unknown asymmetric side: {side!r}")
    raise ValueError(f"Unknown padding mode: {mode!r}")


class PadImageToDivisible:
    """Pad an IMAGE so its width and height are multiples of ``divisor``.

    Works on image and video frame batches (tensor shape ``[B, H, W, C]``).
    Outputs the padded image plus the four pad amounts so a downstream
    :class:`CropImageByPadding` node can reverse the operation after a model
    pass.
    """

    CATEGORY = "image/transform"
    FUNCTION = "pad"

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "pad_left", "pad_right", "pad_top", "pad_bottom")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "divisor": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Both width and height will be padded up to a multiple of this value.",
                    },
                ),
                "mode": (
                    ("symmetric", "asymmetric"),
                    {
                        "default": "symmetric",
                        "tooltip": (
                            "symmetric: split padding between both sides "
                            "(e.g. 720 -> 736 becomes 8 top + 8 bottom). "
                            "asymmetric: put all padding on one side."
                        ),
                    },
                ),
                "asymmetric_side": (
                    ("end", "start"),
                    {
                        "default": "end",
                        "tooltip": (
                            "Only used when mode=asymmetric. "
                            "end = pad bottom and right; start = pad top and left."
                        ),
                    },
                ),
                "pad_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Hex color for the padding, e.g. #000000 (black) or #ffffff (white).",
                    },
                ),
            },
        }

    def pad(self, image, divisor, mode, asymmetric_side, pad_color):
        if image.ndim != 4:
            raise ValueError(
                f"PadImageToDivisible expects [B,H,W,C] tensor, got shape {tuple(image.shape)}"
            )

        b, h, w, c = image.shape
        new_h = ((h + divisor - 1) // divisor) * divisor
        new_w = ((w + divisor - 1) // divisor) * divisor
        dh = new_h - h
        dw = new_w - w

        pad_top, pad_bottom = _split_pad(dh, mode, asymmetric_side)
        pad_left, pad_right = _split_pad(dw, mode, asymmetric_side)

        if dh == 0 and dw == 0:
            return (image, 0, 0, 0, 0)

        r, g, b_col = _parse_hex_color(pad_color)
        # Build padding color matching the channel count. For non-RGB inputs
        # (e.g. mask-like single-channel) repeat the red component.
        if c >= 3:
            color = [r, g, b_col] + [0.0] * (c - 3)
        else:
            color = [r] * c
        color_t = torch.tensor(color, dtype=image.dtype, device=image.device)

        out = torch.empty(
            (b, new_h, new_w, c), dtype=image.dtype, device=image.device
        )
        out[...] = color_t  # broadcast fill
        out[:, pad_top : pad_top + h, pad_left : pad_left + w, :] = image

        return (out, pad_left, pad_right, pad_top, pad_bottom)


class CropImageByPadding:
    """Crop an IMAGE by removing the given pad amounts from each side.

    Pairs with :class:`PadImageToDivisible` to restore the original spatial
    dimensions after a model pass. Works on image and video frame batches
    (tensor shape ``[B, H, W, C]``).
    """

    CATEGORY = "image/transform"
    FUNCTION = "crop"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 8192}),
            },
        }

    def crop(self, image, pad_left, pad_right, pad_top, pad_bottom):
        if image.ndim != 4:
            raise ValueError(
                f"CropImageByPadding expects [B,H,W,C] tensor, got shape {tuple(image.shape)}"
            )

        _, h, w, _ = image.shape
        if pad_top + pad_bottom >= h or pad_left + pad_right >= w:
            raise ValueError(
                "CropImageByPadding: requested crop removes the entire image "
                f"(image {w}x{h}, crop L{pad_left} R{pad_right} T{pad_top} B{pad_bottom})."
            )

        bottom = h - pad_bottom if pad_bottom > 0 else h
        right = w - pad_right if pad_right > 0 else w
        return (image[:, pad_top:bottom, pad_left:right, :].contiguous(),)


NODE_CLASS_MAPPINGS = {
    "PadImageToDivisible": PadImageToDivisible,
    "CropImageByPadding": CropImageByPadding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageToDivisible": "Pad Image To Divisible",
    "CropImageByPadding": "Crop Image By Padding",
}
