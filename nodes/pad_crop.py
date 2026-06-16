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


def _make_padding_tuple(
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    pad_color: str,
) -> dict:
    """Build the dict that flows on a ``PADDING_TUPLE`` link."""
    return {
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
        "pad_color": str(pad_color),
    }


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


class CRTP_PadImageToDivisible:
    """Pad an IMAGE so its width and height are multiples of ``divisor``.

    Works on image and video frame batches (tensor shape ``[B, H, W, C]``).
    Outputs the padded image plus the four pad amounts so a downstream
    :class:`CropImageByPadding` node can reverse the operation after a model
    pass.
    """

    CATEGORY = "image/transform"
    FUNCTION = "pad"

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "PADDING_TUPLE")
    RETURN_NAMES = ("image", "pad_left", "pad_right", "pad_top", "pad_bottom", "padding")

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
            empty = _make_padding_tuple(0, 0, 0, 0, pad_color)
            return (image, 0, 0, 0, 0, empty)

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

        padding = _make_padding_tuple(pad_left, pad_right, pad_top, pad_bottom, pad_color)
        return (out, pad_left, pad_right, pad_top, pad_bottom, padding)


class CRTP_CropImageByPadding:
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
            "optional": {
                "padding": (
                    "PADDING_TUPLE",
                    {
                        "tooltip": (
                            "Optional bundled padding (from PadImageToDivisible or "
                            "PackPaddingTuple). When connected, overrides the four "
                            "individual pad_* inputs."
                        ),
                    },
                ),
            },
        }

    def crop(self, image, pad_left, pad_right, pad_top, pad_bottom, padding=None):
        if padding is not None:
            pad_left = int(padding.get("pad_left", pad_left))
            pad_right = int(padding.get("pad_right", pad_right))
            pad_top = int(padding.get("pad_top", pad_top))
            pad_bottom = int(padding.get("pad_bottom", pad_bottom))
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


class CRTP_ComputePaddingToDivisible:
    """Compute pad amounts to make ``width`` and ``height`` divisible by ``divisor``.

    Same math as :class:`PadImageToDivisible`, but operates purely on integers
    without touching an image. Useful for pre-computing target dimensions for
    empty latents, downstream resolution-aware nodes, etc.
    """

    CATEGORY = "image/transform"
    FUNCTION = "compute"

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "INT", "PADDING_TUPLE",
    )
    RETURN_NAMES = (
        "pad_left", "pad_right", "pad_top", "pad_bottom",
        "padded_width", "padded_height", "padding",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1280, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 720, "min": 1, "max": 16384}),
                "divisor": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Both width and height are rounded up to a multiple of this value.",
                    },
                ),
                "mode": (
                    ("symmetric", "asymmetric"),
                    {"default": "symmetric"},
                ),
                "asymmetric_side": (
                    ("end", "start"),
                    {
                        "default": "end",
                        "tooltip": "Only used when mode=asymmetric. end = bottom/right, start = top/left.",
                    },
                ),
                "pad_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Hex color stored in the PADDING_TUPLE output (no image is produced).",
                    },
                ),
            },
        }

    def compute(self, width, height, divisor, mode, asymmetric_side, pad_color):
        new_w = ((width + divisor - 1) // divisor) * divisor
        new_h = ((height + divisor - 1) // divisor) * divisor
        dw = new_w - width
        dh = new_h - height

        pad_left, pad_right = _split_pad(dw, mode, asymmetric_side)
        pad_top, pad_bottom = _split_pad(dh, mode, asymmetric_side)

        padding = _make_padding_tuple(pad_left, pad_right, pad_top, pad_bottom, pad_color)
        return (pad_left, pad_right, pad_top, pad_bottom, new_w, new_h, padding)


class CRTP_PackPaddingTuple:
    """Bundle four pad amounts and a pad color into a single PADDING_TUPLE."""

    CATEGORY = "image/transform"
    FUNCTION = "pack"

    RETURN_TYPES = ("PADDING_TUPLE",)
    RETURN_NAMES = ("padding",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "pad_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Hex color associated with the padding, e.g. #000000.",
                    },
                ),
            },
        }

    def pack(self, pad_left, pad_right, pad_top, pad_bottom, pad_color):
        return (
            _make_padding_tuple(pad_left, pad_right, pad_top, pad_bottom, pad_color),
        )


class CRTP_UnpackPaddingTuple:
    """Split a PADDING_TUPLE into its four pad amounts and pad color."""

    CATEGORY = "image/transform"
    FUNCTION = "unpack"

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("pad_left", "pad_right", "pad_top", "pad_bottom", "pad_color")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "padding": ("PADDING_TUPLE",),
            },
        }

    def unpack(self, padding):
        if not isinstance(padding, dict):
            raise ValueError(
                f"UnpackPaddingTuple: expected PADDING_TUPLE dict, got {type(padding).__name__}"
            )
        return (
            int(padding.get("pad_left", 0)),
            int(padding.get("pad_right", 0)),
            int(padding.get("pad_top", 0)),
            int(padding.get("pad_bottom", 0)),
            str(padding.get("pad_color", "#000000")),
        )


NODE_CLASS_MAPPINGS = {
    "CRTP_PadImageToDivisible": CRTP_PadImageToDivisible,
    "CRTP_CropImageByPadding": CRTP_CropImageByPadding,
    "CRTP_ComputePaddingToDivisible": CRTP_ComputePaddingToDivisible,
    "CRTP_PackPaddingTuple": CRTP_PackPaddingTuple,
    "CRTP_UnpackPaddingTuple": CRTP_UnpackPaddingTuple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_PadImageToDivisible": "CRTP Pad Image To Divisible",
    "CRTP_CropImageByPadding": "CRTP Crop Image By Padding",
    "CRTP_ComputePaddingToDivisible": "CRTP Compute Padding To Divisible",
    "CRTP_PackPaddingTuple": "CRTP Pack Padding Tuple",
    "CRTP_UnpackPaddingTuple": "CRTP Unpack Padding Tuple",
}
