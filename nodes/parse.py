"""Parse strings into INT or FLOAT values.

Useful when an upstream node (text input, regex extract, file name, etc.)
produces a string that downstream numeric inputs need.
"""

from __future__ import annotations


class CRTP_ParseInt:
    """Parse a STRING into an INT.

    Supports decimal, hex (``0x``), octal (``0o``), and binary (``0b``)
    literals. Floats like ``"3.0"`` are accepted and truncated. On failure,
    ``default`` is returned.
    """

    CATEGORY = "utils/parse"
    FUNCTION = "parse"

    RETURN_TYPES = ("INT", "BOOLEAN")
    RETURN_NAMES = ("value", "ok")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "0", "multiline": False}),
                "default": ("INT", {"default": 0, "min": -2**31, "max": 2**31 - 1}),
            },
        }

    def parse(self, text, default):
        s = (text or "").strip()
        if not s:
            return (int(default), False)

        # Try integer literal (handles 0x.., 0o.., 0b.., decimal, +/-)
        try:
            return (int(s, 0), True)
        except (ValueError, TypeError):
            pass

        # Fall back to float-then-truncate (handles "3.0", "1e3")
        try:
            return (int(float(s)), True)
        except (ValueError, TypeError):
            return (int(default), False)


class CRTP_ParseFloat:
    """Parse a STRING into a FLOAT. On failure, ``default`` is returned."""

    CATEGORY = "utils/parse"
    FUNCTION = "parse"

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("value", "ok")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "0.0", "multiline": False}),
                "default": (
                    "FLOAT",
                    {"default": 0.0, "min": -1e30, "max": 1e30, "step": 0.0001},
                ),
            },
        }

    def parse(self, text, default):
        s = (text or "").strip()
        if not s:
            return (float(default), False)

        try:
            return (float(s), True)
        except (ValueError, TypeError):
            return (float(default), False)


NODE_CLASS_MAPPINGS = {
    "CRTP_ParseInt": CRTP_ParseInt,
    "CRTP_ParseFloat": CRTP_ParseFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_ParseInt": "CRTP Parse Int",
    "CRTP_ParseFloat": "CRTP Parse Float",
}
