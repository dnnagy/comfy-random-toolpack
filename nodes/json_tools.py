"""JSON manipulation nodes for text/data workflows.

ComfyUI has no native object type, so these nodes introduce a lightweight
``JSON`` socket type that carries a parsed Python object (dict/list/scalar)
between nodes:

* ``CRTP_JSONParse``     STRING -> JSON   (parse)
* ``CRTP_JSONStringify`` JSON   -> STRING (stringify, optional indent)
* ``CRTP_JSONMinify``    STRING -> STRING (collapse whitespace)
* ``CRTP_JSONPrettify``  STRING -> STRING (indent)
* ``CRTP_JSONSet``       JSON   -> JSON   (obj[key] = value)
* ``CRTP_JSONDeleteKey`` JSON   -> JSON   (remove key)
* ``CRTP_JSONGet``       JSON   -> value  (return obj[key])

Mutating nodes operate on a deep copy so cached upstream outputs are never
modified in place.
"""

from __future__ import annotations

import copy
import json as _json


class _AnyType(str):
    def __ne__(self, _value):
        return False


_ANY = _AnyType("*")
JSON_TYPE = "JSON"


def _as_obj(data):
    """Accept either a parsed object or a JSON string and return the object."""
    if isinstance(data, str):
        return _json.loads(data)
    return data


def _coerce_value(value: str, value_is_json: bool):
    """Interpret a STRING widget value as JSON (number/bool/obj/...) or raw text."""
    if not value_is_json:
        return value
    try:
        return _json.loads(value)
    except (ValueError, TypeError):
        return value


class CRTP_JSONParse:
    """Parse a JSON STRING into a JSON object. On failure returns ``ok=False``."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = (JSON_TYPE, "BOOLEAN", "STRING")
    RETURN_NAMES = ("json", "ok", "error")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    def run(self, text):
        try:
            return (_json.loads(text or ""), True, "")
        except (ValueError, TypeError) as e:
            return (None, False, str(e))


class CRTP_JSONStringify:
    """Serialize a JSON object to a STRING. ``indent=0`` produces compact output."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (JSON_TYPE,),
                "indent": ("INT", {"default": 0, "min": 0, "max": 16}),
                "sort_keys": ("BOOLEAN", {"default": False}),
                "ensure_ascii": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, json, indent, sort_keys, ensure_ascii):
        if indent and indent > 0:
            text = _json.dumps(
                json, indent=int(indent), sort_keys=sort_keys, ensure_ascii=ensure_ascii
            )
        else:
            text = _json.dumps(
                json, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=ensure_ascii
            )
        return (text,)


class CRTP_JSONMinify:
    """Minify a JSON STRING (remove all insignificant whitespace)."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "ok")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "{}"}),
                "sort_keys": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, text, sort_keys):
        try:
            obj = _json.loads(text or "")
        except (ValueError, TypeError):
            return (text, False)
        return (_json.dumps(obj, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=False), True)


class CRTP_JSONPrettify:
    """Prettify a JSON STRING with the given indent."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "ok")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "{}"}),
                "indent": ("INT", {"default": 2, "min": 1, "max": 16}),
                "sort_keys": ("BOOLEAN", {"default": False}),
            },
        }

    def run(self, text, indent, sort_keys):
        try:
            obj = _json.loads(text or "")
        except (ValueError, TypeError):
            return (text, False)
        return (_json.dumps(obj, indent=int(indent), sort_keys=sort_keys, ensure_ascii=False), True)


class CRTP_JSONSet:
    """Set ``obj[key] = value`` on a JSON object and return the modified object."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = (JSON_TYPE,)
    RETURN_NAMES = ("json",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (JSON_TYPE,),
                "key": ("STRING", {"default": "foo"}),
                "value": ("STRING", {"multiline": True, "default": ""}),
                "value_is_json": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, json, key, value, value_is_json):
        obj = copy.deepcopy(_as_obj(json))
        if not isinstance(obj, dict):
            raise ValueError("CRTP_JSONSet: target JSON is not an object/dict.")
        obj[key] = _coerce_value(value, value_is_json)
        return (obj,)


class CRTP_JSONDeleteKey:
    """Remove ``key`` from a JSON object entirely."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = (JSON_TYPE, "BOOLEAN")
    RETURN_NAMES = ("json", "removed")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (JSON_TYPE,),
                "key": ("STRING", {"default": "foo"}),
            },
        }

    def run(self, json, key):
        obj = copy.deepcopy(_as_obj(json))
        removed = isinstance(obj, dict) and key in obj
        if removed:
            del obj[key]
        return (obj, removed)


class CRTP_JSONGet:
    """Return ``obj[key]`` as a JSON value and as a STRING, plus a ``found`` flag."""

    CATEGORY = "utils/json"
    FUNCTION = "run"

    RETURN_TYPES = (_ANY, "STRING", "BOOLEAN")
    RETURN_NAMES = ("value", "value_str", "found")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (JSON_TYPE,),
                "key": ("STRING", {"default": "foo"}),
            },
            "optional": {
                "default": ("STRING", {"default": ""}),
            },
        }

    def run(self, json, key, default=""):
        obj = _as_obj(json)
        if isinstance(obj, dict) and key in obj:
            value = obj[key]
            found = True
        else:
            value = default
            found = False

        if isinstance(value, str):
            value_str = value
        else:
            value_str = _json.dumps(value, ensure_ascii=False)
        return (value, value_str, found)


NODE_CLASS_MAPPINGS = {
    "CRTP_JSONParse": CRTP_JSONParse,
    "CRTP_JSONStringify": CRTP_JSONStringify,
    "CRTP_JSONMinify": CRTP_JSONMinify,
    "CRTP_JSONPrettify": CRTP_JSONPrettify,
    "CRTP_JSONSet": CRTP_JSONSet,
    "CRTP_JSONDeleteKey": CRTP_JSONDeleteKey,
    "CRTP_JSONGet": CRTP_JSONGet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_JSONParse": "CRTP JSON Parse",
    "CRTP_JSONStringify": "CRTP JSON Stringify",
    "CRTP_JSONMinify": "CRTP JSON Minify",
    "CRTP_JSONPrettify": "CRTP JSON Prettify",
    "CRTP_JSONSet": "CRTP JSON Set Key",
    "CRTP_JSONDeleteKey": "CRTP JSON Delete Key",
    "CRTP_JSONGet": "CRTP JSON Get Key",
}
