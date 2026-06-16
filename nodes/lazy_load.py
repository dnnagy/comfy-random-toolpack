"""Fault-tolerant ("lazy") versions of the built-in Load Image / Load Audio nodes.

The stock ComfyUI ``LoadImage`` / ``LoadAudio`` nodes define ``VALIDATE_INPUTS``
(``validate_inputs``) that returns an error string when the referenced file is
not present in the input directory. ComfyUI treats that as a hard validation
failure and refuses to run the graph. This is exactly what happens when a
workflow is ported to another machine where the input file is missing.

These nodes instead:

* override ``VALIDATE_INPUTS`` to always pass (this also bypasses the implicit
  "value must be one of the combo options" check, so a stale filename from
  another machine does not abort the run),
* return ``None`` for the data output(s) when the file cannot be loaded,
* expose a ``loaded`` BOOLEAN output so downstream nodes can branch on success.

Combine with ``LazyLatentFallback`` (also in this pack) to skip expensive
branches when ``loaded`` is ``False``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

import folder_paths


def _resolve_path(name: str):
    """Return an existing filesystem path for an annotated filename, or None."""
    if not name:
        return None
    try:
        path = folder_paths.get_annotated_filepath(name)
    except Exception:
        return None
    if path and os.path.exists(path):
        return path
    return None


def _list_media_files(content_types: list[str]) -> list[str]:
    input_dir = folder_paths.get_input_directory()
    try:
        os.makedirs(input_dir, exist_ok=True)
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return sorted(folder_paths.filter_files_content_types(files, content_types))
    except Exception:
        return []


def _load_image_impl(image):
    from PIL import Image, ImageOps, ImageSequence

    path = _resolve_path(image)
    if path is None:
        return (None, None, False)

    try:
        img = Image.open(path)

        output_images = []
        output_masks = []
        w, h = None, None

        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            rgb = frame.convert("RGB")

            if len(output_images) == 0:
                w, h = rgb.size

            if rgb.size[0] != w or rgb.size[1] != h:
                continue

            arr = np.array(rgb).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)[None,]

            if "A" in frame.getbands():
                mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32)

            output_images.append(tensor)
            output_masks.append(mask.unsqueeze(0))

        if not output_images:
            return (None, None, False)

        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
        return (output_image, output_mask, True)
    except Exception:
        return (None, None, False)


def _load_audio_impl(audio):
    path = _resolve_path(audio)
    if path is None:
        return (None, False)

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
        result = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (result, True)
    except Exception:
        return (None, False)


def _load_video_impl(file):
    path = _resolve_path(file)
    if path is None:
        return (None, False)

    try:
        from comfy_api.latest import InputImpl

        return (InputImpl.VideoFromFile(path), True)
    except Exception:
        return (None, False)


class CRTP_LazyLoadImageUpload:
    CATEGORY = "image"
    FUNCTION = "load_image"

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_list_media_files(["image"]), {"image_upload": True}),
            },
        }

    def load_image(self, image):
        return _load_image_impl(image)

    @classmethod
    def IS_CHANGED(cls, image):
        path = _resolve_path(image)
        if path is None:
            return "missing:{}".format(image)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True


class CRTP_LazyLoadImageSelect:
    CATEGORY = "image"
    FUNCTION = "load_image"

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_list_media_files(["image"]),),
            },
        }

    def load_image(self, image):
        return _load_image_impl(image)

    @classmethod
    def IS_CHANGED(cls, image):
        path = _resolve_path(image)
        if path is None:
            return "missing:{}".format(image)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True


class CRTP_LazyLoadAudioUpload:
    """Load audio from the input directory, returning None if it is missing.

    Drop-in alternative to the built-in ``Load Audio`` node that never aborts
    the graph when the file is absent. Outputs ``audio`` (AUDIO) and ``loaded``
    (BOOLEAN).
    """

    CATEGORY = "audio"
    FUNCTION = "load_audio"

    RETURN_TYPES = ("AUDIO", "BOOLEAN")
    RETURN_NAMES = ("audio", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (_list_media_files(["audio", "video"]), {"audio_upload": True}),
            },
        }

    def load_audio(self, audio):
        return _load_audio_impl(audio)

    @classmethod
    def IS_CHANGED(cls, audio):
        path = _resolve_path(audio)
        if path is None:
            return "missing:{}".format(audio)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        return True

class CRTP_LazyLoadAudioSelect:
    CATEGORY = "audio"
    FUNCTION = "load_audio"

    RETURN_TYPES = ("AUDIO", "BOOLEAN")
    RETURN_NAMES = ("audio", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (_list_media_files(["audio", "video"]),),
            },
        }

    def load_audio(self, audio):
        return _load_audio_impl(audio)

    @classmethod
    def IS_CHANGED(cls, audio):
        path = _resolve_path(audio)
        if path is None:
            return "missing:{}".format(audio)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        return True


class CRTP_LazyLoadVideoUpload:
    """Load a video from the input directory, returning None if it is missing.

    Drop-in alternative to the built-in ``Load Video`` node that never aborts
    the graph when the file is absent. Outputs ``video`` (VIDEO) and ``loaded``
    (BOOLEAN).
    """

    CATEGORY = "video"
    FUNCTION = "load_video"

    RETURN_TYPES = ("VIDEO", "BOOLEAN")
    RETURN_NAMES = ("video", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_list_media_files(["video"]), {"video_upload": True}),
            },
        }

    def load_video(self, file):
        return _load_video_impl(file)

    @classmethod
    def IS_CHANGED(cls, file):
        path = _resolve_path(file)
        if path is None:
            return "missing:{}".format(file)
        # Match the built-in node: use mtime to avoid rehashing large files.
        return os.path.getmtime(path)

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        return True

class CRTP_LazyLoadVideoSelect:
    CATEGORY = "video"
    FUNCTION = "load_video"

    RETURN_TYPES = ("VIDEO", "BOOLEAN")
    RETURN_NAMES = ("video", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_list_media_files(["video"]),),
            },
        }

    def load_video(self, file):
        return _load_video_impl(file)

    @classmethod
    def IS_CHANGED(cls, file):
        path = _resolve_path(file)
        if path is None:
            return "missing:{}".format(file)
        return os.path.getmtime(path)

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        return True


NODE_CLASS_MAPPINGS = {
    "CRTP_LazyLoadImageUpload": CRTP_LazyLoadImageUpload,
    "CRTP_LazyLoadImageSelect": CRTP_LazyLoadImageSelect,
    "CRTP_LazyLoadAudioUpload": CRTP_LazyLoadAudioUpload,
    "CRTP_LazyLoadAudioSelect": CRTP_LazyLoadAudioSelect,
    "CRTP_LazyLoadVideoUpload": CRTP_LazyLoadVideoUpload,
    "CRTP_LazyLoadVideoSelect": CRTP_LazyLoadVideoSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_LazyLoadImageUpload": "CRTP Lazy Load Image (Upload)",
    "CRTP_LazyLoadImageSelect": "CRTP Lazy Load Image (Select)",
    "CRTP_LazyLoadAudioUpload": "CRTP Lazy Load Audio (Upload)",
    "CRTP_LazyLoadAudioSelect": "CRTP Lazy Load Audio (Select)",
    "CRTP_LazyLoadVideoUpload": "CRTP Lazy Load Video (Upload)",
    "CRTP_LazyLoadVideoSelect": "CRTP Lazy Load Video (Select)",
}
