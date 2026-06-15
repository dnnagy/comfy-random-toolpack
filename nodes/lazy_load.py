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


class LazyLoadImage:
    """Load an image from the input directory, returning None if it is missing.

    Drop-in alternative to the built-in ``Load Image`` node that never aborts
    the graph when the file is absent. Outputs ``image`` (IMAGE), ``mask``
    (MASK), and ``loaded`` (BOOLEAN).
    """

    CATEGORY = "image"
    FUNCTION = "load_image"

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "loaded")

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
            ]
            files = folder_paths.filter_files_content_types(files, ["image"])
        except Exception:
            files = []
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    def load_image(self, image):
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

    @classmethod
    def IS_CHANGED(cls, image):
        path = _resolve_path(image)
        if path is None:
            # Missing files compare equal so the node is not forced to re-run.
            return "missing:{}".format(image)
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        # Always pass: this bypasses both the file-existence check and the
        # implicit combo-membership check, so a missing/stale filename does not
        # abort the run. Missing files are handled gracefully in load_image.
        return True


class LazyLoadAudio:
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
        input_dir = folder_paths.get_input_directory()
        try:
            os.makedirs(input_dir, exist_ok=True)
            files = folder_paths.filter_files_content_types(
                os.listdir(input_dir), ["audio", "video"]
            )
        except Exception:
            files = []
        return {
            "required": {
                "audio": (sorted(files), {"audio_upload": True}),
            },
        }

    def load_audio(self, audio):
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
        # Always pass; missing files are handled gracefully in load_audio.
        return True


NODE_CLASS_MAPPINGS = {
    "LazyLoadImage": LazyLoadImage,
    "LazyLoadAudio": LazyLoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LazyLoadImage": "Lazy Load Image",
    "LazyLoadAudio": "Lazy Load Audio",
}
