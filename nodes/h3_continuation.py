"""Self-contained MiniMax H3 video-continuation helpers.

The nodes in this module use only ComfyUI core types and APIs. In particular,
they do not import ComfyUI_MiniMax_H3_Extender or ComfyUI-H3-Motion-Context.
The workflow uses the source tail as a native Ref2VA video reference and the
exact source boundary as a native first-frame keyframe. The boundary frame is
removed from the decoded result before it is saved and concatenated.
"""

from __future__ import annotations

import math

import torch

import comfy.utils
import node_helpers


FPS = 24
VALID_CONTEXT_LENGTHS = (5, 22, 39, 56)
BUILD = "crtp-h3-continuation-v2-native"


def _fit_frames(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Aspect-fit frames to the generation canvas with symmetric letterboxing."""
    if images is None or images.ndim != 4 or int(images.shape[0]) < 1:
        raise ValueError("CRTP H3 Continuation Tail: source video has no decoded frames.")

    source_h = int(images.shape[1])
    source_w = int(images.shape[2])
    scale = min(float(width) / source_w, float(height) / source_h)
    scaled_w = max(1, min(int(width), int(round(source_w * scale))))
    scaled_h = max(1, min(int(height), int(round(source_h * scale))))

    channels_first = images[..., :3].movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        channels_first, scaled_w, scaled_h, "lanczos", "disabled"
    ).movedim(1, -1)

    if scaled_w == int(width) and scaled_h == int(height):
        return resized

    output = torch.zeros(
        (int(resized.shape[0]), int(height), int(width), int(resized.shape[-1])),
        dtype=resized.dtype,
        device=resized.device,
    )
    left = (int(width) - scaled_w) // 2
    top = (int(height) - scaled_h) // 2
    output[:, top:top + scaled_h, left:left + scaled_w, :] = resized
    return output


def _align_h3_frame_count(frame_count: int) -> int:
    """Round upward to the 17k+5 frame grid required by MiniMax H3."""
    result = max(5, int(frame_count))
    return result + (5 - result % 17) % 17


class CRTP_H3ContinuationTail:
    """Resample the source tail to H3's 24 fps reference-video timeline."""

    CATEGORY = "CRTP/MiniMax H3"
    FUNCTION = "extract"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "tail_frames",
        "last_frame",
        "context_frames",
        "context_duration",
        "source_duration",
        "status",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "width": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 4096, "step": 32}),
                "context_length": ([str(v) for v in VALID_CONTEXT_LENGTHS], {"default": "22"}),
            }
        }

    def extract(self, images, source_fps, width, height, context_length):
        source_fps = float(source_fps)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("CRTP H3 Continuation Tail: source FPS must be positive.")

        source_count = int(images.shape[0])
        requested = int(context_length)
        if requested not in VALID_CONTEXT_LENGTHS:
            raise ValueError(f"CRTP H3 Continuation Tail: unsupported context length {requested}.")

        source_duration = source_count / source_fps
        normalized_count = max(1, int(round(source_duration * FPS)))
        if normalized_count < requested:
            raise ValueError(
                "CRTP H3 Continuation Tail: the normalized source has "
                f"{normalized_count} frames, but {requested} context frames were requested. "
                "Choose a shorter context window or upload a longer video."
            )

        target_positions = torch.arange(
            normalized_count - requested,
            normalized_count,
            device=images.device,
            dtype=torch.float64,
        )
        indices = torch.round(target_positions * source_fps / FPS).to(torch.long)
        indices.clamp_(0, source_count - 1)
        indices[-1] = source_count - 1

        tail = _fit_frames(images.index_select(0, indices), int(width), int(height))
        context_duration = requested / float(FPS)
        status = (
            f"{source_count} frames at {source_fps:g} fps -> final {requested} frames "
            f"at {FPS} fps, {int(width)}x{int(height)} native Ref2VA video reference"
        )
        return (
            tail,
            tail[-1:],
            requested,
            context_duration,
            float(source_duration),
            status,
        )


class CRTP_H3ContinuationLength:
    """Convert requested new duration to an H3 length including one anchor frame."""

    CATEGORY = "CRTP/MiniMax H3"
    FUNCTION = "calculate"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("model_frame_count", "new_frame_count", "actual_new_duration", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 150.0, "step": 0.5}),
            }
        }

    def calculate(self, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("CRTP H3 Continuation Length: duration must be positive.")
        requested_new_frames = max(1, int(round(seconds * FPS)))
        model_frames = _align_h3_frame_count(requested_new_frames + 1)
        new_frames = model_frames - 1
        actual_duration = new_frames / float(FPS)
        return (
            model_frames,
            new_frames,
            actual_duration,
            f"{seconds:g}s requested -> {model_frames} model frames; "
            f"trim 1 boundary frame -> {new_frames} new frames ({actual_duration:.3f}s)",
        )


class CRTP_H3AddFirstFrameAnchor:
    """Attach an exact frame-zero keyframe to native Ref2VA conditioning."""

    CATEGORY = "CRTP/MiniMax H3"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "first_frame": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 4096, "step": 32}),
                "frame_count": ("INT", {"default": 243, "min": 5, "max": 3600}),
            }
        }

    def apply(self, conditioning, first_frame, vae, width, height, frame_count):
        frame_count = int(frame_count)
        if frame_count % 17 != 5:
            raise ValueError(
                "CRTP H3 First Frame Anchor: frame_count must be on H3's 17k+5 grid."
            )
        image = _fit_frames(first_frame[:1], int(width), int(height))
        keyframe = {"resolved_frame_index": 0, "latent": vae.encode(image)}
        anchored = node_helpers.conditioning_set_values(
            conditioning,
            {
                "minimax_keyframes": [keyframe],
                "minimax_frame_count": frame_count,
            },
        )
        return anchored, f"native MiniMax H3 first-frame anchor at frame 0/{frame_count - 1}"


class CRTP_H3TrimContinuationAV:
    """Remove internal boundary frames and the time-aligned decoded audio prefix."""

    CATEGORY = "CRTP/MiniMax H3"
    FUNCTION = "trim"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "trim_frames": ("INT", {"default": 1, "min": 0, "max": 56}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            }
        }

    def trim(self, images, audio, trim_frames, fps):
        trim_frames = int(trim_frames)
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("CRTP H3 Trim Continuation AV: FPS must be positive.")
        if trim_frames >= int(images.shape[0]):
            raise ValueError(
                "CRTP H3 Trim Continuation AV: trim would remove every generated frame."
            )

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        trim_samples = int(round(trim_frames * sample_rate / fps))
        if trim_samples >= int(waveform.shape[-1]):
            raise ValueError(
                "CRTP H3 Trim Continuation AV: trim would remove all generated audio."
            )
        trimmed_audio = {**audio, "waveform": waveform[..., trim_samples:]}
        remaining_frames = int(images.shape[0]) - trim_frames
        return (
            images[trim_frames:],
            trimmed_audio,
            f"trimmed {trim_frames} boundary frame(s) and {trim_samples} audio samples; "
            f"{remaining_frames} generated frames remain",
        )


NODE_CLASS_MAPPINGS = {
    "CRTP_H3ContinuationTail": CRTP_H3ContinuationTail,
    "CRTP_H3ContinuationLength": CRTP_H3ContinuationLength,
    "CRTP_H3AddFirstFrameAnchor": CRTP_H3AddFirstFrameAnchor,
    "CRTP_H3TrimContinuationAV": CRTP_H3TrimContinuationAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_H3ContinuationTail": "CRTP H3 Continuation Tail (24 fps)",
    "CRTP_H3ContinuationLength": "CRTP H3 Continuation Length",
    "CRTP_H3AddFirstFrameAnchor": "CRTP H3 Add First-Frame Anchor",
    "CRTP_H3TrimContinuationAV": "CRTP H3 Trim Continuation AV",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
