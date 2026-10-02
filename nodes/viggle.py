"""Viggle-Animate helpers using native ComfyUI MiniMax H3 support.

Conditioning geometry and reference packing adapted from Saganaki22's
ComfyUI-Viggle-Animate-H3 1.3.2 (Apache-2.0). See licenses/viggle-NOTICE.md.
No dependency on that custom-node package is required.
"""

from __future__ import annotations

import math
import os
import re

import torch

import folder_paths


FPS = 24
MAX_FRAMES = 362
MAX_DECODE_PIXELS = 512_000_000
DEFAULT_RESOLUTION = "864x480 (landscape, 0.4 MP)"
folder_paths.add_model_folder_path("text_cond", os.path.join(folder_paths.models_dir, "text_cond"))


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"Viggle-Animate: {name} must be an integer.")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"Viggle-Animate: {name} must be between {minimum} and {maximum}.")
    return value


def _images(images, name):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or min(images.shape[:3]) < 1 or images.shape[-1] < 3:
        raise ValueError(f"Viggle-Animate: {name} must contain nonempty RGB image frames.")


def _aspect(width, height):
    if width <= 0 or height <= 0 or not 0.25 <= width / height <= 4:
        raise ValueError("Viggle-Animate: aspect ratio must be between 1:4 and 4:1.")


def _canvas(width, height):
    width = _integer(width, "width", 32, 1536)
    height = _integer(height, "height", 32, 1536)
    if width % 32 or height % 32 or width * height > 1024 * 1024:
        raise ValueError("Viggle-Animate: resolution must use multiples of 32 and at most 1,048,576 pixels.")
    _aspect(width, height)
    return width, height


def _reference_size(width, height, short_edge, max_pixels=None):
    """Upstream short-edge resize, optionally area capped, then round to 32."""
    _aspect(width, height)
    scale = short_edge / min(width, height)
    w, h = width * scale, height * scale
    if max_pixels is not None and w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        w, h = w * scale, h * scale
    return max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)


def _generation_frames(count):
    count = max(5, int(count))
    return count + (5 - count) % 17


def _text_conditioning(blob):
    try:
        embeds, tags = blob["prompt_embeds"], blob["text_token_tags"]
    except (KeyError, TypeError):
        raise ValueError("Viggle-Animate: text conditioning requires prompt_embeds and text_token_tags.") from None
    if not isinstance(embeds, torch.Tensor) or tuple(embeds.shape) != (1, 362, 5120) or not embeds.is_floating_point():
        raise ValueError("Viggle-Animate: expected floating-point prompt_embeds with shape [1, 362, 5120].")
    if not isinstance(tags, torch.Tensor) or tuple(tags.shape) != (362,) or tags.dtype != torch.int64:
        raise ValueError("Viggle-Animate: expected int64 text_token_tags with shape [362].")
    return embeds, tags


def _trim_audio(audio, duration):
    if audio is None:
        return None
    try:
        waveform = audio["waveform"]
        rate = _integer(audio["sample_rate"], "audio sample rate", 1, 768000)
    except (KeyError, TypeError):
        raise ValueError("Viggle-Animate: invalid source audio.") from None
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or min(waveform.shape[:2]) < 1:
        raise ValueError("Viggle-Animate: expected audio waveform [batch, channels, samples].")
    count = int(round(duration * rate))
    result = waveform[:1, :, :count]
    if result.shape[-1] < count:
        result = torch.nn.functional.pad(result, (0, count - result.shape[-1]))
    return {"waveform": result, "sample_rate": rate}


def _presentation_timeline(video, source_fps, max_source_frames):
    """Read frame timestamps without materializing full-resolution image arrays.

    Native get_components discards PTS. A bounded timing pass lets us retain
    its color/rotation/audio handling while correctly resampling VFR footage.
    """
    import av

    if not hasattr(video, "get_active_trim_window"):
        raise ValueError("Viggle-Animate: use native LoadVideo for timestamp-aware video preparation.")
    start_time, duration = video.get_active_trim_window()
    source = video.get_stream_source()
    if hasattr(source, "seek"):
        source.seek(0)
    timestamps = []
    last_duration = 1 / source_fps
    following_timestamp = None
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("Viggle-Animate: input has no video stream.")
        stream = container.streams.video[0]
        start_pts = int(start_time / stream.time_base)
        end_pts = int((start_time + duration) / stream.time_base)
        if start_pts:
            container.seek(start_pts, stream=stream)
        done = False
        for packet in container.demux(stream):
            if done:
                break
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                if frame.pts is None:
                    raise ValueError("Viggle-Animate: source frames need valid presentation timestamps.")
                if frame.pts < start_pts:
                    continue
                if duration and frame.pts >= end_pts:
                    following_timestamp = float(frame.pts * stream.time_base) - start_time
                    done = True
                    break
                timestamp = float(frame.pts * stream.time_base) - start_time
                if timestamps and timestamp <= timestamps[-1]:
                    raise ValueError("Viggle-Animate: source presentation timestamps must be strictly increasing.")
                timestamps.append(timestamp)
                if len(timestamps) > max_source_frames:
                    raise ValueError("Viggle-Animate: actual source frames exceed the 512-million-pixel decode budget; resize or shorten the clip.")
                frame_duration = getattr(frame, "duration", 0)
                if frame_duration:
                    last_duration = float(frame_duration * stream.time_base)
                elif len(timestamps) > 1:
                    last_duration = timestamps[-1] - timestamps[-2]
    if not timestamps:
        raise ValueError("Viggle-Animate: the selected clip contains no video frames.")
    # A VFR frame stays on screen until the next presentation timestamp. The
    # codec's nominal frame.duration can be shorter than that interval.
    video_end = following_timestamp if following_timestamp is not None else timestamps[-1] + last_duration
    return torch.tensor(timestamps, dtype=torch.float64), video_end


def _aligned_source_audio(video, duration):
    """Decode source audio by its own time base, preserving offsets and gaps."""
    import av

    start_time, _ = video.get_active_trim_window()
    source = video.get_stream_source()
    if hasattr(source, "seek"):
        source.seek(0)
    with av.open(source, mode="r") as container:
        stream = next((s for s in reversed(container.streams.audio) if s.codec_context is not None), None)
        if stream is None:
            return None
        if start_time:
            container.seek(int(start_time / stream.time_base), stream=stream)
        resampler = av.AudioResampler(format="fltp")
        result = None
        sample_rate = None
        done = False
        for packet in container.demux(stream):
            if done:
                break
            for decoded in packet.decode():
                for frame in resampler.resample(decoded):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError("Viggle-Animate: audio frames need valid timestamps.")
                    rate = int(frame.sample_rate)
                    if sample_rate is None:
                        sample_rate = _integer(rate, "audio sample rate", 1, 768000)
                        result = torch.zeros(1, frame.layout.nb_channels, round(duration * rate))
                    if rate != sample_rate or frame.layout.nb_channels != result.shape[1]:
                        raise ValueError("Viggle-Animate: audio format changes within the selected clip.")
                    offset = round((float(frame.pts * frame.time_base) - start_time) * rate)
                    if offset >= result.shape[-1]:
                        done = True
                        break
                    start = max(0, offset)
                    stop = min(result.shape[-1], offset + frame.samples)
                    if stop > start:
                        array = frame.to_ndarray()
                        result[0, :, start:stop] = torch.from_numpy(array[:, start - offset:stop - offset].copy())
        return None if result is None else {"waveform": result, "sample_rate": sample_rate}


class CRTP_ViggleResolution:
    CATEGORY = "CRTP/Viggle-Animate"
    FUNCTION = "parse"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"resolution": ("STRING", {"default": DEFAULT_RESOLUTION})}}

    def parse(self, resolution):
        match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)(?:\s+\([^\n]*\))?\s*", resolution)
        if not match:
            raise ValueError("Viggle-Animate: resolution must be WIDTHxHEIGHT, optionally followed by a label in parentheses.")
        return _canvas(int(match[1]), int(match[2]))


class CRTP_VigglePrepareVideo:
    """Bound the lazy VIDEO before decoding and sample its timeline at 24 fps."""

    CATEGORY = "CRTP/Viggle-Animate"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "frame_count", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "max_frames": ("INT", {"default": 124, "min": 5, "max": MAX_FRAMES, "step": 1}),
            "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
            "preserve_audio": ("BOOLEAN", {"default": True}),
        }}

    def prepare(self, video, max_frames=124, start_time=0.0, preserve_audio=True):
        max_frames = _integer(max_frames, "max_frames", 5, MAX_FRAMES)
        start_time = float(start_time)
        if not math.isfinite(start_time) or start_time < 0:
            raise ValueError("Viggle-Animate: start_time must be finite and nonnegative.")
        if video is None:
            raise ValueError("Viggle-Animate: a driving video is required.")
        source_fps = float(video.get_frame_rate())
        if not math.isfinite(source_fps) or not 0 < source_fps <= 240:
            raise ValueError("Viggle-Animate: source FPS must be positive and at most 240.")
        source_width, source_height = video.get_dimensions()
        _aspect(source_width, source_height)
        # Native ComfyUI VideoFromFile implements this as a lazy bounded view.
        # Do not decode the original VIDEO and then slice the image tensor.
        clipped = video.as_trimmed(start_time, max_frames / FPS, strict_duration=False)
        if clipped is None:
            raise ValueError("Viggle-Animate: start_time is outside the driving video.")
        duration = min(float(clipped.get_duration()), max_frames / FPS)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Viggle-Animate: selected video duration must be positive.")
        decode_frames = math.ceil(duration * source_fps) + 1
        if source_width * source_height * decode_frames > MAX_DECODE_PIXELS:
            raise ValueError(
                "Viggle-Animate: selected source exceeds the 512-million-pixel decode budget. "
                "Resize the driving video, reduce its FPS, or select fewer frames before running."
            )
        timestamps, video_end = _presentation_timeline(
            clipped, source_fps, MAX_DECODE_PIXELS // (source_width * source_height)
        )
        components = clipped.get_components()
        images = components.images
        _images(images, "driving video")
        source_fps = float(components.frame_rate)
        if not math.isfinite(source_fps) or not 0 < source_fps <= 240:
            raise ValueError("Viggle-Animate: source FPS must be positive and at most 240.")
        if len(timestamps) != int(images.shape[0]):
            raise ValueError("Viggle-Animate: timing and image decode produced different frame counts; re-encode the source clip.")
        # floor avoids extending a partial final frame interval; tolerate only
        # floating-point representation noise, not another frame's duration.
        count = min(max_frames, math.floor(min(duration, video_end) * FPS + 1e-7))
        if count < 5:
            raise ValueError("Viggle-Animate: the selected clip must contain at least 5 frames at 24 fps.")
        positions = torch.arange(count, dtype=torch.float64) / FPS
        after = torch.searchsorted(timestamps, positions).clamp_(0, len(timestamps) - 1)
        before = (after - 1).clamp_(0)
        # On an exact tie choose the newer frame, matching nearest-neighbor
        # CFR resampling. PTS, rather than average FPS, drives this selection.
        indices = torch.where(positions - timestamps[before] < timestamps[after] - positions - 1e-9, before, after)
        indices = indices.to(device=images.device)
        normalized = images.index_select(0, indices)[..., :3]
        # Native ComfyUI's audio decode can apply the stream time base to a
        # resampled frame's PTS. Read frame.time_base explicitly for alignment.
        audio = _aligned_source_audio(clipped, count / FPS) if preserve_audio else None
        return normalized, audio, count, float(FPS)


class CRTP_ViggleTextCondLoader:
    CATEGORY = "CRTP/Viggle-Animate"
    FUNCTION = "load"
    RETURN_TYPES = ("TEXT_COND",)
    RETURN_NAMES = ("text_cond",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_cond": (folder_paths.get_filename_list("text_cond"),)}}

    def load(self, text_cond):
        from safetensors.torch import load_file

        path = folder_paths.get_full_path_or_raise("text_cond", text_cond)
        blob = load_file(path)
        embeds, tags = _text_conditioning(blob)
        return ({"prompt_embeds": embeds, "text_token_tags": tags},)


class CRTP_ViggleAnimateConditioning:
    CATEGORY = "CRTP/Viggle-Animate"
    FUNCTION = "build"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cond_video": ("IMAGE",), "ref_image": ("IMAGE",),
            "text_cond": ("TEXT_COND",), "vae": ("VAE",),
            "width": ("INT", {"default": 864, "min": 32, "max": 1536, "step": 32}),
            "height": ("INT", {"default": 480, "min": 32, "max": 1536, "step": 32}),
        }}

    def build(self, cond_video, ref_image, text_cond, vae, width=864, height=480):
        # Import H3 only when executed, so other CRTP helpers keep loading on
        # older installations. This node requires native MiniMax H3 support.
        from comfy_extras import nodes_minimax_h3 as core_h3
        import comfy.model_management
        import comfy.nested_tensor

        _images(cond_video, "driving video")
        _images(ref_image, "reference image")
        count = _integer(int(cond_video.shape[0]), "prepared frame count", 5, MAX_FRAMES)
        width, height = _canvas(width, height)
        embeds, tags = _text_conditioning(text_cond)
        short_edge = min(width, height)
        vh, vw = cond_video.shape[1:3]
        rw, rh = _reference_size(vw, vh, short_edge, width * height)
        ih, iw = ref_image.shape[1:3]
        # The finetune uses an uncapped still at the target short edge.
        # The 1:4..4:1 check bounds it to four times short_edge squared.
        tw, th = _reference_size(iw, ih, short_edge)
        frames = cond_video[..., :3]
        if (vw, vh) != (rw, rh):
            frames = core_h3._resize(frames, rw, rh, "disabled")
        still = ref_image[:1, ..., :3]
        if (iw, ih) != (tw, th):
            still = core_h3._resize(still, tw, th, "disabled")
        video_latent = vae.encode(frames)
        image_latent = vae.encode(still)
        # Ordering is part of the trained conditioning contract.
        refs = [
            {"kind": "video", "latent_t": video_latent.shape[2], "latent_h": rh // 16,
             "latent_w": rw // 16, "ref_audio_t": 0, "latent": video_latent, "audio_latent": None},
            {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": image_latent},
        ]
        _, latent_t, audio_t = core_h3.temporal_shape(_generation_frames(count))
        device = comfy.model_management.intermediate_device()
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros((1, 24, latent_t, height // 16, width // 16), device=device),
            torch.zeros((1, 32, 2, audio_t), device=device),
        ))}
        return [[embeds, {"minimax_refs": refs, "minimax_token_tags": tags}]], latent


class CRTP_ViggleSigmas:
    CATEGORY = "CRTP/Viggle-Animate"
    FUNCTION = "schedule"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"steps": ("INT", {"default": 3, "min": 3, "max": 7, "step": 2})}}

    def schedule(self, steps=3):
        steps = _integer(steps, "steps", 3, 7)
        if steps not in (3, 5, 7):
            raise ValueError("Viggle-Animate: supported Euler update counts are 3, 5, and 7.")
        # Distilled schedules published with the upstream workflow, exactly
        # steps+1 points. One terminal zero, without scheduler interpolation.
        schedules = {
            3: (1.0, 0.8571428571428571, 0.6, 0.0),
            5: (1.0, 0.9230769230769231, 0.8181818181818182, 0.6666666666666666, 0.42857142857142855, 0.0),
            7: (1.0, 0.9473684210526315, 0.8823529411764706, 0.8, 0.6923076923076923, 0.5454545454545454, 0.3333333333333333, 0.0),
        }
        return (torch.tensor(schedules[steps], dtype=torch.float32),)


class CRTP_ViggleTrimOutput:
    CATEGORY = "CRTP/Viggle-Animate"
    FUNCTION = "trim"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "frame_count": ("INT", {"default": 124, "min": 5, "max": MAX_FRAMES})}}

    def trim(self, images, frame_count):
        _images(images, "decoded video")
        frame_count = _integer(frame_count, "frame_count", 5, MAX_FRAMES)
        if images.shape[0] < frame_count:
            raise ValueError("Viggle-Animate: decoded video is shorter than its prepared driving clip.")
        return (images[:frame_count],)


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls for cls in (
        CRTP_ViggleResolution, CRTP_VigglePrepareVideo, CRTP_ViggleTextCondLoader,
        CRTP_ViggleAnimateConditioning, CRTP_ViggleSigmas, CRTP_ViggleTrimOutput,
    )
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_ViggleResolution": "CRTP Viggle Resolution",
    "CRTP_VigglePrepareVideo": "CRTP Viggle Prepare Video (24 fps)",
    "CRTP_ViggleTextCondLoader": "CRTP Viggle Frozen Text Conditioning",
    "CRTP_ViggleAnimateConditioning": "CRTP Viggle Animate Conditioning",
    "CRTP_ViggleSigmas": "CRTP Viggle Distilled Sigmas",
    "CRTP_ViggleTrimOutput": "CRTP Viggle Trim Output",
}
