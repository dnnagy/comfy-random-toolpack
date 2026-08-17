"""CRTP MiniMax H3 continuation nodes.

These nodes intentionally layer on top of the pinned MiniMax H3 Extender
instead of replacing it.  They add the two pieces that upstream does not
currently provide:

* an H3-compatible audio/video latent made from the tail of an uploaded video;
* an Extender variant that can use that latent for clip zero and exposes all
  three standalone Ref2VA audio slots supported by ComfyUI core.
"""

from __future__ import annotations

import importlib
import math

import torch
import torchaudio

import comfy.model_management
import comfy.nested_tensor
import comfy.utils


FPS = 24
VALID_CONTEXT_LENGTHS = (5, 22, 39, 56)
BUILD = "crtp-h3-continuation-v1"


def _upstream():
    return importlib.import_module("ComfyUI_MiniMax_H3_Extender.extender")


def _fit_frames(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Aspect-fit frames exactly once, using the same letterbox policy as concat."""
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


def _aligned_audio_window(audio, source_duration: float, window_duration: float):
    """Return the audio window aligned to the end of the source video.

    Missing portions are zero padded.  This matters for videos whose audio
    stream starts late or ends slightly before the video stream.
    """
    if audio is None:
        return None

    waveform = audio["waveform"][:1]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim != 3 or sample_rate <= 0:
        raise ValueError("CRTP H3 Encode Continuation Context: invalid source audio.")

    target_samples = max(1, int(round(float(window_duration) * sample_rate)))
    window_end = int(round(float(source_duration) * sample_rate))
    window_start = window_end - target_samples
    audio_samples = int(waveform.shape[-1])

    result = torch.zeros(
        (1, int(waveform.shape[-2]), target_samples),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    copy_start = max(0, window_start)
    copy_end = min(audio_samples, window_end)
    if copy_end > copy_start:
        destination_start = copy_start - window_start
        destination_end = destination_start + (copy_end - copy_start)
        result[..., destination_start:destination_end] = waveform[..., copy_start:copy_end]

    return {"waveform": result, "sample_rate": sample_rate}


def _nearest_h3_frame_count(frame_count: int) -> int:
    """Snap to H3's 17k+5 grid without always lengthening the request."""
    requested = max(5, int(frame_count))
    cycle = max(0, int(round((requested - 5) / 17.0)))
    return 5 + cycle * 17


class CRTP_H3ContinuationTail:
    """Resample only the required source tail to H3's 24 fps timeline."""

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

        # Match FFmpeg's nearest-frame 24 fps resampling while forcing the
        # boundary sample to be the exact final decoded source frame.
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
            f"at {FPS} fps, {int(width)}x{int(height)}"
        )
        return (
            tail,
            tail[-1:],
            requested,
            context_duration,
            float(source_duration),
            status,
        )


class CRTP_H3EncodeContinuationContext:
    """VAE-encode a normalized source tail into a joint H3 AV latent."""

    CATEGORY = "CRTP/MiniMax H3"
    FUNCTION = "encode"
    RETURN_TYPES = ("LATENT", "AUDIO", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "context_latent",
        "aligned_audio",
        "video_latent_steps",
        "audio_latent_steps",
        "status",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tail_frames": ("IMAGE",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "context_duration": ("FLOAT", {"default": 22 / 24}),
                "source_duration": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                "source_audio": ("AUDIO", {"forceInput": True}),
            },
        }

    def encode(
        self,
        tail_frames,
        vae,
        audio_vae,
        context_duration,
        source_duration,
        source_audio=None,
    ):
        frame_count = int(tail_frames.shape[0])
        if frame_count not in VALID_CONTEXT_LENGTHS:
            raise ValueError(
                "CRTP H3 Encode Continuation Context: tail must contain "
                f"one of {VALID_CONTEXT_LENGTHS}, got {frame_count}."
            )

        aligned = _aligned_audio_window(
            source_audio, float(source_duration), float(context_duration)
        )
        vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if aligned is None:
            audio_samples = max(1, int(round(float(context_duration) * vae_sample_rate)))
            waveform = torch.zeros(
                (1, 1, audio_samples),
                dtype=torch.float32,
                device=comfy.model_management.intermediate_device(),
            )
            aligned = {"waveform": waveform, "sample_rate": vae_sample_rate}

        waveform = aligned["waveform"]
        sample_rate = int(aligned["sample_rate"])
        if sample_rate != vae_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)

        video_latent = vae.encode(tail_frames)
        audio_latent = audio_vae.encode(waveform[:1].movedim(1, -1))
        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent))
        }
        status = (
            f"encoded {frame_count} source frames into {int(video_latent.shape[2])} "
            f"video and {int(audio_latent.shape[-1])} audio latent steps"
        )
        return (
            latent,
            aligned,
            int(video_latent.shape[2]),
            int(audio_latent.shape[-1]),
            status,
        )


class CRTP_MiniMaxH3ContinuationExtender:
    """Upstream Extender plus clip-zero motion context and three audio refs."""

    @classmethod
    def INPUT_TYPES(cls):
        upstream = _upstream()
        schema = upstream.MiniMaxH3Extender.INPUT_TYPES()
        optional = dict(schema.get("optional", {}))
        optional.update(
            {
                "initial_context": ("LATENT", {"forceInput": True}),
                "ref_audio_2": ("AUDIO", {"forceInput": True}),
                "ref_audio_3": ("AUDIO", {"forceInput": True}),
            }
        )
        return {
            "required": dict(schema["required"]),
            "optional": optional,
            "hidden": dict(schema.get("hidden", {})),
        }

    RETURN_TYPES = ("H3_MOTION_DISK_CACHE", "INT", "INT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = (
        "cache",
        "clip_count",
        "validated_count",
        "status",
        "cache_size_mb",
        "build",
    )
    FUNCTION = "extend"
    CATEGORY = "CRTP/MiniMax H3"
    OUTPUT_NODE = False

    def extend(
        self,
        model,
        clip,
        vae,
        run_mode,
        width,
        height,
        ref_image_size,
        steps,
        sampler_name,
        scheduler,
        denoise,
        context_length,
        audio_context_length,
        clips_json,
        unique_id=None,
        **kwargs,
    ):
        upstream = _upstream()
        clips = upstream._parse_clips_json(clips_json)
        owner = str(unique_id if unique_id is not None else "h3_continuation_extender")
        data_path, manifest_path, manifest = upstream._manifest_for_extender(owner, upstream.FPS)

        if len(manifest.get("segments", [])) > len(clips):
            manifest = upstream._truncate_chain(
                data_path, manifest_path, manifest, len(clips)
            )

        segments = manifest.get("segments", [])
        if len(segments) < len(clips):
            for i in range(len(segments), len(clips)):
                if clips[i]["validated"]:
                    for j in range(i, len(clips)):
                        clips[j]["validated"] = False
                    break

        refs = [kwargs.get(f"ref_{i}") for i in range(1, 10)]
        audio_vae = kwargs.get("audio_vae")
        ref_items, ref_blocks = upstream._prepare_shared_refs(
            vae,
            audio_vae,
            int(width),
            int(height),
            str(ref_image_size),
            refs,
            ref_audio=None,
        )

        audios = [
            kwargs.get("ref_audio"),
            kwargs.get("ref_audio_2"),
            kwargs.get("ref_audio_3"),
        ]
        seen_audio_gap = False
        for index, audio in enumerate(audios, start=1):
            if audio is None:
                seen_audio_gap = True
                continue
            if seen_audio_gap:
                raise ValueError(
                    "CRTP MiniMax H3 Continuation Extender: audio reference slots "
                    f"must be connected contiguously. <Audio {index}> cannot be used "
                    f"without <Audio {index - 1}>."
                )
            if audio_vae is None:
                raise ValueError(
                    "CRTP MiniMax H3 Continuation Extender: connect audio_vae when "
                    "using reference audio."
                )
            if not ref_items:
                raise ValueError(
                    "CRTP MiniMax H3 Continuation Extender: Ref2VA audio requires "
                    "at least one image reference."
                )
            audio_latent, ref_audio_t = upstream._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append(
                {
                    "kind": "audio",
                    "ref_audio_t": int(ref_audio_t),
                    "audio_latent": audio_latent,
                }
            )

        disk_join = upstream.MiniMaxH3MotionContextDiskJoin()
        motion = upstream.MiniMaxH3MotionContextRAM()
        initial_context = kwargs.get("initial_context")

        previous_handle = None
        previous_proxy = None
        generated = []
        statuses = []

        for i, cfg in enumerate(clips):
            current_manifest = upstream._load_manifest_from_paths(data_path, manifest_path)
            existing_count = len(current_manifest.get("segments", [])) if current_manifest else 0
            existing = i < existing_count

            if cfg["validated"] and existing:
                result = disk_join.join(
                    samples=None,
                    trim_frames=None,
                    validated=True,
                    run_mode=str(run_mode),
                    fps=float(upstream.FPS),
                    previous_cache=previous_handle,
                    unique_id=f"extender_{owner}",
                )
                previous_handle = result[0]
                previous_proxy = result[1]
                statuses.append(result[4])
                continue

            upstream._send_extender_progress(
                owner, i, len(clips), "preparing", f"Preparing clip {i + 1}/{len(clips)}"
            )
            cfg["validated"] = False
            for j in range(i + 1, len(clips)):
                clips[j]["validated"] = False

            context_source = initial_context if i == 0 else previous_proxy
            if context_source is not None:
                # The overlap is decoded and trimmed by Disk Join. Add it to
                # the internal sample length so the user still receives the
                # requested amount of *new* video rather than a shorter clip.
                requested_new_frames = max(5, int(round(float(cfg["duration"]) * upstream.FPS)))
                frame_count = _nearest_h3_frame_count(
                    requested_new_frames + int(context_length)
                )
            else:
                frame_count = upstream._duration_to_frames(cfg["duration"])
            positive, latent = upstream._make_ref2va_conditioning(
                clip,
                vae,
                cfg["prompt"],
                int(width),
                int(height),
                frame_count,
                ref_items,
                ref_blocks,
            )

            trim_frames = None
            if context_source is not None:
                positive, trim_frames, _, _, _ = motion.apply(
                    positive,
                    latent,
                    context_source,
                    str(context_length),
                    int(audio_context_length),
                )
            elif i > 0:
                raise RuntimeError(
                    "CRTP MiniMax H3 Continuation Extender: previous cached latent "
                    "is unavailable."
                )

            upstream._send_extender_progress(
                owner, i, len(clips), "sampling", f"Rendering clip {i + 1}/{len(clips)}"
            )
            sampled = upstream._sample_h3(
                model,
                positive,
                latent,
                cfg["seed"],
                str(sampler_name),
                str(scheduler),
                int(steps),
                float(denoise),
            )
            result = disk_join.join(
                samples=sampled,
                trim_frames=trim_frames,
                validated=False,
                run_mode=str(run_mode),
                fps=float(upstream.FPS),
                previous_cache=previous_handle,
                unique_id=f"extender_{owner}",
            )
            previous_handle = result[0]
            previous_proxy = result[1]
            statuses.append(result[4])
            generated.append(i)

            upstream._send_extender_progress(
                owner, i, len(clips), "complete", f"Clip {i + 1}/{len(clips)} complete"
            )
            del sampled, positive, latent
            if str(run_mode) == "clip_by_clip":
                break

        if previous_handle is None:
            raise RuntimeError(
                "CRTP MiniMax H3 Continuation Extender: sequence produced no cache handle."
            )

        final_manifest = upstream._load_manifest_from_paths(data_path, manifest_path)
        cached_count = len(final_manifest.get("segments", []))
        validated_count = 0
        for descriptor in final_manifest.get("segments", []):
            if bool(descriptor.get("validated", False)):
                validated_count += 1
            else:
                break

        status = (
            f"{str(run_mode)} | cached {cached_count}/{len(clips)} | "
            f"validated {validated_count} | "
            + (
                "generated " + ",".join(str(i + 1) for i in generated)
                if generated
                else "disk only"
            )
            + (" | source context" if initial_context is not None else "")
            + f" | {len([a for a in audios if a is not None])} audio refs"
        )
        cache_mb = upstream._cache_size_mb(data_path, manifest_path)
        upstream._send_extender_progress(owner, -1, len(clips), "idle", status)

        normalized_json = upstream._state_json(clips)
        ui_state = {
            "clips_json": normalized_json,
            "clip_count": len(clips),
            "cached_count": cached_count,
            "validated_count": validated_count,
            "generated": [i + 1 for i in generated],
            "status": status,
            "build": BUILD,
        }
        return {
            "ui": {"h3_extender_state": [ui_state]},
            "result": (
                previous_handle,
                int(len(clips)),
                int(validated_count),
                status,
                float(cache_mb),
                BUILD,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "CRTP_H3ContinuationTail": CRTP_H3ContinuationTail,
    "CRTP_H3EncodeContinuationContext": CRTP_H3EncodeContinuationContext,
    "CRTP_MiniMaxH3ContinuationExtender": CRTP_MiniMaxH3ContinuationExtender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_H3ContinuationTail": "CRTP H3 Continuation Tail (24 fps)",
    "CRTP_H3EncodeContinuationContext": "CRTP H3 Encode Continuation Context",
    "CRTP_MiniMaxH3ContinuationExtender": "CRTP MiniMax H3 Continuation Extender",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
