"""Experimental H3 latent-state nodes. Native guides, never sampler warm starts.

Original implementation; design references and format contract: docs/h3-state.md.
Imports of ComfyUI and safetensors are deferred so header/format tests run on CPU.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
import wave

import torch

FORMAT = "crtp_h3_state_v1"
LAYOUT = "h3_24ch_16x_1-4-4-4-4_v1"
CATEGORY = "CRTP/MiniMax H3/Latent State (Experimental)"
STATE = "CRTP_H3_STATE"
CONTEXT = "CRTP_H3_CONTEXT"
PLAN = "CRTP_H3_PLAN"


def frame_count(t):
    if t == 1:
        return 1
    if t < 2 or t % 5 != 2:
        raise ValueError("H3 video requires 5k+2 temporal tokens; a still requires one.")
    return 5 + (t - 2) // 5 * 17


def split_samples(latent):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        if len(parts) != 2:
            raise ValueError("Expected H3 nested video/audio latent pair.")
        return parts[0], parts[1]
    return samples, None


def validate_header(meta, shapes, dtypes):
    if meta.get("format") != FORMAT or meta.get("layout") != LAYOUT:
        raise ValueError("Unsupported H3 state format/layout; no automatic conversion.")
    if meta.get("kind") not in ("image_keyframe", "av_clip"):
        raise ValueError("Unknown H3 state kind.")
    if meta.get("origin") not in ("sampled", "encoded_pixels"):
        raise ValueError("Unknown latent origin.")
    if set(shapes) not in ({"video"}, {"video", "audio"}):
        raise ValueError("State must contain video and optional audio tensors only.")
    v = shapes["video"]
    if len(v) != 5 or v[:2] != [1, 24] or min(v) < 1:
        raise ValueError("H3 video shape must be [1,24,T,H/16,W/16].")
    frames = frame_count(v[2])
    if (frames == 1) != (meta["kind"] == "image_keyframe"):
        raise ValueError("A video tail token is not an image-keyframe latent.")
    if meta.get("raw_frames") != frames or meta.get("fps") != 24:
        raise ValueError("State timing disagrees with tensor shape.")
    if meta.get("width") != v[4] * 16 or meta.get("height") != v[3] * 16:
        raise ValueError("State canvas disagrees with tensor shape.")
    if meta["width"] % 32 or meta["height"] % 32:
        raise ValueError("H3 generation canvas must use multiples of 32.")
    start, end = meta.get("delivered_start"), meta.get("delivered_end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= frames:
        raise ValueError("Invalid delivered frame range.")
    if not isinstance(meta.get("vae_id"), str) or not meta["vae_id"].strip():
        raise ValueError("An explicit compatible H3 video VAE identifier is required.")
    if not isinstance(meta.get("id"), str) or len(meta["id"]) != 32:
        raise ValueError("Invalid state identity.")
    if any(d not in ("F16", "BF16", "F32", "F64") for d in dtypes.values()):
        raise ValueError("Only floating point H3 latents are supported.")
    if "audio" in shapes:
        a = shapes["audio"]
        if len(a) != 4 or a[:3] != [1, 32, 2] or a[3] != round(frames * 40 / 24):
            raise ValueError("Audio latent shape/timing does not match native H3 AV layout.")
    phase = meta.get("audio_phase", [0, 1])
    if not isinstance(phase, list) or len(phase) != 2 or any(type(x) is not int for x in phase) or phase[1] <= 0:
        raise ValueError("Invalid audio phase.")
    if abs(Fraction(*phase)) > Fraction(1, 80):
        raise ValueError("Audio phase exceeds half an H3 audio token.")


def tensor_description(tensors):
    names = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64"}
    return ({k: list(v.shape) for k, v in tensors.items()},
            {k: names.get(v.dtype, str(v.dtype)) for k, v in tensors.items()})


def make_state(latent, kind, origin, vae_id, plan=None):
    video, audio = split_samples(latent)
    tensors = {"video": video.detach()}
    if audio is not None:
        tensors["audio"] = audio.detach()
    if video.ndim != 5:
        raise ValueError("Expected an H3 video tensor, not a generic image-model latent.")
    frames = frame_count(video.shape[2])
    meta = dict(format=FORMAT, layout=LAYOUT, id=uuid.uuid4().hex, kind=kind,
                origin=origin, vae_id=vae_id, fps=24, width=video.shape[4] * 16,
                height=video.shape[3] * 16, raw_frames=frames,
                delivered_start=0, delivered_end=frames, audio_phase=[0, 1])
    if plan is not None:
        if kind != "av_clip" or plan["raw_frames"] != frames or plan["width"] != meta["width"] or plan["height"] != meta["height"] or plan["vae_id"] != vae_id:
            raise ValueError("Generation plan does not match sampled latent.")
        meta.update(delivered_start=plan["trim_head"], delivered_end=frames,
                    parent=deepcopy(plan.get("parent")), audio_phase=plan["audio_phase"])
    validate_header(meta, *tensor_description(tensors))
    return {"meta": meta, "tensors": tensors}


def signature(path):
    st = Path(path).stat()
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def resolve_input(path):
    import folder_paths
    p = Path(path).expanduser()
    if not p.is_absolute():
        # Saved nodes return absolute paths; relative paths are output-relative.
        p = Path(folder_paths.get_output_directory()) / p
    if p.suffix != ".safetensors" or not p.is_file():
        raise ValueError("Select an existing CRTP H3 .safetensors state file.")
    return p.resolve()


def read_state(path):
    from safetensors import safe_open
    p = resolve_input(path)
    before = signature(p)
    with safe_open(str(p), framework="pt", device="cpu") as f:
        meta = json.loads((f.metadata() or {}).get("crtp_h3", "{}"))
        shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
        dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
        validate_header(meta, shapes, dtypes)
    if signature(p) != before:
        raise ValueError("State changed while reading its header; reload it.")
    return {"meta": meta, "path": str(p), "signature": before}


def read_tensors(state, video_slice=None, audio_slice=None, include_audio=True):
    if "tensors" in state:
        source = state["tensors"]
        tensors = {"video": source["video"][:, :, video_slice or slice(None)]}
        if include_audio and "audio" in source:
            tensors["audio"] = source["audio"][..., audio_slice or slice(None)]
        return tensors
    from safetensors import safe_open
    p = state["path"]
    if signature(p) != state["signature"]:
        raise ValueError("State changed after lazy load; reload it.")
    with safe_open(p, framework="pt", device="cpu") as f:
        tensors = {"video": f.get_slice("video")[:, :, video_slice or slice(None)]}
        if include_audio and "audio" in f.keys():
            tensors["audio"] = f.get_slice("audio")[..., audio_slice or slice(None)]
    if signature(p) != state["signature"]:
        raise ValueError("State changed during tensor load; reload it.")
    return tensors


def output_base(prefix):
    import folder_paths
    root = Path(folder_paths.get_output_directory()).resolve()
    rel = Path(prefix)
    if rel.is_absolute() or ".." in rel.parts or not rel.name:
        raise ValueError("Filename prefix must be relative to ComfyUI output.")
    base = (root / rel).resolve()
    if not base.is_relative_to(root):
        raise ValueError("Output path escapes ComfyUI output.")
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_name(base.name + "_" + uuid.uuid4().hex[:12])


def write_state(state, path):
    from safetensors.torch import save_file
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in read_tensors(state).items()}
    validate_header(state["meta"], *tensor_description(tensors))
    path = Path(path)
    # Atomic publication; callers allocate unique names, never overwrite masters.
    fd, temp = tempfile.mkstemp(prefix=".h3-", suffix=".safetensors", dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, temp, metadata={"crtp_h3": json.dumps(state["meta"], separators=(",", ":"))})
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return str(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CRTP_H3StateFromLatent:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = (STATE,)
    RETURN_NAMES = ("state",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",),
                "kind": (["av_clip", "image_keyframe"],),
                "origin": (["sampled", "encoded_pixels"],),
                "vae_id": ("STRING", {"default": "minimax_h3_video_vae_fp16.safetensors"})},
                "optional": {"plan": (PLAN,)}}

    def run(self, latent, kind, origin, vae_id, plan=None):
        return (make_state(latent, kind, origin, vae_id, plan),)


class CRTP_SaveH3State:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = (STATE, "STRING")
    RETURN_NAMES = ("state", "state_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": (STATE,), "filename_prefix": ("STRING", {"default": "h3_test/state"})}}

    def run(self, state, filename_prefix):
        path = write_state(state, str(output_base(filename_prefix)) + ".safetensors")
        print(f"CRTP H3 saved state: {path}")
        return {"ui": {"text": [path]}, "result": (state, path)}


class CRTP_LazyLoadH3State:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = (STATE, "STRING")
    RETURN_NAMES = ("state", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state_path": ("STRING", {"default": "h3_test/keyframe.safetensors"})}}

    @classmethod
    def IS_CHANGED(cls, state_path):
        try:
            return signature(resolve_input(state_path))
        except (OSError, ValueError):
            return float("nan")

    def run(self, state_path):
        state = read_state(state_path)
        return (state, json.dumps(state["meta"], indent=2))


class CRTP_H3SelectLatentContext:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = (CONTEXT, "STRING")
    RETURN_NAMES = ("context", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": (STATE,), "mode": (["tail_22", "tail_5", "image_keyframe"],),
                             "reuse_audio": ("BOOLEAN", {"default": False})}}

    def run(self, state, mode, reuse_audio):
        meta = state["meta"]
        if mode == "image_keyframe":
            if meta["kind"] != "image_keyframe":
                raise ValueError("Use a genuine single-image latent; a video's final token is not a keyframe.")
            tensors = read_tensors(state, include_audio=False)
            start, end, phase = 0, 1, Fraction(0)
        else:
            if mode not in ("tail_5", "tail_22") or meta["kind"] != "av_clip":
                raise ValueError("Tail selection requires an H3 video state and a 5/22 frame window.")
            n = int(mode.split("_")[1])
            end = meta["delivered_end"]
            start = end - n
            if start < 0 or start % 17 or end % 17 != 5:
                raise ValueError("Tail is too short or off H3's 17-frame grid. No silent snapping/re-encoding.")
            token_start = start // 17 * 5
            token_count = 2 + (n - 5) // 17 * 5
            old_phase = Fraction(*meta["audio_phase"])
            audio_start = round((Fraction(start, 24) - old_phase) * 40)
            phase = old_phase + Fraction(audio_start, 40) - Fraction(start, 24)
            # Slice on the source grid using absolute boundaries. Rounding the
            # window duration independently can ask for one nonexistent token.
            audio_end = min(round(Fraction(meta["raw_frames"], 24) * 40),
                            round((Fraction(end, 24) - old_phase) * 40))
            tensors = read_tensors(state, slice(token_start, token_start + token_count),
                                   slice(audio_start, audio_end), reuse_audio)
            if reuse_audio and ("audio" not in tensors or tensors["audio"].shape[-1] != audio_end - audio_start):
                raise ValueError("Requested source audio is absent or too short; disable reuse_audio explicitly.")
            if not reuse_audio:
                phase = Fraction(0)
        # Own detached copies: neither sampler nor downstream code can mutate the master.
        context = {"video": tensors["video"].clone(), "audio": tensors.get("audio"),
                   "kind": meta["kind"], "frames": end - start, "width": meta["width"],
                   "height": meta["height"], "vae_id": meta["vae_id"],
                   "audio_phase": [phase.numerator, phase.denominator],
                   "parent": {"id": meta["id"], "start": start, "end": end,
                              "origin": meta["origin"], "state_path": state.get("path")}}
        if context["audio"] is not None:
            context["audio"] = context["audio"].clone()
        return (context, f"{mode}: raw source frames [{start}, {end}); {meta['origin']}; no VAE encoding")


class CRTP_H3PrepareLatentGeneration:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("CONDITIONING", "LATENT", PLAN)
    RETURN_NAMES = ("positive", "fresh_latent", "plan")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "context": (CONTEXT,),
                "seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "vae_id": ("STRING", {"default": "minimax_h3_video_vae_fp16.safetensors"})}}

    def run(self, positive, context, seconds, vae_id):
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        if context["vae_id"] != vae_id:
            raise ValueError("Guide VAE identifier mismatch. No implicit latent conversion.")
        if not math.isfinite(seconds) or not 1 <= seconds <= 15:
            raise ValueError("Test generation duration must be 1–15 seconds.")
        trim = context["frames"] if context["kind"] == "av_clip" else 0
        requested = round(seconds * 24) + trim
        # Nearest native 17k+5 length, matching the existing continuation convention.
        length = 5 + 17 * max(0, round((requested - 5) / 17))
        latent, frames = _empty_av_latent(context["width"], context["height"], length)
        guide = {"resolved_frame_index": 0, "latent": context["video"].detach().clone()}
        guides = [guide]
        if context["audio"] is not None:
            # Source audio slices can start between video frames. Put that
            # offset in the native guide's position, not in newly sampled audio.
            # The target AV latent always starts on its own standard zero grid.
            guides.append({"resolved_frame_index": float(Fraction(*context["audio_phase"]) * 24),
                           "audio_latent": context["audio"].detach().clone()})
        result = []
        for embedding, values in positive:
            values = values.copy()
            if values.get("minimax_keyframes"):
                raise ValueError("Conditioning already has temporal guides; use unanchored Ref2VA conditioning.")
            values["minimax_keyframes"] = guides
            result.append([embedding, values])
        plan = dict(raw_frames=frames, trim_head=trim, width=context["width"],
                    height=context["height"], vae_id=vae_id, fps=24,
                    audio_phase=[0, 1], source_audio_phase=context["audio_phase"],
                    parent=context["parent"] if trim else None)
        # This is always a new empty AV latent. Clean guide tensors never enter its samples.
        return (result, latent, plan)


class CRTP_H3DecodeAndTrim:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", STATE)
    RETURN_NAMES = ("images", "audio", "raw_state")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "audio_vae": ("VAE",),
                             "plan": (PLAN,)},
                "optional": {"raw_state": (STATE,), "after_state": (STATE,)}}

    def run(self, samples, vae, audio_vae, plan, raw_state=None, after_state=None):
        from nodes import VAEDecode
        from comfy_extras.nodes_audio import vae_decode_audio
        state = make_state(samples, "av_clip", "sampled", plan["vae_id"], plan)
        if raw_state is not None:
            # Preserve the identity already used by downstream continuations.
            # Rewrapping here would give the parent a different UUID at save time.
            for key, value in state["meta"].items():
                if key != "id" and raw_state["meta"].get(key) != value:
                    raise ValueError("raw_state does not match this generation plan.")
            original = read_tensors(raw_state)
            if set(original) != set(state["tensors"]):
                raise ValueError("raw_state does not match sampled AV tensors.")
            for key, tensor in state["tensors"].items():
                saved = original[key]
                if saved.shape != tensor.shape or saved.dtype != tensor.dtype:
                    raise ValueError("raw_state does not match sampled AV tensors.")
                if saved.data_ptr() != tensor.data_ptr() and not torch.equal(saved.to(tensor.device), tensor):
                    raise ValueError("raw_state does not match sampled AV tensors.")
            state = raw_state
        # after_state is an execution dependency, not conditioning: a later
        # sampler can finish before this decoder starts loading its VAEs.
        images = VAEDecode().decode(vae, samples)[0]
        if images.shape[0] != plan["raw_frames"]:
            raise ValueError("VAE decoded frame count differs from saved raw H3 layout.")
        audio = vae_decode_audio(audio_vae, samples)
        rate = int(audio["sample_rate"])
        # The audio token selected for context can start ±1/120 s from the video grid.
        phase = Fraction(*plan["audio_phase"])
        start = round((Fraction(plan["trim_head"], 24) - phase) * rate)
        count = round(Fraction(plan["raw_frames"] - plan["trim_head"], 24) * rate)
        waveform = audio["waveform"]
        end = start + count
        if start < 0:
            waveform = torch.nn.functional.pad(waveform, (-start, 0))
            end -= start
            start = 0
        if end > waveform.shape[-1]:
            waveform = torch.nn.functional.pad(waveform, (0, end - waveform.shape[-1]))
        audio = {**audio, "waveform": waveform[..., start:end]}
        return (images[plan["trim_head"]:], audio, state)


def encode_take(images, audio, path, crf):
    """Stream frames to FFmpeg; no extra full RGB byte-buffer or shell invocation."""
    if not 0 <= int(crf) <= 51:
        raise ValueError("CRF must be 0–51.")
    with tempfile.TemporaryDirectory(prefix="crtp-h3-") as td:
        wav = Path(td) / "audio.wav"
        waveform = audio["waveform"].detach().cpu()
        if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
            raise ValueError("Expected mono/stereo batch-one audio.")
        pcm = (waveform[0].T.clamp(-1, 1) * 32767).to(torch.int16).numpy().astype("<i2").tobytes()
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(waveform.shape[1]); f.setsampwidth(2)
            f.setframerate(int(audio["sample_rate"])); f.writeframes(pcm)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{images.shape[2]}x{images.shape[1]}",
                   "-r", "24", "-i", "pipe:0", "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-crf", str(int(crf)), "-preset", "fast", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k", "-movflags", "+faststart", str(path)]
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for frame in images:
                    proc.stdin.write((frame[..., :3].detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy().tobytes())
                proc.stdin.close()
                if proc.wait(timeout=1800):
                    raise RuntimeError("FFmpeg failed encoding take.")
            except BaseException:
                proc.kill(); proc.wait()
                errors.seek(0)
                message = errors.read(4000).decode(errors="replace")
                if message:
                    print("CRTP H3 FFmpeg:", message)
                raise


def video_ui(path, text):
    import folder_paths
    relative = Path(path).relative_to(Path(folder_paths.get_output_directory()).resolve())
    return {"text": text, "images": [{"filename": relative.name,
            "subfolder": str(relative.parent) if str(relative.parent) != "." else "",
            "type": "output"}], "animated": (True,)}


class CRTP_H3SaveTake:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = (STATE, "STRING", "STRING")
    RETURN_NAMES = ("state", "state_path", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": (STATE,), "images": ("IMAGE",), "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "h3_test/take"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51})}}

    def run(self, state, images, audio, filename_prefix, crf):
        meta = deepcopy(state["meta"])
        count = meta["delivered_end"] - meta["delivered_start"]
        if meta["kind"] != "av_clip" or list(images.shape) != [count, meta["height"], meta["width"], 3]:
            raise ValueError("Delivered images do not match state range/canvas.")
        if audio["waveform"].shape[-1] != round(count * audio["sample_rate"] / 24):
            raise ValueError("Delivered audio must be trimmed to the exact video duration.")
        base = output_base(filename_prefix)
        video_path = str(base) + ".mp4"
        state_path = str(base) + ".safetensors"
        temporary = str(base) + ".partial.mp4"
        try:
            encode_take(images, audio, temporary, crf)
            os.link(temporary, video_path)
            meta["media"] = {"filename": Path(video_path).name, "sha256": digest(video_path)}
            saved = {**state, "meta": meta}
            write_state(saved, state_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        # A sidecar is the commit point. A failed sidecar write leaves a playable orphan video.
        loaded = read_state(state_path)
        print(f"CRTP H3 saved state: {state_path}\nCRTP H3 saved video: {video_path}")
        return {"ui": video_ui(video_path, [state_path, video_path]), "result": (loaded, state_path, video_path)}


class CRTP_H3AssembleTakes:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state_paths": ("STRING", {"multiline": True, "default": ""}),
                "filename_prefix": ("STRING", {"default": "h3_test/assembled"}),
                "allow_cuts": ("BOOLEAN", {"default": False}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51})},
                "optional": {"state_path_1": ("STRING", {"forceInput": True}),
                             "state_path_2": ("STRING", {"forceInput": True}),
                             "state_path_3": ("STRING", {"forceInput": True}),
                             "pad_final_frame": ("BOOLEAN", {"default": False})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Must re-verify externally edited/swapped media on every explicit export.
        return float("nan")

    def run(self, state_paths, filename_prefix, allow_cuts, crf,
            state_path_1=None, state_path_2=None, state_path_3=None, pad_final_frame=False):
        paths = [line.strip() for line in state_paths.splitlines() if line.strip()]
        wired = (state_path_1, state_path_2, state_path_3)
        if any(path is not None for path in wired):
            if paths:
                raise ValueError("Use either wired state paths or the text list, not both.")
            gap = False
            for path in wired:
                if path is None:
                    gap = True
                elif gap or not path.strip():
                    raise ValueError("Wired state paths must be nonempty and contiguous from slot 1.")
                else:
                    paths.append(path.strip())
        if not paths or len(paths) > 100:
            raise ValueError("Supply 1–100 state paths, one per line, in playback order.")
        if not 0 <= int(crf) <= 51:
            raise ValueError("CRF must be 0–51.")
        states = [read_state(path) for path in paths]
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        filters = []
        previous = None
        canvas = None
        total_frames = 0
        for i, state in enumerate(states):
            meta = state["meta"]
            size = (meta["width"], meta["height"])
            if canvas is not None and size != canvas:
                raise ValueError("Assemble requires matching canvases; no implicit resize of masters.")
            canvas = size
            media = meta.get("media", {})
            name = media.get("filename", "")
            if not name or Path(name).name != name:
                raise ValueError("State has no paired take video.")
            video = Path(state["path"]).parent / name
            if not video.is_file() or digest(video) != media.get("sha256"):
                raise ValueError("Paired video is missing or modified; cannot trust join metadata.")
            if previous is not None and not allow_cuts:
                parent = meta.get("parent") or {}
                if parent.get("id") != previous["id"] or parent.get("end") != previous["delivered_end"]:
                    raise ValueError("Takes are not consecutive parent/child clips; enable allow_cuts for an intentional cut.")
            previous = meta
            count = meta["delivered_end"] - meta["delivered_start"]
            total_frames += count
            duration = count / 24
            command += ["-i", str(video)]
            filters += [f"[{i}:v]trim=end_frame={count},setpts=N/(24*TB),setsar=1[v{i}]",
                        f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=duration={duration:.12f},asetpts=PTS-STARTPTS[a{i}]"]
        filters.append("".join(f"[v{i}][a{i}]" for i in range(len(states))) + f"concat=n={len(states)}:v=1:a=1[vc][ac]")
        if pad_final_frame:
            filters += [f"[vc]tpad=stop_mode=clone:stop=1,trim=end_frame={total_frames+1},setpts=N/(24*TB)[v]",
                        f"[ac]apad,atrim=duration={(total_frames+1)/24:.12f},asetpts=PTS-STARTPTS[a]"]
        else:
            filters += ["[vc]null[v]", "[ac]anull[a]"]
        base = output_base(filename_prefix)
        output = str(base) + ".mp4"
        temporary = str(base) + ".partial.mp4"
        command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
                    # Single output: legacy -vsync works on the pod's FFmpeg 4.4
                    # as well as newer releases; -fps_mode requires FFmpeg 5.1+.
                    "-r", "24", "-vsync", "cfr",
                    "-c:v", "libx264", "-crf", str(int(crf)), "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", temporary]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=1800)
            os.link(temporary, output)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = exc.stderr or "FFmpeg returned no error output."
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            reason = (f"exit status {exc.returncode}" if isinstance(exc, subprocess.CalledProcessError)
                      else f"timeout after {exc.timeout} seconds")
            raise RuntimeError(
                f"H3 assembly failed ({reason}). FFmpeg error:\n{detail[-8000:].strip()}\n"
                "Saved segment videos and states are unchanged. Retry with the assembly-only "
                "workflow; no sampling or VAE decoding is needed."
            ) from exc
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"ui": video_ui(output, [output]), "result": (output,)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    CRTP_H3StateFromLatent, CRTP_SaveH3State, CRTP_LazyLoadH3State,
    CRTP_H3SelectLatentContext, CRTP_H3PrepareLatentGeneration,
    CRTP_H3DecodeAndTrim, CRTP_H3SaveTake, CRTP_H3AssembleTakes,
)}
# Versioned graph IDs make older installs reject the three-segment workflow
# during prompt validation, before expensive sampling. Keep the original IDs
# for existing single-segment graphs; these aliases share the same classes.
NODE_CLASS_MAPPINGS.update({
    "CRTP_H3DecodeAndTrimV2": CRTP_H3DecodeAndTrim,
    "CRTP_H3AssembleTakesV2": CRTP_H3AssembleTakes,
})
NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_H3StateFromLatent": "CRTP H3 State from Latent",
    "CRTP_SaveH3State": "CRTP Save H3 State",
    "CRTP_LazyLoadH3State": "CRTP Lazy Load H3 State",
    "CRTP_H3SelectLatentContext": "CRTP H3 Select Latent Context",
    "CRTP_H3PrepareLatentGeneration": "CRTP H3 Prepare Latent Generation",
    "CRTP_H3DecodeAndTrim": "CRTP H3 Decode and Trim",
    "CRTP_H3DecodeAndTrimV2": "CRTP H3 Decode and Trim (V2)",
    "CRTP_H3SaveTake": "CRTP H3 Save Take",
    "CRTP_H3AssembleTakes": "CRTP H3 Assemble Takes",
    "CRTP_H3AssembleTakesV2": "CRTP H3 Assemble Takes (V2)",
}
