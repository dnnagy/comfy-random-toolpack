"""Drama 3 nodes. Optional inference stacks live in isolated worker processes."""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import tempfile
import threading
import time

MLX_MODEL = "fish-audio-drama-3-4bit"
CUDA_MODEL = "drama-3-w4a16-32gb"
PACK = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_SPEAKER = re.compile(r"<\|speaker:(\d+)\|>")


def _interrupt():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()


def _validate_audio(audio, max_seconds=60):
    import torch
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Reference must be a ComfyUI AUDIO input.")
    wave = audio["waveform"]
    rate = audio.get("sample_rate")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("Reference sample_rate must be a positive integer.")
    if not isinstance(wave, torch.Tensor) or wave.ndim != 3 or wave.shape[0] != 1:
        raise ValueError("Reference audio must have shape [1, channels, samples]; audio batches are unsupported.")
    if wave.shape[1] < 1 or wave.shape[2] < 1 or not torch.isfinite(wave).all():
        raise ValueError("Reference audio is empty or contains non-finite samples.")
    if wave.shape[-1] / rate > max_seconds:
        raise ValueError(f"Reference exceeds {max_seconds} seconds. Trim it before generation.")
    return wave, rate


def _references(reference_audio=None, reference_text="", references=None, max_seconds=60):
    refs = list(references or [])
    if reference_audio is not None:
        if refs:
            raise ValueError("Use either the single reference inputs or the reference chain, not both.")
        refs = [{"audio": reference_audio, "text": reference_text, "speaker_id": 0}]
    elif reference_text.strip():
        raise ValueError("Reference transcript supplied without reference audio.")
    if len(refs) > 16:
        raise ValueError("Drama 3 supports at most 16 speaker references.")
    ids = set()
    for ref in refs:
        speaker = ref["speaker_id"]
        if isinstance(speaker, bool) or not isinstance(speaker, int) or not 0 <= speaker <= 15 or speaker in ids:
            raise ValueError("Reference speaker IDs must be unique integers from 0 through 15.")
        ids.add(speaker)
        transcript = ref["text"].strip()
        if not _SPEAKER.sub("", transcript).strip() or any(int(tag) != speaker for tag in _SPEAKER.findall(transcript)):
            raise ValueError("Provide the reference transcript, with no conflicting speaker tags.")
        _validate_audio(ref["audio"], max_seconds)
    return refs


def _resolve_model(name, backend):
    import folder_paths
    path = Path(name).expanduser()
    roots = [Path(folder_paths.models_dir) / "tts"]
    if "tts" in folder_paths.folder_names_and_paths:
        roots.extend(Path(p) for p in folder_paths.get_folder_paths("tts"))
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        config = json.loads((candidate / "config.json").read_text())
        if backend == "mlx" and config.get("model_type") != "fish_drama3":
            raise ValueError("The MLX node needs the converted fish_drama3 checkpoint, not CUDA weights.")
        if backend == "cuda" and config.get("quantization_config", {}).get("quant_method") != "bsgemm-moe-w4a16-l8":
            raise ValueError("The CUDA node needs darioooooo0o/drama-3-w4a16-32gb (bsgemm-moe-w4a16-l8).")
        index = candidate / "model.safetensors.index.json"
        files = (set(json.loads(index.read_text())["weight_map"].values()) if index.is_file()
                 else {p.name for p in candidate.glob("model*.safetensors")})
        if not files:
            raise FileNotFoundError(f"No model safetensors found in {candidate}; complete the download first.")
        files.update(["tokenizer.json", "codec.safetensors" if backend == "mlx" else "codec.pth"])
        missing = [f for f in files if not (candidate / f).is_file() or (candidate / f).stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Incomplete Drama 3 download in {candidate}: {', '.join(sorted(missing))}")
        return str(candidate.resolve())
    raise FileNotFoundError(f"Drama 3 model not found: {name}. Install under ComfyUI/models/tts or enter an absolute directory.")


def _python(backend):
    override = os.environ.get(f"CRTP_DRAMA3_{backend.upper()}_PYTHON")
    path = Path(override).expanduser() if override else PACK / f".venv-drama3-{backend}" / "bin/python"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {backend.upper()} runtime: {path}. See docs/drama3.md or set CRTP_DRAMA3_{backend.upper()}_PYTHON.")
    return str(path)


def _stop_process_group(proc):
    # The worker and its SGLang children share this private session. Always
    # signal the group, even when the worker failed but left children alive.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=8)


def _run_worker(python, request, directory, timeout):
    request_path = Path(directory) / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Never let another Python environment leak into the isolated runtime.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen([python, str(PACK / "scripts/drama3_worker.py"), str(request_path)],
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                _interrupt()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Drama 3 exceeded the {timeout}s startup + generation timeout.")
                time.sleep(0.2)
            _interrupt()
            if proc.returncode:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 12000))
                raise RuntimeError("Drama 3 worker failed:\n" + log.read().decode(errors="replace"))
            log.seek(0, 2)
            log.seek(max(0, log.tell() - 2000))
            print("[CRTP Drama 3] " + log.read().decode(errors="replace"), flush=True)
        finally:
            _stop_process_group(proc)


class CRTP_Drama3Reference:
    CATEGORY = "CRTP/audio/Drama 3"
    FUNCTION = "append"
    RETURN_TYPES = ("CRTP_DRAMA3_REFERENCES",)
    RETURN_NAMES = ("references",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",),
                             "transcript": ("STRING", {"multiline": True}),
                             "speaker_id": ("INT", {"default": 0, "min": 0, "max": 15})},
                "optional": {"references": ("CRTP_DRAMA3_REFERENCES",)}}

    def append(self, audio, transcript, speaker_id=0, references=None):
        refs = [*(references or []), {"audio": audio, "text": transcript, "speaker_id": speaker_id}]
        return (_references(references=refs),)


class _Drama3Generate:
    CATEGORY = "CRTP/audio/Drama 3"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": ("STRING", {"default": MLX_MODEL if cls.BACKEND == "mlx" else CUDA_MODEL,
                                      "tooltip": "Directory under models/tts, or an absolute path. No automatic download."}),
            "text": ("STRING", {"multiline": True, "default": "This is a local test of Drama Three."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295, "control_after_generate": True,
                             "tooltip": "MLX sampling seed. CUDA: rerun/cache control only; upstream does not support request seeds."}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01}),
            "top_p": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 30, "min": 1, "max": 1024}),
            "max_new_tokens": ("INT", {"default": 1024, "min": 16, "max": 1024,
                                        "tooltip": "Frame limit per text chunk; 1024 is about 47 seconds. Long text is chunked."}),
            "timeout_seconds": ("INT", {"default": 1800, "min": 30, "max": 14400}),
        }, "optional": {
            "reference_audio": ("AUDIO",),
            "reference_text": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": "Exact words in reference_audio. Target words go in text."}),
            "references": ("CRTP_DRAMA3_REFERENCES",),
        }}

    def generate(self, model_path, text, seed=42, temperature=0.7, top_p=0.7, top_k=30,
                 max_new_tokens=1024, timeout_seconds=1800, reference_audio=None,
                 reference_text="", references=None, generation_mode="sentences", max_silence_seconds=6.0):
        if not text.strip():
            raise ValueError("Target text must not be empty.")
        if any(int(s) > 15 for s in _SPEAKER.findall(text)):
            raise ValueError("Target speaker IDs must be between 0 and 15.")
        if not (isinstance(seed, int) and 0 <= seed <= 4294967295):
            raise ValueError("Seed must be a uint32 integer.")
        if not (math.isfinite(temperature) and 0 < temperature <= 2 and math.isfinite(top_p) and 0 < top_p <= 1):
            raise ValueError("Invalid temperature or top_p.")
        if not (isinstance(top_k, int) and 1 <= top_k <= 1024 and isinstance(max_new_tokens, int) and 16 <= max_new_tokens <= 1024):
            raise ValueError("Invalid top_k or per-chunk frame limit.")
        if not 30 <= timeout_seconds <= 14400:
            raise ValueError("Timeout must be between 30 and 14400 seconds.")
        if generation_mode not in ("sentences", "native"):
            raise ValueError("Unknown generation_mode.")
        if not math.isfinite(max_silence_seconds) or not 0 <= max_silence_seconds <= 60:
            raise ValueError("max_silence_seconds must be between 0 and 60.")
        refs = _references(reference_audio, reference_text, references, 20 if self.BACKEND == "cuda" else 60)
        if self.BACKEND == "mlx" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
            raise RuntimeError("The MLX node requires an Apple Silicon Mac. Use the CUDA node on NVIDIA Linux.")
        if self.BACKEND == "cuda" and platform.system() != "Linux":
            raise RuntimeError("The CUDA W4A16 runtime requires NVIDIA Linux (including WSL2).")
        model = _resolve_model(model_path, self.BACKEND)
        python = _python(self.BACKEND)
        while not _LOCK.acquire(timeout=0.2):
            _interrupt()
        try:
            import numpy as np
            import soundfile as sf
            import torch
            import comfy.model_management as mm
            _interrupt()
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
            with tempfile.TemporaryDirectory(prefix="crtp-drama3-") as directory:
                serialized = []
                for i, ref in enumerate(refs):
                    wave, rate = _validate_audio(ref["audio"], 20 if self.BACKEND == "cuda" else 60)
                    path = Path(directory) / f"reference-{i}.wav"
                    mono = wave.detach().float().cpu()[0].mean(dim=0).numpy()
                    sf.write(path, mono, rate, subtype="FLOAT")
                    transcript = _SPEAKER.sub("", ref["text"]).strip()
                    serialized.append({"audio_path": str(path),
                                       "text": f"<|speaker:{ref['speaker_id']}|> {transcript}"})
                output = Path(directory) / "output.wav"
                request = {"backend": self.BACKEND, "model_path": model, "text": text,
                           "seed": seed, "temperature": temperature, "top_p": top_p, "top_k": top_k,
                           "max_new_tokens": max_new_tokens, "timeout_seconds": timeout_seconds,
                           "references": serialized, "output": str(output)}
                if self.BACKEND == "mlx":
                    request.update(generation_mode=generation_mode, max_silence_seconds=max_silence_seconds)
                _run_worker(python, request, directory, timeout_seconds)
                data, rate = sf.read(output, dtype="float32", always_2d=True)
                if data.size == 0 or not np.isfinite(data).all() or rate != 44100:
                    raise RuntimeError("Drama 3 returned empty, invalid, or unexpected-rate audio.")
                return ({"waveform": torch.from_numpy(data.T.copy()).unsqueeze(0), "sample_rate": rate},)
        finally:
            _LOCK.release()


class CRTP_Drama3MLXGenerate(_Drama3Generate):
    BACKEND = "mlx"
    DESCRIPTION = "Local Apple Silicon Drama 3. Sentence chunks reduce silent stalls; failed takes retry once with seed + 1."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["optional"].update({
            "generation_mode": (["sentences", "native"], {"default": "sentences",
                "tooltip": "Sentences: generate one sentence at a time with shared voice/history. Native: upstream paragraph chunks."}),
            "max_silence_seconds": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 60.0, "step": 0.5,
                "tooltip": "Reject sustained near-silence and retry the full take once with seed + 1. Set 0 to disable for intentional long pauses. This is not transcript verification."}),
        })
        return inputs


class CRTP_Drama3CUDAGenerate(_Drama3Generate):
    BACKEND = "cuda"
    DESCRIPTION = "Local NVIDIA Drama 3 W4A16 32 GB. Requires the patched SGLang-Omni runtime; starts and stops it per run."


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (CRTP_Drama3Reference, CRTP_Drama3MLXGenerate, CRTP_Drama3CUDAGenerate)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_Drama3Reference": "CRTP Drama 3 · Speaker Reference",
    "CRTP_Drama3MLXGenerate": "CRTP Drama 3 · Generate (MLX / Mac)",
    "CRTP_Drama3CUDAGenerate": "CRTP Drama 3 · Generate (CUDA / 32 GB)",
}
