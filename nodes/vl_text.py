"""Native Comfy VL or local llama.cpp GGUF+mmproj, selected lazily.

llama-server is an optional executable, not a Python dependency. Each GGUF job
owns its process, binds to loopback, and tears it down even on cancellation.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import gc
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

NATIVE = "Qwen3-VL-8B-Instruct"
GGUF = "Qwen3.8-27B-HauhauCS-Q5_K_P"
GEMMA = "Gemma-4-E4B-HauhauCS-Q8_K_P"
GEMMA_MODEL_FILE = "Gemma-4-E4B-HauhauCS-Q8_K_P/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
GEMMA_PROJECTOR_FILE = "Gemma-4-E4B-HauhauCS-Q8_K_P/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
LEGACY_GGUF = "Qwen3.8-27B-UD-Q5_K_M"
MODEL_FILE = "Qwen3.8-27B-HauhauCS-Q5_K_P/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf"
PROJECTOR_FILE = "Qwen3.8-27B-HauhauCS-Q5_K_P/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
_LOCK = threading.Lock()


def _interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _resolve_gguf(name):
    import folder_paths
    # Absolute paths are useful on desktop; relative paths search models/llm
    # and any llm roots configured in extra_model_paths.yaml.
    roots = [Path(folder_paths.models_dir) / "llm"]
    if "llm" in folder_paths.folder_names_and_paths:
        roots.extend(map(Path, folder_paths.get_folder_paths("llm")))
    candidates = [Path(name)] if Path(name).is_absolute() else [p / name for p in roots]
    for path in candidates:
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError(f"Not a complete GGUF file: {path}")
            return str(path.resolve())
    raise FileNotFoundError(f"Missing GGUF model/projector: {name}. Install it under ComfyUI/models/llm.")


def _server_binary():
    configured = os.environ.get("CRTP_LLAMA_SERVER")
    bundled = Path(__file__).resolve().parents[1] / "bin" / "llama-server"
    candidate = configured or shutil.which("llama-server") or str(bundled)
    path = shutil.which(candidate)
    if not path:
        raise FileNotFoundError("Install llama-server with multimodal support, or set CRTP_LLAMA_SERVER to its executable path.")
    return path


def _request(url, token, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    # Loopback inference must never go through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc


def _command(binary, model, projector, port, token, context_size, image_max_tokens):
    return [binary, "--model", model, "--mmproj", projector,
            "--host", "127.0.0.1", "--port", str(port), "--api-key", token,
            "--ctx-size", str(context_size), "--parallel", "1",
            "--n-gpu-layers", "99", "--image-max-tokens", str(image_max_tokens),
            "--jinja", "--reasoning-format", "deepseek", "--no-context-shift"]


@contextmanager
def _server(model, projector, context_size=16384, image_max_tokens=1024):
    binary = _server_binary()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_hex(24)
    url = f"http://127.0.0.1:{port}"
    startup_timeout = float(os.environ.get("CRTP_LLAMA_START_TIMEOUT", "600"))
    with tempfile.TemporaryFile(mode="w+b") as log:
        # Do not inherit LLAMA_ARG_* overrides from an unrelated server setup.
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        proc = subprocess.Popen(_command(binary, model, projector, port, token,
                                        context_size, image_max_tokens),
                                stdout=log, stderr=log, env=env)
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                _interrupt()
                if proc.poll() is not None:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 6000))
                    raise RuntimeError("llama-server failed to start: " + log.read().decode("utf-8", errors="replace"))
                try:
                    if _request(url + "/health", token, timeout=1).get("status") == "ok":
                        break
                except (OSError, RuntimeError):
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("llama-server model/projector startup timed out")
            yield url, token
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


def _image_content(image):
    from PIL import Image
    import numpy as np
    if image is None or len(image.shape) != 4 or not 1 <= image.shape[0] <= 10:
        raise ValueError("Supply an IMAGE batch containing 1–10 images")
    content = []
    for index, frame in enumerate(image):
        pixels = (frame.detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pixels).convert("RGB").save(buf, format="PNG")
        content.append({"type": "text", "text": f"Image {index + 1}:"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}})
    return content


def _payload(prompt, content, max_length, sampling_mode, temperature, top_k,
             top_p, min_p, repetition_penalty, seed, presence_penalty, thinking):
    sample = sampling_mode == "on"
    return {"model": "local", "messages": [{"role": "user", "content": [
                *content, {"type": "text", "text": prompt}]}],
            "max_tokens": max_length, "stream": False,
            "temperature": temperature if sample else 0.0,
            "top_k": top_k if sample else 0, "top_p": top_p if sample else 1.0,
            "min_p": min_p if sample else 0.0,
            "repeat_penalty": repetition_penalty if sample else 1.0,
            "presence_penalty": presence_penalty if sample else 0.0,
            "seed": seed if sample else 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}


def _complete(url, token, payload):
    timeout = float(os.environ.get("CRTP_LLAMA_GENERATION_TIMEOUT", "1800"))
    # Poll interruption while the HTTP request is running. On cancellation the
    # enclosing context terminates the process, releasing the pending request.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_request, url + "/v1/chat/completions", token, payload, timeout)
    deadline = time.monotonic() + timeout
    try:
        while True:
            _interrupt()
            if time.monotonic() >= deadline:
                raise TimeoutError("llama-server generation timed out")
            try:
                result = future.result(timeout=0.25)
                break
            except FutureTimeout:
                if future.done():
                    raise
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("llama-server returned no choices")
        message = choices[0].get("message", {})
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(text, str) or not isinstance(reasoning, str):
            raise RuntimeError("llama-server returned malformed text")
        if reasoning:
            text = f"<think>\n{reasoning}\n</think>\n\n{text}"
        if not text.strip():
            raise RuntimeError("llama-server returned no text; increase the token limit if thinking is enabled")
        return text
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


class CRTP_VLTextGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ([NATIVE, GGUF, GEMMA, LEGACY_GGUF], {"default": NATIVE}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "image": ("IMAGE",),
            "max_length": ("INT", {"default": 512, "min": 1, "max": 4096}),
            "sampling_mode": (["off", "on"], {"default": "off"}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2}),
            "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1}),
            "min_p": ("FLOAT", {"default": 0.05, "min": 0, "max": 1}),
            "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.01, "max": 5}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": False}),
            "presence_penalty": ("FLOAT", {"default": 0, "min": 0, "max": 5}),
            "thinking": ("BOOLEAN", {"default": False}),
            "use_default_template": ("BOOLEAN", {"default": True, "tooltip": "GGUF vision requires the model chat template."}),
            "gguf_model": ("STRING", {"default": MODEL_FILE}),
            "mmproj": ("STRING", {"default": PROJECTOR_FILE}),
        }, "optional": {"clip": ("CLIP", {"lazy": True}),
            "gemma_gguf_model": ("STRING", {"default": GEMMA_MODEL_FILE}),
            "gemma_mmproj": ("STRING", {"default": GEMMA_PROJECTOR_FILE})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "CRTP/text"

    def check_lazy_status(self, model, clip=None, **kwargs):
        return ["clip"] if model == NATIVE and clip is None else []

    def generate(self, model, prompt, image, max_length=512, sampling_mode="off",
                 temperature=0.7, top_k=64, top_p=0.95, min_p=0.05,
                 repetition_penalty=1.05, seed=0, presence_penalty=0.0,
                 thinking=False, use_default_template=True,
                 gguf_model=MODEL_FILE, mmproj=PROJECTOR_FILE, clip=None,
                 gemma_gguf_model=GEMMA_MODEL_FILE, gemma_mmproj=GEMMA_PROJECTOR_FILE):
        if model not in (NATIVE, GGUF, GEMMA, LEGACY_GGUF):
            raise ValueError(f"Unknown VL model: {model}")
        if sampling_mode not in ("on", "off"):
            raise ValueError("Sampling mode must be on or off")
        if model == NATIVE:
            if clip is None:
                raise ValueError("Connect the existing Qwen3-VL-8B CLIP loader")
            tokens = clip.tokenize(prompt, image=image, skip_template=not use_default_template,
                                   min_length=1, thinking=thinking)
            generated = clip.generate(tokens, do_sample=sampling_mode == "on", max_length=max_length,
                temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                seed=seed, mtp=False)
            return (clip.decode(generated),)
        if not use_default_template:
            raise ValueError("GGUF vision requires 'Use model chat template' enabled")
        if model == GEMMA:
            gguf_model, mmproj = gemma_gguf_model, gemma_mmproj
        if not gguf_model or not mmproj:
            raise ValueError("Both the GGUF model and matching mmproj are required for vision")
        model_path, projector_path = _resolve_gguf(gguf_model), _resolve_gguf(mmproj)
        if model_path == projector_path:
            raise ValueError("The model and mmproj must be separate files")
        _server_binary()  # Fail before unloading models when runtime is absent.
        payload = _payload(prompt, _image_content(image), max_length, sampling_mode,
                           temperature, top_k, top_p, min_p, repetition_penalty, seed,
                           presence_penalty, thinking)
        while not _LOCK.acquire(timeout=0.25):
            _interrupt()
        try:
            import comfy.model_management as mm
            _interrupt()
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            with _server(model_path, projector_path) as (url, token):
                return (_complete(url, token, payload),)
        finally:
            _LOCK.release()


NODE_CLASS_MAPPINGS = {"CRTP_VLTextGenerate": CRTP_VLTextGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTP_VLTextGenerate": "CRTP Vision Language · Model Selector"}
