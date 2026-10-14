"""Private one-job worker for the ComfyUI Drama 3 nodes; no Comfy imports.

Run with the backend's isolated Python and one JSON request path. The caller
owns the process session and terminates the entire session on cancellation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def sentence_chunks(text, native_split):
    """Keep sentences separate while preserving speaker IDs and inline scopes.

    The native splitter packs several sentences into one ~300-byte chunk, which
    can make this MLX checkpoint dwell in silence at punctuation. Use the native
    generator's first/previous-chunk history, but do not pack sentences together.
    """
    protected = re.compile(r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S)
    masked = protected.sub(lambda m: "_" * len(m.group()), text)
    cuts = {len(text)}
    for match in re.finditer(r"[.!?…。！？]+[\"'”’»)]*(?=\s|$)", masked):
        # Avoid splitting common titles and initialisms. Decimal points don't
        # match because they are followed by a digit rather than whitespace.
        prefix = text[:match.end()].rstrip()
        if re.search(r"(?:\b(?:dr|mr|mrs|ms|prof|st)|\b[A-Za-z]|\b(?:[A-Za-z]\.)+[A-Za-z])\.$", prefix, re.I):
            continue
        cuts.add(match.end())
    speakers = re.compile(r"<\|speaker:(\d+)\|>")
    cuts.update(m.start() for m in speakers.finditer(text) if m.start())
    chunks, start, speaker, pending = [], 0, None, ""
    for end in sorted(cuts):
        part = (pending + text[start:end]).strip()
        start, pending = end, ""
        tags = list(speakers.finditer(part))
        if tags:
            speaker = tags[-1].group()
        # A direction by itself belongs to the next spoken sentence.
        spoken = re.sub(r"<[^>]*>|\[[^\[\]]*\]", "", part)
        if not any(c.isalnum() for c in spoken):
            pending = part + " "
            continue
        if speaker and not speakers.search(part):
            part = speaker + " " + part
        chunks.extend(native_split(part))
    if pending.strip():
        if chunks:
            chunks[-1] += " " + pending.strip()
        else:
            chunks = native_split(text)
    return chunks


class GenerationFailure(RuntimeError):
    """A failed take that can be retried with a different sampling seed."""


class SilenceGuard:
    """Detect sustained near-silence; never trim or pretend missing words exist."""

    def __init__(self, sample_rate, seconds):
        import numpy as np
        self.rate, self.seconds = sample_rate, seconds
        self.window = max(1, round(sample_rate * 0.1))
        self.pending = np.empty(0, dtype=np.float32)
        self.peak, self.quiet = 0.0, 0

    def feed(self, audio):
        import numpy as np
        if self.seconds == 0:
            return
        data = np.concatenate((self.pending, audio))
        end = len(data) // self.window * self.window
        for block in data[:end].reshape(-1, self.window):
            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            self.peak = max(self.peak, rms)
            threshold = max(0.003, min(0.01, self.peak * 0.05))
            self.quiet = self.quiet + self.window if rms < threshold else 0
            if self.quiet / self.rate >= self.seconds:
                raise GenerationFailure(f"Drama 3 generated {self.seconds:g}s of sustained near-silence. "
                                        "The take stalled; no partial audio will be returned.")
        self.pending = data[end:]


def mlx_references(refs, sample_rate):
    """Read our WAV handoff and avoid the pinned runtime's linear upsampler."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audios = []
    for ref in refs:
        audio, rate = sf.read(ref["audio_path"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != sample_rate:
            divisor = math.gcd(rate, sample_rate)
            audio = resample_poly(audio, sample_rate // divisor, rate // divisor)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if not np.isfinite(audio).all() or peak < 1e-6:
            raise ValueError("Reference audio is empty, silent, or non-finite.")
        # Codec conditioning is sensitive to reference gain. Comfy's arithmetic
        # stereo mean and equal-power decoders otherwise yield different voices/
        # stalls from the same recording. Set a consistent peak without clipping.
        audio = audio * (0.95 / peak)
        audios.append(np.asarray(audio, dtype=np.float32))
    return audios


def run_mlx(request):
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load_model
    import mlx_audio.tts.models.fish_drama3.drama3 as drama

    print("Loading Drama 3 MLX", flush=True)
    model = load_model(request["model_path"])
    refs = request["references"]
    kwargs = {"text": request["text"], "max_tokens": request["max_new_tokens"],
              "temperature": request["temperature"], "top_p": request["top_p"],
              "top_k": request["top_k"], "stream": True}
    if refs:
        kwargs.update(ref_audio=mlx_references(refs, model.sample_rate),
                      ref_text=[ref["text"] for ref in refs])
    native_split = drama.split_text
    mode = request.get("generation_mode", "sentences")
    if mode == "sentences":
        # This is a private worker process. Restore the pinned upstream hook even
        # on failure; no shared ComfyUI modules or installed packages are edited.
        drama.split_text = lambda text: sentence_chunks(text, native_split)
    elif mode != "native":
        raise ValueError("Unknown MLX generation_mode.")
    try:
        expected_segments = len(drama.split_text(request["text"]))
        for attempt in range(2):
            seed = (request["seed"] + attempt) % 2**32
            mx.random.seed(seed)
            samples, frames_per_segment = 0, {}
            guard = SilenceGuard(model.sample_rate, request.get("max_silence_seconds", 6.0))
            started = time.monotonic()
            try:
                # Reopening discards the entire failed take before retrying.
                with sf.SoundFile(request["output"], "w", samplerate=model.sample_rate,
                                  channels=1, subtype="FLOAT") as output:
                    for result in model.generate(**kwargs):
                        segment = result.segment_idx
                        frames_per_segment[segment] = frames_per_segment.get(segment, 0) + result.token_count
                        if frames_per_segment[segment] >= request["max_new_tokens"]:
                            raise GenerationFailure("Drama 3 reached the per-chunk frame limit without finishing. "
                                                    "Shorten the passage or raise max_new_tokens.")
                        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                        if result.sample_rate != model.sample_rate or not np.isfinite(audio).all():
                            raise RuntimeError("Invalid MLX audio chunk.")
                        guard.feed(audio)
                        output.write(audio)
                        samples += audio.size
                if not samples:
                    raise GenerationFailure("Drama 3 generated no audio.")
                if any(frames_per_segment.get(i, 0) == 0 for i in range(expected_segments)):
                    raise GenerationFailure("Drama 3 skipped a text chunk without generating audio.")
            except GenerationFailure as exc:
                Path(request["output"]).unlink(missing_ok=True)
                print(f"MLX take {attempt + 1} (seed {seed}) failed: {exc}", flush=True)
                if attempt == 1:
                    raise
                print("Retrying the complete text once with seed + 1.", flush=True)
                continue
            print(json.dumps({"audio_seconds": samples / model.sample_rate,
                              "generation_seconds": time.monotonic() - started,
                              "generation_mode": mode, "effective_seed": seed,
                              "chunks": len(frames_per_segment),
                              "peak_memory_gib": mx.get_peak_memory() / 1024**3}), flush=True)
            return
    finally:
        drama.split_text = native_split


def cuda_command(binary, request, port):
    return [str(binary), "serve", "--model-path", request["model_path"],
            "--host", "127.0.0.1", "--port", str(port),
            "--allowed-local-media-path", str(Path(request["output"]).parent),
            "--tts_engine.gpu_memory_fraction", "0.50", "--vocoder.gpu_memory_fraction", "0.12",
            "--tts_engine.factory.max_new_tokens", str(request["max_new_tokens"]),
            "--preprocessing.total_reserve_bytes", "3GiB",
            "--preprocessing.factory.max_reference_audio_seconds", "20"]


def cuda_payload(request):
    # Drama3SpeechPreprocessor explicitly rejects per-request seed. The Comfy
    # seed widget is only a cache invalidator on this backend, never sent here.
    return {"input": request["text"], "references": request["references"],
            "response_format": "wav", "stream": False,
            "max_new_tokens": request["max_new_tokens"],
            "temperature": request["temperature"], "top_p": request["top_p"],
            "top_k": request["top_k"]}


def _http(opener, url, payload=None, timeout=2):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang-Omni HTTP {exc.code}: {detail}") from exc


def run_cuda(request):
    import sglang_omni

    # Let SGLang validate the GPU; probing with torch here creates another
    # CUDA context competing for the 32 GB card's limited free memory.
    quant_file = Path(sglang_omni.__file__).parent / "models/fishaudio_drama3/quantization.py"
    if "ExpertW4A16L8Config" not in quant_file.read_text():
        raise RuntimeError("Missing W4A16-L8 loader patch. Run scripts/setup_drama3.py cuda; see docs/drama3.md.")
    binary = Path(sys.executable).with_name("sgl-omni")
    if not binary.is_file():
        raise FileNotFoundError(f"Missing CUDA runtime executable: {binary}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + request["timeout_seconds"]
    # Inherit the worker's session so Comfy cancellation also kills engine and
    # vocoder children. SGLang logs go into the caller's bounded error report.
    proc = subprocess.Popen(cuda_command(binary, request, port))
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang-Omni exited during startup ({proc.returncode}); see startup log above.")
            try:
                with _http(opener, url + "/health") as response:
                    if response.status == 200:
                        break
            except (OSError, RuntimeError):
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError("SGLang-Omni did not become ready before the timeout.")
        print("Generating Drama 3 CUDA audio (per-request seeds unsupported upstream)", flush=True)
        with _http(opener, url + "/v1/audio/speech", cuda_payload(request),
                   timeout=max(1, deadline - time.monotonic())) as response, open(request["output"], "wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=8)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    if request["backend"] == "mlx":
        run_mlx(request)
    elif request["backend"] == "cuda":
        run_cuda(request)
    else:
        raise ValueError("Unsupported Drama 3 backend.")


if __name__ == "__main__":
    main()
