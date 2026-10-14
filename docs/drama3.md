# Drama 3 on Apple Silicon and NVIDIA CUDA

Three nodes in **CRTP/audio/Drama 3**:

| Node | Runtime | Checkpoint |
| --- | --- | --- |
| CRTP Drama 3 · Generate (MLX / Mac) | MLX Audio, Apple Silicon | `mlx-community/fish-audio-drama-3-4bit` |
| CRTP Drama 3 · Generate (CUDA / 32 GB) | Patched SGLang-Omni, NVIDIA Linux/WSL2 | `darioooooo0o/drama-3-w4a16-32gb` |
| CRTP Drama 3 · Speaker Reference | ComfyUI | Combines an AUDIO recording, its transcript, and a speaker ID |

Both generation nodes return ordinary ComfyUI `AUDIO` at 44.1 kHz, with shape
`[1, channels, samples]`. Connect it to Preview Audio, Save Audio, or video nodes.
Neither node calls Fish Audio's hosted API. Models are local directories; node
execution never downloads weights or installs packages.

## Quick workflows

Open [Drama3-MLX.json](../examples/drama3/Drama3-MLX.json) or
[Drama3-CUDA.json](../examples/drama3/Drama3-CUDA.json) in ComfyUI. After setup,
queue for plain TTS. An unconnected Load Audio node is provided for cloning:
select/upload your recording, connect it to `reference_audio`, and fill in
`reference_text` with exactly what the recording says. Put the new words in `text`.
You do not supply target audio. Restart ComfyUI after installing these nodes.

Matching `Drama3-MLX.api.json` and `Drama3-CUDA.api.json` files in the same
directory are API prompt objects for automation (wrap in `{"prompt": ...}`
when posting to ComfyUI's `/prompt`). They are not editor workflow files.

For multiple speakers, chain Speaker Reference nodes with different IDs (0–15),
then connect the last `references` output to either Generate node. Write target
text such as `<|speaker:0|> Are you coming? <|speaker:1|> Give me a minute.`
The helper adds speaker tags to transcripts. Do not also connect the single
reference inputs. Duplicate IDs and conflicting transcript tags are rejected.

Delivery cues go inline, for example `[sighs] Fine. [laughs] Maybe you're right.`
Only one reference recording per speaker is supported by this helper. Audio
batches are rejected; stereo references are downmixed to mono. The MLX worker
normalizes each reference to a 0.95 peak and resamples to 44.1 kHz with SciPy's
polyphase FIR filter, avoiding the pinned
MLX Audio runtime's linear-upsample artifacts. CUDA handles resampling upstream. Use clean references
at 44.1 kHz or above when possible; upsampling a poor recording adds no detail.

## MLX setup

Requires an Apple Silicon Mac, Python 3.12 (managed by uv), Git, and `uv` on PATH.
From the toolpack directory:

```bash
python3 scripts/setup_drama3.py mlx \
  --model-dir /absolute/path/to/fish-audio-drama-3-4bit
```

This installs a pinned Drama 3 MLX Audio revision into `.venv-drama3-mlx` without
changing ComfyUI's dependencies. The checkpoint is about 22 GB; the publisher
reports about 26 GB peak generation memory. The M4 Max with 64 GB is suitable.
Set `model_path` to the absolute model directory, or symlink the whole directory
into `ComfyUI/models/tts/fish-audio-drama-3-4bit` and use the default node value.
Configured `tts` roots in `extra_model_paths.yaml` are also searched.

For a separately installed runtime, set `CRTP_DRAMA3_MLX_PYTHON` to that
environment's absolute `bin/python` path before starting ComfyUI. An ordinary
MLX Audio release may not yet contain Drama 3; use the pinned setup script.

## CUDA setup for RTX 5090

Run setup **on the NVIDIA Linux worker**, from the toolpack directory:

```bash
python3 scripts/setup_drama3.py cuda \
  --model-dir /app/ComfyUI/models/tts/drama-3-w4a16-32gb
```

Use your actual ComfyUI model directory if different. This installs the pinned
SGLang-Omni source in `.drama3-runtime`, applies the model author's quantization
patch after checking its SHA-256, and installs an isolated `.venv-drama3-cuda`.
The checkpoint download is pinned too. It does not replace a system driver or
CUDA toolkit. The selected SGLang-Omni revision has **CUDA 13 dependencies**;
use a compatible driver and CUDA development environment (including a compiler
for Marlin). The older model card's CUDA 12.8 minimum alone does not establish
compatibility with this newer runtime pin. Installation and GPU inference must
be validated on the deployment host.

For another installed environment, set `CRTP_DRAMA3_CUDA_PYTHON` to its Python.
It must have `sgl-omni` beside that executable and the W4A16-L8 patch applied.
The worker checks for that patch before loading weights. Use `CUDA_VISIBLE_DEVICES`
when launching ComfyUI to select the target card; SGLang uses visible GPU 0.

The node starts a private loopback server per run with the model author's profile:

```text
--tts_engine.gpu_memory_fraction 0.50
--vocoder.gpu_memory_fraction 0.12
--tts_engine.factory.max_new_tokens 1024
--preprocessing.total_reserve_bytes 3GiB
--preprocessing.factory.max_reference_audio_seconds 20
```

It posts to `/v1/audio/speech`, collects the WAV, and stops the server. References
longer than 20 seconds are rejected before startup. The maximum frame count is
per text chunk: 1024 is roughly 47 seconds; longer text is chunked upstream.
The author's 30.7 GiB measurement was on an A6000, **not a 5090**. This profile
is tight and must be measured on a real 5090. Keep other GPU workloads off it.
The newer runtime pin may have different memory use; do not treat fit as proven.

**CUDA seeds:** SGLang-Omni's Drama 3 preprocessor rejects per-request seeds.
The node's `seed` widget only invalidates ComfyUI's cached output to request a
new take on CUDA. It is not sent as a sampling seed. MLX uses an actual sampling
seed, but different runtime versions or hardware can still change results.

## Memory and cancellation

Nodes serialize Drama 3 runs, unload Comfy-managed models before inference, and
use separate processes so MLX/SGLang dependencies do not conflict with ComfyUI.
Workers are stopped after every run, error, timeout, or ComfyUI cancellation.
CUDA worker descendants are included in the process-group cleanup. Reference
WAVs and request files live in a temporary directory and are removed afterward.
Startup plus generation share the `timeout_seconds` budget (default 30 minutes).

This deliberately reloads the model per run: the upstream cached-voice first-audio
latency does not apply. ComfyUI returns the completed audio, not a live stream.
MLX consumes streaming chunks internally to keep decoding memory bounded.
### MLX silence mitigation

The community checkpoint can generate a word or sentence followed by long
near-silence, including with a clean reference. Increasing `max_new_tokens`
does not fix that failure. It also occurs outside ComfyUI and with full rather
than streaming audio decoding.

`generation_mode=sentences` is now the default. It generates individual
sentences using the upstream first/previous-chunk history to preserve the voice
and context. It keeps speaker IDs, bracketed cues, and paired-tag scopes intact.
Long sentences still use the native size limit; punctuation inside a paired
scope is not split. Sentence detection is heuristic, so unusual abbreviations
may introduce extra boundaries. `native` restores upstream paragraph chunks
for comparisons or performances requiring continuous multi-sentence delivery.

`max_silence_seconds=6` rejects sustained near-silence. Set it to `0` to disable
this check for intentionally long pauses, or raise it as needed. The threshold uses
the take's level with a 0.003 RMS floor to catch an initially silent/noisy take.
This is a heuristic: background noise can
hide a stall, and very quiet performances can trigger it. It does not transcribe
or prove that every word was spoken. No silence is trimmed automatically.

A skipped/empty chunk, exhausted frame budget, or detected silent stall discards
the entire take and retries once with `seed + 1` (wrapping at uint32). Logs show
the effective seed. A second failure raises an error and returns no partial
audio. Normal pauses and successful takes are preserved. These changes only
apply to MLX; CUDA generation is unchanged.

## Verification

Run with ComfyUI's Python, with torch, numpy, soundfile, and scipy:

```bash
python -m unittest discover -s tests -p 'test_drama3.py' -v
```

Tests cover reference validation, stereo conversion, sample rates, format/shard
checks, MLX chunk assembly, a real local CUDA-protocol HTTP round trip, error
propagation, and process cancellation. HTTP tests use a fake inference server;
they do not establish CUDA GPU compatibility or speech quality.

Local validation on 2026-10-14 used an M4 Max with 64 GB, MLX 0.32.3,
Transformers 5.19.0, and the pinned MLX Audio source. The full toolpack suite
passed 54 tests and both API examples passed ComfyUI's prompt validator.
The default single-sentence take completed (2.55 seconds of audio, 23.8 GiB
reported peak MLX memory, about 35 seconds including model loading). The
upstream cookbook's reference-conditioned test generated 2.37 seconds and
Whisper transcribed the requested sentence exactly. This is a functional
smoke test, not a perceptual voice-similarity benchmark. Two earlier takes
with multi-sentence text stalled after the first sentence; those failures
motivated the frame-limit check. CUDA GPU inference remains untested here.

Follow-up silence investigation on the same Mac reproduced a one-word failure
with a neutral English multi-sentence prompt. Full and streaming decoding of
the same codes agreed within 1.3e-6, locating the silence in generated tokens,
not WAV concatenation. Sentence chunks completed that English passage, a
Hungarian three-sentence passage, and a longer single English sentence in local
ASR checks. A user-supplied Hungarian coffee-invitation prompt and 35-second
reference failed with native chunks at seeds 42, 43, and 44; sentence chunks
produced all three sentences at all three seeds. Whisper base recovered the
sentences with spelling/recognition errors, so these checks establish improved
completion, not perfect pronunciation. Some multi-second pauses remain.
The updated regression suite covers silence detection,
whole-take retries, missing-chunk errors, scope/speaker preservation, and
reference-resampling quality.

Reference-level follow-up: ComfyUI's arithmetic stereo downmix produced a
quieter reference than FFmpeg's equal-power downmix. With the exact same supplied
Hungarian text, the quieter reference could still stall even with sentence
chunks. Normalizing reference peak level to 0.95 completed the prompt with its
directions intact in the diagnostic run. Reference preprocessing now applies
that normalization, without trimming pauses or changing transcripts.

The final installed-node test used ComfyUI's actual audio loader and the supplied
Hungarian prompt/reference at seed 42. It completed three chunks in 11.94 seconds
of audio (42.12 seconds including startup, 25.95 GiB peak MLX memory). The output
was sample-identical to the normalized diagnostic clip whose three sentences
were recovered by local ASR. The full toolpack suite passed 63 tests.

## Upstream sources and licenses

- [MLX checkpoint and model card](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit)
- [MLX Audio Drama 3 implementation](https://github.com/stickerdaniel/mlx-audio/tree/478f3189974fb7dcafe886afba09887f03996aec)
- [CUDA checkpoint and author patch](https://huggingface.co/darioooooo0o/drama-3-w4a16-32gb)
- [SGLang-Omni cookbook](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)

These nodes are adapters; they do not redistribute model weights or upstream
runtime source. Checkpoint LICENSE and NOTICE files remain with each download.
The Fish Audio Community License applies to the models and includes separate
terms for commercial use and hosted services.
