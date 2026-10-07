# Vision-language model selector

`CRTP_VLTextGenerate` has a model dropdown:

- `Qwen3-VL-8B-Instruct`: uses a lazily connected native ComfyUI CLIP loader.
- `Qwen3.8-27B-UD-Q5_K_M`: runs a local llama.cpp server with a GGUF language
  model **and its matching vision projector**. The CLIP branch is not evaluated.

Put these files under `ComfyUI/models/llm/Qwen3.8-27B-UD-Q5_K_M/`:

- `Qwen3.8-27B-UD-Q5_K_M.gguf`
- `mmproj-F16.gguf`

Use the same revision of `unsloth/Qwen3.8-27B-GGUF` for both. The `gguf_model`
and `mmproj` widgets also accept absolute paths, or paths relative to additional
`llm` roots configured in `extra_model_paths.yaml`. There is no text-only fallback
if the projector is absent. The complete image batch (1–10 RGB images) is sent as
ordered image inputs, with the prompt, to the local multimodal endpoint.

Install current multimodal `llama-server` on PATH, set `CRTP_LLAMA_SERVER` to its
executable, or place/link the executable at `comfy-random-toolpack/bin/llama-server`.
The LosFeliz worker builds the CUDA runtime from pinned llama.cpp commit
`8a118ee86c3b818ce7e1524e48fc7cc65f1dc69b`. CPU-only llama.cpp works but is slow.
This node does not require llama-cpp-python or a separately managed server.

The 27B branch unloads Comfy models, clears the torch cache, then launches one
loopback-only authenticated subprocess. Context is bounded to 16,384 tokens,
parallelism to one, and image tokens to 1,024 per image. Context shifting is
disabled: oversized requests fail rather than silently discarding references.
The process exits after every request (including failure/cancellation), releasing
GPU memory for later Comfy jobs. Model load time is paid on each uncached run.
Do not run concurrent unrelated GPU jobs when relying on this memory budget.

Sampling, seed, penalties, thinking, and response-token limit are shared controls.
Sampling off is greedy with neutral penalties on the GGUF path; native 8B retains
Comfy's normal greedy behavior. GGUF multimodal requests require the model chat
template. Disabling `use_default_template` on that branch is an explicit error.
Thinking output, if returned separately by llama.cpp, is included in `<think>`
tags; the token limit includes thinking. No MTP draft model is loaded.

`CRTP_LLAMA_START_TIMEOUT` defaults to 600 seconds and
`CRTP_LLAMA_GENERATION_TIMEOUT` to 1800 seconds. Cancellation is polled during
startup and generation. Prompt/image payloads never use an external service.

Restart ComfyUI after updating the pack. Run the CPU integration tests with:

```sh
python -m unittest discover -s tests -p test_vl_text.py
```

For an opt-in inference test with synthetic images, run
`tests/vl_native_integration.py --model /absolute/model.gguf --mmproj /absolute/mmproj.gguf`.
Use `--images 10` to exercise the maximum image batch. This test loads the real
weights and requires enough GPU/unified memory; it is not part of the CPU suite.
