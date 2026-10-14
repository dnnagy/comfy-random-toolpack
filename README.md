# Comfy Random Toolpack

A grab-bag of small, useful ComfyUI custom nodes.

## Nodes

### CRTP MiniMax H3 Continuation Nodes

These nodes are self-contained and use only ComfyUI core's native MiniMax H3
conditioning. Neither `ComfyUI_MiniMax_H3_Extender` nor
`ComfyUI-H3-Motion-Context` is required.

- **CRTP_H3ContinuationTail** — resamples the final source window to H3's
  24 fps timeline, aspect-fits it to the generation canvas, and extracts a
  valid 5, 22, 39, or 56-frame native Ref2VA video reference plus its exact
  final frame.
- **CRTP_H3ContinuationLength** — adds the guide overlap to the requested new
  video and snaps the combined sampling length to H3's 17k+5 frame grid.
- **CRTP_H3AlignedAudioTail** — extracts the source-audio window exactly
  aligned to the selected final video frames.
- **CRTP_H3TrimContinuationAV** — removes the native multi-frame guide overlap
  and its time-aligned audio prefix after ComfyUI AV decoding.

The source tail should be connected both to core
`MiniMaxH3ReferenceToVideo` as `<Video 1>` and to core `MiniMaxH3AddGuide` as
the frame-zero hard AV guide. The source's complete soundtrack and optional
new voices use Ref2VA's standalone audio inputs as `<Audio 1>` through
`<Audio 3>`. The requested duration represents new continuation material
after the guide is trimmed. Use 22 guide/reference frames normally and 5 for a
source shorter than roughly one second. This graph requires ComfyUI commit
`e01fb4c56b7a88149d469b99cbbfe3223d715054` or newer.

### CRTP_ReferenceImagePicker

Pick a reference image deterministically from a versioned manifest, or load an
exact stable asset ID. Each node has its own `catalog_path` input, so faces,
clothing, and backgrounds can live in separate directories.

- Inputs: `catalog_path`, `catalog` (`name@version`), `category`, `value`,
  `selection_seed` (ordinary INT), optional `asset_id`.
- Outputs: `image` (IMAGE), `asset_id` (STRING), `description` (STRING).
- Manifest and selected-file content hashes invalidate cached results.
- Includes a standalone resolver for saving references when creating jobs.

See [catalog setup, manifest template, and H3 integration](docs/reference-picker.md).

### CRTP_LazyLatentFallback

Pick between two `LATENT` inputs *without* evaluating both upstream branches.
Built on top of ComfyUI's
[lazy input evaluation](https://docs.comfy.org/custom-nodes/backend/lazy_evaluation),
so the unselected branch is never executed. This makes it safe to wire up a
VAE encode that would otherwise crash on a missing input.

Inputs:

- `use_real_latent` (BOOLEAN): If `True`, output `real_latent`; otherwise output `empty_latent`.
- `real_latent` (LATENT, lazy, optional): The "real" latent (e.g. from a VAE encode).
- `empty_latent` (LATENT, lazy, optional): A fallback empty latent.

Output:

- `latent` (LATENT)

Typical pattern:

```text
             ┌─ valid source → VAE Encode ─────────┐
has_source ──┤                                     ├─ CRTP_LazyLatentFallback → LATENT
             └─ Empty Latent ──────────────────────┘
```

Because `real_latent` is lazy, the VAE encode branch is only executed when
`use_real_latent` is `True`, and the empty latent branch is only executed when
it is `False`.

### CRTP_PadImageToDivisible

Pad an `IMAGE` (or video frame batch — same `[B, H, W, C]` layout) so both
width and height are multiples of `divisor`. Useful for models like LTX-Video
2.3, which requires both dimensions to be divisible by 32 (e.g. 1280×720 →
1280×736).

Inputs:

- `image` (IMAGE)
- `divisor` (INT, default `32`): pad up to a multiple of this value.
- `mode` (`symmetric` | `asymmetric`): `symmetric` splits padding between both
  sides (e.g. 720 → 736 becomes 8 top + 8 bottom). `asymmetric` puts all
  padding on one side.
- `asymmetric_side` (`end` | `start`): only used in `asymmetric` mode.
  `end` = pad bottom and right; `start` = pad top and left.
- `pad_color` (STRING, default `#000000`): hex color used for the pad area.

Outputs:

- `image` (IMAGE)
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)
- `padding` (PADDING_TUPLE): the four pad amounts plus `pad_color` bundled on a single link.

### CRTP_CropImageByPadding

Reverse of `CRTP_PadImageToDivisible`. Removes the supplied pad amounts from
each side. Wire either the four `pad_*` outputs or the bundled `padding`
output from the pad node into this one after running the model.

Inputs:

- `image` (IMAGE)
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)
- `padding` (PADDING_TUPLE, optional): when connected, overrides the four individual `pad_*` inputs.

Output:

- `image` (IMAGE)

### CRTP_PadImageByPadding

Pad an `IMAGE` using provided pad values (instead of computing from a divisor).
This pairs naturally with `CRTP_ComputePaddingToDivisible`:

```text
width/height ──► CRTP_ComputePaddingToDivisible ──► padding (PADDING_TUPLE)
image        ─────────────────────────────────────► CRTP_PadImageByPadding
```

Inputs:

- `image` (IMAGE)
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)
- `pad_color` (STRING)
- `padding` (PADDING_TUPLE, optional): when connected, overrides individual
  pad values and color.

Outputs:

- `image` (IMAGE)
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)
- `padding` (PADDING_TUPLE)

### CRTP_ComputePaddingToDivisible

Same math as `CRTP_PadImageToDivisible`, but works on integers only — no image
is touched. Useful for pre-computing target dimensions for empty latents or
other resolution-aware nodes.

Inputs:

- `width`, `height` (INT)
- `divisor` (INT, default `32`)
- `mode` (`symmetric` | `asymmetric`)
- `asymmetric_side` (`end` | `start`)
- `pad_color` (STRING, default `#000000`) — stored in the PADDING_TUPLE output.

Outputs:

- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)
- `padded_width`, `padded_height` (INT)
- `padding` (PADDING_TUPLE)

### CRTP_PackPaddingTuple / CRTP_UnpackPaddingTuple

`PADDING_TUPLE` is a small custom socket type carrying
`pad_left`, `pad_right`, `pad_top`, `pad_bottom`, and `pad_color` together.

- **CRTP_PackPaddingTuple** — inputs: four INTs + `pad_color` STRING; output: `padding` (PADDING_TUPLE).
- **CRTP_UnpackPaddingTuple** — input: `padding` (PADDING_TUPLE); outputs: four INTs + `pad_color` STRING.

Typical LTX-Video 2.3 wiring:

```text
1280x720 image ──► CRTP_PadImageToDivisible (divisor=32, symmetric)
                       │     pad_left/right/top/bottom ──┐
                       ▼                                  │
                 1280x736 image ──► LTX 2.3 ──► output ───┼─► CRTP_CropImageByPadding ──► 1280x720
                                                          │
                                                          └─────────────────────────────┘
```

### CRTP_ParseInt / CRTP_ParseFloat

Parse a STRING into a numeric value. On parse failure the supplied `default`
is returned, and an `ok` BOOLEAN reports whether parsing succeeded.

- **CRTP_ParseInt** — accepts decimal, hex (`0x...`), octal (`0o...`), binary
  (`0b...`), and float-like strings (`"3.0"`, `"1e3"`, truncated). Outputs
  `(value: INT, ok: BOOLEAN)`.
- **CRTP_ParseFloat** — outputs `(value: FLOAT, ok: BOOLEAN)`.

### CRTP JSON Nodes

Manipulate JSON text/data. These introduce a lightweight `JSON` socket type
that carries a parsed object (dict/list/scalar) between nodes; `parse` and
`stringify` bridge to/from `STRING`, while `minify`/`prettify` are plain
string-to-string conveniences. Mutating nodes operate on a deep copy, so
cached upstream outputs are never modified in place.

- **CRTP_JSONParse** — `STRING -> (JSON, ok: BOOLEAN, error: STRING)`.
- **CRTP_JSONStringify** — `JSON -> STRING`. `indent=0` is compact; options for
  `sort_keys` and `ensure_ascii`.
- **CRTP_JSONMinify** — `STRING -> (STRING, ok)`; removes insignificant
  whitespace.
- **CRTP_JSONPrettify** — `STRING -> (STRING, ok)`; re-indents (default `2`).
- **CRTP_JSONSet** — `obj[key] = value`. `value_is_json` (default `true`) parses
  the value as JSON (numbers/bools/objects/arrays); otherwise it is set as a
  raw string.
- **CRTP_JSONDeleteKey** — remove `key` entirely; outputs `(JSON, removed)`.
- **CRTP_JSONGet** — return `obj[key]` as `(value: *, value_str: STRING,
  found: BOOLEAN)` with an optional `default`.

Chain like: `JSON Parse -> JSON Set -> JSON Delete Key -> JSON Stringify`.

### CRTP_GetTensorShape

Generic tensor shape inspector.

- Input: `tensor` (accepts any connected value; for `LATENT` it reads
  `samples`, for `AUDIO` it reads `waveform`).
- Outputs: `num_dims`, `dim0..dim7`, `shape` (STRING).

Useful for quickly checking shapes while wiring complex workflows.

### CRTP_PadLatentToSize

Pad a `LATENT` tensor to a target size on a selected dimension using a fill
value (default `0.0`).

- Inputs:
  - `latent` (LATENT)
  - `dimension` (INT): index in latent `samples` shape `[B, C, H, W]`
  - `target_size` (INT)
  - `value` (FLOAT, default `0.0`)
- Outputs:
  - `latent` (LATENT)
  - `padded_by` (INT)
  - `new_size` (INT)

If `target_size` is smaller than or equal to the current size, it returns the
latent unchanged with `padded_by = 0`.

### CRTP Ideogram 4 Nodes (experimental)

Ideogram 4's text encoder is a Qwen3-VL-8B model, whose tokenizer can accept
images via `clip.tokenize(prompt, images=[...])` (same mechanism as the
built-in Qwen image-edit nodes). These nodes wire that path into standard
`CONDITIONING` so you can use an image as a prompt and chain prompts together.

> **Experimental:** this only sets up the encoder plumbing. Whether Ideogram 4
> meaningfully uses the image signal depends on how the model was trained.

- **CRTP_Ideogram4ImagePrompt** — encode an `image` (+ optional `text`) into
  `CONDITIONING`. Optional `conditioning` input appends the result onto an
  existing conditioning (sequence concat). The image is passed at **native
  resolution** by default — the Qwen-VL processor smart-resizes internally
  (clamped to ~12.8 MP) and snaps to the patch grid. Use `max_megapixels` only
  to cap resolution for VRAM/compute (0 = native; never upscales).
- **CRTP_Ideogram4TextPrompt** — encode `text` into `CONDITIONING`, with the
  same optional `conditioning` append input.
- **CRTP_ConditioningAppend** — generic helper that appends
  `conditioning_from` onto `conditioning_to` (same as `ConditioningConcat`).

Because each prompt node has an optional `conditioning` passthrough, you can
build chains such as:

```text
CLIP text prompt -> Image prompt -> Image prompt 2 -> Sampling
Image prompt     -> CLIP text prompt -> Image prompt 2 -> Sampling
```

All prompts in a chain must come from the **same Ideogram 4 CLIP** (matching
feature dimension) to be appended.

### CRTP Audio Nodes

Merged from `comfy-audio-nodes` into this pack:

- **CRTP_NormalizeAudio** — normalize `AUDIO` by `peak` or `rms` to `target_dB`.
- **CRTP_AudioProperties** — outputs `samples`, `duration`, `sample_rate`,
  `num_channels`, `isNone` (`audio` input is optional and returns `-1` +
  `isNone=True` when missing).
- **CRTP_SplitAudioChannels** — splits `AUDIO` into `left` and `right` mono
  outputs (mono input is duplicated).
- **CRTP_MergeAudioChannels** — merges `left` + `right` audio into stereo,
  with sample-rate validation and zero-padding for mismatched lengths.

### CRTP Lazy Load Nodes

Fault-tolerant alternatives to the built-in `Load Image` / `Load Audio` nodes.
The stock loaders define a `VALIDATE_INPUTS` check that aborts the whole graph
when the referenced file is missing — which is exactly what happens when a
workflow is ported to a machine that does not have the input file.

These versions instead:

- override `VALIDATE_INPUTS` to always pass (this also bypasses the implicit
  "value must be one of the combo options" check, so a stale filename does not
  abort the run),
- return `None` for the data output(s) when the file cannot be loaded,
- expose a `loaded` BOOLEAN output for downstream branching.

- **CRTP_LazyLoadImageUpload** and **CRTP_LazyLoadImageSelect** — outputs `image` (IMAGE), `mask` (MASK), `loaded` (BOOLEAN).
- **CRTP_LazyLoadAudioUpload** and **CRTP_LazyLoadAudioSelect** — outputs `audio` (AUDIO), `loaded` (BOOLEAN).
- **CRTP_LazyLoadVideoUpload** and **CRTP_LazyLoadVideoSelect** — outputs `video` (VIDEO), `loaded` (BOOLEAN).

Note: if your frontend still has the `uploadAudio.ts` bug on legacy nodes,
prefer `CRTP_LazyLoadAudioSelect` (plain selector) over the upload variant.

Pair `loaded` with `CRTP_LazyLatentFallback` to skip an expensive encode branch
when the source is absent.

### CRTP Viggle-Animate Nodes

Native MiniMax H3 helpers for bounded 24 fps driving video, frozen text
conditioning, video-first reference packing, distilled sigma schedules, and
output trimming. Includes Resolution, Prepare Video, Frozen Text Conditioning,
Animate Conditioning, Distilled Sigmas, and Trim Output nodes.
See [Viggle-Animate setup and node wiring](docs/viggle.md).

### CRTP Vision Language · Model Selector

Generate text from images using native Qwen3-VL 8B, Qwen3.8 27B GGUF, or
Gemma 4 E4B GGUF. GGUF options use a local llama.cpp runtime and the matching
vision projector. See [model setup and controls](docs/vl-text.md).

### CRTP Drama 3 Speech Nodes

Local speech generation and reference voice cloning, with separate runtimes
for Apple Silicon testing and NVIDIA CUDA workflows:

| Node | Purpose |
| --- | --- |
| **CRTP Drama 3 · Generate (MLX / Mac)** | Runs `mlx-community/fish-audio-drama-3-4bit` on Apple Silicon. |
| **CRTP Drama 3 · Generate (CUDA / 32 GB)** | Runs `darioooooo0o/drama-3-w4a16-32gb` through patched SGLang-Omni on NVIDIA Linux/WSL2. |
| **CRTP Drama 3 · Speaker Reference** | Combines a recording, its exact transcript, and a speaker ID; chain nodes for multiple voices. |

Both generators return standard ComfyUI AUDIO at 44.1 kHz. For a single cloned
voice, connect `reference_audio`, enter its exact transcript in `reference_text`,
and put the new speech in `text`. Target audio is not required. For multiple
speakers, connect chained Speaker Reference nodes to `references` instead.

MLX uses sentence chunks by default, normalizes reference levels, and retries a
failed take once when it encounters a skipped chunk, frame limit, or sustained
near-silence. These checks improve completion but do not verify every spoken word.
Each generator uses an isolated runtime process and releases it after the run.
Weights and dependencies are installed explicitly, never during node execution.

**Verification:** MLX generation has been exercised on an M4 Max with 64 GB.
The CUDA adapter has protocol and process-cleanup tests; actual RTX 5090 memory
fit and GPU inference remain unverified. See the setup guide for runtime and
CUDA requirements before downloading the checkpoint.

- [Setup, reference preparation, controls, and validation](docs/drama3.md)
- [MLX workflow](examples/drama3/Drama3-MLX.json) · [CUDA workflow](examples/drama3/Drama3-CUDA.json)
- [MLX API prompt](examples/drama3/Drama3-MLX.api.json) · [CUDA API prompt](examples/drama3/Drama3-CUDA.api.json)

### CRTP Audio Waveform Editor

A LosslessCut-style editor for standard ComfyUI AUDIO, under **CRTP → audio**.
It runs on the CPU and can prepare references for either MLX or CUDA, or edit
speech after generation.

- Drag across the waveform to select a range, split at the playhead, and drag
  segment boundaries to refine cuts.
- Enable or disable segments using their checkboxes; disabled sections stay
  visible and can be restored. The source recording is never overwritten.
- Mark quiet gaps with adjustable silence threshold, minimum duration, and
  padding; review the suggested cuts before running.
- Audition the original, selected range, or joined enabled sections. Includes
  zoom, an expanded editor, keyboard controls, and session undo/redo.
- Save cuts with the workflow. Changed input audio requires reviewing the cuts
  again; optional short edge fades reduce clicks in the rendered output.

```text
Load Audio → Audio Waveform Editor → Drama 3 reference_audio
                                  → Preview Audio / Save Audio
```

Click **Load waveform** to queue the editor and its upstream inputs without
running downstream generators. Select gaps and click **Disable**, or use
**Mark silence…**. Click **▶ Kept** to audition, then run the workflow normally
to output the enabled sections joined in order. Outputs are `edited_audio`,
`removed_seconds`, and an `edit_map` JSON string.

If you remove only silence from a voice reference, keep its transcript unchanged.
If you remove spoken words, update the transcript to match.

[Usage guide and shortcuts](docs/audio-waveform-editor.md) ·
[Example workflow](examples/audio-editor/waveform-editor.json)

## Installation and updates

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dnnagy/comfy-random-toolpack.git
```

To update an existing installation:

```bash
cd ComfyUI/custom_nodes/comfy-random-toolpack
git pull --ff-only
```

Restart ComfyUI and refresh its browser page after installing or updating.
All nodes use the `CRTP_` prefix for namespacing. The waveform editor uses
ComfyUI's existing torch, NumPy, and soundfile dependencies; its browser UI is
bundled with the pack.

Drama 3 requires a separate runtime and checkpoint. From the toolpack directory,
choose the setup command for your machine (requires `uv` and Git):

```bash
# Apple Silicon Mac
python3 scripts/setup_drama3.py mlx --model-dir /absolute/path/to/fish-audio-drama-3-4bit

# NVIDIA Linux/WSL2 worker with compatible CUDA development tools
python3 scripts/setup_drama3.py cuda --model-dir /absolute/path/to/drama-3-w4a16-32gb
```

Set the generator's `model_path` to the downloaded directory. The installer
keeps runtime dependencies separate from ComfyUI. Read the
[Drama 3 setup guide](docs/drama3.md) for pinned versions and platform requirements.

## Development checks

Run Python tests with ComfyUI's Python environment (torch, NumPy, soundfile,
SciPy, and Pillow available). JavaScript edit-state tests use Node.js 22+:

```bash
python -m unittest discover -s tests -v
node --test tests/audio_editor_core.test.mjs
```

The CPU suite includes local loopback HTTP tests. Model inference integration
checks are separate and require the relevant weights and hardware.

## License

MIT for this toolpack. Model weights and external runtimes retain their own
licenses; see [Drama 3 upstream sources and license notes](docs/drama3.md#upstream-sources-and-licenses).
