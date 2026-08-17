# Comfy Random Toolpack

A grab-bag of small, useful ComfyUI custom nodes.

## Nodes

### CRTP MiniMax H3 Continuation Nodes

These nodes extend the separately installed, commit-pinned
`ComfyUI_MiniMax_H3_Extender` with uploaded-video motion context and all three
standalone Ref2VA audio references supported by ComfyUI core.

- **CRTP_H3ContinuationTail** — resamples the final source window to H3's
  24 fps timeline, aspect-fits it to the generation canvas, and extracts a
  valid 5, 22, 39, or 56-frame context tail plus its exact final frame.
- **CRTP_H3EncodeContinuationContext** — VAE-encodes that video tail and its
  time-aligned source audio into a joint H3 audio/video latent.
- **CRTP_MiniMaxH3ContinuationExtender** — applies the source latent as
  clip-zero Motion Context and exposes `<Audio 1>`, optional `<Audio 2>`, and
  optional `<Audio 3>` reference inputs. Audio slots must be contiguous so
  MiniMax cannot silently renumber them.

The overlap exists only inside sampling and is trimmed before final decode.
The requested clip duration therefore represents new continuation material.
Use 22 context frames normally and 5 for a source shorter than roughly one
second.

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

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dnnagy/comfy-random-toolpack.git
```

All nodes in this pack use the `CRTP_` prefix for namespacing.

## License

MIT
