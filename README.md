# Comfy Random Toolpack

A grab-bag of small, useful ComfyUI custom nodes.

## Nodes

### Lazy Latent Fallback

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
has_source ──┤                                     ├─ Lazy Latent Fallback → LATENT
             └─ Empty Latent ──────────────────────┘
```

Because `real_latent` is lazy, the VAE encode branch is only executed when
`use_real_latent` is `True`, and the empty latent branch is only executed when
it is `False`.

### Pad Image To Divisible

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

### Crop Image By Padding

Reverse of `Pad Image To Divisible`. Removes the supplied pad amounts from
each side. Wire either the four `pad_*` outputs or the bundled `padding`
output from the pad node into this one after running the model.

Inputs:

- `image` (IMAGE)
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)
- `padding` (PADDING_TUPLE, optional): when connected, overrides the four individual `pad_*` inputs.

Output:

- `image` (IMAGE)

### Compute Padding To Divisible

Same math as `Pad Image To Divisible`, but works on integers only — no image
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

### Pack / Unpack Padding Tuple

`PADDING_TUPLE` is a small custom socket type carrying
`pad_left`, `pad_right`, `pad_top`, `pad_bottom`, and `pad_color` together.

- **Pack Padding Tuple** — inputs: four INTs + `pad_color` STRING; output: `padding` (PADDING_TUPLE).
- **Unpack Padding Tuple** — input: `padding` (PADDING_TUPLE); outputs: four INTs + `pad_color` STRING.

Typical LTX-Video 2.3 wiring:

```text
1280x720 image ──► Pad Image To Divisible (divisor=32, symmetric)
                       │     pad_left/right/top/bottom ──┐
                       ▼                                  │
                 1280x736 image ──► LTX 2.3 ──► output ───┼─► Crop Image By Padding ──► 1280x720
                                                          │
                                                          └─────────────────────────────┘
```

### Parse Int / Parse Float

Parse a STRING into a numeric value. On parse failure the supplied `default`
is returned, and an `ok` BOOLEAN reports whether parsing succeeded.

- **Parse Int** — accepts decimal, hex (`0x...`), octal (`0o...`), binary
  (`0b...`), and float-like strings (`"3.0"`, `"1e3"`, truncated). Outputs
  `(value: INT, ok: BOOLEAN)`.
- **Parse Float** — outputs `(value: FLOAT, ok: BOOLEAN)`.

### Lazy Load Image / Lazy Load Audio

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

- **Lazy Load Image** — outputs `image` (IMAGE), `mask` (MASK), `loaded` (BOOLEAN).
- **Lazy Load Audio** — outputs `audio` (AUDIO), `loaded` (BOOLEAN).
- **Lazy Load Video** — outputs `video` (VIDEO), `loaded` (BOOLEAN).

Note: `Lazy Load Audio` currently uses a standard dropdown selector (no upload
widget) to avoid a frontend `uploadAudio.ts` crash path seen with custom
legacy-node `audio_upload` metadata.

Pair `loaded` with `Lazy Latent Fallback` to skip an expensive encode branch
when the source is absent.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dnnagy/comfy-random-toolpack.git
```

Restart ComfyUI, then search the node menu under `latent/conditional` for
`Lazy Latent Fallback`.

## License

MIT
