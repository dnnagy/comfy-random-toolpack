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

### Crop Image By Padding

Reverse of `Pad Image To Divisible`. Removes the supplied pad amounts from
each side. Wire the four pad outputs from the pad node straight into this one
after running the model.

Inputs:

- `image` (IMAGE)
- `pad_left`, `pad_right`, `pad_top`, `pad_bottom` (INT)

Output:

- `image` (IMAGE)

Typical LTX-Video 2.3 wiring:

```text
1280x720 image ──► Pad Image To Divisible (divisor=32, symmetric)
                       │     pad_left/right/top/bottom ──┐
                       ▼                                  │
                 1280x736 image ──► LTX 2.3 ──► output ───┼─► Crop Image By Padding ──► 1280x720
                                                          │
                                                          └─────────────────────────────┘
```

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
