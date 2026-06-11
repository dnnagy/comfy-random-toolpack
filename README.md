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
