from __future__ import annotations

import torch


class _AnyType(str):
    def __ne__(self, _value):
        return False


_ANY = _AnyType("*")


def _extract_shape(data) -> tuple[int, ...]:
    if isinstance(data, dict):
        if "samples" in data and hasattr(data["samples"], "shape"):
            return tuple(int(x) for x in data["samples"].shape)
        if "waveform" in data and hasattr(data["waveform"], "shape"):
            return tuple(int(x) for x in data["waveform"].shape)
        for value in data.values():
            if hasattr(value, "shape"):
                return tuple(int(x) for x in value.shape)

    if hasattr(data, "shape"):
        return tuple(int(x) for x in data.shape)

    if isinstance(data, (list, tuple)):
        return (len(data),)

    return ()


class CRTP_GetTensorShape:
    """Get shape information from tensor-like input."""

    CATEGORY = "utils/tensor"
    FUNCTION = "get_shape"

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "num_dims",
        "dim0",
        "dim1",
        "dim2",
        "dim3",
        "dim4",
        "dim5",
        "dim6",
        "dim7",
        "shape",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": (_ANY,),
            },
        }

    def get_shape(self, tensor):
        shape = _extract_shape(tensor)
        dims = list(shape[:8])
        while len(dims) < 8:
            dims.append(-1)

        shape_text = str(shape) if shape else "()"
        return (len(shape), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], shape_text)


class CRTP_PadLatentToSize:
    """Pad LATENT samples on one dimension up to target_size using a fill value."""

    CATEGORY = "latent/transform"
    FUNCTION = "pad"

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "padded_by", "new_size")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "dimension": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 3,
                        "tooltip": "Dimension in latent samples [B,C,H,W]. 0=batch, 1=channels, 2=height, 3=width.",
                    },
                ),
                "target_size": ("INT", {"default": 96, "min": 0, "max": 16384}),
                "value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
            },
        }

    def pad(self, latent, dimension, target_size, value):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("CRTP_PadLatentToSize: LATENT input must contain 'samples'.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("CRTP_PadLatentToSize: latent['samples'] must be a torch.Tensor.")

        dim = int(dimension)
        if dim < 0 or dim >= samples.ndim:
            raise ValueError(
                f"CRTP_PadLatentToSize: dimension {dim} out of range for samples shape {tuple(samples.shape)}."
            )

        current_size = int(samples.shape[dim])
        target = int(target_size)
        if target <= current_size:
            return (latent, 0, current_size)

        pad_amount = target - current_size
        pad_shape = list(samples.shape)
        pad_shape[dim] = pad_amount

        pad_tensor = torch.full(
            pad_shape,
            float(value),
            dtype=samples.dtype,
            device=samples.device,
        )
        out_samples = torch.cat([samples, pad_tensor], dim=dim)

        out = dict(latent)
        out["samples"] = out_samples
        return (out, pad_amount, int(out_samples.shape[dim]))


NODE_CLASS_MAPPINGS = {
    "CRTP_GetTensorShape": CRTP_GetTensorShape,
    "CRTP_PadLatentToSize": CRTP_PadLatentToSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_GetTensorShape": "CRTP Get Tensor Shape",
    "CRTP_PadLatentToSize": "CRTP Pad Latent To Size",
}
