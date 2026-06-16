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


class CRTP_FitLatentToSize:
    """Pad or crop a LATENT on a chosen dimension to match target_size.

    If the current size on ``dimension`` is smaller than ``target_size``, the
    tensor is padded with ``value``. If it is larger, it is cropped. ``align``
    controls which side the operation happens on.
    """

    CATEGORY = "latent/transform"
    FUNCTION = "fit"

    RETURN_TYPES = ("LATENT", "STRING", "INT", "INT")
    RETURN_NAMES = ("latent", "operation", "delta", "new_size")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "dimension": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 7,
                        "tooltip": "Dimension index in latent samples.",
                    },
                ),
                "target_size": ("INT", {"default": 32, "min": 0, "max": 16384}),
                "value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Fill value used when padding.",
                    },
                ),
                "align": (
                    ("end", "start", "center"),
                    {
                        "default": "end",
                        "tooltip": (
                            "end: pad/crop the end of the dimension. "
                            "start: pad/crop the start. "
                            "center: split symmetrically."
                        ),
                    },
                ),
            },
        }

    @staticmethod
    def _split(amount: int, align: str) -> tuple[int, int]:
        if amount <= 0:
            return 0, 0
        if align == "end":
            return 0, amount
        if align == "start":
            return amount, 0
        if align == "center":
            before = amount // 2
            after = amount - before
            return before, after
        raise ValueError(f"Unknown align: {align!r}")

    def fit(self, latent, dimension, target_size, value, align):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("CRTP_FitLatentToSize: LATENT must contain 'samples'.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("CRTP_FitLatentToSize: latent['samples'] must be a torch.Tensor.")

        dim = int(dimension)
        if dim < 0 or dim >= samples.ndim:
            raise ValueError(
                f"CRTP_FitLatentToSize: dimension {dim} out of range for samples shape {tuple(samples.shape)}."
            )

        current = int(samples.shape[dim])
        target = int(target_size)

        if target == current:
            return (latent, "noop", 0, current)

        if target > current:
            amount = target - current
            before, after = self._split(amount, align)

            new_shape = list(samples.shape)
            parts = [samples]

            if before > 0:
                pad_shape = list(samples.shape)
                pad_shape[dim] = before
                parts.insert(
                    0,
                    torch.full(pad_shape, float(value), dtype=samples.dtype, device=samples.device),
                )
            if after > 0:
                pad_shape = list(samples.shape)
                pad_shape[dim] = after
                parts.append(
                    torch.full(pad_shape, float(value), dtype=samples.dtype, device=samples.device),
                )

            out_samples = torch.cat(parts, dim=dim)
            op = "pad"
            delta = amount
        else:
            amount = current - target
            before, after = self._split(amount, align)
            start = before
            length = current - before - after
            out_samples = samples.narrow(dim, start, length).contiguous()
            op = "crop"
            delta = amount

        out = dict(latent)
        out["samples"] = out_samples
        return (out, op, delta, int(out_samples.shape[dim]))


NODE_CLASS_MAPPINGS = {
    "CRTP_GetTensorShape": CRTP_GetTensorShape,
    "CRTP_PadLatentToSize": CRTP_PadLatentToSize,
    "CRTP_FitLatentToSize": CRTP_FitLatentToSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_GetTensorShape": "CRTP Get Tensor Shape",
    "CRTP_PadLatentToSize": "CRTP Pad Latent To Size",
    "CRTP_FitLatentToSize": "CRTP Fit Latent To Size",
}
