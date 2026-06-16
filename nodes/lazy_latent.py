"""Lazy conditional latent fallback nodes.

These nodes use ComfyUI's lazy input evaluation so that only the selected
branch is executed. This is useful when one branch (e.g. a VAE encode) would
fail or be expensive when the source input is missing.

Reference: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation
"""


class CRTP_LazyLatentFallback:
    """Pick between two LATENT inputs without evaluating both branches.

    Only the LATENT input selected by ``use_real_latent`` is evaluated, so the
    other upstream graph (e.g. a VAE encode that would crash on missing input)
    is never executed.
    """

    CATEGORY = "latent/conditional"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_real_latent": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "real_latent": ("LATENT", {"lazy": True}),
                "empty_latent": ("LATENT", {"lazy": True}),
            },
        }

    @classmethod
    def check_lazy_status(cls, use_real_latent, real_latent=None, empty_latent=None):
        needed = []
        if use_real_latent:
            if real_latent is None:
                needed.append("real_latent")
        else:
            if empty_latent is None:
                needed.append("empty_latent")
        return needed

    def run(self, use_real_latent, real_latent=None, empty_latent=None):
        if use_real_latent:
            if real_latent is None:
                raise ValueError(
                    "LazyLatentFallback: use_real_latent is True but "
                    "real_latent is not connected."
                )
            return (real_latent,)

        if empty_latent is None:
            raise ValueError(
                "LazyLatentFallback: use_real_latent is False but "
                "empty_latent is not connected."
            )
        return (empty_latent,)


NODE_CLASS_MAPPINGS = {
    "CRTP_LazyLatentFallback": CRTP_LazyLatentFallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_LazyLatentFallback": "CRTP Lazy Latent Fallback",
}
