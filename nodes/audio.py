import torch


class CRTP_NormalizeAudio:
    """Normalize audio volume using peak or RMS normalization."""

    CATEGORY = "audio"
    FUNCTION = "normalize"

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "method": (("peak", "rms"), {"default": "peak"}),
                "target_dB": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -60.0,
                        "max": 0.0,
                        "step": 0.1,
                        "tooltip": (
                            "Target level in dBFS. "
                            "0.0 = maximum digital level, "
                            "-1.0 = slight headroom (recommended for peak), "
                            "-14.0 = typical broadcast loudness (recommended for RMS)."
                        ),
                    },
                ),
            },
        }

    def normalize(self, audio, method="peak", target_dB=-1.0):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        target_linear = 10.0 ** (target_dB / 20.0)

        if method == "peak":
            waveform = self._peak_normalize(waveform, target_linear)
        else:
            waveform = self._rms_normalize(waveform, target_linear)

        return ({"waveform": waveform, "sample_rate": sample_rate},)

    @staticmethod
    def _peak_normalize(waveform: torch.Tensor, target: float) -> torch.Tensor:
        peak = waveform.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return waveform * (target / peak)

    @staticmethod
    def _rms_normalize(waveform: torch.Tensor, target: float) -> torch.Tensor:
        rms = waveform.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
        scaled = waveform * (target / rms)
        peak = scaled.abs().amax(dim=(-2, -1), keepdim=True)
        if (peak > 1.0).any():
            scaled = scaled / peak.clamp(min=1e-8)
        return scaled


class CRTP_AudioProperties:
    """Get audio properties and return sentinel values for None input."""

    CATEGORY = "audio"
    FUNCTION = "get_properties"

    RETURN_TYPES = ("INT", "FLOAT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("samples", "duration", "sample_rate", "num_channels", "isNone")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    def get_properties(self, audio=None):
        if audio is None:
            return (-1, -1.0, -1, -1, True)

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        samples = int(waveform.shape[-1])
        num_channels = int(waveform.shape[-2])
        duration = samples / sample_rate if sample_rate > 0 else 0.0
        return (samples, float(duration), sample_rate, num_channels, False)


class CRTP_SplitAudioChannels:
    """Split a multi-channel audio into two mono outputs."""

    CATEGORY = "audio"
    FUNCTION = "split"

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("left", "right")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    def split(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        num_channels = waveform.shape[-2]

        left = waveform[..., 0:1, :]
        if num_channels >= 2:
            right = waveform[..., 1:2, :]
        else:
            right = left.clone()

        return (
            {"waveform": left.contiguous(), "sample_rate": sample_rate},
            {"waveform": right.contiguous(), "sample_rate": sample_rate},
        )


class CRTP_MergeAudioChannels:
    """Merge two mono audio inputs into a stereo audio output."""

    CATEGORY = "audio"
    FUNCTION = "merge"

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("AUDIO",),
                "right": ("AUDIO",),
            },
        }

    def merge(self, left, right):
        lw = left["waveform"]
        rw = right["waveform"]
        lsr = int(left["sample_rate"])
        rsr = int(right["sample_rate"])

        if lsr != rsr:
            raise ValueError(
                f"CRTP_MergeAudioChannels: sample rates differ ({lsr} vs {rsr})."
            )

        lw = lw[..., 0:1, :]
        rw = rw[..., 0:1, :]

        ll = lw.shape[-1]
        rl = rw.shape[-1]
        if ll < rl:
            pad = torch.zeros(*lw.shape[:-1], rl - ll, dtype=lw.dtype, device=lw.device)
            lw = torch.cat([lw, pad], dim=-1)
        elif rl < ll:
            pad = torch.zeros(*rw.shape[:-1], ll - rl, dtype=rw.dtype, device=rw.device)
            rw = torch.cat([rw, pad], dim=-1)

        merged = torch.cat([lw, rw], dim=-2)
        return ({"waveform": merged.contiguous(), "sample_rate": lsr},)


NODE_CLASS_MAPPINGS = {
    "CRTP_NormalizeAudio": CRTP_NormalizeAudio,
    "CRTP_AudioProperties": CRTP_AudioProperties,
    "CRTP_SplitAudioChannels": CRTP_SplitAudioChannels,
    "CRTP_MergeAudioChannels": CRTP_MergeAudioChannels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTP_NormalizeAudio": "CRTP Normalize Audio",
    "CRTP_AudioProperties": "CRTP Audio Properties",
    "CRTP_SplitAudioChannels": "CRTP Split Audio Channels",
    "CRTP_MergeAudioChannels": "CRTP Merge Audio Channels",
}
