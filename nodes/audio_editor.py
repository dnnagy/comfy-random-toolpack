"""Non-destructive, sample-indexed waveform editing for ComfyUI AUDIO."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

MAX_SEGMENTS = 2048


def validate_audio(audio):
    import torch
    if not isinstance(audio, dict):
        raise ValueError("Connect an AUDIO input first.")
    wave, rate = audio.get("waveform"), audio.get("sample_rate")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("Audio sample_rate must be a positive integer.")
    if not isinstance(wave, torch.Tensor) or wave.ndim != 3 or wave.shape[0] != 1:
        raise ValueError("Waveform Editor accepts one audio item [1, channels, samples], not a batch.")
    if not wave.is_floating_point() or not wave.shape[1] or not wave.shape[2] or not torch.isfinite(wave).all():
        raise ValueError("Audio must be nonempty, finite floating-point samples.")
    return wave.detach().to(device="cpu", dtype=torch.float32).contiguous(), rate


def source_id(wave, rate):
    digest = hashlib.sha256(f"{rate}:{tuple(wave.shape)}:".encode())
    digest.update(memoryview(wave.numpy()).cast('B'))
    return digest.hexdigest()


def parse_edits(value, identity, samples):
    if not value or value == "{}":
        return [{"start": 0, "end": samples, "enabled": True}]
    if len(value) > 1024 * 1024:
        raise ValueError("Editor state is too large.")
    try:
        state = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid waveform editor state. Reload the waveform.") from exc
    if not isinstance(state, dict) or state.get('version') != 1:
        raise ValueError("Unsupported waveform editor state version.")
    if state.get('source_id') != identity or state.get('samples') != samples:
        raise ValueError("The input audio changed. Click Load waveform to review/reset cuts before running again.")
    segments = state.get('segments')
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError(f"Keep between 1 and {MAX_SEGMENTS} segments.")
    end = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Invalid audio segment.")
        a, b, enabled = segment.get('start'), segment.get('end'), segment.get('enabled')
        if (type(a) is not int or type(b) is not int or type(enabled) is not bool
                or a != end or not a < b <= samples):
            raise ValueError("Segments must be ordered, contiguous sample ranges covering the original audio.")
        end = b
    if end != samples:
        raise ValueError("Segments must cover the complete source audio.")
    return segments


def kept_ranges(segments):
    ranges = []
    for segment in segments:
        if segment['enabled']:
            if ranges and ranges[-1][1] == segment['start']:
                ranges[-1][1] = segment['end']
            else:
                ranges.append([segment['start'], segment['end']])
    return ranges


def render_audio(wave, rate, segments, edge_fade_ms):
    import torch
    if not math.isfinite(edge_fade_ms) or not 0 <= edge_fade_ms <= 20:
        raise ValueError("edge_fade_ms must be between 0 and 20.")
    ranges = kept_ranges(segments)
    if not ranges:
        raise ValueError("Every segment is disabled. Enable at least one segment before rendering audio.")
    pieces, mapping, offset = [], [], 0
    fade_samples = round(rate * edge_fade_ms / 1000)
    for start, end in ranges:
        part = wave[..., start:end].clone()
        fade = min(fade_samples, (end - start) // 2)
        if fade:
            ramp = torch.linspace(0, 1, fade, dtype=part.dtype)
            if start > 0:
                part[..., :fade] *= ramp
            if end < wave.shape[-1]:
                part[..., -fade:] *= ramp.flip(0)
        pieces.append(part)
        mapping.append({'source_start': start, 'source_end': end, 'output_start': offset, 'output_end': offset + end - start})
        offset += end - start
    return torch.cat(pieces, dim=-1), mapping


def preview_data(wave, rate, identity):
    import numpy as np
    import soundfile as sf
    import folder_paths
    array = wave[0].numpy()
    samples = array.shape[-1]
    # Content-addressed previews survive repeated edits without duplicated files.
    directory = Path(folder_paths.get_temp_directory()) / 'crtp_audio_editor'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{identity}.wav'
    if not target.is_file():
        fd, temporary = tempfile.mkstemp(suffix='.wav', dir=directory)
        os.close(fd)
        try:
            # Browser preview supports mono/stereo. Rendering preserves all channels.
            sf.write(temporary, array[:2].T, rate, subtype='FLOAT')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    stride = max(1, math.ceil(samples / 8192))
    pad = (-samples) % stride
    low = np.pad(array.min(axis=0), (0, pad)).reshape(-1, stride).min(axis=1)
    high = np.pad(array.max(axis=0), (0, pad)).reshape(-1, stride).max(axis=1)
    # Bounded RMS envelope for interactive silence detection, without uploading
    # audio to a service or decoding the whole recording just to draw its shape.
    rms_stride = max(1, round(rate * .02), math.ceil(samples / 30000))
    power = np.max(np.square(array, dtype=np.float64), axis=0)
    padded = np.pad(power, (0, (-samples) % rms_stride)).reshape(-1, rms_stride)
    counts = np.full(padded.shape[0], rms_stride); counts[-1] = samples - rms_stride * (len(counts) - 1)
    db = 10 * np.log10(np.maximum(padded.sum(axis=1) / counts, 1e-12))
    return {'source_id': identity, 'samples': samples, 'sample_rate': rate, 'channels': array.shape[0],
            'duration': samples / rate, 'peak_stride': stride,
            'peaks_low': np.round(low, 5).tolist(), 'peaks_high': np.round(high, 5).tolist(),
            'rms_stride': rms_stride, 'rms_db': np.round(db, 1).tolist(),
            'file': {'filename': target.name, 'subfolder': 'crtp_audio_editor', 'type': 'temp'}}


class CRTP_AudioWaveformEditor:
    CATEGORY = 'CRTP/audio'
    FUNCTION = 'edit'
    OUTPUT_NODE = True
    RETURN_TYPES = ('AUDIO', 'FLOAT', 'STRING')
    RETURN_NAMES = ('edited_audio', 'removed_seconds', 'edit_map')
    DESCRIPTION = ('Load waveform, select/split ranges, and toggle Keep/Cut. Edits are saved with the workflow. '
                   'Load waveform queues only this node and its inputs; Run applies enabled segments to downstream audio.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': ('AUDIO',),
                             'segments_json': ('STRING', {'default': '', 'multiline': False}),
                             'edge_fade_ms': ('FLOAT', {'default': 3.0, 'min': 0.0, 'max': 20.0, 'step': 0.5,
                                 'tooltip': 'Short fades at cut boundaries reduce clicks. 0 preserves exact retained samples.'})},
                'optional': {'preview_only': ('BOOLEAN', {'default': False})}}

    @classmethod
    def IS_CHANGED(cls, preview_only=False, **kwargs):
        # Recreate expired temporary previews even when source and cuts are cached.
        return float('nan') if preview_only else False

    def edit(self, audio, segments_json='', edge_fade_ms=3.0, preview_only=False):
        wave, rate = validate_audio(audio)
        identity = source_id(wave, rate)
        segments = parse_edits('' if preview_only else segments_json, identity, wave.shape[-1])
        edited, mapping = render_audio(wave, rate, segments, edge_fade_ms)
        info = preview_data(wave, rate, identity)
        removed = (wave.shape[-1] - edited.shape[-1]) / rate
        info.update(rendered_samples=edited.shape[-1], preview_only=bool(preview_only))
        edit_map = json.dumps({'version': 1, 'source_id': identity, 'sample_rate': rate, 'ranges': mapping})
        return {'ui': {'audio_editor': [info]},
                'result': ({'waveform': edited, 'sample_rate': rate}, removed, edit_map)}


NODE_CLASS_MAPPINGS = {'CRTP_AudioWaveformEditor': CRTP_AudioWaveformEditor}
NODE_DISPLAY_NAME_MAPPINGS = {'CRTP_AudioWaveformEditor': 'CRTP Audio Waveform Editor'}
