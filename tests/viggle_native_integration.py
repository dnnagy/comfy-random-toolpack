"""Run inside the pinned ComfyUI worker image, on CPU, without model weights."""
import sys
sys.argv = ["viggle-integration", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()

import av
from fractions import Fraction
import importlib.util
import numpy as np
from pathlib import Path
import tempfile
import subprocess
from unittest.mock import patch
import torch
from types import SimpleNamespace

from comfy_api.latest import InputImpl
from comfy_extras import nodes_minimax_h3, nodes_video

node_path = Path(__file__).resolve().parents[1] / "nodes" / "viggle.py"
if not node_path.exists():
    node_path = Path(__file__).with_name("viggle.py")
spec = importlib.util.spec_from_file_location("viggle", node_path)
viggle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viggle)

with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / "variable-rate.mkv")
    times = [i * .04 for i in range(50)] + [2 + i * .1 for i in range(20)]
    with av.open(path, mode="w") as output:
        video = output.add_stream("ffv1", rate=25)
        video.width = video.height = 32
        video.pix_fmt = "bgr0"
        video.time_base = video.codec_context.time_base = Fraction(1, 1000)
        audio = output.add_stream("pcm_f32le", rate=24000, layout="mono")
        packets = []
        for index, seconds in enumerate(times):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 3, dtype=np.uint8), format="rgb24")
            frame.pts = round(seconds * 1000)
            frame.time_base = Fraction(1, 1000)
            for packet in video.encode(frame):
                packets.append(packet)
        for packet in video.encode():
            packets.append(packet)
        wave = (np.arange(96000, dtype=np.float32) / 240000).reshape(1, -1)
        for begin in range(0, 96000, 1200):
            frame = av.AudioFrame.from_ndarray(wave[:, begin:begin + 1200], format="flt", layout="mono")
            frame.sample_rate = 24000
            frame.pts = begin
            frame.time_base = Fraction(1, 24000)
            packets.extend(audio.encode(frame))
        for packet in audio.encode():
            packets.append(packet)
        for packet in sorted(packets, key=lambda p: float(p.pts * p.time_base)):
            output.mux(packet)

    source = InputImpl.VideoFromFile(path)
    with patch.object(InputImpl.VideoFromFile, "get_components", side_effect=AssertionError("No full source tensor decode")):
        images, audio, count, fps = viggle.CRTP_VigglePrepareVideo().prepare(source, 36, 1.5, True)
    assert tuple(images.shape) == (36, 32, 32, 3)  # Never upscale the driver.
    assert (count, fps) == (36, 24.0), (count, fps)
    selected_times = [(i, t) for i, t in enumerate(times) if 1.5 <= t < 3]
    expected = [min(selected_times, key=lambda x: (abs(x[1] - (1.5 + k / 24)), -x[1]))[0] for k in range(count)]
    observed = torch.round(images[:, 0, 0, 0] * 255 / 3).long().tolist()
    assert observed == expected, (observed, expected)
    assert audio["waveform"].shape == (1, 1, 36000), audio["waveform"].shape
    assert abs(audio["waveform"][0, 0, 0].item() - .15) < 1e-5, audio["waveform"][0, 0, :10]
    assert abs(audio["waveform"][0, 0, -1].item() - 71999 / 240000) < 1e-5
    print("PASS: native VFR decode -> exact24fps,36frames, nonzerooffsetaudioaligned")

    # Real native H3 geometry, NestedTensor, CreateVideo and video export.
    calls = []
    def encode(tensor):
        calls.append(tuple(tensor.shape))
        n, h, w, _ = tensor.shape
        return torch.zeros(1, 24, nodes_minimax_h3.temporal_shape(n)[1], h // 16, w // 16)
    cond, latent = viggle.CRTP_ViggleAnimateConditioning().build(
        images, images[:1], {"prompt_embeds": torch.zeros(1,362,5120,dtype=torch.bfloat16),
                            "text_token_tags": torch.zeros(362,dtype=torch.int64)},
        SimpleNamespace(encode=encode), 64, 32)
    assert [x["kind"] for x in cond[0][1]["minimax_refs"]] == ["video", "image"]
    assert latent["samples"].unbind()[0].shape == (1,24,12,2,4)
    output_video = nodes_video.CreateVideo.execute(images=images, fps=fps, audio=audio)[0]
    output_path = str(Path(directory) / "output.mp4")
    output_video.save_to(output_path)
    saved = InputImpl.VideoFromFile(output_path)
    assert saved.get_frame_count() == 36
    assert float(saved.get_frame_rate()) == 24
    print("PASS: native H3 conditioning/NestedTensor and CreateVideo/save 24fps roundtrip")

    # Phone-style display rotation and a source width that is not 32 aligned.
    # 60 fps input must be downscaled before floating RGB conversion and return
    # exactly 24 fps. A rotated MP4 also exercises actual decoder side data.
    source_path = str(Path(directory) / "phone-landscape.mp4")
    rotated_path = str(Path(directory) / "phone-portrait.mp4")
    pixels = np.zeros((406, 702, 3), dtype=np.uint8)
    pixels[:203, :351] = (255, 0, 0)
    pixels[:203, 351:] = (0, 255, 0)
    pixels[203:, :351] = (0, 0, 255)
    pixels[203:, 351:] = (255, 255, 255)
    with av.open(source_path, mode="w") as output:
        stream = output.add_stream("libx264", rate=60)
        stream.width, stream.height = 702, 406
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "12"}
        for index in range(30):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 60)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source_path, "-c", "copy",
                    "-metadata:s:v:0", "rotate=90", rotated_path], check=True)
    with av.open(rotated_path) as container:
        probe = next(container.decode(video=0))
        rotation = int(round(probe.rotation // 90)) % 4
        assert rotation % 2, probe.rotation
    with patch.object(InputImpl.VideoFromFile, "get_components", side_effect=AssertionError("No full source tensor decode")):
        resized, silent, count, fps = viggle.CRTP_VigglePrepareVideo().prepare(
            InputImpl.VideoFromFile(rotated_path), 12, preserve_audio=False, width=64, height=128)
    assert (count, fps) == (12, 24.0)
    assert tuple(resized.shape) == (12, 96, 64, 3), resized.shape
    assert silent is None
    expected_pixels = np.rot90(pixels, k=rotation)
    corners = ((.25, .25), (.25, .75), (.75, .25), (.75, .75))
    for y, x in corners:
        observed = resized[0, int(y * 96), int(x * 64)]
        expected = torch.from_numpy(expected_pixels[int(y * 702), int(x * 406)].copy()).float() / 255
        torch.testing.assert_close(observed, expected, atol=.04, rtol=0)
    assert torch.isfinite(resized).all()
    print("PASS: native 60fps phone MP4 -> resized24fps, rotation preserved, non-aligned source float RGB")
