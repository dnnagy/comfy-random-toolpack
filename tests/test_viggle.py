"""CPU tests for Viggle media timing and native H3 conditioning contracts."""

import importlib.util
from fractions import Fraction
import numpy as np
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "viggle.py"
if not MODULE_PATH.exists():
    MODULE_PATH = Path(__file__).with_name("viggle.py")
folders = types.ModuleType("folder_paths")
folders.models_dir = "/models"
folders.add_model_folder_path = lambda *args: None
folders.get_filename_list = lambda name: ["fixed_embed_fwd_anyframe.safetensors"]
folders.get_full_path_or_raise = lambda category, name: name
spec = importlib.util.spec_from_file_location("crtp_viggle_under_test", MODULE_PATH)
viggle = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"folder_paths": folders}):
    spec.loader.exec_module(viggle)


def text_cond():
    return {"prompt_embeds": torch.zeros(1, 362, 5120, dtype=torch.bfloat16),
            "text_token_tags": torch.zeros(362, dtype=torch.int64)}


class LazyVideo:
    def __init__(self, frame_count=300, fps=30, audio=True):
        self.images = torch.arange(frame_count).float().view(-1, 1, 1, 1).expand(-1, 2, 2, 3)
        self.fps = fps
        self.audio = {"waveform": torch.arange(int(frame_count / fps * 24000)).float().view(1, 1, -1), "sample_rate": 24000} if audio else None
        self.calls = []
        self.dimensions = (32, 32)
        self.converted = []
        self.timestamps = None

    def get_components(self):
        raise AssertionError("The full source must never be decoded")

    def get_frame_rate(self):
        return self.fps

    def get_dimensions(self):
        return self.dimensions

    def as_trimmed(self, start_time, duration, strict_duration):
        self.calls.append((start_time, duration, strict_duration))
        begin = round(start_time * self.fps)
        end = min(self.images.shape[0], round((start_time + duration) * self.fps))
        if begin >= end:
            return None
        audio = self.audio
        if audio is not None:
            start = round(start_time * audio["sample_rate"])
            stop = round((start_time + duration) * audio["sample_rate"])
            audio = {**audio, "waveform": audio["waveform"][..., start:stop]}
        clip = types.SimpleNamespace(get_duration=lambda: (end - begin) / self.fps,
                                     _audio=audio)
        clip.get_components = lambda: (_ for _ in ()).throw(AssertionError("Never build source-resolution tensors"))
        clip.get_active_trim_window = lambda: (0.0, (end - begin) / self.fps)
        clip.get_stream_source = lambda: clip
        times = self.timestamps if self.timestamps is not None else [i / self.fps for i in range(end - begin)]
        clip.frames = [FakeFrame(begin + i, timestamp, self.fps, self.dimensions, self.converted)
                       for i, timestamp in enumerate(times)]
        return clip


class FakeFrame:
    def __init__(self, value, timestamp, fps, dimensions, converted):
        self.value = value
        self.pts = timestamp
        self.time_base = Fraction(1, 1)
        self.duration = 1 / fps
        self.width, self.height = dimensions
        self.format = types.SimpleNamespace(name="yuv420p")
        self.rotation = 0
        self.converted = converted

    def to_ndarray(self, *args, **kwargs):
        raise AssertionError("Source frame must be resized before RGB array conversion")

    def reformat(self, width, height, format, interpolation):
        self.converted.append((self.value, width, height))
        return types.SimpleNamespace(to_ndarray=lambda: np.full((height, width, 3), self.value, np.float32))


class FakeContainer:
    def __init__(self, clip):
        self.clip = clip
        self.streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=Fraction(1, 1000000))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def demux(self, stream):
        for frame in self.clip.frames:
            yield types.SimpleNamespace(decode=lambda frame=frame: [frame])

    def seek(self, *args, **kwargs):
        pass


class PrepareTests(unittest.TestCase):
    def setUp(self):
        decoder = patch("av.open", lambda clip, mode: FakeContainer(clip))
        decoder.start()
        self.addCleanup(decoder.stop)
        audio = patch.object(viggle, "_aligned_source_audio", lambda clip, duration: viggle._trim_audio(clip._audio, duration))
        audio.start()
        self.addCleanup(audio.stop)

    def test_bounds_lazy_decode_resamples_and_aligns_audio(self):
        source = LazyVideo()
        images, audio, count, fps = viggle.CRTP_VigglePrepareVideo().prepare(source, 124, 2.0, True)
        self.assertEqual(source.calls, [(2.0, 124 / 24, False)])
        self.assertEqual((count, fps), (124, 24.0))
        expected = (60 + torch.floor(torch.arange(124) * 30 / 24 + 0.5))
        torch.testing.assert_close(images[:, 0, 0, 0], expected)
        self.assertEqual(audio["waveform"].shape, (1, 1, 124000))
        self.assertEqual(audio["waveform"][0, 0, 0], 48000)

    def test_shorter_source_and_low_fps_are_preserved(self):
        images, audio, count, _ = viggle.CRTP_VigglePrepareVideo().prepare(LazyVideo(12, 12), 124)
        self.assertEqual(count, 24)
        self.assertEqual(images[-1, 0, 0, 0], 11)
        self.assertEqual(audio["waveform"].shape[-1], 24000)

    def test_silent_and_disabled_audio(self):
        for source, enabled in ((LazyVideo(audio=False), True), (LazyVideo(), False)):
            self.assertIsNone(viggle.CRTP_VigglePrepareVideo().prepare(source, preserve_audio=enabled)[1])

    def test_trim_audio_pads_short_audio_to_video_duration(self):
        audio = {"waveform": torch.ones(1, 2, 5), "sample_rate": 24}
        result = viggle._trim_audio(audio, 1)
        self.assertEqual(result["waveform"].shape, (1, 2, 24))
        self.assertTrue(torch.equal(result["waveform"][..., :5], audio["waveform"]))
        self.assertEqual(result["waveform"][..., 5:].sum(), 0)

    def test_invalid_and_too_short_media_rejected(self):
        for source, args in (
            (None, {}), (LazyVideo(4, 24), {}), (LazyVideo(), {"start_time": 500}),
            (LazyVideo(), {"start_time": float("nan")}), (LazyVideo(), {"max_frames": 363}),
            (LazyVideo(), {"max_frames": 5.5}), (LazyVideo(), {"max_frames": True}),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                viggle.CRTP_VigglePrepareVideo().prepare(source, **args)

    def test_4k_60fps_is_downscaled_before_tensor_conversion(self):
        source = LazyVideo(600, 60, audio=False)
        source.dimensions = (3840, 2160)
        images, _, count, fps = viggle.CRTP_VigglePrepareVideo().prepare(
            source, 124, width=864, height=480)
        self.assertEqual((count, fps), (124, 24.0))
        self.assertEqual(tuple(images.shape), (124, 480, 832, 3))
        self.assertLessEqual(len(source.converted), 124)
        self.assertTrue(all((w, h) == (832, 480) for _, w, h in source.converted))
        torch.testing.assert_close(images[:, 0, 0, 0], torch.floor(torch.arange(124) * 60 / 24 + .5))

    def test_source_fps_validation_precedes_decoding(self):
        for fps in (0, float("nan"), 241):
            source = LazyVideo()
            source.fps = fps
            with self.assertRaisesRegex(ValueError, "source FPS"):
                viggle.CRTP_VigglePrepareVideo().prepare(source)
            self.assertEqual(source.calls, [])

    def test_variable_frame_timestamps_control_resampling(self):
        source = LazyVideo(12, 24, audio=False)
        source.timestamps = [0, .02, .04, .06, .08, .10, .12, .20, .28, .36, .42, .46]
        images, _, count, fps = viggle.CRTP_VigglePrepareVideo().prepare(source)
        self.assertEqual((count, fps), (12, 24.0))
        self.assertEqual(images[3, 0, 0, 0], 6)  # .125 sec, nearest PTS .12
        self.assertEqual(images[6, 0, 0, 0], 8)  # .25 sec, nearest PTS .28

    def test_small_source_is_not_upscaled_and_high_fps_is_resampled(self):
        source = LazyVideo(120, 120, audio=False)
        source.dimensions = (320, 180)
        images, _, count, fps = viggle.CRTP_VigglePrepareVideo().prepare(source, 24, width=864, height=480)
        self.assertEqual(tuple(images.shape), (24, 160, 320, 3))
        self.assertEqual((count, fps), (24, 24.0))
        torch.testing.assert_close(images[:, 0, 0, 0], torch.arange(24).float() * 5)

    def test_mismatched_aspect_fits_within_target_without_cropping(self):
        source = LazyVideo(24, 24, audio=False)
        source.dimensions = (1080, 1920)
        images, _, _, _ = viggle.CRTP_VigglePrepareVideo().prepare(source, 5, width=864, height=480)
        self.assertEqual(tuple(images.shape), (5, 480, 256, 3))

    def test_display_rotation_is_applied_after_small_frame_conversion(self):
        frame = FakeFrame(1, 0, 24, (1920, 1080), [])
        frame.rotation = 90
        image = viggle._frame_image(frame, 256, 480)
        self.assertEqual(tuple(image.shape), (480, 256, 3))
        self.assertEqual(frame.converted, [(1, 480, 256)])

    def test_optional_size_inputs_preserve_old_workflow_contract(self):
        optional = viggle.CRTP_VigglePrepareVideo.INPUT_TYPES()["optional"]
        self.assertEqual(optional["width"][1]["default"], 864)
        self.assertEqual(optional["height"][1]["default"], 480)


class ContractTests(unittest.TestCase):
    def test_frame_grid_rounds_up_without_extra_cycle(self):
        self.assertEqual([viggle._generation_frames(v) for v in (5, 6, 21, 22, 23, 124, 125, 362)],
                         [5, 22, 22, 22, 39, 124, 141, 362])

    def test_sigmas_have_exact_evaluation_count_and_one_terminal_zero(self):
        for steps in (3, 5, 7):
            sigmas, = viggle.CRTP_ViggleSigmas().schedule(steps)
            self.assertEqual(len(sigmas), steps + 1)
            self.assertEqual(sigmas[0], 1)
            self.assertEqual(sigmas[-1], 0)
            self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
        torch.testing.assert_close(viggle.CRTP_ViggleSigmas().schedule(3)[0], torch.tensor([1, 6 / 7, 3 / 5, 0]))
        for value in (4, 6, 3.5):
            with self.assertRaises(ValueError):
                viggle.CRTP_ViggleSigmas().schedule(value)

    def test_resolutions_and_guards(self):
        for w, h in ((864, 480), (480, 864), (640, 640), (1024, 576), (576, 1024),
                     (768, 768), (1312, 736), (736, 1312), (992, 992)):
            self.assertEqual(viggle.CRTP_ViggleResolution().parse(f"{w}x{h} (label)"), (w, h))
        for value in ("1920x1080", "768x432", "32x1536", "1536x1536", "864x480garbage"):
            with self.assertRaises(ValueError):
                viggle.CRTP_ViggleResolution().parse(value)

    def test_reference_geometry_matches_upstream(self):
        self.assertEqual(viggle._reference_size(1920, 1080, 480, 864 * 480), (864, 480))
        self.assertEqual(viggle._reference_size(1, 4, 480), (480, 1920))
        with self.assertRaises(ValueError):
            viggle._reference_size(1, 5, 480)

    def test_real_safetensors_load_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "conditioning.safetensors")
            save_file(text_cond(), path)
            loaded, = viggle.CRTP_ViggleTextCondLoader().load(path)
            self.assertEqual(loaded["prompt_embeds"].shape, (1, 362, 5120))
            save_file({"prompt_embeds": torch.zeros(1, 1, 1)}, path)
            with self.assertRaises(ValueError):
                viggle.CRTP_ViggleTextCondLoader().load(path)

    def test_trim_removes_padding_but_never_silently_shortens(self):
        images = torch.zeros(39, 2, 2, 3)
        trimmed, = viggle.CRTP_ViggleTrimOutput().trim(images, 23)
        self.assertEqual(len(trimmed), 23)
        with self.assertRaises(ValueError):
            viggle.CRTP_ViggleTrimOutput().trim(images, 40)

    def test_conditioning_packs_video_first_and_allocates_correct_av_shape(self):
        comfy = types.ModuleType("comfy")
        management = types.ModuleType("comfy.model_management")
        management.intermediate_device = lambda: "cpu"
        nested = types.ModuleType("comfy.nested_tensor")
        nested.NestedTensor = tuple
        comfy.model_management, comfy.nested_tensor = management, nested
        extras = types.ModuleType("comfy_extras")
        h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
        h3.temporal_shape = lambda n: (n, ((n - 5) // 17) * 5 + 2, round(n / 24 * 40))
        h3._resize = lambda image, w, h, crop: torch.zeros(image.shape[0], h, w, 3)
        extras.nodes_minimax_h3 = h3
        calls = []
        def encode(images):
            calls.append(tuple(images.shape))
            return torch.zeros(1, 24, 7 if len(images) > 1 else 1, images.shape[1] // 16, images.shape[2] // 16)
        vae = types.SimpleNamespace(encode=encode)
        modules = {"comfy": comfy, "comfy.model_management": management, "comfy.nested_tensor": nested,
                   "comfy_extras": extras, "comfy_extras.nodes_minimax_h3": h3}
        with patch.dict(sys.modules, modules):
            cond, latent = viggle.CRTP_ViggleAnimateConditioning().build(
                torch.zeros(23, 32, 32, 3), torch.zeros(1, 8, 8, 3), text_cond(), vae, 64, 32)
        self.assertEqual(calls, [(23, 32, 32, 3), (1, 32, 32, 3)])
        refs = cond[0][1]["minimax_refs"]
        self.assertEqual([r["kind"] for r in refs], ["video", "image"])
        self.assertEqual(refs[0]["ref_audio_t"], 0)
        self.assertEqual(latent["samples"][0].shape, (1, 24, 12, 2, 4))
        self.assertEqual(latent["samples"][1].shape, (1, 32, 2, 65))


if __name__ == "__main__":
    unittest.main()
