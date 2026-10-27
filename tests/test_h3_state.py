"""Real torch/safetensors/FFmpeg CPU tests; only model execution is stubbed."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location("crtp_h3_state_test", Path(__file__).parents[1] / "nodes/h3_state.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


class Nested:
    is_nested = True
    def __init__(self, video, audio):
        self.parts = (video, audio)
    def unbind(self):
        return self.parts


def av(frames=56, dtype=torch.float32):
    t = (frames - 5) // 17 * 5 + 2
    video = torch.arange(24*t*4).reshape(1,24,t,2,2).to(dtype)
    audio = torch.arange(32*2*round(frames*40/24)).reshape(1,32,2,-1).to(dtype)
    return {"samples": Nested(video, audio)}


def empty(width, height, length):
    return {"samples": Nested(torch.zeros(1,24,(length-5)//17*5+2,height//16,width//16),
                              torch.zeros(1,32,2,round(length*40/24)))}, length


class H3StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folders = types.SimpleNamespace(get_output_directory=lambda: str(self.root))
        self.folder_patch = patch.dict(sys.modules, {"folder_paths": self.folders})
        self.folder_patch.start()
    def tearDown(self):
        self.folder_patch.stop()
        self.temp.cleanup()
    def state(self, frames=56, dtype=torch.float32):
        return h3.make_state(av(frames, dtype), "av_clip", "sampled", "h3_vae")
    def save(self, state, filename="state.safetensors"):
        return h3.write_state(state, self.root / filename)
    def prepare(self, context, seconds=1):
        native = types.SimpleNamespace(_empty_av_latent=empty)
        with patch.dict(sys.modules, {"comfy_extras.nodes_minimax_h3": native}):
            return h3.CRTP_H3PrepareLatentGeneration().run([[torch.ones(1), {"minimax_refs": ["original-reference"]}]], context, seconds, "h3_vae")

    def test_eight_nodes_registered(self):
        self.assertEqual(len(h3.NODE_CLASS_MAPPINGS), 8)
        for cls in h3.NODE_CLASS_MAPPINGS.values():
            self.assertIn("required", cls.INPUT_TYPES())
            self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))

    def test_dtype_exact_roundtrip_and_lazy_header(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            original = self.state(dtype=dtype)
            path = self.save(original, str(dtype) + ".safetensors")
            loaded, info = h3.CRTP_LazyLoadH3State().run(path)
            self.assertNotIn("tensors", loaded)
            self.assertIn("raw_frames", info)
            for key, value in h3.read_tensors(loaded).items():
                self.assertEqual(value.dtype, dtype)
                self.assertTrue(torch.equal(value, original["tensors"][key]))

    def test_lazy_tail_matches_memory_and_never_loads_full_tensor(self):
        state = self.state(243)
        disk = h3.read_state(self.save(state))
        for window in (5, 22):
            a, _ = h3.CRTP_H3SelectLatentContext().run(state, f"tail_{window}", True)
            b, _ = h3.CRTP_H3SelectLatentContext().run(disk, f"tail_{window}", True)
            self.assertTrue(torch.equal(a["video"], b["video"]))
            self.assertTrue(torch.equal(a["audio"], b["audio"]))
            self.assertEqual(h3.frame_count(b["video"].shape[2]), window)
            self.assertEqual(b["parent"]["end"], 243)
            b["video"].zero_()
            self.assertTrue(torch.equal(h3.read_tensors(disk)["video"], state["tensors"]["video"]))

    def test_prepare_guides_are_clean_sampler_is_fresh(self):
        context, _ = h3.CRTP_H3SelectLatentContext().run(self.state(), "tail_22", True)
        positive, latent, plan = self.prepare(context)
        self.assertEqual(positive[0][1]["minimax_refs"], ["original-reference"])
        self.assertTrue(torch.equal(positive[0][1]["minimax_keyframes"][0]["latent"], context["video"]))
        for tensor in latent["samples"].unbind():
            self.assertEqual(torch.count_nonzero(tensor), 0)
        self.assertNotIn("noise_mask", latent)
        self.assertEqual(plan["trim_head"], 22)
        self.assertEqual(plan["raw_frames"] % 17, 5)
        self.assertGreater(plan["raw_frames"], 22)

    def test_keyframe_retained_and_video_token_not_accepted(self):
        image = h3.make_state({"samples": torch.ones(1,24,1,2,2)}, "image_keyframe", "encoded_pixels", "h3_vae")
        context, _ = h3.CRTP_H3SelectLatentContext().run(image, "image_keyframe", False)
        _, _, plan = self.prepare(context)
        self.assertEqual(plan["trim_head"], 0)
        self.assertIsNone(plan["parent"])
        with self.assertRaisesRegex(ValueError, "genuine"):
            h3.CRTP_H3SelectLatentContext().run(self.state(), "image_keyframe", False)
        with self.assertRaises(ValueError):
            h3.make_state({"samples": torch.ones(1,4,1,2,2)}, "image_keyframe", "sampled", "h3_vae")
        still = {"samples": Nested(torch.ones(1,24,1,2,2), torch.ones(1,32,2,2))}
        state = h3.make_state(still, "image_keyframe", "sampled", "h3_vae")
        context, _ = h3.CRTP_H3SelectLatentContext().run(state, "image_keyframe", False)
        self.assertIsNone(context["audio"])

    def test_short_offgrid_and_missing_audio_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            h3.CRTP_H3SelectLatentContext().run(self.state(5), "tail_22", False)
        state = self.state()
        state["meta"]["delivered_end"] -= 1
        with self.assertRaisesRegex(ValueError, "off H3"):
            h3.CRTP_H3SelectLatentContext().run(state, "tail_5", False)
        state = self.state(); del state["tensors"]["audio"]
        with self.assertRaisesRegex(ValueError, "absent"):
            h3.CRTP_H3SelectLatentContext().run(state, "tail_22", True)
        context, _ = h3.CRTP_H3SelectLatentContext().run(state, "tail_22", False)
        self.assertIsNone(context["audio"])

    def test_chain_keeps_raw_prefix_and_bounds_audio_phase(self):
        from fractions import Fraction
        state = self.state(124)
        for i in range(20):
            context, _ = h3.CRTP_H3SelectLatentContext().run(state, "tail_22", True)
            _, latent, plan = self.prepare(context, 5)
            state = h3.make_state(latent, "av_clip", "sampled", "h3_vae", plan)
            self.assertEqual(state["meta"]["delivered_start"], 22)
            self.assertEqual(state["meta"]["raw_frames"], h3.frame_count(state["tensors"]["video"].shape[2]))
            self.assertLessEqual(abs(Fraction(*state["meta"]["audio_phase"])), Fraction(1,80))

    def test_decode_trim_preserves_state_and_exact_av_duration(self):
        context, _ = h3.CRTP_H3SelectLatentContext().run(self.state(124), "tail_22", True)
        _, samples, plan = self.prepare(context)
        n = plan["raw_frames"]
        pixels = torch.arange(n).view(-1,1,1,1).expand(-1,32,32,3).float()
        decoder = types.SimpleNamespace(VAEDecode=lambda: types.SimpleNamespace(decode=lambda *_: (pixels,)))
        audio_decoder = types.SimpleNamespace(vae_decode_audio=lambda *_: {"waveform": torch.ones(1,2,round(n/24*32000)), "sample_rate": 32000})
        with patch.dict(sys.modules, {"nodes": decoder, "comfy_extras.nodes_audio": audio_decoder}):
            images, audio, state = h3.CRTP_H3DecodeAndTrim().run(samples, None, None, plan)
        self.assertEqual(images[0,0,0,0], 22)
        self.assertEqual(audio["waveform"].shape[-1], round((n-22)/24*32000))
        self.assertEqual(h3.frame_count(state["tensors"]["video"].shape[2]), n)

    def test_corrupt_layout_shapes_and_stale_lazy_handle(self):
        state = self.state()
        path = self.save(state)
        loaded = h3.read_state(path)
        Path(path).touch()
        with self.assertRaisesRegex(ValueError, "changed"):
            h3.read_tensors(loaded)
        state["meta"]["width"] = 64
        save_file(state["tensors"], str(self.root / "bad.safetensors"), metadata={"crtp_h3": json.dumps(state["meta"])})
        with self.assertRaisesRegex(ValueError, "canvas"):
            h3.read_state(str(self.root / "bad.safetensors"))

    def test_save_passthrough_no_overwrite_and_path_traversal(self):
        state = self.state()
        returned, path = h3.CRTP_SaveH3State().run(state, "sub/test")["result"]
        self.assertIs(returned, state)
        self.assertTrue(Path(path).is_file())
        with self.assertRaises(FileExistsError):
            h3.write_state(state, path)
        with self.assertRaises(ValueError):
            h3.CRTP_SaveH3State().run(state, "../escape")

    def test_vae_mismatch_and_existing_guides_are_rejected(self):
        context, _ = h3.CRTP_H3SelectLatentContext().run(self.state(), "tail_5", False)
        native = types.SimpleNamespace(_empty_av_latent=empty)
        with patch.dict(sys.modules, {"comfy_extras.nodes_minimax_h3": native}):
            with self.assertRaisesRegex(ValueError, "identifier mismatch"):
                h3.CRTP_H3PrepareLatentGeneration().run([[torch.zeros(1), {}]], context, 1, "other_vae")
            with self.assertRaisesRegex(ValueError, "already has"):
                h3.CRTP_H3PrepareLatentGeneration().run([[torch.zeros(1), {"minimax_keyframes": [1]}]], context, 1, "h3_vae")

    def test_ui_api_graphs_and_custom_node_schemas(self):
        folder = Path(__file__).parents[1] / "examples/h3_state"
        for api_path in folder.glob("*_api.json"):
            api = json.loads(api_path.read_text())
            ui = json.loads(api_path.with_name(api_path.name.replace("_api.json", ".json")).read_text())
            ui_nodes = {str(n["id"]): n for n in ui["nodes"]}
            for id, node in api.items():
                self.assertEqual(node["class_type"], ui_nodes[id]["type"])
                cls = h3.NODE_CLASS_MAPPINGS.get(node["class_type"])
                if cls:
                    schema = cls.INPUT_TYPES()
                    self.assertTrue(set(schema["required"]) <= set(node["inputs"]))
                    self.assertTrue(set(node["inputs"]) <= set(schema["required"]) | set(schema.get("optional", {})))
                for name, value in node["inputs"].items():
                    if isinstance(value, list):
                        source, output = value
                        self.assertIn(source, api)
                        self.assertLess(output, len(ui_nodes[source]["outputs"]))
                        socket = next(i for i in ui_nodes[id]["inputs"] if i["name"] == name)
                        link = next(link for link in ui["links"] if link[0] == socket["link"])
                        self.assertEqual(link[1:4], [int(source), output, int(id)])
            # The sample state is always fresh Prepare output, not a loaded state.
            self.assertEqual(api["13"]["inputs"]["latent_image"], ["8", 1])
            self.assertEqual(api["10"]["inputs"]["denoise"], 1.0)
            self.assertNotIn("25", api)  # optional assembly is muted by default
            if "20" in api:
                # Stock LoadImage validates file existence even on an unused
                # lazy branch, breaking latent-only runs without an image.
                self.assertEqual(api["20"]["class_type"], "CRTP_LazyLoadImageUpload")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_save_take_assemble_frame_counts_and_hash_checks(self):
        import subprocess
        parent = self.state(22)
        pixels = torch.zeros(22,32,32,3)
        audio = {"waveform": torch.zeros(1,2,44000), "sample_rate": 48000}
        first = h3.CRTP_H3SaveTake().run(parent, pixels, audio, "take", 18)["result"]
        context, _ = h3.CRTP_H3SelectLatentContext().run(first[0], "tail_5", True)
        _, latent, plan = self.prepare(context)
        child = h3.make_state(latent, "av_clip", "sampled", "h3_vae", plan)
        count = plan["raw_frames"] - 5
        second = h3.CRTP_H3SaveTake().run(child, torch.ones(count,32,32,3),
                    {"waveform": torch.zeros(1,2,count*2000), "sample_rate": 48000}, "take", 18)["result"]
        ordered = first[1] + "\n" + second[1]
        out = h3.CRTP_H3AssembleTakes().run(ordered, "assembly", False, 18)["result"][0]
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "json", out]))
        self.assertEqual(int(probe["streams"][0]["nb_read_frames"]), 22 + count)
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            h3.CRTP_H3AssembleTakes().run(second[1] + "\n" + first[1], "bad", False, 18)
        with open(first[2], "ab") as f:
            f.write(b"modified")
        with self.assertRaisesRegex(ValueError, "modified"):
            h3.CRTP_H3AssembleTakes().run(ordered, "bad", False, 18)


if __name__ == "__main__":
    unittest.main()
