"""CPU-only compatibility check against a real ComfyUI source checkout.

Executes the checkout's exact tensor/layout utilities without importing its GPU
runtime. This checks native shape/guide contracts, not transformer inference.
Usage: python scripts/check_h3_native_contract.py /app/ComfyUI
"""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    picked = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in picked} == set(names), f"Native API drift in {path}"
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)


def check(root):
    namespace = {"torch": torch, "math": math, "FRAME_PER_TOKEN": (1,4,4,4,4), "FRAME_RESCALE": 5/3, "FPS": 24, "AUDIO_LATENT_FPS": 40}
    definitions(root / "comfy/nested_tensor.py", ["NestedTensor"], namespace)
    comfy = types.SimpleNamespace(nested_tensor=types.SimpleNamespace(NestedTensor=namespace["NestedTensor"]),
                                  model_management=types.SimpleNamespace(intermediate_device=lambda: torch.device("cpu")))
    namespace["comfy"] = comfy
    definitions(root / "comfy_extras/nodes_minimax_h3.py", ["align_frame_count", "video_latent_t", "temporal_shape", "_empty_av_latent"], namespace)
    definitions(root / "comfy/ldm/minimax/model.py", ["_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid", "_ref_t_span", "_audio_grid", "_video_grid", "PackedLayout"], namespace)
    spec = importlib.util.spec_from_file_location("h3_state_check", Path(__file__).parents[1] / "nodes/h3_state.py")
    h3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h3)
    native = types.SimpleNamespace(_empty_av_latent=namespace["_empty_av_latent"])
    for window in (1,5,22):
        if window == 1:
            # Fizgig stills have one video token AND two audio tokens.
            samples = {"samples": namespace["NestedTensor"]((torch.ones(1,24,1,2,2), torch.ones(1,32,2,2)))}
            state = h3.make_state(samples, "image_keyframe", "sampled", "h3")
            context, _ = h3.CRTP_H3SelectLatentContext().run(state, "image_keyframe", False)
        else:
            samples, _ = native._empty_av_latent(32,32,124)
            samples["samples"].tensors[0].fill_(0.5)
            state = h3.make_state(samples, "av_clip", "sampled", "h3")
            context, _ = h3.CRTP_H3SelectLatentContext().run(state, f"tail_{window}", True)
        with patch.dict(sys.modules, {"comfy_extras.nodes_minimax_h3": native}):
            cond, fresh, plan = h3.CRTP_H3PrepareLatentGeneration().run([[torch.zeros(1), {}]], context, 1, "h3")
        video, audio = fresh["samples"].unbind()
        guides = cond[0][1]["minimax_keyframes"]
        layout = namespace["PackedLayout"](4,video.shape[2],2,2,audio.shape[-1],keyframes=guides)
        fixed_video_rows = context["video"].shape[2]
        assert not layout.img_update[:fixed_video_rows].any()
        assert layout.img_update[fixed_video_rows:].all()
        assert not video.any() and not audio.any()
        assert torch.equal(guides[0]["latent"], context["video"])
        if window > 1:
            fixed_audio_rows = context["audio"].shape[-1]*2
            assert not layout.audio_update[:fixed_audio_rows].any()
            assert layout.audio_update[fixed_audio_rows:].all()
            assert math.isclose(guides[1]["resolved_frame_index"] / 24, float(__import__("fractions").Fraction(*context["audio_phase"])))
        assert plan["audio_phase"] == [0,1]
        print(f"PASS native {window}-frame context: fixed guide rows, fresh target, valid H3 grid")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "/app/ComfyUI"))
