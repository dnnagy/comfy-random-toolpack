"""Opt-in real GGUF+mmproj smoke test; uses synthetic images and no Comfy install.

python tests/vl_native_integration.py --model /path/model.gguf --mmproj /path/mmproj.gguf
Set CRTP_LLAMA_SERVER when llama-server is not on PATH. Requires torch, numpy, Pillow.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mmproj', required=True)
    parser.add_argument('--selector', default='Qwen3.8-27B-HauhauCS-Q5_K_P')
    parser.add_argument('--images', type=int, choices=(2, 10), default=2)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    spec = importlib.util.spec_from_file_location('vl_text', Path(__file__).parents[1] / 'nodes/vl_text.py')
    vl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vl)
    mm = types.ModuleType('comfy.model_management')
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    comfy = types.ModuleType('comfy')
    comfy.model_management = mm
    sys.modules.update({'comfy': comfy, 'comfy.model_management': mm})
    sys.modules['folder_paths'] = types.SimpleNamespace(models_dir='/unused', folder_names_and_paths={})
    images = []
    for index in range(args.images):
        image = Image.new('RGB', (768, 768), 'white')
        draw = ImageDraw.Draw(image)
        if index % 2 == 0:
            draw.rectangle((160, 160, 608, 608), fill='red')
        else:
            draw.ellipse((160, 160, 608, 608), fill='blue')
        images.append(torch.from_numpy(np.array(image).astype(np.float32) / 255))
    start = time.monotonic()
    answer = vl.CRTP_VLTextGenerate().generate(args.selector,
        'Identify the shape and color in each image. Answer in numbered lines, one per image.',
        torch.stack(images), max_length=256,
        gguf_model=str(Path(args.model).resolve()), mmproj=str(Path(args.mmproj).resolve()),
        gemma_gguf_model=str(Path(args.model).resolve()), gemma_mmproj=str(Path(args.mmproj).resolve()))[0]
    print(json.dumps({'answer': answer, 'elapsed_seconds': round(time.monotonic() - start, 2),
                      'images': args.images}, indent=2))
    assert all(word in answer.lower() for word in ('red', 'square', 'blue', 'circle')), answer
    # A harmless introductory sentence is allowed; verify each numbered image.
    lines = re.findall(r'^\s*(\d+)[.)]\s+(.+)$', answer, re.MULTILINE)
    assert [int(number) for number, _ in lines] == list(range(1, args.images + 1)), answer
    for index, (_, line) in enumerate(lines):
        expected = ('red', 'square') if index % 2 == 0 else ('blue', 'circle')
        assert all(word in line.lower() for word in expected), answer


if __name__ == '__main__':
    main()
