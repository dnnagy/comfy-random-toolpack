"""Opt-in real GGUF+mmproj smoke test; uses synthetic images and no Comfy install.

python tests/vl_native_integration.py --model /path/model.gguf --mmproj /path/mmproj.gguf
Set CRTP_LLAMA_SERVER when llama-server is not on PATH. Requires torch, numpy, Pillow.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mmproj', required=True)
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
    answer = vl.CRTP_VLTextGenerate().generate(vl.GGUF,
        'Identify the shape and color in each image. Answer in numbered lines, one per image.',
        torch.stack(images), max_length=256,
        gguf_model=str(Path(args.model).resolve()), mmproj=str(Path(args.mmproj).resolve()))[0]
    print(json.dumps({'answer': answer, 'elapsed_seconds': round(time.monotonic() - start, 2),
                      'images': args.images}, indent=2))
    assert all(word in answer.lower() for word in ('red', 'square', 'blue', 'circle')), answer
    assert len([line for line in answer.splitlines() if line.strip()]) == args.images, answer


if __name__ == '__main__':
    main()
