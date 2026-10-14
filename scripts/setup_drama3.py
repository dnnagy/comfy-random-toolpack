"""Install isolated Drama 3 dependencies and optionally download pinned weights.

Requires uv and Git on PATH. Run --help for examples. Never installs into
ComfyUI's environment and never changes a global CUDA installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

PACK = Path(__file__).resolve().parents[1]
MLX_REV = "478f3189974fb7dcafe886afba09887f03996aec"
CUDA_REV = "94730a87b2b3b2f8383274918c3c4ed041821ecb"
MODELS = {
    "mlx": ("mlx-community/fish-audio-drama-3-4bit", "cc639e27aba7edb73429d3e3806f471dcd79d316"),
    "cuda": ("darioooooo0o/drama-3-w4a16-32gb", "8061bada1a8bd097e8cb107de7528c3610ddd302"),
}
PATCH_SHA256 = "95a2d8ac797181980ba67c58733696707fd376b6224061b59ee61ce69ef02111"


def run(*command, cwd=None):
    subprocess.run([str(c) for c in command], check=True, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=MODELS)
    parser.add_argument("--model-dir", type=Path, help="Download the checkpoint to this exact directory (about 22/25 GB).")
    parser.add_argument("--runtime-dir", type=Path, help="Override the default toolpack .venv-drama3-BACKEND directory.")
    parser.add_argument("--download-only", action="store_true", help="Use an already installed runtime to download weights.")
    args = parser.parse_args()
    if args.backend == "mlx" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        parser.error("MLX requires an Apple Silicon Mac.")
    if args.backend == "cuda" and platform.system() != "Linux":
        parser.error("CUDA setup requires NVIDIA Linux/WSL2, a compatible driver, and CUDA development tools.")
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv before running this setup script.")
    venv = (args.runtime_dir or PACK / f".venv-drama3-{args.backend}").expanduser().resolve()
    python = venv / "bin/python"
    if not args.download_only:
        if not python.exists():
            run(uv, "venv", "--python", "3.12", venv)
        if args.backend == "mlx":
            run(uv, "pip", "install", "--python", python,
                f"mlx-audio @ git+https://github.com/stickerdaniel/mlx-audio@{MLX_REV}",
                "mlx==0.32.3", "transformers==5.19.0", "soundfile", "scipy")
        else:
            # Dedicated pinned checkout: never patch the user's shared SGLang.
            source = PACK / ".drama3-runtime" / f"sglang-omni-{CUDA_REV}"
            if not (source / ".git").exists():
                source.mkdir(parents=True, exist_ok=True)
                run("git", "init", source)
                run("git", "-C", source, "remote", "add", "origin", "https://github.com/sgl-project/sglang-omni.git")
            current = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
            if current.returncode:
                run("git", "-C", source, "fetch", "--depth", "1", "origin", CUDA_REV)
                run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
            head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if head != CUDA_REV:
                raise RuntimeError(f"Unexpected runtime checkout revision: {head}")
            repo, revision = MODELS["cuda"]
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/apply_patch.py", timeout=60) as response:
                patch = response.read()
            if hashlib.sha256(patch).hexdigest() != PATCH_SHA256:
                raise RuntimeError("Quantization loader patch checksum mismatch; refusing to execute it.")
            patch_path = source.parent / "apply_w4a16_patch.py"
            patch_path.write_bytes(patch)
            run(python, patch_path, source)
            # cwd matters: upstream's uv protobuf override is needed by fishaudio.
            run(uv, "pip", "install", "--python", python, "--prerelease=allow", "-e", ".[fishaudio]", "soundfile", "ninja", cwd=source)
    if not python.is_file():
        parser.error(f"Runtime missing: {python}. Run without --download-only first.")
    if args.model_dir:
        repo, revision = MODELS[args.backend]
        target = args.model_dir.expanduser().resolve()
        code = "from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=4)"
        run(python, "-c", code, repo, revision, target)
        (target / "crtp-download.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2) + "\n")
    print(f"Runtime ready: {python}")
    print(f"For a custom runtime path, set CRTP_DRAMA3_{args.backend.upper()}_PYTHON={python}")


if __name__ == "__main__":
    main()
