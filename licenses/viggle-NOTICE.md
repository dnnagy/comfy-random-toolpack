# Viggle-Animate adaptation

The reference geometry and conditioning packing in `nodes/viggle.py` are adapted
from **ComfyUI-Viggle-Animate-H3 1.3.2**, published by **Saganaki22**, licensed under
Apache License 2.0. The original repository URL is
https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3.

The source was obtained from the Comfy registry's published 1.3.2 archive:
https://cdn.comfy.org/saganaki22/comfyui-viggle-animate-h3/1.3.2/node.zip

Archive SHA-256:
`6d7c203ed76141e7030eaf2940e02a45251ebc88992fb7c7ab2ea3ea501ddd95`

The original license is preserved in [Apache-2.0-Viggle.txt](Apache-2.0-Viggle.txt).
The registry archive contains no separate upstream NOTICE file. CRTP changes
include validation, bounded native-video preparation and FPS normalization,
audio alignment, resolution parsing, fixed distilled sampler schedules, output
padding removal, and tests. Chunked sampling, UI extensions, caches, and optional
third-party node dependencies are not included.
