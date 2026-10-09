# Viggle-Animate (MiniMax H3)

These six CRTP nodes use ComfyUI's native MiniMax H3 model, VAE, sigma shift,
sampler, and video APIs. ComfyUI v0.37.0 or newer with native H3 support is
required. The unavailable ComfyUI-Viggle-Animate-H3 GitHub repository is not an
installation dependency. See [upstream attribution](../licenses/viggle-NOTICE.md).

## Model files

For the compact INT8 ConvRot configuration from
[drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI):

* `models/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors`
* `models/loras/viggle_animate_dmd_lora_r64.safetensors`
* `models/text_cond/fixed_embed_fwd_anyframe.safetensors`
* `models/vae/minimax_h3_video_vae_fp16.safetensors` (reuse the base H3 video VAE)

The frozen conditioning replaces the Qwen text encoder. No audio VAE is needed
when decoding only the video stream and remuxing the driving video's audio.

## Wiring and recommended defaults

1. Native `LoadVideo` → `CRTP_VigglePrepareVideo`: 124 frame cap, start time 0,
   preserve audio enabled. Connect the selected resolution's width and height
   to this node as well as the conditioning node. Optional size inputs default
   to 864×480 for older workflows. It streams a bounded lazy trim, selects frames
   using actual presentation timestamps (including VFR), and **always resamples
   to 24 fps**. Only selected frames are resized and converted to RGB tensors;
   the full-resolution source clip is never materialized in memory. The driving
   dimensions fit inside both the selected canvas and the source, never upscale,
   and round down to the H3 32-pixel grid. Display rotation is preserved, and
   resizing keeps the source aspect ratio apart from grid rounding; no crop is
   applied. Outputs are images, aligned audio, actual frame count, and 24 fps.
   Inputs shorter than five normalized frames fail clearly. The cap supports
   5–362 frames; this is a single-shot workflow. Connect an uncropped native
   LoadVideo so timestamps are available; assembled/cropped VIDEO objects are
   rejected rather than silently losing their transforms.
   Audio is decoded against each audio frame's own time base so nonzero start
   offsets and delayed audio tracks remain aligned with the video timeline.
2. `CRTP_ViggleResolution` parses a `WIDTHxHEIGHT (optional label)` string into
   width and height. Default: `864x480 (landscape, 0.4 MP)`. Dimensions must be
   multiples of 32, no more than 1536 per axis, at most 1,048,576 pixels, and
   between 1:4 and 4:1. Higher resolutions and frame counts need more memory.
3. `CRTP_ViggleTextCondLoader` loads the fixed embedding from `text_cond`.
4. `CRTP_ViggleAnimateConditioning` receives prepared frames, a reference still,
   frozen conditioning, video VAE, width, and height. It packs the driving video
   **before** the image, as required by the finetune. The driving video preserves
   aspect ratio within the selected canvas and source bounds on a 32-pixel grid.
   Prepared smaller driving frames are never upscaled by conditioning. The still
   preserves aspect ratio and targets the short edge
   without an area cap, matching upstream; its aspect ratio is restricted to
   1:4–4:1 to bound its size. Use a repainted frame from the driving video with
   matching pose, framing, and background for the intended workflow.
5. Load the pruned model, apply the r64 LoRA at strength 1.0, then native
   `MiniMaxH3SigmaShift` with video and audio shifts 3.0. Use `BasicGuider` (CFG 1),
   Euler, and `CRTP_ViggleSigmas` default **3 sampler evaluations**. This is the
   upstream four-point schedule `[1, 6/7, 3/5, 0]`. Optional 5- and 7-evaluation
   schedules use the published six- and eight-point sequences; more evaluations
   are not necessarily better and can oversharpen. Do not append another zero.
6. Decode the video latent, then `CRTP_ViggleTrimOutput` with the prepare node's
   actual frame count. H3 sampling pads up to `17k+5`; trimming removes this
   padding from the delivered video. Native `CreateVideo` consumes the trimmed
   frames and the prepare node's audio/fps, then `SaveVideo` writes it.

Source audio is trimmed with the source video and muxed into the result; it is
not an audio conditioning or lip-sync feature. Missing or disabled audio yields
`None`, which native `CreateVideo` accepts. Short audio is padded with silence to
the output duration. Source FPS must be positive and at most 240. Source
resolution and FPS affect streaming decode work, but no source-size pixel budget
rejects otherwise valid 1080p/4K footage. RGB tensor memory is bounded by the
selected resolution and 362-frame maximum; no source-sized frame batch or
full-resolution RGB conversion is created. The video codec still needs a few
source-sized decoder surfaces. At the maximum 1 MP and 362 frames, the float32
RGB output alone can occupy about 4.6 GB; smaller defaults need less memory.

## Validation

Run `python -m unittest discover -s tests -p 'test_viggle.py'` with PyTorch and
safetensors installed. Tests exercise real tensor operations, frozen embedding
loading, bounded decode calls, frame-rate conversion, audio alignment, geometry,
padding removal, native H3 reference ordering, and AV latent shapes. A GPU render
is still necessary to assess visual quality and measured performance.

With ComfyUI and its dependencies available, run
`PYTHONPATH=/path/to/ComfyUI python tests/viggle_native_integration.py` for a CPU
integration test. It creates a small variable-frame-rate file with seekable audio,
checks exact 24 fps sampling and a nonzero audio start offset, exercises native
H3/NestedTensor packing, verifies high-FPS downscaling and phone display rotation
with a non-aligned source width, and exports/reloads an MP4 with native CreateVideo.
