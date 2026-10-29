# Experimental H3 latent-state toolkit

Eight nodes and five standalone test workflows. No product template changes, no
replacement of the existing CRTP pixel-based continuation helpers, no refinement.
This branch is based on CRTP `6f24baf8c53cf7fc82664d93120af9c1a5a79da4`.

## Start testing

Open these **UI** JSON files in ComfyUI (the matching `_api.json` files are for
API submission, not the canvas):

- `examples/h3_state/H3-State-Initial-Video-Test.json` — start here with an image
- `examples/h3_state/H3-State-First-Frame-Test.json` — start from a saved image latent
- `examples/h3_state/H3-State-Continuation-Test.json`
- `examples/h3_state/H3-State-Three-Segments-30s-Test.json` — three prompts, one queue, automatic assembly
- `examples/h3_state/H3-State-Three-Segments-30s-Turbo-8Step-Test.json` — the same chain with Turbo v1.1, 8 steps

Dependencies: ComfyUI with native H3 multi-frame guides and the guide/reference
packing fix, CRTP from this branch, KJNodes (lazy switch in the first workflow),
the usual H3 Ref2VA model/text encoder/video and audio VAEs, and FFmpeg on PATH.
Torch and safetensors are already ComfyUI dependencies. No extra node pack is
needed for continuation. The native contract was checked against the pod's core
commit `73c9bad4d21e7addbe1d13bc92eee0f1431b017d`.

The four base graphs start with the **base Ref2VA model, RES multistep, Simple,
30 steps, full denoise, fixed seed**. The separate Turbo graph adds the adapter
and uses **Euler, Simple, 8 steps**. None applies refinement. Generation
resolution is inherited from the guide state, not the unused latent output of
the prompt node. In Initial Video, the **Output canvas** image-scale node defines
that guide's resolution. Start at a modest resolution for the first GPU smoke test.

### Turbo 8-step variant

Open **H3-State-Three-Segments-30s-Turbo-8Step-Test.json** for the faster variant.
It uses the same Ref2VA base checkpoint, patched once by `LoraLoaderModelOnly`
with [LightX2V's Ref2V Turbo v1.1 8-step ComfyUI BF16 adapter](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors):

`minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors`

Install it under `ComfyUI/models/loras/`. The older 4-step adapter is not a
substitute. Strength is **1.0**, sampler **Euler**, scheduler **Simple**, **8 steps**,
and denoise **1.0**. All three guiders and the scheduler share the patched model.
No new CRTP node implementation is needed beyond the V2 IDs used by the base chain.

The three prompts, fixed seeds, 1024x576 canvas, 22-frame latent/audio context,
overlap trimming, state saves, deferred decoding and exact 720-frame export are
unchanged. Outputs have their own prefix: `output/h3_test/three_segments_turbo/`.
The adapter is named for 768p; this smoke-test canvas is deliberately kept equal
to the base graph for comparisons. Generation quality still needs a GPU test.

### Three prompts → one 30-second continuous shot

Open **H3-State-Three-Segments-30s-Test.json**, upload the first-frame image,
choose the canvas, and edit the three numbered prompts. Queue once. The graph:

1. Shares exactly one diffusion-model loader, one text-encoder loader, one video
   VAE loader and one audio VAE loader. Scheduler/sampler selection is also shared.
2. Generates the first clip from the image and passes the **raw sampled state**
   directly to segment two, then segment three. Each continuation uses 22 frames
   of motion context, with source audio reuse enabled by default.
3. Finishes all three samplers before decoding output clips. The optional
   `after_state` input is a scheduling dependency, not additional conditioning.
   The optional `raw_state` input preserves the parent IDs already used by the
   continuation chain; it validates that the plan and sampled tensors match.
4. Trims each continuation's 22-frame generated overlap, saves all three original
   clips and states, and wires their returned paths to Assemble Takes. No path
   copying or separate export queue is needed.

Outputs are under `output/h3_test/three_segments/`: `segment_1_*`, `segment_2_*`,
`segment_3_*` (each MP4 + safetensors), and `continuous_30s_*.mp4`.

At native 24 fps, the default ten-second settings produce 243 frames in the first
clip and 260 raw frames in each continuation. Removing 22 from each continuation
leaves **719 frames** (29.9583 s). This graph explicitly enables `pad_final_frame`
on assembly: repeat only the very last frame once and pad the soundtrack to give
**720 frames / exactly 30 seconds**. The three segment masters remain unchanged.
Changing segment durations changes the total; this option always adds one frame,
not an arbitrary stretch or duration clamp.

Shared loader nodes avoid duplicate model instantiation. ComfyUI can still offload
and reload weights between RAM and VRAM according to memory pressure; this graph
does not force all models to remain GPU-resident. Continuity of appearance/motion
still depends on generation quality and prompts, not just correct concatenation.
No refinement, seam blending, or optical flow is applied.

Update CRTP and restart ComfyUI before using this graph: the optional decoder and
assembly inputs are newer than the original eight-node release. Existing test
workflow files retain their previous behavior and wiring.

### Generate the initial video (no saved latent required)

1. Open **H3-State-Initial-Video-Test.json** and upload the opening image.
2. Set **Output canvas** width/height (multiples of 32; default 1024×576).
   It center-crops the image to that aspect ratio. The resized image supplies
   both the opening-frame guide and `<Picture 1>`; extra references start at
   `<Picture 2>`.
3. Edit the prompt and set duration on **Prepare Latent Generation** (default
   10 seconds, snapped to H3's native frame grid). Queue the workflow.
4. **Save Take** saves `output/h3_test/root_<id>.mp4` and the paired
   `root_<id>.safetensors`, containing the complete sampled video/audio latents.
   The opening clip is retained in full; no continuation overlap is removed.
5. Paste the saved `.safetensors` path into **Lazy Load H3 State** in
   **H3-State-Continuation-Test.json**, choose `tail_5` or `tail_22`, and generate
   the next segment. Paths are printed in the console and returned in node history.

This starter has no state-file loader or bootstrap switch. It uses the same eight
nodes already published; no additional node implementation or model is needed.

### First frame from a saved latent

1. Supply an `image_keyframe` state path to **Lazy Load H3 State**. Absolute paths
   work; relative paths are under ComfyUI's output directory.
2. Keep the lazy switch **true**. The image upload/encode branch is not executed.
3. Edit the prompt and queue. Save Take writes a root MP4 and `.safetensors` file.
   The opening frame is retained in the delivered clip.

If no state exists yet, the same graph has a one-time bootstrap route: set the
switch **false**, upload an image with width/height divisible by 32, and queue.
Save H3 State writes a keyframe file. Use that exact path and switch **true** for
subsequent generations. Source pixels are encoded only during that bootstrap.

For a **sampled H3 still**, connect its final sampler LATENT output directly to
State from Latent, select `image_keyframe` and `origin=sampled`, and use that
branch. Fizgig's one-video-token/two-audio-token nested still output is supported;
its audio is stored but not used as keyframe conditioning. This bypasses image
decoding and encoding. A single arbitrary token cut from a video is not a valid
standalone last-image representation. Use a 5/22-frame tail instead.

### Continue from a saved video state

1. Load the `.safetensors` file saved beside a root or continuation MP4.
2. Select `tail_22` (default) or `tail_5`; enable `reuse_audio` only if wanted.
3. Edit the new segment's prompt and queue. Save Take writes **only new delivered
   frames** to the MP4, but saves the **full raw sampled latent including context**.
4. For the next continuation, load this new sidecar, not a concatenated MP4.

The loader only reads the safetensors header. The context node reads the selected
tensor slices on CPU; it never loads/decodes the source MP4. In-memory states work
too: connect a Save Take state output straight to the next Select node.

Optional images/voices can be connected to the native Ref2VA prompt node. Use
`<Picture 1>`, etc. only for references actually connected there. A latent guide
is positional conditioning, **not** automatically a numbered Qwen image/video
reference. Reuse original reference images when semantic identity guidance is
needed. The guide itself never undergoes VAE re-encoding.

### Assemble

The continuation graph contains a muted **Assemble Takes** node. Unmute it only
for export and enter the root plus successive child state paths, one per line.
It verifies each paired MP4's hash, canvas and parent/end boundary, trims decoded
streams to their recorded delivered durations, and encodes the ordered original
segments once. Export does not create a new latent master. It does not apply
seam smoothing, color correction, latent refinement, or stream-copy optimization.
`allow_cuts=true` permits intentionally unrelated takes; it does not repair seams.

For an assembly-only queue, mute Save Take (and the optional standalone keyframe
saver) so generation is not requested. Masters are never overwritten. Paths are
printed in the server console and returned in node history, and video saves expose
a native ComfyUI video preview.

## Nodes

| Node | Contract |
| --- | --- |
| `CRTP_H3StateFromLatent` | Wrap a **clean final** native H3 sampler/encoder latent with explicit kind, origin and video-VAE identifier; optional generation plan. |
| `CRTP_SaveH3State` | Atomic safetensors save, exact dtype, state passthrough and saved path. |
| `CRTP_LazyLoadH3State` | Header-only handle, explicit path, file-change cache invalidation; no model loading. |
| `CRTP_H3SelectLatentContext` | Genuine image keyframe or phase-aligned final 5/22-frame window; optional audio, no silent snapping. |
| `CRTP_H3PrepareLatentGeneration` | Native fixed guide conditioning plus a **new empty AV latent** and explicit trim plan. |
| `CRTP_H3DecodeAndTrim` | Native full-latent decode, then frame/audio trim; keep complete raw latent state. |
| `CRTP_H3SaveTake` | Delivered MP4 plus verified raw-state sidecar, video preview and paths. |
| `CRTP_H3AssembleTakes` | Single delivery export from original verified takes and join metadata. |

## Why the old latent is not denoised again

Prepare attaches source tensors to `minimax_keyframes`. Native H3 packs these as
condition rows with update masks false. The sampler receives a fresh empty joint
AV latent and fresh noise through the ordinary full schedule. Source tensors never
become the sampler's starting samples. No custom sampler patch, noise-mask trick,
or second denoising pass is used. An exact guide does not guarantee that the model
reproduces the overlapped frames exactly; those generated frames are trimmed.
Latent preservation removes a VAE round trip; it cannot guarantee no generative
texture drift, a perfect seam, or faithful skin detail across arbitrary durations.

H3 video timing uses `(1,4,4,4,4)` pixel-frame spans per five latent tokens.
Valid full video states have `5k+2` temporal tokens (`17k+5` pixel frames). Tail
windows must start at a 17-frame group boundary. A short/off-grid state fails
explicitly. A continuation's delivered range is stored separately from its raw
range, so trimming never destroys the latent prefix needed by later operations.

Audio uses a 40 Hz latent grid versus 24 fps video. Selection rounds **absolute
source boundaries**, not the duration independently. The selected audio's offset
is represented by a separate native guide position, including fractional/negative
frame positions. Newly sampled AV starts on its own zero grid. Delivered waveform
length is exact to the frame duration, with only decoder-end padding if needed.
The CPU checks verify the timing arithmetic, not perceptual audio seam quality.

## Storage and limitations

Format `crtp_h3_state_v1` stores `video`, optional `audio`, and a JSON header in
safetensors metadata. The header records layout, kind, source origin, explicit
video-VAE identifier, canvas, 24 fps timing, raw and delivered ranges, audio phase,
state/parent IDs, and paired media filename/hash. No pickle, prompt, credentials,
or full workflow is serialized. Saves preserve FP16/BF16/FP32/FP64 tensors without
quantization; MP4 delivery is 8-bit H.264/AAC. Input tensors must be clean final
sampler output (sigma zero); numerical contents cannot prove that automatically.

VAE identity is an explicit user contract: use the identifier of the actual VAE
and keep it consistent with the loader. No arbitrary resolution conversion,
unknown foreign sidecar import, multi-batch states, arbitrary edited cuts, or
automatic MP4 reconstruction is supported. To import a known native H3 tensor,
load it with its appropriate loader and use State from Latent with correct
metadata. Foreign sidecars need an explicit adapter, not a renamed file.

Saves use unique names and atomic hard-link publication. A crash before sidecar
publication can leave a playable orphan MP4, which is not a valid take. Keep the
paired files together. A delivery video whose bytes change cannot be used for
verified assembly; latent-only continuation remains possible from its original
unmodified state. Disk capacity must account for raw latent sidecars.

## Verification

```bash
python -m unittest discover -s tests -p 'test_h3_state.py' -v
python scripts/check_h3_native_contract.py /app/ComfyUI
python scripts/build_h3_state_workflows.py
python scripts/build_h3_three_segment_workflow.py
python scripts/build_h3_three_segment_workflow.py --turbo
```

CPU tests exercise real torch, safetensors and FFmpeg, dtype-exact round trips,
tail slices, lineage, corruption rejection, fresh-sampling separation, timing,
graph wiring, and export frame counts. The native contract script executes the
installed ComfyUI source's tensor/layout utilities without model weights. A GPU
generation is still needed to evaluate appearance, seams, runtime and VRAM.

## Design references

This is an original implementation using ComfyUI's existing native guide API.
No source code was copied from these projects, and none is a runtime dependency:

- [OBVPM Timeline](https://github.com/chanon/comfyui-obvpm-timeline): separate raw
  latent masters, delivered ranges, lineage-aware clip assembly and audio timing.
- [pmhaidn H3 Extender](https://github.com/pmhaidn/ComfyUI-Minimax-H3-Extender): pass
  previous sampled latents directly as continuation context, avoiding re-encoding.
- [LongTakeNoCuts](https://github.com/xyzDist/H3-LongTakeNoCuts): distinguish loss
  from reconstruction and residual generative drift. Its refinement/blending
  approach is deliberately deferred; this release establishes a clean baseline.

### Version mismatch: unexpected `raw_state`

`CRTP_H3DecodeAndTrim.run() got an unexpected keyword argument 'raw_state'`
means the running decoder class predates the three-segment workflow. Fetching
files or updating the local pod dependency pin does not update a running Python
class or an already published image. Update the CRTP checkout and restart the
ComfyUI process, or rebuild the pod image from the updated dependency manifest.
A browser refresh alone is insufficient. Restarting clears unsaved sampler caches;
already saved MP4/state files survive.

The three-segment graph now uses `CRTP_H3DecodeAndTrimV2` and
`CRTP_H3AssembleTakesV2`. These are aliases of the eight core nodes, not new
sampling logic. An older installation reports missing nodes during prompt
validation rather than sampling first and failing at decode. Load the updated
workflow JSON to get this protection. The original node IDs and single-segment
workflows remain supported.
