# Audio Waveform Editor

`CRTP Audio Waveform Editor` lives under **CRTP → audio**. It edits standard ComfyUI AUDIO on the CPU, so the same node works before MLX or CUDA voice generation. It requires no additional packages beyond ComfyUI's existing torch, NumPy and soundfile.

## Reference-audio workflow

Connect `Load Audio → CRTP Audio Waveform Editor → CRTP Drama 3 Reference` (or the generator's reference AUDIO input). You can also connect the output to Preview Audio or Save Audio.

1. Click **Load waveform**. This queues the editor and its upstream inputs only. If the input comes from a generator, that generator may need to run; downstream nodes do not run for this action.
2. Click **Expand ↗** for a large editor. Drag across the waveform to select a range, then click **Disable**. The range becomes a disabled segment, while the original waveform remains visible.
3. Click a colored segment or its card to select it. Toggle its checkbox, or click **Enable/Disable**. Drag a boundary in the lower segment bar to adjust a cut. **In s / Out s → Select** provides numeric precision.
4. Click **▶ Kept** to audition the enabled sections joined together. **▶ Source** plays the original from the playhead; **▶ Selection** auditions the selected source range. Stop playback before adjusting cuts.
5. Run the workflow normally. `edited_audio` contains only enabled sections, in their original order. Save the workflow to retain cuts.

The source file is never overwritten. **Keep all** restores the full recording; Undo/Redo restore previous editing states within the current session. Saved cuts are bound to the exact source samples: if the input changes, normal execution stops and asks you to load/review the new waveform.

## Quickly shorten pauses

Open **Mark silence…**, then **Mark gaps as cuts**. Suggested starting values:

- **Below dB: −38** — level below which a region is considered quiet (dBFS).
- **Min gap s: 0.5** — leave short, natural pauses alone.
- **Keep edge s: 0.08** — retain 80 ms on either side of an internal quiet region, preserving breath and word endings. A long internal pause retains approximately 160 ms in total.

This adds cuts without enabling sections you already disabled. Audition the result; quiet speech can resemble silence. Undo restores the previous state. Lower the threshold, for example to −45, if quiet words are being marked. Increase edge padding if word endings sound clipped.

For voice cloning, keep a little natural breathing room. If you remove **only silence**, the reference transcript stays the same. If you remove any spoken words, update the transcript to exactly match the retained speech.

## Controls and output

- Click the waveform to seek; drag above the colored bar to select.
- **S** splits at the playhead. **Delete/Backspace** disables the selected range or segment; **E** enables it.
- **Space** plays/stops the kept audio. **⌘/Ctrl+Z** undoes; **⇧⌘/Ctrl+Z** redoes.
- Arrow keys seek 0.1 seconds; Shift+Arrow seeks 1 second.
- Use the zoom slider and timeline scrollbar; Ctrl/⌘+wheel zooms. **Fit** shows the full clip.
- **Escape** closes the expanded view.

`edge_fade_ms` defaults to 3 ms and softens retained edges next to cuts to reduce clicks. It does not overlap audio or change output length. Set it to 0 for exact retained float32 samples. The browser audition skips cuts; the final node output applies the edge fades.

Outputs are `edited_audio` (AUDIO), `removed_seconds` (FLOAT), and `edit_map` (JSON STRING containing original/output sample ranges). The original sample rate and all channels are retained. Browser playback previews the first two channels for recordings with more than two channels. One audio item is supported per node (batch size 1), with up to 2,048 segments.

Waveform peaks and edit state are saved in the workflow. Playback WAVs live in ComfyUI's temporary directory. After a restart or temp cleanup, click **Load waveform** to recreate playback; cuts survive when the source is unchanged. Undo history is session-only. Large recordings require browser memory to decode for audition.

## Validation

```sh
python -m unittest discover -s tests -p 'test_audio_editor.py' -v
node --test tests/audio_editor_core.test.mjs
```
