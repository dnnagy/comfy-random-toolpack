"""Build a model-free export graph for already saved H3 takes."""
from build_h3_state_workflows import Graph


def main():
    g = Graph()
    paths = "\n".join(f"h3_test/three_segments_turbo/REPLACE_segment_{i}.safetensors" for i in (1,2,3))
    g.node(300,"CRTP_H3AssembleTakesV2",[30,30],[],[("video_path","STRING")],
           [("state_paths",paths),("filename_prefix","h3_test/recovered/continuous_30s"),
            ("allow_cuts",False),("crf",18),("pad_final_frame",True)],
           "FINAL — assemble saved takes only",size=[700,360])
    g.node(99,"Note",[30,440],[],[],[("text",
           "No models or samplers. Paste saved .safetensors paths into state_paths, one per line in playback order. "
           "The paired MP4 files must remain alongside their states. Do not list the MP4 paths.\n"
           "Keep allow_cuts false to verify the parent/child sequence. The three-segment 30-second graphs "
           "already trim their overlaps; this export does not trim them again.\n"
           "pad_final_frame adds exactly one final repeated frame (719 → 720 frames for the default three segments). "
           "Turn it off for exports that should preserve the original total frame count.\n"
           "Queue this graph alone. Updating/restarting CRTP is safe for already saved takes. "
           "The originals remain unchanged, and failed partial exports are removed.")],size=[700,260])
    g.save("H3-State-Assemble-Saved-Takes")


if __name__ == "__main__":
    main()
