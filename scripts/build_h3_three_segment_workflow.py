"""One model-loader set, three sequential latent-guided takes, automatic export."""
from copy import deepcopy

from build_h3_state_workflows import build, VAE


def replace_link(g, source, slot, target, name):
    node = g.sockets[target]
    entry = next(i for i in node["inputs"] if i["name"] == name)
    old = next(link for link in g.links if link[0] == entry["link"])
    g.sockets[old[1]]["outputs"][old[2]]["links"].remove(old[0])
    old[1:3] = [source, slot]
    g.sockets[source]["outputs"][slot]["links"].append(old[0])
    g.api[str(target)]["inputs"][name] = [str(source), slot]


def widgets(g, id, updates):
    api = g.api[str(id)]["inputs"]
    # Widget order is explicit for the nodes changed below.
    names = {5: ["prompt", "width", "height", "length", "ref_image_size"],
             15: ["filename_prefix", "crf"]}[id % 100]
    for name, value in updates.items():
        api[name] = value
        g.sockets[id]["widgets_values"][names.index(name)] = value


def main(turbo=False):
    g = build(False, initial=True)
    original = deepcopy(g.api)
    originals = deepcopy(g.sockets)
    # Reuse all four loaders, original image/reference, scheduler and sampler.
    per_segment = (5,7,8,9,12,13,14,15)
    for offset in (100,200):
        for id in per_segment:
            n = originals[id]
            clone = deepcopy(n)
            clone["id"] += offset
            clone["pos"][1] += offset * 16
            clone["order"] = len(g.nodes)
            for entry in clone["inputs"]:
                entry["link"] = None
            for entry in clone["outputs"]:
                entry["links"] = []
            g.nodes.append(clone); g.sockets[clone["id"]] = clone
            g.api[str(clone["id"])] = deepcopy(original[str(id)])
        for id in per_segment:
            for name,value in original[str(id)]["inputs"].items():
                if isinstance(value,list):
                    source,slot = int(value[0]),value[1]
                    source = source+offset if source in per_segment else source
                    g.link(source,slot,id+offset,name)
        g.api[str(7+offset)]["inputs"].update(mode="tail_22", reuse_audio=True)
        g.sockets[7+offset]["widgets_values"] = ["tail_22", True]

    actions = ["The person looks toward the window and slowly smiles.",
               "Continuing the same movement, the person slowly turns back toward the camera and rests a hand on the table.",
               "Continuing immediately, the person picks up the cup from the table, takes a small sip, and lowers it."]
    for index,offset in enumerate((0,100,200)):
        prompt = ("subject_definitions:\nThe person and room shown in <Picture 1>.\n\nsummary:\n"
                  + ("Video continuation. " if index else "")
                  + "One continuous documentary shot with no cuts.\n\nretention_analysis:\n"
                  "Preserve the person's appearance, clothes, room and lighting.\n\ndetailed_description:\n[Shot 1] "
                  + actions[index] + " The camera remains steady. Preserve natural skin texture and movement. "
                  "Do not restart the action or introduce a cut.\n\noverall_soundscape:\nContinuous quiet room ambience.\n\nnon_diegetic_music:\nN/A")
        widgets(g,5+offset,{"prompt":prompt})
        widgets(g,15+offset,{"filename_prefix":f"h3_test/three_segments/segment_{index+1}"})
        g.sockets[5+offset]["title"] = f"PROMPT {index+1} — next 10 seconds"
        g.sockets[12+offset]["widgets_values"][0] = 10104+index
        g.api[str(12+offset)]["inputs"]["noise_seed"] = 10104+index
        # Preserve parent identity before decoding; pass these in-memory states
        # forward so the next sampler has no dependency on an intermediate MP4.
        state_id = 30+offset
        g.node(state_id,"CRTP_H3StateFromLatent",[2550,30+offset*16],
               [("latent","LATENT"),("plan","CRTP_H3_PLAN")],[("state","CRTP_H3_STATE")],
               [("kind","av_clip"),("origin","sampled"),("vae_id",VAE)],
               f"Segment {index+1}: raw sampled state (no decode)")
        g.link(13+offset,0,state_id,"latent"); g.link(8+offset,2,state_id,"plan")
        g.link(state_id,0,14+offset,"raw_state")
        if offset:
            replace_link(g,state_id-100,0,7+offset,"state")
    # All generation finishes before any output decoder is executed. ComfyUI
    # may still offload shared models according to available VRAM.
    for offset in (0,100,200):
        g.link(230,0,14+offset,"after_state")
        # A legacy install must reject the prompt before executing any sampler.
        g.sockets[14+offset]["type"] = "CRTP_H3DecodeAndTrimV2"
        g.sockets[14+offset]["properties"]["Node name for S&R"] = "CRTP_H3DecodeAndTrimV2"
        g.api[str(14+offset)]["class_type"] = "CRTP_H3DecodeAndTrimV2"
    g.node(300,"CRTP_H3AssembleTakesV2",[3350,300],
           [("state_path_1","STRING"),("state_path_2","STRING"),("state_path_3","STRING")],
           [("video_path","STRING")],[("state_paths",""),("filename_prefix","h3_test/three_segments/continuous_30s"),
                                      ("allow_cuts",False),("crf",18),("pad_final_frame",True)],
           "FINAL — 30-second continuous shot",size=[440,310])
    for i,offset in enumerate((0,100,200),1):
        g.link(15+offset,1,300,f"state_path_{i}")
    g.sockets[99].update(pos=[3350,700],size=[550,440],widgets_values=[
        "Upload the first image and edit PROMPT 1, 2, 3. Queue once.\n"
        "One shared diffusion loader, text encoder loader, video VAE and audio VAE; the scheduler and sampler selection are shared too.\n"
        "Segment 1 starts from the image. Segments 2/3 use the previous raw sampled state's last 22 frames, including audio context.\n"
        "The generated overlap is removed once. Three original segment videos/states and the assembled output are saved automatically.\n"
        "All sampling finishes before output decoding. Shared loaders avoid duplicate graph loads; ComfyUI can still move models between RAM and VRAM.\n"
        "Native timing: 243 + (260-22) + (260-22) = 719 frames. Final export repeats ONLY the final frame once and pads audio to reach 720 frames / 30 seconds. Masters remain unmodified.\n"
        "Changing durations changes the total; pad_final_frame always adds exactly one frame.\n"
        "Requires CRTP V2 decoder/assembler IDs; update CRTP and restart ComfyUI if missing. No refinement or seam smoothing. A continuous shot is prompted, not guaranteed by concatenation."])
    name = "H3-State-Three-Segments-30s-Test"
    if turbo:
        lora = "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"
        g.node(40,"LoraLoaderModelOnly",[30,1320],[("model","MODEL")],[("MODEL","MODEL")],
               [("lora_name",lora),("strength_model",1.0)],
               "Shared Turbo v1.1 — 8 steps",size=[410,150])
        g.sockets[40]["properties"]["models"] = [{
            "name": lora, "directory": "loras",
            "url": "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/" + lora}]
        g.link(1,0,40,"model")
        for target in (9,109,209,10):
            replace_link(g,40,0,target,"model")
        g.api["10"]["inputs"]["steps"] = 8
        g.sockets[10]["widgets_values"][1] = 8
        g.api["11"]["inputs"]["sampler_name"] = "euler"
        g.sockets[11]["widgets_values"] = ["euler"]
        for index,offset in enumerate((0,100,200),1):
            widgets(g,15+offset,{"filename_prefix":f"h3_test/three_segments_turbo/segment_{index}"})
        prefix = "h3_test/three_segments_turbo/continuous_30s"
        g.api["300"]["inputs"]["filename_prefix"] = prefix
        g.sockets[300]["widgets_values"][1] = prefix
        g.sockets[99]["widgets_values"][0] = (
            "TURBO: Ref2V v1.1 8-step LoRA, strength 1.0, Euler + Simple, full denoise. "
            "One shared LoRA-patched base model serves all three samplers. "
            "Canvas remains 1024x576 for comparison with the base test.\n"
            + g.sockets[99]["widgets_values"][0])
        g.sockets[99]["size"][1] += 100
        name = "H3-State-Three-Segments-30s-Turbo-8Step-Test"
    g.save(name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbo", action="store_true", help="Build the separate Turbo v1.1 8-step graph")
    main(turbo=parser.parse_args().turbo)
