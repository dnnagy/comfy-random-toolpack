"""Rebuild the isolated H3 test graphs (UI and API); no product templates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "examples/h3_state"
VAE = "minimax_h3_video_vae_fp16.safetensors"


class Graph:
    def __init__(self):
        self.nodes = []; self.links = []; self.api = {}; self.sockets = {}
    def node(self, id, cls, xy, inputs, outputs, widgets=(), title=None, mode=0, size=None):
        n = {"id": id, "type": cls, "pos": xy, "size": size or [340, max(100, 55+len(inputs)*22+len(widgets)*28)],
             "flags": {}, "order": len(self.nodes), "mode": mode, "inputs": [],
             "outputs": [{"name": name, "type": kind, "links": []} for name, kind in outputs],
             "properties": {"Node name for S&R": cls}, "widgets_values": [value for _, value in widgets]}
        if title: n["title"] = title
        for name, kind in inputs:
            n["inputs"].append({"name": name, "type": kind, "link": None})
        self.nodes.append(n); self.sockets[id] = n
        if cls != "Note" and mode == 0:
            self.api[str(id)] = {"class_type": cls, "inputs": dict(widgets), "_meta": {"title": title or cls}}
            # Frontend-only widgets do not belong in API input dictionaries.
            self.api[str(id)]["inputs"].pop("control_after_generate", None)
            self.api[str(id)]["inputs"].pop("upload", None)
        return id
    def link(self, source, output, target, name, widget=False):
        a, b = self.sockets[source], self.sockets[target]
        slot = next((i for i, v in enumerate(b["inputs"]) if v["name"] == name), None)
        kind = a["outputs"][output]["type"]
        if slot is None:
            slot = len(b["inputs"])
            entry = {"name": name, "type": kind, "link": None}
            if widget: entry["widget"] = {"name": name}
            b["inputs"].append(entry)
        link = len(self.links)+1
        self.links.append([link, source, output, target, slot, kind])
        a["outputs"][output]["links"].append(link); b["inputs"][slot]["link"] = link
        if str(target) in self.api:
            self.api[str(target)]["inputs"][name] = [str(source), output]
    def save(self, name):
        ROOT.mkdir(parents=True, exist_ok=True)
        ui = {"last_node_id": max(self.sockets), "last_link_id": len(self.links), "nodes": self.nodes,
              "links": self.links, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.65, "offset": [0,0]}}, "version": 0.4}
        (ROOT / (name + ".json")).write_text(json.dumps(ui, indent=2)+"\n")
        (ROOT / (name + "_api.json")).write_text(json.dumps(self.api, indent=2)+"\n")


def build(continuation, initial=False):
    g = Graph()
    g.node(1,"UNETLoader",[30,600],[],[("MODEL","MODEL")],[("unet_name","minimax_h3_ref2va_pruned_int8_convrot.safetensors"),("weight_dtype","default")])
    g.node(2,"CLIPLoader",[30,760],[],[("CLIP","CLIP")],[("clip_name","qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),("type","minimax"),("device","default")])
    g.node(3,"VAELoader",[30,960],[],[("VAE","VAE")],[("vae_name",VAE)])
    g.node(4,"VAELoader",[30,1100],[],[("VAE","VAE")],[("vae_name","minimax_h3_audio_vae_fp32.safetensors")])
    prompt = ("subject_definitions:\nThe person and room established by the latent guide.\n\n"
              "summary:\n" + ("Video continuation. " if continuation else "") +
              "A natural documentary shot.\n\nretention_analysis:\nPreserve the person's appearance, clothing, room and lighting.\n\n"
              "detailed_description:\n[Shot 1] The person looks toward the window and slowly smiles. "
              "The camera remains steady. Preserve natural skin texture and incoming motion. No cuts.\n\n"
              "overall_soundscape:\nQuiet room ambience.\n\nnon_diegetic_music:\nN/A")
    g.node(5,"MiniMaxH3ReferenceToVideo",[450,650],[("clip","CLIP"),("vae","VAE"),("audio_vae","VAE"),
           ("ref_images.ref_image_0","IMAGE"),("ref_images.ref_image_1","IMAGE"),("ref_audios.ref_audio_0","AUDIO")],
           [("positive","CONDITIONING"),("unused_latent","LATENT")],
           [("prompt",prompt),("width",1024),("height",576),("length",243),("ref_image_size","max")],
           "Prompt + optional references (guide defines actual canvas)",size=[470,650])
    for src, name in [(2,"clip"),(3,"vae"),(4,"audio_vae")]: g.link(src,0,5,name)
    if not initial:
        g.node(6,"CRTP_LazyLoadH3State",[30,30],[],[("state","CRTP_H3_STATE"),("info","STRING")],
               [("state_path","h3_test/REPLACE_WITH_"+("TAKE" if continuation else "KEYFRAME")+".safetensors")])
    source = 6
    if not continuation:
        g.node(20,"CRTP_LazyLoadImageUpload",[30,1350],[],[("image","IMAGE"),("mask","MASK"),("loaded","BOOLEAN")],[("image","first_frame.png"),("upload","image")],"Optional one-time bootstrap: upload a 32-aligned image")
        g.node(21,"VAEEncode",[450,1400],[("pixels","IMAGE"),("vae","VAE")],[("LATENT","LATENT")])
        if not initial:
            g.link(20,0,21,"pixels")
        g.link(3,0,21,"vae")
        g.node(22,"CRTP_H3StateFromLatent",[850,1350],[("latent","LATENT"),("plan","CRTP_H3_PLAN")],[("state","CRTP_H3_STATE")],
               [("kind","image_keyframe"),("origin","encoded_pixels"),("vae_id",VAE)],"Bootstrap OR connect a sampled Fizgig still latent here")
        g.link(21,0,22,"latent")
        if initial:
            source = 22
            g.sockets[20]["title"] = "Upload first-frame image (required)"
            g.sockets[22]["title"] = "Encoded opening frame — generation starts here"
            g.sockets[20].update(pos=[30,30], size=[340,360])
            g.sockets[21]["pos"] = [850,30]
            g.sockets[22]["pos"] = [850,210]
            for id in (20,22):
                g.api[str(id)]["_meta"]["title"] = g.sockets[id]["title"]
            g.node(26,"ImageScale",[450,30],[("image","IMAGE")],[("IMAGE","IMAGE")],
                   [("upscale_method","lanczos"),("width",1024),("height",576),("crop","center")],
                   "Output canvas — center-crops the first frame")
            for src, out, dst, name in [(20,0,26,"image"),(26,0,21,"pixels"),(26,0,5,"ref_images.ref_image_0")]:
                g.link(src,out,dst,name)
            g.api["5"]["inputs"]["prompt"] = prompt.replace("The person and room established by the latent guide.",
                "The person and room shown in <Picture 1>, which also supplies the opening frame.")
            g.sockets[5]["widgets_values"][0] = g.api["5"]["inputs"]["prompt"]
        else:
            g.node(23,"LazySwitchKJ",[450,30],[("on_false","CRTP_H3_STATE"),("on_true","CRTP_H3_STATE")],[("state","CRTP_H3_STATE")],
                   [("switch",True)],"TRUE: saved latent / FALSE: one-time image encode")
            g.link(22,0,23,"on_false"); g.link(6,0,23,"on_true")
            g.node(24,"CRTP_SaveH3State",[850,30],[("state","CRTP_H3_STATE")],[("state","CRTP_H3_STATE"),("state_path","STRING")],
                   [("filename_prefix","h3_test/keyframe")])
            g.link(23,0,24,"state"); source = 24
    g.node(7,"CRTP_H3SelectLatentContext",[1250,30],[("state","CRTP_H3_STATE")],[("context","CRTP_H3_CONTEXT"),("info","STRING")],
           [("mode","tail_22" if continuation else "image_keyframe"),("reuse_audio",False)])
    g.link(source,0,7,"state")
    g.node(8,"CRTP_H3PrepareLatentGeneration",[1250,330],[("positive","CONDITIONING"),("context","CRTP_H3_CONTEXT")],
           [("positive","CONDITIONING"),("fresh_latent","LATENT"),("plan","CRTP_H3_PLAN")],[("seconds",10.0),("vae_id",VAE)])
    g.link(5,0,8,"positive"); g.link(7,0,8,"context")
    g.node(9,"BasicGuider",[1700,300],[("model","MODEL"),("conditioning","CONDITIONING")],[("GUIDER","GUIDER")])
    g.link(1,0,9,"model"); g.link(8,0,9,"conditioning")
    g.node(10,"BasicScheduler",[1700,500],[("model","MODEL")],[("SIGMAS","SIGMAS")],[("scheduler","simple"),("steps",30),("denoise",1.0)])
    g.link(1,0,10,"model")
    g.node(11,"KSamplerSelect",[1700,710],[],[("SAMPLER","SAMPLER")],[("sampler_name","res_multistep")])
    g.node(12,"RandomNoise",[1700,30],[],[("NOISE","NOISE")],[("noise_seed",10104),("control_after_generate","fixed")])
    g.node(13,"SamplerCustomAdvanced",[2150,300],[("noise","NOISE"),("guider","GUIDER"),("sampler","SAMPLER"),("sigmas","SIGMAS"),("latent_image","LATENT")],
           [("output","LATENT"),("denoised_output","LATENT")])
    for src,name in [(12,"noise"),(9,"guider"),(11,"sampler"),(10,"sigmas")]: g.link(src,0,13,name)
    g.link(8,1,13,"latent_image")
    g.node(14,"CRTP_H3DecodeAndTrim",[2550,300],[("samples","LATENT"),("vae","VAE"),("audio_vae","VAE"),("plan","CRTP_H3_PLAN")],
           [("images","IMAGE"),("audio","AUDIO"),("raw_state","CRTP_H3_STATE")])
    g.link(13,0,14,"samples"); g.link(3,0,14,"vae"); g.link(4,0,14,"audio_vae"); g.link(8,2,14,"plan")
    g.node(15,"CRTP_H3SaveTake",[2950,300],[("state","CRTP_H3_STATE"),("images","IMAGE"),("audio","AUDIO")],
           [("state","CRTP_H3_STATE"),("state_path","STRING"),("video_path","STRING")],[("filename_prefix","h3_test/"+("continuation" if continuation else "root")),("crf",18)])
    g.link(14,2,15,"state"); g.link(14,0,15,"images"); g.link(14,1,15,"audio")
    if continuation:
        g.node(25,"CRTP_H3AssembleTakes",[2950,650],[],[("video_path","STRING")],
               [("state_paths",""),("filename_prefix","h3_test/assembled"),("allow_cuts",False),("crf",18)],
               "OPTIONAL EXPORT (muted): ordered sidecar paths, one per line",mode=2,size=[450,300])
    note = ("EXPERIMENTAL — no changes to product workflows.\n"
            "Base model, 30 steps, RES multistep, Simple. No Turbo.\n"
            "The guide is fixed conditioning. The sampler gets a NEW empty AV latent.\n"
            "Canvas comes from the state. Keep VAE identifiers consistent.\n"
            "Optional pictures/audio can be connected to Ref2VA. Number only references actually supplied; the latent guide is not a <Picture 1> token.\n"
            + ("Set the loader to a root/continuation sidecar. Select tail_5 or tail_22. Audio reuse is explicit.\n"
               "Save Take outputs the new clip + full raw latent sidecar. Continue using the sidecar.\n"
               "To export, unmute Assemble Takes and paste root + child sidecar paths in order. It encodes original clips once."
               if continuation else
               "Default TRUE loads a saved keyframe state without encoding pixels.\n"
               "To bootstrap: select FALSE, upload a 32-aligned image, then copy the saved keyframe state path into the loader and switch TRUE.\n"
               "For a generated H3 still: connect its sampler output to State from Latent, set origin=sampled and switch FALSE. Never use a sliced video token as a still.\n"
               "Save Take produces the root clip and state needed by the continuation test."))
    if initial:
        note = ("START HERE — generate the first video and save continuation state.\n"
                "1. Upload a first-frame image. Set width/height on Output canvas (multiples of 32). Center crop is explicit.\n"
                "2. Edit the prompt; <Picture 1> is the resized first frame. Extra references start at <Picture 2>.\n"
                "3. Set duration on Prepare Latent Generation and queue. Defaults: 10 seconds, base model, 30 steps, RES multistep + Simple.\n"
                "4. Save Take writes h3_test/root_<id>.mp4 AND root_<id>.safetensors, containing the final sampled video/audio latent.\n"
                "5. Paste the .safetensors path into Lazy Load H3 State in H3-State-Continuation-Test. Select tail_5 or tail_22.\n"
                "No previous latent file is required. The complete opening clip is kept. Existing workflows are unchanged.")
    g.node(99,"Note",[2150,950],[],[],[("text",note)],size=[750,340])
    g.save("H3-State-Initial-Video-Test" if initial else ("H3-State-Continuation-Test" if continuation else "H3-State-First-Frame-Test"))
    return g


if __name__ == "__main__":
    build(False); build(True); build(False, initial=True)
