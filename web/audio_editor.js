import {app} from '../../scripts/app.js';
import {api} from '../../scripts/api.js';
import {AudioEditor} from './audio_editor_ui.js';
import {upstreamPrompt} from './audio_editor_core.js';

function hide(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    if (widget.inputEl) widget.inputEl.style.display = 'none';
}
app.registerExtension({
    name: 'CRTP.AudioWaveformEditor',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'CRTP_AudioWaveformEditor') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            const stateWidget = this.widgets.find(w => w.name === 'segments_json');
            const previewWidget = this.widgets.find(w => w.name === 'preview_only');
            hide(stateWidget); hide(previewWidget);
            // Preview is an execution override, never persisted into the workflow.
            previewWidget.value = false;
            previewWidget.serializeValue = () => false;
            const host = document.createElement('div');
            host.style.cssText = 'width:100%;height:490px;';
            const editor = new AudioEditor(host, {
                saved: () => stateWidget.value,
                save: (state, info) => {
                    const value = JSON.stringify(state);
                    const changed = stateWidget.value !== value;
                    stateWidget.value = value;
                    this.properties ||= {};
                    this.properties.crtp_audio_editor_preview = info;
                    if (changed) this.graph?.change();
                    this.setDirtyCanvas(true, true);
                },
                url: file => api.apiURL('/view?' + new URLSearchParams(file)),
                load: async () => {
                    const prompt = await app.graphToPrompt();
                    const id = String(this.id);
                    const output = upstreamPrompt(prompt.output, id);
                    output[id].inputs.preview_only = true;
                    output[id].inputs.segments_json = '';
                    try { await api.queuePrompt(-1, {output, workflow: prompt.workflow}); }
                    catch (error) { throw new Error(error?.response?.error?.message || error.message || 'Could not queue waveform. Check the AUDIO connection.'); }
                },
            });
            this.crtpAudioEditor = editor;
            const widget = this.addDOMWidget('audio_waveform', 'crtp_audio_waveform', host, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 490, getMaxHeight: () => 490, getHeight: () => 490,
            });
            widget.computeSize = () => [760, 500];
            this.size = [Math.max(this.size[0], 860), Math.max(this.computeSize()[1], 590)];
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            executed?.apply(this, arguments);
            if (message?.audio_editor?.[0]) this.crtpAudioEditor?.setInfo(message.audio_editor[0]);
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            configured?.apply(this, arguments);
            this.widgets.find(w => w.name === 'preview_only').value = false;
            const info = this.properties?.crtp_audio_editor_preview;
            if (info) this.crtpAudioEditor?.setInfo(info);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            this.crtpAudioEditor?.dispose();
            return removed?.apply(this, arguments);
        };
    },
});
