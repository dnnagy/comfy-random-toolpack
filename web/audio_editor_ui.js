import {clone, clamp, initialState, validState, splitAt, setRange, keptRanges, keptSamples, detectSilence} from './audio_editor_core.js';

const cssURL = new URL('./audio_editor.css', import.meta.url).href;
const color = i => `hsl(${(i * 63 + 157) % 360} 42% 43%)`;
const stamp = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toFixed(3).padStart(6, '0')}`;
function stylesheet(root) { const link = document.createElement('link'); link.rel = 'stylesheet'; link.href = cssURL; root.append(link); }
function element(tag, text, className) { const e = document.createElement(tag); if (text) e.textContent = text; if (className) e.className = className; return e; }

export class AudioEditor {
    constructor(host, hooks) {
        this.host = host; this.hooks = hooks; this.info = null; this.state = null;
        this.cursor = 0; this.selection = null; this.selected = -1; this.zoom = 1; this.viewStart = 0;
        this.undoStack = []; this.redoStack = []; this.sources = []; this.playing = null; this.revision = 0;
        this.shadow = host.attachShadow({mode: 'open'}); stylesheet(this.shadow);
        this.frame = element('div', '', 'editor'); this.frame.tabIndex = 0; this.frame.setAttribute('aria-label', 'Audio waveform editor');
        this.shadow.append(this.frame);
        this.build();
        this.resize = new ResizeObserver(() => this.draw()); this.resize.observe(this.frame);
        this.frame.addEventListener('keydown', e => this.key(e));
        for (const event of ['pointerdown','pointerup','click','dblclick','wheel']) this.frame.addEventListener(event, e => e.stopPropagation());
    }
    button(parent, text, callback, className = '') {
        const b = element('button', text, className); b.type = 'button'; b.addEventListener('click', () => this.safe(callback)); parent.append(b); return b;
    }
    safe(callback) { try { Promise.resolve(callback()).catch(e => this.status(e.message || String(e), true)); } catch(e) { this.status(e.message || String(e), true); } }
    number(parent, title, value, step, min, max) {
        const label = element('label', title); const input = document.createElement('input');
        input.type = 'number'; input.value = value; input.step = step; input.min = min; input.max = max;
        input.setAttribute('aria-label', title); label.append(input); parent.append(label); return input;
    }
    build() {
        const header = element('div', '', 'header'); header.append(element('span', 'CRTP / AUDIO CUT', 'brand'));
        this.summary = element('span', 'Prepare audio, then make your cuts.', 'summary'); header.append(this.summary);
        this.expand = this.button(header, 'Expand ↗', () => this.toggleModal()); this.frame.append(header);
        const toolbar = element('div', '', 'toolbar');
        this.load = this.button(toolbar, 'Load waveform', async () => {
            this.load.disabled = true; this.stop(); this.status('Preparing waveform — only this node and its inputs are queued.');
            try { await this.hooks.load(); } finally { this.load.disabled = false; }
        }, 'primary');
        this.button(toolbar, 'Mark silence…', () => { this.silence.hidden = !this.silence.hidden; this.draw(); });
        toolbar.append(element('span', '', 'spacer'));
        this.undoButton = this.button(toolbar, 'Undo', () => this.undo());
        this.redoButton = this.button(toolbar, 'Redo', () => this.redo());
        this.button(toolbar, 'Keep all', () => this.change(() => { this.state.segments = initialState(this.info).segments; this.selected = -1; this.selection = null; }));
        this.frame.append(toolbar);
        this.silence = element('div', '', 'silence'); this.silence.hidden = true;
        this.threshold = this.number(this.silence, 'Below dB', -38, 1, -100, 0);
        this.minimum = this.number(this.silence, 'Min gap s', .5, .05, .05, 30);
        this.padding = this.number(this.silence, 'Keep edge s', .08, .01, 0, 2);
        this.button(this.silence, 'Mark gaps as cuts', () => this.markSilence(), 'cut');
        this.frame.append(this.silence);
        const timeline = element('div', '', 'timeline'); this.canvas = document.createElement('canvas'); this.canvas.tabIndex = 0;
        this.canvas.setAttribute('aria-label', 'Waveform timeline. Drag to select a range; click to seek.'); timeline.append(this.canvas);
        this.empty = element('div', '', 'empty'); this.empty.append(element('strong', 'Make room for the words.'), element('span', 'Connect AUDIO → Load waveform → select and disable gaps.')); timeline.append(this.empty); this.frame.append(timeline);
        const navigation = element('div', '', 'navigation'); navigation.append(element('span', 'View'));
        this.pan = document.createElement('input'); this.pan.type = 'range'; this.pan.min = 0; this.pan.max = 10000; this.pan.value = 0; this.pan.className = 'pan'; this.pan.setAttribute('aria-label', 'Scroll timeline');
        this.pan.addEventListener('input', () => { if (this.info) { this.viewStart = Number(this.pan.value) / 10000 * (this.info.samples - this.span()); this.draw(); } }); navigation.append(this.pan);
        navigation.append(element('span', 'Zoom'));
        this.zoomSlider = document.createElement('input'); this.zoomSlider.type = 'range'; this.zoomSlider.min = 1; this.zoomSlider.max = 40; this.zoomSlider.step = .25; this.zoomSlider.value = 1; this.zoomSlider.setAttribute('aria-label', 'Zoom waveform');
        this.zoomSlider.addEventListener('input', () => this.setZoom(Number(this.zoomSlider.value))); navigation.append(this.zoomSlider);
        this.button(navigation, 'Fit', () => { this.zoom = 1; this.viewStart = 0; this.zoomSlider.value = 1; this.draw(); }); this.frame.append(navigation);
        const transport = element('div', '', 'transport');
        this.button(transport, '↤', () => this.seek(0)).title = 'Go to start';
        this.playSource = this.button(transport, '▶ Source', () => this.play('source'));
        this.playKept = this.button(transport, '▶ Kept', () => this.play('kept'), 'primary');
        this.button(transport, '■ Stop', () => this.stop());
        this.time = element('span', '00:00.000', 'time'); transport.append(this.time, element('span', '', 'spacer'));
        const volumeLabel = element('label', 'Volume'); this.volume = document.createElement('input'); this.volume.type = 'range'; this.volume.min = 0; this.volume.max = 1; this.volume.step = .05; this.volume.value = .8; this.volume.setAttribute('aria-label','Playback volume');
        this.volume.addEventListener('input', () => { if (this.gain) this.gain.gain.value = Number(this.volume.value); }); volumeLabel.append(this.volume); transport.append(volumeLabel); this.frame.append(transport);
        const selection = element('div', '', 'selection');
        this.startInput = this.number(selection, 'In s', 0, .001, 0, 86400); this.endInput = this.number(selection, 'Out s', 0, .001, 0, 86400);
        this.button(selection, 'Select', () => this.selectNumbers());
        this.button(selection, 'Split · S', () => this.change(() => splitAt(this.state, this.cursor)));
        this.button(selection, 'Disable', () => this.setEnabled(false), 'cut');
        this.button(selection, 'Enable', () => this.setEnabled(true), 'keep');
        this.button(selection, '▶ Selection', () => this.play('selection'));
        this.frame.append(selection);
        this.segmentList = element('div', '', 'segments'); this.segmentList.setAttribute('aria-label', 'Audio segments'); this.frame.append(this.segmentList);
        this.statusLine = element('div', 'Drag a range to select it. Split at the playhead, or drag segment boundaries. Space: play/stop.', 'status'); this.statusLine.setAttribute('role', 'status'); this.frame.append(this.statusLine);
        this.canvas.addEventListener('pointerdown', e => this.pointerDown(e));
        this.canvas.addEventListener('pointermove', e => this.pointerMove(e));
        this.canvas.addEventListener('pointerup', e => this.pointerUp(e));
        this.canvas.addEventListener('pointercancel', () => { if (this.drag?.before) this.state = this.drag.before; this.drag = null; this.refresh(); });
        this.canvas.addEventListener('wheel', e => { if (!this.info) return; e.preventDefault(); if (e.ctrlKey || e.metaKey) this.setZoom(this.zoom * (e.deltaY < 0 ? 1.2 : .8)); else { this.viewStart = clamp(this.viewStart + (e.deltaX || e.deltaY) * this.span() / 1000, 0, this.info.samples - this.span()); this.draw(); } }, {passive: false});
        this.refresh();
    }
    status(message, error = false) { this.statusLine.textContent = message; this.statusLine.classList.toggle('error', error); }
    setInfo(info, saved = this.hooks.saved()) {
        if (!info?.source_id || !info.samples || !info.sample_rate) return;
        const changed = this.info?.source_id !== info.source_id;
        if (changed) { this.stop(); this.buffer = null; this.undoStack = []; this.redoStack = []; this.cursor = 0; this.selection = null; this.selected = -1; this.zoom = 1; this.viewStart = 0; this.zoomSlider.value = 1; }
        this.info = info;
        let parsed; try { parsed = typeof saved === 'string' ? JSON.parse(saved) : saved; } catch {}
        const reset = !validState(parsed, info);
        this.state = reset ? initialState(info) : clone(parsed);
        this.hooks.save(this.state, this.info); this.refresh();
        this.status(reset && saved && saved !== '{}' ? 'Source changed or cuts were invalid; starting with all audio enabled. Undo history cleared.' : 'Waveform ready. Enabled segments are joined on the next workflow run.');
    }
    change(fn) {
        if (!this.state) { this.status('Connect AUDIO and click Load waveform first.'); return; }
        this.stop(); const before = clone(this.state);
        try { fn(); } catch (e) { this.state = before; this.refresh(); throw e; }
        if (JSON.stringify(before) !== JSON.stringify(this.state)) { this.undoStack.push(before); if (this.undoStack.length > 100) this.undoStack.shift(); this.redoStack = []; this.hooks.save(this.state, this.info); }
        this.refresh();
    }
    undo() { if (!this.undoStack.length) return; this.stop(); this.redoStack.push(clone(this.state)); this.state = this.undoStack.pop(); this.selected = -1; this.selection = null; this.hooks.save(this.state, this.info); this.refresh(); }
    redo() { if (!this.redoStack.length) return; this.stop(); this.undoStack.push(clone(this.state)); this.state = this.redoStack.pop(); this.selected = -1; this.selection = null; this.hooks.save(this.state, this.info); this.refresh(); }
    selectNumbers() {
        if (!this.info) return;
        const a = Number(this.startInput.value), b = Number(this.endInput.value);
        if (!Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b <= a || b > this.info.duration + .001) throw new Error('Choose an In time before Out, within the audio duration.');
        this.selection = [Math.round(a * this.info.sample_rate), Math.min(this.info.samples, Math.round(b * this.info.sample_rate))]; this.selected = -1; this.seek(this.selection[0]); this.refresh();
    }
    setEnabled(enabled) {
        this.change(() => {
            if (this.selection) setRange(this.state, ...this.selection, enabled);
            else if (this.selected >= 0 && this.state.segments[this.selected]) this.state.segments[this.selected].enabled = enabled;
            else throw new Error('Drag a range or select a segment first.');
        });
    }
    markSilence() {
        if (!this.info) throw new Error('Load a waveform first.');
        const threshold = Number(this.threshold.value), min = Number(this.minimum.value), pad = Number(this.padding.value);
        if (![threshold, min, pad].every(Number.isFinite) || threshold < -100 || threshold > 0 || min < .05 || pad < 0 || pad > 2) throw new Error('Check silence threshold, minimum gap, and edge padding.');
        const ranges = detectSilence(this.info, threshold, min, pad);
        this.change(() => { for (const r of ranges) setRange(this.state, r.start, r.end, false); this.selection = null; this.selected = -1; });
        this.status(`${ranges.length} quiet gaps marked as cuts. Review the shaded sections; Undo restores the previous cuts.`);
    }
    refresh() {
        this.empty.hidden = !!this.info; this.undoButton.disabled = !this.undoStack.length; this.redoButton.disabled = !this.redoStack.length;
        this.playSource.disabled = !this.info; this.playKept.disabled = !this.state || keptSamples(this.state) === 0;
        if (!this.info) return;
        const rate = this.info.sample_rate, kept = keptSamples(this.state) / rate;
        this.summary.textContent = `${this.info.duration.toFixed(2)}s source → ${kept.toFixed(2)}s kept · ${(this.info.duration - kept).toFixed(2)}s removed · ${this.info.channels} ch`;
        const range = this.selection || (this.state.segments[this.selected] ? [this.state.segments[this.selected].start, this.state.segments[this.selected].end] : [this.cursor, this.cursor]);
        if (!this.drag || this.drag.kind !== 'boundary') { this.startInput.value = (range[0] / rate).toFixed(3); this.endInput.value = (range[1] / rate).toFixed(3); }
        this.segmentList.replaceChildren();
        this.state.segments.forEach((s, i) => {
            const row = element('div', '', 'segment'); row.style.setProperty('--color', color(i)); row.classList.toggle('active', i === this.selected); row.classList.toggle('disabled', !s.enabled);
            const toggle = document.createElement('input'); toggle.type = 'checkbox'; toggle.checked = s.enabled; toggle.setAttribute('aria-label', `Keep segment ${i + 1}`);
            toggle.addEventListener('click', e => e.stopPropagation()); toggle.addEventListener('change', () => this.safe(() => this.change(() => { this.state.segments[i].enabled = toggle.checked; })));
            const label = element('span'); label.append(element('b', `${String(i + 1).padStart(2,'0')}  ${s.enabled ? 'KEEP' : 'CUT'}`), element('small', `${(s.start / rate).toFixed(2)}–${(s.end / rate).toFixed(2)}s`));
            row.append(toggle, label); row.addEventListener('click', () => { this.selected = i; this.selection = null; this.seek(s.start); this.refresh(); }); this.segmentList.append(row);
        });
        this.draw();
    }
    span() { return this.info ? this.info.samples / this.zoom : 1; }
    setZoom(value) { if (!this.info) return; const middle = this.viewStart + this.span() / 2; this.zoom = clamp(value, 1, 40); this.zoomSlider.value = this.zoom; this.viewStart = clamp(middle - this.span() / 2, 0, this.info.samples - this.span()); this.draw(); }
    seek(sample) { this.stop(); this.cursor = clamp(Math.round(sample), 0, this.info?.samples || 0); this.draw(); }
    eventSample(e) { const r = this.canvas.getBoundingClientRect(); return Math.round(clamp(this.viewStart + (e.clientX - r.left) / r.width * this.span(), 0, this.info.samples)); }
    pointerDown(e) {
        if (!this.info || e.button !== 0) return; e.preventDefault(); this.frame.focus(); this.stop(); this.canvas.setPointerCapture(e.pointerId);
        const rect = this.canvas.getBoundingClientRect(), y = (e.clientY - rect.top) / rect.height * this.canvas.clientHeight;
        const sample = this.eventSample(e), tolerance = this.span() * 7 / rect.width;
        const boundary = y > this.canvas.clientHeight - 47 ? this.state.segments.findIndex((s, i) => i > 0 && Math.abs(s.start - sample) < tolerance) : -1;
        if (boundary > 0) { this.drag = {kind: 'boundary', index: boundary, before: clone(this.state)}; return; }
        if (y > this.canvas.clientHeight - 47) { this.selected = this.state.segments.findIndex(s => s.start <= sample && sample < s.end); this.selection = null; this.cursor = sample; this.refresh(); return; }
        this.drag = {kind: 'selection', start: sample, x: e.clientX}; this.selection = null; this.selected = -1; this.cursor = sample; this.draw();
    }
    pointerMove(e) {
        if (!this.drag) return;
        const sample = this.eventSample(e);
        if (this.drag.kind === 'boundary') {
            const i = this.drag.index, a = this.state.segments[i - 1], b = this.state.segments[i];
            const edge = clamp(sample, a.start + 1, b.end - 1); a.end = edge; b.start = edge; this.selected = i; this.selection = null; this.draw();
        } else if (Math.abs(e.clientX - this.drag.x) > 3) { this.selection = [Math.min(this.drag.start, sample), Math.max(this.drag.start, sample)]; this.draw(); }
    }
    pointerUp(e) {
        if (!this.drag) return;
        const drag = this.drag; this.drag = null;
        if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
        if (drag.kind === 'boundary' && JSON.stringify(drag.before) !== JSON.stringify(this.state)) { this.undoStack.push(drag.before); this.redoStack = []; this.hooks.save(this.state, this.info); }
        this.refresh();
    }
    draw() {
        const canvas = this.canvas; if (!canvas) return;
        const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return;
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
        const c = canvas.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
        if (!this.info) return;
        const n = this.info.samples, span = this.span(), left = this.viewStart, rate = this.info.sample_rate;
        const x = sample => (sample - left) / span * w, bar = h - 44, center = (bar + 27) / 2, amp = Math.max(8, (bar - 35) / 2);
        const maximum = Math.max(.02, ...this.info.peaks_high.map(Math.abs), ...this.info.peaks_low.map(Math.abs));
        for (let i = 0; i < this.state.segments.length; i++) {
            const s = this.state.segments[i], a = clamp(x(s.start), 0, w), b = clamp(x(s.end), 0, w); if (b <= a) continue;
            c.globalAlpha = s.enabled ? .09 : .28; c.fillStyle = s.enabled ? color(i) : '#e2757c'; c.fillRect(a, 26, b-a, bar-26);
            c.globalAlpha = s.enabled ? .8 : .28; c.fillStyle = s.enabled ? color(i) : '#6b434d'; c.fillRect(a+1, bar+3, Math.max(0,b-a-2), 35); c.globalAlpha = 1;
            if (!s.enabled) { c.save(); c.beginPath(); c.rect(a,26,b-a,h-26); c.clip(); c.strokeStyle = '#d7868344'; for (let k=a-h; k<b; k+=13) { c.beginPath(); c.moveTo(k,h); c.lineTo(k+h,26); c.stroke(); } c.restore(); }
            if (i === this.selected) { c.strokeStyle = '#e7f7ff'; c.lineWidth=2; c.strokeRect(a+1,bar+3,b-a-2,35); c.lineWidth=1; }
            if (b-a > 36) { c.fillStyle='#fff'; c.font='11px system-ui'; c.fillText(`${i+1} ${s.enabled ? '✓' : '×'}`,a+7,bar+25); }
            if (i) { c.strokeStyle='#afbdd373'; c.beginPath(); c.moveTo(x(s.start),bar+4); c.lineTo(x(s.start),h-5); c.stroke(); }
        }
        c.strokeStyle = '#ccebe7'; c.beginPath();
        for (let px=0; px<w; px++) {
            const a=Math.max(0,Math.floor((left+px/w*span)/this.info.peak_stride));
            const b=Math.min(this.info.peaks_low.length, Math.max(a+1,Math.ceil((left+(px+1)/w*span)/this.info.peak_stride)));
            let low=0,high=0; for (let i=a;i<b;i++) {low=Math.min(low,this.info.peaks_low[i]);high=Math.max(high,this.info.peaks_high[i]);}
            c.moveTo(px,center-high/maximum*amp); c.lineTo(px,center-low/maximum*amp);
        } c.stroke();
        const seconds=span/rate, target=seconds/(w/100), magnitude=10**Math.floor(Math.log10(target));
        const tick=[1,2,5,10].find(v=>v*magnitude>=target)*magnitude;
        c.font='10px system-ui'; c.fillStyle='#92a4bc'; c.strokeStyle='#8094ae30';
        for(let t=Math.ceil(left/rate/tick)*tick;t<=(left+span)/rate;t+=tick) {const px=x(t*rate);c.fillText(t.toFixed(tick<1?2:1)+'s',px+4,16);c.beginPath();c.moveTo(px,22);c.lineTo(px,bar);c.stroke();}
        if(this.selection) {const a=x(this.selection[0]),b=x(this.selection[1]);c.fillStyle='#65e4dc30';c.fillRect(a,26,b-a,bar-26);c.strokeStyle='#7cf5e6';c.strokeRect(a,26,b-a,bar-26);}
        const cursorX=x(this.cursor); c.strokeStyle='#fcce80';c.lineWidth=1.5;c.beginPath();c.moveTo(cursorX,22);c.lineTo(cursorX,h);c.stroke();c.lineWidth=1;
        this.time.textContent=stamp(this.cursor/rate);
        this.pan.value=n>span?Math.round(left/(n-span)*10000):0; this.pan.disabled=this.zoom===1;
    }
    async audioBuffer() {
        if (!this.context) { this.context=new AudioContext(); this.gain=this.context.createGain();this.gain.gain.value=Number(this.volume.value);this.gain.connect(this.context.destination); }
        await this.context.resume();
        if (this.buffer) return this.buffer;
        const identity=this.info.source_id;
        const response=await fetch(this.hooks.url(this.info.file));
        if(!response.ok)throw new Error('Preview expired. Click Load waveform to recreate it.');
        const buffer=await this.context.decodeAudioData(await response.arrayBuffer());
        if(this.info?.source_id===identity)this.buffer=buffer;
        return buffer;
    }
    async play(mode) {
        if(!this.info)return;
        this.stop(); const revision=this.revision; const identity=this.info.source_id;
        const buffer=await this.audioBuffer();if(revision!==this.revision || this.info.source_id!==identity)return;
        let ranges;
        if(mode==='kept')ranges=keptRanges(this.state);
        else if(mode==='selection') {const s=this.selection||(this.state.segments[this.selected]?[this.state.segments[this.selected].start,this.state.segments[this.selected].end]:null);if(!s)throw new Error('Select a range or segment first.');ranges=[{start:s[0],end:s[1]}];}
        else ranges=[{start:this.cursor>=this.info.samples?0:this.cursor,end:this.info.samples}];
        if(!ranges.length)throw new Error('Enable at least one segment.');
        let offset=0;const startTime=this.context.currentTime+.04,rate=this.info.sample_rate;
        const scheduled=[];
        for(const r of ranges) {const duration=(r.end-r.start)/rate;if(duration<=0)continue;const source=this.context.createBufferSource();source.buffer=buffer;source.connect(this.gain);source.start(startTime+offset,r.start/rate,duration);this.sources.push(source);scheduled.push({...r,offset,duration});offset+=duration;}
        this.playing={startTime,ranges:scheduled,duration:offset};this.status(mode==='kept'?'Playing enabled sections only. Cuts are skipped.':'Playing source audio.');
        const tick=()=>{if(!this.playing)return;const elapsed=this.context.currentTime-startTime;
            if(elapsed>=offset){this.cursor=scheduled.at(-1)?.end||0;this.stop();this.draw();return;}
            const r=scheduled.find(r=>elapsed<r.offset+r.duration)||scheduled.at(-1);if(r)this.cursor=r.start+Math.max(0,elapsed-r.offset)*rate;
            if(this.cursor<this.viewStart||this.cursor>this.viewStart+this.span())this.viewStart=clamp(this.cursor-this.span()*.2,0,this.info.samples-this.span());this.draw();this.animation=requestAnimationFrame(tick);};tick();
    }
    stop() {this.revision++;for(const s of this.sources){try{s.stop();}catch{}s.disconnect();}this.sources=[];this.playing=null;cancelAnimationFrame(this.animation);}
    key(e) {
        const input=['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName);
        if(e.key==='Escape'&&this.modal){e.preventDefault();this.toggleModal();return;}
        if(input){e.stopPropagation();return;}
        const action=()=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.shiftKey?this.redo():this.undo();return true;}
            if(e.ctrlKey||e.metaKey)return false;
            if(e.code==='Space'){this.playing?this.stop():this.safe(()=>this.play('kept'));return true;}
            if(e.key.toLowerCase()==='s'){this.change(()=>splitAt(this.state,this.cursor));return true;}
            if(e.key==='Delete'||e.key==='Backspace'){this.setEnabled(false);return true;}
            if(e.key.toLowerCase()==='e'){this.setEnabled(true);return true;}
            if((e.key==='ArrowLeft'||e.key==='ArrowRight')&&this.info){this.seek(this.cursor+(e.key==='ArrowRight'?1:-1)*this.info.sample_rate*(e.shiftKey?1:.1));return true;}return false;};
        try{if(action()){e.preventDefault();e.stopPropagation();}}catch(error){e.preventDefault();e.stopPropagation();this.status(error.message,true);}
    }
    toggleModal() {
        if(this.modal){this.shadow.append(this.frame);this.modal.remove();this.modal=null;this.expand.textContent='Expand ↗';this.frame.focus();this.draw();return;}
        const modal=document.createElement('div');modal.style.cssText='position:fixed;inset:0;z-index:100000;';const shadow=modal.attachShadow({mode:'open'});stylesheet(shadow);
        const overlay=element('div','','modal');overlay.setAttribute('role','dialog');overlay.setAttribute('aria-label','Audio waveform editor');overlay.setAttribute('aria-modal','true');shadow.append(overlay);overlay.append(this.frame);
        overlay.addEventListener('click',e=>{if(e.target===overlay)this.toggleModal();});
        this.frame.addEventListener('keydown',this.trapFocus ||= e=>{if(!this.modal||e.key!=='Tab')return;const items=[...this.frame.querySelectorAll('button,input,canvas')].filter(el=>!el.disabled&&el.getClientRects().length);if(!items.length)return;const first=items[0],last=items.at(-1);if(e.shiftKey&&e.target===first){last.focus();e.preventDefault();}else if(!e.shiftKey&&e.target===last){first.focus();e.preventDefault();}});
        document.body.append(modal);this.modal=modal;this.expand.textContent='Close ↙';this.frame.focus();this.draw();
    }
    dispose(){this.stop();this.resize.disconnect();this.modal?.remove();this.context?.close();this.host.remove();}
}
