// Sample-indexed edit operations, independent of ComfyUI and the DOM.
export const MAX_SEGMENTS = 2048;
export const clone = (value) => JSON.parse(JSON.stringify(value));
export const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
export function initialState(info) {
    return {version: 1, source_id: info.source_id, samples: info.samples,
        segments: [{start: 0, end: info.samples, enabled: true}]};
}
export function validState(state, info) {
    if (!state || state.version !== 1 || state.source_id !== info.source_id || state.samples !== info.samples ||
        !Array.isArray(state.segments) || !state.segments.length || state.segments.length > MAX_SEGMENTS) return false;
    let end = 0;
    for (const s of state.segments) {
        if (!Number.isSafeInteger(s.start) || !Number.isSafeInteger(s.end) || typeof s.enabled !== 'boolean' ||
            s.start !== end || s.end <= s.start || s.end > info.samples) return false;
        end = s.end;
    }
    return end === info.samples;
}
export function splitAt(state, sample) {
    sample = Math.round(sample);
    const i = state.segments.findIndex(s => s.start < sample && sample < s.end);
    if (i < 0) return false;
    if (state.segments.length >= MAX_SEGMENTS) throw new Error(`Maximum ${MAX_SEGMENTS} segments. Undo or reset some cuts.`);
    const s = state.segments[i];
    state.segments.splice(i, 1, {...s, end: sample}, {...s, start: sample});
    return true;
}
export function setRange(state, start, end, enabled) {
    [start, end] = [Math.round(Math.min(start, end)), Math.round(Math.max(start, end))];
    start = clamp(start, 0, state.samples); end = clamp(end, 0, state.samples);
    if (start >= end) return;
    splitAt(state, start); splitAt(state, end);
    for (const s of state.segments) if (s.start >= start && s.end <= end) s.enabled = enabled;
}
export function keptRanges(state) {
    const ranges = [];
    for (const s of state.segments) {
        if (!s.enabled) continue;
        const last = ranges.at(-1);
        if (last && last.end === s.start) last.end = s.end;
        else ranges.push({start: s.start, end: s.end});
    }
    return ranges;
}
export function keptSamples(state) { return keptRanges(state).reduce((n, s) => n + s.end - s.start, 0); }
export function detectSilence(info, thresholdDB, minimumSeconds, paddingSeconds) {
    const ranges = [];
    let start = null;
    const stride = info.rms_stride, rate = info.sample_rate;
    for (let i = 0; i <= info.rms_db.length; i++) {
        if (i < info.rms_db.length && info.rms_db[i] < thresholdDB) {
            if (start === null) start = i * stride;
        } else if (start !== null) {
            const end = Math.min(info.samples, i * stride);
            if ((end - start) / rate >= minimumSeconds) {
                const a = start === 0 ? 0 : start + Math.round(paddingSeconds * rate);
                const b = end === info.samples ? end : end - Math.round(paddingSeconds * rate);
                if (b > a) ranges.push({start: a, end: b});
            }
            start = null;
        }
    }
    return ranges;
}
// Follow actual input links only (not arbitrary two-item literal arrays).
export function upstreamPrompt(output, target) {
    const keep = new Set();
    function visit(id) {
        id = String(id);
        if (keep.has(id)) return;
        const node = output[id];
        if (!node) throw new Error('Editor node is unavailable. Enable it and reconnect its AUDIO input.');
        keep.add(id);
        for (const value of Object.values(node.inputs || {})) {
            if (Array.isArray(value) && value.length === 2 && typeof value[0] === 'string' &&
                Number.isInteger(value[1]) && output[value[0]]) visit(value[0]);
        }
    }
    visit(target);
    return Object.fromEntries([...keep].map(id => [id, clone(output[id])]));
}
