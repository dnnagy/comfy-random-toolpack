import test from 'node:test';
import assert from 'node:assert/strict';
import {initialState,validState,splitAt,setRange,keptRanges,keptSamples,detectSilence,upstreamPrompt} from '../web/audio_editor_core.js';
const info={source_id:'a',samples:1000,sample_rate:100};
test('range cuts split boundaries, remain reversible and merge kept spans',()=>{
 const s=initialState(info);setRange(s,200,700,false);
 assert.equal(validState(s,info),true);assert.equal(keptSamples(s),500);
 assert.deepEqual(keptRanges(s),[{start:0,end:200},{start:700,end:1000}]);
 setRange(s,200,700,true);splitAt(s,800);assert.deepEqual(keptRanges(s),[{start:0,end:1000}]);
 assert.equal(validState(s,{...info,source_id:'b'}),false);
});
test('silence detection preserves edge padding and ignores short gaps',()=>{
 const fixture={...info,rms_stride:100,rms_db:[-100,-10,-10,-100,-100,-100,-10,-100,-10,-100]};
 assert.deepEqual(detectSilence(fixture,-38,1.5,.1),[{start:310,end:590}]);
 assert.deepEqual(detectSilence(fixture,-38,.5,.1),[{start:0,end:90},{start:310,end:590},{start:710,end:790},{start:910,end:1000}]);
});
test('preview queues upstream only and does not mutate original prompt',()=>{
 const output={'1':{inputs:{audio:['2',0]}},'2':{inputs:{path:'a.wav'}},'3':{inputs:{audio:['1',0]}}};
 const preview=upstreamPrompt(output,'1');assert.deepEqual(Object.keys(preview),['1','2']);
 preview['1'].inputs.preview_only=true;assert.equal(output['1'].inputs.preview_only,undefined);
 assert.throws(()=>upstreamPrompt(output,'4'),/unavailable/);
});
