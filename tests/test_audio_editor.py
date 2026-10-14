"""Sample accuracy, state safety, and waveform preview contracts."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
import soundfile as sf

spec = importlib.util.spec_from_file_location('audio_editor', Path(__file__).parents[1] / 'nodes/audio_editor.py')
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)

class AudioEditorTests(unittest.TestCase):
    def setUp(self):
        self.wave = torch.stack((torch.arange(1000) / 1000, -torch.arange(1000) / 1000))[None]
        self.audio = {'waveform': self.wave, 'sample_rate': 1000}
        self.identity = node.source_id(self.wave, 1000)
        self.segments = [{'start': 0, 'end': 200, 'enabled': True}, {'start': 200, 'end': 700, 'enabled': False}, {'start': 700, 'end': 1000, 'enabled': True}]
        self.state = dict(version=1, source_id=self.identity, samples=1000, segments=self.segments)

    def test_load_waveform_bypasses_cache_to_recreate_expired_preview(self):
        self.assertTrue(math.isnan(node.CRTP_AudioWaveformEditor.IS_CHANGED(preview_only=True)))
        self.assertEqual(node.CRTP_AudioWaveformEditor.IS_CHANGED(), False)

    def test_cut_preserves_channels_samples_and_source(self):
        before = self.wave.clone()
        edited, mapping = node.render_audio(self.wave, 1000, self.segments, 0)
        self.assertTrue(torch.equal(edited, torch.cat((self.wave[..., :200], self.wave[..., 700:]), -1)))
        self.assertTrue(torch.equal(self.wave, before))
        self.assertEqual(mapping[-1], dict(source_start=700, source_end=1000, output_start=200, output_end=500))

    def test_visual_splits_do_not_fade_or_change_audio(self):
        for s in self.segments: s['enabled'] = True
        edited, mapping = node.render_audio(self.wave, 1000, self.segments, 20)
        self.assertTrue(torch.equal(edited, self.wave))
        self.assertEqual(len(mapping), 1)

    def test_fades_only_at_cut_edges(self):
        edited, _ = node.render_audio(self.wave, 1000, self.segments, 3)
        self.assertTrue(torch.equal(edited[..., :197], self.wave[..., :197]))
        self.assertEqual(edited[0, 0, 199].item(), 0)
        self.assertEqual(edited[0, 0, 200].item(), 0)
        self.assertTrue(torch.equal(edited[..., -20:], self.wave[..., -20:]))

    def test_invalid_states_rejected(self):
        self.assertEqual(node.parse_edits(json.dumps(self.state), self.identity, 1000), self.segments)
        with self.assertRaisesRegex(ValueError, 'changed'): node.parse_edits(json.dumps(self.state), 'different', 1000)
        for broken in [dict(start=201,end=700,enabled=False), dict(start=200,end=700,enabled=0), dict(start=200,end=1001,enabled=False)]:
            self.state['segments'][1] = broken
            with self.assertRaises(ValueError): node.parse_edits(json.dumps(self.state), self.identity, 1000)

    def test_empty_output_and_invalid_audio_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Every segment'): node.render_audio(self.wave, 1000, [dict(start=0,end=1000,enabled=False)], 0)
        for wave in [torch.zeros(2, 1, 20), torch.zeros(1, 1, 0), torch.full((1,1,10),float('nan'))]:
            with self.assertRaises(ValueError): node.validate_audio(dict(waveform=wave,sample_rate=1000))

    def test_preview_and_execution_contract(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'folder_paths': types.SimpleNamespace(get_temp_directory=lambda: directory)}):
            result = node.CRTP_AudioWaveformEditor().edit(self.audio, json.dumps(self.state), 0)
            info = result['ui']['audio_editor'][0]
            self.assertEqual(result['result'][1], .5)
            self.assertEqual(info['samples'], 1000)
            self.assertGreater(max(info['peaks_high']), .9)  # Antiphase stereo cannot cancel the display.
            wave, rate = sf.read(Path(directory) / info['file']['subfolder'] / info['file']['filename'])
            self.assertEqual(wave.shape, (1000,2))
            self.assertEqual(rate, 1000)
            preview = node.CRTP_AudioWaveformEditor().edit(self.audio, 'bad state', 0, True)
            self.assertEqual(preview['result'][0]['waveform'].shape[-1], 1000)
            self.assertEqual(len(list(Path(directory).rglob('*.wav'))), 1)

    def test_envelopes_are_bounded_and_tail_rms_is_correct(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'folder_paths': types.SimpleNamespace(get_temp_directory=lambda: directory)}):
            wave = torch.ones(1, 1, 100003) * .1
            info = node.preview_data(wave, 1000, node.source_id(wave, 1000))
            self.assertLessEqual(len(info['peaks_high']),8192)
            self.assertLessEqual(len(info['rms_db']),30000)
            self.assertAlmostEqual(info['rms_db'][-1],-20,places=1)

if __name__ == '__main__': unittest.main()
