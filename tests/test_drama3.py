"""Audio contracts, isolated process cleanup, and CUDA HTTP integration without GPUs."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly  # Load before sys.modules patch contexts.

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


node = load("drama3", ROOT / "nodes/drama3.py")
worker = load("drama3_worker", ROOT / "scripts/drama3_worker.py")


def audio(seconds=0.1, channels=1, batch=1):
    return {"waveform": torch.ones(batch, channels, int(44100 * seconds)) * 0.1, "sample_rate": 44100}


class ReferencesTest(unittest.TestCase):
    def test_reference_chain_is_immutable_and_preserves_speaker_ids(self):
        first = node.CRTP_Drama3Reference().append(audio(), "First voice.", 2)[0]
        second = node.CRTP_Drama3Reference().append(audio(), "Second voice.", 7, first)[0]
        self.assertEqual([r['speaker_id'] for r in first], [2])
        self.assertEqual([r['speaker_id'] for r in second], [2, 7])
        with self.assertRaisesRegex(ValueError, "unique"):
            node.CRTP_Drama3Reference().append(audio(), "Duplicate.", 2, first)

    def test_incomplete_or_ambiguous_references_fail_before_loading(self):
        cases = [dict(reference_audio=audio()), dict(reference_text="Orphan transcript"),
                 dict(reference_audio=audio(), reference_text="Hello", references=[{}]),
                 dict(reference_audio=audio(), reference_text="<|speaker:4|> Conflict")]
        for kwargs in cases:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                node._references(**kwargs)
        self.assertEqual(node._references(), [])

    def test_audio_batches_nonfinite_and_long_cuda_references_are_rejected(self):
        bad = audio(); bad['waveform'][0, 0, 0] = float('nan')
        for value in (audio(batch=2), bad, audio(seconds=20.01), {'waveform': torch.zeros(1, 1, 0), 'sample_rate': 44100}):
            with self.assertRaises(ValueError):
                node._validate_audio(value, 20)
        node._validate_audio(audio(seconds=20), 20)

    def test_model_formats_and_missing_shards_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp = types.SimpleNamespace(models_dir=root, folder_names_and_paths={})
            model = root / 'tts/test'; model.mkdir(parents=True)
            (model / 'config.json').write_text(json.dumps({'model_type': 'fish_drama3'}))
            for f in ['model-00001.safetensors', 'codec.safetensors', 'tokenizer.json']:
                (model / f).write_bytes(b'fixture')
            with patch.dict(sys.modules, {'folder_paths': fp}):
                self.assertEqual(node._resolve_model('test', 'mlx'), str(model.resolve()))
                with self.assertRaisesRegex(ValueError, 'W4A16|w4a16'):
                    node._resolve_model('test', 'cuda')
                (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'missing.safetensors'}}))
                with self.assertRaisesRegex(FileNotFoundError, 'missing.safetensors'):
                    node._resolve_model('test', 'mlx')

    def test_node_serializes_reference_without_changing_rate_and_returns_comfy_audio(self):
        mm = types.ModuleType('comfy.model_management')
        mm.unload_all_models = Mock(); mm.soft_empty_cache = Mock()
        comfy = types.ModuleType('comfy'); comfy.model_management = mm
        ref = audio(channels=2)
        ref['waveform'][:, 1] *= 3
        def fake_worker(python, request, directory, timeout):
            self.assertEqual(request['references'][0]['text'], '<|speaker:0|> Hello reference.')
            data, sr = sf.read(request['references'][0]['audio_path'])
            self.assertEqual(sr, 44100)
            np.testing.assert_allclose(data, 0.2, atol=1e-7)
            sf.write(request['output'], np.ones(4000, dtype=np.float32) * 0.25, 44100, subtype='FLOAT')
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': mm}), \
             patch.object(node, '_interrupt'), patch.object(node, '_python', return_value=sys.executable), \
             patch.object(node, '_resolve_model', return_value='/fixture'), \
             patch.object(node.platform, 'system', return_value='Darwin'), \
             patch.object(node.platform, 'machine', return_value='arm64'), \
             patch.object(node, '_run_worker', side_effect=fake_worker):
            result, = node.CRTP_Drama3MLXGenerate().generate('model', 'Target.', reference_audio=ref,
                                                         reference_text='Hello reference.')
        self.assertEqual(result['waveform'].shape, (1, 1, 4000))
        self.assertEqual(result['sample_rate'], 44100)
        mm.unload_all_models.assert_called_once()


class MLXTextAndSilenceTest(unittest.TestCase):
    def split(self, text):
        return worker.sentence_chunks(text, lambda text: [text])

    def test_sentences_are_not_repacked_and_cues_stay_with_speech(self):
        self.assertEqual(self.split('[Speak in Hungarian]\n[laugh] Jó reggelt! Hogy vagy? Rég láttalak...'),
                         ['[Speak in Hungarian]\n[laugh] Jó reggelt!', 'Hogy vagy?', 'Rég láttalak...'])

    def test_speaker_ids_continue_across_sentences_and_turns(self):
        self.assertEqual(self.split('<|speaker:2|> Hello. Still here! <|speaker:7|> Good. Goodbye.'),
                         ['<|speaker:2|> Hello.', '<|speaker:2|> Still here!', '<|speaker:7|> Good.', '<|speaker:7|> Goodbye.'])

    def test_scopes_directions_decimals_and_titles_are_preserved(self):
        self.assertEqual(self.split('[Quiet. Very quiet.] Dr. Smith paid 3.14 euros. <whisper>Wait. Listen.</whisper>'),
                         ['[Quiet. Very quiet.] Dr. Smith paid 3.14 euros.', '<whisper>Wait. Listen.</whisper>'])
        self.assertEqual(self.split('Hello. [pause]'), ['Hello. [pause]'])

    def test_quiet_speech_and_short_pauses_are_not_removed(self):
        guard = worker.SilenceGuard(1000, 1)
        for _ in range(5):
            guard.feed(np.ones(200) * .004)
            guard.feed(np.zeros(900))
        guard = worker.SilenceGuard(1000, 0)
        guard.feed(np.zeros(10000))

    def test_sustained_noise_floor_is_detected_across_chunk_boundaries(self):
        guard = worker.SilenceGuard(1000, 1)
        guard.feed(np.ones(200) * .1)
        guard.feed(np.ones(475) * .001)
        with self.assertRaisesRegex(worker.GenerationFailure, 'near-silence'):
            guard.feed(np.ones(625) * .001)


class RuntimeTest(unittest.TestCase):
    def fake_mlx_modules(self, mx, util):
        mlx = types.ModuleType('mlx'); mlx.core = mx
        drama = types.ModuleType('mlx_audio.tts.models.fish_drama3.drama3')
        drama.split_text = lambda text: [text]
        return {'mlx': mlx, 'mlx.core': mx, 'mlx_audio.tts.utils': util,
                'mlx_audio': types.ModuleType('mlx_audio'),
                'mlx_audio.tts': types.ModuleType('mlx_audio.tts'),
                'mlx_audio.tts.models': types.ModuleType('mlx_audio.tts.models'),
                'mlx_audio.tts.models.fish_drama3': types.ModuleType('mlx_audio.tts.models.fish_drama3'),
                'mlx_audio.tts.models.fish_drama3.drama3': drama}

    def test_mlx_consumes_each_stream_chunk_once_and_seeds_sampling(self):
        mx = types.ModuleType('mlx.core'); mx.random = Mock(); mx.get_peak_memory = lambda: 123
        model = Mock(sample_rate=44100)
        model.generate.return_value = [types.SimpleNamespace(audio=np.full(n, v), sample_rate=44100, segment_idx=0, token_count=1 if n else 0)
                                       for n, v in [(12, .1), (8, .2), (0, 0)]]
        util = types.ModuleType('mlx_audio.tts.utils'); util.load_model = Mock(return_value=model)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, self.fake_mlx_modules(mx, util)):
            request = self.request(directory)
            worker.run_mlx(request)
            data, sr = sf.read(request['output'])
        self.assertEqual(len(data), 20)
        self.assertEqual(sr, 44100)
        np.testing.assert_allclose(data[:12], .1)
        mx.random.seed.assert_called_once_with(42)
        self.assertTrue(model.generate.call_args.kwargs['stream'])
        self.assertEqual(model.generate.call_args.kwargs['ref_text'], ['<|speaker:0|> Reference.'])

    def test_mlx_frame_limit_does_not_silently_return_a_truncated_take(self):
        mx = types.ModuleType('mlx.core'); mx.random = Mock()
        model = Mock(sample_rate=44100)
        model.generate.return_value = [types.SimpleNamespace(audio=np.ones(100), sample_rate=44100, segment_idx=0, token_count=128)]
        util = types.ModuleType('mlx_audio.tts.utils'); util.load_model = Mock(return_value=model)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, self.fake_mlx_modules(mx, util)):
            with self.assertRaisesRegex(RuntimeError, 'frame limit'):
                worker.run_mlx(self.request(directory))
            self.assertFalse((Path(directory) / 'output.wav').exists())
        self.assertEqual(mx.random.seed.call_args_list[-1].args, (43,))

    def test_failed_take_is_discarded_and_split_hook_restored(self):
        mx = types.ModuleType('mlx.core'); mx.random = Mock(); mx.get_peak_memory = lambda: 123
        model = Mock(sample_rate=1000)
        def result(a):
            return types.SimpleNamespace(audio=np.asarray(a), sample_rate=1000, segment_idx=0, token_count=1)
        model.generate.side_effect = [iter([result(np.ones(100) * .1), result(np.zeros(1000))]),
                                      iter([result(np.ones(150) * .2)])]
        util = types.ModuleType('mlx_audio.tts.utils'); util.load_model = Mock(return_value=model)
        modules = self.fake_mlx_modules(mx, util)
        drama = modules['mlx_audio.tts.models.fish_drama3.drama3']; original = drama.split_text
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            request = self.request(directory); request.update(max_silence_seconds=.5, references=[])
            worker.run_mlx(request)
            data, _ = sf.read(request['output'])
            self.assertIs(drama.split_text, original)
        np.testing.assert_allclose(data, np.full(150, .2))
        self.assertEqual([c.args[0] for c in mx.random.seed.call_args_list], [42, 43])

    def test_reference_resampling_preserves_duration_and_filters_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tone.wav'
            t = np.arange(16000) / 16000
            sf.write(path, .5 * np.sin(2 * np.pi * 7000 * t), 16000, subtype='FLOAT')
            data, = worker.mlx_references([{'audio_path': str(path)}], 44100)
            self.assertEqual(data.shape, (44100,))
            spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data)))) ** 2
            frequencies = np.fft.rfftfreq(len(data), 1 / 44100)
            self.assertLess(spectrum[frequencies > 8100].sum() / spectrum.sum(), .001)

    def test_reference_gain_is_normalized_and_silence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wave = np.sin(np.arange(4410) * 2 * np.pi * 440 / 44100).astype(np.float32)
            for name, gain in [('quiet', .1), ('loud', .8), ('silent', 0)]:
                sf.write(root / (name + '.wav'), wave * gain, 44100, subtype='FLOAT')
            quiet, loud = worker.mlx_references([{'audio_path': str(root / (n + '.wav'))}
                                                for n in ('quiet', 'loud')], 44100)
            np.testing.assert_allclose(quiet, loud, atol=1e-6)
            self.assertAlmostEqual(float(np.max(np.abs(quiet))), .95, places=6)
            with self.assertRaisesRegex(ValueError, 'silent'):
                worker.mlx_references([{'audio_path': str(root / 'silent.wav')}], 44100)

    def test_empty_text_chunk_fails_instead_of_silently_skipping_words(self):
        mx = types.ModuleType('mlx.core'); mx.random = Mock()
        model = Mock(sample_rate=44100)
        model.generate.return_value = [types.SimpleNamespace(audio=np.ones(100) * .1,
                                      sample_rate=44100, segment_idx=1, token_count=1)]
        util = types.ModuleType('mlx_audio.tts.utils'); util.load_model = Mock(return_value=model)
        modules = self.fake_mlx_modules(mx, util)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            request = self.request(directory); request.update(text='First. Second.', references=[])
            with self.assertRaisesRegex(worker.GenerationFailure, 'skipped a text chunk'):
                worker.run_mlx(request)
            self.assertFalse(Path(request['output']).exists())

    def request(self, directory):
        sf.write(Path(directory) / 'ref.wav', np.ones(100, dtype=np.float32) * .1, 44100)
        return {'model_path': '/model', 'text': 'Target.', 'references': [{'audio_path': str(Path(directory) / 'ref.wav'), 'text': '<|speaker:0|> Reference.'}],
                'seed': 42, 'temperature': .7, 'top_p': .7, 'top_k': 30, 'max_new_tokens': 128,
                'timeout_seconds': 30, 'output': str(Path(directory) / 'output.wav')}

    def test_cuda_real_http_roundtrip_and_error_details(self):
        # Real local HTTP transport and child process; only GPU probing and the
        # inference executable are replaced. Captures the actual posted body.
        server_source = '''
import io, json, sys, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"healthy"}')
 def do_POST(self):
  data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  open(sys.argv[2], 'w').write(json.dumps(data))
  if data['input'] == 'fail':
   self.send_response(400); self.end_headers(); self.wfile.write(b'reference rejected'); return
  b = io.BytesIO()
  with wave.open(b, 'wb') as w:
   w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(b'\\x01\\x00'*100)
  self.send_response(200); self.end_headers(); self.wfile.write(b.getvalue())
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quant = root / 'models/fishaudio_drama3/quantization.py'; quant.parent.mkdir(parents=True)
            quant.write_text('ExpertW4A16L8Config')
            executable = root / 'sgl-omni'; executable.touch()
            script = root / 'server.py'; script.write_text(server_source)
            fake_sgl = types.SimpleNamespace(__file__=str(root / '__init__.py'))
            request = self.request(directory)
            captured = root / 'captured.json'
            command_python = sys.executable
            with patch.dict(sys.modules, {'sglang_omni': fake_sgl}), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_capability', return_value=(12, 0)), \
                 patch.object(worker, 'cuda_command', side_effect=lambda b,r,p: [command_python, str(script), str(p), str(captured)]), \
                 patch.object(worker.sys, 'executable', str(root / 'python')):
                worker.run_cuda(request)
                self.assertEqual(sf.info(request['output']).frames, 100)
                body = json.loads(captured.read_text())
                self.assertNotIn('seed', body)
                self.assertEqual(body['references'], request['references'])
                request['text'] = 'fail'
                with self.assertRaisesRegex(RuntimeError, 'HTTP 400: reference rejected'):
                    worker.run_cuda(request)

    def test_worker_failure_and_cancellation_release_process_group(self):
        for mode in ['failure', 'cancel']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); (root / 'scripts').mkdir()
                source = "import time\ntime.sleep(30)" if mode == 'cancel' else "raise RuntimeError('fixture error')"
                (root / 'scripts/drama3_worker.py').write_text(source)
                real_popen = node.subprocess.Popen
                processes = []
                def spawn(*args, **kwargs):
                    proc = real_popen(*args, **kwargs); processes.append(proc); return proc
                with patch.object(node, 'PACK', root), patch.object(node.subprocess, 'Popen', side_effect=spawn), \
                     patch.object(node, '_interrupt', side_effect=KeyboardInterrupt if mode == 'cancel' else None):
                    with self.assertRaises(KeyboardInterrupt if mode == 'cancel' else RuntimeError):
                        node._run_worker(sys.executable, {}, directory, 10)
                self.assertIsNotNone(processes[0].poll())
                with self.assertRaises(ProcessLookupError):
                    os.killpg(processes[0].pid, 0)


if __name__ == '__main__':
    unittest.main()
