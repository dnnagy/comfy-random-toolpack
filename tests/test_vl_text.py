"""Routing, real HTTP/process lifecycle, and projector contracts without a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('vl_text', Path(__file__).parents[1] / 'nodes/vl_text.py')
vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vl)


class VLTests(unittest.TestCase):
    def test_lazy_routing_never_requests_clip_for_gguf(self):
        node = vl.CRTP_VLTextGenerate()
        self.assertEqual(node.check_lazy_status(vl.NATIVE), ['clip'])
        self.assertEqual(node.check_lazy_status(vl.NATIVE, clip=object()), [])
        self.assertEqual(node.check_lazy_status(vl.GGUF), [])
        clip = Mock()
        clip.decode.return_value = 'native answer'
        with patch.object(vl, '_server', side_effect=AssertionError('GGUF must stay unloaded')):
            self.assertEqual(node.generate(vl.NATIVE, 'question', object(), clip=clip,
                sampling_mode='on', temperature=.3, top_k=0, thinking=True,
                use_default_template=False), ('native answer',))
        self.assertTrue(clip.tokenize.call_args.kwargs['skip_template'])
        self.assertTrue(clip.generate.call_args.kwargs['do_sample'])
        self.assertEqual(clip.generate.call_args.kwargs['temperature'], .3)
        self.assertFalse(clip.generate.call_args.kwargs['mtp'])

    def test_missing_projector_or_raw_template_is_rejected(self):
        node = vl.CRTP_VLTextGenerate()
        for kw in ({'mmproj': ''}, {'use_default_template': False}):
            with self.assertRaises(ValueError), patch.object(vl, '_server') as server:
                node.generate(vl.GGUF, 'question', object(), **kw)
            server.assert_not_called()

    def test_paths_resolve_models_and_projector_without_silent_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            fp = types.SimpleNamespace(models_dir=directory, folder_names_and_paths={})
            path = Path(directory) / 'llm' / vl.PROJECTOR_FILE
            path.parent.mkdir(parents=True)
            path.write_bytes(b'GGUFdata')
            with patch.dict('sys.modules', {'folder_paths': fp}):
                self.assertEqual(vl._resolve_gguf(vl.PROJECTOR_FILE), str(path.resolve()))
                with self.assertRaises(FileNotFoundError):
                    vl._resolve_gguf(vl.MODEL_FILE)
                path.write_bytes(b'<html>')
                with self.assertRaises(ValueError):
                    vl._resolve_gguf(vl.PROJECTOR_FILE)

    def test_payload_preserves_images_order_and_sampling(self):
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,one'}},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,two'}}]
        for mode in ('on', 'off'):
            p = vl._payload('compare', content, 512, mode, .4, 23, .8, .1, 1.2, 123, .5, True)
            self.assertEqual(p['messages'][0]['content'][:2], content)
            self.assertEqual(p['max_tokens'], 512)
            self.assertEqual(p['temperature'], .4 if mode == 'on' else 0)
            self.assertEqual(p['repeat_penalty'], 1.2 if mode == 'on' else 1)
            self.assertTrue(p['chat_template_kwargs']['enable_thinking'])

    def test_command_always_attaches_projector_and_bounds_context(self):
        args = vl._command('llama-server', '/model', '/projector', 123, 'token', 16384, 1024)
        for flag, value in [('--mmproj', '/projector'), ('--host', '127.0.0.1'),
                            ('--ctx-size', '16384'), ('--parallel', '1'),
                            ('--image-max-tokens', '1024')]:
            self.assertEqual(args[args.index(flag)+1], value)

    def test_server_terminates_on_success_error_and_interruption(self):
        for failure in (None, RuntimeError('generation failed'), KeyboardInterrupt()):
            proc = Mock(); proc.poll.return_value = None
            with patch.object(vl, '_server_binary', return_value='/bin/fake'), \
                 patch.object(vl, '_interrupt'), patch.object(vl, '_request', return_value={'status': 'ok'}), \
                 patch.object(vl.subprocess, 'Popen', return_value=proc):
                try:
                    with vl._server('model', 'projector'):
                        if failure is not None:
                            raise failure
                except (RuntimeError, KeyboardInterrupt) as exc:
                    self.assertIs(exc, failure)
                proc.terminate.assert_called_once()
                proc.wait.assert_called_once()

    def test_complete_surfaces_http_errors_and_reasoning(self):
        with patch.object(vl, '_interrupt'), patch.object(vl, '_request', return_value={
                'choices': [{'message': {'content': 'answer', 'reasoning_content': 'reason'}}]}):
            self.assertEqual(vl._complete('url', 'token', {}), '<think>\nreason\n</think>\n\nanswer')
        with patch.object(vl, '_interrupt'), patch.object(vl, '_request', side_effect=RuntimeError('bad projector')):
            with self.assertRaisesRegex(RuntimeError, 'bad projector'):
                vl._complete('url', 'token', {})

    def test_real_loopback_http_process_is_cleaned_up(self):
        # Exercise actual auth, startup polling, JSON transport, response parsing,
        # and process teardown with a tiny server rather than a GPU model.
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'llama-server'
            script.write_text('''#!/usr/bin/env python3
import http.server,json,sys
port=int(sys.argv[sys.argv.index('--port')+1]); token=sys.argv[sys.argv.index('--api-key')+1]
assert '--mmproj' in sys.argv
class H(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def reply(self,obj):
  assert self.headers['Authorization']=='Bearer '+token
  self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(obj).encode())
 def do_GET(self): self.reply({'status':'ok'})
 def do_POST(self):
  p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  self.reply({'choices':[{'message':{'content':p['messages'][0]['content'][-1]['text']}}]})
http.server.HTTPServer(('127.0.0.1',port),H).serve_forever()
''')
            script.chmod(0o755)
            with patch.dict(os.environ, {'CRTP_LLAMA_SERVER': str(script)}), patch.object(vl, '_interrupt'):
                with vl._server('model', 'projector') as (url, token):
                    p = vl._payload('round trip', [], 10, 'off', .7, 64, .95, .05, 1.05, 0, 0, False)
                    self.assertEqual(vl._complete(url, token, p), 'round trip')
                with self.assertRaises(OSError):
                    vl._request(url + '/health', token)


if __name__ == '__main__':
    unittest.main()
