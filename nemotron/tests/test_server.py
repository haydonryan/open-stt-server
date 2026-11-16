"""Protocol, upload, error, admission and disconnect regression tests."""

import asyncio
import base64
import io
import subprocess
import unittest
import wave

import numpy as np
from aiohttp import FormData
from aiohttp.client_exceptions import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

import nemotron.server as server_module
from nemotron.server import Config, create_app


def wav_bytes(seconds=1.0, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b'\x00\x00' * round(seconds * rate))
    return buffer.getvalue()


class FakeAdapter:
    def __init__(self):
        self.reset_calls = 0
        self.feed_calls = 0
        self.samples = 0
        self.text = ''
        self.closed = False

    def reset(self):
        self.reset_calls += 1
        self.samples = 0
        self.text = ''

    def feed(self, pcm):
        self.feed_calls += 1
        self.samples += len(pcm)
        self.text = 'heard' + '!' * self.feed_calls
        return self.text

    def finish(self):
        return f'final {self.samples}'

    def close(self):
        self.closed = True


async def make_client(config=None, adapter=None):
    adapter = adapter or FakeAdapter()
    app = create_app(config or Config(), lambda: adapter)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, adapter


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_and_models_preserve_whisper_alias(self):
        client, _adapter = await make_client()
        try:
            health = await client.get('/health')
            self.assertEqual(health.status, 200)
            self.assertEqual(await health.json(), {'status': 'ok'})

            models = await client.get('/v1/models')
            self.assertEqual(models.status, 200)
            payload = await models.json()
            ids = [row['id'] for row in payload['data']]
            self.assertIn('whisper-1', ids)
            self.assertIn('nemotron-speech-streaming-en-0.6b', ids)
        finally:
            await client.close()

    async def test_multipart_json_and_text_uploads(self):
        client, adapter = await make_client()
        try:
            audio = wav_bytes()
            form = FormData()
            form.add_field('file', audio, filename='audio.wav', content_type='audio/wav')
            form.add_field('model', 'whisper-1')
            form.add_field('response_format', 'json')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 200)
            payload = await response.json()
            self.assertEqual(payload['text'], 'final 16000')
            self.assertAlmostEqual(payload['usage']['seconds'], 1.0, places=3)
            self.assertGreater(adapter.feed_calls, 0)

            form = FormData()
            form.add_field('file', audio, filename='audio.wav', content_type='audio/wav')
            form.add_field('model', 'nemotron')
            form.add_field('response_format', 'text')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers['Content-Type'].split(';')[0], 'text/plain')
            self.assertEqual(await response.text(), 'final 16000')

            flac = subprocess.run(
                ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
                 '-f', 'flac', 'pipe:1'],
                input=audio, stdout=subprocess.PIPE, check=True,
            ).stdout
            form = FormData()
            form.add_field('file', flac, filename='audio.flac', content_type='audio/flac')
            form.add_field('model', 'whisper-1')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())['text'], 'final 16000')
        finally:
            await client.close()

    async def test_upload_errors(self):
        client, _adapter = await make_client()
        try:
            form = FormData(default_to_multipart=True)
            form.add_field('model', 'whisper-1')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())['error']['message'], 'Missing required field: file')

            form = FormData()
            form.add_field('file', b'not audio', filename='bad.bin', content_type='application/octet-stream')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertIn('Failed to decode audio', (await response.json())['error']['message'])

            form = FormData()
            form.add_field('file', wav_bytes(), filename='audio.wav', content_type='audio/wav')
            form.add_field('model', 'unknown-model')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 404)
            self.assertEqual((await response.json())['error']['param'], 'model')

            form = FormData()
            form.add_field('file', io.BytesIO(b'\0' * (1024 * 1024 + 1)), filename='big.wav', content_type='audio/wav')
            client_big, _adapter_big = await make_client(Config(max_upload_mb=1))
            try:
                response = await client_big.post('/v1/audio/transcriptions', data=form)
                self.assertEqual(response.status, 413)
            finally:
                await client_big.close()
        finally:
            await client.close()

    async def test_realtime_protocol_and_append_only_partials(self):
        client, _adapter = await make_client()
        try:
            ws = await client.ws_connect('/v1/realtime')
            created = await ws.receive_json(timeout=5)
            self.assertEqual(created['type'], 'session.created')
            self.assertTrue(created['id'].startswith('sess-'))

            await ws.send_json({'type': 'session.update', 'model': 'whisper-1'})
            await ws.send_json({'type': 'input_audio_buffer.commit'})
            pcm = np.zeros(1600, dtype='<i2').tobytes()
            await ws.send_json({
                'type': 'input_audio_buffer.append',
                'audio': base64.b64encode(pcm).decode('ascii'),
            })

            first = await ws.receive_json(timeout=5)
            self.assertEqual(first['type'], 'transcription.delta')
            self.assertEqual(first['delta'], 'heard!')
            second = await ws.receive_json(timeout=5)
            self.assertEqual(second['type'], 'transcription.delta')
            self.assertEqual(second['delta'], '!')

            await ws.send_json({'type': 'input_audio_buffer.commit', 'final': True})
            done = await ws.receive_json(timeout=5)
            self.assertEqual(done['type'], 'transcription.done')
            self.assertEqual(done['text'], 'final 1600')
            self.assertEqual(done['received_samples'], 1600)
            self.assertEqual(done['usage'], {'type': 'duration', 'seconds': 0.1})
            self.assertGreaterEqual(done['compute_seconds'], 0)
            await ws.close()
        finally:
            await client.close()

    async def test_realtime_error_events(self):
        client, _adapter = await make_client()
        try:
            ws = await client.ws_connect('/v1/realtime')
            await ws.receive_json(timeout=5)

            await ws.send_json({'type': 'input_audio_buffer.commit', 'final': 'yes'})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'invalid_event')

            await ws.send_json([1, 2])
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'invalid_event')

            await ws.send_json({'type': 'input_audio_buffer.commit'})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'model_not_validated')

            await ws.send_json({'type': 'session.update'})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'invalid_event')

            await ws.send_json({'type': 'session.update', 'model': 'unknown-model'})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'model_not_found')

            await ws.send_json({'type': 'input_audio_buffer.append', 'audio': '!!!'})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'invalid_audio')

            await ws.send_json({'type': 'input_audio_buffer.append', 'audio': 'AA=='})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'invalid_audio')

            await ws.send_json({'type': 'unexpected'})
            event = await ws.receive_json(timeout=5)
            self.assertEqual(event['code'], 'unknown_event')
            await ws.close()
        finally:
            await client.close()

    async def test_admission_rejects_second_active_stream(self):
        client, _adapter = await make_client()
        try:
            first = await client.ws_connect('/v1/realtime')
            await first.receive_json(timeout=5)
            with self.assertRaises(WSServerHandshakeError) as context:
                await client.ws_connect('/v1/realtime')
            self.assertEqual(context.exception.status, 409)
            await first.close()
            await asyncio.sleep(0.2)

            second = await client.ws_connect('/v1/realtime')
            self.assertEqual((await second.receive_json(timeout=5))['type'], 'session.created')
            await second.close()
        finally:
            await client.close()

    async def test_disconnect_cleanup_and_successive_sessions(self):
        client, adapter = await make_client()
        try:
            ws = await client.ws_connect('/v1/realtime')
            await ws.receive_json(timeout=5)
            await ws.send_json({'type': 'session.update', 'model': 'whisper-1'})
            await ws.send_json({'type': 'input_audio_buffer.commit'})
            await ws.send_json({
                'type': 'input_audio_buffer.append',
                'audio': base64.b64encode(np.zeros(1600, dtype='<i2').tobytes()).decode('ascii'),
            })
            await ws.receive_json(timeout=5)
            baseline = adapter.reset_calls
            await ws.close()
            await asyncio.sleep(0.2)
            self.assertGreater(adapter.reset_calls, baseline)

            second = await client.ws_connect('/v1/realtime')
            await second.receive_json(timeout=5)
            await second.send_json({'type': 'session.update', 'model': 'whisper-1'})
            await second.send_json({'type': 'input_audio_buffer.commit'})
            await second.send_json({
                'type': 'input_audio_buffer.append',
                'audio': base64.b64encode(np.zeros(1600, dtype='<i2').tobytes()).decode('ascii'),
            })
            await second.send_json({'type': 'input_audio_buffer.commit', 'final': True})
            while True:
                event = await second.receive_json(timeout=5)
                if event['type'] == 'transcription.done':
                    done = event
                    break
            self.assertEqual(done['text'], 'final 1600')
            await second.close()
        finally:
            await client.close()
        self.assertTrue(adapter.closed)

    async def test_multipart_field_and_option_validation(self):
        client, _adapter = await make_client()
        try:
            audio = wav_bytes()

            # Only the file field exercises the awaited multipart reader and alias default.
            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())['text'], 'final 16000')

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('file', audio, filename='b.wav', content_type='audio/wav')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())['error']['param'], 'file')

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('response_format', 'srt')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())['error']['param'], 'response_format')

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('language', 'fr')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())['error']['param'], 'language')

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('language', 'en')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 200)

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('prompt', 'hello')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())['error']['param'], 'prompt')

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('translate', 'true')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())['error']['param'], 'translate')

            form = FormData()
            form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            form.add_field('model', 'x' * 5000)
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertIn('exceeds 4096 bytes', (await response.json())['error']['message'])
        finally:
            await client.close()

    async def test_bounded_decode_rejects_oversized_pcm(self):
        client, _adapter = await make_client(Config(max_upload_mb=1))
        try:
            silence = subprocess.run(
                ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                 '-i', 'anullsrc=r=16000:cl=mono', '-t', '140', '-f', 'flac', 'pipe:1'],
                stdout=subprocess.PIPE, check=True,
            ).stdout
            self.assertLess(len(silence), 1024 * 1024)

            form = FormData()
            form.add_field('file', silence, filename='silence.flac', content_type='audio/flac')
            form.add_field('model', 'whisper-1')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 400)
            self.assertIn('Decoded audio exceeds', (await response.json())['error']['message'])

            form = FormData()
            form.add_field('file', wav_bytes(), filename='audio.wav', content_type='audio/wav')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())['text'], 'final 16000')
        finally:
            await client.close()

    async def test_http_admission_bounds_waiting(self):
        client, _adapter = await make_client(Config(gate_timeout_s=0.2))
        try:
            ws = await client.ws_connect('/v1/realtime')
            await ws.receive_json(timeout=5)
            form = FormData()
            form.add_field('file', wav_bytes(), filename='audio.wav', content_type='audio/wav')
            form.add_field('model', 'whisper-1')
            response = await client.post('/v1/audio/transcriptions', data=form)
            self.assertEqual(response.status, 409)
            self.assertIn('Model busy', (await response.json())['error']['message'])
            await ws.close()
        finally:
            await client.close()

    async def test_http_admission_queue_limit(self):
        client, _adapter = await make_client(Config(max_pending_http=1))
        original = server_module._decode_audio

        async def slow_decode(data, max_bytes, timeout_s):
            await asyncio.sleep(0.3)
            return b'\x00\x00' * 8000

        server_module._decode_audio = slow_decode
        try:
            audio = wav_bytes()
            first_form = FormData()
            first_form.add_field('file', audio, filename='a.wav', content_type='audio/wav')
            second_form = FormData()
            second_form.add_field('file', audio, filename='b.wav', content_type='audio/wav')
            first_task = asyncio.create_task(
                client.post('/v1/audio/transcriptions', data=first_form))
            await asyncio.sleep(0.1)
            second = await client.post('/v1/audio/transcriptions', data=second_form)
            self.assertEqual(second.status, 409)
            self.assertIn('Model queue is full', (await second.json())['error']['message'])
            first = await first_task
            self.assertEqual(first.status, 200)
            self.assertEqual((await first.json())['text'], 'final 8000')
        finally:
            server_module._decode_audio = original
            await client.close()

    async def test_auth_rejects_missing_key(self):
        client, _adapter = await make_client(Config(api_key='secret'))
        try:
            response = await client.get('/v1/models')
            self.assertEqual(response.status, 401)
            response = await client.get('/v1/models', headers={'Authorization': 'Bearer secret'})
            self.assertEqual(response.status, 200)
        finally:
            await client.close()


if __name__ == '__main__':
    unittest.main()
