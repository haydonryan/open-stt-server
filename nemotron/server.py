"""Nemotron Speech Streaming backend behind the Voxtral-compatible ASR API.

The HTTP surface mirrors open-stt-server/OpenAI (`/health`, `/v1/models`,
`/v1/audio/transcriptions`) and the deployed Voxtral vLLM realtime protocol
(`/v1/realtime`: session.created, session.update, input_audio_buffer.commit,
input_audio_buffer.append, transcription.delta, transcription.done, error).

Model inference stays in the official NeMo cache-aware RNNT pipeline. The HTTP
handler and WebSocket handler only feed newly received PCM; they never reread a
whole recording or replay an utterance prefix into the model.
"""

import asyncio
import base64
import binascii
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from aiohttp import WSMsgType, web


MODEL_ID = 'nemotron-speech-streaming-en-0.6b'
MODEL_ALIASES = frozenset({
    MODEL_ID,
    'nemotron',
    'nvidia/nemotron-speech-streaming-en-0.6b',
    'whisper-1',
})
ENGLISH_LANGUAGES = frozenset({'en', 'english', 'en-us', 'en_us', 'auto'})
MAX_FEED_SAMPLES = 1280  # 80 ms at 16 kHz, the adapter's native ingress block.
HTTP_PCM_CHUNK_BYTES = MAX_FEED_SAMPLES * 2


@dataclass(frozen=True)
class Config:
    host: str = '0.0.0.0'
    port: int = 8080
    checkpoint: str = '/models/nemotron-speech-streaming-en-0.6b.nemo'
    device: str = 'cpu'
    threads: int = 8
    api_key: str | None = None
    max_upload_mb: int = 25
    model_alias: str = 'whisper-1'
    max_pending_http: int = 2
    gate_timeout_s: float = 30.0
    decode_timeout_s: float = 120.0
    max_field_bytes: int = 4096

    @classmethod
    def from_env(cls):
        return cls(
            host=os.environ.get('NEMOTRON_HOST', '0.0.0.0'),
            port=int(os.environ.get('NEMOTRON_PORT', '8080')),
            checkpoint=os.environ.get(
                'NEMOTRON_CHECKPOINT', '/models/nemotron-speech-streaming-en-0.6b.nemo'),
            device=os.environ.get('NEMOTRON_DEVICE', 'cpu'),
            threads=int(os.environ.get('NEMOTRON_THREADS', '8')),
            api_key=os.environ.get('NEMOTRON_API_KEY') or None,
            max_upload_mb=int(os.environ.get('NEMOTRON_MAX_UPLOAD_MB', '25')),
            model_alias=os.environ.get('NEMOTRON_MODEL_ALIAS', 'whisper-1'),
            max_pending_http=int(os.environ.get('NEMOTRON_MAX_PENDING_HTTP', '2')),
            gate_timeout_s=float(os.environ.get('NEMOTRON_GATE_TIMEOUT_S', '30')),
            decode_timeout_s=float(os.environ.get('NEMOTRON_DECODE_TIMEOUT_S', '120')),
            max_field_bytes=int(os.environ.get('NEMOTRON_MAX_FIELD_BYTES', '4096')),
        )


class Runtime:
    """Single-model process state: one inference thread, one active stream."""

    def __init__(self, config, adapter_factory):
        self.config = config
        self.factory = adapter_factory
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nemotron-model')
        self.gate = asyncio.Lock()
        self.stream_active = False
        self.http_pending = 0
        self.adapter = None

    async def start(self):
        loop = asyncio.get_running_loop()
        self.adapter = await loop.run_in_executor(self.pool, self.factory)

    async def infer(self, method, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, method, *args)

    async def reset(self):
        await self.infer(self.adapter.reset)

    async def close(self):
        if self.adapter is not None:
            try:
                await self.infer(self.adapter.close)
            except Exception:
                pass
        self.pool.shutdown(wait=True)


def _authorized(request, config):
    if not config.api_key:
        return True
    return request.headers.get('Authorization') == f'Bearer {config.api_key}'


def _error(status, message, error_type, code=None, param=None):
    payload = {'error': {'message': message, 'type': error_type}}
    if code is not None:
        payload['error']['code'] = code
    if param is not None:
        payload['error']['param'] = param
    return web.json_response(payload, status=status)


def _model_list(config):
    ids = [config.model_alias]
    if MODEL_ID not in ids:
        ids.append(MODEL_ID)
    return {
        'object': 'list',
        'data': [
            {'id': model_id, 'object': 'model', 'owned_by': 'nemotron', 'root': MODEL_ID}
            for model_id in ids
        ],
    }


def _known_model(name, config):
    return name in MODEL_ALIASES or name == config.model_alias


def _resolve_model(requested, config):
    name = requested.strip() if requested else config.model_alias
    if not name:
        name = config.model_alias
    return name if _known_model(name, config) else None


async def _read_field_text(field, limit):
    """Read a bounded non-file multipart field as UTF-8 text."""
    buffer = bytearray()
    while True:
        chunk = await field.read_chunk(64 * 1024)
        if not chunk:
            break
        if len(buffer) + len(chunk) > limit:
            raise ValueError(f'Multipart field `{field.name}` exceeds {limit} bytes')
        buffer.extend(chunk)
    return buffer.decode('utf-8', errors='replace').strip()


async def _decode_audio(data, max_bytes, timeout_s):
    """Decode arbitrary container audio to mono 16 kHz PCM16 via ffmpeg.

    stdout is read incrementally and capped so decoded PCM cannot exceed the
    upload budget, and the subprocess is killed and reaped on timeout or
    cancellation.
    """
    limit = max_bytes * 4
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
        '-f', 's16le', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', '-',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    decoded = bytearray()
    errors = bytearray()

    async def pump_stdin():
        try:
            for offset in range(0, len(data), 64 * 1024):
                proc.stdin.write(data[offset:offset + 64 * 1024])
                await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def read_stdout():
        while True:
            chunk = await proc.stdout.read(64 * 1024)
            if not chunk:
                return
            if len(decoded) + len(chunk) > limit:
                raise ValueError(f'Decoded audio exceeds {limit} bytes')
            decoded.extend(chunk)

    async def read_stderr():
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                return
            if len(errors) < 4096:
                errors.extend(chunk[:4096 - len(errors)])

    try:
        async with asyncio.timeout(timeout_s):
            await asyncio.gather(pump_stdin(), read_stdout(), read_stderr())
            await proc.wait()
    except BaseException:
        proc.kill()
        try:
            await asyncio.shield(asyncio.create_task(proc.wait()))
        except asyncio.CancelledError:
            pass
        raise

    if proc.returncode != 0:
        detail = bytes(errors).decode('utf-8', errors='replace').strip() or 'ffmpeg failed'
        raise ValueError(detail[:500])
    if not decoded:
        raise ValueError('Decoded audio is empty')
    if len(decoded) % 2:
        del decoded[-1]
    return bytes(decoded)


def create_app(config=None, adapter_factory=None):
    config = config or Config.from_env()

    if adapter_factory is None:
        def adapter_factory():
            from .adapter import make_adapter
            return make_adapter('nemotron', config.device, config.threads, config.checkpoint)

    runtime = Runtime(config, adapter_factory)
    app = web.Application()

    async def startup(_app):
        await runtime.start()

    async def cleanup(_app):
        await runtime.close()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)

    async def health(_request):
        return web.json_response({'status': 'ok'})

    async def models(request):
        if not _authorized(request, config):
            return _error(401, 'Invalid or missing API key', 'authentication_error')
        return web.json_response(_model_list(config))

    async def transcribe(request):
        if not _authorized(request, config):
            return _error(401, 'Invalid or missing API key', 'authentication_error')
        if not (request.content_type or '').startswith('multipart/form-data'):
            return _error(
                400, 'Request must be multipart/form-data', 'invalid_request_error')

        if runtime.http_pending >= config.max_pending_http:
            return _error(
                409,
                f'Model queue is full: {config.max_pending_http} HTTP transcriptions '
                'are already waiting',
                'invalid_request_error',
            )
        runtime.http_pending += 1
        acquired = False
        try:
            max_bytes = config.max_upload_mb * 1024 * 1024
            audio = bytearray()
            file_seen = False
            model_name = None
            response_format = 'json'
            language = None
            try:
                reader = await request.multipart()
                async for field in reader:
                    name = field.name
                    if name == 'file':
                        if file_seen:
                            return _error(
                                400, 'Duplicate multipart field: file',
                                'invalid_request_error', param='file')
                        file_seen = True
                        while True:
                            chunk = await field.read_chunk(64 * 1024)
                            if not chunk:
                                break
                            if len(audio) + len(chunk) > max_bytes:
                                return _error(
                                    413,
                                    f'Audio file exceeds {config.max_upload_mb} MB limit',
                                    'invalid_request_error',
                                    param='file',
                                )
                            audio.extend(chunk)
                    elif name == 'model':
                        model_name = await _read_field_text(field, config.max_field_bytes)
                    elif name == 'response_format':
                        response_format = await _read_field_text(field, config.max_field_bytes)
                    elif name == 'language':
                        language = await _read_field_text(field, config.max_field_bytes)
                    else:
                        return _error(
                            400,
                            f'Unsupported multipart field: {name}',
                            'invalid_request_error',
                            param=name,
                        )
            except Exception as exc:
                return _error(
                    400, f'Invalid multipart request: {exc}', 'invalid_request_error')

            if response_format not in ('json', 'text'):
                return _error(
                    400,
                    f'Unsupported response_format `{response_format}`; supported: json, text',
                    'invalid_request_error',
                    param='response_format',
                )
            if language and language.lower() not in ENGLISH_LANGUAGES:
                return _error(
                    400,
                    f'Unsupported language `{language}`; this model streams English only',
                    'invalid_request_error',
                    param='language',
                )
            if not audio:
                return _error(400, 'Missing required field: file', 'invalid_request_error')

            resolved = _resolve_model(model_name, config)
            if resolved is None:
                requested = model_name or '<default>'
                return _error(
                    404,
                    f"The model `{requested}` does not exist. Available: {', '.join(sorted(MODEL_ALIASES))}",
                    'invalid_request_error',
                    param='model',
                )

            try:
                pcm = await _decode_audio(bytes(audio), max_bytes, config.decode_timeout_s)
            except asyncio.TimeoutError:
                return _error(
                    504,
                    f'Audio decode exceeded {config.decode_timeout_s} seconds',
                    'server_error',
                )
            except Exception as exc:
                return _error(
                    400, f'Failed to decode audio: {exc}', 'invalid_request_error')

            seconds = len(pcm) / 2 / 16000
            try:
                await asyncio.wait_for(runtime.gate.acquire(), timeout=config.gate_timeout_s)
                acquired = True
            except asyncio.TimeoutError:
                return _error(
                    409,
                    f'Model busy: no inference slot within {config.gate_timeout_s} seconds',
                    'invalid_request_error',
                )

            try:
                await runtime.reset()
                for offset in range(0, len(pcm), HTTP_PCM_CHUNK_BYTES):
                    chunk_bytes = pcm[offset:offset + HTTP_PCM_CHUNK_BYTES]
                    samples = np.frombuffer(chunk_bytes, dtype='<i2').astype(np.float32) / 32768.0
                    await runtime.infer(runtime.adapter.feed, samples)
                text = await runtime.infer(runtime.adapter.finish)
            except Exception as exc:
                return _error(
                    500, f'Transcription failed: {exc}', 'server_error')
            finally:
                try:
                    await asyncio.shield(runtime.reset())
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                if acquired:
                    runtime.gate.release()
                    acquired = False

            if response_format == 'text':
                return web.Response(text=text, content_type='text/plain')
            return web.json_response({
                'text': text,
                'usage': {'type': 'duration', 'seconds': round(seconds, 3)},
            })
        finally:
            runtime.http_pending -= 1

    async def realtime(request):
        if not _authorized(request, config):
            return _error(401, 'Invalid or missing API key', 'authentication_error')
        if runtime.stream_active or runtime.gate.locked():
            return web.Response(status=409, text='One active stream per model process')

        runtime.stream_active = True
        try:
            await runtime.gate.acquire()
        except Exception:
            runtime.stream_active = False
            raise

        ws = web.WebSocketResponse(max_msg_size=2**22, heartbeat=30)
        try:
            await ws.prepare(request)
            await ws.send_json({
                'type': 'session.created',
                'id': f'sess-{uuid.uuid4().hex}',
                'created': int(time.time()),
            })

            validated = False
            previous = ''
            received_samples = 0

            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        await ws.send_json({
                            'type': 'error',
                            'error': 'Invalid JSON',
                            'code': 'invalid_json',
                        })
                        continue
                    if not isinstance(payload, dict):
                        await ws.send_json({
                            'type': 'error',
                            'error': 'Event payload must be a JSON object',
                            'code': 'invalid_event',
                        })
                        continue

                    event_type = payload.get('type')
                    if event_type == 'session.update':
                        model = payload.get('model')
                        if not isinstance(model, str) or not model.strip():
                            await ws.send_json({
                                'type': 'error',
                                'error': 'Missing required field: model',
                                'code': 'invalid_event',
                            })
                            continue
                        if not _known_model(model.strip(), config):
                            await ws.send_json({
                                'type': 'error',
                                'error': f"The model `{model}` does not exist.",
                                'code': 'model_not_found',
                            })
                            continue
                        validated = True

                    elif event_type == 'input_audio_buffer.append':
                        raw = payload.get('audio')
                        if not isinstance(raw, str):
                            await ws.send_json({
                                'type': 'error',
                                'error': 'Invalid audio data',
                                'code': 'invalid_audio',
                            })
                            continue
                        try:
                            audio_bytes = base64.b64decode(raw, validate=True)
                        except (binascii.Error, ValueError):
                            await ws.send_json({
                                'type': 'error',
                                'error': 'Invalid audio data',
                                'code': 'invalid_audio',
                            })
                            continue
                        if not audio_bytes or len(audio_bytes) % 2:
                            await ws.send_json({
                                'type': 'error',
                                'error': 'Invalid audio data',
                                'code': 'invalid_audio',
                            })
                            continue
                        if len(audio_bytes) > config.max_upload_mb * 1024 * 1024:
                            await ws.send_json({
                                'type': 'error',
                                'error': f'Audio append exceeds {config.max_upload_mb} MB limit',
                                'code': 'audio_filesize',
                            })
                            continue

                        samples = np.frombuffer(audio_bytes, dtype='<i2').astype(np.float32) / 32768.0
                        received_samples += samples.size
                        try:
                            for offset in range(0, samples.size, MAX_FEED_SAMPLES):
                                chunk = samples[offset:offset + MAX_FEED_SAMPLES]
                                text = await runtime.infer(runtime.adapter.feed, chunk)
                                if text is None or text == previous:
                                    continue
                                if text.startswith(previous):
                                    delta = text[len(previous):]
                                    previous = text
                                    if delta:
                                        await ws.send_json({
                                            'type': 'transcription.delta',
                                            'delta': delta,
                                        })
                                elif not previous:
                                    previous = text
                                    await ws.send_json({
                                        'type': 'transcription.delta',
                                        'delta': text,
                                    })
                                # A non-prefix revision cannot be represented by the
                                # append-only delta protocol; keep partials append-only
                                # and let the authoritative final correct the text.
                        except Exception as exc:
                            await ws.send_json({
                                'type': 'error',
                                'error': str(exc)[:500],
                                'code': 'processing_error',
                            })
                            await runtime.reset()
                            previous = ''
                            received_samples = 0

                    elif event_type == 'input_audio_buffer.commit':
                        final = payload.get('final', False)
                        if not isinstance(final, bool):
                            await ws.send_json({
                                'type': 'error',
                                'error': 'commit.final must be a boolean',
                                'code': 'invalid_event',
                            })
                            continue
                        if not validated:
                            await ws.send_json({
                                'type': 'error',
                                'error': ('Model not validated. Make sure to validate the '
                                          'model by sending a session.update event.'),
                                'code': 'model_not_validated',
                            })
                            continue
                        if final:
                            started = time.perf_counter()
                            try:
                                text = await runtime.infer(runtime.adapter.finish)
                            except Exception as exc:
                                await ws.send_json({
                                    'type': 'error',
                                    'error': str(exc)[:500],
                                    'code': 'processing_error',
                                })
                                await runtime.reset()
                                previous = ''
                                received_samples = 0
                                continue
                            elapsed = time.perf_counter() - started
                            audio_seconds = received_samples / 16000
                            await runtime.reset()
                            await ws.send_json({
                                'type': 'transcription.done',
                                'text': text,
                                'usage': {'type': 'duration', 'seconds': round(audio_seconds, 3)},
                                'received_samples': received_samples,
                                'compute_seconds': round(elapsed, 3),
                            })
                            previous = ''
                            received_samples = 0
                        # A non-final commit is the protocol's generation marker;
                        # audio is processed incrementally as append events arrive.

                    else:
                        await ws.send_json({
                            'type': 'error',
                            'error': f'Unknown event type: {event_type}',
                            'code': 'unknown_event',
                        })

                elif message.type == WSMsgType.BINARY:
                    await ws.send_json({
                        'type': 'error',
                        'error': 'Binary messages are not accepted; send JSON events',
                        'code': 'invalid_event',
                    })
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            runtime.gate.release()
            runtime.stream_active = False
            try:
                await asyncio.shield(runtime.reset())
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        return ws

    app.router.add_get('/health', health)
    app.router.add_get('/v1/models', models)
    app.router.add_post('/v1/audio/transcriptions', transcribe)
    app.router.add_get('/v1/realtime', realtime)
    return app


def main():
    config = Config.from_env()
    app = create_app(config)
    web.run_app(app, host=config.host, port=config.port, access_log=None)


if __name__ == '__main__':
    main()
