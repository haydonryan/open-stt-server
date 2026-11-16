# Nemotron Speech Streaming backend

This repository adds a self-contained Python sidecar that serves the deployed
Voxtral-compatible ASR surface with NVIDIA Nemotron Speech Streaming 0.6B. The
model stays in the official NeMo cache-aware RNNT pipeline; the Rust Whisper
server remains unchanged and independent.

## Compatibility surface

The sidecar preserves the routes and client behavior used by FastClaw and the
existing ASR benchmark:

| Route | Behavior |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/models` | OpenAI list containing `whisper-1` and `nemotron-speech-streaming-en-0.6b` |
| `POST /v1/audio/transcriptions` | Multipart `file`, optional `model`, `response_format` (`json`/`text`), `language` (English only) |
| `GET /v1/realtime` (WebSocket upgrade) | Voxtral vLLM protocol: `session.created`, `session.update(model=whisper-1)`, `input_audio_buffer.commit`, `input_audio_buffer.append` (base64 PCM16), `transcription.delta`, `transcription.done`, `error` |

HTTP uploads are decoded by ffmpeg to mono 16 kHz PCM16, so WAV, MP3, FLAC, OGG,
M4A and other common containers work. JSON responses return `{"text": ...,
"usage": {"type": "duration", "seconds": ...}}`; `response_format=text` returns
plain text. The reported duration is the decoded audio duration.

Unsupported requests are rejected honestly rather than silently ignored:

- `response_format` other than `json` or `text` → HTTP 400 (`param=response_format`).
- `language` outside `en`, `english`, `en-us`, `en_us`, `auto` → HTTP 400 (`param=language`).
- `prompt`, `temperature`, `translate`, or any unknown multipart field → HTTP 400
  (`param=<field>`); the backend does not support prompt biasing, temperature
  sampling, or translation.
- Duplicate `file` fields → HTTP 400 (`param=file`).
- Non-file multipart fields are capped at `NEMOTRON_MAX_FIELD_BYTES` (4096 bytes)
  and the aggregate upload at `NEMOTRON_MAX_UPLOAD_MB`.

Realtime ingress accepts any even-length PCM16 chunk, including FastClaw's
3200-byte/100 ms chunks and the benchmark's 2560-byte/80 ms chunks. The server
splits each chunk into the adapter's native 80 ms blocks (≤1280 samples). The
adapter retains a bounded pending buffer (8960-sample native frame + 1280-sample
ingress block = 10240 samples), so memory does not grow with utterance length.

Partial text is emitted as append-only `transcription.delta` events before EOF.
A non-prefix model revision is not emitted as a delta; the authoritative
`transcription.done.text` is the only final transcript. This matches FastClaw's
append-only partial accumulation and strict final policy.

`transcription.done.usage.seconds` reports the input audio duration
(`received_samples / 16000`), while `compute_seconds` reports the inference time
of the finalization call. They are separate fields and are not interchangeable.

WebSocket events must be JSON objects; a non-object payload, a non-string model,
or a non-boolean `commit.final` returns an `error` event with code
`invalid_event`.

## Admission and lifecycle

- One active realtime stream per model process. A concurrent WebSocket upgrade
  receives HTTP 409 before upgrade.
- HTTP transcriptions are admitted through a bounded queue: at most
  `NEMOTRON_MAX_PENDING_HTTP` (default 2) requests may wait, and each request
  waits at most `NEMOTRON_GATE_TIMEOUT_S` (default 30 s) for the shared model
  gate. Excess queueing or gate waiting returns HTTP 409 before inference.
- ffmpeg decoding is bounded: stdout is read incrementally and capped at four
  times the upload budget, decoding is limited by `NEMOTRON_DECODE_TIMEOUT_S`
  (default 120 s), and the ffmpeg subprocess is killed and reaped on timeout,
  cancellation, or budget overrun. A bounded decode failure returns HTTP 400
  (budget) or HTTP 504 (timeout).
- Client disconnect, protocol error, or final completion resets the adapter and
  frees native encoder/feature slots. Cleanup runs on the model executor and is
  shielded against handler cancellation so disconnect cleanup still runs.
- An optional `NEMOTRON_API_KEY` enforces `Authorization: Bearer <key>` on HTTP
  and WebSocket requests.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `NEMOTRON_HOST` | `0.0.0.0` | Bind host |
| `NEMOTRON_PORT` | `8080` | Bind port (production compose uses `8000`) |
| `NEMOTRON_CHECKPOINT` | `/models/nemotron-speech-streaming-en-0.6b.nemo` | NeMo checkpoint path |
| `NEMOTRON_DEVICE` | `cpu` | `cpu` or `cuda`; CUDA never falls back to CPU |
| `NEMOTRON_THREADS` | `8` | Torch intra-op threads |
| `NEMOTRON_MAX_UPLOAD_MB` | `25` | Multipart/append audio bound; decoded PCM is capped at 4× this budget |
| `NEMOTRON_MODEL_ALIAS` | `whisper-1` | Default model alias |
| `NEMOTRON_MAX_PENDING_HTTP` | `2` | HTTP transcriptions allowed to wait on the model gate |
| `NEMOTRON_GATE_TIMEOUT_S` | `30` | Maximum seconds an HTTP transcription waits for the gate |
| `NEMOTRON_DECODE_TIMEOUT_S` | `120` | Maximum seconds for ffmpeg decode |
| `NEMOTRON_MAX_FIELD_BYTES` | `4096` | Bound for each non-file multipart field |
| `NEMOTRON_API_KEY` | unset | Optional bearer token |

The checkpoint is revision `ebe59e5a817142986528bbbee5dba8db7b38ed50` of
`nvidia/nemotron-speech-streaming-en-0.6b` (2.47 GB). CPU8 with FP32 weights is
the provisional default; CUDA uses BF16 weights.

## Build and run

```bash
just nemotron-build
# or
docker build -f Dockerfile.nemotron -t open-stt-server:nemotron .

docker run -d --name nemotron-asr-candidate \
  --cpus 8 --memory 12g \
  -e NEMOTRON_PORT=8000 \
  -p 10083:8000 \
  -v /mnt/warp-shared/huggingface-cache/hub/asr-benchmark-nemotron:/models:ro \
  open-stt-server:nemotron
```

The tracked compose profile mirrors the production runtime and can also serve
as an isolated candidate on a non-production host port:

```bash
NEMOTRON_HOST_PORT=10083 docker compose --profile nemotron up -d --force-recreate
# production-shaped profile (root-owned cutover):
just nemotron-deploy
```

The model directory is mounted read-only; it is not baked into the image. The
compose profile preserves the backend network alias `voxtral-realtime`, the
internal listen port `8000`, host port `10082`, both existing Traefik routers
(`realtime-speech.lab.hnrglobal.com`, `speech.lab.hnrglobal.com`), a Python
healthcheck against `/health`, and CPU/memory limits (`cpus: 8`, `mem_limit:
12g`).

## Verification

Focused regression suite (protocol, upload, errors, admission, disconnect,
decode bounds):

```bash
PYTHONPATH=. python -m unittest discover -s nemotron/tests -v
```

Candidate smoke:

```bash
curl -s http://127.0.0.1:10083/health
curl -s http://127.0.0.1:10083/v1/models
curl -s -X POST http://127.0.0.1:10083/v1/audio/transcriptions \
  -F 'file=@/path/audio.wav' -F 'model=whisper-1' -F 'response_format=json'
curl -s -X POST http://127.0.0.1:10083/v1/audio/transcriptions \
  -F 'file=@/path/audio.wav' -F 'model=whisper-1' -F 'response_format=text'
```

Paced realtime smoke with the existing benchmark client (unchanged client,
same float32→PCM16 conversion and authoritative final policy):

```bash
/home/claw/clawd/projects/asr-streaming-benchmark/.venv/bin/python \
  /home/claw/clawd/projects/asr-streaming-benchmark/live/voxtral/client.py \
  --url http://127.0.0.1:10083 \
  --manifest /home/claw/clawd/projects/asr-streaming-benchmark/data/manifest.jsonl \
  --limit 2 --output /tmp/nemotron-candidate.jsonl
```

Record the final image ID/digest (`docker inspect --format '{{.Id}}'`) rather
than relying on the mutable tag. Run `just test` and `just check` for the Rust
repository; they remain green because the Rust server is untouched.

### Final-image candidate record (2026-11-15)

- Image ID: `sha256:17417e021f3ad557d0a3499f5e68defda28fe88abf3575fa20da419135dab3a5`
  (`open-stt-server:nemotron`, built from the reviewed source).
- Candidate: `nemotron-asr-candidate` on host port `10083:8000` (internal
  `NEMOTRON_PORT=8000`), `--cpus 8`, `--memory 12g`, checkpoint mounted
  read-only.
- `GET /health` → `{"status":"ok"}`; `/v1/models` lists `whisper-1` and
  `nemotron-speech-streaming-en-0.6b`.
- Multipart JSON upload (12.64 s clip) → HTTP 200 in ~9.0 s, transcript text,
  `usage.seconds=12.64`; `response_format=text` → HTTP 200 in ~6.7 s plain text.
- Unsupported `response_format=srt` and `language=fr` → HTTP 400 with the
  documented `param`.
- Paced WebSocket client (unchanged benchmark client, 80 ms chunks, 2 clips):
  2/2 authoritative finals, 0 errors, 0 empty finals, partials before audio end
  for both clips; mean first partial 2.18 s, mean final after audio end 0.61 s;
  `transcription.done.usage.seconds` equal to clip duration (12.64, 7.92) and
  `compute_seconds` 0.337/0.343.
- Resources during paced run: ~5.16 GiB RSS (43% of the 12 GiB limit), peak
  ~655% CPU across the 8 allocated cores; idle ~0%.
- Compose healthcheck command verified inside the container against port 8000.
- Existing `voxtral-realtime` container remained running and untouched
  (`0.0.0.0:10082->8000/tcp`).

## Production cutover (root-owned)

The deployed `voxtral-realtime` service remains untouched until root performs
the verified switch:

1. Rebuild the image after any source change and pin the explicit image ID/digest
   for the candidate.
2. Verify the candidate container's `/health`, `/v1/models`, multipart JSON/text,
   paced WebSocket partials/final, two successive sessions, and disconnect
   cleanup on its candidate port.
3. Stop Voxtral only at cutover, not for benchmarks. Root owns this step.
4. Point the production compose profile at the pinned image (`NEMOTRON_IMAGE`),
   keep host port `10082:8000`, backend alias `voxtral-realtime`, and the two
   Traefik routers; keep `model=whisper-1` and existing FastClaw config.
5. Run a post-cutover live check on both HTTP and WebSocket paths.

## Rollback

Restore the existing `voxtral-realtime` container configuration and Traefik
routing to the Voxtral service, then restart only the affected container. The
Nemotron candidate image and compose profile can remain built for later retry.
No model weights or data are destroyed by rollback.

## Limitations

- English streaming model only; automatic EOU, ITN, NMT, prompt biasing,
  temperature sampling, and translation are disabled or rejected.
- One active stream per process; concurrent WebSocket streams are rejected with
  HTTP 409. HTTP requests queue behind the model gate with a bounded wait.
- CPU8 is slower than the Voxtral GPU service; CPU peak RSS is dominated by the
  2.47 GB checkpoint plus NeMo/PyTorch runtime. CUDA BF16 uses ~5.5 GB process
  VRAM with a higher startup allocation while restoring the checkpoint.
- Partial deltas are append-only; non-prefix revisions are suppressed and
  corrected only by the authoritative final.
- Silent audio can yield an empty authoritative final; the protocol still sends
  `transcription.done`.
