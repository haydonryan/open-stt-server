# open-stt-server

A self-hosted, OpenAI Whisper-compatible speech-to-text API server written in Rust. It exposes the same `/v1/audio/transcriptions` endpoint as the OpenAI API, making it a drop-in replacement for any client that already speaks that protocol.

Models are loaded once at startup and kept resident in memory for low-latency inference. All model weights are downloaded automatically from HuggingFace Hub on first run and cached locally.

---

## Credits

The model loading and inference code is adapted from [**super-stt**](https://github.com/jorge-menjivar/super-stt) by **Jorge Menjivar**. Super-stt is a high-performance speech-to-text daemon for the COSMIC desktop environment that uses the same Candle ML framework and supports the same model families. This project extracts that inference engine and wraps it in a standard HTTP API server.

---

## Supported Models

| Model | HuggingFace ID |
|---|---|
| `whisper-tiny` | openai/whisper-tiny |
| `whisper-tiny.en` | openai/whisper-tiny.en |
| `whisper-base` | openai/whisper-base |
| `whisper-base.en` | openai/whisper-base.en |
| `whisper-small` | openai/whisper-small |
| `whisper-small.en` | openai/whisper-small.en |
| `whisper-medium` | openai/whisper-medium |
| `whisper-medium.en` | openai/whisper-medium.en |
| `whisper-large` | openai/whisper-large |
| `whisper-large-v2` | openai/whisper-large-v2 |
| `whisper-large-v3` | openai/whisper-large-v3 |
| `whisper-large-v3-turbo` | openai/whisper-large-v3-turbo |
| `whisper-distil-medium.en` | distil-whisper/distil-medium.en |
| `whisper-distil-large-v2` | distil-whisper/distil-large-v2 |
| `whisper-distil-large-v3` | distil-whisper/distil-large-v3 |
| `voxtral-mini` | mistralai/Voxtral-Mini-3B-2507 |
| `voxtral-small` | mistralai/Voxtral-Small-24B-2507 |

The alias `whisper-1` is also accepted and maps to `whisper-tiny` for OpenAI client compatibility.

---

## API

### `GET /health`

Returns `{"status":"ok"}` when the server is ready.

### `GET /v1/models`

Lists all loaded models in OpenAI format.

```json
{
  "object": "list",
  "data": [
    { "id": "whisper-base", "object": "model", "owned_by": "open-stt-server" }
  ]
}
```

### `POST /v1/audio/transcriptions`

Transcribe an audio file. Accepts `multipart/form-data`.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | yes | Audio file (WAV, MP3, FLAC, OGG, M4A, …) |
| `model` | string | no | Model name. Defaults to the configured default model. |
| `response_format` | string | no | `json` (default) or `text` |
| `language` | string | no | Accepted but currently unused |
| `prompt` | string | no | Accepted but currently unused |
| `temperature` | float | no | Accepted but currently unused |

**Response (`json`):**

```json
{ "text": "The transcribed text." }
```

**Response (`text`):**

```
The transcribed text.
```

**Example with curl:**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-base"
```

**Example with the OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",  # required by the SDK but ignored if no key is configured
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-base", file=f)

print(result.text)
```

---

## Configuration

All options are available as CLI flags and environment variables.

| Flag | Env Var | Default | Description |
|---|---|---|---|
| `--port` | `OPEN_STT_PORT` | `8080` | Port to listen on |
| `--model` | `OPEN_STT_MODELS` | *(required)* | Model(s) to load. Comma-separated in env var, repeated flag on CLI. |
| `--default-model` | `OPEN_STT_DEFAULT_MODEL` | first model | Model used when the request does not specify one |
| `--force-cpu` | `OPEN_STT_FORCE_CPU` | `false` | Disable CUDA even if available |
| `--download` | `OPEN_STT_DOWNLOAD` | `false` | Download missing model files on startup |
| `--api-key` | `OPEN_STT_API_KEY` | *(none)* | If set, all requests must include `Authorization: Bearer <key>` |
| | `RUST_LOG` | `info` | Log level (`error`, `warn`, `info`, `debug`, `trace`) |

The server always binds to `0.0.0.0`.

### Loading multiple models

```bash
# CLI
open-stt-server --model whisper-base --model whisper-large-v3 --default-model whisper-base

# Environment variable
OPEN_STT_MODELS=whisper-base,whisper-large-v3 open-stt-server
```

---

## Running Natively

### Prerequisites

- Rust 1.82+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Build

```bash
# CPU only
cargo build --release

# With CUDA support
cargo build --release --features cuda

# With CUDA + cuDNN
cargo build --release --features cuda,cudnn

# With flash-attention (requires CUDA)
cargo build --release --features cuda,flash-attn
```

### Run

```bash
# Download whisper-base on first run, then serve
./target/release/open-stt-server --model whisper-base --download

# Serve with an API key
OPEN_STT_API_KEY=secret ./target/release/open-stt-server --model whisper-small --download

# Multiple models, custom port
./target/release/open-stt-server \
  --model whisper-tiny \
  --model whisper-base \
  --default-model whisper-base \
  --port 9000 \
  --download
```

---

## Docker

Two image variants are provided.

| Variant | Dockerfile | Base | Notes |
|---|---|---|---|
| Debian slim | `Dockerfile.debian` | `debian:bookworm-slim` | Best compatibility |
| Alpine | `Dockerfile.alpine` | `alpine:3.21` | Smaller final image |

### Build manually

```bash
# Debian
docker build -f Dockerfile.debian -t open-stt-server:debian .

# Alpine
docker build -f Dockerfile.alpine -t open-stt-server:alpine .
```

### Run manually

```bash
docker run -p 8080:8080 \
  -v hf_cache:/root/.cache/huggingface \
  -e OPEN_STT_MODELS=whisper-base \
  -e OPEN_STT_DOWNLOAD=true \
  open-stt-server:debian
```

### Docker Compose

A `docker-compose.yml` is included with both variants available as profiles.

```bash
# Start the Debian variant (default)
docker compose --profile default up

# Start the Alpine variant
docker compose --profile alpine up

# Override the model and port
OPEN_STT_MODELS=whisper-small OPEN_STT_PORT=9000 docker compose --profile default up
```

Create a `.env` file to persist your configuration:

```env
OPEN_STT_MODELS=whisper-base
OPEN_STT_PORT=8080
OPEN_STT_API_KEY=your-secret-key
RUST_LOG=info
```

Model weights are stored in a named Docker volume (`hf_cache`) and survive container restarts.

---

## Model Storage

Models are cached in the standard HuggingFace Hub layout at `~/.cache/huggingface/hub/` (or `/root/.cache/huggingface/hub/` inside Docker). Once downloaded, they are reused on subsequent starts without re-downloading.

Approximate download sizes:

| Model | Size |
|---|---|
| whisper-tiny | ~150 MB |
| whisper-base | ~290 MB |
| whisper-small | ~970 MB |
| whisper-medium | ~3 GB |
| whisper-large-v3 | ~6 GB |
| whisper-large-v3-turbo | ~3 GB |
| voxtral-mini | ~6 GB |
| voxtral-small | ~47 GB |

---

## License

MIT
