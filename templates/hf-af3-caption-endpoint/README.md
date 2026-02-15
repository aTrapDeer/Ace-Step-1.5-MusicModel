# Audio Flamingo 3 Caption Endpoint Template

Use this as a custom `handler.py` runtime for a Hugging Face Dedicated Endpoint.

## Request contract

```json
{
  "inputs": {
    "prompt": "Analyze this full song and summarize arrangement changes.",
    "audio_base64": "<base64-encoded WAV bytes>",
    "max_new_tokens": 1200,
    "temperature": 0.1
  }
}
```

## Response contract

```json
{
  "generated_text": "..."
}
```

## Setup

Fastest path from this repo:

```bash
python scripts/hf_clone.py af3-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_ENDPOINT_REPO
```

Then deploy a Dedicated Endpoint from that model repo.

Important: make sure your endpoint repo contains top-level:
- `handler.py`
- `requirements.txt`
- `README.md`

Use endpoint task `custom` so the runtime loads `handler.py` instead of a default Transformers pipeline.

## Endpoint env vars

Required:
- `AF3_MODEL_ID=nvidia/audio-flamingo-3-hf`

Optional runtime bootstrap (defaults shown):
- `AF3_BOOTSTRAP_RUNTIME=1`
- `AF3_TRANSFORMERS_SPEC=transformers==5.1.0`
- `AF3_RUNTIME_DIR=/tmp/af3_runtime`
- `AF3_STUB_TORCHVISION=1`

## Notes

- Audio Flamingo 3 is large; use a GPU endpoint.
- First boot can take longer because the handler installs AF3-compatible runtime dependencies.
- This handler returns raw prose analysis. Use the local AF3+ChatGPT pipeline to normalize to LoRA sidecar JSON.
