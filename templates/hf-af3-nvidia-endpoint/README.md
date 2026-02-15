# Audio Flamingo 3 NVIDIA-Stack Endpoint Template

This template uses the same core runtime pattern as NVIDIA's Space:
- `llava` code from `nvidia/audio-flamingo-3` (space repo)
- base checkpoint from `nvidia/audio-flamingo-3` (model repo)
- optional `stage35` think/long adapter

## Request contract

```json
{
  "inputs": {
    "prompt": "Please describe the audio in detail.",
    "audio_base64": "<base64 WAV bytes>",
    "think_mode": true,
    "max_new_tokens": 2048,
    "temperature": 0.2
  }
}
```

## Response contract

```json
{
  "generated_text": "...",
  "mode": "think"
}
```

## Bootstrap command

```bash
python scripts/hf_clone.py af3-nvidia-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_NVIDIA_ENDPOINT_REPO
```

## Endpoint settings

- Task: `custom`
- GPU instance required
- Secrets:
  - `HF_TOKEN=<your_token>`

## Optional env vars

- `AF3_NV_CODE_REPO_ID=nvidia/audio-flamingo-3`
- `AF3_NV_MODEL_REPO_ID=nvidia/audio-flamingo-3`
- `AF3_NV_CODE_REPO_TYPE=space`
- `AF3_NV_MODEL_REPO_TYPE=model`
- `AF3_NV_DEFAULT_MODE=think`
- `AF3_NV_LOAD_THINK=1`
- `AF3_NV_LOAD_SINGLE=0`

Default behavior loads think/long mode for higher-quality long-form reasoning.
