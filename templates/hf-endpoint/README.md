# ACE-Step Custom Endpoint Repo

This repo is intended for a Hugging Face **Dedicated Inference Endpoint** with a custom `handler.py`.

## Contents

- `handler.py`: Endpoint request/response logic.
- `acestep/`: Core inference utilities.
- `requirements.txt`: Python dependencies.
- `packages.txt`: System dependencies.

## Expected Request Payload

```json
{
  "inputs": {
    "prompt": "upbeat pop rap with emotional guitar",
    "lyrics": "[Verse] city lights and midnight rain",
    "duration_sec": 12,
    "sample_rate": 44100,
    "seed": 42,
    "guidance_scale": 7.0,
    "steps": 50,
    "use_lm": true
  }
}
```

## Quick Setup

1. Create a model repo on Hugging Face.
2. Push this folder content to that repo.
3. Create a new dedicated endpoint from this custom repo.
4. Set environment variables on the endpoint as needed:
   - `ACE_CONFIG_PATH` (default `acestep-v15-sft`)
   - `ACE_LM_MODEL_PATH` (default `acestep-5Hz-lm-4B`)
   - `ACE_DOWNLOAD_SOURCE` (`huggingface` or `modelscope`)
5. Scale down or pause when idle to control cost.
