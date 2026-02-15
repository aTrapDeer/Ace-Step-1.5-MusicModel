# Deploy AF3 NVIDIA-Stack Endpoint (Space-Parity Runtime)

This path uses NVIDIA's `llava` stack + `stage35` think adapter, which matches the quality profile of:
- `https://huggingface.co/spaces/nvidia/audio-flamingo-3`

## 1) Create endpoint runtime repo

```bash
python scripts/hf_clone.py af3-nvidia-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_NVIDIA_ENDPOINT_REPO
```

This pushes:
- `handler.py`
- `requirements.txt`
- `README.md`

from `templates/hf-af3-nvidia-endpoint/`.

## 2) Create Dedicated Endpoint

1. Create endpoint from `YOUR_USERNAME/YOUR_AF3_NVIDIA_ENDPOINT_REPO`.
2. Set task to `custom`.
3. Use a GPU instance.
4. Add secret:
   - `HF_TOKEN=hf_xxx`

## 3) Recommended endpoint env vars

- `AF3_NV_DEFAULT_MODE=think`
- `AF3_NV_LOAD_THINK=1`
- `AF3_NV_LOAD_SINGLE=0`
- `AF3_NV_CODE_REPO_ID=nvidia/audio-flamingo-3`
- `AF3_NV_MODEL_REPO_ID=nvidia/audio-flamingo-3`

## 4) Request shape from local scripts

Current scripts send:

```json
{
  "inputs": {
    "prompt": "...",
    "audio_base64": "...",
    "max_new_tokens": 3200,
    "temperature": 0.2
  }
}
```

Optional extra flag for this endpoint:

```json
{
  "inputs": {
    "think_mode": true
  }
}
```

## 5) Notes

- First boot is slow because runtime deps + model artifacts must load.
- Keep at least one warm replica if you want consistent latency.
- This runtime is heavier than the HF-converted `audio-flamingo-3-hf` endpoint path.
