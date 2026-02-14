# Deploy Inference To Your Own HF Dedicated Endpoint

This guide deploys the custom `handler.py` inference runtime to a Hugging Face Dedicated Inference Endpoint.

## Prerequisites

- Hugging Face account
- `HF_TOKEN` with repo write access
- Dedicated Endpoint access on your HF plan

## 1) Create/Update Your Endpoint Repo

```bash
python scripts/hf_clone.py endpoint --repo-id YOUR_USERNAME/YOUR_ENDPOINT_REPO
```

This uploads:

- `handler.py`
- `acestep/`
- `requirements.txt`
- `packages.txt`
- endpoint-specific README template

## 2) Create Endpoint In HF UI

1. Go to **Inference Endpoints** -> **New endpoint**.
2. Select your custom model repo: `YOUR_USERNAME/YOUR_ENDPOINT_REPO`.
3. Choose GPU hardware.
4. Deploy.

## 3) Recommended Endpoint Environment Variables

- `ACE_CONFIG_PATH` (default: `acestep-v15-sft`)
- `ACE_LM_MODEL_PATH` (default: `acestep-5Hz-lm-4B`)
- `ACE_LM_BACKEND` (default: `pt`)
- `ACE_DOWNLOAD_SOURCE` (`huggingface` or `modelscope`)
- `ACE_ENABLE_FALLBACK` (`false` recommended for strict failure visibility)

## 4) Test The Endpoint

Set credentials:

```bash
# Linux/macOS
export HF_TOKEN=hf_xxx
export HF_ENDPOINT_URL=https://your-endpoint-url.endpoints.huggingface.cloud

# Windows PowerShell
$env:HF_TOKEN="hf_xxx"
$env:HF_ENDPOINT_URL="https://your-endpoint-url.endpoints.huggingface.cloud"
```

Test with:

- `python scripts/endpoint/generate_interactive.py`
- `scripts/endpoint/test.ps1`

## Request Contract

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

## Cost Control

- Use scale-to-zero for idle periods.
- Pause endpoint for immediate spend stop.
- Expect cold starts when scaled to zero.
