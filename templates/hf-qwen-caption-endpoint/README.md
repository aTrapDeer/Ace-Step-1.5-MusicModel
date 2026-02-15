# Qwen2-Audio Caption Endpoint Template

Use this as a custom `handler.py` runtime for a Hugging Face Dedicated Endpoint.

## Request contract

```json
{
  "inputs": {
    "prompt": "Analyze and describe this music segment.",
    "audio_base64": "<base64-encoded WAV bytes>",
    "sample_rate": 16000,
    "max_new_tokens": 384,
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

Fastest way from this repo:

```bash
python scripts/hf_clone.py qwen-endpoint --repo-id YOUR_USERNAME/YOUR_QWEN_ENDPOINT_REPO
```

Then deploy a Dedicated Endpoint from that repo with task `custom`.

Manual path:

1. Create a new model repo for your endpoint runtime.
2. Copy `handler.py` from this folder into that repo as top-level `handler.py`.
3. Add a `requirements.txt` containing at least:
   - `torch`
   - `torchaudio`
   - `transformers>=4.53.0,<4.58.0`
   - `soundfile`
   - `numpy`
4. Deploy a Dedicated Endpoint from that repo.
5. Optional endpoint env var:
   - `QWEN_MODEL_ID=Qwen/Qwen2-Audio-7B-Instruct`

Then point `qwen_caption_app.py` backend `hf_endpoint` at that endpoint URL.

## Quick local test script

From this repo:

```bash
python scripts/endpoint/test_qwen_caption_endpoint.py \
  --url https://YOUR_ENDPOINT.endpoints.huggingface.cloud \
  --token hf_xxx \
  --audio path/to/song.wav
```
