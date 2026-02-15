# Deploy Audio Flamingo 3 Caption Endpoint (Dedicated Endpoint)

Note: this guide is for the HF-converted `audio-flamingo-3-hf` runtime path.
For NVIDIA Space stack parity (`llava` + `stage35` think adapter), use:
`docs/deploy/AF3_NVIDIA_ENDPOINT.md`.

## 1) Create endpoint runtime repo

```bash
python scripts/hf_clone.py af3-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_ENDPOINT_REPO
```

This pushes:

- `handler.py`
- `requirements.txt`
- `README.md`

from `templates/hf-af3-caption-endpoint/`.

## 2) Create endpoint from that model repo

In Hugging Face Endpoints:

1. Create endpoint from `YOUR_USERNAME/YOUR_AF3_ENDPOINT_REPO`.
2. Choose a GPU instance.
3. Set task to `custom`.
4. Set env vars:
   - `AF3_MODEL_ID=nvidia/audio-flamingo-3-hf`
   - `AF3_BOOTSTRAP_RUNTIME=1`
   - `AF3_TRANSFORMERS_SPEC=transformers==5.1.0`

## 3) Validate startup

If logs contain:

- `No custom pipeline found at /repository/handler.py`

then `handler.py` is not in repo root. Re-upload the runtime template files.

If logs contain:

- `Failed to load AF3 processor classes after runtime bootstrap`

keep endpoint task as `custom`, then check that startup could install runtime deps (network + disk). The first cold start can take several minutes.

## 4) Connect from local pipeline

Set:

- `HF_AF3_ENDPOINT_URL`
- `HF_TOKEN`
- `OPENAI_API_KEY`

Recommended local `.env`:

```env
HF_AF3_ENDPOINT_URL=https://bc3r76slij67lskb.us-east-1.aws.endpoints.huggingface.cloud
HF_TOKEN=hf_xxx
OPENAI_API_KEY=sk-...
```

`.env` is git-ignored in this repo. Do not commit real credentials.

Then run:

```bash
python scripts/pipeline/run_af3_chatgpt_pipeline.py \
  --audio ./train-dataset/track.mp3 \
  --backend hf_endpoint \
  --endpoint-url "$HF_AF3_ENDPOINT_URL" \
  --openai-api-key "$OPENAI_API_KEY"
```

Or launch full GUI stack:

```bash
python af3_gui_app.py
```
