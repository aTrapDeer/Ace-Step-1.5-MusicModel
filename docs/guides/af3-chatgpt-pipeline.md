# Audio Flamingo 3 + ChatGPT Pipeline (Local Orchestration)

This guide sets up a cloud-first annotation workflow:

1. **Audio Flamingo 3** generates raw audio analysis text.
2. **ChatGPT** cleans and structures that output into Ace Step 1.5 LoRA sidecar JSON.
3. Optional human edits are applied before LoRA training.

## Endpoint vs Space

For 100+ tracks, use an **HF Dedicated Endpoint** for AF3 inference.

- Endpoint: production API, autoscaling options, stable URL, easier local integration.
- Space: better for interactive demos/tools, less ideal for bulk API workloads.

Use a Space only if you want a hosted UI. Keep heavy batch inference on Endpoint.

## Files in this repo

- Pipeline core: `af3_chatgpt_pipeline.py`
- Batch CLI: `scripts/pipeline/run_af3_chatgpt_pipeline.py`
- Local API: `services/pipeline_api.py`
- React UI: `react-ui/`
- AF3 endpoint template: `templates/hf-af3-caption-endpoint/`

## 1) Deploy AF3 endpoint

Create/push endpoint runtime repo:

```bash
python scripts/hf_clone.py af3-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_ENDPOINT_REPO
```

If you want NVIDIA Space parity (llava + stage35 think adapter), use:

```bash
python scripts/hf_clone.py af3-nvidia-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_NVIDIA_ENDPOINT_REPO
```

Then create a Hugging Face Dedicated Endpoint from that model repo.

If startup logs show:

- `No custom pipeline found at /repository/handler.py`

your repo root is missing `handler.py` (copy from `templates/hf-af3-caption-endpoint/handler.py`).

## 2) Configure env

Set values in `.env` (or shell env vars):

```env
HF_TOKEN=hf_xxx
HF_AF3_ENDPOINT_URL=https://bc3r76slij67lskb.us-east-1.aws.endpoints.huggingface.cloud
AF3_MODEL_ID=nvidia/audio-flamingo-3-hf
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini
```

`.env` is git-ignored by default. Keep all real secrets in local `.env` only.

## 3) Run one track from CLI

```bash
python scripts/pipeline/run_af3_chatgpt_pipeline.py \
  --audio "E:/Coding/hf-music-gen/train-dataset/Andrew Spacey - Wonder (Prod Beat It AT).mp3" \
  --backend hf_endpoint \
  --endpoint-url "$HF_AF3_ENDPOINT_URL" \
  --hf-token "$HF_TOKEN" \
  --openai-api-key "$OPENAI_API_KEY" \
  --artist-name "Andrew Spacey" \
  --track-name "Wonder"
```

Default behavior writes JSON next to the audio file (`same_stem.json`).

## 4) Batch all tracks

```bash
python scripts/pipeline/run_af3_chatgpt_pipeline.py \
  --dataset-dir ./train-dataset \
  --backend hf_endpoint \
  --endpoint-url "$HF_AF3_ENDPOINT_URL" \
  --openai-api-key "$OPENAI_API_KEY" \
  --enable-web-search
```

Use `--output-dir` if you want sidecars in a separate folder.

## 5) Run GUI stack

One command (recommended):

```bash
python af3_gui_app.py
```

This builds React and serves it from FastAPI. Open `http://127.0.0.1:8008`.

PowerShell:

```powershell
.\scripts\dev\run_af3_gui.ps1
```

Manual mode:

```bash
uvicorn services.pipeline_api:app --host 0.0.0.0 --port 8008 --reload

cd react-ui
npm install
npm run dev
```

Open `http://localhost:5173`.

UI supports:

- Local file path mode or upload mode
- AF3 backend toggle (`hf_endpoint` or `local`)
- Optional user context
- Optional web-search-enhanced ChatGPT cleanup
- Artist/track hints for better metadata normalization

## 6) Human-in-the-loop refinement

Recommended loop:

1. Generate sidecars with AF3+ChatGPT.
2. Review/edit core fields (`caption`, `bpm`, `keyscale`, `timesignature`, `duration`).
3. Keep rich analysis fields for traceability.
4. Train LoRA with `lora_train.py` on the folder containing audio + JSON sidecars.

## Output compatibility

The pipeline keeps Ace Step core sidecar fields:

- `caption`
- `lyrics`
- `bpm`
- `keyscale`
- `timesignature`
- `vocal_language`
- `duration`

And adds richer analysis fields in `music_analysis` + `pipeline` for auditability.

## Note on "guarantee"

No model can guarantee perfect music metadata. This pipeline improves reliability by:

- Schema-constrained ChatGPT output
- Normalization/defaulting in `build_lora_sidecar(...)`
- Optional human review pass before training
