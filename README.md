---
title: ACE-Step 1.5 LoRA Studio
emoji: music
colorFrom: blue
colorTo: teal
sdk: gradio
app_file: app.py
pinned: false
---

# ACE-Step 1.5 LoRA Studio
- Andrew Rapier

Train ACE-Step 1.5 LoRA adapters, deploy your own Hugging Face Space, and run production-style inference through a Dedicated Endpoint.

[![Create HF Space](https://img.shields.io/badge/Create-HF%20Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/new-space)
[![Create HF Endpoint Repo](https://img.shields.io/badge/Create-HF%20Endpoint%20Repo-FFB000?logo=huggingface&logoColor=black)](https://huggingface.co/new-model)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What you get

- LoRA training UI and workflow: `app.py`, `lora_ui.py`
- CLI LoRA trainer for local/HF datasets: `lora_train.py`
- Qwen2-Audio captioning/annotation pipeline: `qwen_caption_app.py`, `qwen_audio_captioning.py`, `scripts/annotations/`
- Audio Flamingo 3 + ChatGPT cleanup pipeline: `af3_chatgpt_pipeline.py`, `scripts/pipeline/`, `services/pipeline_api.py`
- React orchestration UI for AF3+ChatGPT: `react-ui/`
- Custom endpoint runtime: `handler.py`, `acestep/`
- Bootstrap automation for cloning into your HF account: `scripts/hf_clone.py`
- Endpoint test clients and HF job launcher: `scripts/endpoint/`, `scripts/jobs/`

## Quick start (local)

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

## End-to-end setup (recommended)

Use this sequence when setting up from scratch.

1. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Create local `.env` from `.env.example` and fill secrets

```env
HF_TOKEN=hf_xxx
HF_AF3_ENDPOINT_URL=https://YOUR_AF3_ENDPOINT.endpoints.huggingface.cloud
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini
AF3_MODEL_ID=nvidia/audio-flamingo-3-hf
```

3. Bootstrap your Hugging Face repos (Space + endpoint templates)

```bash
python scripts/hf_clone.py space --repo-id YOUR_USERNAME/YOUR_SPACE_NAME
python scripts/hf_clone.py af3-nvidia-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_NVIDIA_ENDPOINT_REPO
```

4. Deploy endpoint from the cloned AF3 NVIDIA endpoint repo

- Set endpoint task to `custom`.
- Confirm top-level `handler.py` exists in the endpoint repo.
- Set endpoint env vars if needed (`HF_TOKEN`, `AF3_NV_DEFAULT_MODE=think`).

5. Generate analysis sidecars from audio

```bash
python scripts/pipeline/run_af3_chatgpt_pipeline.py \
  --dataset-dir ./train-dataset \
  --backend hf_endpoint \
  --endpoint-url "$HF_AF3_ENDPOINT_URL" \
  --openai-api-key "$OPENAI_API_KEY"
```

6. Normalize existing JSONs into LoRA-ready shape (optional but recommended)

```bash
python scripts/pipeline/refine_dataset_json_with_openai.py \
  --dataset-dir ./train-dataset \
  --enable-web-search
```

This script keeps core fields needed by ACE-Step LoRA training and preserves rich analysis context in `source.rich_details`.

7. Train LoRA

```bash
python app.py
```

Then in UI:
- Load model.
- Scan/upload dataset.
- Start LoRA training.

8. Test generation with your new adapter

- Use the endpoint scripts in `scripts/endpoint/`.
- Or test through the Gradio UI flow.
- In **Step 4 - Evaluate**, you can now upload your own LoRA adapter (`.zip` or adapter files),
  then load it without retraining in this Space.

## AF3 GUI one-command startup

1. Configure `.env` (never commit this file):

```env
HF_TOKEN=hf_xxx
HF_AF3_ENDPOINT_URL=https://bc3r76slij67lskb.us-east-1.aws.endpoints.huggingface.cloud
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini
AF3_MODEL_ID=nvidia/audio-flamingo-3-hf
```

2. Launch API + GUI together:

```bash
python af3_gui_app.py
```

PowerShell alternative:

```powershell
.\scripts\dev\run_af3_gui.ps1
```

This command builds the React UI and serves it from the FastAPI backend.
Open `http://127.0.0.1:8008`.

## Clone to your HF account

Use the two buttons near the top of this README to create target repos in your HF account, then run:

Set token once:

```bash
# Linux/macOS
export HF_TOKEN=hf_xxx

# Windows PowerShell
$env:HF_TOKEN="hf_xxx"
```

Clone your own Space:

```bash
python scripts/hf_clone.py space --repo-id YOUR_USERNAME/YOUR_SPACE_NAME
```

Clone your own Endpoint repo:

```bash
python scripts/hf_clone.py endpoint --repo-id YOUR_USERNAME/YOUR_ENDPOINT_REPO
```

Clone a Qwen2-Audio caption endpoint repo:

```bash
python scripts/hf_clone.py qwen-endpoint --repo-id YOUR_USERNAME/YOUR_QWEN_ENDPOINT_REPO
```

Clone an Audio Flamingo 3 caption endpoint repo:

```bash
python scripts/hf_clone.py af3-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_ENDPOINT_REPO
```

When creating that endpoint, set task to `custom` so it loads the custom `handler.py`.

Clone an AF3 NVIDIA-stack endpoint repo (matches NVIDIA Space stack better):

```bash
python scripts/hf_clone.py af3-nvidia-endpoint --repo-id YOUR_USERNAME/YOUR_AF3_NVIDIA_ENDPOINT_REPO
```

Use this path when you want think/long quality behavior closer to NVIDIA's public demo.

Clone both in one run:

```bash
python scripts/hf_clone.py all \
  --space-repo-id YOUR_USERNAME/YOUR_SPACE_NAME \
  --endpoint-repo-id YOUR_USERNAME/YOUR_ENDPOINT_REPO
```

## Project layout

```text
.
|- app.py
|- lora_ui.py
|- lora_train.py
|- qwen_caption_app.py
|- qwen_audio_captioning.py
|- af3_chatgpt_pipeline.py
|- af3_gui_app.py
|- handler.py
|- acestep/
|- scripts/
|  |- hf_clone.py
|  |- dev/
|  |  |- run_af3_gui.py
|  |  `- run_af3_gui.ps1
|  |- annotations/
|  |  `- qwen_caption_dataset.py
|  |- pipeline/
|  |  `- run_af3_chatgpt_pipeline.py
|  |- endpoint/
|  |  |- generate_interactive.py
|  |  |- test.ps1
|  |  |- test.bat
|  |  |- test_rnb.bat
|  |  `- test_rnb_2min.bat
|  `- jobs/
|     `- submit_hf_lora_job.ps1
|     `- submit_hf_qwen_caption_job.ps1
|- services/
|  `- pipeline_api.py
|- react-ui/
|- utils/
|  `- env_config.py
|- docs/
|  |- deploy/
|  `- guides/
|- summaries/
|  `- findings.md
`- templates/hf-endpoint/
```

## Dataset format

Supported audio:

- `.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a`, `.aac`

Optional sidecar metadata per track:

- `song_001.wav`
- `song_001.json`

```json
{
  "caption": "melodic emotional rnb pop with warm pads",
  "lyrics": "[Verse]\\n...",
  "bpm": 92,
  "keyscale": "Am",
  "timesignature": "4/4",
  "vocal_language": "en",
  "duration": 120
}
```

## Qwen2-Audio annotation pipeline (music captioning)

Run the dedicated annotation UI:

```bash
python qwen_caption_app.py
```

Batch caption from CLI:

```bash
python scripts/annotations/qwen_caption_dataset.py \
  --dataset-dir ./dataset_inbox \
  --backend local \
  --model-id Qwen/Qwen2-Audio-7B-Instruct \
  --output-dir ./qwen_annotations \
  --copy-audio
```

This also writes `.json` sidecars next to source audio by default for direct ACE-Step LoRA training.

Then train LoRA on the exported dataset:

```bash
python lora_train.py --dataset-dir ./qwen_annotations/dataset --model-config acestep-v15-base
```

## Audio Flamingo 3 + ChatGPT pipeline (analysis -> normalized sidecar JSON)

This stack runs:

1. Audio Flamingo 3 for raw music analysis prose.
2. ChatGPT for cleanup/normalization into LoRA-ready fields.
3. Sidecar JSON export next to each audio file (or in a custom output folder).

CLI single track:

```bash
python scripts/pipeline/run_af3_chatgpt_pipeline.py \
  --audio "./train-dataset/Andrew Spacey - Wonder (Prod Beat It AT).mp3" \
  --backend hf_endpoint \
  --endpoint-url "$HF_AF3_ENDPOINT_URL" \
  --hf-token "$HF_TOKEN" \
  --openai-api-key "$OPENAI_API_KEY" \
  --artist-name "Andrew Spacey" \
  --track-name "Wonder"
```

CLI dataset batch:

```bash
python scripts/pipeline/run_af3_chatgpt_pipeline.py \
  --dataset-dir ./train-dataset \
  --backend hf_endpoint \
  --endpoint-url "$HF_AF3_ENDPOINT_URL" \
  --openai-api-key "$OPENAI_API_KEY"
```

Refine already-generated JSON files in place:

```bash
python scripts/pipeline/refine_dataset_json_with_openai.py \
  --dataset-dir ./train-dataset \
  --enable-web-search
```

Write refined files to a separate folder:

```bash
python scripts/pipeline/refine_dataset_json_with_openai.py \
  --dataset-dir ./train-dataset \
  --recursive \
  --enable-web-search \
  --output-dir ./train-dataset-refined
```

Single-command GUI (recommended):

```bash
python af3_gui_app.py
```

Manual API + React UI:

```bash
uvicorn services.pipeline_api:app --host 0.0.0.0 --port 8008 --reload
```

```bash
cd react-ui
npm install
npm run dev
```

Open `http://localhost:5173` (manual) or `http://127.0.0.1:8008` (single-command).

## Endpoint testing

```bash
python scripts/endpoint/generate_interactive.py
```

Or run scripted tests:

- `scripts/endpoint/test.ps1`
- `scripts/endpoint/test.bat`

## Findings and notes

Current baseline analysis and improvement ideas are tracked in:

- `summaries/findings.md`

## Docs

- Space deployment: `docs/deploy/SPACE.md`
- Qwen caption Space deployment: `docs/deploy/QWEN_SPACE.md`
- Endpoint deployment: `docs/deploy/ENDPOINT.md`
- AF3 endpoint deployment: `docs/deploy/AF3_ENDPOINT.md`
- AF3 NVIDIA-stack endpoint deployment: `docs/deploy/AF3_NVIDIA_ENDPOINT.md`
- Additional guides: `docs/guides/qwen2-audio-train.md`, `docs/guides/af3-chatgpt-pipeline.md`

## Open-source readiness checklist

- Secrets are env-driven (`HF_TOKEN`, `HF_AF3_ENDPOINT_URL`, `OPENAI_API_KEY`, `.env`).
- Local artifacts are ignored via `.gitignore`.
- MIT license included.
- Reproducible clone/deploy paths documented.
- `.env` is git-ignored; keep real credentials only in local `.env`.

## GitHub publish flow

1. Check status

```bash
git status
```

2. Stage and commit

```bash
git add .
git commit -m "Consolidate AF3/Qwen pipelines, endpoint templates, and docs"
```

3. Push to GitHub remote

```bash
git push github main
```
