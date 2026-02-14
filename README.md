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
|- handler.py
|- acestep/
|- scripts/
|  |- hf_clone.py
|  |- endpoint/
|  |  |- generate_interactive.py
|  |  |- test.ps1
|  |  |- test.bat
|  |  |- test_rnb.bat
|  |  `- test_rnb_2min.bat
|  `- jobs/
|     `- submit_hf_lora_job.ps1
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
- Endpoint deployment: `docs/deploy/ENDPOINT.md`
- Additional guides: `docs/guides/qwen2-audio-train.md`

## Open-source readiness checklist

- Secrets are env-driven (`HF_TOKEN`, `HF_ENDPOINT_URL`, `.env`).
- Local artifacts are ignored via `.gitignore`.
- MIT license included.
- Reproducible clone/deploy paths documented.
