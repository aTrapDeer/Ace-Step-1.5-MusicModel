# Qwen2-Audio Captioning -> Human Refinement -> ACE-Step LoRA Dataset

This guide adds a full annotation pipeline around `Qwen/Qwen2-Audio-7B-Instruct` so you can:

1. Caption full songs with timestamped segment analysis.
2. Refine/expand annotations manually.
3. Export LoRA-ready sidecar JSON for ACE-Step 1.5 training.

## What was added

- Reusable captioning module: `qwen_audio_captioning.py`
- Gradio UI for upload/analyze/edit/export: `qwen_caption_app.py`
- Batch CLI for local/HF jobs: `scripts/annotations/qwen_caption_dataset.py`
- HF Job launcher for batch captioning: `scripts/jobs/submit_hf_qwen_caption_job.ps1`
- Optional endpoint handler template: `templates/hf-qwen-caption-endpoint/handler.py`

## Why use `Qwen2-Audio-7B-Instruct`

Use `Qwen/Qwen2-Audio-7B-Instruct` for this task because your prompt is instruction-heavy and structured (musical elements, mix/effects, vocals, and timestamped interactions).

## Default analysis prompt

The pipeline defaults to:

> Analyze and detail the musical elements, tones, instruments, genre and effects. Describe the effects and mix of instruments and vocals. Vocals may use modern production techniques such as pitch correction and tuning effects. Explain how musical elements interact throughout the song with timestamps. Go in depth on vocal performance and musical writing. Be concise but detail-rich.

You can override this in the UI or CLI.

## Run locally (recommended first)

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the captioning UI:

```bash
python qwen_caption_app.py
```

Open `http://localhost:7860`.

### UI flow

1. **Load Audio** tab:
   - Scan a folder and/or upload files.
2. **Run Qwen Captioning** tab:
   - Backend:
     - `local` (model runs in same app process), or
     - `hf_endpoint` (calls a remote endpoint URL).
   - Tune segmentation (`segment_seconds`, `overlap_seconds`) for timestamp granularity.
3. **Human Annotation + Export** tab:
   - Load JSON per track.
   - Manually refine timelines, instrument/mix notes, caption text.
   - Export sidecars + manifest.

## Run batch from CLI

Example local batch:

```bash
python scripts/annotations/qwen_caption_dataset.py \
  --dataset-dir ./dataset_inbox \
  --backend local \
  --model-id Qwen/Qwen2-Audio-7B-Instruct \
  --segment-seconds 30 \
  --overlap-seconds 2 \
  --max-new-tokens 384 \
  --temperature 0.1 \
  --output-dir ./qwen_annotations \
  --copy-audio
```

Sidecars are written next to each source audio file by default.  
Disable with `--no-write-inplace-sidecars`.

Outputs:

- `qwen_annotations/dataset/*.audio` (if `--copy-audio`)
- `qwen_annotations/dataset/*.json` (LoRA sidecars)
- `qwen_annotations/annotations_manifest.jsonl`
- `qwen_annotations/annotations_index.json`

## Run batch on Hugging Face Jobs

PowerShell:

```powershell
.\scripts\jobs\submit_hf_qwen_caption_job.ps1 `
  -CodeRepo "YOUR_USERNAME/ace-step-lora-studio" `
  -DatasetRepo "YOUR_USERNAME/YOUR_AUDIO_DATASET" `
  -ModelId "Qwen/Qwen2-Audio-7B-Instruct" `
  -Flavor "a10g-large" `
  -Timeout "8h" `
  -CopyAudio `
  -UploadRepo "YOUR_USERNAME/YOUR_ANNOTATED_DATASET"
```

## Use on Hugging Face Space

To run this UI as a dedicated Space app, set Space `README.md` front matter:

- `sdk: gradio`
- `app_file: qwen_caption_app.py`

Then push this repo content to that Space.

## Optional: remote endpoint backend

If you want local UI to call a remote endpoint:

1. Deploy dedicated endpoint runtime from this template:
   - `python scripts/hf_clone.py qwen-endpoint --repo-id YOUR_USERNAME/YOUR_QWEN_ENDPOINT_REPO`
2. In UI select `backend=hf_endpoint`.
3. Set endpoint URL + token.

## Sidecar schema and ACE-Step compatibility

The exported JSON keeps ACE-Step core fields:

- `caption`
- `lyrics`
- `bpm`
- `keyscale`
- `timesignature`
- `vocal_language`
- `duration`

And adds rich fields:

- `music_analysis.timeline` (timestamped segment notes)
- `music_analysis.instruments`, `effects`, `vocal_characteristics`, `mix_notes`
- `analysis_prompt`, `analysis_model`, `analysis_generated_at`

ACE-Step loader ignores unknown keys, so rich fields stay available for later refinement while training still works with core fields.

## Train ACE-Step LoRA from exported dataset

Local:

```bash
python lora_train.py \
  --dataset-dir ./qwen_annotations/dataset \
  --model-config acestep-v15-base \
  --device auto \
  --num-epochs 20 \
  --batch-size 1 \
  --grad-accum 1 \
  --output-dir ./lora_output
```

HF Job (existing script):

```powershell
.\scripts\jobs\submit_hf_lora_job.ps1 `
  -CodeRepo "YOUR_USERNAME/ace-step-lora-studio" `
  -DatasetRepo "YOUR_USERNAME/YOUR_ANNOTATED_DATASET" `
  -ModelConfig "acestep-v15-base"
```

## Recommended iterative loop

1. Auto-caption with segment timestamps.
2. Human refine 10-20% highest-impact tracks first.
3. Export updated sidecars.
4. Train LoRA.
5. Evaluate structural/timing control.
6. Feed findings back into prompt + schema refinements.
