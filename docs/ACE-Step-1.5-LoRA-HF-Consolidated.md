# ACE-Step 1.5 LoRA Pipeline (Simple + HF Spaces)

Last updated: 2026-02-12

## 1. What is already implemented in this repo
- Drag/drop dataset loading and folder scan.
- Optional per-track sidecar JSON (`song.wav` + `song.json`).
- New **Auto-Label All** option in `lora_ui.py`:
  - Uses ACE audio understanding (`audio -> semantic codes -> caption/lyrics/metadata`).
  - Writes/updates sidecar JSON for each track.
- LoRA training with ACE flow-matching defaults and adapter checkpoints.
- Training log now shows device plus elapsed time and ETA.

## 2. Direct answers to your core questions

### Is LoRA using HF GPU?
Yes, if the Space hardware is GPU and model device is `auto`/`cuda`, training runs on that Space GPU.

### Do we get time estimates?
Yes. The training status now shows elapsed time and ETA in the log.

### How are metadata and lyrics paired per song?
By basename in the same folder:
- `track01.wav`
- `track01.json`

### Do you need all metadata?
No. In this pipeline, metadata is optional.
- Required minimum: audio files.
- Strongly recommended: `caption` and/or `lyrics` for better conditioning quality.
- Optional but helpful: `bpm`, `keyscale`, `timesignature`, `vocal_language`, `duration`.

### Where are trained adapters saved?
- Local run: `lora_output/...` by default.
- HF Space run: `/data/lora_output/...` by default (as configured in UI code).
- Final adapter checkpoint is saved under a `final` subfolder.

### Cloud GPU + local files?
- Training on Spaces uses cloud GPU and writes artifacts to the Space filesystem.
- To keep results outside the Space, download them or upload to a Hub model repo.

### Can HF Endpoint GPU train this?
Not the right product. Inference Endpoints are for model serving/inference; use Spaces (interactive) or Jobs (batch) for training.

## 3. Minimal dataset format

Drop files into one folder:

```text
dataset_inbox/
  song_a.wav
  song_b.flac
  song_c.mp3
```

Optional sidecar for tighter control:

```text
dataset_inbox/
  song_a.wav
  song_a.json
```

Example `song_a.json`:

```json
{
  "caption": "emotional indie pop with airy female vocal and warm pads",
  "lyrics": "[Verse]\n...",
  "bpm": 96,
  "keyscale": "Am",
  "timesignature": "4/4",
  "vocal_language": "en",
  "duration": 120
}
```

## 4. Super simple training flow (UI)
1. Start UI:
   - Local: `python app.py`
   - Space: app starts automatically from `app.py`.
2. Step 1 tab: initialize `acestep-v15-base` (best LoRA plasticity).
3. Step 2 tab: scan folder or drag/drop files.
4. Optional: initialize auto-label LM and click **Auto-Label All**.
5. Step 3 tab: keep defaults for first run, click **Start Training**.
6. Click **Refresh Log** to monitor status/loss/ETA.
7. Step 4 tab: load adapter from output folder and A/B test against base.

## 5. HF Spaces setup (step by step)
1. Create a new Hugging Face **Space** with SDK = `Gradio`.
2. Push this repo to that Space repo.
3. Ensure Space metadata/front matter includes:
   - `sdk: gradio`
   - `app_file: app.py`
4. In Space `Settings -> Hardware`, select a GPU tier.
5. In Space `Settings -> Variables and secrets`, add any needed tokens as secrets (never hardcode).
6. Open the Space and run the 4-step UI flow.

## 6. GPU association and cost control

### Pick hardware for your stage
- Fast/cheap iteration: start with T4 or A10G.
- Heavier runs or bigger LM usage: A100/L40S/H100 class.

### Keep spend under control
1. Use smaller auto-label LM (`0.6B`) unless you need higher quality labels.
2. Train with `acestep-v15-base` only for final-quality runs; iterate on turbo variants if needed.
3. Pause or downgrade hardware immediately when idle.
4. Export/upload adapters right after training so you can shut hardware down.

### Current billing behavior to remember
HF Spaces docs indicate upgraded hardware is billed by minute while the Space is running, and you should pause/stop upgraded hardware when not in use.

## 7. Suggested first-run defaults
- Model: `acestep-v15-base`
- LoRA rank/alpha/dropout: `64 / 64 / 0.1`
- Optimizer: `adamw_8bit`
- LR: `1e-4`
- Warmup: `0.03`
- Scheduler: `constant_with_warmup`
- Shift: `3.0`
- Max grad norm: `1.0`

## 8. Source links (official)
- ACE-Step Gradio guide: https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5/blob/main/docs/en/GRADIO_GUIDE.md
- ACE-Step README: https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5/blob/main/README.md
- ACE-Step LoRA model card note (DiT-only LoRA): https://huggingface.co/ACE-Step/Ace-Step-v1.5-lo-ra-new-year
- HF Spaces overview: https://huggingface.co/docs/hub/en/spaces-overview
- HF Spaces GPU/hardware docs: https://huggingface.co/docs/hub/en/spaces-gpus
- HF Spaces config reference: https://huggingface.co/docs/hub/en/spaces-config-reference
- HF Inference Endpoints overview: https://huggingface.co/docs/inference-endpoints/en/index
