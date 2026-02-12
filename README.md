---
title: ACE-Step 1.5 LoRA Studio
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# ACE-Step 1.5 LoRA Studio (HF Space)

Train and evaluate ACE-Step 1.5 LoRA adapters from a Gradio UI.

## Space Setup

1. Create a new **Gradio Space** on Hugging Face.
2. Push this repo to that Space repo.
3. Select a **GPU hardware** flavor in Space settings (A10G/A100/etc.).
4. Outputs default to `/data/lora_output` on Spaces.
   - If persistent storage is available on your plan/tier, enable it for checkpoint retention.

## Run Locally

```bash
python -m pip install -r requirements.txt
python app.py
```

## Data Format

Put audio files in a folder or drag/drop them in the UI.

Supported audio extensions:
- `.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a`, `.aac`

Optional sidecar metadata file with same basename:

- `track001.wav`
- `track001.json`

Example sidecar:

```json
{
  "caption": "melodic emotional rnb pop with warm pads",
  "lyrics": "[Verse]\n...",
  "bpm": 92,
  "keyscale": "Am",
  "timesignature": "4/4",
  "vocal_language": "en",
  "duration": 120
}
```

If sidecars are missing, use Step 2 **Auto-Label All** in the UI.

## Local Calls To Your Space

This UI exposes named API endpoints (e.g. `/generate_sample`, `/ab_test`, `/start_training`).

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
print(client.view_api())

# Example: single generation
audio_path, status = client.predict(
    "upbeat pop rock with electric guitar",  # prompt
    "[Instrumental]",                        # lyrics
    30,                                      # duration
    0,                                       # bpm (0=auto)
    8,                                       # steps
    7.0,                                     # guidance
    42,                                      # seed
    True,                                    # use_lora
    1.0,                                     # lora_scale
    api_name="/generate_sample",
)
print(status, audio_path)
```

## Notes

- Inference Endpoints are best for serving; Spaces/Jobs are better for interactive or long-running training.
- If you want headless scheduled training on HF GPUs, use `lora_train.py` CLI with HF Jobs.
