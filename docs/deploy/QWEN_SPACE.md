# Deploy Qwen Captioning UI To HF Space

This deploys the music-captioning app (`qwen_caption_app.py`) as its own Space.

## Prerequisites

- Hugging Face account
- `HF_TOKEN` with write access

## Steps

1. Create a new Hugging Face Space (SDK: `Gradio`).
2. Push this repo content to that Space.
3. In Space `README.md` front matter, set:
   - `sdk: gradio`
   - `app_file: qwen_caption_app.py`
4. Pick GPU hardware (A10G or better recommended for local backend).
5. Optional secrets/env:
   - `HF_TOKEN` (if accessing private datasets or endpoint backend)

## Runtime notes

- `local` backend loads `Qwen/Qwen2-Audio-7B-Instruct` in the Space runtime.
- `hf_endpoint` backend can call a dedicated endpoint URL instead.
- Export defaults to `/data/qwen_annotations` on Spaces when persistent storage is enabled.

