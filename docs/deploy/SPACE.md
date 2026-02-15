# Deploy LoRA Studio To Your Own HF Space

This guide deploys the full LoRA Studio UI to your own Hugging Face Space.
For the dedicated Qwen captioning UI, see `docs/deploy/QWEN_SPACE.md`.

## Prerequisites

- Hugging Face account
- `HF_TOKEN` with repo write access
- Python environment with `requirements.txt` installed

## Fast Path (Recommended)

```bash
python scripts/hf_clone.py space --repo-id YOUR_USERNAME/YOUR_SPACE_NAME
```

Optional private Space:

```bash
python scripts/hf_clone.py space --repo-id YOUR_USERNAME/YOUR_SPACE_NAME --private
```

## Manual Path

1. Create a new Space on Hugging Face:
   - SDK: `Gradio`
2. Push this repo content (excluding local artifacts) to that Space repo.
3. Ensure README front matter has:
   - `sdk: gradio`
   - `app_file: app.py`
4. In Space settings:
   - select GPU hardware (A10G/A100/etc.) if needed
   - add secrets (`HF_TOKEN`) if your flow requires private Hub access

## Runtime Notes

- Space output defaults to `/data/lora_output` on Hugging Face Spaces.
- Enable persistent storage if you need checkpoint retention across restarts.
- For long-running non-interactive training, HF Jobs may be more cost-efficient than keeping a Space running.
