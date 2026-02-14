# Contributing

## Development Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

## Before Opening A PR

1. Keep secrets out of git (`HF_TOKEN`, endpoint URLs, `.env`).
2. Do not commit local artifacts (`checkpoints/`, `lora_output/`, generated audio).
3. Run quick CLI sanity checks:
   - `python lora_train.py --help`
   - `python scripts/hf_clone.py --help`
   - `python scripts/endpoint/generate_interactive.py --help`
4. Update docs (`README.md`, `docs/deploy/*`) if behavior or workflows changed.

## Scope Guidelines

- UI + training workflow changes belong in `lora_ui.py` / `lora_train.py`.
- Inference endpoint changes belong in `handler.py`.
- Shared ACE-Step runtime logic belongs in `acestep/`.
