from pathlib import Path

content = """# ACE-Step 1.5 Deployment Context (Hugging Face Endpoint) — Handoff Notes

## Objective
Set up `ACE-Step/Ace-Step1.5` on Hugging Face so we can:
1. Serve music generations through a private endpoint (token-protected),
2. Run on GPU (A100 preferred),
3. Control costs aggressively (scale-to-zero / pause when idle),
4. Call the endpoint from local scripts/app backend,
5. Transition from a sine-wave smoke test to real ACE-Step generations.

---

## Current State (What’s Already Done)

- Hugging Face auth is working in terminal (`hf auth` usable).
- A private dedicated endpoint exists:
  - `https://xr81s77sis7hoggq.us-east-1.aws.endpoints.huggingface.cloud`
- Endpoint is connected to a custom repo containing `handler.py`.
- Smoke-test `handler.py` was deployed and tested successfully:
  - Returns base64-encoded WAV generated as sine wave/noise.
- Local `.bat` + `.ps1` testing flow works to hit endpoint and save `.wav`.

---

## Key Constraint Discovered

`ACE-Step/Ace-Step1.5` is **not** a one-click “Model Catalog verified” endpoint deployment.
HF warning indicates:
- no verified config,
- missing `handler.py` if trying to deploy model repo directly.

### Implication
Use a **custom endpoint repo** (our own repo) with:
- `handler.py`
- `requirements.txt`
Then load `ACE-Step/Ace-Step1.5` from code at runtime.

---

## Important Product/Infra Notes

### ZeroGPU vs Dedicated Endpoints
- **ZeroGPU** applies to **Spaces** (good for demos/prototypes).
- For production-like API serving, use **Dedicated Inference Endpoints**.

### Idle Scaling
- In Dedicated Endpoints UI, minimum idle scale-to-zero window observed is **15 minutes**.
- Faster than 15 min is not available in current UI setting.
- To stop all billing immediately, use **Pause** endpoint.
- Scale-to-zero (min replicas 0) is good for auto-wake behavior with cold starts.

---

## Files in Custom Endpoint Repo (Expected)

- `handler.py`  -> custom inference logic
- `requirements.txt` -> runtime dependencies
- `README.md` -> optional docs/config context

---

## Smoke-Test Handler Behavior (Current)

Current handler:
- Does NOT load ACE-Step model.
- Generates synthetic audio via numpy sine + noise.
- Returns:
  - `audio_base64_wav`
  - `sample_rate`
  - `duration_sec`

This validated endpoint wiring, auth, request/response format, and client decode pipeline.

---

## What Needs to Happen Next (Critical Path)

## 1) Replace fallback generation with real ACE-Step inference

In `handler.py`:

- `__init__`:
  - Load ACE-Step pipeline/model once at container startup.
  - Use model source `ACE-Step/Ace-Step1.5`.
  - Move model to CUDA when available.

- `__call__`:
  - Parse request inputs:
    - `prompt`
    - `lyrics`
    - `duration_sec`
    - `sample_rate`
    - `seed`
    - optional: `guidance_scale`, `steps`, `use_lm`
  - Execute ACE-Step generation.
  - Convert output waveform to WAV bytes.
  - Return base64 WAV in JSON.

## 2) Ensure dependencies are correct in `requirements.txt`
At minimum for current scaffold:
- `numpy`
- `soundfile`

Likely needed for ACE runtime:
- `torch`
- `torchaudio`
- `transformers`
- `accelerate`
- `huggingface_hub`
- any ACE-Step-specific package requirements from ACE docs/repo.

## 3) Push repo changes and redeploy/rebuild endpoint
- `git add .`
- `git commit -m "..."`
- `git push`
- wait for endpoint rebuild healthy status.

## 4) Run generation call with real payload
Use same client script and include `prompt/lyrics`.

---

## Request Payload Contract (Target)

```json
{
  "inputs": {
    "prompt": "upbeat pop rap with emotional guitar",
    "lyrics": "[Verse] city lights and midnight rain",
    "duration_sec": 12,
    "sample_rate": 44100,
    "seed": 42,
    "guidance_scale": 7.0,
    "steps": 50,
    "use_lm": true
  }
}
