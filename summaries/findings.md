# ACE-Step 1.5 Annotation and LoRA Findings (My Notes)

[Back to project README](../README.md)

## What I was trying to build

I wanted a reliable pipeline to:

1. Analyze my songs with AF3/Qwen-style timestamped musical detail.
2. Clean and structure the results with ChatGPT.
3. Save sidecar JSON files that ACE-Step 1.5 LoRA training can consume directly.
4. Keep enough detail for future iteration (human edits, richer annotations, timeline/event work).

## What ACE-Step 1.5 actually reads during LoRA training

Based on this repo's loader (`lora_train.py`), the training loop directly reads these JSON keys:

- `caption`
- `lyrics`
- `bpm`
- `keyscale`
- `timesignature`
- `vocal_language`
- `duration`

Anything else is effectively extra metadata for my own workflow. This is why I moved rich analysis detail into `caption` so it is not ignored by the model.

## Endpoint stack comparison I observed

I tested two serving stacks on the same tracks/prompts.

### Stack A (lower quality)

- Model path: `nvidia/audio-flamingo-3-hf`
- Runtime style: generic Transformers path with custom endpoint handler
- Behavior I observed:
  - Often short outputs
  - Sometimes repetitive segment text
  - Less convincing section-by-section progression
- Latency I observed:
  - Fast short runs
  - Medium-length think runs

### Stack B (higher quality)

- Model path:
  - base: `nvidia/audio-flamingo-3`
  - think adapter: `stage35`
- Runtime style: NVIDIA-style `llava`/`generate_content` stack
- Behavior I observed:
  - Longer, richer timestamped prose
  - Better flow across sections
  - Better musical interaction detail (vocals + instruments + arrangement)
- Latency I observed:
  - Slower than Stack A
  - Roughly around 1 minute per track in think/long style runs

### My conclusion

If I care about annotation quality, Stack B is clearly better even if it is slower.

## Main issues I hit and how I resolved them

### 1) Endpoint failed with `Unknown task custom`

Observed error:

- `KeyError: "Unknown task custom ..."`

What caused it:

- Endpoint fell back to default pipeline path instead of loading my custom `handler.py`.
- Log showed: `No custom pipeline found at /repository/handler.py`.

Fix:

- Ensure endpoint repo has top-level `handler.py`.
- Deploy using the custom endpoint template files exactly.

### 2) AF3 architecture not recognized

Observed error:

- `model type audioflamingo3 not recognized`

What caused it:

- Endpoint base runtime had older Transformers stack that could not load AF3 model classes.

Fix:

- Bootstrap runtime dependencies compatible with AF3 in custom handler/template.
- Avoid relying on plain default endpoint image assumptions.

### 3) Processor load failures for HF-converted AF3 repo

Observed error:

- `Unrecognized processing class in nvidia/audio-flamingo-3-hf`

What caused it:

- Mismatch between model repo packaging and runtime loader expectations.

Fix:

- Move to NVIDIA stack template path and serving format that matches expected classes/runtime behavior.

### 4) Dependency conflicts after forced upgrades

Observed logs showed conflicts around:

- `transformers`
- `huggingface_hub`
- `torch`/`torchaudio`/`torchvision`
- `huggingface-inference-toolkit` pinned versions

What caused it:

- Upgrading one package in place inside endpoint image caused incompatibility with toolkit pins.

Fix:

- Use curated endpoint template/runtime setup instead of ad-hoc package upgrades.

### 5) Token/auth confusion

Observed warning:

- Unauthenticated requests to HF Hub even though I had a token in `.env`.

What caused it:

- Variable name mismatch (`hf_token` vs expected runtime env var names like `HF_TOKEN`) in some contexts.

Fix:

- Normalize env variable names and pass token consistently in endpoint/runtime settings.

### 6) Very short or repetitive analysis output

What caused it:

- Wrong stack path (HF-converted flow) and/or non-think-compatible runtime behavior.

Fix:

- Migrate to NVIDIA think-capable stack.
- Use longer token budgets and think-oriented prompt settings.

## Dataset run results and quality checks

### Batch throughput I observed

- 22 tracks processed in about 22 minutes.
- Roughly 60 seconds per track average.

### Repetition audit outcome

- No exact duplicate full outputs across tracks.
- But strong template reuse in phrasing and sentence structures.

Interpretation:

- The model output varied by track, but stylistically collapsed into repeated wording patterns.

## JSON shaping decisions I made

### Flattening for LoRA compatibility

I flattened each sidecar to core fields used by `lora_train.py`:

- `artist`, `caption`, `lyrics`, `bpm`, `keyscale`, `timesignature`, `vocal_language`, `duration`, `source`

### Keeping rich detail without losing trainability

I preserved detail under `source.rich_details` and then pushed high-value content into `caption` so training sees it.

### Global normalization applied

- `timesignature`: `"4"`
- `vocal_language`: `"en"`
- Captions prefixed with `Andrew Spacey:`

## Important remaining data limitations

Even after cleanup, these are still weak points in current sidecars:

- `bpm` is mostly null
- `keyscale` is mostly unknown/blank

These are optional for training, but adding reliable BPM/key would likely improve control and consistency.

## My current recommendation

1. Keep NVIDIA stack as default for annotation generation quality.
2. Keep core LoRA fields simple and valid.
3. Keep rich details in `source.rich_details` for traceability.
4. Keep detail-rich caption text for actual conditioning.
5. Add a BPM/key estimation pass next if I want stronger metadata conditioning.

## Next technical step I want

I should run a structured event pass (`events` list with start/end/type/intensity) on a subset first, then test whether event-aware captions improve generated song structure over the current caption-only approach.
