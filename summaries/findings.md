# Improving ACE-Step LoRA with Time-Event-Based Annotation

[Back to project README](../README.md)

## Baseline context in this repo

This project already provides a solid end-to-end workflow:

- Train LoRA adapters with `lora_train.py` and the Gradio UI (`app.py`, `lora_ui.py`).
- Deploy generation through a custom endpoint runtime (`handler.py`, `acestep/`).
- Test prompts and lyrics quickly with endpoint client scripts in `scripts/endpoint/`.

Today, most conditioning in this pipeline is still global (caption, lyrics, BPM, key, tags). That is a strong baseline, but it does not explicitly teach *when* events happen inside a track.

## Core limitation

Current annotations usually describe *what* a song is, not *when* events occur. The model can learn style and texture, but temporal structure is weaker:

- Verse/chorus transitions are often less deliberate than human-produced songs.
- Build-ups, drops, or effect changes can feel averaged or blurred.
- Subgenre-specific arrangement timing is harder to reproduce consistently.

## Observed baseline behavior (working assumption)

From current prompt and endpoint testing workflows in this repo, the baseline appears to do best on:

- overall timbre/style conditioning from caption-like prompts,
- short-form motif continuity,
- broad genre direction.

The baseline appears weaker on:

- section-level planning across longer durations,
- predictable timing of transitions (intro/verse/chorus/bridge),
- reliable callback motifs that should reappear at known timestamps.

These are expected gaps for globally conditioned generation and provide a clear target for time-event experiments.

## Why time-event labels are promising

1. Better musical structure: teach the model where sections start/end and where key transitions occur.
2. Better genre fidelity: encode timing differences between styles that share similar instruments.
3. Better control at inference: allow prompting for both content and structure (what + when).

## Practical direction for this codebase

A useful next step is to extend the current sidecar metadata approach with optional timed events.

Example direction:

- Keep existing fields (`caption`, `lyrics`, `bpm`, etc.).
- Add an `events` list with event type + start/end times.
- Start with a small, high-quality subset before scaling.

Illustrative shape:

```json
{
  "caption": "emotional rnb pop with warm pads",
  "bpm": 92,
  "events": [
    {"type": "intro", "start": 0.0, "end": 8.0},
    {"type": "verse", "start": 8.0, "end": 32.0},
    {"type": "chorus", "start": 32.0, "end": 48.0}
  ]
}
```

Optional extension fields that may help later:

- `intensity` (0-1) per event,
- `instrument_focus` tags per section,
- `transition_type` (hard cut, riser, filtered handoff, etc.).

## Early experiments worth running

- Compare baseline LoRA vs time-event LoRA on the same curated mini-dataset.
- Score structural accuracy (section order, transition timing tolerance).
- Run blind listening tests for perceived musical arc and arrangement coherence.
- Track whether time labels improve consistency without reducing creativity.

## Suggested evaluation rubric (v1)

Use a simple shared scorecard to keep comparisons objective:

1. Structure match (0-5): generated section order vs target plan.
2. Timing adherence (0-5): transition timestamps within tolerance window.
3. Musical coherence (0-5): transitions feel intentional, not abrupt/noisy.
4. Genre fit (0-5): arrangement behavior matches requested subgenre.
5. Prompt fidelity (0-5): requested mood/style/lyrics alignment.

This makes iteration easier than relying only on subjective listening notes.

## Incremental execution plan

Phase 1: Data and schema

- Define the minimal `events` schema and annotation guidelines.
- Build a small seed set (for example 50-200 clips) with high label quality.

Phase 2: Training and ablation

- Train a baseline LoRA and an event-aware LoRA with matched settings.
- Run ablations (with/without `events`, coarse vs fine event types).

Phase 3: Inference controls

- Add optional event-aware controls in the UI and endpoint payload.
- Keep backward compatibility so existing prompts still work.

Phase 4: Evaluation and docs

- Publish scorecard results + examples.
- Document tradeoffs (quality, speed, annotation effort).

## Expected outcomes

If this works, this repo can evolve from "style-conditioned generation" toward "structure-aware generation":

- More intentional song progression.
- Stronger subgenre identity.
- Better controllability for creators.

This is still a baseline research note, but it gives a clear technical direction that fits the current project architecture.
