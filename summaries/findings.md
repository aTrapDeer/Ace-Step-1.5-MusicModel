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

## Early experiments worth running

- Compare baseline LoRA vs time-event LoRA on the same curated mini-dataset.
- Score structural accuracy (section order, transition timing tolerance).
- Run blind listening tests for perceived musical arc and arrangement coherence.
- Track whether time labels improve consistency without reducing creativity.

## Expected outcomes

If this works, this repo can evolve from "style-conditioned generation" toward "structure-aware generation":

- More intentional song progression.
- Stronger subgenre identity.
- Better controllability for creators.

This is still a baseline research note, but it gives a clear technical direction that fits the current project architecture.
