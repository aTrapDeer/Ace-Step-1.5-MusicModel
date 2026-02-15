#!/usr/bin/env python
"""
Annotate one audio file with Qwen2-Audio and save a sidecar JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qwen_audio_captioning import (
    DEFAULT_ANALYSIS_PROMPT,
    DEFAULT_LONG_ANALYSIS_PROMPT,
    DEFAULT_MODEL_ID,
    build_captioner,
    generate_track_annotation,
    read_prompt_file,
)


def read_dotenv_value(path: str, key: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate a single audio file with Qwen2-Audio")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--backend", default="hf_endpoint", choices=["local", "hf_endpoint"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--endpoint-url", default=os.getenv("HF_QWEN_ENDPOINT_URL", ""))
    parser.add_argument("--token", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--prompt", default=DEFAULT_ANALYSIS_PROMPT)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--include-long-analysis", action="store_true")
    parser.add_argument("--long-analysis-prompt", default=DEFAULT_LONG_ANALYSIS_PROMPT)
    parser.add_argument("--long-analysis-prompt-file", default="")
    parser.add_argument("--long-analysis-max-new-tokens", type=int, default=1200)
    parser.add_argument("--long-analysis-temperature", type=float, default=0.1)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--keep-raw-outputs", action="store_true")
    parser.add_argument("--output-json", default="", help="Output JSON path (default: audio sidecar)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    prompt = read_prompt_file(args.prompt_file) if args.prompt_file else args.prompt
    long_prompt = (
        read_prompt_file(args.long_analysis_prompt_file)
        if args.long_analysis_prompt_file
        else args.long_analysis_prompt
    )
    token = (
        args.token
        or os.getenv("HF_TOKEN", "")
        or read_dotenv_value(".env", "HF_TOKEN")
        or read_dotenv_value(".env", "hf_token")
    )

    captioner = build_captioner(
        backend=args.backend,
        model_id=args.model_id,
        endpoint_url=args.endpoint_url,
        token=token,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    sidecar = generate_track_annotation(
        audio_path=str(audio_path),
        captioner=captioner,
        prompt=prompt,
        segment_seconds=float(args.segment_seconds),
        overlap_seconds=float(args.overlap_seconds),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        keep_raw_outputs=bool(args.keep_raw_outputs),
        include_long_analysis=bool(args.include_long_analysis),
        long_analysis_prompt=long_prompt,
        long_analysis_max_new_tokens=int(args.long_analysis_max_new_tokens),
        long_analysis_temperature=float(args.long_analysis_temperature),
    )

    out_path = Path(args.output_json) if args.output_json else audio_path.with_suffix(".json")
    out_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "saved_to": str(out_path),
                "caption": sidecar.get("caption", ""),
                "bpm": sidecar.get("bpm"),
                "keyscale": sidecar.get("keyscale", ""),
                "duration": sidecar.get("duration"),
                "segment_count": sidecar.get("music_analysis", {}).get("segment_count"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
