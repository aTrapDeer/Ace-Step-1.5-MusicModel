#!/usr/bin/env python
"""
Run AF3 -> ChatGPT cleanup pipeline on one file or a dataset folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from af3_chatgpt_pipeline import (
    DEFAULT_AF3_MODEL_ID,
    DEFAULT_AF3_PROMPT,
    DEFAULT_AF3_PROMPT_THINK_LONG,
    DEFAULT_OPENAI_MODEL,
    AF3EndpointClient,
    AF3LocalClient,
    run_af3_chatgpt_pipeline,
)
from qwen_audio_captioning import list_audio_files
from utils.env_config import get_env, load_project_env


def build_parser() -> argparse.ArgumentParser:
    load_project_env()
    p = argparse.ArgumentParser(description="AF3 + ChatGPT LoRA metadata pipeline")
    p.add_argument("--audio", default="", help="Single audio path")
    p.add_argument("--dataset-dir", default="", help="Dataset folder")
    p.add_argument("--backend", default="hf_endpoint", choices=["hf_endpoint", "local"])
    p.add_argument("--endpoint-url", default=get_env("HF_AF3_ENDPOINT_URL", "hf_af3_endpoint_url"))
    p.add_argument("--hf-token", default="")
    p.add_argument("--model-id", default=get_env("AF3_MODEL_ID", "af3_model_id", default=DEFAULT_AF3_MODEL_ID))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--prompt", default=DEFAULT_AF3_PROMPT)
    p.add_argument(
        "--think-long",
        action="store_true",
        help="Use long-form AF3 prompt + higher token budget defaults.",
    )
    p.add_argument("--af3-max-new-tokens", type=int, default=1400)
    p.add_argument("--af3-temperature", type=float, default=0.1)
    p.add_argument("--openai-api-key", default="")
    p.add_argument("--openai-model", default=get_env("OPENAI_MODEL", "openai_model", default=DEFAULT_OPENAI_MODEL))
    p.add_argument("--user-context", default="")
    p.add_argument("--artist-name", default="")
    p.add_argument("--track-name", default="")
    p.add_argument("--enable-web-search", action="store_true")
    p.add_argument("--output-dir", default="", help="If set, save sidecars here instead of next to audio")
    return p


def resolve_audio_paths(args) -> List[str]:
    if args.audio:
        p = Path(args.audio)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {p}")
        return [str(p)]
    if args.dataset_dir:
        files = list_audio_files(args.dataset_dir)
        if not files:
            raise RuntimeError(f"No audio files found in {args.dataset_dir}")
        return files
    raise ValueError("Provide --audio or --dataset-dir")


def main() -> int:
    args = build_parser().parse_args()
    hf_token = args.hf_token or get_env("HF_TOKEN", "hf_token")
    openai_key = (
        args.openai_api_key
        or get_env("OPENAI_API_KEY", "openai_api_key")
    )
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required for cleanup step.")

    if args.backend == "hf_endpoint":
        if not args.endpoint_url:
            raise RuntimeError("HF endpoint backend requires --endpoint-url")
        af3_client = AF3EndpointClient(
            endpoint_url=args.endpoint_url,
            token=hf_token,
            model_id=args.model_id,
        )
    else:
        af3_client = AF3LocalClient(
            model_id=args.model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )

    af3_prompt = args.prompt
    af3_max_new_tokens = int(args.af3_max_new_tokens)
    af3_temperature = float(args.af3_temperature)
    if args.think_long:
        if af3_prompt == DEFAULT_AF3_PROMPT:
            af3_prompt = DEFAULT_AF3_PROMPT_THINK_LONG
        if af3_max_new_tokens == 1400:
            af3_max_new_tokens = 3200
        if abs(af3_temperature - 0.1) < 1e-9:
            af3_temperature = 0.2

    audio_paths = resolve_audio_paths(args)
    failures = []
    saved = []
    for audio_path in tqdm(audio_paths, desc="AF3+ChatGPT"):
        try:
            result = run_af3_chatgpt_pipeline(
                audio_path=audio_path,
                af3_client=af3_client,
                af3_prompt=af3_prompt,
                af3_max_new_tokens=af3_max_new_tokens,
                af3_temperature=af3_temperature,
                openai_api_key=openai_key,
                openai_model=args.openai_model,
                user_context=args.user_context,
                artist_name=args.artist_name,
                track_name=args.track_name,
                enable_web_search=bool(args.enable_web_search),
            )
            sidecar = result["sidecar"]
            if args.output_dir:
                out_path = Path(args.output_dir) / (Path(audio_path).stem + ".json")
            else:
                out_path = Path(audio_path).with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
            saved.append(str(out_path))
        except Exception as exc:
            failures.append(f"{Path(audio_path).name}: {exc}")

    print(
        json.dumps(
            {
                "processed": len(audio_paths),
                "saved": len(saved),
                "failed": len(failures),
                "saved_paths": saved[:20],
                "failures": failures[:20],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
