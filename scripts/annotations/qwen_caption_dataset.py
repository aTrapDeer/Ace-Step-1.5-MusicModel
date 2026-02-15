#!/usr/bin/env python
"""
Batch caption a music dataset with Qwen2-Audio and export LoRA-ready sidecars.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, snapshot_download
from loguru import logger
from tqdm import tqdm

from qwen_audio_captioning import (
    DEFAULT_ANALYSIS_PROMPT,
    DEFAULT_LONG_ANALYSIS_PROMPT,
    DEFAULT_MODEL_ID,
    build_captioner,
    export_annotation_records,
    generate_track_annotation,
    list_audio_files,
    read_prompt_file,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen2-Audio batch captioning for LoRA datasets")

    # Data source
    p.add_argument("--dataset-dir", type=str, default="", help="Local dataset folder")
    p.add_argument("--dataset-repo", type=str, default="", help="HF dataset repo id")
    p.add_argument("--dataset-revision", type=str, default="main", help="HF dataset revision")
    p.add_argument("--dataset-subdir", type=str, default="", help="Subdirectory inside dataset")

    # Backend
    p.add_argument("--backend", type=str, default="local", choices=["local", "hf_endpoint"])
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--endpoint-url", type=str, default="")
    p.add_argument("--hf-token", type=str, default="", help="HF token (or use HF_TOKEN env var)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    # Prompt + generation controls
    p.add_argument("--prompt", type=str, default=DEFAULT_ANALYSIS_PROMPT)
    p.add_argument("--prompt-file", type=str, default="", help="Text file to override --prompt")
    p.add_argument("--include-long-analysis", action="store_true", help="Also request long prose analysis")
    p.add_argument("--long-analysis-prompt", type=str, default=DEFAULT_LONG_ANALYSIS_PROMPT)
    p.add_argument("--long-analysis-prompt-file", type=str, default="", help="Text file to override --long-analysis-prompt")
    p.add_argument("--long-analysis-max-new-tokens", type=int, default=1200)
    p.add_argument("--long-analysis-temperature", type=float, default=0.1)
    p.add_argument("--segment-seconds", type=float, default=30.0)
    p.add_argument("--overlap-seconds", type=float, default=2.0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--keep-raw-outputs", action="store_true", help="Store per-segment raw outputs in sidecar JSON")

    # Export
    p.add_argument("--output-dir", type=str, default="qwen_annotations")
    p.add_argument("--copy-audio", action="store_true", help="Copy audio files into output_dir/dataset")
    p.add_argument(
        "--write-inplace-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write sidecars next to source audio (default: true). Use --no-write-inplace-sidecars to disable.",
    )

    # Optional upload of exported folder
    p.add_argument("--upload-repo", type=str, default="", help="Optional HF dataset repo to upload exports")
    p.add_argument("--upload-private", action="store_true", help="Create upload repo as private")
    p.add_argument("--upload-path", type=str, default="", help="Optional path inside upload repo")

    return p


def resolve_dataset_dir(args) -> str:
    if args.dataset_dir:
        if not Path(args.dataset_dir).is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {args.dataset_dir}")
        return args.dataset_dir

    if not args.dataset_repo:
        raise ValueError("Provide --dataset-dir or --dataset-repo")

    token = args.hf_token or os.getenv("HF_TOKEN", "")
    temp_root = tempfile.mkdtemp(prefix="qwen_caption_dataset_")
    local_dir = os.path.join(temp_root, "dataset")
    logger.info(f"Downloading dataset {args.dataset_repo}@{args.dataset_revision} -> {local_dir}")
    snapshot_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        revision=args.dataset_revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token or None,
    )
    if args.dataset_subdir:
        sub = os.path.join(local_dir, args.dataset_subdir)
        if not Path(sub).is_dir():
            raise FileNotFoundError(f"Dataset subdir not found: {sub}")
        return sub
    return local_dir


def upload_export_if_requested(args, output_dir: str):
    if not args.upload_repo:
        return
    token = args.hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF token missing. Set --hf-token or HF_TOKEN.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.upload_repo,
        repo_type="dataset",
        private=bool(args.upload_private),
        exist_ok=True,
    )
    path_in_repo = args.upload_path.strip().strip("/") if args.upload_path else ""
    logger.info(f"Uploading {output_dir} -> {args.upload_repo}/{path_in_repo}")
    api.upload_folder(
        repo_id=args.upload_repo,
        repo_type="dataset",
        folder_path=output_dir,
        path_in_repo=path_in_repo,
        commit_message="Upload Qwen2-Audio annotations",
    )
    logger.info("Upload complete")


def main() -> int:
    args = build_parser().parse_args()
    prompt = read_prompt_file(args.prompt_file) if args.prompt_file else args.prompt
    long_prompt = (
        read_prompt_file(args.long_analysis_prompt_file)
        if args.long_analysis_prompt_file
        else args.long_analysis_prompt
    )
    token = args.hf_token or os.getenv("HF_TOKEN", "")

    dataset_dir = resolve_dataset_dir(args)
    audio_files: List[str] = list_audio_files(dataset_dir)
    if not audio_files:
        raise RuntimeError(f"No audio files found in {dataset_dir}")
    logger.info(f"Found {len(audio_files)} audio files")

    captioner = build_captioner(
        backend=args.backend,
        model_id=args.model_id,
        endpoint_url=args.endpoint_url,
        token=token,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    records = []
    failed = []
    for path in tqdm(audio_files, desc="Captioning audio"):
        try:
            sidecar = generate_track_annotation(
                audio_path=path,
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
            records.append({"audio_path": path, "sidecar": sidecar})
        except Exception as exc:
            failed.append(f"{Path(path).name}: {exc}")
            logger.exception(f"Failed: {path}")

    export_result = export_annotation_records(
        records=records,
        output_dir=args.output_dir,
        copy_audio=bool(args.copy_audio),
        write_inplace_sidecars=bool(args.write_inplace_sidecars),
    )

    logger.info(
        "Done. analyzed={} failed={} manifest={}",
        len(records),
        len(failed),
        export_result["manifest_path"],
    )
    if failed:
        logger.warning("First failures:\n" + "\n".join(failed[:20]))

    upload_export_if_requested(args, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
