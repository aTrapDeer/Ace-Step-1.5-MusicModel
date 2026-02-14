#!/usr/bin/env python
"""
Bootstrap this project into your own Hugging Face Space and/or Endpoint repo.

Examples:
  python scripts/hf_clone.py space --repo-id your-name/ace-step-lora-studio
  python scripts/hf_clone.py endpoint --repo-id your-name/ace-step-endpoint
  python scripts/hf_clone.py all --space-repo-id your-name/ace-step-lora-studio --endpoint-repo-id your-name/ace-step-endpoint
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMON_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    ".cache",
    ".huggingface",
    ".gradio",
    "checkpoints",
    "lora_output",
    "outputs",
    "artifacts",
    "models",
    "datasets",
    "Lora-ace-step",
}

COMMON_SKIP_FILES = {
    ".env",
}

COMMON_SKIP_PREFIXES = (
    "song_summaries_llm",
)

COMMON_SKIP_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".pt",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".log",
    ".pyc",
    ".pyo",
    ".pyd",
}

MAX_FILE_BYTES = 30 * 1024 * 1024  # 30MB safety cap for upload snapshot


def _should_skip_common(rel_path: Path, is_dir: bool) -> bool:
    if any(part in COMMON_SKIP_DIRS for part in rel_path.parts):
        return True
    if rel_path.name in COMMON_SKIP_FILES:
        return True
    if any(rel_path.name.startswith(prefix) for prefix in COMMON_SKIP_PREFIXES):
        return True
    if not is_dir and rel_path.suffix.lower() in COMMON_SKIP_SUFFIXES:
        return True
    return False


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _stage_space_snapshot(staging_dir: Path) -> tuple[int, int, list[str]]:
    copied = 0
    bytes_total = 0
    skipped: list[str] = []

    for src in PROJECT_ROOT.rglob("*"):
        rel = src.relative_to(PROJECT_ROOT)

        if src.is_dir():
            if _should_skip_common(rel, is_dir=True):
                skipped.append(f"{rel}/")
            continue

        if _should_skip_common(rel, is_dir=False):
            skipped.append(str(rel))
            continue

        size = src.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append(f"{rel} (>{MAX_FILE_BYTES // (1024 * 1024)}MB)")
            continue

        dst = staging_dir / rel
        _copy_file(src, dst)
        copied += 1
        bytes_total += size

    return copied, bytes_total, skipped


def _iter_endpoint_paths() -> Iterable[Path]:
    # Minimal runtime set for custom endpoint repos.
    required = [
        PROJECT_ROOT / "handler.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "packages.txt",
        PROJECT_ROOT / "acestep",
    ]
    for p in required:
        if p.exists():
            yield p

    template_readme = PROJECT_ROOT / "templates" / "hf-endpoint" / "README.md"
    if template_readme.exists():
        yield template_readme


def _stage_endpoint_snapshot(staging_dir: Path) -> tuple[int, int]:
    copied = 0
    bytes_total = 0

    for src in _iter_endpoint_paths():
        if src.is_file():
            rel_dst = Path("README.md") if src.name == "README.md" and "templates" in src.parts else Path(src.name)
            dst = staging_dir / rel_dst
            _copy_file(src, dst)
            copied += 1
            bytes_total += src.stat().st_size
            continue

        if src.is_dir():
            for nested in src.rglob("*"):
                rel_nested = nested.relative_to(src)
                if nested.is_dir():
                    if _should_skip_common(Path(src.name) / rel_nested, is_dir=True):
                        continue
                    continue
                if _should_skip_common(Path(src.name) / rel_nested, is_dir=False):
                    continue
                if nested.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    continue

                dst = staging_dir / src.name / rel_nested
                _copy_file(nested, dst)
                copied += 1
                bytes_total += nested.stat().st_size

    return copied, bytes_total


def _resolve_token(arg_token: str) -> str | None:
    if arg_token:
        return arg_token
    return os.getenv("HF_TOKEN")


def _ensure_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    space_sdk: str | None = None,
) -> None:
    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "private": private,
        "exist_ok": True,
    }
    if repo_type == "space" and space_sdk:
        kwargs["space_sdk"] = space_sdk
    api.create_repo(**kwargs)


def _upload_snapshot(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    folder_path: Path,
    commit_message: str,
) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder_path),
        commit_message=commit_message,
        delete_patterns=[],
    )


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def clone_space(repo_id: str, private: bool, token: str | None, dry_run: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="hf_space_clone_") as tmp:
        staging = Path(tmp)
        copied, bytes_total, skipped = _stage_space_snapshot(staging)
        print(f"[space] staged files: {copied}, size: {_fmt_mb(bytes_total)}")
        if skipped:
            print(f"[space] skipped entries: {len(skipped)}")
            for item in skipped[:20]:
                print(f"  - {item}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more")

        if dry_run:
            print("[space] dry-run complete (nothing uploaded).")
            return

        api = HfApi(token=token)
        _ensure_repo(api, repo_id=repo_id, repo_type="space", private=private, space_sdk="gradio")
        _upload_snapshot(
            api,
            repo_id=repo_id,
            repo_type="space",
            folder_path=staging,
            commit_message="Bootstrap ACE-Step LoRA Studio Space",
        )
        print(f"[space] uploaded to https://huggingface.co/spaces/{repo_id}")


def clone_endpoint(repo_id: str, private: bool, token: str | None, dry_run: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="hf_endpoint_clone_") as tmp:
        staging = Path(tmp)
        copied, bytes_total = _stage_endpoint_snapshot(staging)
        print(f"[endpoint] staged files: {copied}, size: {_fmt_mb(bytes_total)}")

        if dry_run:
            print("[endpoint] dry-run complete (nothing uploaded).")
            return

        api = HfApi(token=token)
        _ensure_repo(api, repo_id=repo_id, repo_type="model", private=private)
        _upload_snapshot(
            api,
            repo_id=repo_id,
            repo_type="model",
            folder_path=staging,
            commit_message="Bootstrap ACE-Step custom endpoint repo",
        )
        print(f"[endpoint] uploaded to https://huggingface.co/{repo_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clone this project into your own HF Space/Endpoint repos.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_space = subparsers.add_parser("space", help="Create/update your HF Space from this project.")
    p_space.add_argument("--repo-id", required=True, help="Target space repo id, e.g. username/my-space.")
    p_space.add_argument("--private", action="store_true", help="Create repo as private.")
    p_space.add_argument("--token", type=str, default="", help="HF token (default: HF_TOKEN env var).")
    p_space.add_argument("--dry-run", action="store_true", help="Stage files only; do not upload.")

    p_endpoint = subparsers.add_parser("endpoint", help="Create/update your custom endpoint model repo.")
    p_endpoint.add_argument("--repo-id", required=True, help="Target model repo id, e.g. username/my-endpoint.")
    p_endpoint.add_argument("--private", action="store_true", help="Create repo as private.")
    p_endpoint.add_argument("--token", type=str, default="", help="HF token (default: HF_TOKEN env var).")
    p_endpoint.add_argument("--dry-run", action="store_true", help="Stage files only; do not upload.")

    p_all = subparsers.add_parser("all", help="Run both Space and Endpoint bootstrap.")
    p_all.add_argument("--space-repo-id", required=True, help="Target space repo id.")
    p_all.add_argument("--endpoint-repo-id", required=True, help="Target endpoint model repo id.")
    p_all.add_argument("--space-private", action="store_true", help="Create Space as private.")
    p_all.add_argument("--endpoint-private", action="store_true", help="Create endpoint repo as private.")
    p_all.add_argument("--token", type=str, default="", help="HF token (default: HF_TOKEN env var).")
    p_all.add_argument("--dry-run", action="store_true", help="Stage files only; do not upload.")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = _resolve_token(args.token)

    if not token and not args.dry_run:
        print("HF token not found. Set HF_TOKEN or pass --token.")
        return 1

    if args.cmd == "space":
        clone_space(args.repo_id, private=bool(args.private), token=token, dry_run=bool(args.dry_run))
    elif args.cmd == "endpoint":
        clone_endpoint(args.repo_id, private=bool(args.private), token=token, dry_run=bool(args.dry_run))
    else:
        clone_space(args.space_repo_id, private=bool(args.space_private), token=token, dry_run=bool(args.dry_run))
        clone_endpoint(
            args.endpoint_repo_id,
            private=bool(args.endpoint_private),
            token=token,
            dry_run=bool(args.dry_run),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
