#!/usr/bin/env python
"""
Refine existing dataset JSON annotations into Ace-Step 1.5 LoRA-ready sidecars.

This script:
1. Reads existing JSON files (typically containing AF3 `generated_text`).
2. Uses OpenAI cleanup (optionally with web search) to normalize/expand metadata.
3. Writes normalized sidecar JSON in-place (or to an output directory).
4. Creates backup copies before overwrite by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from af3_chatgpt_pipeline import (  # noqa: E402
    DEFAULT_AF3_PROMPT,
    DEFAULT_OPENAI_MODEL,
    build_lora_sidecar,
    cleanup_with_chatgpt,
)
from qwen_audio_captioning import AUDIO_EXTENSIONS  # noqa: E402
from utils.env_config import get_env, load_project_env  # noqa: E402


def _parse_args() -> argparse.Namespace:
    load_project_env()
    p = argparse.ArgumentParser(
        description="Refine dataset JSONs into Ace-Step 1.5 LoRA-ready metadata using OpenAI."
    )
    p.add_argument("--dataset-dir", default="train-dataset", help="Directory containing source JSON files")
    p.add_argument("--recursive", action="store_true", help="Include nested folders")
    p.add_argument("--pattern", default="*.json", help="Filename glob pattern")
    p.add_argument("--output-dir", default="", help="Optional output folder. Default: overwrite in place")
    p.add_argument(
        "--backup-ext",
        default=".backup-before-openai.json",
        help="Backup extension for in-place writes",
    )
    p.add_argument("--no-backup", action="store_true", help="Disable backup creation for in-place writes")
    p.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    p.add_argument("--artist-default", default="Andrew Spacey", help="Fallback artist if parsing fails")
    p.add_argument("--user-context", default="", help="Extra guidance passed to OpenAI cleanup")
    p.add_argument("--openai-api-key", default="", help="Overrides OPENAI_API_KEY from .env")
    p.add_argument(
        "--openai-model",
        default=get_env("OPENAI_MODEL", "openai_model", default=DEFAULT_OPENAI_MODEL),
        help="OpenAI model id",
    )
    p.add_argument(
        "--enable-web-search",
        action="store_true",
        help="Enable web search tool for artist/track context lookup",
    )
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    return p.parse_args()


def _iter_json_files(dataset_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(dataset_dir.rglob(pattern))
    return sorted(dataset_dir.glob(pattern))


def _load_json(path: Path) -> Dict:
    # Handle both standard UTF-8 and UTF-8 with BOM.
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON is not an object")
    return data


def _detect_audio_path(json_path: Path) -> Optional[Path]:
    stem = json_path.stem
    for ext in AUDIO_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    # Fallback to case-insensitive scan.
    parent = json_path.parent
    for f in parent.iterdir():
        if f.is_file() and f.stem == stem and f.suffix.lower() in AUDIO_EXTENSIONS:
            return f
    return None


def _try_duration_seconds(audio_path: Optional[Path], fallback: float = 0.0) -> float:
    if audio_path is None or not audio_path.exists():
        return float(fallback or 0.0)
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return float(fallback or 0.0)


def _parse_artist_track_from_stem(stem: str, artist_default: str) -> Tuple[str, str]:
    parts = stem.split(" - ", 1)
    if len(parts) == 2:
        artist, track = parts[0].strip(), parts[1].strip()
        if artist and track:
            return artist, track
    return artist_default.strip() or "Unknown Artist", stem.strip()


def _extract_raw_analysis(data: Dict) -> str:
    checks: Iterable[object] = (
        data.get("generated_text"),
        data.get("af3_analysis"),
        data.get("analysis_long"),
        data.get("analysis_short"),
        (data.get("music_analysis") or {}).get("summary_long") if isinstance(data.get("music_analysis"), dict) else None,
        data.get("caption"),
    )
    for value in checks:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ensure_output_path(src_json: Path, output_dir: Optional[Path]) -> Path:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / src_json.name
    return src_json


def _create_backup(src: Path, backup_ext: str) -> Optional[Path]:
    backup_path = src.with_name(src.stem + backup_ext)
    if backup_path.exists():
        return backup_path
    shutil.copy2(src, backup_path)
    return backup_path


def _finalize_sidecar(
    *,
    cleaned: Dict,
    raw_analysis: str,
    duration: float,
    source_audio: Optional[Path],
    source_json: Path,
    artist: str,
    track_name: str,
    openai_model: str,
    web_search_used: bool,
) -> Dict:
    source_audio_str = str(source_audio) if source_audio else ""
    sidecar = build_lora_sidecar(
        cleaned,
        af3_text=raw_analysis,
        af3_prompt=DEFAULT_AF3_PROMPT,
        af3_backend="existing_json_refine",
        af3_model_id="nvidia/audio-flamingo-3",
        source_audio=source_audio_str,
        duration=duration,
        chatgpt_model=openai_model,
        web_search_used=web_search_used,
    )
    sidecar["artist"] = artist
    sidecar["track_name"] = track_name
    sidecar["source"] = {
        "input_json": str(source_json),
        "input_audio": source_audio_str,
        "refined_from_existing_json": True,
    }
    return sidecar


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    openai_key = args.openai_api_key or get_env("OPENAI_API_KEY", "openai_api_key")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in .env or pass --openai-api-key).")

    files = _iter_json_files(dataset_dir, pattern=args.pattern, recursive=bool(args.recursive))
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise RuntimeError(f"No files matched {args.pattern} in {dataset_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    failures: List[str] = []
    saved: List[str] = []
    backups: List[str] = []

    for json_path in tqdm(files, desc="Refine JSON"):
        try:
            data = _load_json(json_path)
            raw_analysis = _extract_raw_analysis(data)
            if not raw_analysis:
                raise ValueError("No analysis text found (generated_text/analysis/caption missing)")

            artist, track_name = _parse_artist_track_from_stem(json_path.stem, args.artist_default)
            artist = str(data.get("artist") or artist).strip() or artist
            track_name = str(data.get("track_name") or data.get("title") or track_name).strip() or track_name

            source_audio = _detect_audio_path(json_path)
            duration = _try_duration_seconds(source_audio, fallback=float(data.get("duration") or 0.0))

            try:
                cleaned = cleanup_with_chatgpt(
                    raw_analysis,
                    openai_api_key=openai_key,
                    model=args.openai_model,
                    duration=duration,
                    user_context=args.user_context,
                    artist_name=artist,
                    track_name=track_name,
                    enable_web_search=bool(args.enable_web_search),
                )
                web_used = bool(args.enable_web_search)
            except Exception:
                # If web-search tool compatibility fails on this runtime, retry without it.
                if not args.enable_web_search:
                    raise
                cleaned = cleanup_with_chatgpt(
                    raw_analysis,
                    openai_api_key=openai_key,
                    model=args.openai_model,
                    duration=duration,
                    user_context=args.user_context,
                    artist_name=artist,
                    track_name=track_name,
                    enable_web_search=False,
                )
                web_used = False

            sidecar = _finalize_sidecar(
                cleaned=cleaned,
                raw_analysis=raw_analysis,
                duration=duration,
                source_audio=source_audio,
                source_json=json_path,
                artist=artist,
                track_name=track_name,
                openai_model=args.openai_model,
                web_search_used=web_used,
            )

            out_path = _ensure_output_path(json_path, output_dir)
            if not args.dry_run and output_dir is None and not args.no_backup:
                backup = _create_backup(json_path, args.backup_ext)
                if backup:
                    backups.append(str(backup))

            if not args.dry_run:
                out_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
            saved.append(str(out_path))
        except Exception as exc:
            failures.append(f"{json_path.name}: {exc}")
            if args.fail_fast:
                break

    summary = {
        "processed": len(files),
        "saved": len(saved),
        "failed": len(failures),
        "backup_count": len(backups),
        "output_mode": "separate_dir" if output_dir else ("dry_run" if args.dry_run else "in_place"),
        "sample_saved": saved[:10],
        "sample_failures": failures[:10],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

