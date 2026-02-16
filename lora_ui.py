"""
ACE-Step 1.5 LoRA Training and Evaluation UI.

Gradio interface with four tabs:
  1. Model Setup: initialize base DiT, VAE, and text encoder
  2. Dataset: scan folder or drop files, then edit/save sidecars
  3. Training: configure hyperparameters and run LoRA training
  4. Evaluation: load adapters and run deterministic A/B generation
"""
import os
import sys
import json
import math
import random
import threading
import tempfile
import time
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import gradio as gr
# On Hugging Face Spaces Zero, `spaces` must be imported before CUDA-related modules.
if os.getenv("SPACE_ID"):
    try:
        import spaces  # noqa: F401
    except Exception:
        pass
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `acestep` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from acestep.handler import AceStepHandler
from acestep.audio_utils import AudioSaver
from acestep.llm_inference import LLMHandler
from acestep.inference import understand_music
from lora_train import (
    LoRATrainConfig,
    LoRATrainer,
    TrackEntry,
    scan_dataset_folder,
    scan_uploaded_files,
)

# ---------------------------------------------------------------------------
# Globals (shared across Gradio callbacks)
# ---------------------------------------------------------------------------
handler = AceStepHandler()
llm_handler = LLMHandler()
trainer: Optional[LoRATrainer] = None
dataset_entries: List[TrackEntry] = []
_training_thread: Optional[threading.Thread] = None
_training_log: List[str] = []
_training_status: str = "idle"  # idle | running | stopped | done
_training_started_at: Optional[float] = None
_model_init_ok: bool = False
_model_init_status: str = ""
_last_model_init_args: Optional[dict] = None
_lm_init_ok: bool = False
_last_lm_init_args: Optional[dict] = None
_auto_label_cursor: int = 0
_last_loaded_adapter_path: Optional[str] = None

audio_saver = AudioSaver(default_format="wav")
IS_SPACE = bool(os.getenv("SPACE_ID"))
DEFAULT_UPLOADED_ADAPTER_SUBDIR = "uploaded_adapters"


def _resolve_writable_output_dir(preferred_dir: Optional[str] = None) -> Path:
    """Resolve a writable output directory for training artifacts and uploaded adapters."""
    candidates: List[Path] = []
    if preferred_dir:
        candidates.append(Path(preferred_dir))
    env_output = os.getenv("LORA_OUTPUT_DIR")
    if env_output:
        candidates.append(Path(env_output))
    if IS_SPACE:
        candidates.extend([
            Path("/data/lora_output"),
            Path("/tmp/lora_output"),
            Path(PROJECT_ROOT) / "lora_output",
        ])
    else:
        candidates.append(Path("lora_output"))

    checked = set()
    for candidate in candidates:
        key = str(candidate)
        if key in checked:
            continue
        checked.add(key)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            continue

    # Final fallback in the current working tree.
    fallback = Path(PROJECT_ROOT) / "lora_output"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


DEFAULT_OUTPUT_DIR = str(_resolve_writable_output_dir())

if IS_SPACE:
    try:
        import spaces as _hf_spaces
        _gpu_callback = _hf_spaces.GPU(duration=300)
    except Exception:
        _gpu_callback = lambda fn: fn
else:
    _gpu_callback = lambda fn: fn


def _rows_from_entries(entries: List[TrackEntry]):
    rows = []
    for e in entries:
        rows.append([
            Path(e.audio_path).name,
            f"{e.duration:.1f}s" if e.duration else "?",
            e.caption or "(none)",
            e.lyrics[:60] + "..." if len(e.lyrics) > 60 else (e.lyrics or "(none)"),
            e.vocal_language,
        ])
    return rows


# ===========================================================================
# Tab 1 - Model Setup
# ===========================================================================

def get_available_models():
    models = handler.get_available_acestep_v15_models()
    return models if models else ["acestep-v15-base"]


def init_model(
    model_name: str,
    device: str,
    offload_cpu: bool,
    offload_dit_cpu: bool,
):
    global _model_init_ok, _model_init_status, _last_model_init_args
    _last_model_init_args = dict(
        project_root=PROJECT_ROOT,
        config_path=model_name,
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=offload_cpu,
        offload_dit_to_cpu=offload_dit_cpu,
    )
    status, ok = _init_model_gpu(**_last_model_init_args)
    _model_init_ok = bool(ok)
    _model_init_status = status or ""
    return status


@_gpu_callback
def _init_model_gpu(**kwargs):
    return _init_model_impl(**kwargs)


def _init_model_impl(**kwargs):
    return handler.initialize_service(**kwargs)


def _ensure_model_ready_for_eval() -> tuple[bool, str]:
    """Ensure the base model is initialized before adapter load/generation callbacks."""
    global _model_init_ok, _model_init_status
    if handler.model is not None:
        return True, ""
    if _model_init_ok and _last_model_init_args:
        status, ok = _init_model_impl(**_last_model_init_args)
        _model_init_ok = bool(ok)
        _model_init_status = status or ""
        if ok:
            return True, "Model reloaded for this session."
        return False, f"Model reload failed:\n{status}"
    return False, "Model not initialized. Please initialize model in Step 1 first."


def _looks_like_lora_loaded(status: str) -> bool:
    s = (status or "").strip().lower()
    return ("lora loaded from" in s) or s.startswith("✅")


def _ensure_lora_ready_for_eval(use_lora: bool) -> tuple[bool, str]:
    """Ensure requested LoRA state is available in current callback context."""
    global _last_loaded_adapter_path
    if not use_lora:
        return True, ""
    if handler.lora_loaded:
        return True, ""
    if _last_loaded_adapter_path and os.path.exists(_last_loaded_adapter_path):
        status = handler.load_lora(_last_loaded_adapter_path)
        if _looks_like_lora_loaded(status):
            return True, f"LoRA reloaded for this session from {_last_loaded_adapter_path}."
        return False, f"LoRA reload failed:\n{status}"
    return False, "LoRA requested but no adapter is loaded in this session. Click 'Load Adapter' first."


# ===========================================================================
# Tab 2 - Dataset
# ===========================================================================

def scan_folder(folder_path: str):
    global dataset_entries, _auto_label_cursor
    if not folder_path or not os.path.isdir(folder_path):
        return "Provide a valid folder path.", []
    dataset_entries = scan_dataset_folder(folder_path)
    _auto_label_cursor = 0
    rows = _rows_from_entries(dataset_entries)
    msg = f"Found {len(dataset_entries)} audio files."
    return msg, rows


def load_uploaded(file_paths: List[str]):
    global dataset_entries, _auto_label_cursor
    if not file_paths:
        return "Drop audio files (and optional .json sidecars) first.", []
    sidecar_count = sum(
        1 for p in file_paths if isinstance(p, str) and Path(p).suffix.lower() == ".json"
    )
    dataset_entries = scan_uploaded_files(file_paths)
    _auto_label_cursor = 0
    rows = _rows_from_entries(dataset_entries)
    msg = (
        f"Loaded {len(dataset_entries)} dropped audio files."
        + (f" Matched {sidecar_count} uploaded sidecar JSON file(s)." if sidecar_count else "")
    )
    return msg, rows


def save_sidecar(index: int, caption: str, lyrics: str, bpm: str, keyscale: str, lang: str):
    """Save metadata edits back to a JSON sidecar and update in-memory entry."""
    global dataset_entries
    if index < 0 or index >= len(dataset_entries):
        return "Invalid track index."
    entry = dataset_entries[index]
    entry.caption = caption
    entry.lyrics = lyrics
    if bpm.strip():
        try:
            entry.bpm = int(float(bpm))
        except ValueError:
            return "Invalid BPM value. Use an integer or leave empty."
    else:
        entry.bpm = None
    entry.keyscale = keyscale
    entry.vocal_language = lang

    sidecar_path = Path(entry.audio_path).with_suffix(".json")
    meta = {
        "caption": entry.caption,
        "lyrics": entry.lyrics,
        "bpm": entry.bpm,
        "keyscale": entry.keyscale,
        "timesignature": entry.timesignature,
        "vocal_language": entry.vocal_language,
        "duration": entry.duration,
    }
    sidecar_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Saved sidecar for {Path(entry.audio_path).name}"


def init_auto_label_lm(lm_model_path: str, lm_backend: str, lm_device: str):
    global _lm_init_ok, _last_lm_init_args
    _last_lm_init_args = dict(
        lm_model_path=lm_model_path,
        lm_backend=lm_backend,
        lm_device=lm_device,
    )
    status = _init_auto_label_lm_gpu(**_last_lm_init_args)
    _lm_init_ok = not str(status).startswith("LM init failed:") and not str(status).startswith("LM init exception:")
    return status


@_gpu_callback
def _init_auto_label_lm_gpu(lm_model_path: str, lm_backend: str, lm_device: str):
    return _init_auto_label_lm_impl(lm_model_path, lm_backend, lm_device)


def _init_auto_label_lm_impl(lm_model_path: str, lm_backend: str, lm_device: str):
    """Initialize LLM for dataset auto-labeling."""
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    full_lm_path = os.path.join(checkpoint_dir, lm_model_path)

    try:
        if not os.path.exists(full_lm_path):
            from pathlib import Path as _Path
            from acestep.model_downloader import ensure_main_model, ensure_lm_model

            if lm_model_path == "acestep-5Hz-lm-1.7B":
                ok, msg = ensure_main_model(
                    checkpoints_dir=_Path(checkpoint_dir),
                    prefer_source="huggingface",
                )
            else:
                ok, msg = ensure_lm_model(
                    model_name=lm_model_path,
                    checkpoints_dir=_Path(checkpoint_dir),
                    prefer_source="huggingface",
                )
            if not ok:
                return f"Failed to download LM model: {msg}"

        status, ok = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model_path,
            backend=lm_backend,
            device=lm_device,
            offload_to_cpu=False,
        )
        return status if ok else f"LM init failed:\n{status}"
    except Exception as exc:
        logger.exception("LM init failed for auto-label")
        return f"LM init exception: {exc}"


def _write_entry_sidecar(entry: TrackEntry):
    sidecar_path = Path(entry.audio_path).with_suffix(".json")
    meta = {
        "caption": entry.caption,
        "lyrics": entry.lyrics,
        "bpm": entry.bpm,
        "keyscale": entry.keyscale,
        "timesignature": entry.timesignature,
        "vocal_language": entry.vocal_language,
        "duration": entry.duration,
    }
    sidecar_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


@_gpu_callback
def auto_label_all(overwrite_existing: bool, caption_only: bool, max_files_per_run: int = 6, reset_cursor: bool = False):
    """Auto-label all loaded tracks using ACE audio understanding (audio->codes->metadata)."""
    global dataset_entries, _auto_label_cursor

    if handler.model is None:
        if _model_init_ok and _last_model_init_args:
            status, ok = _init_model_impl(**_last_model_init_args)
            if not ok:
                return f"Model reload failed before auto-label:\n{status}", [], "Auto-label skipped."
        else:
            return "Initialize model first in Step 1.", [], "Auto-label skipped."
    if not dataset_entries:
        return "Load dataset first in Step 2.", [], "Auto-label skipped."
    if not llm_handler.llm_initialized:
        if _lm_init_ok and _last_lm_init_args:
            status = _init_auto_label_lm_impl(**_last_lm_init_args)
            if not llm_handler.llm_initialized:
                return (
                    f"Auto-label LM reload failed:\n{status}",
                    _rows_from_entries(dataset_entries),
                    "Auto-label skipped.",
                )
        else:
            return "Initialize Auto-Label LM first.", _rows_from_entries(dataset_entries), "Auto-label skipped."

    if max_files_per_run <= 0:
        max_files_per_run = 6
    if reset_cursor:
        _auto_label_cursor = 0
    if _auto_label_cursor < 0 or _auto_label_cursor >= len(dataset_entries):
        _auto_label_cursor = 0

    start_idx = _auto_label_cursor
    end_idx = min(len(dataset_entries), start_idx + int(max_files_per_run))

    updated = 0
    skipped = 0
    failed = 0
    logs: List[str] = []

    for idx in range(start_idx, end_idx):
        entry = dataset_entries[idx]
        try:
            missing_fields = []
            if not (entry.caption or "").strip():
                missing_fields.append("caption")
            if (not caption_only) and (not (entry.lyrics or "").strip()):
                missing_fields.append("lyrics")
            if entry.bpm is None:
                missing_fields.append("bpm")
            if not (entry.keyscale or "").strip():
                missing_fields.append("keyscale")
            if entry.duration is None:
                missing_fields.append("duration")

            # Skip only when every core field is already available.
            if (not overwrite_existing) and (len(missing_fields) == 0):
                skipped += 1
                logs.append(f"[{idx}] Skipped (already fully labeled): {Path(entry.audio_path).name}")
                continue

            codes = handler.convert_src_audio_to_codes(entry.audio_path)
            if not codes or codes.startswith("❌"):
                failed += 1
                logs.append(f"[{idx}] Failed to convert audio to codes: {Path(entry.audio_path).name}")
                continue

            result = understand_music(
                llm_handler=llm_handler,
                audio_codes=codes,
                temperature=0.85,
                use_constrained_decoding=True,
                constrained_decoding_debug=False,
            )
            if not result.success:
                failed += 1
                logs.append(f"[{idx}] Failed to label: {Path(entry.audio_path).name} ({result.error or result.status_message})")
                continue

            # Update fields. If overwrite is false, fill only missing values.
            if overwrite_existing or not (entry.caption or "").strip():
                entry.caption = (result.caption or entry.caption or "").strip()
            if not caption_only:
                if overwrite_existing or not (entry.lyrics or "").strip():
                    entry.lyrics = (result.lyrics or entry.lyrics or "").strip()
            if entry.bpm is None and result.bpm is not None:
                entry.bpm = int(result.bpm)
            if (not entry.keyscale) and result.keyscale:
                entry.keyscale = result.keyscale
            if (not entry.timesignature) and result.timesignature:
                entry.timesignature = result.timesignature
            if (not entry.vocal_language) and result.language:
                entry.vocal_language = result.language
            if entry.duration is None and result.duration is not None:
                entry.duration = float(result.duration)

            _write_entry_sidecar(entry)
            updated += 1
            logs.append(f"[{idx}] Labeled: {Path(entry.audio_path).name}")
        except Exception as exc:
            failed += 1
            logs.append(f"[{idx}] Exception: {Path(entry.audio_path).name} ({exc})")

    _auto_label_cursor = 0 if end_idx >= len(dataset_entries) else end_idx
    mode = "caption-only" if caption_only else "caption+lyrics"
    progress_msg = (
        f"Processed batch {start_idx + 1}-{end_idx} of {len(dataset_entries)}. "
        if len(dataset_entries) > 0 else ""
    )
    if _auto_label_cursor == 0 and len(dataset_entries) > 0:
        progress_msg += "Reached end of dataset."
    else:
        progress_msg += f"Next start index: {_auto_label_cursor}."
    summary = (
        f"Auto-label ({mode}) complete. Updated={updated}, Skipped={skipped}, Failed={failed}. "
        f"{progress_msg}"
    )
    detail = "\n".join(logs[-40:]) if logs else "No logs."
    return summary, _rows_from_entries(dataset_entries), detail


# ===========================================================================
# Tab 3 - Training
# ===========================================================================

def _run_training(config_dict: dict):
    """Target for the background training thread."""
    global trainer, _training_status, _training_log, _training_started_at
    _training_status = "running"
    _training_log.clear()
    _training_started_at = time.time()

    try:
        cfg = LoRATrainConfig(**config_dict)
        trainer = LoRATrainer(handler, cfg)
        trainer.prepare()
        _training_log.append(f"Training device: {handler.device}")

        def _cb(step, total, loss, epoch):
            elapsed = 0.0 if _training_started_at is None else max(0.0, time.time() - _training_started_at)
            rate = (step / elapsed) if elapsed > 0 else 0.0
            remaining = max(0, total - step)
            eta_sec = (remaining / rate) if rate > 0 else -1.0
            eta_msg = f"{eta_sec/60:.1f}m" if eta_sec >= 0 else "unknown"
            msg = (
                f"Step {step}/{total}  Epoch {epoch+1}  Loss {loss:.6f}  "
                f"Elapsed {elapsed/60:.1f}m  ETA {eta_msg}"
            )
            _training_log.append(msg)

        result = trainer.train(dataset_entries, progress_callback=_cb)
        _training_log.append(result)
        _training_status = "done"
    except Exception as exc:
        _training_log.append(f"ERROR: {exc}")
        _training_status = "stopped"
        logger.exception("Training failed")


def start_training(
    lora_rank, lora_alpha, lora_dropout,
    lr, weight_decay, optimizer_name,
    max_grad_norm, warmup_ratio, scheduler_name,
    num_epochs, batch_size, grad_accum,
    save_every, log_every, shift,
    max_duration, output_dir, resume_from,
):
    global _training_thread, _training_status

    if handler.model is None:
        return "Model not initialised. Go to Model Setup first."
    if not dataset_entries:
        return "No dataset loaded. Go to Dataset tab first."
    if _training_status == "running":
        return "Training already in progress."

    config_dict = dict(
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        learning_rate=float(lr),
        weight_decay=float(weight_decay),
        optimizer=optimizer_name,
        max_grad_norm=float(max_grad_norm),
        warmup_ratio=float(warmup_ratio),
        scheduler=scheduler_name,
        num_epochs=int(num_epochs),
        batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        save_every_n_epochs=int(save_every),
        log_every_n_steps=int(log_every),
        shift=float(shift),
        max_duration_sec=float(max_duration),
        output_dir=output_dir,
        resume_from=(resume_from.strip() if isinstance(resume_from, str) and resume_from.strip() else None),
        device=str(handler.device),
    )

    steps_per_epoch = math.ceil(len(dataset_entries) / int(batch_size))
    total_steps = steps_per_epoch * int(num_epochs)
    total_optim_steps = math.ceil(total_steps / int(grad_accum))

    _training_thread = threading.Thread(target=_run_training, args=(config_dict,), daemon=True)
    _training_thread.start()
    return (
        f"Training started on {handler.device}. "
        f"Estimated optimiser steps: {total_optim_steps}."
    )


def stop_training():
    global trainer, _training_status
    if trainer:
        trainer.request_stop()
        _training_status = "stopped"
        return "Stop requested - will finish current step."
    return "No training in progress."


def poll_training():
    """Return current log + loss chart data."""
    log_text = "\n".join(_training_log[-50:]) if _training_log else "(no output yet)"

    # Build loss curve data
    chart_data = []
    if trainer and trainer.loss_history:
        chart_data = [[h["step"], h["loss"]] for h in trainer.loss_history]

    status = _training_status
    device_line = f"Device: {handler.device}"
    if torch.cuda.is_available() and str(handler.device).startswith("cuda"):
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
            device_line = (
                f"Device: {handler.device} ({name}) | "
                f"VRAM allocated={allocated:.2f}GB reserved={reserved:.2f}GB"
            )
        except Exception:
            pass

    return f"Status: {status}\n{device_line}\n\n{log_text}", chart_data


# ===========================================================================
# Tab 4 - Evaluation / A-B Test
# ===========================================================================

def list_adapters(output_dir: str):
    try:
        resolved_output = _resolve_writable_output_dir(output_dir)
    except Exception:
        return ["(none found)"]
    adapters = LoRATrainer.list_adapters(str(resolved_output))
    return adapters if adapters else ["(none found)"]


def _safe_adapter_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return f"adapter_{int(time.time())}"
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._")
    return cleaned or f"adapter_{int(time.time())}"


def _safe_extract_zip(zip_path: str, target_dir: Path) -> int:
    extracted = 0
    target_resolved = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if not str(member_path).startswith(str(target_resolved)):
                raise RuntimeError(f"Unsafe archive path detected: {member.filename}")
        zf.extractall(target_dir)
        extracted = len(zf.namelist())
    return extracted


def upload_adapter_files(uploaded_files: List[str], adapter_dir: str, adapter_name: str):
    """Upload LoRA adapter files/zip and make them available in adapter dropdown."""
    if not uploaded_files:
        adapters = list_adapters(adapter_dir)
        return "Please upload .zip or adapter files first.", gr.update(choices=adapters, value=adapters[0] if adapters else None)

    root_dir = _resolve_writable_output_dir(adapter_dir or DEFAULT_OUTPUT_DIR)
    target_root = root_dir / DEFAULT_UPLOADED_ADAPTER_SUBDIR
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / _safe_adapter_name(adapter_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    extracted = 0
    try:
        # If a single zip is uploaded, extract it; otherwise copy files directly.
        if len(uploaded_files) == 1 and str(uploaded_files[0]).lower().endswith(".zip"):
            zip_path = uploaded_files[0]
            extracted = _safe_extract_zip(zip_path, target_dir)
        else:
            for src in uploaded_files:
                src_path = Path(src)
                if not src_path.exists():
                    continue
                dst = target_dir / src_path.name
                shutil.copy2(src_path, dst)
                copied += 1

        found = sorted({str(p.parent) for p in target_dir.rglob("adapter_config.json")})
        if not found:
            adapters = list_adapters(str(root_dir))
            return (
                f"Uploaded to {target_dir}, but no adapter_config.json found. "
                "Upload a valid LoRA adapter folder or zip.",
                gr.update(choices=adapters, value=adapters[0] if adapters else None),
            )

        adapters = list_adapters(str(root_dir))
        primary = found[0]
        msg = (
            f"Adapter upload complete. Copied {copied} file(s), extracted {extracted} archive entries. "
            f"Detected {len(found)} adapter path(s). Primary: {primary}"
        )
        return msg, gr.update(choices=adapters, value=primary)
    except Exception as exc:
        logger.exception("Adapter upload failed")
        adapters = list_adapters(str(root_dir))
        return f"Adapter upload failed: {exc}", gr.update(choices=adapters, value=adapters[0] if adapters else None)


@_gpu_callback
def load_adapter(adapter_path: str):
    global _last_loaded_adapter_path
    if not adapter_path or adapter_path == "(none found)":
        return "Select a valid adapter path."
    ok, msg = _ensure_model_ready_for_eval()
    if not ok:
        return msg
    status = handler.load_lora(adapter_path)
    if _looks_like_lora_loaded(status):
        _last_loaded_adapter_path = adapter_path
    if msg:
        return f"{msg}\n{status}"
    return status


@_gpu_callback
def unload_adapter():
    global _last_loaded_adapter_path
    ok, msg = _ensure_model_ready_for_eval()
    if not ok:
        return msg
    status = handler.unload_lora()
    if "unloaded" in (status or "").strip().lower():
        _last_loaded_adapter_path = None
    return status


def set_lora_scale(scale: float):
    return handler.set_lora_scale(scale)


@_gpu_callback
def generate_sample(
    prompt: str,
    lyrics: str,
    duration: float,
    bpm: int,
    steps: int,
    guidance: float,
    seed: int,
    use_lora: bool,
    lora_scale: float,
):
    """Generate a single audio sample for evaluation."""
    ok, msg = _ensure_model_ready_for_eval()
    if not ok:
        return None, msg
    ok_lora, lora_msg = _ensure_lora_ready_for_eval(use_lora)
    if not ok_lora:
        return None, lora_msg

    # Toggle LoRA if loaded
    if handler.lora_loaded:
        handler.set_use_lora(use_lora)
        if use_lora:
            handler.set_lora_scale(lora_scale)

    actual_seed = int(seed) if seed >= 0 else random.randint(0, 2**32 - 1)

    result = handler.generate_music(
        captions=prompt,
        lyrics=lyrics,
        bpm=bpm if bpm > 0 else None,
        inference_steps=steps,
        guidance_scale=guidance,
        use_random_seed=False,
        seed=actual_seed,
        audio_duration=duration,
        batch_size=1,
    )

    if not result.get("success", False):
        return None, result.get("error", "Generation failed.")

    audios = result.get("audios", [])
    if not audios:
        return None, "No audio produced."

    # Save to temp file
    audio_data = audios[0]
    wav_tensor = audio_data.get("tensor")
    sr = audio_data.get("sample_rate", 48000)

    if wav_tensor is None:
        path = audio_data.get("path")
        if path and os.path.exists(path):
            return path, f"Generated (from file), seed={actual_seed}."
        return None, "No audio tensor."

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_saver.save_audio(wav_tensor, tmp.name, sample_rate=sr)
    session_msgs = [m for m in (msg, lora_msg) if m]
    prefix = ("\n".join(session_msgs) + "\n") if session_msgs else ""
    return tmp.name, f"{prefix}Generated successfully, seed={actual_seed}."


@_gpu_callback
def ab_test(
    prompt, lyrics, duration, bpm, steps, guidance, seed,
    lora_scale_b,
):
    """Generate two samples: A = base, B = LoRA at given scale."""
    resolved_seed = int(seed) if seed >= 0 else random.randint(0, 2**32 - 1)
    results = {}
    for label, use, scale in [("A (base)", False, 0.0), ("B (LoRA)", True, lora_scale_b)]:
        path, msg = generate_sample(
            prompt, lyrics, duration, bpm, steps, guidance, resolved_seed,
            use_lora=use, lora_scale=scale,
        )
        results[label] = (path, msg)

    return (
        results["A (base)"][0],
        results["A (base)"][1],
        results["B (LoRA)"][0],
        results["B (LoRA)"][1],
    )


# ===========================================================================
# Build the Gradio App
# ===========================================================================

def get_workflow_status():
    global _last_loaded_adapter_path
    model_is_ready = (handler.model is not None) or _model_init_ok
    model_ready = "Ready" if model_is_ready else "Not initialized"
    tracks = len(dataset_entries)
    training_state = _training_status
    lora_status = handler.get_lora_status() if handler.model is not None else {"loaded": False, "active": False, "scale": 1.0}
    init_note = ""
    if IS_SPACE and _model_init_ok and handler.model is None:
        init_note = " (Zero GPU callback context)"
    adapter_hint = _last_loaded_adapter_path or "(none)"
    lora_loaded_text = str(lora_status.get('loaded', False))
    if IS_SPACE and handler.model is None and _last_loaded_adapter_path:
        lora_loaded_text = "Session-pending (auto-reload on eval)"
    return (
        f"Model: {model_ready}{init_note}\n"
        f"Tracks Loaded: {tracks}\n"
        f"Training: {training_state}\n"
        f"LoRA Loaded: {lora_loaded_text}\n"
        f"LoRA Active: {lora_status.get('active', False)}\n"
        f"LoRA Scale: {lora_status.get('scale', 1.0)}\n"
        f"Last Adapter Path: {adapter_hint}"
    )


def init_model_and_status(
    model_name: str,
    device: str,
    offload_cpu: bool,
    offload_dit_cpu: bool,
):
    status = init_model(model_name, device, offload_cpu, offload_dit_cpu)
    return status, get_workflow_status()


def build_ui():
    available_models = get_available_models()

    with gr.Blocks(title="ACE-Step 1.5 LoRA Studio", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# ACE-Step 1.5 LoRA Studio\n"
            "Use this guided workflow from left to right.\n\n"
            "**Step 1:** Initialize model  \n"
            "**Step 2:** Load dataset  \n"
            "**Step 3:** Start training  \n"
            "**Step 4:** Evaluate adapter"
        )
        with gr.Row():
            workflow_status = gr.Textbox(label="Workflow Status", value=get_workflow_status(), lines=6, interactive=False)
            refresh_status_btn = gr.Button("Refresh Status")
        refresh_status_btn.click(get_workflow_status, outputs=workflow_status, api_name="workflow_status")

        # ---- Step 1 ----
        with gr.Tab("Step 1 - Initialize Model"):
            gr.Markdown(
                "### Instructions\n"
                "1. Pick a model (`acestep-v15-base` recommended for LoRA).\n"
                "2. Keep device on `auto` unless you need manual override.\n"
                "3. Click **Initialize Model** and confirm status is success."
            )
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="DiT Model",
                )
                device_dd = gr.Dropdown(
                    choices=["auto", "cuda", "mps", "cpu"],
                    value="auto",
                    label="Device",
                )
            with gr.Row():
                offload_cb = gr.Checkbox(label="Offload To CPU (optional)", value=False)
                offload_dit_cb = gr.Checkbox(label="Offload DiT To CPU (optional)", value=False)
            init_btn = gr.Button("Initialize Model", variant="primary")
            init_out = gr.Textbox(label="Initialization Output", lines=8, interactive=False)
            init_btn.click(
                init_model_and_status,
                [model_dd, device_dd, offload_cb, offload_dit_cb],
                [init_out, workflow_status],
                api_name="init_model",
            )

        # ---- Step 2 ----
        with gr.Tab("Step 2 - Load Dataset"):
            gr.Markdown(
                "### Instructions\n"
                "1. Either scan a folder or drag/drop audio files (+ optional .json sidecars).\n"
                "2. Confirm tracks appear in the table.\n"
                "3. Optional: run Auto-Label All to fill caption/lyrics/metas.\n"
                "4. Optional: edit metadata manually and save sidecar JSON."
            )
            with gr.Row():
                folder_input = gr.Textbox(label="Dataset Folder Path", placeholder="e.g. ./dataset_inbox")
                scan_btn = gr.Button("Scan Folder")
            with gr.Row():
                upload_files = gr.Files(
                    label="Drag/Drop Audio Files (+ Optional JSON Sidecars)",
                    file_count="multiple",
                    file_types=["audio", ".json"],
                    type="filepath",
                )
                upload_btn = gr.Button("Load Dropped Files")
            scan_msg = gr.Textbox(label="Dataset Result", interactive=False)
            dataset_table = gr.Dataframe(
                headers=["File", "Duration", "Caption", "Lyrics", "Language"],
                datatype=["str", "str", "str", "str", "str"],
                label="Tracks",
                interactive=False,
            )
            scan_btn.click(
                scan_folder,
                folder_input,
                [scan_msg, dataset_table],
                api_name="scan_folder",
            )
            upload_btn.click(
                load_uploaded,
                upload_files,
                [scan_msg, dataset_table],
                api_name="load_uploaded",
            )

            with gr.Accordion("Auto-Label (ACE audio understanding)", open=False):
                gr.Markdown(
                    "Auto-label uses ACE: audio -> semantic codes -> metadata/lyrics.\n"
                    "Initialize LM first, then run Auto-Label All.\n"
                    "Use Caption-Only if your dataset has no lyrics.\n"
                    "On Zero GPU, process in smaller batches and click Auto-Label All repeatedly."
                )
                with gr.Row():
                    lm_model_dd = gr.Dropdown(
                        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"],
                        value="acestep-5Hz-lm-0.6B",
                        label="Auto-Label LM Model",
                    )
                    lm_backend_dd = gr.Dropdown(
                        choices=["pt", "vllm", "mlx"],
                        value="pt",
                        label="LM Backend",
                    )
                    lm_device_dd = gr.Dropdown(
                        choices=["auto", "cuda", "mps", "xpu", "cpu"],
                        value="auto",
                        label="LM Device",
                    )
                with gr.Row():
                    init_lm_btn = gr.Button("Initialize Auto-Label LM")
                    overwrite_cb = gr.Checkbox(label="Overwrite Existing Caption/Lyrics", value=False)
                    caption_only_cb = gr.Checkbox(label="Caption-Only (Skip Lyrics)", value=True)
                    auto_label_btn = gr.Button("Auto-Label All", variant="primary")
                with gr.Row():
                    max_files_per_run = gr.Slider(1, 25, value=6, step=1, label="Files Per Run (Zero GPU Safe)")
                    reset_cursor_cb = gr.Checkbox(label="Restart From First Track", value=False)
                lm_init_status = gr.Textbox(label="Auto-Label LM Status", lines=5, interactive=False)
                auto_label_status = gr.Textbox(label="Auto-Label Summary", interactive=False)
                auto_label_log = gr.Textbox(label="Auto-Label Log", lines=8, interactive=False)
                init_lm_btn.click(
                    init_auto_label_lm,
                    [lm_model_dd, lm_backend_dd, lm_device_dd],
                    lm_init_status,
                    api_name="init_auto_label_lm",
                )
                auto_label_btn.click(
                    auto_label_all,
                    [overwrite_cb, caption_only_cb, max_files_per_run, reset_cursor_cb],
                    [auto_label_status, dataset_table, auto_label_log],
                    api_name="auto_label_all",
                )

            with gr.Accordion("Optional: Edit Metadata Sidecar", open=False):
                with gr.Row():
                    edit_idx = gr.Number(label="Track Index (0-based)", value=0, precision=0)
                    edit_caption = gr.Textbox(label="Caption")
                    edit_lyrics = gr.Textbox(label="Lyrics", lines=3)
                with gr.Row():
                    edit_bpm = gr.Textbox(label="BPM", placeholder="e.g. 120")
                    edit_key = gr.Textbox(label="Key/Scale", placeholder="e.g. Am")
                    edit_lang = gr.Textbox(label="Language", value="en")
                save_btn = gr.Button("Save Sidecar")
                save_msg = gr.Textbox(label="Save Result", interactive=False)
                save_btn.click(
                    save_sidecar,
                    [edit_idx, edit_caption, edit_lyrics, edit_bpm, edit_key, edit_lang],
                    save_msg,
                    api_name="save_sidecar",
                )

        # ---- Step 3 ----
        with gr.Tab("Step 3 - Train LoRA"):
            gr.Markdown(
                "### Instructions\n"
                "1. Keep default settings for first run.\n"
                "2. Set output directory (defaults are good).\n"
                "3. Click **Start Training** and monitor logs/loss.\n"
                "4. Use **Stop Training** for graceful stop."
            )
            with gr.Row():
                t_epochs = gr.Slider(1, 500, value=50, step=1, label="Epochs")
                t_bs = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                t_accum = gr.Slider(1, 16, value=1, step=1, label="Grad Accumulation")
            with gr.Row():
                t_outdir = gr.Textbox(label="Output Directory", value=DEFAULT_OUTPUT_DIR)
                t_resume = gr.Textbox(label="Resume From Adapter Directory (optional)", value="")

            with gr.Accordion("Advanced Training Settings (optional)", open=False):
                with gr.Row():
                    t_rank = gr.Slider(4, 256, value=64, step=4, label="LoRA Rank")
                    t_alpha = gr.Slider(4, 256, value=64, step=4, label="LoRA Alpha")
                    t_dropout = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="LoRA Dropout")
                with gr.Row():
                    t_lr = gr.Number(label="Learning Rate", value=1e-4)
                    t_wd = gr.Number(label="Weight Decay", value=0.01)
                    t_optim = gr.Dropdown(["adamw", "adamw_8bit"], value="adamw_8bit", label="Optimizer")
                with gr.Row():
                    t_grad_norm = gr.Number(label="Max Grad Norm", value=1.0)
                    t_warmup = gr.Number(label="Warmup Ratio", value=0.03)
                    t_sched = gr.Dropdown(
                        ["constant_with_warmup", "linear", "cosine"],
                        value="constant_with_warmup",
                        label="Scheduler",
                    )
                with gr.Row():
                    t_save = gr.Slider(1, 100, value=10, step=1, label="Save Every N Epochs")
                    t_log = gr.Slider(1, 100, value=5, step=1, label="Log Every N Steps")
                    t_shift = gr.Number(label="Timestep Shift", value=3.0)
                    t_maxdur = gr.Number(label="Max Audio Duration (s)", value=240)

            with gr.Row():
                train_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
                poll_btn = gr.Button("Refresh Log")

            train_status = gr.Textbox(label="Training Log", lines=12, interactive=False)
            loss_chart = gr.LinePlot(
                x="Step",
                y="Loss",
                title="Training Loss",
                x_title="Step",
                y_title="Loss",
            )

            train_btn.click(
                start_training,
                [
                    t_rank, t_alpha, t_dropout,
                    t_lr, t_wd, t_optim,
                    t_grad_norm, t_warmup, t_sched,
                    t_epochs, t_bs, t_accum,
                    t_save, t_log, t_shift,
                    t_maxdur, t_outdir, t_resume,
                ],
                train_status,
                api_name="start_training",
            )
            stop_btn.click(stop_training, outputs=train_status, api_name="stop_training")

            def _poll_and_format():
                log_text, chart_data = poll_training()
                if chart_data:
                    import pandas as pd
                    df = pd.DataFrame(chart_data, columns=["Step", "Loss"])
                else:
                    import pandas as pd
                    df = pd.DataFrame({"Step": [], "Loss": []})
                return log_text, df

            poll_btn.click(_poll_and_format, outputs=[train_status, loss_chart], api_name="poll_training")

        # ---- Step 4 ----
        with gr.Tab("Step 4 - Evaluate"):
            gr.Markdown(
                "### Instructions\n"
                "1. Refresh adapter list and load a trained adapter.\n"
                "2. Run single generation or A/B test.\n"
                "3. Use same seed for fair comparison."
            )

            with gr.Accordion("Adapter Management", open=True):
                with gr.Row():
                    adapter_dir = gr.Textbox(label="Adapters Directory", value=DEFAULT_OUTPUT_DIR)
                    refresh_btn = gr.Button("Refresh List")
                adapter_dd = gr.Dropdown(label="Select Adapter", choices=[])
                with gr.Row():
                    upload_adapter_files_input = gr.Files(
                        label="Upload LoRA Adapter (.zip or adapter files)",
                        file_count="multiple",
                        file_types=[".zip", ".json", ".safetensors", ".bin", ".pt", ".pth"],
                        type="filepath",
                    )
                    upload_adapter_name = gr.Textbox(
                        label="Uploaded Adapter Name (optional)",
                        placeholder="my-lora-adapter",
                    )
                    upload_adapter_btn = gr.Button("Upload Adapter")
                with gr.Row():
                    load_btn = gr.Button("Load Adapter", variant="primary")
                    unload_btn = gr.Button("Unload Adapter")
                adapter_status = gr.Textbox(label="Adapter Status", interactive=False)

                def _refresh(d):
                    adapters = list_adapters(d)
                    return gr.update(choices=adapters, value=adapters[0] if adapters else None)

                refresh_btn.click(_refresh, adapter_dir, adapter_dd, api_name="list_adapters")
                upload_adapter_btn.click(
                    upload_adapter_files,
                    [upload_adapter_files_input, adapter_dir, upload_adapter_name],
                    [adapter_status, adapter_dd],
                    api_name="upload_adapter_files",
                )
                load_evt = load_btn.click(load_adapter, adapter_dd, adapter_status, api_name="load_adapter")
                load_evt.then(get_workflow_status, outputs=workflow_status)
                unload_evt = unload_btn.click(unload_adapter, outputs=adapter_status, api_name="unload_adapter")
                unload_evt.then(get_workflow_status, outputs=workflow_status)

            with gr.Accordion("Generation Settings", open=True):
                with gr.Row():
                    eval_prompt = gr.Textbox(label="Prompt / Caption", lines=2, placeholder="upbeat pop rock with electric guitar")
                    eval_lyrics = gr.Textbox(label="Lyrics", lines=3, placeholder="[Instrumental]")
                with gr.Row():
                    eval_dur = gr.Slider(10, 300, value=30, step=5, label="Duration (s)")
                    eval_bpm = gr.Number(label="BPM (0 = auto)", value=0)
                    eval_steps = gr.Slider(1, 100, value=8, step=1, label="Inference Steps")
                with gr.Row():
                    eval_guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance Scale")
                    eval_seed = gr.Number(label="Seed (-1 = random)", value=-1)

            with gr.Row():
                sg_use_lora = gr.Checkbox(label="Use LoRA", value=True)
                sg_scale = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="LoRA Scale")
                sg_btn = gr.Button("Generate", variant="primary")
            sg_audio = gr.Audio(label="Single Output", type="filepath")
            sg_msg = gr.Textbox(label="Generation Status", interactive=False)
            sg_btn.click(
                generate_sample,
                [eval_prompt, eval_lyrics, eval_dur, eval_bpm, eval_steps, eval_guidance, eval_seed, sg_use_lora, sg_scale],
                [sg_audio, sg_msg],
                api_name="generate_sample",
            )

            gr.Markdown("#### A/B Test (Base vs LoRA)")
            with gr.Row():
                ab_scale = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="LoRA Scale for B")
                ab_btn = gr.Button("Run A/B Test")
            with gr.Row():
                ab_audio_a = gr.Audio(label="A - Base", type="filepath")
                ab_audio_b = gr.Audio(label="B - Base + LoRA", type="filepath")
            with gr.Row():
                ab_msg_a = gr.Textbox(label="Status A", interactive=False)
                ab_msg_b = gr.Textbox(label="Status B", interactive=False)

            ab_btn.click(
                ab_test,
                [eval_prompt, eval_lyrics, eval_dur, eval_bpm, eval_steps, eval_guidance, eval_seed, ab_scale],
                [ab_audio_a, ab_msg_a, ab_audio_b, ab_msg_b],
                api_name="ab_test",
            )

    app.queue(default_concurrency_limit=1)
    return app


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

