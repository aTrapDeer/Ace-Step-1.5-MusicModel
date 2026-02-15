import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import torchaudio

# On Hugging Face Spaces Zero, `spaces` must be imported before CUDA-related modules.
if os.getenv("SPACE_ID"):
    try:
        import spaces  # noqa: F401
    except Exception:
        pass

from qwen_audio_captioning import (
    DEFAULT_ANALYSIS_PROMPT,
    DEFAULT_MODEL_ID,
    build_captioner,
    export_annotation_records,
    generate_track_annotation,
    list_audio_files,
)


IS_SPACE = bool(os.getenv("SPACE_ID"))
DEFAULT_EXPORT_DIR = "/data/qwen_annotations" if IS_SPACE else "qwen_annotations"

_captioner_cache: Dict[str, Any] = {"key": None, "obj": None}


def _audio_duration_sec(path: str) -> Optional[float]:
    try:
        info = torchaudio.info(path)
        if info.sample_rate <= 0:
            return None
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return None


def _dedupe_paths(paths: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        if not isinstance(p, str):
            continue
        pp = p.strip()
        if not pp:
            continue
        key = str(Path(pp).resolve()) if Path(pp).exists() else pp
        if key in seen:
            continue
        seen.add(key)
        out.append(pp)
    return out


def _files_table(paths: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for p in paths:
        duration = _audio_duration_sec(p)
        rows.append([
            Path(p).name,
            f"{duration:.2f}" if duration is not None else "?",
            p,
        ])
    return rows


def _records_table(records: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for rec in records:
        sidecar = rec.get("sidecar", {})
        analysis = sidecar.get("music_analysis", {})
        rows.append([
            Path(rec.get("audio_path", "")).name,
            f"{sidecar.get('duration', '?')}",
            str(analysis.get("segment_count", "?")),
            str(sidecar.get("bpm", "")),
            str(sidecar.get("keyscale", "")),
            str(sidecar.get("caption", ""))[:160],
            str(rec.get("status", "ok")),
        ])
    return rows


def _get_captioner(
    backend: str,
    model_id: str,
    endpoint_url: str,
    token: str,
    device: str,
    dtype: str,
):
    cache_key = (backend, model_id, endpoint_url, device, dtype, token if backend == "hf_endpoint" else "")
    if _captioner_cache["obj"] is not None and _captioner_cache["key"] == cache_key:
        return _captioner_cache["obj"]

    cap = build_captioner(
        backend=backend,
        model_id=model_id,
        endpoint_url=endpoint_url,
        token=token,
        device=device,
        torch_dtype=dtype,
    )
    _captioner_cache["obj"] = cap
    _captioner_cache["key"] = cache_key
    return cap


def scan_folder(folder_path: str, current_paths: List[str]):
    current_paths = current_paths or []
    if not folder_path or not Path(folder_path).is_dir():
        return "Provide a valid folder path.", current_paths, _files_table(current_paths)
    merged = _dedupe_paths(current_paths + list_audio_files(folder_path))
    return f"Loaded {len(merged)} audio files.", merged, _files_table(merged)


def add_uploaded(uploaded_paths: List[str], current_paths: List[str]):
    current_paths = current_paths or []
    uploaded_paths = uploaded_paths or []
    merged = _dedupe_paths(current_paths + uploaded_paths)
    if not merged:
        return "Upload one or more audio files first.", merged, _files_table(merged)
    return f"Loaded {len(merged)} audio files.", merged, _files_table(merged)


def clear_files():
    return "Cleared file list.", [], []


def load_existing_sidecars(audio_paths: List[str], records: List[Dict[str, Any]]):
    audio_paths = audio_paths or []
    records = records or []
    existing_by_path = {r.get("audio_path"): r for r in records}
    loaded = 0
    for audio_path in audio_paths:
        sidecar_path = Path(audio_path).with_suffix(".json")
        if not sidecar_path.exists():
            continue
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        existing_by_path[audio_path] = {
            "audio_path": audio_path,
            "sidecar": data,
            "status": "loaded-existing",
        }
        loaded += 1

    merged_records = list(existing_by_path.values())
    choices = [r.get("audio_path", "") for r in merged_records]
    return (
        f"Loaded {loaded} existing sidecar(s). Total editable records: {len(merged_records)}.",
        merged_records,
        _records_table(merged_records),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def run_analysis(
    audio_paths: List[str],
    backend: str,
    model_id: str,
    endpoint_url: str,
    token: str,
    device: str,
    dtype: str,
    prompt: str,
    segment_seconds: float,
    overlap_seconds: float,
    max_new_tokens: int,
    temperature: float,
    keep_raw_outputs: bool,
    existing_records: List[Dict[str, Any]],
):
    audio_paths = audio_paths or []
    existing_records = existing_records or []
    if not audio_paths:
        return (
            "No audio files loaded.",
            existing_records,
            _records_table(existing_records),
            gr.update(choices=[], value=None),
        )
    prompt = (prompt or "").strip() or DEFAULT_ANALYSIS_PROMPT

    captioner = _get_captioner(
        backend=backend,
        model_id=model_id or DEFAULT_MODEL_ID,
        endpoint_url=endpoint_url,
        token=token,
        device=device,
        dtype=dtype,
    )

    existing_by_path = {r.get("audio_path"): r for r in existing_records}
    failures: List[str] = []

    for audio_path in audio_paths:
        try:
            sidecar = generate_track_annotation(
                audio_path=audio_path,
                captioner=captioner,
                prompt=prompt,
                segment_seconds=float(segment_seconds),
                overlap_seconds=float(overlap_seconds),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                keep_raw_outputs=bool(keep_raw_outputs),
            )
            # Persist immediately so dataset folder stays LoRA-ready.
            Path(audio_path).with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            existing_by_path[audio_path] = {
                "audio_path": audio_path,
                "sidecar": sidecar,
                "status": "analyzed+saved",
            }
        except Exception as exc:
            failures.append(f"{Path(audio_path).name}: {exc}")
            fallback = existing_by_path.get(audio_path, {"audio_path": audio_path, "sidecar": {}})
            fallback["status"] = f"failed: {exc}"
            existing_by_path[audio_path] = fallback

    merged_records = list(existing_by_path.values())
    choices = [r.get("audio_path", "") for r in merged_records]
    message = (
        f"Analyzed {len(audio_paths)} file(s). "
        f"Failures: {len(failures)}."
    )
    if failures:
        message += "\n" + "\n".join(failures[:12])
    return (
        message,
        merged_records,
        _records_table(merged_records),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def load_record_json(selected_audio_path: str, records: List[Dict[str, Any]]):
    records = records or []
    if not selected_audio_path:
        return "{}", "", "", "", "", "", ""
    for rec in records:
        if rec.get("audio_path") == selected_audio_path:
            sidecar = rec.get("sidecar", {})
            return (
                json.dumps(sidecar, indent=2, ensure_ascii=False),
                str(sidecar.get("caption", "")),
                str(sidecar.get("lyrics", "")),
                str(sidecar.get("bpm", "")),
                str(sidecar.get("keyscale", "")),
                str(sidecar.get("vocal_language", "")),
                str(sidecar.get("duration", "")),
            )
    return "{}", "", "", "", "", "", ""


def save_record_json(
    selected_audio_path: str,
    edited_json: str,
    records: List[Dict[str, Any]],
):
    records = records or []
    if not selected_audio_path:
        return "Select a track first.", records, _records_table(records)
    try:
        payload = json.loads(edited_json)
        if not isinstance(payload, dict):
            return "Edited payload must be a JSON object.", records, _records_table(records)
    except Exception as exc:
        return f"Invalid JSON: {exc}", records, _records_table(records)

    updated = False
    for rec in records:
        if rec.get("audio_path") == selected_audio_path:
            rec["sidecar"] = payload
            rec["status"] = "edited+saved"
            updated = True
            break
    if not updated:
        records.append({"audio_path": selected_audio_path, "sidecar": payload, "status": "edited+saved"})

    # Persist edits next to source audio for LoRA-ready folder layout.
    Path(selected_audio_path).with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return "Saved edits and wrote sidecar next to source audio.", records, _records_table(records)


def export_records(
    records: List[Dict[str, Any]],
    output_dir: str,
    copy_audio: bool,
    write_inplace_sidecars: bool,
):
    records = records or []
    valid: List[Dict[str, Any]] = []
    for rec in records:
        if not rec.get("audio_path") or not isinstance(rec.get("sidecar"), dict):
            continue
        valid.append({"audio_path": rec["audio_path"], "sidecar": rec["sidecar"]})
    if not valid:
        return "No valid analyzed/edited records to export."

    out_dir = (output_dir or "").strip() or DEFAULT_EXPORT_DIR
    result = export_annotation_records(
        records=valid,
        output_dir=out_dir,
        copy_audio=bool(copy_audio),
        write_inplace_sidecars=bool(write_inplace_sidecars),
    )
    return (
        f"Exported {result['written_count']} sidecar(s).\n"
        f"Manifest: {result['manifest_path']}\n"
        f"Index: {result['index_path']}\n"
        f"Dataset root: {result['dataset_root'] or '(audio copy disabled)'}"
    )


def build_ui():
    with gr.Blocks(title="Qwen2-Audio Music Captioning", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Qwen2-Audio Music Captioning + Annotation Export\n"
            "Upload songs, run structured timestamped music analysis, optionally edit annotations, "
            "then export ACE-Step LoRA sidecars."
        )

        audio_paths_state = gr.State([])
        records_state = gr.State([])

        with gr.Tab("1) Load Audio"):
            with gr.Row():
                folder_input = gr.Textbox(label="Dataset Folder", placeholder="e.g. ./dataset_inbox")
                scan_btn = gr.Button("Scan Folder")
            with gr.Row():
                upload_files = gr.Files(
                    label="Upload Audio Files",
                    file_count="multiple",
                    file_types=["audio"],
                    type="filepath",
                )
                add_upload_btn = gr.Button("Add Uploaded Files")
                clear_btn = gr.Button("Clear")
            files_status = gr.Textbox(label="Load Status", interactive=False)
            files_table = gr.Dataframe(
                headers=["File", "Duration(s)", "Path"],
                datatype=["str", "str", "str"],
                label="Loaded Audio",
                interactive=False,
            )

            scan_btn.click(
                scan_folder,
                [folder_input, audio_paths_state],
                [files_status, audio_paths_state, files_table],
            )
            add_upload_btn.click(
                add_uploaded,
                [upload_files, audio_paths_state],
                [files_status, audio_paths_state, files_table],
            )
            clear_btn.click(
                clear_files,
                outputs=[files_status, audio_paths_state, files_table],
            )

        with gr.Tab("2) Run Qwen Captioning"):
            with gr.Row():
                backend_dd = gr.Dropdown(
                    choices=["local", "hf_endpoint"],
                    value="local",
                    label="Backend",
                )
                model_id = gr.Textbox(label="Model ID", value=DEFAULT_MODEL_ID)
                endpoint_url = gr.Textbox(label="HF Endpoint URL (for hf_endpoint backend)", value="")
            with gr.Row():
                hf_token = gr.Textbox(label="HF Token (optional)", type="password", value="")
                device_dd = gr.Dropdown(
                    choices=["auto", "cuda", "cpu", "mps"],
                    value="auto",
                    label="Local Device",
                )
                dtype_dd = gr.Dropdown(
                    choices=["auto", "float16", "bfloat16", "float32"],
                    value="auto",
                    label="Torch DType",
                )
            prompt_box = gr.Textbox(
                label="Analysis Prompt",
                lines=6,
                value=DEFAULT_ANALYSIS_PROMPT,
            )
            with gr.Row():
                segment_seconds = gr.Slider(10, 120, value=30, step=1, label="Segment Seconds")
                overlap_seconds = gr.Slider(0, 20, value=2, step=1, label="Overlap Seconds")
                max_new_tokens = gr.Slider(64, 2048, value=384, step=32, label="Max New Tokens")
            with gr.Row():
                temperature = gr.Slider(0.0, 1.2, value=0.1, step=0.05, label="Temperature")
                keep_raw = gr.Checkbox(value=True, label="Keep Raw Segment Responses In JSON")
                analyze_btn = gr.Button("Run Captioning", variant="primary")
            with gr.Row():
                load_existing_btn = gr.Button("Load Existing Sidecars")
            analysis_status = gr.Textbox(label="Analysis Status", lines=5, interactive=False)
            gr.Markdown("Sidecars are auto-saved next to each source audio file during analysis.")
            records_table = gr.Dataframe(
                headers=["File", "Duration", "Segments", "BPM", "Key", "Caption", "Status"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                label="Annotation Records",
            )
            track_selector = gr.Dropdown(choices=[], label="Select Track For Editing")

            analyze_btn.click(
                run_analysis,
                [
                    audio_paths_state,
                    backend_dd,
                    model_id,
                    endpoint_url,
                    hf_token,
                    device_dd,
                    dtype_dd,
                    prompt_box,
                    segment_seconds,
                    overlap_seconds,
                    max_new_tokens,
                    temperature,
                    keep_raw,
                    records_state,
                ],
                [analysis_status, records_state, records_table, track_selector],
            )
            load_existing_btn.click(
                load_existing_sidecars,
                [audio_paths_state, records_state],
                [analysis_status, records_state, records_table, track_selector],
            )

        with gr.Tab("3) Human Annotation + Export"):
            with gr.Row():
                load_record_btn = gr.Button("Load Selected JSON")
                save_record_btn = gr.Button("Save JSON Edits")
            json_editor = gr.Textbox(label="Editable Annotation JSON", lines=24)
            with gr.Row():
                caption_preview = gr.Textbox(label="Caption", interactive=False)
                bpm_preview = gr.Textbox(label="BPM", interactive=False)
                key_preview = gr.Textbox(label="Key/Scale", interactive=False)
            with gr.Row():
                lang_preview = gr.Textbox(label="Vocal Language", interactive=False)
                duration_preview = gr.Textbox(label="Duration", interactive=False)
                lyrics_preview = gr.Textbox(label="Lyrics", interactive=False)
            edit_status = gr.Textbox(label="Edit Status", interactive=False)
            gr.Markdown("Saving JSON edits also writes the sidecar next to the source audio file.")

            load_record_btn.click(
                load_record_json,
                [track_selector, records_state],
                [
                    json_editor,
                    caption_preview,
                    lyrics_preview,
                    bpm_preview,
                    key_preview,
                    lang_preview,
                    duration_preview,
                ],
            )
            save_record_btn.click(
                save_record_json,
                [track_selector, json_editor, records_state],
                [edit_status, records_state, records_table],
            )

            gr.Markdown("### Export LoRA-Ready Dataset")
            with gr.Row():
                export_dir = gr.Textbox(label="Export Directory", value=DEFAULT_EXPORT_DIR)
                copy_audio_cb = gr.Checkbox(value=True, label="Copy Audio Into Export Dataset")
                inplace_cb = gr.Checkbox(value=True, label="Also Write Sidecars Next To Source Audio")
                export_btn = gr.Button("Export", variant="primary")
            export_status = gr.Textbox(label="Export Status", lines=5, interactive=False)

            export_btn.click(
                export_records,
                [records_state, export_dir, copy_audio_cb, inplace_cb],
                export_status,
            )

    app.queue(default_concurrency_limit=1)
    return app


app = build_ui()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
