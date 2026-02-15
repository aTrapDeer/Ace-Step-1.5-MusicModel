"""
Qwen2-Audio captioning utilities for music annotation workflows.

This module supports:
1) Local inference with Qwen2-Audio models via transformers.
2) Remote inference via a Hugging Face Endpoint with a simple JSON contract.
3) Segment-based analysis with timestamped aggregation.
4) Export helpers for ACE-Step LoRA sidecars and manifest files.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torchaudio


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


DEFAULT_ANALYSIS_PROMPT = (
    "Analyze and detail the musical elements, tones, instruments, genre and effects. "
    "Describe the effects and mix of instruments and vocals. Vocals may use modern production "
    "techniques such as pitch correction and tuning effects. Explain how musical elements interact "
    "throughout the song with timestamps. Go in depth on vocal performance and musical writing. "
    "Be concise but detail-rich."
)

DEFAULT_LONG_ANALYSIS_PROMPT = (
    "Analyze the full song and return a concise but detailed timestamped prose breakdown. "
    "Use sections every 10 to 20 seconds (or major arrangement changes). For each section, "
    "describe vocals, instrumentation, genre cues, effects, mix/energy changes, and how elements "
    "interact. End with a short overall summary paragraph."
)


SEGMENT_JSON_SCHEMA_HINT = (
    'Return JSON only with keys: "segment_summary" (string), "section_label" (string), '
    '"genre" (array of strings), "instruments" (array of strings), "effects" (array of strings), '
    '"vocal_characteristics" (array of strings), "mix_notes" (array of strings), '
    '"interaction_notes" (string), "bpm_guess" (number or null), "key_guess" (string or ""), '
    '"notable_moments" (array of objects with "timestamp_sec" and "note").'
)


@dataclass
class SegmentResult:
    index: int
    start_sec: float
    end_sec: float
    prompt: str
    raw_response: str
    parsed: Dict[str, Any]


def list_audio_files(folder: str) -> List[str]:
    root = Path(folder)
    if not root.is_dir():
        return []
    files: List[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(str(path))
    return files


def _load_audio_with_fallback(path: str) -> Tuple[np.ndarray, int]:
    """Load audio to mono float32 numpy array with fallback decode path."""
    try:
        wav, sr = torchaudio.load(path)
        wav = wav.float().numpy()
        if wav.ndim == 1:
            mono = wav
        else:
            mono = wav.mean(axis=0)
        return mono.astype(np.float32), int(sr)
    except Exception as torchaudio_exc:
        try:
            audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
            mono = audio_np.mean(axis=1)
            return mono.astype(np.float32), int(sr)
        except Exception as sf_exc:
            # Last fallback: ffmpeg decode (works when local libsndfile lacks mp3 codec).
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_wav = tmp.name
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    tmp_wav,
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc.returncode != 0:
                    tail = (proc.stderr or "")[-800:]
                    raise RuntimeError(f"ffmpeg decode failed: {tail}")
                audio_np, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
                mono = audio_np.mean(axis=1)
                return mono.astype(np.float32), int(sr)
            except Exception as ffmpeg_exc:
                raise RuntimeError(
                    f"Audio decode failed for '{path}'. "
                    f"torchaudio_error={torchaudio_exc}; "
                    f"soundfile_error={sf_exc}; "
                    f"ffmpeg_error={ffmpeg_exc}"
                ) from ffmpeg_exc
            finally:
                try:
                    if "tmp_wav" in locals():
                        Path(tmp_wav).unlink(missing_ok=True)
                except Exception:
                    pass


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    audio, sr = _load_audio_with_fallback(path)
    if sr == target_sr:
        return audio, sr

    wav = torch_audio_from_numpy(audio)
    resampled = torchaudio.functional.resample(wav, sr, target_sr)
    return resampled.squeeze(0).cpu().numpy().astype(np.float32), target_sr


def torch_audio_from_numpy(audio: np.ndarray):
    import torch

    if audio.ndim != 1:
        raise ValueError(f"Expected mono waveform [T], got shape={audio.shape}")
    return torch.from_numpy(audio).unsqueeze(0)


def split_audio_segments(
    audio: np.ndarray,
    sample_rate: int,
    segment_seconds: float,
    overlap_seconds: float,
) -> List[Tuple[float, float, np.ndarray]]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= segment_seconds:
        raise ValueError("overlap_seconds must be smaller than segment_seconds")

    total_samples = int(audio.shape[0])
    segment_samples = max(1, int(round(segment_seconds * sample_rate)))
    step_samples = max(1, int(round((segment_seconds - overlap_seconds) * sample_rate)))

    segments: List[Tuple[float, float, np.ndarray]] = []
    start = 0
    idx = 0
    while start < total_samples:
        end = min(total_samples, start + segment_samples)
        seg_audio = audio[start:end]
        start_sec = start / sample_rate
        end_sec = end / sample_rate
        segments.append((start_sec, end_sec, seg_audio))
        idx += 1
        if end >= total_samples:
            break
        start = idx * step_samples
    return segments


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    # Direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Parse markdown code fence if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence_match:
        block = fence_match.group(1)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback: first brace-balanced object.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def _ensure_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    out: List[str] = []
    if isinstance(value, Sequence):
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
    deduped: List[str] = []
    seen = set()
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


_GENRE_KEYWORDS = [
    "pop",
    "rock",
    "hip-hop",
    "hip hop",
    "rap",
    "r&b",
    "rnb",
    "electronic",
    "edm",
    "trap",
    "house",
    "techno",
    "ambient",
    "indie",
    "soul",
    "jazz",
    "metal",
    "punk",
    "country",
    "lo-fi",
    "lofi",
    "drill",
]

_INSTRUMENT_KEYWORDS = [
    "drums",
    "kick",
    "snare",
    "hihat",
    "hi-hat",
    "808",
    "bass",
    "synth",
    "piano",
    "guitar",
    "electric guitar",
    "acoustic guitar",
    "strings",
    "pad",
    "lead",
    "pluck",
    "vocal",
    "choir",
]

_EFFECT_KEYWORDS = [
    "reverb",
    "delay",
    "distortion",
    "saturation",
    "autotune",
    "auto tune",
    "pitch correction",
    "compression",
    "eq",
    "sidechain",
    "chorus",
    "flanger",
    "phaser",
    "stereo widening",
]

_VOCAL_KEYWORDS = [
    "autotune",
    "auto tune",
    "pitch correction",
    "harmonies",
    "ad-libs",
    "ad libs",
    "falsetto",
    "breathy",
    "raspy",
    "processed vocals",
]


def _clean_model_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    # Remove repetitive leading boilerplate often produced when JSON is requested.
    s = re.sub(r"^\s*The output should be a JSON object with these fields\.?\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*This is the requested information for the given song segment:?\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*From\s+\d+(\.\d+)?s\s+to\s+\d+(\.\d+)?s\s*", "", s, flags=re.I)
    return s.strip()


def _extract_bpm_guess(text: str) -> Optional[float]:
    for pat in [r"\b(\d{2,3}(?:\.\d+)?)\s*bpm\b", r"\btempo\s*(?:of|is|:)?\s*(\d{2,3}(?:\.\d+)?)\b"]:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                val = float(m.group(1))
                if 30 <= val <= 300:
                    return val
            except Exception:
                continue
    return None


def _extract_key_guess(text: str) -> str:
    patterns = [
        r"\b([A-G](?:#|b)?\s*(?:major|minor))\b",
        r"\b([A-G](?:#|b)?m)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            key = m.group(1).strip()
            return key[0].upper() + key[1:]
    return ""


def _extract_keyword_hits(text: str, keywords: List[str]) -> List[str]:
    lower = text.lower()
    found: List[str] = []
    for kw in keywords:
        if kw.lower() in lower:
            label = kw.replace("rnb", "R&B").replace("hip-hop", "hip-hop")
            if label.lower() not in {x.lower() for x in found}:
                found.append(label)
    return found


class BaseCaptioner:
    backend_name = "base"
    model_id = DEFAULT_MODEL_ID

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        raise NotImplementedError


class LocalQwen2AudioCaptioner(BaseCaptioner):
    backend_name = "local"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._processor = None
        self._model = None

    def _load(self):
        if self._processor is not None and self._model is not None:
            return

        import torch
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Qwen2-Audio classes are unavailable. Install a recent transformers build "
                "(for example transformers>=4.53.0) and retry."
            ) from exc

        if self.torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        device_map = "auto" if self.device == "auto" else None
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
        )
        if device_map is None:
            if self.device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = self.device
            self._model.to(target_device)

    def _model_device(self):
        import torch

        if self._model is None:
            return torch.device("cpu")
        return next(self._model.parameters()).device

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        self._load()
        import torch

        conversation = [
            {"role": "system", "content": "You are a precise music analysis assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "local://segment.wav"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=text,
            audio=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        device = self._model_device()
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(device)

        do_sample = bool(temperature and temperature > 0)
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-5)

        with torch.no_grad():
            generated = self._model.generate(**inputs, **gen_kwargs)
        prompt_tokens = inputs["input_ids"].size(1)
        generated_new = generated[:, prompt_tokens:]
        text_out = self._processor.batch_decode(
            generated_new,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if not text_out.strip():
            text_out = self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return text_out.strip()


class HFEndpointCaptioner(BaseCaptioner):
    backend_name = "hf_endpoint"

    def __init__(
        self,
        endpoint_url: str,
        token: Optional[str] = None,
        model_id: str = DEFAULT_MODEL_ID,
        timeout_seconds: int = 180,
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required for HFEndpointCaptioner")
        self.endpoint_url = endpoint_url.strip()
        self.token = token or os.getenv("HF_TOKEN", "")
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        # Serialize to wav bytes for endpoint transport.
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        wav_bytes = buffer.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        payload = {
            "inputs": {
                "prompt": prompt,
                "audio_base64": audio_b64,
                "sample_rate": sample_rate,
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "model_id": self.model_id,
            }
        }

        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)

        # Accept common endpoint output shapes.
        if isinstance(data, dict):
            if isinstance(data.get("generated_text"), str):
                return data["generated_text"].strip()
            if isinstance(data.get("text"), str):
                return data["text"].strip()
            if isinstance(data.get("output_text"), str):
                return data["output_text"].strip()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
                return first["generated_text"].strip()
        return str(data).strip()


def build_segment_prompt(
    base_prompt: str,
    start_sec: float,
    end_sec: float,
) -> str:
    return (
        f"{base_prompt}\n\n"
        f"Analyze only the song segment from {start_sec:.2f}s to {end_sec:.2f}s.\n"
        "Use timestamp references in absolute song seconds.\n"
        f"{SEGMENT_JSON_SCHEMA_HINT}"
    )


def _make_fallback_segment_dict(raw_text: str) -> Dict[str, Any]:
    summary = _clean_model_text(raw_text)
    if not summary:
        summary = "No analysis generated."
    bpm_guess = _extract_bpm_guess(summary)
    key_guess = _extract_key_guess(summary)
    genres = _extract_keyword_hits(summary, _GENRE_KEYWORDS)
    instruments = _extract_keyword_hits(summary, _INSTRUMENT_KEYWORDS)
    effects = _extract_keyword_hits(summary, _EFFECT_KEYWORDS)
    vocal_chars = _extract_keyword_hits(summary, _VOCAL_KEYWORDS)
    return {
        "segment_summary": summary,
        "section_label": "",
        "genre": genres,
        "instruments": instruments,
        "effects": effects,
        "vocal_characteristics": vocal_chars,
        "mix_notes": [],
        "interaction_notes": summary,
        "bpm_guess": bpm_guess,
        "key_guess": key_guess,
        "notable_moments": [],
    }


def _parse_segment_output(raw_text: str) -> Dict[str, Any]:
    parsed = _extract_json_from_text(raw_text)
    if not parsed:
        return _make_fallback_segment_dict(raw_text)

    out = dict(parsed)
    out["segment_summary"] = str(out.get("segment_summary", "")).strip()
    out["section_label"] = str(out.get("section_label", "")).strip()
    out["genre"] = _ensure_string_list(out.get("genre"))
    out["instruments"] = _ensure_string_list(out.get("instruments"))
    out["effects"] = _ensure_string_list(out.get("effects"))
    out["vocal_characteristics"] = _ensure_string_list(out.get("vocal_characteristics"))
    out["mix_notes"] = _ensure_string_list(out.get("mix_notes"))
    out["interaction_notes"] = str(out.get("interaction_notes", "")).strip()
    out["bpm_guess"] = _float_or_none(out.get("bpm_guess"))
    out["key_guess"] = str(out.get("key_guess", "")).strip()

    nm = out.get("notable_moments")
    cleaned_nm: List[Dict[str, Any]] = []
    if isinstance(nm, Sequence):
        for item in nm:
            if not isinstance(item, dict):
                continue
            ts = _float_or_none(item.get("timestamp_sec"))
            note = str(item.get("note", "")).strip()
            if ts is None and not note:
                continue
            cleaned_nm.append({"timestamp_sec": ts, "note": note})
    out["notable_moments"] = cleaned_nm
    return out


def _pick_common_key(values: List[str]) -> str:
    counts: Dict[str, int] = {}
    first_original: Dict[str, str] = {}
    for v in values:
        s = (v or "").strip()
        if not s:
            continue
        k = s.lower()
        counts[k] = counts.get(k, 0) + 1
        if k not in first_original:
            first_original[k] = s
    if not counts:
        return ""
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return first_original[best]


def _collect_unique(items: List[List[str]], limit: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in items:
        for item in group:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item.strip())
            if len(out) >= limit:
                return out
    return out


def _derive_caption(genres: List[str], instruments: List[str], vocals: List[str]) -> str:
    parts: List[str] = []
    if genres:
        parts.append(", ".join(genres[:2]))
    if instruments:
        parts.append("with " + ", ".join(instruments[:3]))
    if vocals:
        parts.append("and modern processed vocals")
    if not parts:
        return "music track with detailed arrangement and production dynamics"
    return " ".join(parts)


def generate_track_annotation(
    audio_path: str,
    captioner: BaseCaptioner,
    prompt: str = DEFAULT_ANALYSIS_PROMPT,
    segment_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
    max_new_tokens: int = 384,
    temperature: float = 0.1,
    keep_raw_outputs: bool = True,
    include_long_analysis: bool = False,
    long_analysis_prompt: str = DEFAULT_LONG_ANALYSIS_PROMPT,
    long_analysis_max_new_tokens: int = 1200,
    long_analysis_temperature: float = 0.1,
) -> Dict[str, Any]:
    audio, sr = load_audio_mono(audio_path, target_sr=16000)
    duration_sec = float(audio.shape[0]) / float(sr) if sr > 0 else 0.0
    segments = split_audio_segments(
        audio=audio,
        sample_rate=sr,
        segment_seconds=segment_seconds,
        overlap_seconds=overlap_seconds,
    )

    results: List[SegmentResult] = []
    for idx, (start_sec, end_sec, seg_audio) in enumerate(segments):
        seg_prompt = build_segment_prompt(prompt, start_sec=start_sec, end_sec=end_sec)
        raw = captioner.generate(
            audio=seg_audio,
            sample_rate=sr,
            prompt=seg_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed = _parse_segment_output(raw)
        results.append(
            SegmentResult(
                index=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                prompt=seg_prompt,
                raw_response=raw,
                parsed=parsed,
            )
        )

    timeline: List[Dict[str, Any]] = []
    all_genres: List[List[str]] = []
    all_instruments: List[List[str]] = []
    all_effects: List[List[str]] = []
    all_vocals: List[List[str]] = []
    all_mix_notes: List[List[str]] = []
    bpm_values: List[float] = []
    keys: List[str] = []
    interaction_summary: List[str] = []

    for seg in results:
        p = seg.parsed
        all_genres.append(_ensure_string_list(p.get("genre")))
        all_instruments.append(_ensure_string_list(p.get("instruments")))
        all_effects.append(_ensure_string_list(p.get("effects")))
        all_vocals.append(_ensure_string_list(p.get("vocal_characteristics")))
        all_mix_notes.append(_ensure_string_list(p.get("mix_notes")))

        bpm = _float_or_none(p.get("bpm_guess"))
        if bpm is not None and bpm > 0:
            bpm_values.append(bpm)
        key_guess = str(p.get("key_guess", "")).strip()
        if key_guess:
            keys.append(key_guess)
        if p.get("interaction_notes"):
            interaction_summary.append(str(p["interaction_notes"]).strip())

        timeline_entry = {
            "segment_index": seg.index,
            "start_sec": round(seg.start_sec, 3),
            "end_sec": round(seg.end_sec, 3),
            "section_label": str(p.get("section_label", "")).strip(),
            "segment_summary": str(p.get("segment_summary", "")).strip(),
            "instruments": _ensure_string_list(p.get("instruments")),
            "effects": _ensure_string_list(p.get("effects")),
            "vocal_characteristics": _ensure_string_list(p.get("vocal_characteristics")),
            "interaction_notes": str(p.get("interaction_notes", "")).strip(),
            "mix_notes": _ensure_string_list(p.get("mix_notes")),
            "notable_moments": p.get("notable_moments", []),
        }
        if keep_raw_outputs:
            timeline_entry["raw_response"] = seg.raw_response
        timeline.append(timeline_entry)

    genres = _collect_unique(all_genres, limit=10)
    instruments = _collect_unique(all_instruments, limit=16)
    effects = _collect_unique(all_effects, limit=16)
    vocal_traits = _collect_unique(all_vocals, limit=12)
    mix_notes = _collect_unique(all_mix_notes, limit=24)
    keyscale = _pick_common_key(keys)
    bpm = int(round(sum(bpm_values) / len(bpm_values))) if bpm_values else None
    caption = _derive_caption(genres=genres, instruments=instruments, vocals=vocal_traits)

    sidecar: Dict[str, Any] = {
        "caption": caption,
        "lyrics": "",
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": "4/4",
        "vocal_language": "unknown",
        "duration": round(duration_sec, 3),
        "annotation_version": "qwen2_audio_music_v1",
        "source_audio": str(audio_path),
        "analysis_prompt": prompt,
        "analysis_backend": captioner.backend_name,
        "analysis_model": captioner.model_id,
        "analysis_generated_at": datetime.now(timezone.utc).isoformat(),
        "music_analysis": {
            "genres": genres,
            "instruments": instruments,
            "effects": effects,
            "vocal_characteristics": vocal_traits,
            "mix_notes": mix_notes,
            "interaction_summary": interaction_summary,
            "timeline": timeline,
            "segment_seconds": segment_seconds,
            "overlap_seconds": overlap_seconds,
            "segment_count": len(timeline),
        },
    }

    if include_long_analysis:
        long_prompt = (long_analysis_prompt or "").strip() or DEFAULT_LONG_ANALYSIS_PROMPT
        try:
            long_raw = captioner.generate(
                audio=audio,
                sample_rate=sr,
                prompt=long_prompt,
                max_new_tokens=int(long_analysis_max_new_tokens),
                temperature=float(long_analysis_temperature),
            )
            long_text = _clean_model_text(long_raw)
            sidecar["analysis_long_prompt"] = long_prompt
            sidecar["analysis_long"] = long_text
            sidecar["music_analysis"]["summary_long"] = long_text
        except Exception as exc:
            sidecar["analysis_long_prompt"] = long_prompt
            sidecar["analysis_long"] = ""
            sidecar["analysis_long_error"] = str(exc)

    return sidecar


def build_captioner(
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    endpoint_url: str = "",
    token: str = "",
    device: str = "auto",
    torch_dtype: str = "auto",
) -> BaseCaptioner:
    backend = (backend or "").strip().lower()
    if backend in {"local", "hf_space_local"}:
        return LocalQwen2AudioCaptioner(
            model_id=model_id or DEFAULT_MODEL_ID,
            device=device,
            torch_dtype=torch_dtype,
        )
    if backend in {"endpoint", "hf_endpoint"}:
        return HFEndpointCaptioner(
            endpoint_url=endpoint_url,
            token=token,
            model_id=model_id or DEFAULT_MODEL_ID,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def export_annotation_records(
    records: List[Dict[str, Any]],
    output_dir: str,
    copy_audio: bool = True,
    write_inplace_sidecars: bool = True,
) -> Dict[str, Any]:
    """
    Export analyzed tracks as LoRA-ready sidecars + manifest.

    records item schema:
      {
        "audio_path": "...",
        "sidecar": {...annotation json...}
      }
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_root = out_root / "dataset"
    if copy_audio:
        dataset_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "annotations_manifest.jsonl"
    index_path = out_root / "annotations_index.json"

    manifest_lines: List[str] = []
    index_items: List[Dict[str, Any]] = []
    written_count = 0

    for rec in records:
        src_audio = Path(rec["audio_path"])
        sidecar = dict(rec["sidecar"])
        if not src_audio.exists():
            continue

        if copy_audio:
            dst_audio = dataset_root / src_audio.name
            if src_audio.resolve() != dst_audio.resolve():
                shutil.copy2(src_audio, dst_audio)
            dst_sidecar = dst_audio.with_suffix(".json")
        else:
            dst_sidecar = (out_root / src_audio.name).with_suffix(".json")

        dst_sidecar.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
        written_count += 1

        if write_inplace_sidecars:
            inplace_sidecar = src_audio.with_suffix(".json")
            inplace_sidecar.write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        manifest_row = {
            "audio_path": str(dst_sidecar.with_suffix(src_audio.suffix).as_posix()) if copy_audio else str(src_audio),
            "sidecar_path": str(dst_sidecar),
            "caption": sidecar.get("caption", ""),
            "duration": sidecar.get("duration"),
            "bpm": sidecar.get("bpm"),
            "keyscale": sidecar.get("keyscale", ""),
        }
        manifest_lines.append(json.dumps(manifest_row, ensure_ascii=False))
        index_items.append(
            {
                "source_audio": str(src_audio),
                "exported_sidecar": str(dst_sidecar),
                "caption": sidecar.get("caption", ""),
            }
        )

    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "records": index_items,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "written_count": written_count,
        "manifest_path": str(manifest_path),
        "index_path": str(index_path),
        "dataset_root": str(dataset_root) if copy_audio else "",
    }


def read_prompt_file(prompt_file: str) -> str:
    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return text
