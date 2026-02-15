"""
Audio Flamingo 3 -> ChatGPT cleanup pipeline for Ace Step 1.5 LoRA metadata.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import soundfile as sf

from qwen_audio_captioning import AUDIO_EXTENSIONS, load_audio_mono


DEFAULT_AF3_MODEL_ID = "nvidia/audio-flamingo-3-hf"
DEFAULT_AF3_PROMPT = (
    "Analyze this full song and provide concise, timestamped sections describing vocals, "
    "instrumentation, production effects, mix changes, energy flow, and genre cues. End with "
    "a short overall summary."
)
DEFAULT_AF3_PROMPT_THINK_LONG = (
    "Analyze the entire song from start to finish and produce a detailed, timestamped breakdown. "
    "Cover the full duration with many sections, describing vocals, instrumentation, effects, mix, "
    "arrangement, and energy transitions. Include notable moments and end with a concise overall summary."
)
DEFAULT_OPENAI_MODEL = "gpt-5-mini"


LUNA_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "lyrics": {"type": "string"},
        "bpm": {"type": ["integer", "null"]},
        "keyscale": {"type": "string"},
        "timesignature": {"type": "string"},
        "vocal_language": {"type": "string"},
        "duration": {"type": "number"},
        "analysis_short": {"type": "string"},
        "analysis_long": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "summary": {"type": "string"},
                    "vocal_notes": {"type": "string"},
                    "instrument_notes": {"type": "string"},
                    "effects": {"type": "array", "items": {"type": "string"}},
                    "mix_notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "start_sec",
                    "end_sec",
                    "summary",
                    "vocal_notes",
                    "instrument_notes",
                    "effects",
                    "mix_notes",
                ],
                "additionalProperties": False,
            },
        },
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "caption",
        "lyrics",
        "bpm",
        "keyscale",
        "timesignature",
        "vocal_language",
        "duration",
        "analysis_short",
        "analysis_long",
        "sections",
        "tags",
    ],
    "additionalProperties": False,
}


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
                break
    raise ValueError("Failed to parse JSON object from model output")


def _ensure_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _ensure_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ensure_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        iv = int(float(value))
    except Exception:
        return None
    if iv <= 0:
        return None
    return iv


def _ensure_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        s = _ensure_str(item)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _normalize_sections(sections: Any, duration: float) -> List[Dict[str, Any]]:
    if not isinstance(sections, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections):
        if not isinstance(sec, dict):
            continue
        start = _ensure_float(sec.get("start_sec"), default=0.0)
        end = _ensure_float(sec.get("end_sec"), default=start)
        if end < start:
            end = start
        if duration > 0:
            start = max(0.0, min(start, duration))
            end = max(start, min(end, duration))
        out.append(
            {
                "index": idx,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "summary": _ensure_str(sec.get("summary")),
                "vocal_notes": _ensure_str(sec.get("vocal_notes")),
                "instrument_notes": _ensure_str(sec.get("instrument_notes")),
                "effects": _ensure_str_list(sec.get("effects")),
                "mix_notes": _ensure_str_list(sec.get("mix_notes")),
            }
        )
    return out


def _audio_to_wav_base64(audio_path: str, sample_rate: int = 16000) -> str:
    audio, sr = load_audio_mono(audio_path, target_sr=sample_rate)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class AF3EndpointClient:
    backend_name: ClassVar[str] = "hf_endpoint"
    endpoint_url: str
    token: str
    model_id: str = DEFAULT_AF3_MODEL_ID
    timeout_seconds: int = 300

    def analyze(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        audio_b64 = _audio_to_wav_base64(audio_path, sample_rate=16000)
        payload = {
            "inputs": {
                "prompt": prompt,
                "audio_base64": audio_b64,
                "sample_rate": 16000,
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "model_id": self.model_id,
            }
        }
        req = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
            return data["generated_text"].strip()
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return data["text"].strip()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
                return first["generated_text"].strip()
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return first["text"].strip()
        return str(data).strip()


@dataclass
class AF3LocalClient:
    backend_name: ClassVar[str] = "local"
    model_id: str = DEFAULT_AF3_MODEL_ID
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    def __post_init__(self):
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None and self._processor is not None:
            return
        import torch

        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
            model_cls = AudioFlamingo3ForConditionalGeneration
        except Exception as exc:
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
                model_cls = AutoModelForImageTextToText
            except Exception:
                raise RuntimeError(
                    "Audio Flamingo 3 classes are unavailable. Install transformers>=4.57.0."
                ) from exc

        if self.torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        self._model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "auto" else None,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device != "auto":
            self._model.to(self.device)

    def analyze(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        self._load()
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        device = next(self._model.parameters()).device
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature > 0),
        }
        if temperature > 0:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        start = int(inputs["input_ids"].shape[1])
        text = self._processor.batch_decode(outputs[:, start:], skip_special_tokens=True)[0].strip()
        if not text:
            text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return text


def cleanup_with_chatgpt(
    af3_text: str,
    *,
    openai_api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    duration: float = 0.0,
    user_context: str = "",
    artist_name: str = "",
    track_name: str = "",
    enable_web_search: bool = False,
) -> Dict[str, Any]:
    if not openai_api_key:
        raise ValueError("openai_api_key is required for ChatGPT cleanup.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed. Add `openai` to dependencies.") from exc

    client = OpenAI(api_key=openai_api_key)

    system = (
        "You transform raw audio-analysis prose into high-quality LoRA training metadata for Ace Step 1.5. "
        "Always return compact, truthful JSON. Never invent precise music facts not supported by input."
    )
    user = (
        f"Raw AF3 analysis:\n{af3_text}\n\n"
        f"Track duration seconds: {duration}\n"
        f"Artist (optional): {artist_name or 'unknown'}\n"
        f"Track name (optional): {track_name or 'unknown'}\n"
        f"User context (optional): {user_context or 'none'}\n\n"
        "Return output matching the JSON schema exactly. "
        "Keep caption concise and useful for LoRA conditioning."
    )
    if hasattr(client, "responses"):
        req_kwargs: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ace_step_luna_metadata",
                    "schema": LUNA_OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        }
        if enable_web_search:
            req_kwargs["tools"] = [{"type": "web_search"}]

        try:
            response = client.responses.create(**req_kwargs)
        except Exception:
            if enable_web_search:
                # Fallback for SDK/runtime variants that still use the preview tool id.
                req_kwargs["tools"] = [{"type": "web_search_preview"}]
                response = client.responses.create(**req_kwargs)
            else:
                raise

        output_text = getattr(response, "output_text", "") or ""
        if not output_text and hasattr(response, "output"):
            chunks: List[str] = []
            for item in getattr(response, "output", []):
                for content in getattr(item, "content", []):
                    text_val = getattr(content, "text", None)
                    if text_val:
                        chunks.append(str(text_val))
            output_text = "\n".join(chunks).strip()
    else:
        if enable_web_search:
            raise RuntimeError(
                "enable_web_search requires an OpenAI SDK/runtime with Responses API support. "
                "Upgrade openai package to a recent version."
            )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ace_step_luna_metadata",
                        "schema": LUNA_OUTPUT_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except Exception:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            user
                            + "\n\nReturn valid JSON with keys exactly matching this set: "
                            "caption, lyrics, bpm, keyscale, timesignature, vocal_language, "
                            "duration, analysis_short, analysis_long, sections, tags."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
        output_text = ""
        if getattr(response, "choices", None):
            message = response.choices[0].message
            output_text = getattr(message, "content", "") or ""

    cleaned = _extract_json_object(output_text)
    return cleaned


def build_lora_sidecar(
    cleaned: Dict[str, Any],
    *,
    af3_text: str,
    af3_prompt: str,
    af3_backend: str,
    af3_model_id: str,
    source_audio: str,
    duration: float,
    chatgpt_model: str,
    web_search_used: bool,
) -> Dict[str, Any]:
    caption = _ensure_str(cleaned.get("caption"), "music track with evolving arrangement")
    lyrics = _ensure_str(cleaned.get("lyrics"), "")
    bpm = _ensure_int_or_none(cleaned.get("bpm"))
    keyscale = _ensure_str(cleaned.get("keyscale"), "")
    timesignature = _ensure_str(cleaned.get("timesignature"), "4/4") or "4/4"
    vocal_language = _ensure_str(cleaned.get("vocal_language"), "unknown") or "unknown"
    duration_val = _ensure_float(cleaned.get("duration"), duration)
    analysis_short = _ensure_str(cleaned.get("analysis_short"), caption)
    analysis_long = _ensure_str(cleaned.get("analysis_long"), af3_text)
    sections = _normalize_sections(cleaned.get("sections"), duration=duration_val)
    tags = _ensure_str_list(cleaned.get("tags"))

    sidecar: Dict[str, Any] = {
        "caption": caption,
        "lyrics": lyrics,
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": timesignature,
        "vocal_language": vocal_language,
        "duration": round(duration_val, 3),
        "analysis_short": analysis_short,
        "analysis_long": analysis_long,
        "source_audio": source_audio,
        "annotation_version": "af3_chatgpt_luna_v1",
        "music_analysis": {
            "timeline": sections,
            "tags": tags,
            "summary_long": analysis_long,
            "segment_count": len(sections),
        },
        "pipeline": {
            "af3_prompt": af3_prompt,
            "af3_backend": af3_backend,
            "af3_model_id": af3_model_id,
            "af3_raw_analysis": af3_text,
            "chatgpt_model": chatgpt_model,
            "chatgpt_web_search_used": bool(web_search_used),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return sidecar


def run_af3_chatgpt_pipeline(
    *,
    audio_path: str,
    af3_client: Any,
    af3_prompt: str = DEFAULT_AF3_PROMPT,
    af3_max_new_tokens: int = 1400,
    af3_temperature: float = 0.1,
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_MODEL,
    user_context: str = "",
    artist_name: str = "",
    track_name: str = "",
    enable_web_search: bool = False,
) -> Dict[str, Any]:
    audio_path = str(Path(audio_path))
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if Path(audio_path).suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio extension: {Path(audio_path).suffix}")

    audio, sr = load_audio_mono(audio_path, target_sr=16000)
    duration = (float(audio.shape[0]) / float(sr)) if sr > 0 else 0.0

    af3_text = af3_client.analyze(
        audio_path=audio_path,
        prompt=af3_prompt,
        max_new_tokens=af3_max_new_tokens,
        temperature=af3_temperature,
    )
    cleaned = cleanup_with_chatgpt(
        af3_text,
        openai_api_key=openai_api_key,
        model=openai_model,
        duration=duration,
        user_context=user_context,
        artist_name=artist_name,
        track_name=track_name,
        enable_web_search=enable_web_search,
    )
    sidecar = build_lora_sidecar(
        cleaned,
        af3_text=af3_text,
        af3_prompt=af3_prompt,
        af3_backend=getattr(af3_client, "backend_name", type(af3_client).__name__),
        af3_model_id=getattr(af3_client, "model_id", DEFAULT_AF3_MODEL_ID),
        source_audio=audio_path,
        duration=duration,
        chatgpt_model=openai_model,
        web_search_used=enable_web_search,
    )
    return {
        "af3_analysis": af3_text,
        "cleaned": cleaned,
        "sidecar": sidecar,
    }


def save_sidecar(sidecar: Dict[str, Any], output_json: str) -> str:
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)
