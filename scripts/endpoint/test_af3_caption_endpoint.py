#!/usr/bin/env python
"""
Send one audio file to an Audio Flamingo 3 endpoint and print/save the response.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from af3_chatgpt_pipeline import DEFAULT_AF3_PROMPT, DEFAULT_AF3_PROMPT_THINK_LONG
from qwen_audio_captioning import load_audio_mono
from utils.env_config import get_env, load_project_env


def load_audio_b64(audio_path: str, target_sr: int = 16000) -> str:
    mono, sr = load_audio_mono(audio_path, target_sr=target_sr)
    buf = io.BytesIO()
    sf.write(buf, mono, int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def send(url: str, token: str, payload: dict) -> dict:
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            **({"Authorization": f"Bearer {token}"} if token else {}),
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=600) as resp:
            text = resp.read().decode("utf-8")
        return json.loads(text)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        lower = body.lower()
        if "endpoint is in error" in lower:
            body += (
                "\nHint: open the endpoint page and restart/redeploy. "
                "This is a remote runtime failure, not a local script issue."
            )
        if "no custom pipeline found" in lower:
            body += (
                "\nHint: endpoint repo root must contain handler.py; "
                "ensure you deployed templates/hf-af3-caption-endpoint files."
            )
        if "audioflamingo3" in lower and "does not recognize" in lower:
            body += (
                "\nHint: runtime transformers is too old. "
                "Use templates/hf-af3-caption-endpoint/handler.py bootstrap runtime "
                "(AF3_TRANSFORMERS_SPEC=transformers==5.1.0) and redeploy."
            )
        if "failed to load af3 processor classes after runtime bootstrap" in lower:
            body += (
                "\nHint: endpoint startup could not install/load AF3 runtime deps. "
                "Check startup logs for pip/network/disk issues and keep task=custom."
            )
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e}") from e


def main() -> int:
    load_project_env()
    parser = argparse.ArgumentParser(description="Test AF3 caption endpoint")
    parser.add_argument(
        "--url",
        default=get_env("HF_AF3_ENDPOINT_URL", "hf_af3_endpoint_url"),
        required=False,
    )
    parser.add_argument(
        "--token",
        default=get_env("HF_TOKEN", "hf_token"),
        required=False,
    )
    parser.add_argument("--audio", required=True, help="Path to local audio file")
    parser.add_argument("--prompt", default=DEFAULT_AF3_PROMPT)
    parser.add_argument(
        "--mode",
        choices=["auto", "think", "single"],
        default="auto",
        help="Optional AF3 mode selector for NVIDIA-stack endpoints.",
    )
    parser.add_argument(
        "--think-long",
        action="store_true",
        help="Use long-form AF3 prompt + higher token budget defaults.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    if not args.url:
        raise RuntimeError("Missing endpoint URL. Pass --url or set HF_AF3_ENDPOINT_URL.")
    if not Path(args.audio).is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    audio_b64 = load_audio_b64(args.audio, target_sr=16000)
    prompt = args.prompt
    max_new_tokens = int(args.max_new_tokens)
    temperature = float(args.temperature)
    if args.think_long:
        if prompt == DEFAULT_AF3_PROMPT:
            prompt = DEFAULT_AF3_PROMPT_THINK_LONG
        if max_new_tokens == 1400:
            max_new_tokens = 3200
        if abs(temperature - 0.1) < 1e-9:
            temperature = 0.2

    payload = {
        "inputs": {
            "prompt": prompt,
            "audio_base64": audio_b64,
            "sample_rate": 16000,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
    }
    if args.mode != "auto":
        payload["inputs"]["think_mode"] = bool(args.mode == "think")

    result = send(args.url, args.token, payload)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    try:
        print(rendered)
    except UnicodeEncodeError:
        # Fallback for Windows cp1252 terminals when model emits non-ASCII punctuation.
        print(json.dumps(result, indent=2, ensure_ascii=True))
    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved: {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
