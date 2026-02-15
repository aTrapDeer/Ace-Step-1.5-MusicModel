#!/usr/bin/env python
"""
Send one audio file to a Qwen caption endpoint and print/save the response.

Request contract expected by templates/hf-qwen-caption-endpoint/handler.py:
{
  "inputs": {
    "prompt": "...",
    "audio_base64": "...",
    "sample_rate": 16000,
    "max_new_tokens": 384,
    "temperature": 0.1
  }
}
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import soundfile as sf

from qwen_audio_captioning import DEFAULT_ANALYSIS_PROMPT, load_audio_mono


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


def load_audio_b64(audio_path: str, target_sr: int) -> str:
    mono, sr = load_audio_mono(audio_path, target_sr=target_sr)

    import io

    buf = io.BytesIO()
    sf.write(buf, mono, int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def send(url: str, token: str, payload: dict) -> dict:
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=600) as resp:
            text = resp.read().decode("utf-8")
        return json.loads(text)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Qwen caption endpoint")
    parser.add_argument(
        "--url",
        default=os.getenv("HF_QWEN_ENDPOINT_URL", "") or read_dotenv_value(".env", "HF_QWEN_ENDPOINT_URL"),
        required=False,
    )
    parser.add_argument(
        "--token",
        default=(
            os.getenv("HF_TOKEN", "")
            or os.getenv("hf_token", "")
            or read_dotenv_value(".env", "HF_TOKEN")
            or read_dotenv_value(".env", "hf_token")
        ),
        required=False,
    )
    parser.add_argument("--audio", required=True, help="Path to local audio file")
    parser.add_argument("--prompt", default=DEFAULT_ANALYSIS_PROMPT)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    if not args.url:
        raise RuntimeError("Missing endpoint URL. Pass --url or set HF_QWEN_ENDPOINT_URL.")
    if not args.token:
        raise RuntimeError("Missing HF token. Pass --token or set HF_TOKEN.")
    if not Path(args.audio).is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    audio_b64 = load_audio_b64(args.audio, target_sr=args.sample_rate)
    payload = {
        "inputs": {
            "prompt": args.prompt,
            "audio_base64": audio_b64,
            "sample_rate": args.sample_rate,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
    }

    result = send(args.url, args.token, payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved: {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
