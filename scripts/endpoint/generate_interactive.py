import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "https://your-endpoint-url.endpoints.huggingface.cloud"
DEFAULT_SAMPLE_RATE = 44100


def read_dotenv_value(key: str, dotenv_path: str = ".env") -> str:
    path = Path(dotenv_path)
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def prompt_text(label: str, default: str = "", required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        if not value:
            value = default
        if value or not required:
            return value
        print("Value required.")


def prompt_int(label: str, default: int | None = None, allow_blank: bool = False) -> int | None:
    while True:
        default_str = "" if default is None else str(default)
        value = prompt_text(label, default_str, required=not allow_blank)
        if not value and allow_blank:
            return None
        try:
            return int(value)
        except ValueError:
            print("Enter a valid integer.")


def prompt_float(label: str, default: float) -> float:
    while True:
        value = prompt_text(label, str(default), required=True)
        try:
            return float(value)
        except ValueError:
            print("Enter a valid number.")


def prompt_yes_no(label: str, default: bool) -> bool:
    default_text = "y" if default else "n"
    while True:
        value = prompt_text(f"{label} (y/n)", default_text, required=True).lower()
        if value in {"y", "yes", "1", "true", "t"}:
            return True
        if value in {"n", "no", "0", "false", "f"}:
            return False
        print("Please answer y or n.")


def prompt_multiline(label: str, end_token: str = "END") -> str:
    print(label)
    print(f"Finish lyrics by typing {end_token} on its own line.")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_lyrics_optional() -> str:
    use_lyrics = prompt_yes_no("Add custom lyrics", True)
    if not use_lyrics:
        return ""
    return prompt_multiline("Paste lyrics (or just type END for none)")


def send_request(url: str, token: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=3600) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {text}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e}") from e


def resolve_token(cli_token: str) -> str:
    if cli_token:
        return cli_token
    env_token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
    if env_token:
        return env_token
    dotenv_token = read_dotenv_value("hf_token") or read_dotenv_value("HF_TOKEN")
    return dotenv_token


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive ACE-Step endpoint generator")
    parser.add_argument("--url", default=os.getenv("HF_ENDPOINT_URL", DEFAULT_URL), help="Inference endpoint URL")
    parser.add_argument("--token", default="", help="HF token. If omitted, uses env/.env")
    parser.add_argument("--prompt", default="", help="Initial prompt")
    parser.add_argument("--out-file", default="", help="Output WAV file path")
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Ask advanced generation options (seed/guidance/steps/sample-rate/LM).",
    )
    args = parser.parse_args()

    print("=== ACE-Step Interactive Generation ===")

    token = resolve_token(args.token)
    if not token:
        print("No token found. Set HF_TOKEN or hf_token in .env, or pass --token.")
        return 1

    url = prompt_text("Endpoint URL", args.url, required=True)
    music_prompt = prompt_text("Music prompt", args.prompt, required=True)
    bpm = prompt_int("BPM (blank for auto)", None, allow_blank=True)
    duration_sec = prompt_int("Duration seconds", 120)
    instrumental = prompt_yes_no("Instrumental (no vocals)", False)
    lyrics = "" if instrumental else prompt_lyrics_optional()

    # Quality-first defaults: use SFT + LM path configured on the endpoint.
    sample_rate = DEFAULT_SAMPLE_RATE
    steps = 50
    guidance_scale = 7.0
    seed = 42
    use_lm = True
    allow_fallback = False
    simple_prompt = False

    if args.advanced:
        print("\nAdvanced options:")
        sample_rate = prompt_int("Sample rate", DEFAULT_SAMPLE_RATE)
        steps = prompt_int("Steps", 50)
        guidance_scale = prompt_float("Guidance scale", 7.0)
        seed = prompt_int("Seed", 42)
        use_lm = prompt_yes_no("Use LM planning (higher quality, slower)", True)
        allow_fallback = prompt_yes_no("Allow fallback sine audio", False)

    default_out = args.out_file or f"music_{int(time.time())}.wav"
    out_file = prompt_text("Output file", default_out, required=True)

    inputs = {
        "prompt": music_prompt,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "use_lm": use_lm,
        "simple_prompt": simple_prompt,
        "instrumental": instrumental,
        "allow_fallback": allow_fallback,
    }
    if bpm is not None:
        inputs["bpm"] = bpm
    if lyrics:
        inputs["lyrics"] = lyrics

    payload = {"inputs": inputs}

    print("\nSending request...")
    try:
        response = send_request(url, token, payload)
    except Exception as e:
        print(f"Request failed: {e}")
        return 1

    print("Response summary:")
    print(json.dumps({
        "used_fallback": response.get("used_fallback"),
        "model_loaded": response.get("model_loaded"),
        "model_error": response.get("model_error"),
        "sample_rate": response.get("sample_rate"),
        "duration_sec": response.get("duration_sec"),
    }, indent=2))

    if response.get("error"):
        print(f"Endpoint error: {response['error']}")
        return 1

    audio_b64 = response.get("audio_base64_wav")
    if not audio_b64:
        print("No audio_base64_wav in response.")
        return 1

    audio_bytes = base64.b64decode(audio_b64)
    Path(out_file).write_bytes(audio_bytes)
    print(f"Saved audio: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
