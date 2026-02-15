import base64
import importlib
import importlib.machinery
import importlib.util
import io
import os
import subprocess
import sys
import types
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch


def _resolve_model_id(model_dir: str) -> str:
    default_id = os.getenv("AF3_MODEL_ID", "nvidia/audio-flamingo-3-hf")
    if model_dir and os.path.isdir(model_dir):
        has_local = os.path.exists(os.path.join(model_dir, "config.json"))
        if has_local:
            return model_dir
    return default_id


def _log(msg: str) -> None:
    print(f"[AF3 handler] {msg}", flush=True)


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _install_torchvision_stub() -> None:
    if not _env_true("AF3_STUB_TORCHVISION", True):
        return
    interpolation_mode = types.SimpleNamespace(
        NEAREST=0,
        BILINEAR=2,
        BICUBIC=3,
        BOX=4,
        HAMMING=5,
        LANCZOS=1,
    )
    transforms_stub = types.ModuleType("torchvision.transforms")
    setattr(transforms_stub, "InterpolationMode", interpolation_mode)
    setattr(
        transforms_stub,
        "__spec__",
        importlib.machinery.ModuleSpec(name="torchvision.transforms", loader=None),
    )
    tv_stub = types.ModuleType("torchvision")
    setattr(tv_stub, "transforms", transforms_stub)
    setattr(
        tv_stub,
        "__spec__",
        importlib.machinery.ModuleSpec(name="torchvision", loader=None),
    )
    sys.modules["torchvision"] = tv_stub
    sys.modules["torchvision.transforms"] = transforms_stub


_FIND_SPEC_PATCHED = False


def _patch_optional_backend_discovery() -> None:
    global _FIND_SPEC_PATCHED
    if _FIND_SPEC_PATCHED:
        return
    blocked = {"torchvision", "librosa"}
    original_find_spec = importlib.util.find_spec

    def wrapped_find_spec(name: str, package: str | None = None):
        root = name.split(".", 1)[0]
        if root in blocked:
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = wrapped_find_spec  # type: ignore[assignment]
    _FIND_SPEC_PATCHED = True


def _clear_python_modules(prefixes: Tuple[str, ...]) -> None:
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(f"{p}.") for p in prefixes):
            sys.modules.pop(name, None)


def _patch_torch_compat() -> None:
    try:
        import torch._dynamo._trace_wrapped_higher_order_op as dyn_wrap
    except Exception:
        return
    if hasattr(dyn_wrap, "TransformGetItemToIndex"):
        return

    class TransformGetItemToIndex:  # pragma: no cover - runtime compatibility shim
        pass

    setattr(dyn_wrap, "TransformGetItemToIndex", TransformGetItemToIndex)


def _af3_classes_available() -> tuple[bool, str]:
    try:
        from transformers import AudioFlamingo3ForConditionalGeneration  # noqa: F401
        from transformers import AudioFlamingo3Processor  # noqa: F401

        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _bootstrap_runtime_transformers(target_dir: str) -> None:
    packages = [
        os.getenv("AF3_TRANSFORMERS_SPEC", "transformers==5.1.0"),
        "numpy<2",
        "accelerate>=1.1.0",
        "sentencepiece",
        "safetensors",
        "soxr",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "--target", target_dir, *packages]
    _log("Installing runtime deps for AF3 (first boot can take a few minutes).")
    subprocess.check_call(cmd)


def _ensure_af3_transformers():
    _patch_optional_backend_discovery()
    _install_torchvision_stub()
    _patch_torch_compat()

    import transformers

    ok, err = _af3_classes_available()
    if ok:
        _log(f"Using bundled transformers={transformers.__version__}")
        return transformers

    if not _env_true("AF3_BOOTSTRAP_RUNTIME", True):
        raise RuntimeError(
            "AF3 classes are unavailable in bundled transformers "
            f"({transformers.__version__}) and AF3_BOOTSTRAP_RUNTIME is disabled. "
            f"Last import error: {err}"
        )

    target_dir = os.getenv("AF3_RUNTIME_DIR", "/tmp/af3_runtime")
    os.makedirs(target_dir, exist_ok=True)
    _bootstrap_runtime_transformers(target_dir)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    _clear_python_modules(("transformers", "tokenizers", "huggingface_hub", "safetensors"))
    _patch_optional_backend_discovery()
    _install_torchvision_stub()
    _patch_torch_compat()
    importlib.invalidate_caches()
    transformers = importlib.import_module("transformers")

    ok, err = _af3_classes_available()
    if not ok:
        raise RuntimeError(
            "Failed to load AF3 processor classes after runtime bootstrap. "
            f"transformers={getattr(transformers, '__version__', 'unknown')} "
            f"error={err}"
        )
    _log(f"Bootstrapped transformers={transformers.__version__}")
    return transformers


def _resample_audio_mono(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.arange(audio.shape[0], dtype=np.float64)
    dst_len = int(round(audio.shape[0] * float(dst_sr) / float(src_sr)))
    dst_len = max(dst_len, 1)
    dst_idx = np.linspace(0.0, float(max(audio.shape[0] - 1, 0)), dst_len, dtype=np.float64)
    out = np.interp(dst_idx, src_idx, audio.astype(np.float64, copy=False))
    return out.astype(np.float32, copy=False)


def _decode_audio_from_b64(audio_b64: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(audio_b64)
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    if data.ndim != 1:
        data = np.asarray(data).reshape(-1)
    target_sr = 16000
    if int(sr) != target_sr:
        data = _resample_audio_mono(data, int(sr), target_sr)
        sr = target_sr
    return data.astype(np.float32, copy=False), int(sr)


class EndpointHandler:
    """
    Hugging Face Dedicated Endpoint custom handler.

    Request:
      {
        "inputs": {
          "prompt": "...",
          "audio_base64": "...",
          "max_new_tokens": 1200,
          "temperature": 0.1
        }
      }

    Response:
      {"generated_text": "..."}
    """

    def __init__(self, model_dir: str = ""):
        self.model_id = _resolve_model_id(model_dir)
        self.transformers = _ensure_af3_transformers()
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        _log(
            f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"transformers={self.transformers.__version__} model_id={self.model_id}"
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _build_inputs(self, audio: np.ndarray, sample_rate: int, prompt: str) -> Dict[str, Any]:
        conversation: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            return self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                audio_kwargs={"sampling_rate": int(sample_rate)},
            )
        except Exception:
            return self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = data.get("inputs", data) if isinstance(data, dict) else {}
        prompt = str(payload.get("prompt", "Analyze this full song and summarize arrangement changes.")).strip()
        audio_b64 = payload.get("audio_base64")
        if not audio_b64:
            return {"error": "audio_base64 is required"}

        max_new_tokens = int(payload.get("max_new_tokens", 1200))
        temperature = float(payload.get("temperature", 0.1))

        try:
            audio, sample_rate = _decode_audio_from_b64(audio_b64)
            inputs = self._build_inputs(audio, sample_rate, prompt)
            device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    if hasattr(value, "dtype") and torch.is_floating_point(value):
                        inputs[key] = value.to(device=device, dtype=model_dtype)
                    else:
                        inputs[key] = value.to(device)

            do_sample = bool(temperature > 0)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            start = int(inputs["input_ids"].shape[1])
            text = self.processor.batch_decode(outputs[:, start:], skip_special_tokens=True)[0].strip()
            if not text:
                text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return {"generated_text": text}
        except Exception as exc:
            return {"error": str(exc)}
