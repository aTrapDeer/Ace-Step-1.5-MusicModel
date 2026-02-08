# handler.py
import base64
import inspect
import io
import os
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf

# Optional torch import for dtype/device handling
try:
    import torch
except Exception:
    torch = None


class EndpointHandler:
    """
    Hugging Face Inference Endpoints custom handler for ACE-Step 1.5.

    Request body shape:
    {
      "inputs": {
        "prompt": "upbeat pop rap, emotional guitar",
        "lyrics": "[Verse] ...",
        "duration_sec": 12,
        "sample_rate": 44100,
        "seed": 42,
        "guidance_scale": 7.0,
        "steps": 50,
        "use_lm": true,
        "simple_prompt": false,
        "model_repo": "ACE-Step/Ace-Step1.5"
      }
    }

    Also supported for simple mode:
    {
      "inputs": "upbeat pop rap with emotional guitar"
    }

    Response:
    {
      "audio_base64_wav": "...",
      "sample_rate": 44100,
      "duration_sec": 12,
      "used_fallback": false,
      "model_loaded": true,
      "model_error": null,
      "meta": {...}
    }
    """

    def __init__(self, path: str = ""):
        self.path = path
        self.model = None
        self.model_error = None
        self.model_repo = os.getenv("ACE_MODEL_REPO", "ACE-Step/Ace-Step1.5")
        self.default_sr = int(os.getenv("DEFAULT_SAMPLE_RATE", "44100"))

        # Runtime knobs
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.dtype = "float16" if self.device == "cuda" else "float32"

        # Try to initialize ACE-Step pipeline from repo code paths.
        # Repo mentions Python API and module path `acestep.acestep_v15_pipeline`.
        self._init_model()

    # --------------------------
    # Initialization helpers
    # --------------------------
    def _init_model(self) -> None:
        err_msgs = []

        # Strategy A: class/factory in acestep.acestep_v15_pipeline
        try:
            from acestep import acestep_v15_pipeline as m  # type: ignore

            # Try common factory patterns
            if hasattr(m, "from_pretrained"):
                self.model = m.from_pretrained(self.model_repo)  # type: ignore
            elif hasattr(m, "AceStepV15Pipeline"):
                cls = getattr(m, "AceStepV15Pipeline")
                if hasattr(cls, "from_pretrained"):
                    self.model = cls.from_pretrained(self.model_repo)
                else:
                    self.model = cls(model_path=self.model_repo)
            elif hasattr(m, "Pipeline"):
                cls = getattr(m, "Pipeline")
                if hasattr(cls, "from_pretrained"):
                    self.model = cls.from_pretrained(self.model_repo)
                else:
                    self.model = cls(self.model_repo)
            else:
                raise RuntimeError("No known pipeline class/factory found in acestep_v15_pipeline")

            # Move device if supported
            if self.model is not None and hasattr(self.model, "to"):
                try:
                    self.model.to(self.device)
                except Exception:
                    pass

            return
        except Exception as e:
            err_msgs.append(f"Strategy A failed: {type(e).__name__}: {e}")

        # Strategy B: import root `acestep` and find a likely pipeline symbol
        try:
            import acestep  # type: ignore

            candidates = [
                "AceStepV15Pipeline",
                "AceStepPipeline",
                "Pipeline",
                "create_pipeline",
                "build_pipeline",
                "load_pipeline",
            ]

            obj = None
            for name in candidates:
                if hasattr(acestep, name):
                    obj = getattr(acestep, name)
                    break

            if obj is None:
                raise RuntimeError("No known pipeline symbol found in `acestep` package")

            if callable(obj):
                # class or factory
                if hasattr(obj, "from_pretrained"):
                    self.model = obj.from_pretrained(self.model_repo)
                else:
                    # try keyword variants
                    try:
                        self.model = obj(model_path=self.model_repo)
                    except TypeError:
                        self.model = obj(self.model_repo)
            else:
                self.model = obj

            if self.model is not None and hasattr(self.model, "to"):
                try:
                    self.model.to(self.device)
                except Exception:
                    pass
            return
        except Exception as e:
            err_msgs.append(f"Strategy B failed: {type(e).__name__}: {e}")

        self.model = None
        self.model_error = " | ".join(err_msgs)

    # --------------------------
    # Audio helpers
    # --------------------------
    @staticmethod
    def _as_float32(audio: Any) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            arr = audio
        elif torch is not None and isinstance(audio, torch.Tensor):
            arr = audio.detach().cpu().numpy()
        else:
            arr = np.asarray(audio)

        # Convert common tensor shape [channels, samples] to [samples, channels].
        if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
            arr = arr.T

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        return np.clip(arr, -1.0, 1.0)

    @staticmethod
    def _wav_b64(audio: np.ndarray, sr: int) -> str:
        bio = io.BytesIO()
        sf.write(bio, audio, sr, format="WAV")
        return base64.b64encode(bio.getvalue()).decode("utf-8")

    @staticmethod
    def _fallback_sine(duration_sec: int, sr: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        y = (0.07 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        return np.clip(y, -1.0, 1.0)

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        return default

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _pick_text(inputs: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            v = inputs.get(key)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    def _normalize_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_inputs = data.get("inputs", data)
        if isinstance(raw_inputs, str):
            raw_inputs = {"prompt": raw_inputs, "simple_prompt": True}
        if not isinstance(raw_inputs, dict):
            raise ValueError("`inputs` must be an object or string")

        prompt = self._pick_text(raw_inputs, "prompt", "query", "caption", "text", "description")
        lyrics = self._pick_text(raw_inputs, "lyrics")

        simple_prompt = self._to_bool(raw_inputs.get("simple_prompt"), False) or self._to_bool(
            raw_inputs.get("simple"), False
        )
        instrumental = self._to_bool(raw_inputs.get("instrumental"), False)
        if not lyrics and (instrumental or simple_prompt):
            lyrics = "[Instrumental]"

        duration_sec = self._to_int(raw_inputs.get("duration_sec", raw_inputs.get("duration", 10)), 10)
        duration_sec = max(1, min(duration_sec, 600))

        sample_rate = self._to_int(raw_inputs.get("sample_rate", self.default_sr), self.default_sr)
        sample_rate = max(8000, min(sample_rate, 48000))

        seed = self._to_int(raw_inputs.get("seed", 42), 42)
        guidance_scale = self._to_float(raw_inputs.get("guidance_scale", 7.0), 7.0)
        steps = self._to_int(raw_inputs.get("steps", raw_inputs.get("inference_steps", 50)), 50)
        steps = max(1, min(steps, 500))
        use_lm = self._to_bool(raw_inputs.get("use_lm", raw_inputs.get("thinking", True)), True)
        task_type = self._pick_text(raw_inputs, "task_type") or "text2music"

        model_repo = raw_inputs.get("model_repo")

        model_kwargs = {
            "task_type": task_type,
            "prompt": prompt,
            "caption": prompt,
            "query": prompt,
            "lyrics": lyrics,
            "duration_sec": duration_sec,
            "duration": duration_sec,
            "sample_rate": sample_rate,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "inference_steps": steps,
            "num_inference_steps": steps,
            "use_lm": use_lm,
            "thinking": use_lm,
            "instrumental": instrumental,
        }

        return {
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "use_lm": use_lm,
            "instrumental": instrumental,
            "simple_prompt": simple_prompt,
            "model_repo": model_repo,
            "model_kwargs": model_kwargs,
        }

    @staticmethod
    def _invoke_with_supported_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Any:
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if has_var_kw:
                return fn(**kwargs)
            accepted = {
                name
                for name, p in sig.parameters.items()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            return fn(**filtered)
        except Exception:
            # Fallback for C-extension callables or dynamic signatures.
            return fn(**kwargs)

    def _normalize_model_output(self, out: Any, default_sr: int) -> Tuple[np.ndarray, int]:
        if out is None:
            raise RuntimeError("Model returned None")

        if hasattr(out, "success") and not getattr(out, "success"):
            err = getattr(out, "error", "unknown model error")
            raise RuntimeError(str(err))

        if hasattr(out, "audios"):
            audios = getattr(out, "audios") or []
            if not audios:
                raise RuntimeError("Model result has no audios")
            first = audios[0]
            if isinstance(first, dict):
                audio = first.get("tensor", first.get("audio", first.get("waveform", first.get("wav"))))
                sr = first.get("sample_rate", default_sr)
            else:
                audio = getattr(first, "tensor", getattr(first, "audio", None))
                sr = getattr(first, "sample_rate", default_sr)
            if audio is None:
                raise RuntimeError("Model result audio entry is missing tensor/audio")
            return self._as_float32(audio), int(sr)

        if isinstance(out, tuple) and len(out) >= 1:
            audio = out[0]
            sr = int(out[1]) if len(out) > 1 and out[1] is not None else default_sr
            return self._as_float32(audio), sr

        if isinstance(out, dict):
            if "audios" in out:
                audios = out.get("audios") or []
                if not audios:
                    raise RuntimeError("Model output `audios` is empty")
                first = audios[0]
                if not isinstance(first, dict):
                    raise RuntimeError("Model output `audios[0]` must be a dict")
                audio = first.get("tensor", first.get("audio", first.get("waveform", first.get("wav"))))
                sr = first.get("sample_rate", default_sr)
                if audio is None:
                    raise RuntimeError("Model output `audios[0]` missing tensor/audio")
                return self._as_float32(audio), int(sr)

            audio = out.get("audio", out.get("waveform", out.get("wav", out.get("tensor"))))
            sr = out.get("sample_rate", out.get("sr", default_sr))
            if audio is None:
                raise RuntimeError("Model dict output missing audio/waveform field")
            return self._as_float32(audio), int(sr)

        for name in ("audio", "waveform", "wav", "tensor"):
            if hasattr(out, name):
                audio = getattr(out, name)
                if audio is not None:
                    sr = getattr(out, "sample_rate", getattr(out, "sr", default_sr))
                    return self._as_float32(audio), int(sr)

        return self._as_float32(out), default_sr

    # --------------------------
    # Inference
    # --------------------------
    def _call_model(
        self,
        model_kwargs: Dict[str, Any],
        sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Tries multiple invocation styles to tolerate minor ACE-Step API differences.
        Returns (audio_np, sample_rate).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # Common callable entrypoints
        methods = [
            "__call__",
            "generate",
            "infer",
            "inference",
            "text_to_music",
            "run",
        ]

        last_err = None
        for m in methods:
            try:
                fn = self.model if m == "__call__" else getattr(self.model, m, None)
                if fn is None:
                    continue

                # Try full kwargs
                try:
                    out = self._invoke_with_supported_kwargs(fn, model_kwargs)
                except TypeError:
                    # Narrow payload if signature is strict
                    skinny = {
                        "prompt": model_kwargs.get("prompt"),
                        "caption": model_kwargs.get("caption"),
                        "lyrics": model_kwargs.get("lyrics"),
                        "duration": model_kwargs.get("duration"),
                        "seed": model_kwargs.get("seed"),
                    }
                    skinny = {k: v for k, v in skinny.items() if v is not None and (k != "prompt" or str(v).strip())}
                    out = self._invoke_with_supported_kwargs(fn, skinny)

                return self._normalize_model_output(out, sample_rate)

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"No compatible inference method worked: {last_err}")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            req = self._normalize_request(data)

            # Optional override
            model_repo = req.get("model_repo")
            if model_repo and model_repo != self.model_repo:
                # hot-switch model only if user asks
                self.model_repo = str(model_repo)
                self._init_model()

            used_fallback = False

            if self.model is not None:
                try:
                    audio, out_sr = self._call_model(
                        model_kwargs=req["model_kwargs"],
                        sample_rate=req["sample_rate"],
                    )
                except Exception as e:
                    used_fallback = True
                    self.model_error = f"Inference failed: {type(e).__name__}: {e}"
                    audio = self._fallback_sine(req["duration_sec"], req["sample_rate"], req["seed"])
                    out_sr = req["sample_rate"]
            else:
                used_fallback = True
                audio = self._fallback_sine(req["duration_sec"], req["sample_rate"], req["seed"])
                out_sr = req["sample_rate"]

            return {
                "audio_base64_wav": self._wav_b64(audio, out_sr),
                "sample_rate": int(out_sr),
                "duration_sec": int(req["duration_sec"]),
                "used_fallback": used_fallback,
                "model_loaded": self.model is not None,
                "model_repo": self.model_repo,
                "model_error": self.model_error,
                "meta": {
                    "device": self.device,
                    "dtype": self.dtype,
                    "prompt_len": len(req["prompt"]),
                    "lyrics_len": len(req["lyrics"]),
                    "seed": req["seed"],
                    "guidance_scale": req["guidance_scale"],
                    "steps": req["steps"],
                    "use_lm": req["use_lm"],
                    "simple_prompt": req["simple_prompt"],
                    "instrumental": req["instrumental"],
                    "resolved_prompt": req["prompt"],
                    "resolved_lyrics": req["lyrics"],
                },
            }

        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
                "audio_base64_wav": None,
                "sample_rate": None,
                "duration_sec": None,
            }
