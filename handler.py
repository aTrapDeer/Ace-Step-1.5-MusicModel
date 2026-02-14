# handler.py
import base64
import io
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import torch
except Exception:
    torch = None


class EndpointHandler:
    """
    Hugging Face Inference Endpoints custom handler for ACE-Step 1.5.

    Supported request shapes:
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
        "instrumental": false,
        "allow_fallback": false
      }
    }

    Or simple mode:
    {
      "inputs": "upbeat pop rap with emotional guitar"
    }

    Notes:
    - This handler uses ACE-Step's official Python API internally.
    - Fallback sine generation is disabled by default so model failures are explicit.
    """

    def __init__(self, path: str = ""):
        self.path = path
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        self.model_repo = os.getenv("ACE_MODEL_REPO", "ACE-Step/Ace-Step1.5")
        # Default to the larger quality-oriented setup.
        # Override via ACE_CONFIG_PATH / ACE_LM_MODEL_PATH when needed.
        self.config_path = os.getenv("ACE_CONFIG_PATH", "acestep-v15-sft")
        self.lm_model_path = os.getenv("ACE_LM_MODEL_PATH", "acestep-5Hz-lm-4B")
        self.lm_backend = os.getenv("ACE_LM_BACKEND", "pt")
        self.download_source = os.getenv("ACE_DOWNLOAD_SOURCE", "huggingface")

        self.default_sr = int(os.getenv("DEFAULT_SAMPLE_RATE", "44100"))
        self.enable_fallback = self._to_bool(os.getenv("ACE_ENABLE_FALLBACK"), False)
        self.init_lm_on_start = self._to_bool(os.getenv("ACE_INIT_LLM"), False)
        self.skip_init = self._to_bool(os.getenv("ACE_SKIP_INIT"), False)

        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.dtype = "float16" if self.device == "cuda" else "float32"

        self.model_loaded = False
        self.model_error: Optional[str] = None
        self.init_details: Dict[str, Any] = {}

        self.dit_handler = None
        self.llm_handler = None
        self.llm_initialized = False
        self.llm_error: Optional[str] = None

        self._GenerationParams = None
        self._GenerationConfig = None
        self._generate_music = None
        self._create_sample = None

        if self.skip_init:
            self.model_error = "Initialization skipped because ACE_SKIP_INIT=true"
        else:
            self._init_model()

    # --------------------------
    # Initialization
    # --------------------------
    def _init_model(self) -> None:
        err_msgs = []

        # ACE-Step dynamic config imports layer_type_validation from transformers.
        # Some endpoint base images ship a transformers build without this helper.
        self._patch_transformers_layer_validation()
        # Some CUDA/torch combinations used by managed endpoint images don't support
        # sorting bool tensors on CUDA. ACE-Step/Transformers paths can hit this.
        self._patch_torch_sort_bool_cuda()

        try:
            from acestep.handler import AceStepHandler
            from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
            from acestep.llm_inference import LLMHandler
        except Exception as e:
            self.model_error = f"ACE-Step import failed: {type(e).__name__}: {e}"
            return

        self._GenerationParams = GenerationParams
        self._GenerationConfig = GenerationConfig
        self._generate_music = generate_music
        self._create_sample = create_sample

        try:
            self.dit_handler = AceStepHandler()
            prefer_source = self.download_source if self.download_source in {"huggingface", "modelscope"} else None
            init_status, ok = self.dit_handler.initialize_service(
                project_root=self.project_root,
                config_path=self.config_path,
                device=self.device,
                use_flash_attention=False,
                compile_model=False,
                offload_to_cpu=False,
                offload_dit_to_cpu=False,
                prefer_source=prefer_source,
            )
            self.init_details["dit_status"] = init_status
            if not ok:
                raise RuntimeError(init_status)
        except Exception as e:
            err_msgs.append(f"DiT init failed: {type(e).__name__}: {e}")

        try:
            self.llm_handler = LLMHandler()
            if self.init_lm_on_start:
                self._ensure_llm_initialized()
        except Exception as e:
            err_msgs.append(f"LLM bootstrap failed: {type(e).__name__}: {e}")

        if err_msgs:
            self.model_loaded = False
            self.model_error = " | ".join(err_msgs)
            return

        self.model_loaded = True
        self.model_error = None

    @staticmethod
    def _patch_transformers_layer_validation() -> None:
        try:
            from transformers import configuration_utils as cu
        except Exception:
            return

        if hasattr(cu, "layer_type_validation"):
            return

        def _fallback_layer_type_validation(layer_types, num_hidden_layers=None):
            if layer_types is None:
                return
            if not isinstance(layer_types, (list, tuple)):
                raise TypeError("`layer_types` must be a list/tuple")
            if num_hidden_layers is not None and len(layer_types) != int(num_hidden_layers):
                raise ValueError("`layer_types` length must match `num_hidden_layers`")

        cu.layer_type_validation = _fallback_layer_type_validation

    @staticmethod
    def _patch_torch_sort_bool_cuda() -> None:
        if torch is None or not hasattr(torch, "sort"):
            return
        if getattr(torch.sort, "__name__", "") == "_sort_bool_cuda_compat":
            return

        _orig_sort = torch.sort
        _orig_tensor_sort = getattr(torch.Tensor, "sort", None)
        _orig_argsort = getattr(torch, "argsort", None)
        _orig_tensor_argsort = getattr(torch.Tensor, "argsort", None)

        def _sort_bool_cuda_compat(input_tensor, *args, **kwargs):
            if (
                isinstance(input_tensor, torch.Tensor)
                and input_tensor.is_cuda
                and input_tensor.dtype == torch.bool
            ):
                out = _orig_sort(input_tensor.to(torch.uint8), *args, **kwargs)
                values = out.values.to(torch.bool) if hasattr(out, "values") else out[0].to(torch.bool)
                indices = out.indices if hasattr(out, "indices") else out[1]
                return values, indices
            return _orig_sort(input_tensor, *args, **kwargs)

        _sort_bool_cuda_compat.__name__ = "_sort_bool_cuda_compat"
        torch.sort = _sort_bool_cuda_compat

        if callable(_orig_tensor_sort):
            def _tensor_sort_bool_cuda_compat(self, *args, **kwargs):
                if self.is_cuda and self.dtype == torch.bool:
                    out = _orig_tensor_sort(self.to(torch.uint8), *args, **kwargs)
                    values = out.values.to(torch.bool) if hasattr(out, "values") else out[0].to(torch.bool)
                    indices = out.indices if hasattr(out, "indices") else out[1]
                    return values, indices
                return _orig_tensor_sort(self, *args, **kwargs)

            _tensor_sort_bool_cuda_compat.__name__ = "_tensor_sort_bool_cuda_compat"
            torch.Tensor.sort = _tensor_sort_bool_cuda_compat

        if callable(_orig_argsort):
            def _argsort_bool_cuda_compat(input_tensor, *args, **kwargs):
                if (
                    isinstance(input_tensor, torch.Tensor)
                    and input_tensor.is_cuda
                    and input_tensor.dtype == torch.bool
                ):
                    return _orig_argsort(input_tensor.to(torch.uint8), *args, **kwargs)
                return _orig_argsort(input_tensor, *args, **kwargs)

            _argsort_bool_cuda_compat.__name__ = "_argsort_bool_cuda_compat"
            torch.argsort = _argsort_bool_cuda_compat

        if callable(_orig_tensor_argsort):
            def _tensor_argsort_bool_cuda_compat(self, *args, **kwargs):
                if self.is_cuda and self.dtype == torch.bool:
                    return _orig_tensor_argsort(self.to(torch.uint8), *args, **kwargs)
                return _orig_tensor_argsort(self, *args, **kwargs)

            _tensor_argsort_bool_cuda_compat.__name__ = "_tensor_argsort_bool_cuda_compat"
            torch.Tensor.argsort = _tensor_argsort_bool_cuda_compat

    def _ensure_llm_initialized(self) -> bool:
        if self.llm_handler is None:
            self.llm_error = "LLM handler is not available"
            return False

        if self.llm_initialized:
            return True

        try:
            checkpoint_dir = os.path.join(self.project_root, "checkpoints")
            full_lm_model_path = os.path.join(checkpoint_dir, self.lm_model_path)
            if not os.path.exists(full_lm_model_path):
                try:
                    from acestep.model_downloader import ensure_lm_model, ensure_main_model
                except Exception as e:
                    self.llm_error = f"LM download helper import failed: {type(e).__name__}: {e}"
                    return False

                # 1.7B ships with main; 0.6B/4B are standalone submodels.
                if self.lm_model_path == "acestep-5Hz-lm-1.7B":
                    dl_ok, dl_msg = ensure_main_model(
                        checkpoints_dir=Path(checkpoint_dir),
                        prefer_source=self.download_source,
                    )
                else:
                    dl_ok, dl_msg = ensure_lm_model(
                        model_name=self.lm_model_path,
                        checkpoints_dir=Path(checkpoint_dir),
                        prefer_source=self.download_source,
                    )
                self.init_details["llm_download"] = dl_msg
                if not dl_ok:
                    self.llm_error = f"LM download failed: {dl_msg}"
                    return False

            status, ok = self.llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=self.lm_model_path,
                backend=self.lm_backend,
                device=self.device,
                offload_to_cpu=False,
            )
            self.init_details["llm_status"] = status
            if not ok:
                self.llm_error = status
                self.llm_initialized = False
                return False

            self.llm_error = None
            self.llm_initialized = True
            return True
        except Exception as e:
            self.llm_error = f"LLM init exception: {type(e).__name__}: {e}"
            self.llm_initialized = False
            return False

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

    # --------------------------
    # Request normalization
    # --------------------------
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

        duration_sec = self._to_int(raw_inputs.get("duration_sec", raw_inputs.get("duration", 12)), 12)
        duration_sec = max(10, min(duration_sec, 600))

        sample_rate = self._to_int(raw_inputs.get("sample_rate", self.default_sr), self.default_sr)
        sample_rate = max(8000, min(sample_rate, 48000))

        seed = self._to_int(raw_inputs.get("seed", 42), 42)
        guidance_scale = self._to_float(raw_inputs.get("guidance_scale", 7.0), 7.0)
        steps = self._to_int(raw_inputs.get("steps", raw_inputs.get("inference_steps", 50)), 50)
        steps = max(1, min(steps, 200))
        bpm_raw = raw_inputs.get("bpm")
        bpm = None
        if bpm_raw is not None and str(bpm_raw).strip() != "":
            bpm = self._to_int(bpm_raw, 0)
            if bpm <= 0:
                bpm = None
        use_lm = self._to_bool(raw_inputs.get("use_lm", raw_inputs.get("thinking", True)), True)
        allow_fallback = self._to_bool(raw_inputs.get("allow_fallback"), self.enable_fallback)

        return {
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "bpm": bpm,
            "use_lm": use_lm,
            "instrumental": instrumental,
            "simple_prompt": simple_prompt,
            "allow_fallback": allow_fallback,
        }

    # --------------------------
    # ACE-Step invocation
    # --------------------------
    def _build_generation_inputs(self, req: Dict[str, Any], llm_ready: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        caption = req["prompt"]
        lyrics = req["lyrics"]

        extras: Dict[str, Any] = {
            "simple_expansion_used": False,
            "simple_expansion_error": None,
        }

        bpm = req.get("bpm")
        keyscale = ""
        timesignature = ""
        vocal_language = "unknown"
        duration = float(req["duration_sec"])

        if req["simple_prompt"] and req["use_lm"] and llm_ready and caption:
            try:
                sample = self._create_sample(
                    llm_handler=self.llm_handler,
                    query=caption,
                    instrumental=req["instrumental"],
                )
                if getattr(sample, "success", False):
                    caption = getattr(sample, "caption", "") or caption
                    lyrics = getattr(sample, "lyrics", "") or lyrics
                    sample_bpm = getattr(sample, "bpm", None)
                    if bpm is None:
                        bpm = sample_bpm
                    keyscale = getattr(sample, "keyscale", "") or ""
                    timesignature = getattr(sample, "timesignature", "") or ""
                    vocal_language = getattr(sample, "language", "") or "unknown"
                    sample_duration = getattr(sample, "duration", None)
                    if sample_duration:
                        duration = float(sample_duration)
                    extras["simple_expansion_used"] = True
                else:
                    extras["simple_expansion_error"] = getattr(sample, "error", "create_sample failed")
            except Exception as e:
                extras["simple_expansion_error"] = f"{type(e).__name__}: {e}"

        params = self._GenerationParams(
            task_type="text2music",
            caption=caption,
            lyrics=lyrics,
            instrumental=req["instrumental"],
            duration=duration,
            inference_steps=req["steps"],
            guidance_scale=req["guidance_scale"],
            seed=req["seed"],
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            vocal_language=vocal_language,
            thinking=bool(req["use_lm"] and llm_ready),
            use_cot_metas=bool(req["use_lm"] and llm_ready),
            use_cot_caption=bool(req["use_lm"] and llm_ready and not req["simple_prompt"]),
            use_cot_language=bool(req["use_lm"] and llm_ready),
        )

        config = self._GenerationConfig(
            batch_size=1,
            allow_lm_batch=False,
            use_random_seed=False,
            seeds=[req["seed"]],
            audio_format="wav",
        )

        extras["resolved_prompt"] = caption
        extras["resolved_lyrics"] = lyrics
        extras["resolved_duration"] = duration

        return {"params": params, "config": config}, extras

    def _call_model(self, req: Dict[str, Any]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        if not self.model_loaded or self.dit_handler is None:
            raise RuntimeError(self.model_error or "Model is not loaded")

        llm_ready = False
        if req["use_lm"]:
            llm_ready = self._ensure_llm_initialized()

        generation_inputs, extras = self._build_generation_inputs(req, llm_ready)

        result = self._generate_music(
            self.dit_handler,
            self.llm_handler if llm_ready else None,
            generation_inputs["params"],
            generation_inputs["config"],
            save_dir=None,
            progress=None,
        )

        if not getattr(result, "success", False):
            raise RuntimeError(getattr(result, "error", "generation failed"))

        audios = getattr(result, "audios", None) or []
        if not audios:
            raise RuntimeError("generation succeeded but no audio was returned")

        first = audios[0]
        audio_tensor = first.get("tensor") if isinstance(first, dict) else None
        if audio_tensor is None:
            raise RuntimeError("generated audio tensor is missing")

        sample_rate = int(first.get("sample_rate", req["sample_rate"]))
        status_message = getattr(result, "status_message", "")

        meta = {
            "llm_requested": req["use_lm"],
            "llm_initialized": llm_ready,
            "llm_error": self.llm_error,
            "status_message": status_message,
        }
        meta.update(extras)

        return self._as_float32(audio_tensor), sample_rate, meta

    # --------------------------
    # Endpoint entry
    # --------------------------
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            req = self._normalize_request(data)

            used_fallback = False
            runtime_meta: Dict[str, Any] = {}

            try:
                audio, out_sr, runtime_meta = self._call_model(req)
            except Exception as model_exc:
                self.model_error = f"Inference failed: {type(model_exc).__name__}: {model_exc}"
                if not req["allow_fallback"]:
                    raise RuntimeError(self.model_error)

                used_fallback = True
                audio = self._fallback_sine(req["duration_sec"], req["sample_rate"], req["seed"])
                out_sr = req["sample_rate"]

            return {
                "audio_base64_wav": self._wav_b64(audio, out_sr),
                "sample_rate": int(out_sr),
                "duration_sec": int(req["duration_sec"]),
                "used_fallback": used_fallback,
                "model_loaded": self.model_loaded,
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
                    "bpm": req.get("bpm"),
                    "use_lm": req["use_lm"],
                    "simple_prompt": req["simple_prompt"],
                    "instrumental": req["instrumental"],
                    "allow_fallback": req["allow_fallback"],
                    "resolved_prompt": runtime_meta.get("resolved_prompt", req["prompt"]),
                    "resolved_lyrics": runtime_meta.get("resolved_lyrics", req["lyrics"]),
                    "simple_expansion_used": runtime_meta.get("simple_expansion_used", False),
                    "simple_expansion_error": runtime_meta.get("simple_expansion_error"),
                    "llm_requested": runtime_meta.get("llm_requested", False),
                    "llm_initialized": runtime_meta.get("llm_initialized", False),
                    "llm_error": runtime_meta.get("llm_error"),
                    "status_message": runtime_meta.get("status_message", ""),
                    "init_details": self.init_details,
                },
            }

        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4),
                "audio_base64_wav": None,
                "sample_rate": None,
                "duration_sec": None,
                "used_fallback": False,
                "model_loaded": self.model_loaded,
                "model_repo": self.model_repo,
                "model_error": self.model_error,
                "meta": {
                    "device": self.device,
                    "dtype": self.dtype,
                    "init_details": self.init_details,
                    "llm_error": self.llm_error,
                },
            }
