import base64
import copy
import os
import sys
import tempfile
from typing import Any, Dict

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel


DEFAULT_PROMPT = "Please describe the audio in detail."


def _log(msg: str) -> None:
    print(f"[AF3 NVIDIA handler] {msg}", flush=True)


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _strip_state_dict_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        key2 = key[6:] if key.startswith("model.") else key
        out[key2] = value
    return out


class EndpointHandler:
    """
    NVIDIA AF3 stack endpoint handler (matches Space architecture closely).

    Request:
      {
        "inputs": {
          "prompt": "...",
          "audio_base64": "...",
          "think_mode": true,
          "max_new_tokens": 2048,
          "temperature": 0.2
        }
      }

    Response:
      {"generated_text": "...", "mode": "think|single"}
    """

    def __init__(self, model_dir: str = ""):
        del model_dir
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.code_repo_id = os.getenv("AF3_NV_CODE_REPO_ID", "nvidia/audio-flamingo-3")
        self.model_repo_id = os.getenv("AF3_NV_MODEL_REPO_ID", "nvidia/audio-flamingo-3")
        self.code_repo_type = os.getenv("AF3_NV_CODE_REPO_TYPE", "space")
        self.model_repo_type = os.getenv("AF3_NV_MODEL_REPO_TYPE", "model")
        self.default_mode = os.getenv("AF3_NV_DEFAULT_MODE", "think").strip().lower()
        if self.default_mode not in {"think", "single"}:
            self.default_mode = "think"

        self.load_think = _env_true("AF3_NV_LOAD_THINK", True)
        self.load_single = _env_true("AF3_NV_LOAD_SINGLE", self.default_mode == "single")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={self.device}")
        _log(
            f"code_repo={self.code_repo_type}:{self.code_repo_id} "
            f"model_repo={self.model_repo_type}:{self.model_repo_id} default_mode={self.default_mode}"
        )

        self.llava = self._load_llava_runtime()
        self.model_root = self._download_model_root()

        self.model_single = None
        self.model_think = None

        if self.load_single:
            self.model_single = self._load_single_model()
        if self.load_think:
            self.model_think = self._load_think_model()

        if self.model_single is None and self.model_think is None:
            raise RuntimeError("No model loaded. Enable AF3_NV_LOAD_THINK or AF3_NV_LOAD_SINGLE.")

    def _load_llava_runtime(self):
        code_root = snapshot_download(
            repo_id=self.code_repo_id,
            repo_type=self.code_repo_type,
            allow_patterns=["llava/**"],
            token=self.hf_token or None,
        )
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        import llava  # type: ignore

        _log(f"Loaded llava runtime from {code_root}")
        return llava

    def _download_model_root(self) -> str:
        model_root = snapshot_download(
            repo_id=self.model_repo_id,
            repo_type=self.model_repo_type,
            token=self.hf_token or None,
        )
        _log(f"Model root: {model_root}")
        return model_root

    def _load_single_model(self):
        _log("Loading single-turn model...")
        model = self.llava.load(self.model_root, model_base=None)
        model = model.to(self.device)
        model.eval()
        return model

    def _load_think_model(self):
        _log("Loading think/long model (stage35 adapter)...")
        stage35_dir = os.path.join(self.model_root, "stage35")
        non_lora_path = os.path.join(stage35_dir, "non_lora_trainables.bin")
        if not os.path.exists(non_lora_path):
            raise RuntimeError(f"stage35 non_lora_trainables missing: {non_lora_path}")

        model = self.llava.load(self.model_root, model_base=None)
        model = model.to(self.device)

        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
        non_lora_trainables = _strip_state_dict_prefixes(non_lora_trainables)
        model.load_state_dict(non_lora_trainables, strict=False)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = PeftModel.from_pretrained(
            model,
            stage35_dir,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        )
        model.eval()
        return model

    def _select_model(self, think_mode: bool):
        if think_mode and self.model_think is not None:
            return self.model_think, "think"
        if (not think_mode) and self.model_single is not None:
            return self.model_single, "single"
        if self.model_think is not None:
            return self.model_think, "think"
        return self.model_single, "single"

    def _build_generation_config(self, model, max_new_tokens: int, temperature: float):
        base_cfg = getattr(model, "default_generation_config", None)
        if base_cfg is None:
            return None
        cfg = copy.deepcopy(base_cfg)
        if max_new_tokens > 0:
            setattr(cfg, "max_new_tokens", int(max_new_tokens))
        if temperature > 0:
            setattr(cfg, "temperature", float(temperature))
            setattr(cfg, "do_sample", True)
        else:
            setattr(cfg, "do_sample", False)
        return cfg

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = data.get("inputs", data) if isinstance(data, dict) else {}
        audio_b64 = payload.get("audio_base64")
        if not audio_b64:
            return {"error": "audio_base64 is required"}

        prompt = str(payload.get("prompt", DEFAULT_PROMPT)).strip() or DEFAULT_PROMPT
        think_mode_val = payload.get("think_mode")
        if think_mode_val is None:
            think_mode = self.default_mode == "think"
        else:
            think_mode = bool(think_mode_val)

        max_new_tokens = int(payload.get("max_new_tokens", 2048))
        temperature = float(payload.get("temperature", 0.2))
        model, mode = self._select_model(think_mode)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(base64.b64decode(audio_b64))

        try:
            sound = self.llava.Sound(tmp_path)
            full_prompt = f"<sound>\n{prompt}"
            gen_cfg = self._build_generation_config(model, max_new_tokens=max_new_tokens, temperature=temperature)

            with torch.inference_mode():
                if gen_cfg is not None:
                    response = model.generate_content([sound, full_prompt], generation_config=gen_cfg)
                else:
                    response = model.generate_content([sound, full_prompt])
            return {"generated_text": str(response).strip(), "mode": mode}
        except Exception as exc:
            return {"error": str(exc), "mode": mode}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
