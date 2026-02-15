import base64
import io
import os
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def _decode_audio_b64(audio_b64: str):
    raw = base64.b64decode(audio_b64)
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return mono, int(sr)


class EndpointHandler:
    """
    HF Dedicated Endpoint custom handler contract:
      request:
        {
          "inputs": {
            "prompt": "...",
            "audio_base64": "...",
            "sample_rate": 16000,
            "max_new_tokens": 384,
            "temperature": 0.1
          }
        }
      response:
        {"generated_text": "..."}
    """

    def __init__(self, model_dir: str = ""):
        model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
        # Only load from model_dir when actual weights/config are packaged there.
        if model_dir and os.path.isdir(model_dir):
            has_local_model = (
                os.path.exists(os.path.join(model_dir, "config.json"))
                and (
                    os.path.exists(os.path.join(model_dir, "model.safetensors"))
                    or any(name.endswith(".safetensors") for name in os.listdir(model_dir))
                )
            )
            if has_local_model:
                model_id = model_dir

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = data.get("inputs", data) if isinstance(data, dict) else {}
        prompt = str(payload.get("prompt", "Analyze this music audio.")).strip()
        audio_b64 = payload.get("audio_base64")
        if not audio_b64:
            return {"error": "audio_base64 is required"}

        max_new_tokens = int(payload.get("max_new_tokens", 384))
        temperature = float(payload.get("temperature", 0.1))

        audio, sr = _decode_audio_b64(audio_b64)
        sampling_rate = int(payload.get("sample_rate", sr))

        # Use direct audio token format to force audio conditioning.
        chat_text = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}\n"
        inputs = self.processor(
            text=chat_text,
            audio=[audio],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        device = next(self.model.parameters()).device
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
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        prompt_tokens = inputs["input_ids"].shape[1]
        generated_new = generated_ids[:, prompt_tokens:]
        text = self.processor.batch_decode(
            generated_new,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if not text.strip():
            # Some backends may return generated-only ids without prefix tokens.
            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return {"generated_text": text.strip()}
