"""
Local orchestration API for AF3 captioning + ChatGPT cleanup pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from af3_chatgpt_pipeline import (
    DEFAULT_AF3_MODEL_ID,
    DEFAULT_AF3_PROMPT,
    DEFAULT_OPENAI_MODEL,
    AF3EndpointClient,
    AF3LocalClient,
    run_af3_chatgpt_pipeline,
    save_sidecar,
)
from utils.env_config import get_env, load_project_env


load_project_env()


def _resolve_token(name_upper: str, name_lower: str) -> str:
    return get_env(name_upper, name_lower)


def _build_af3_client(
    backend: str,
    endpoint_url: str,
    hf_token: str,
    model_id: str,
    device: str,
    torch_dtype: str,
):
    if backend == "hf_endpoint":
        if not endpoint_url:
            raise HTTPException(status_code=400, detail="AF3 endpoint backend requires endpoint_url")
        return AF3EndpointClient(
            endpoint_url=endpoint_url,
            token=hf_token,
            model_id=model_id or DEFAULT_AF3_MODEL_ID,
        )
    return AF3LocalClient(
        model_id=model_id or DEFAULT_AF3_MODEL_ID,
        device=device,
        torch_dtype=torch_dtype,
    )


app = FastAPI(title="AF3 + ChatGPT Pipeline API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "react-ui" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"
if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="assets")


class PipelinePathRequest(BaseModel):
    audio_path: str
    backend: str = "hf_endpoint"
    endpoint_url: str = ""
    hf_token: str = ""
    model_id: str = DEFAULT_AF3_MODEL_ID
    device: str = "auto"
    torch_dtype: str = "auto"
    af3_prompt: str = DEFAULT_AF3_PROMPT
    af3_max_new_tokens: int = 1400
    af3_temperature: float = 0.1
    openai_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL
    user_context: str = ""
    artist_name: str = ""
    track_name: str = ""
    enable_web_search: bool = False
    output_json: str = ""


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/", include_in_schema=False)
def serve_root():
    if FRONTEND_DIST.exists():
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
    return JSONResponse(
        {
            "ok": True,
            "message": "Frontend build not found. Run `python af3_gui_app.py` or `npm --prefix react-ui run build`.",
        }
    )


@app.get("/api/config")
def config():
    return {
        "defaults": {
            "backend": "hf_endpoint",
            "endpoint_url": _resolve_token("HF_AF3_ENDPOINT_URL", "hf_af3_endpoint_url"),
            "model_id": _resolve_token("AF3_MODEL_ID", "af3_model_id") or DEFAULT_AF3_MODEL_ID,
            "openai_model": _resolve_token("OPENAI_MODEL", "openai_model") or DEFAULT_OPENAI_MODEL,
            "af3_prompt": DEFAULT_AF3_PROMPT,
        }
    }


@app.post("/api/pipeline/run-path")
def run_pipeline_path(req: PipelinePathRequest):
    audio_path = Path(req.audio_path)
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio not found: {audio_path}")

    hf_token = req.hf_token or _resolve_token("HF_TOKEN", "hf_token")
    openai_key = req.openai_api_key or _resolve_token("OPENAI_API_KEY", "openai_api_key")
    if not openai_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required.")

    endpoint_url = req.endpoint_url or _resolve_token("HF_AF3_ENDPOINT_URL", "hf_af3_endpoint_url")
    af3_client = _build_af3_client(
        backend=req.backend,
        endpoint_url=endpoint_url,
        hf_token=hf_token,
        model_id=req.model_id,
        device=req.device,
        torch_dtype=req.torch_dtype,
    )
    try:
        result = run_af3_chatgpt_pipeline(
            audio_path=str(audio_path),
            af3_client=af3_client,
            af3_prompt=req.af3_prompt,
            af3_max_new_tokens=req.af3_max_new_tokens,
            af3_temperature=req.af3_temperature,
            openai_api_key=openai_key,
            openai_model=req.openai_model,
            user_context=req.user_context,
            artist_name=req.artist_name,
            track_name=req.track_name,
            enable_web_search=req.enable_web_search,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_json = req.output_json or str(audio_path.with_suffix(".json"))
    save_path = save_sidecar(result["sidecar"], output_json)
    return {
        "saved_to": save_path,
        "af3_analysis": result["af3_analysis"],
        "cleaned": result["cleaned"],
        "sidecar": result["sidecar"],
    }


@app.post("/api/pipeline/run-upload")
async def run_pipeline_upload(
    audio_file: UploadFile = File(...),
    backend: str = Form("hf_endpoint"),
    endpoint_url: str = Form(""),
    hf_token: str = Form(""),
    model_id: str = Form(DEFAULT_AF3_MODEL_ID),
    device: str = Form("auto"),
    torch_dtype: str = Form("auto"),
    af3_prompt: str = Form(DEFAULT_AF3_PROMPT),
    af3_max_new_tokens: int = Form(1400),
    af3_temperature: float = Form(0.1),
    openai_api_key: str = Form(""),
    openai_model: str = Form(DEFAULT_OPENAI_MODEL),
    user_context: str = Form(""),
    artist_name: str = Form(""),
    track_name: str = Form(""),
    enable_web_search: bool = Form(False),
    output_json: str = Form(""),
):
    suffix = Path(audio_file.filename or "uploaded.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_audio = Path(tmp.name)
    try:
        content = await audio_file.read()
        temp_audio.write_bytes(content)

        hf_token_val = hf_token or _resolve_token("HF_TOKEN", "hf_token")
        openai_key = openai_api_key or _resolve_token("OPENAI_API_KEY", "openai_api_key")
        if not openai_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required.")

        endpoint_url_val = endpoint_url or _resolve_token("HF_AF3_ENDPOINT_URL", "hf_af3_endpoint_url")
        af3_client = _build_af3_client(
            backend=backend,
            endpoint_url=endpoint_url_val,
            hf_token=hf_token_val,
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )

        result = run_af3_chatgpt_pipeline(
            audio_path=str(temp_audio),
            af3_client=af3_client,
            af3_prompt=af3_prompt,
            af3_max_new_tokens=af3_max_new_tokens,
            af3_temperature=af3_temperature,
            openai_api_key=openai_key,
            openai_model=openai_model,
            user_context=user_context,
            artist_name=artist_name,
            track_name=track_name,
            enable_web_search=enable_web_search,
        )
        default_out = Path("outputs") / "af3_chatgpt" / (Path(audio_file.filename or "track").stem + ".json")
        save_path = save_sidecar(result["sidecar"], output_json or str(default_out))
        return {
            "saved_to": save_path,
            "af3_analysis": result["af3_analysis"],
            "cleaned": result["cleaned"],
            "sidecar": result["sidecar"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            temp_audio.unlink(missing_ok=True)
        except Exception:
            pass
