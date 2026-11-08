"""Colab bridge for F5PdfStudio + LTX-2.5 + Fish Speech.

Run this file INSIDE the Colab runtime that already contains the models.

Environment variables:
  LTX_ROOT=/content/LTX-2
  LTX_PYTHON=python
  LTX_TRANSFORMER=/content/.../ltx-2.5-22b-distilled-transformer-bf16.safetensors
  LTX_TEXT_ENCODER=/content/.../gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
  LTX_VIDEO_VAE=/content/.../ltx-2.5-video-vae-bf16.safetensors
  LTX_AUDIO_VAE=/content/.../ltx-2.5-audio-vae-bf16.safetensors
  LTX_UPSCALER=/content/.../ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
  FISH_URL=http://127.0.0.1:8080
  BRIDGE_API_KEY=optional-secret

The desktop application sends files to /generate. The bridge runs the official
LTX-2.5 distilled pipeline as a subprocess, so it does not assume how the user's
Colab notebook was originally started.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, Response
import requests
import uvicorn

APP = FastAPI(title="F5PdfStudio Colab AI Bridge", version="1.0")
API_KEY = os.getenv("BRIDGE_API_KEY", "").strip()
LTX_ROOT = Path(os.getenv("LTX_ROOT", "/content/LTX-2")).resolve()
LTX_PYTHON = os.getenv("LTX_PYTHON", "python")
LTX_TRANSFORMER = os.getenv("LTX_TRANSFORMER", "")
LTX_TEXT_ENCODER = os.getenv("LTX_TEXT_ENCODER", "")
LTX_VIDEO_VAE = os.getenv("LTX_VIDEO_VAE", "")
LTX_AUDIO_VAE = os.getenv("LTX_AUDIO_VAE", "")
LTX_UPSCALER = os.getenv("LTX_UPSCALER", "")
LTX_DISTILLED_LORA = os.getenv("LTX_DISTILLED_LORA", "")
FISH_URL = os.getenv("FISH_URL", "http://127.0.0.1:8080").rstrip("/")


def auth(authorization: str | None) -> None:
    if API_KEY and (authorization or "").removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid bridge API key")


def ltx_paths() -> list[str]:
    required = {
        "LTX_TRANSFORMER": LTX_TRANSFORMER,
        "LTX_TEXT_ENCODER": LTX_TEXT_ENCODER,
        "LTX_VIDEO_VAE": LTX_VIDEO_VAE,
        "LTX_AUDIO_VAE": LTX_AUDIO_VAE,
        "LTX_UPSCALER": LTX_UPSCALER,
    }
    missing = [k for k, v in required.items() if not v or not Path(v).is_file()]
    if missing:
        raise RuntimeError("Missing LTX-2.5 model path(s): " + ", ".join(missing))
    return [
        "--transformer-path", LTX_TRANSFORMER,
        "--text-encoder-path", LTX_TEXT_ENCODER,
        "--video-vae-path", LTX_VIDEO_VAE,
        "--audio-vae-path", LTX_AUDIO_VAE,
        "--spatial-upsampler-path", LTX_UPSCALER,
    ]


@APP.get("/health")
def health(authorization: str | None = Header(default=None)):
    auth(authorization)
    fish_ok = False
    try:
        r = requests.get(f"{FISH_URL}/v1/health", timeout=3)
        fish_ok = r.ok
    except Exception:
        pass
    return {
        "status": "ok",
        "backend": "LTX-2.5 + Fish Speech bridge",
        "ltx_root": str(LTX_ROOT),
        "fish_available": fish_ok,
    }


@APP.get("/v1/health")
def fish_health_proxy(authorization: str | None = Header(default=None)):
    auth(authorization)
    try:
        r = requests.get(f"{FISH_URL}/v1/health", timeout=10)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    except Exception as exc:
        raise HTTPException(502, f"Fish Speech health proxy failed: {exc}") from exc


@APP.post("/v1/tts")
async def fish_tts_proxy(request: Request, authorization: str | None = Header(default=None)):
    """Forward the official Fish Speech msgpack request to the local Colab server."""
    auth(authorization)
    body = await request.body()
    headers = {"Content-Type": request.headers.get("content-type", "application/msgpack")}
    if authorization:
        headers["Authorization"] = authorization
    try:
        r = requests.post(
            f"{FISH_URL}/v1/tts",
            params=dict(request.query_params),
            data=body,
            headers=headers,
            timeout=900,
        )
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "audio/wav" if r.ok else "application/json"),
        )
    except Exception as exc:
        raise HTTPException(502, f"Fish Speech proxy failed: {exc}") from exc


@APP.post("/generate")
async def generate(
    prompt: str = Form(...),
    mode: str = Form("t2v"),
    width: int = Form(768),
    height: int = Form(512),
    num_frames: int = Form(81),
    frame_rate: float = Form(25.0),
    seed: int = Form(-1),
    offload: str = Form("cpu"),
    quantization: str = Form("fp8-cast"),
    enhance_prompt: str = Form("false"),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
):
    auth(authorization)
    mode = mode.lower().strip()
    if mode not in {"t2v", "i2v", "a2v"}:
        raise HTTPException(400, "mode must be t2v, i2v or a2v")
    if mode in {"i2v", "a2v"} and image is None:
        raise HTTPException(400, "image is required for i2v/a2v")
    if mode == "a2v" and audio is None:
        raise HTTPException(400, "audio is required for a2v")

    temp = Path(tempfile.mkdtemp(prefix="f5pdfstudio_ltx_"))
    try:
        image_path = None
        audio_path = None
        output_path = temp / "ltx_output.mp4"
        if image:
            image_path = temp / (Path(image.filename or "input.png").name)
            image_path.write_bytes(await image.read())
        if audio:
            audio_path = temp / (Path(audio.filename or "input.wav").name)
            audio_path.write_bytes(await audio.read())

        cmd = [
            LTX_PYTHON, "-m", "ltx_pipelines.distilled",
            *ltx_paths(),
            "--prompt", prompt,
            "--output-path", str(output_path),
            "--width", str(max(64, int(width))),
            "--height", str(max(64, int(height))),
            "--num-frames", str(max(9, int(num_frames))),
            "--frame-rate", str(float(frame_rate)),
            "--offload", offload,
        ]
        if quantization:
            cmd += ["--quantization", quantization]
        if enhance_prompt.lower() == "true":
            cmd.append("--enhance-prompt")
        if seed >= 0:
            cmd += ["--seed", str(seed)]
        if image_path:
            cmd += ["--image", str(image_path), "0", "1.0"]

        if mode == "a2v":
            # A2V uses a separate pipeline. Rebuild the command using the same
            # split model paths and the input audio.
            cmd = [
                LTX_PYTHON, "-m", "ltx_pipelines.a2vid_two_stage",
                *ltx_paths(),
                "--prompt", prompt,
                "--output-path", str(output_path),
                "--width", str(max(64, int(width))),
                "--height", str(max(64, int(height))),
                "--num-frames", str(max(9, int(num_frames))),
                "--frame-rate", str(float(frame_rate)),
                "--audio-path", str(audio_path),
                "--offload", offload,
            ]
            if not LTX_DISTILLED_LORA or not Path(LTX_DISTILLED_LORA).is_file():
                raise HTTPException(500, "LTX_DISTILLED_LORA is required for A2V and was not found.")
            cmd += ["--distilled-lora", LTX_DISTILLED_LORA, "0.8"]
            if quantization:
                cmd += ["--quantization", quantization]
            if enhance_prompt.lower() == "true":
                cmd.append("--enhance-prompt")
            if seed >= 0:
                cmd += ["--seed", str(seed)]
            if image_path:
                cmd += ["--image", str(image_path), "0", "1.0"]

        completed = subprocess.run(
            cmd,
            cwd=str(LTX_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        log = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        if completed.returncode != 0:
            raise HTTPException(500, f"LTX generation failed (exit {completed.returncode}).\n{log[-12000:]}")
        if not output_path.is_file():
            raise HTTPException(500, f"LTX finished without creating {output_path}.\n{log[-8000:]}")
        return FileResponse(str(output_path), media_type="video/mp4", filename="ltx_output.mp4")
    finally:
        # FileResponse streams after the endpoint returns, so the temp directory
        # must remain available. It is intentionally cleaned by the OS/Colab
        # session rather than deleted here.
        pass


if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=int(os.getenv("BRIDGE_PORT", "7861")))
