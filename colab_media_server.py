"""Single-GPU Google Colab media server for F5PdfStudio.

Hosts the three remote engines requested by the application:
  * Fish Speech S2/S2-Pro HTTP TTS
  * Wan2.2 video generation
  * LTX-2.5 distilled text/image-to-video

The Windows application talks only to this FastAPI service. None of these
three model stacks are imported or installed in the local F5PdfStudio env.

Environment variables (Colab):
  F5PDFSTUDIO_TOKEN             optional bearer token
  FISH_API_URL                  Fish server base, default http://127.0.0.1:8080
  WAN_ROOT                      Wan2.2 repository root
  WAN_T2V_CKPT                  T2V-A14B checkpoint
  WAN_TI2V_CKPT                 TI2V-5B checkpoint
  WAN_I2V_CKPT                  I2V-A14B checkpoint
  WAN_S2V_CKPT                  S2V-14B checkpoint
  LTX_ROOT                      LTX-2 repository root
  LTX_TRANSFORMER               LTX 2.5 distilled transformer
  LTX_TEXT_ENCODER              LTX 2.5 Gemma4 text encoder
  LTX_VIDEO_VAE                 LTX 2.5 video VAE
  LTX_AUDIO_VAE                 LTX 2.5 audio VAE
  LTX_DURATION_HEAD             optional duration head
  LTX_SPATIAL_UPSAMPLER         LTX 2.5 spatial upsampler
  LTX_WIDTH / LTX_HEIGHT        base stage dimensions, default 512x768
  LTX_FP8 / LTX_OFFLOAD         optional memory-saving flags
"""
from __future__ import annotations

import os
import json
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

FISH_PROCESS = None

import requests
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

APP_ROOT = Path(os.getenv("F5PDFSTUDIO_MEDIA_ROOT", "/content/f5pdfstudio_media"))
APP_ROOT.mkdir(parents=True, exist_ok=True)
TOKEN = os.getenv("F5PDFSTUDIO_TOKEN", "").strip()
FISH_API_URL = os.getenv("FISH_API_URL", "http://127.0.0.1:8081").rstrip("/")
FISH_REFERENCE_ROOT = Path(os.getenv("F5PDFSTUDIO_FISH_REFERENCE_ROOT", os.getenv("F5PDFSTUDIO_WORKSPACE", "/content/drive/MyDrive/AI_Video_Workspace"))) / "references" / "fish"
FISH_REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
WAN_ROOT = Path(os.getenv("WAN_ROOT", "/content/Wan2.2"))
WAN_GENERATE = WAN_ROOT / "generate.py"
LTX_ROOT = Path(os.getenv("LTX_ROOT", "/content/LTX-2"))
LTX_PYTHON = os.getenv("LTX_PYTHON", str(LTX_ROOT / ".venv" / "bin" / "python"))
LTX_TRANSFORMER = os.getenv("LTX_TRANSFORMER", str(LTX_ROOT / "models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"))
LTX_DEV_TRANSFORMER = os.getenv("LTX_DEV_TRANSFORMER", str(LTX_ROOT / "models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"))
LTX_DISTILLED_LORA = os.getenv("LTX_DISTILLED_LORA", str(LTX_ROOT / "models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"))
LTX_TEXT_ENCODER = os.getenv("LTX_TEXT_ENCODER", str(LTX_ROOT / "models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"))
LTX_VIDEO_VAE = os.getenv("LTX_VIDEO_VAE", str(LTX_ROOT / "models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"))
LTX_AUDIO_VAE = os.getenv("LTX_AUDIO_VAE", str(LTX_ROOT / "models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors"))
LTX_DURATION_HEAD = os.getenv("LTX_DURATION_HEAD", str(LTX_ROOT / "models/ltx-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors"))
LTX_SPATIAL_UPSAMPLER = os.getenv("LTX_SPATIAL_UPSAMPLER", str(LTX_ROOT / "models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"))
LTX_WIDTH = int(os.getenv("LTX_WIDTH", "512"))
LTX_HEIGHT = int(os.getenv("LTX_HEIGHT", "768"))

app = FastAPI(title="F5PdfStudio Colab Media Server", version="1.0")


def _auth(authorization: Optional[str] = Header(default=None)):
    if not TOKEN:
        return
    expected = f"Bearer {TOKEN}"
    if not authorization or not secrets.compare_digest(authorization.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


class FishRequest(BaseModel):
    text: str
    reference_id: str = ""
    language: str = "hi"
    format: str = "wav"


def ensure_fish_server():
    global FISH_PROCESS
    try:
        if requests.get(f"{FISH_API_URL}/v1/health", timeout=2).ok:
            _restore_fish_references()
            return
    except Exception:
        pass
    root = Path(os.getenv("FISH_ROOT", "/content/fish-speech"))
    py = os.getenv("FISH_PYTHON", str(root / ".venv/bin/python"))
    llama = os.getenv("FISH_LLAMA_CHECKPOINT", "/content/fish-speech/checkpoints/s2-pro")
    decoder = os.getenv("FISH_DECODER_CHECKPOINT", "/content/fish-speech/checkpoints/s2-pro/codec.pth")
    port = int(os.getenv("FISH_PORT", "8081"))
    cmd = [py if Path(py).is_file() else "python", str(root / "tools/api_server.py"), "--llama-checkpoint-path", llama, "--decoder-checkpoint-path", decoder, "--listen", f"127.0.0.1:{port}", "--half"]
    if os.getenv("FISH_LAZY_LOAD", "1") == "1": cmd.append("--lazy-load")
    FISH_PROCESS = __import__("subprocess").Popen(cmd, cwd=str(root))
    for _ in range(180):
        try:
            if requests.get(f"{FISH_API_URL}/v1/health", timeout=2).ok:
                return
        except Exception:
            pass
        import time; time.sleep(1)
    _restore_fish_references()
    return


def _fish_reference_dir(reference_id: str) -> Path:
    safe = Path(str(reference_id or "").strip()).name
    if not safe or safe in {".", ".."}:
        raise HTTPException(400, "Invalid Fish Speech reference ID")
    return FISH_REFERENCE_ROOT / safe


def _native_fish_references() -> list[str]:
    try:
        r = requests.get(f"{FISH_API_URL}/v1/references/list", timeout=30)
        if r.ok:
            data = r.json()
            return [str(x) for x in (data.get("reference_ids") or [])]
    except Exception:
        pass
    return []


def _register_fish_reference(reference_id: str, audio_path: Path, text: str) -> None:
    with audio_path.open("rb") as f:
        r = requests.post(
            f"{FISH_API_URL}/v1/references/add",
            files={"audio": (audio_path.name, f, "audio/wav")},
            data={"id": reference_id, "text": text},
            timeout=300,
        )
    if not r.ok:
        raise RuntimeError(f"Fish Speech reference registration failed ({r.status_code}): {r.text[:3000]}")


def _restore_fish_references() -> None:
    native = set(_native_fish_references())
    if not FISH_REFERENCE_ROOT.exists():
        return
    for directory in sorted(FISH_REFERENCE_ROOT.iterdir()):
        if not directory.is_dir() or directory.name in native:
            continue
        audio = directory / "reference.wav"
        text_file = directory / "reference.txt"
        if audio.is_file() and text_file.is_file():
            try:
                _register_fish_reference(directory.name, audio, text_file.read_text(encoding="utf-8").strip())
            except Exception as exc:
                print(f"[Fish Speech] reference restore skipped for {directory.name}: {exc}")


def _save_upload(upload: UploadFile, directory: Path, suffix: str = "") -> Path:
    name = Path(upload.filename or "input.bin").name
    if suffix and not Path(name).suffix:
        name += suffix
    path = directory / name
    with path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


@app.get("/health")
def health(_: None = Depends(_auth)):
    fish_ok = False
    try:
        fish_ok = requests.get(f"{FISH_API_URL}/v1/health", timeout=5).ok
    except Exception:
        pass
    return {
        "status": "ok",
        "fish_speech": fish_ok,
        "wan2.2": WAN_GENERATE.is_file(),
        "ltx2.5": all(Path(p).is_file() for p in [LTX_TRANSFORMER, LTX_TEXT_ENCODER, LTX_VIDEO_VAE, LTX_AUDIO_VAE, LTX_SPATIAL_UPSAMPLER]),
        "ltx2.5_a2vid": all(Path(p).is_file() for p in [LTX_DEV_TRANSFORMER, LTX_DISTILLED_LORA, LTX_TEXT_ENCODER, LTX_VIDEO_VAE, LTX_AUDIO_VAE, LTX_SPATIAL_UPSAMPLER]),
        "gpu": _gpu_name(),
    }


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "unknown"


@app.post("/admin/release-fish")
def release_fish(_: None = Depends(_auth)):
    global FISH_PROCESS
    fish_pid = os.getenv("FISH_PID", "").strip()
    if FISH_PROCESS is not None:
        try:
            FISH_PROCESS.terminate()
            FISH_PROCESS.wait(timeout=15)
        except Exception:
            try: FISH_PROCESS.kill()
            except Exception: pass
        FISH_PROCESS = None
    elif fish_pid:
        try:
            import signal
            os.kill(int(fish_pid), signal.SIGTERM)
        except Exception:
            pass
        os.environ.pop("FISH_PID", None)
    return {"status": "fish-released"}


@app.get("/v1/fish/references/list")
def fish_references_list(_: None = Depends(_auth)):
    ensure_fish_server()
    native = _native_fish_references()
    persistent = []
    if FISH_REFERENCE_ROOT.exists():
        persistent = sorted(p.name for p in FISH_REFERENCE_ROOT.iterdir() if p.is_dir())
    return {"reference_ids": sorted(set(native) | set(persistent)), "native_reference_ids": native, "persistent_reference_ids": persistent}


@app.post("/v1/fish/reference/add")
async def fish_reference_add(
    reference_id: str = Form(...),
    text: str = Form(...),
    audio: UploadFile = File(...),
    _: None = Depends(_auth),
):
    reference_id = Path(str(reference_id).strip()).name
    text = str(text or "").strip()
    if not reference_id or reference_id in {".", ".."} or not text:
        raise HTTPException(400, "Reference ID and exact reference transcript are required")
    ensure_fish_server()
    directory = _fish_reference_dir(reference_id)
    directory.mkdir(parents=True, exist_ok=True)
    audio_path = directory / "reference.wav"
    with audio_path.open("wb") as f:
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    (directory / "reference.txt").write_text(text, encoding="utf-8")
    try:
        _register_fish_reference(reference_id, audio_path, text)
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"success": True, "reference_id": reference_id, "path": str(directory)}


@app.delete("/v1/fish/reference/{reference_id}")
def fish_reference_delete(reference_id: str, _: None = Depends(_auth)):
    ensure_fish_server()
    try:
        r = requests.request("DELETE", f"{FISH_API_URL}/v1/references/delete", json={"reference_id": reference_id}, timeout=60)
        if not r.ok and r.status_code != 404:
            raise HTTPException(r.status_code, r.text[:3000])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    directory = _fish_reference_dir(reference_id)
    if directory.exists():
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
    return {"success": True, "reference_id": reference_id}


@app.post("/v1/fish/tts")
def fish_tts(req: FishRequest, _: None = Depends(_auth)):
    ensure_fish_server()
    payload = {
        "text": req.text,
        "reference_id": req.reference_id,
        "language": req.language,
        "format": req.format,
        "streaming": False,
        "normalize": True,
    }
    try:
        r = requests.post(f"{FISH_API_URL}/v1/tts", json=payload, timeout=3600)
    except Exception as exc:
        raise HTTPException(502, f"Fish Speech server is unreachable: {exc}") from exc
    if not r.ok:
        raise HTTPException(r.status_code, r.text[:3000])
    return Response(content=r.content, media_type="audio/wav" if req.format == "wav" else r.headers.get("content-type", "application/octet-stream"))


def _wan_ckpt(task: str, requested: str) -> str:
    if requested:
        return requested
    mapping = {
        "t2v-A14B": os.getenv("WAN_T2V_CKPT", str(WAN_ROOT / "Wan2.2-T2V-A14B")),
        "ti2v-5B": os.getenv("WAN_TI2V_CKPT", str(WAN_ROOT / "Wan2.2-TI2V-5B")),
        "i2v-A14B": os.getenv("WAN_I2V_CKPT", str(WAN_ROOT / "Wan2.2-I2V-A14B")),
        "s2v-14B": os.getenv("WAN_S2V_CKPT", str(WAN_ROOT / "Wan2.2-S2V-14B")),
    }
    return mapping.get(task, mapping["t2v-A14B"])


def _run(cmd: list[str], timeout: int = 7200) -> None:
    proc = subprocess.run(cmd, cwd=str(WAN_ROOT if "Wan2.2" in str(cmd[0]) or str(WAN_ROOT) in " ".join(cmd) else Path.cwd()), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "process failed")[-8000:])


@app.post("/v1/wan/generate")
def wan_generate(
    prompt: str = Form(...),
    task: str = Form("t2v-A14B"),
    checkpoint_dir: str = Form(""),
    size: str = Form("1280*720"),
    frame_num: int = Form(81),
    sample_steps: int = Form(20),
    seed: int = Form(-1),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    _: None = Depends(_auth),
):
    release_fish()
    work = Path(tempfile.mkdtemp(prefix="wan_", dir=str(APP_ROOT)))
    out = work / "result.mp4"
    image_path = _save_upload(image, work) if image else None
    audio_path = _save_upload(audio, work, ".wav") if audio else None
    ckpt = _wan_ckpt(task, checkpoint_dir)
    cmd = [
        os.getenv("WAN_PYTHON", "python"), str(WAN_GENERATE),
        "--task", task, "--ckpt_dir", ckpt, "--size", size,
        "--frame_num", str(int(frame_num)), "--sample_steps", str(int(sample_steps)),
        "--base_seed", str(int(seed)), "--offload_model", "True", "--convert_model_dtype",
        "--t5_cpu", "True", "--save_file", str(out), "--prompt", prompt,
    ]
    if image_path:
        cmd += ["--image", str(image_path)]
    if audio_path:
        cmd += ["--audio", str(audio_path)]
    if task == "s2v-14B":
        cmd += ["--start_from_ref", "True"]
    try:
        _run(cmd)
    except Exception as exc:
        raise HTTPException(500, f"Wan2.2 generation failed: {exc}") from exc
    if not out.is_file():
        raise HTTPException(500, "Wan2.2 completed without producing an MP4")
    return FileResponse(out, media_type="video/mp4", filename="wan_result.mp4")


@app.post("/v1/ltx/generate")
def ltx_generate(
    prompt: str = Form(...),
    mode: str = Form("t2v_i2v_single"),
    duration: float = Form(5.0),
    fps: int = Form(24),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    _: None = Depends(_auth),
):
    release_fish()
    work = Path(tempfile.mkdtemp(prefix="ltx25_", dir=str(APP_ROOT)))
    out = work / "result.mp4"
    image_path = _save_upload(image, work) if image else None
    audio_path = _save_upload(audio, work, ".wav") if audio else None
    use_a2vid = str(mode).lower() in {"a2vid", "audio_image", "audio2video"} and audio_path
    if use_a2vid:
        cmd = [
            LTX_PYTHON if Path(LTX_PYTHON).is_file() else "python", "-m", "ltx_pipelines.a2vid_two_stage",
            "--transformer-path", LTX_DEV_TRANSFORMER,
            "--text-encoder-path", LTX_TEXT_ENCODER,
            "--video-vae-path", LTX_VIDEO_VAE,
            "--audio-vae-path", LTX_AUDIO_VAE,
            "--spatial-upsampler-path", LTX_SPATIAL_UPSAMPLER,
            "--distilled-lora", LTX_DISTILLED_LORA,
            "--height", str(LTX_HEIGHT), "--width", str(LTX_WIDTH),
            "--frame-rate", str(int(fps)), "--audio-max-duration", str(float(duration)),
            "--seed", "42", "--output-path", str(out), "--prompt", prompt,
            "--audio-path", str(audio_path),
        ]
        if image_path:
            cmd += ["--image", str(image_path), "0", "1.0"]
        if os.getenv("LTX_FP8", "1").lower() in {"1", "true", "yes"}:
            cmd += ["--quantization", "fp8-cast"]
        offload = os.getenv("LTX_OFFLOAD", "cpu").strip()
        if offload:
            cmd += ["--offload", offload]
    else:
        num_frames = max(9, int(float(duration) * int(fps)) + 1)
        num_frames = 8 * ((num_frames - 1 + 7) // 8) + 1
        cmd = [
            LTX_PYTHON if Path(LTX_PYTHON).is_file() else "python", "-m", "ltx_pipelines.distilled",
            "--transformer-path", LTX_TRANSFORMER,
            "--text-encoder-path", LTX_TEXT_ENCODER,
            "--video-vae-path", LTX_VIDEO_VAE,
            "--audio-vae-path", LTX_AUDIO_VAE,
            "--spatial-upsampler-path", LTX_SPATIAL_UPSAMPLER,
            "--height", str(LTX_HEIGHT), "--width", str(LTX_WIDTH),
            "--num-frames", str(num_frames), "--frame-rate", str(int(fps)),
            "--seed", "42", "--output-path", str(out), "--prompt", prompt,
        ]
        if Path(LTX_DURATION_HEAD).is_file(): cmd += ["--duration-head-path", LTX_DURATION_HEAD]
        if image_path and mode != "t2v": cmd += ["--image", str(image_path), "0", "1.0"]
        if os.getenv("LTX_FP8", "1").lower() in {"1", "true", "yes"}: cmd += ["--quantization", "fp8-cast"]
        offload = os.getenv("LTX_OFFLOAD", "cpu").strip()
        if offload: cmd += ["--offload", offload]
    try:
        _run(cmd, timeout=10800)
    except Exception as exc:
        raise HTTPException(500, f"LTX 2.5 generation failed: {exc}") from exc
    if not out.is_file():
        raise HTTPException(500, "LTX 2.5 completed without producing an MP4")
    return FileResponse(out, media_type="video/mp4", filename="ltx25_result.mp4")
