"""LTX-2.5 integration for F5PdfStudio.

The desktop app talks to a small HTTP bridge. This is intentional: the LTX-2.5
model is normally installed on a GPU machine/Colab, while F5PdfStudio runs on
Windows. Set LTX25_BASE_URL to the public/tunnel URL of colab_ltx_fish_bridge.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

LTX25_BASE_URL = os.getenv("LTX25_BASE_URL", "http://127.0.0.1:7861").rstrip("/")
LTX25_API_KEY = os.getenv("LTX25_API_KEY", "")


def _headers(api_key: str = "") -> dict[str, str]:
    key = str(api_key or LTX25_API_KEY or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def check_ltx25(base_url: str = LTX25_BASE_URL, api_key: str = "", timeout: int = 8) -> tuple[bool, str]:
    base = (base_url or LTX25_BASE_URL).rstrip("/")
    try:
        r = requests.get(f"{base}/health", headers=_headers(api_key), timeout=timeout)
        r.raise_for_status()
        data = r.json() if r.content else {}
        return True, f"LTX-2.5 bridge: Ready ({data.get('backend', 'unknown')})"
    except Exception as exc:
        return False, f"LTX-2.5 bridge: Not available — {exc}"


def generate_ltx25_video(
    prompt: str,
    output_path: str | Path,
    mode: str = "t2v",
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    base_url: str = LTX25_BASE_URL,
    api_key: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 81,
    frame_rate: float = 25.0,
    seed: int = -1,
    offload: str = "cpu",
    quantization: str = "fp8-cast",
    enhance_prompt: bool = False,
    timeout: int = 3600,
) -> str:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("LTX-2.5 prompt is empty.")

    mode = str(mode or "t2v").strip().lower()
    if mode not in {"t2v", "i2v", "a2v"}:
        raise ValueError("LTX-2.5 mode must be t2v, i2v or a2v.")
    if mode in {"i2v", "a2v"} and (not image_path or not Path(image_path).is_file()):
        raise FileNotFoundError("LTX-2.5 image-to-video/audio-to-video mode needs a valid image.")
    if mode == "a2v" and (not audio_path or not Path(audio_path).is_file()):
        raise FileNotFoundError("LTX-2.5 audio-to-video mode needs a valid audio file.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "prompt": prompt,
        "mode": mode,
        "width": str(int(width)),
        "height": str(int(height)),
        "num_frames": str(int(num_frames)),
        "frame_rate": str(float(frame_rate)),
        "seed": str(int(seed)),
        "offload": str(offload or "cpu"),
        "quantization": str(quantization or ""),
        "enhance_prompt": "true" if enhance_prompt else "false",
    }
    files = {}
    image_handle = None
    audio_handle = None
    try:
        if image_path:
            image_handle = open(image_path, "rb")
            files["image"] = (Path(image_path).name, image_handle, "application/octet-stream")
        if audio_path:
            audio_handle = open(audio_path, "rb")
            files["audio"] = (Path(audio_path).name, audio_handle, "audio/wav")

        r = requests.post(
            f"{(base_url or LTX25_BASE_URL).rstrip('/')}/generate",
            headers=_headers(api_key),
            data=data,
            files=files or None,
            timeout=max(60, int(timeout)),
        )
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:4000]
            raise RuntimeError(f"LTX-2.5 bridge HTTP {r.status_code}: {detail}")

        content_type = (r.headers.get("content-type") or "").lower()
        if "video/" in content_type or r.content[:4] == b"\x00\x00\x00\x18":
            path.write_bytes(r.content)
            return str(path.resolve())

        # Allow bridge implementations that return {"video_path": "..."} or a URL.
        try:
            body = r.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            candidate = body.get("video_path") or body.get("path") or body.get("video_url")
            if candidate:
                if str(candidate).startswith(("http://", "https://")):
                    vr = requests.get(str(candidate), headers=_headers(api_key), timeout=timeout)
                    vr.raise_for_status()
                    path.write_bytes(vr.content)
                elif Path(str(candidate)).is_file():
                    path.write_bytes(Path(str(candidate)).read_bytes())
                else:
                    raise RuntimeError(f"LTX bridge returned a missing video path: {candidate}")
                return str(path.resolve())
        raise RuntimeError("LTX-2.5 bridge returned no video file.")
    finally:
        if image_handle:
            image_handle.close()
        if audio_handle:
            audio_handle.close()
