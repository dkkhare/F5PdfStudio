"""Client for the single Google Colab media server used by F5PdfStudio.

The local Windows application never installs or imports Wan2.2, LTX-2.5 or
Fish Speech. It only uploads inputs to the Colab HTTP endpoint and downloads
finished media into the current project.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

DEFAULT_COLAB_URL = os.getenv("F5PDFSTUDIO_COLAB_URL", "http://127.0.0.1:7861").rstrip("/")
DEFAULT_TIMEOUT = int(os.getenv("F5PDFSTUDIO_COLAB_TIMEOUT", "7200"))


def _headers(token: str = "") -> dict[str, str]:
    token = str(token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _url(base_url: str, path: str) -> str:
    return str(base_url or DEFAULT_COLAB_URL).rstrip("/") + "/" + path.lstrip("/")


def check_colab_server(base_url: str = DEFAULT_COLAB_URL, token: str = "") -> str:
    try:
        r = requests.get(_url(base_url, "/health"), headers=_headers(token), timeout=30)
        r.raise_for_status()
        body = r.json()
        return "Colab media server: READY\n" + "\n".join(f"{k}: {v}" for k, v in body.items())
    except Exception as exc:
        return f"Colab media server: NOT READY\n{type(exc).__name__}: {exc}"


def _download_response(response: requests.Response, output_path: str | Path) -> str:
    response.raise_for_status()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    if path.stat().st_size < 1000:
        raise RuntimeError(f"Colab returned an unusably small media file: {path}")
    return str(path.resolve())


def fish_speech_tts(*, base_url: str, token: str, text: str, output_path: str | Path,
                    reference_id: str = "", language: str = "hi") -> str:
    """Use Fish Speech's OpenAI-compatible endpoint exposed by the Colab wrapper."""
    payload = {
        "text": str(text),
        "reference_id": str(reference_id or ""),
        "language": str(language or "hi"),
        "response_format": "wav",
        "stream": False,
    }
    r = requests.post(_url(base_url, "/v1/fish/tts"), headers={**_headers(token), "Content-Type": "application/json"}, json=payload, timeout=DEFAULT_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"Fish Speech Colab API {r.status_code}: {r.text[:3000]}")
    return _download_response(r, output_path)



def list_fish_references(*, base_url: str, token: str) -> list[str]:
    r = requests.get(_url(base_url, "/v1/fish/references/list"), headers=_headers(token), timeout=60)
    if not r.ok:
        raise RuntimeError(f"Fish Speech references API {r.status_code}: {r.text[:3000]}")
    body = r.json()
    return [str(x) for x in (body.get("reference_ids") or [])]

def _multipart_file(path: str, field: str):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return (field, (p.name, p.read_bytes(), "application/octet-stream"))


def _post_media(endpoint: str, *, base_url: str, token: str, data: dict, files: dict,
                output_path: str | Path) -> str:
    response = requests.post(_url(base_url, endpoint), headers=_headers(token), data=data, files=files,
                             timeout=DEFAULT_TIMEOUT)
    return _download_response(response, output_path)


def generate_wan_remote(base_url: str, token: str, *, prompt: str, image_path: str = "",
                        audio_path: str = "", task: str = "t2v-A14B", checkpoint_dir: str = "",
                        size: str = "1280*720", frame_num: int = 81, sample_steps: int = 20,
                        seed: int = -1, output_path: str | Path) -> str:
    data = {
        "prompt": prompt, "task": task, "checkpoint_dir": checkpoint_dir,
        "size": size, "frame_num": str(int(frame_num)), "sample_steps": str(int(sample_steps)),
        "seed": str(int(seed)),
    }
    files = {}
    if image_path:
        p = Path(image_path); files["image"] = (p.name, p.read_bytes(), "application/octet-stream")
    if audio_path:
        p = Path(audio_path); files["audio"] = (p.name, p.read_bytes(), "audio/wav")
    return _post_media("/v1/wan/generate", base_url=base_url, token=token, data=data, files=files, output_path=output_path)


def generate_ltx_remote(base_url: str, token: str, *, prompt: str, image_path: str = "", audio_path: str = "",
                        mode: str = "t2v_i2v_single", duration: float = 5.0, fps: int = 24,
                        output_path: str | Path) -> str:
    data = {"prompt": prompt, "mode": mode, "duration": str(float(duration)), "fps": str(int(fps))}
    files = {}
    if image_path:
        p = Path(image_path); files["image"] = (p.name, p.read_bytes(), "application/octet-stream")
    if audio_path:
        p = Path(audio_path); files["audio"] = (p.name, p.read_bytes(), "audio/wav")
    return _post_media("/v1/ltx/generate", base_url=base_url, token=token, data=data, files=files, output_path=output_path)
