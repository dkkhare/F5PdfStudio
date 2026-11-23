"""ComfyUI API bridge for the Chatterbox multilingual TTS node."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests

DEFAULT_COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
DEFAULT_CHATTERBOX_MODEL_PACK = "chatterbox-multilingual-hi"
DEFAULT_CHATTERBOX_LANGUAGE = "hi"


def _base_url(value: str | None = None) -> str:
    return str(value or DEFAULT_COMFYUI_URL).strip().rstrip("/")


def check_chatterbox_comfyui(
    base_url: str | None = None,
    model_pack: str = DEFAULT_CHATTERBOX_MODEL_PACK,
    timeout: int = 8,
) -> str:
    """Check ComfyUI connectivity, node availability and the requested Hindi pack."""
    base = _base_url(base_url)
    try:
        response = requests.get(f"{base}/object_info", timeout=timeout)
        response.raise_for_status()
        info = response.json()
    except Exception as exc:
        return (
            f"ComfyUI is not reachable at {base}. "
            f"Start ComfyUI and confirm its address.\n{type(exc).__name__}: {exc}"
        )

    node = info.get("ChatterboxMTLTTS")
    if not isinstance(node, dict):
        return (
            f"ComfyUI is reachable at {base}, but the ChatterboxMTLTTS node "
            "was not found. Check that ComfyUI-Chatterbox is installed and "
            "restart ComfyUI."
        )

    required = node.get("input", {}).get("required", {})
    pack_spec = required.get("model_pack_name", [])
    available_packs = pack_spec[0] if pack_spec and isinstance(pack_spec[0], list) else []
    if not available_packs:
        return (
            f"ComfyUI and ChatterboxMTLTTS are ready at {base}, but the node "
            "did not report any installed model packs. Open the Chatterbox node "
            "in ComfyUI and confirm its Hindi pack is available."
        )
    if model_pack not in available_packs:
        return (
            f"ComfyUI and ChatterboxMTLTTS are ready at {base}, but the Hindi "
            f"model pack '{model_pack}' is not currently listed. Available packs: "
            + ", ".join(str(item) for item in available_packs)
        )

    return (
        f"Connected: {base}\n"
        "Chatterbox Multilingual TTS node: available\n"
        f"Model pack: {model_pack}\n"
        "Language: Hindi (hi)"
    )


def _find_audio_output(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        if value.get("filename") and value.get("type"):
            return {
                "filename": str(value["filename"]),
                "subfolder": str(value.get("subfolder") or ""),
                "type": str(value["type"]),
            }
        for child in value.values():
            found = _find_audio_output(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_audio_output(child)
            if found:
                return found
    return None


def synthesize_chatterbox_comfyui(
    text: str,
    output_path: str | Path,
    reference_audio: str | Path,
    *,
    base_url: str | None = None,
    model_pack: str = DEFAULT_CHATTERBOX_MODEL_PACK,
    language: str = DEFAULT_CHATTERBOX_LANGUAGE,
    timeout_seconds: int = 1800,
    progress_callback=None,
    seed: int | None = None,
) -> None:
    """Generate one section using ChatterboxMTLTTS and save a mono PCM-16 WAV."""
    import soundfile as sf
    import numpy as np

    text = str(text or "").strip()
    if not text:
        raise ValueError("Chatterbox received an empty text section.")
    reference = Path(reference_audio).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError("Chatterbox voice cloning requires a valid reference audio file.")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = _base_url(base_url)
    client_id = uuid.uuid4().hex

    if progress_callback:
        progress_callback("Uploading Chatterbox reference voice to ComfyUI…", 0, 3)

    with reference.open("rb") as handle:
        upload = requests.post(
            f"{base}/upload/image",
            data={"type": "input", "overwrite": "true"},
            files={"image": (f"f5pdf_chatterbox_{uuid.uuid4().hex}{reference.suffix or '.wav'}", handle)},
            timeout=60,
        )
    upload.raise_for_status()
    uploaded_name = str(upload.json().get("name") or "").strip()
    if not uploaded_name:
        raise RuntimeError(f"ComfyUI did not return an uploaded reference filename: {upload.text[:500]}")

    prompt_id_seed = int(seed) if seed is not None and int(seed) >= 0 else int(time.time_ns() % (2**32))
    prompt = {
        "1": {
            "class_type": "LoadAudio",
            "inputs": {"audio": uploaded_name},
        },
        "2": {
            "class_type": "ChatterboxMTLTTS",
            "inputs": {
                "model_pack_name": str(model_pack or DEFAULT_CHATTERBOX_MODEL_PACK),
                "model_version": "t3_mtl23ls_v3.safetensors",
                "language": str(language or DEFAULT_CHATTERBOX_LANGUAGE),
                "text": text,
                "exaggeration": 0.5,
                "temperature": 0.8,
                "cfg_weight": 0.5,
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "top_p": 1.0,
                "seed": prompt_id_seed,
                "use_watermark": False,
                "device": "auto",
                "dtype": "auto",
                "attention_mode": "sdpa",
                "crash_protection_template": "hmm ,, {seg} hmm ,,",
                "audio_prompt": ["1", 0],
            },
        },
        "3": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["2", 0],
                "filename_prefix": f"F5PdfStudio/Chatterbox/{destination.stem}",
            },
        },
    }

    if progress_callback:
        progress_callback("Queueing Chatterbox Hindi TTS on ComfyUI…", 1, 3)
    queued = requests.post(
        f"{base}/prompt",
        json={"prompt": prompt, "client_id": client_id},
        timeout=60,
    )
    if not queued.ok:
        raise RuntimeError(f"ComfyUI rejected the Chatterbox workflow ({queued.status_code}): {queued.text[:2000]}")
    prompt_id = str(queued.json().get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {queued.text[:1000]}")

    deadline = time.monotonic() + max(60, int(timeout_seconds))
    history_entry = None
    while time.monotonic() < deadline:
        time.sleep(2)
        response = requests.get(f"{base}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        history = response.json()
        history_entry = history.get(prompt_id)
        if history_entry:
            status = history_entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False and status.get("messages"):
                raise RuntimeError(f"ComfyUI Chatterbox job failed: {status.get('messages', status)}")
            outputs = history_entry.get("outputs") or {}
            audio_info = _find_audio_output(outputs)
            if audio_info or status.get("completed") is True:
                break
        if progress_callback:
            progress_callback("Chatterbox is generating the current section in ComfyUI…", 2, 3)
    else:
        raise TimeoutError(
            f"Chatterbox job {prompt_id} did not finish within {timeout_seconds} seconds. "
            "The ComfyUI queue may still be processing it."
        )

    if not history_entry:
        raise RuntimeError(f"ComfyUI returned no history for Chatterbox job {prompt_id}.")
    audio_info = _find_audio_output(history_entry.get("outputs") or {})
    if not audio_info:
        status = history_entry.get("status", {})
        raise RuntimeError(f"ComfyUI job finished without an audio file. Status: {status}")

    if progress_callback:
        progress_callback("Downloading Chatterbox WAV from ComfyUI…", 3, 3)
    audio_response = requests.get(
        f"{base}/view",
        params={
            "filename": audio_info["filename"],
            "subfolder": audio_info["subfolder"],
            "type": audio_info["type"],
        },
        timeout=120,
    )
    audio_response.raise_for_status()
    temporary = destination.with_suffix(destination.suffix + ".comfyui_download")
    temporary.write_bytes(audio_response.content)
    try:
        audio, sample_rate = sf.read(str(temporary), dtype="float32", always_2d=True)
        if audio.size == 0 or sample_rate <= 0:
            raise RuntimeError("ComfyUI returned an empty or invalid audio file.")
        mono = np.mean(audio, axis=1)
        sf.write(str(destination), mono, int(sample_rate), subtype="PCM_16", format="WAV")
    finally:
        temporary.unlink(missing_ok=True)

    if not destination.is_file() or destination.stat().st_size < 1000:
        raise RuntimeError("Chatterbox output WAV was not written correctly.")
