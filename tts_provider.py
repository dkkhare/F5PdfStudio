"""Unified TTS providers for F5PdfStudio.

Providers:
- f5tts: handled by the existing local F5-Hindi pipeline in tts_service.py
- voicestudio: local VoiceStudio desktop API at http://127.0.0.1:3900/v1
- voicebox: local Voicebox REST API at http://127.0.0.1:17493
- sarvam: Sarvam Bulbul v3 cloud REST API

This module deliberately keeps provider-specific HTTP code separate from the
existing audiobook/checkpoint logic.
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import time

VOICESTUDIO_BASE_URL = os.getenv("VOICESTUDIO_BASE_URL", "http://127.0.0.1:3900/v1").rstrip("/")
VOICESTUDIO_API_KEY = os.getenv("VOICESTUDIO_API_KEY", "local")
VOICEBOX_BASE_URL = os.getenv("VOICEBOX_BASE_URL", "http://127.0.0.1:17493").rstrip("/")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = "bulbul:v3"
SARVAM_SPEAKERS = [
    "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "ashutosh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali",
]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "output")).strip("._-") or "output"


def check_voicestudio(base_url: str = VOICESTUDIO_BASE_URL, timeout: int = 5) -> tuple[bool, str]:
    base = base_url.rstrip("/")
    try:
        r = requests.get(base.rsplit("/v1", 1)[0] + "/health", timeout=timeout)
        r.raise_for_status()
        data = r.json() if r.content else {}
        return True, f"VoiceStudio: Ready (device={data.get('device', 'unknown')}, version={data.get('version', 'unknown')})"
    except Exception as exc:
        return False, f"VoiceStudio: Not available — {exc}"


def list_voicestudio_voices(base_url: str = VOICESTUDIO_BASE_URL, timeout: int = 8) -> list[dict[str, str]]:
    r = requests.get(
        f"{base_url.rstrip('/')}/audio/voices",
        headers={"Authorization": f"Bearer {VOICESTUDIO_API_KEY or 'local'}"},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    voices = payload.get("voices", []) if isinstance(payload, dict) else []
    result = [{
        "id": str(v.get("voice_id") or v.get("id") or "default"),
        "name": str(v.get("name") or v.get("voice_id") or "default"),
        "language": str(v.get("language") or ""),
        "type": str(v.get("type") or ""),
    } for v in voices if isinstance(v, dict)]
    return result



def check_voicebox(base_url: str = VOICEBOX_BASE_URL, timeout: int = 5) -> tuple[bool, str]:
    base = base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        r.raise_for_status()
        data = r.json() if r.content else {}
        return True, f"Voicebox: Ready (version={data.get('version', 'unknown')})"
    except Exception as exc:
        return False, f"Voicebox: Not available — {exc}"


def list_voicebox_profiles(base_url: str = VOICEBOX_BASE_URL, timeout: int = 8) -> list[dict[str, str]]:
    r = requests.get(f"{base_url.rstrip('/')}/profiles", timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    profiles = payload if isinstance(payload, list) else payload.get("profiles", []) if isinstance(payload, dict) else []
    result = []
    for profile in profiles:
        if not isinstance(profile, dict) or not profile.get("id"):
            continue
        result.append({
            "id": str(profile.get("id")),
            "name": str(profile.get("name") or profile.get("id")),
            "language": str(profile.get("language") or ""),
        })
    return result


def check_sarvam(api_key: Optional[str] = None, timeout: int = 8) -> tuple[bool, str]:
    key = (api_key or SARVAM_API_KEY).strip()
    if not key:
        return False, "Sarvam: API key not configured"
    # A tiny synthesis check would consume credits, so only validate that the key is present here.
    return True, "Sarvam: API key configured (Bulbul v3)"


def _save_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return str(path.resolve())


def synthesize_voicestudio(
    text: str,
    output_path: str | Path,
    voice: str = "default",
    model: str = "tts-1",
    response_format: str = "wav",
    reference_audio: str | Path | None = None,
    reference_text: str | None = None,
    language: str = "hi",
    speed: float = 1.0,
    timeout: int = 300,
) -> str:
    """Generate with VoiceStudio's task-oriented /generate endpoint.

    When reference_audio is supplied, VoiceStudio receives it as ref_audio and
    the exact transcript as ref_text, so the same reference clip used by
    F5-Hindi can be used for zero-profile voice cloning. A persistent profile
    ID may also be supplied through ``voice``.
    """
    if not text.strip():
        raise ValueError("VoiceStudio received empty text.")

    path = Path(output_path)
    root = VOICESTUDIO_BASE_URL.rsplit("/v1", 1)[0].rstrip("/")
    url = f"{root}/generate"

    data: dict[str, Any] = {
        "text": text,
        "language": language or "hi",
        "speed": max(0.25, min(4.0, float(speed))),
        "num_step": 16,
        "guidance_scale": 2,
        "denoise": "true",
        "postprocess_output": "true",
        "effect_preset": "broadcast",
        "max_chunk_chars": 800,
        "crossfade_ms": 50,
        "pronounce": "true",
        "stream": "false",
        "hosted": "false",
    }

    voice_id = (voice or "").strip()
    if voice_id and voice_id.lower() != "default":
        data["profile_id"] = voice_id

    ref_path = Path(reference_audio) if reference_audio else None
    files = None
    try:
        if ref_path and ref_path.exists():
            data["ref_text"] = str(reference_text or "")
            files = {
                "ref_audio": (ref_path.name, ref_path.open("rb"), "application/octet-stream")
            }
        elif reference_audio:
            raise FileNotFoundError(f"VoiceStudio reference audio not found: {ref_path}")

        headers = {"Authorization": f"Bearer {VOICESTUDIO_API_KEY or 'local'}"}
        r = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"VoiceStudio API {r.status_code}: {r.text[:1500]}")

        # /generate is documented as JSON. Be liberal about the exact audio
        # field because VoiceStudio versions may return base64, a path, or URL.
        try:
            body = r.json()
        except ValueError:
            body = None

        if isinstance(body, dict):
            candidates = [
                body.get("audio"), body.get("audio_base64"), body.get("audio_b64"),
                body.get("wav"), body.get("audio_url"), body.get("url"),
                body.get("path"), body.get("file"),
            ]
            audios = body.get("audios")
            if isinstance(audios, list):
                candidates.extend(audios)

            for value in candidates:
                if not value:
                    continue
                if isinstance(value, dict):
                    value = value.get("audio") or value.get("url") or value.get("path")
                if not isinstance(value, str):
                    continue
                if value.startswith("http://") or value.startswith("https://"):
                    ar = requests.get(value, timeout=timeout)
                    ar.raise_for_status()
                    return _save_bytes(path, ar.content)
                if Path(value).exists():
                    return _save_bytes(path, Path(value).read_bytes())
                try:
                    decoded = base64.b64decode(value, validate=True)
                    if decoded:
                        return _save_bytes(path, decoded)
                except Exception:
                    pass

            # Some builds return an artifact object with a nested data field.
            nested = body.get("data")
            if isinstance(nested, str):
                try:
                    return _save_bytes(path, base64.b64decode(nested, validate=True))
                except Exception:
                    pass

            raise RuntimeError(
                "VoiceStudio /generate succeeded but no audio was found in the JSON response. "
                f"Response keys: {', '.join(body.keys())}"
            )

        # Compatibility fallback if a VoiceStudio build returns raw WAV bytes.
        if r.content:
            return _save_bytes(path, r.content)
        raise RuntimeError("VoiceStudio returned an empty response.")
    finally:
        if files:
            file_obj = files["ref_audio"][1]
            try:
                file_obj.close()
            except Exception:
                pass



def synthesize_voicebox(
    text: str,
    output_path: str | Path,
    profile_id: str,
    language: str = "hi",
    model_size: str = "0.6B",
    engine: str = "qwen",
    seed: Optional[int] = None,
    timeout: int = 600,
) -> str:
    """Generate WAV through Voicebox, which may complete asynchronously.

    We first submit /generate, then poll /history/{generation_id} until the
    generation is complete.  Only after completion do we try /audio/{id} and
    /history/{id}/export-audio.  This avoids the common Voicebox race where
    /generate returns an ID while /audio/{id} still returns HTTP 500.
    """
    if not text.strip():
        raise ValueError("Voicebox received empty text.")
    profile = str(profile_id or "").strip()
    if not profile:
        raise ValueError("Select a Voicebox profile before generating audio.")

    payload: dict[str, Any] = {
        "profile_id": profile,
        "text": text,
        "language": str(language or "hi").strip(),
        "model_size": str(model_size or "0.6B").strip(),
    }
    if engine:
        payload["engine"] = str(engine).strip()
    if seed is not None:
        payload["seed"] = int(seed)

    r = requests.post(
        f"{VOICEBOX_BASE_URL}/generate",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"Voicebox API {r.status_code}: {r.text[:2000]}")

    try:
        body = r.json()
    except ValueError as exc:
        raise RuntimeError(f"Voicebox returned a non-JSON response: {r.text[:1000]}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Voicebox returned an unexpected response: {body!r}")

    path = Path(output_path)

    def copy_audio_file(candidate: Any) -> Optional[str]:
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return _save_bytes(path, candidate_path.read_bytes())
        return None

    # Some Voicebox versions return the completed file immediately.
    for key in ("audio_path", "path", "file"):
        saved = copy_audio_file(body.get(key))
        if saved:
            return saved

    generation_id = body.get("id") or body.get("generation_id")

    # Inline/base64 audio, if supplied.
    audio_value = body.get("audio") or body.get("audio_base64") or body.get("audio_b64")
    if isinstance(audio_value, str):
        try:
            return _save_bytes(path, base64.b64decode(audio_value, validate=True))
        except Exception:
            pass

    if not generation_id:
        raise RuntimeError(
            "Voicebox did not return a generation ID or usable audio. "
            f"Response keys: {', '.join(body.keys())}"
        )

    generation_id = str(generation_id)
    deadline = time.monotonic() + max(1, int(timeout))
    poll_interval = 2.0
    last_status = str(body.get("status") or "generating")
    last_error = ""

    terminal_success = {
        "completed", "complete", "success", "succeeded", "done",
        "finished", "ready", "generated"
    }
    terminal_failure = {
        "failed", "failure", "error", "cancelled", "canceled"
    }

    while time.monotonic() < deadline:
        # Check history first because /audio/{id} can return HTTP 500 while
        # Voicebox is still processing the generation.
        try:
            hr = requests.get(
                f"{VOICEBOX_BASE_URL}/history/{generation_id}",
                timeout=min(30, max(5, int(deadline - time.monotonic())))
            )
            if hr.ok:
                try:
                    history = hr.json()
                except ValueError:
                    history = {}
                if isinstance(history, dict):
                    last_status = str(
                        history.get("status")
                        or history.get("state")
                        or last_status
                    ).strip().lower()
                    last_error = str(
                        history.get("error")
                        or history.get("error_message")
                        or ""
                    )

                    for key in ("audio_path", "path", "file"):
                        saved = copy_audio_file(history.get(key))
                        if saved:
                            return saved

                    if last_status in terminal_failure:
                        raise RuntimeError(
                            "Voicebox generation failed. "
                            f"generation_id={generation_id}; status={last_status}; "
                            f"error={last_error or 'no error message returned'}"
                        )

                    # Even if the server uses a non-standard status, a newly
                    # available audio_path is sufficient to finish successfully.
                    if last_status in terminal_success:
                        break
            else:
                last_error = f"history HTTP {hr.status_code}: {hr.text[:500]}"
        except requests.RequestException as exc:
            last_error = str(exc)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    # Once generation is complete (or after the polling window), try the
    # download endpoints.  Some Voicebox builds expose audio only here.
    for endpoint in (
        f"{VOICEBOX_BASE_URL}/audio/{generation_id}",
        f"{VOICEBOX_BASE_URL}/history/{generation_id}/export-audio",
    ):
        try:
            ar = requests.get(endpoint, timeout=60)
            if ar.ok and ar.content:
                return _save_bytes(path, ar.content)
            last_error = f"{endpoint}: HTTP {ar.status_code}"
        except requests.RequestException as exc:
            last_error = f"{endpoint}: {exc}"

    waited = max(0, int(timeout))
    raise RuntimeError(
        "Voicebox did not create/retrieve an audio file. "
        f"generation_id={generation_id}; status={last_status or 'unknown'}; "
        f"waited_up_to={waited}s; {last_error or 'no audio_path was returned'}"
    )

def synthesize_sarvam(
    text: str,
    output_path: str | Path,
    api_key: Optional[str] = None,
    speaker: str = "shubh",
    language_code: str = "hi-IN",
    pace: float = 1.0,
    timeout: int = 300,
) -> str:
    if not text.strip():
        raise ValueError("Sarvam received empty text.")
    key = (api_key or SARVAM_API_KEY).strip()
    if not key:
        raise RuntimeError("Sarvam API key is missing. Enter it in the F5PdfStudio Generation tab or set SARVAM_API_KEY.")
    if len(text) > 2500:
        raise ValueError(f"Sarvam Bulbul v3 REST accepts at most 2500 characters per request; received {len(text)}.")
    speaker = (speaker or "shubh").strip().lower()
    if speaker not in SARVAM_SPEAKERS:
        raise ValueError(f"Unsupported Sarvam Bulbul v3 speaker: {speaker}")
    payload = {
        "text": text,
        "target_language_code": language_code,
        "speaker": speaker,
        "model": SARVAM_MODEL,
        "speech_sample_rate": 24000,
        "output_audio_codec": "wav",
        "pace": max(0.5, min(2.0, float(pace))),
    }
    # Sarvam currently documents language_code; accept either spelling server-side.
    payload["language_code"] = payload.pop("target_language_code")
    r = requests.post(
        f"{SARVAM_BASE_URL}/text-to-speech",
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"Sarvam API {r.status_code}: {r.text[:1500]}")
    try:
        body = r.json()
        encoded = body.get("audios", [None])[0]
        if not encoded:
            raise ValueError("No audio returned")
        audio = base64.b64decode(encoded)
    except Exception as exc:
        raise RuntimeError(f"Sarvam returned an unexpected audio response: {exc}") from exc
    return _save_bytes(Path(output_path), audio)


def provider_label(engine: str) -> str:
    return {
        "f5tts": "F5-Hindi",
        "voicestudio": "VoiceStudio",
        "voicebox": "Voicebox",
        "sarvam": "Sarvam Bulbul v3",
    }.get(str(engine).lower(), str(engine))


__all__ = [
    "VOICESTUDIO_BASE_URL", "VOICESTUDIO_API_KEY", "SARVAM_BASE_URL", "SARVAM_API_KEY",
    "SARVAM_SPEAKERS", "check_voicestudio", "list_voicestudio_voices",
    "VOICEBOX_BASE_URL", "check_voicebox", "list_voicebox_profiles",
    "check_sarvam", "synthesize_voicestudio", "synthesize_voicebox",
    "synthesize_sarvam", "provider_label",
]
