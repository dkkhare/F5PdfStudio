"""Client for the kVoive Hindi TTS worker API.

The API key is supplied by the caller or TTS_API_KEY environment variable;
secrets are deliberately never written to project JSON or source files.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests

DEFAULT_API_URL = "https://definitions-taxation-essay-excited.trycloudflare.com"
CONTENT_TYPES = ("auto", "poetry", "drama", "story", "novel", "prose")


def _config(api_url: str | None = None, api_key: str | None = None):
    url = (api_url or os.getenv("TTS_API_URL") or DEFAULT_API_URL).rstrip("/")
    key = api_key or os.getenv("TTS_API_KEY", "")
    if not url:
        raise ValueError("kVoive API URL is required.")
    if not key:
        raise ValueError("kVoive API key is required. Enter it in Audio Production or set TTS_API_KEY.")
    return url, key


def _headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}"}


def _check(response: requests.Response):
    if response.status_code >= 400:
        raise RuntimeError(f"kVoive API {response.status_code}: {response.text[:500]}")
    return response


def check_health(api_url: str | None = None, timeout: int = 30):
    url = (api_url or os.getenv("TTS_API_URL") or DEFAULT_API_URL).rstrip("/")
    response = requests.get(f"{url}/health", timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"kVoive health check {response.status_code}: {response.text[:300]}")
    return response.json()


def tts(text: str, out_path: str | Path, *, api_url=None, api_key=None, voice=None,
        language="hi", fmt="wav", respect_lines=True, master=True, lufs=None,
        styles=None, content_type=None, speakers=None, type_options=None, gender=None):
    url, key = _config(api_url, api_key)
    payload = {
        "text": text, "voice": voice or None, "language": language, "format": fmt,
        "respect_lines": bool(respect_lines), "master": bool(master), "lufs": lufs,
        "styles": styles, "content_type": content_type, "speakers": speakers,
        "type_options": type_options, "gender": gender,
    }
    response = _check(requests.post(f"{url}/v1/tts", headers=_headers(key),
                                     timeout=1800, json=payload))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(response.content)
    try:
        notes = json.loads(response.headers.get("X-Voice-Notes", "[]"))
    except (TypeError, ValueError):
        notes = []
    return {"audio_seconds": float(response.headers.get("X-Audio-Seconds", 0) or 0),
            "voice_notes": notes}


def _section(item):
    data = {"id": item[0], "text": item[1]}
    if len(item) > 2 and item[2]:
        data["content_type"] = item[2]
    if len(item) > 3 and item[3]:
        data["speakers"] = item[3]
    return data


def submit_batch(book, part, sections, *, api_url=None, api_key=None, voice=None,
                 language="hi", fmt="wav", respect_lines=True, resume=True, lufs=None,
                 styles=None, content_type=None, speakers=None, type_options=None, gender=None):
    url, key = _config(api_url, api_key)
    response = _check(requests.post(f"{url}/v1/batch", headers=_headers(key), timeout=60, json={
        "book": book, "part": part, "voice": voice, "language": language, "format": fmt,
        "respect_lines": respect_lines, "resume": resume, "lufs": lufs, "styles": styles,
        "content_type": content_type, "speakers": speakers, "type_options": type_options,
        "gender": gender, "sections": [_section(s) for s in sections],
    }))
    return response.json()["job_id"]


def job_status(job_id, *, api_url=None, api_key=None):
    url, key = _config(api_url, api_key)
    return _check(requests.get(f"{url}/v1/jobs/{job_id}", headers=_headers(key), timeout=60)).json()


def _job_action(job_id, action, *, api_url=None, api_key=None):
    url, key = _config(api_url, api_key)
    return _check(requests.post(f"{url}/v1/jobs/{job_id}/{action}",
                                headers=_headers(key), timeout=60)).json()


def pause(job_id, **kwargs): return _job_action(job_id, "pause", **kwargs)
def resume(job_id, **kwargs): return _job_action(job_id, "resume", **kwargs)
def stop(job_id, **kwargs): return _job_action(job_id, "stop", **kwargs)


def wait(job_id, *, api_url=None, api_key=None, poll=5, on_update=None):
    while True:
        status = job_status(job_id, api_url=api_url, api_key=api_key)
        if on_update:
            on_update(status)
        if status.get("status") in ("done", "done_with_errors", "stopped", "failed"):
            return status
        time.sleep(poll)


def download_book(book, out_dir, *, api_url=None, api_key=None):
    url, key = _config(api_url, api_key)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = _check(requests.get(f"{url}/v1/files/{book}", headers=_headers(key), timeout=60)).json()
    for name in names:
        dest = out / Path(name).name
        if dest.exists():
            continue
        response = _check(requests.get(f"{url}/v1/files/{book}/{name}", headers=_headers(key), timeout=600))
        dest.write_bytes(response.content)
    return names


def register_voice(voice_id, ref_wav, ref_text="", *, api_url=None, api_key=None,
                   lang="hi", replace=False, gender=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(voice_id or "")):
        raise ValueError("Voice ID may contain only ASCII letters, digits, underscore and hyphen.")
    url, key = _config(api_url, api_key)
    audio = Path(ref_wav).read_bytes()
    payload = {"voice_id": voice_id, "ref_text": ref_text, "lang": lang, "mode": "icl_spk",
               "replace": bool(replace), "gender": gender,
               "ref_audio_b64": base64.b64encode(audio).decode("ascii")}
    return _check(requests.post(f"{url}/v1/voices", headers=_headers(key),
                                timeout=600, json=payload)).json()


def list_voices(*, api_url=None, api_key=None):
    url, key = _config(api_url, api_key)
    return _check(requests.get(f"{url}/v1/voices", headers=_headers(key), timeout=60)).json()


def delete_voice(voice_id, *, api_url=None, api_key=None):
    url, key = _config(api_url, api_key)
    return _check(requests.delete(f"{url}/v1/voices/{voice_id}", headers=_headers(key), timeout=60)).json()


def list_content_types(*, api_url=None, api_key=None):
    url, key = _config(api_url, api_key)
    return _check(requests.get(f"{url}/v1/content-types", headers=_headers(key), timeout=60)).json()
