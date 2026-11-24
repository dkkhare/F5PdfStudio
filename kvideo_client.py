"""ComfyUI API client for the ID_LoRA / LTX image+audio -> video workflow."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".gif")


def _session(api_key: str = "") -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    if str(api_key or "").strip():
        session.headers["Authorization"] = f"Bearer {str(api_key).strip()}"
    return session


def _find(prompt: dict, class_type: str, title=None, startswith=None):
    hits = []
    for node in prompt.values():
        if node.get("class_type") != class_type:
            continue
        node_title = (node.get("_meta") or {}).get("title", "")
        if title is not None and node_title != title:
            continue
        if startswith is not None and not node_title.startswith(startswith):
            continue
        hits.append(node)
    return hits


def _one(prompt: dict, class_type: str, **kwargs):
    hits = _find(prompt, class_type, **kwargs)
    if len(hits) != 1:
        raise RuntimeError(
            f"Workflow has {len(hits)} node(s) of type {class_type} {kwargs or ''}; "
            "expected exactly one. Export the correct workflow using ComfyUI > Workflow > Export (API)."
        )
    return hits[0]


def check_server(server: str, api_key: str = "", timeout: int = 15) -> str:
    base = str(server or "").strip().rstrip("/")
    if not base:
        raise ValueError("Enter the kVideo API URL.")
    session = _session(api_key)
    errors = []
    for endpoint in ("/system_stats", "/queue"):
        try:
            response = session.get(base + endpoint, timeout=timeout)
            if response.status_code in (401, 403):
                raise PermissionError(f"Authentication rejected at {endpoint} (HTTP {response.status_code}). Check the API key.")
            response.raise_for_status()
            return f"ComfyUI server reachable: {base}{endpoint} (HTTP {response.status_code})"
        except PermissionError:
            raise
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("Could not reach the kVideo ComfyUI server. " + " | ".join(errors))


def _upload(server: str, path: str | os.PathLike, session: requests.Session) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    unique_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_{source.name}"
    with source.open("rb") as handle:
        response = session.post(
            f"{server}/upload/image",
            files={"image": (unique_name, handle)},
            data={"overwrite": "true", "type": "input"},
            timeout=600,
        )
    response.raise_for_status()
    result = response.json()
    if not result.get("name"):
        raise RuntimeError(f"ComfyUI upload returned no filename: {result}")
    return (result.get("subfolder", "") + "/" if result.get("subfolder") else "") + result["name"]


def generate(
    server: str,
    workflow_api: str | os.PathLike | dict,
    image: str | os.PathLike,
    audio: str | os.PathLike,
    out: str | os.PathLike = "result.mp4",
    extra_seconds: float | None = 2.0,
    max_seconds: float | None = 9.6,
    prompt: str | None = None,
    negative: str | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    timeout: int = 3600,
    poll: float = 3.0,
    log=print,
    api_key: str = "",
) -> str:
    base = str(server or "").strip().rstrip("/")
    if not base:
        raise ValueError("Enter the kVideo API URL.")
    if isinstance(workflow_api, (str, os.PathLike)):
        with open(workflow_api, "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    else:
        workflow = json.loads(json.dumps(workflow_api))
    if "nodes" in workflow and "links" in workflow:
        raise ValueError("This is a normal ComfyUI workflow. Upload an API-format JSON exported with Workflow > Export (API).")

    session = _session(api_key)
    log("Uploading image and audio to kVideo…")
    image_name = _upload(base, image, session)
    audio_name = _upload(base, audio, session)
    _one(workflow, "LoadImage")["inputs"]["image"] = image_name
    _one(workflow, "LoadAudio")["inputs"]["audio"] = audio_name

    pad_inputs = _one(workflow, "AudioPadToVideoLength")["inputs"]
    if extra_seconds is not None:
        pad_inputs["extra_seconds"] = float(extra_seconds)
    if max_seconds is not None:
        pad_inputs["max_seconds"] = float(max_seconds)
    if prompt is not None or negative is not None:
        conditioning = _one(workflow, "LTXVConditioning")["inputs"]
        if prompt is not None:
            workflow[str(conditioning["positive"][0])]["inputs"]["text"] = str(prompt)
        if negative is not None:
            workflow[str(conditioning["negative"][0])]["inputs"]["text"] = str(negative)
    if seed is not None:
        for node in _find(workflow, "RandomNoise"):
            node["inputs"]["noise_seed"] = int(seed)
    if width is not None:
        _one(workflow, "PrimitiveInt", startswith="Width")["inputs"]["value"] = int(width)
    if height is not None:
        _one(workflow, "PrimitiveInt", startswith="Height")["inputs"]["value"] = int(height)

    audio_stem = Path(audio).stem
    _one(workflow, "SaveVideo")["inputs"]["filename_prefix"] = f"video/{audio_stem}_{int(time.time())}"
    response = session.post(
        f"{base}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"kVideo rejected the workflow (HTTP {response.status_code}): {response.text[:2000]}")
    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"kVideo returned no prompt_id: {response.text[:1000]}")
    log(f"Queued kVideo job {prompt_id}; waiting for render…")

    started = time.monotonic()
    last_log = 0.0
    while True:
        history_response = session.get(f"{base}/history/{prompt_id}", timeout=60)
        history_response.raise_for_status()
        history = history_response.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                message = next(
                    (item[1].get("exception_message") for item in status.get("messages", [])
                     if isinstance(item, (list, tuple)) and len(item) > 1 and item[0] == "execution_error"),
                    "unknown ComfyUI execution error",
                )
                raise RuntimeError(f"kVideo workflow failed: {message}")
            if entry.get("outputs"):
                break
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"Timed out waiting for kVideo job {prompt_id}.")
        if time.monotonic() - last_log > 20:
            log(f"kVideo is rendering… {int(time.monotonic() - started)} seconds elapsed")
            last_log = time.monotonic()
        time.sleep(max(0.5, float(poll)))

    output_files = [
        item
        for output in entry["outputs"].values()
        for value in output.values()
        if isinstance(value, list)
        for item in value
        if isinstance(item, dict) and item.get("filename")
    ]
    videos = [item for item in output_files if item["filename"].lower().endswith(VIDEO_EXTENSIONS)]
    if not videos:
        raise RuntimeError("kVideo workflow finished but returned no video file.")
    selected = videos[-1]
    download = session.get(
        f"{base}/view",
        params={
            "filename": selected["filename"],
            "subfolder": selected.get("subfolder", ""),
            "type": selected.get("type", "output"),
        },
        timeout=600,
        stream=True,
    )
    download.raise_for_status()
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("wb") as handle:
            for chunk in download.iter_content(1 << 20):
                if chunk:
                    handle.write(chunk)
        if not temporary.is_file() or temporary.stat().st_size < 1000:
            raise RuntimeError("kVideo returned an empty or incomplete video download.")
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    log(f"kVideo finished in {int(time.monotonic() - started)} seconds: {destination}")
    return str(destination)
