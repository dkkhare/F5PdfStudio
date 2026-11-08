"""Windows client for asynchronous LTX ComfyUI generation via the Colab bridge.

Supports two workflows (auto-selected):
  * LTX-2.3 image + audio            -> ltx23_workflow_api.json
  * LTX-2.5 audio + reference video  -> ltx25_pose_workflow_api.json
    (reference video drives head/hand/body motion; its first frame is the start image)
"""
from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import requests

DEFAULT_LTX23_BRIDGE_URL = os.getenv("F5PDFSTUDIO_LTX23_URL", "").strip().rstrip("/")
DEFAULT_LTX23_API_KEY = os.getenv("F5PDFSTUDIO_LTX23_API_KEY", "").strip()
DEFAULT_WORKFLOW_PATH = Path(__file__).with_name("ltx23_workflow_api.json")
DEFAULT_POSE_WORKFLOW_PATH = Path(__file__).with_name("ltx25_pose_workflow_api.json")


def _headers(api_key: str = "") -> dict[str, str]:
    key = str(api_key or DEFAULT_LTX23_API_KEY).strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def workflow_kind(workflow: dict[str, Any]) -> str:
    """Return 'ltx25_pose', 'ltx23', or '' if the API workflow is not recognised."""
    if isinstance(workflow, dict):
        if all(k in workflow for k in ("503", "506", "453")):
            return "ltx25_pose"
        if all(k in workflow for k in ("269", "276", "341")):
            return "ltx23"
    return ""


def check_ltx23_bridge(base_url: str = "", api_key: str = "", timeout: int = 10) -> tuple[bool, str]:
    base = str(base_url or DEFAULT_LTX23_BRIDGE_URL).strip().rstrip("/")
    if not base:
        return False, "LTX bridge URL is empty. Start the Colab bridge and enter its public tunnel URL."
    try:
        response = requests.get(f"{base}/health", headers=_headers(api_key), timeout=timeout)
        response.raise_for_status()
        data = response.json()
        comfy = data.get("comfyui", {})
        api_version = data.get("api_version", 1)
        if not comfy.get("reachable", False):
            return False, f"Bridge responds but ComfyUI is not reachable: {comfy.get('detail', 'unknown error')}"
        modes = ", ".join(data.get("modes", [])) or "ltx23"
        note = "" if api_version >= 3 else " (old bridge: update it to enable the reference-video workflow)"
        return True, (f"LTX bridge ready (API v{api_version}; workflows: {modes}). "
                      f"ComfyUI reachable at {comfy.get('url', 'local')}. Async job polling enabled.{note}")
    except Exception as exc:
        return False, f"LTX bridge not available: {exc}"


def generate_ltx23_video(*, prompt: str, output_path: str | Path, image_path: str | Path | None = None,
                         audio_path: str | Path, base_url: str = "", api_key: str = "",
                         workflow_path: str | Path = DEFAULT_WORKFLOW_PATH, duration: float = 5,
                         fps: int = 24, width: int = 1280, height: int = 720, seed: int = -1,
                         timeout: int = 3600, poll_interval: float = 5.0,
                         reference_video_path: str | Path | None = None,
                         extra_positive: str = "", negative_prompt: str = "", cfg: float = 1.0,
                         on_status: Callable[[str], None] | None = None) -> str:
    """Queue job promptly, poll status endpoints, then download the finished MP4.

    Pass ``reference_video_path`` to use the LTX-2.5 pose-reference workflow (body/head/hand motion
    copied from that video). If it is omitted on a later run with the pose workflow, the bridge
    reuses the last reference video it received. ``image_path`` is used only by the LTX-2.3 workflow.

    ``extra_positive`` is appended to the positive prompt; ``negative_prompt`` is appended to the workflow's
    default negative prompt. Both workflows sample at CFG 1 (distilled model), where the negative prompt is
    ignored, so set ``cfg`` above 1 (for example 1.5 to 3) to make the negative prompt take effect.
    """
    def say(message: str) -> None:
        if on_status:
            try:
                on_status(message)
            except Exception:
                pass

    base = str(base_url or DEFAULT_LTX23_BRIDGE_URL).strip().rstrip("/")
    if not base:
        raise ValueError("Enter the public LTX Colab bridge URL.")
    audio_file = Path(str(audio_path or "")).expanduser()
    video_file = Path(str(reference_video_path)).expanduser() if reference_video_path else None
    image_file = Path(str(image_path)).expanduser() if image_path else None
    workflow_file = Path(workflow_path)
    if video_file is not None and workflow_file == DEFAULT_WORKFLOW_PATH:
        workflow_file = DEFAULT_POSE_WORKFLOW_PATH  # a reference video means the pose workflow
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if video_file is not None and not video_file.is_file():
        raise FileNotFoundError(f"Reference video not found: {video_file}")
    if not workflow_file.is_file():
        raise FileNotFoundError(f"LTX API workflow not found: {workflow_file}")
    workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    kind = workflow_kind(workflow)
    if not kind:
        raise ValueError("Workflow JSON is not a recognised LTX API workflow "
                         "(expected LTX-2.3 image+audio or LTX-2.5 pose-reference).")
    if kind == "ltx23":
        video_file = None  # not used by this workflow
        if image_file is not None and not image_file.is_file():
            raise FileNotFoundError(f"Reference image not found: {image_file}")
    elif image_file is not None and not image_file.is_file():
        raise FileNotFoundError(f"Start image not found: {image_file}")
    # pose workflow: an optional image replaces the reference video's first frame as the start frame/identity

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = {"prompt": str(prompt or "").strip(), "duration": str(max(1.0, min(60.0, float(duration)))),
              "fps": str(max(1, min(60, int(fps)))), "width": str(max(256, min(2048, int(width)))),
              "height": str(max(256, min(2048, int(height)))), "seed": str(int(seed)),
              "extra_positive": str(extra_positive or "").strip(), "negative_prompt": str(negative_prompt or "").strip(),
              "cfg": str(max(1.0, min(8.0, float(cfg or 1.0)))),
              "workflow_json": json.dumps(workflow, ensure_ascii=False)}

    say("Uploading inputs to the Colab bridge...")
    with ExitStack() as stack:
        files: dict[str, tuple[str, Any, str]] = {
            "audio": (audio_file.name, stack.enter_context(audio_file.open("rb")), "application/octet-stream")}
        if image_file is not None:
            files["image"] = (image_file.name, stack.enter_context(image_file.open("rb")), "application/octet-stream")
        if video_file is not None:
            files["video"] = (video_file.name, stack.enter_context(video_file.open("rb")), "application/octet-stream")
        response = requests.post(f"{base}/generate", headers=_headers(api_key), data=fields, files=files,
                                 timeout=(30, 900 if video_file is not None else 300))
    if not response.ok:
        try:
            detail: Any = response.json()
        except Exception:
            detail = response.text[:8000]
        raise RuntimeError(f"LTX bridge submit HTTP {response.status_code}: {detail}")
    try:
        submitted = response.json()
        job_id = str(submitted["job_id"])
    except Exception as exc:
        raise RuntimeError(f"Bridge did not return an async job ID (is the updated bridge running?): {response.text[:2000]}") from exc
    for warning in submitted.get("warnings") or []:
        say(f"Warning: {warning}")
    say(f"Queued job {job_id[:8]} ({submitted.get('mode', kind)}).")

    deadline = time.monotonic() + max(60, int(timeout))
    last_status = "running"
    started = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(max(1.0, float(poll_interval)))
        try:
            status_response = requests.get(f"{base}/jobs/{job_id}", headers=_headers(api_key), timeout=45)
            if status_response.status_code == 404:
                raise RuntimeError("Bridge no longer knows this job (the Colab bridge restarted). "
                                   "Check ComfyUI's output folder or submit again.")
            status_response.raise_for_status()
            data = status_response.json()
            last_status = str(data.get("status", "unknown"))
            say(f"{last_status.capitalize()}... [{int(time.monotonic() - started)}s]")
            if last_status == "failed":
                raise RuntimeError(f"LTX ComfyUI job failed: {data.get('error') or data.get('last_poll_error') or data}")
            if last_status == "completed":
                say("Downloading video...")
                video_response = requests.get(f"{base}/jobs/{job_id}/video", headers=_headers(api_key), timeout=300)
                video_response.raise_for_status()
                if len(video_response.content) < 1000:
                    raise RuntimeError("Bridge returned an empty/incomplete video file.")
                destination.write_bytes(video_response.content)
                if not destination.is_file() or destination.stat().st_size < 1000:
                    raise RuntimeError("Downloaded video is empty or incomplete.")
                return str(destination.resolve())
        except requests.RequestException as exc:
            # A temporary tunnel hiccup shouldn't resubmit the generation job.
            last_status = f"poll retry after {type(exc).__name__}: {exc}"
            continue
    raise TimeoutError(f"Timed out waiting for LTX job {job_id}; last status: {last_status}. The ComfyUI job may still be "
                       f"running; check /jobs/{job_id} before retrying to avoid duplicate generation.")


def generate_ltx25_pose_video(*, prompt: str, output_path: str | Path, audio_path: str | Path,
                              reference_video_path: str | Path | None = None, **kwargs: Any) -> str:
    """Convenience wrapper: LTX-2.5 audio + reference-video workflow."""
    return generate_ltx23_video(prompt=prompt, output_path=output_path, audio_path=audio_path,
                                reference_video_path=reference_video_path,
                                workflow_path=kwargs.pop("workflow_path", DEFAULT_POSE_WORKFLOW_PATH), **kwargs)
