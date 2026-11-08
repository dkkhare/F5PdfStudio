"""Async Colab-side bridge for F5PdfStudio -> ComfyUI (LTX-2.3 and LTX-2.5 pose-reference).

POST /generate queues the ComfyUI prompt and returns promptly with job_id.
GET /jobs/{job_id} reports running/completed/failed.
GET /jobs/{job_id}/video returns the finished video.

Two workflows are supported; the bridge detects which one it received by node IDs/classes:
  * "ltx23"      - image + audio -> video          (LoadImage 269, LoadAudio 276, SaveVideo 341)
  * "ltx25_pose" - audio + reference video -> video (LoadAudio 503, LoadVideo 506, SaveVideo 453)
                   The reference video drives head/hand/body motion (DWPose + IC-LoRA). Its first
                   frame is used as the start image, so no separate image is needed.

Audio is mandatory. Reference image (ltx23) and reference video (ltx25_pose) are optional on later
runs: the last uploaded one is cached in ComfyUI's input directory and reused.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8190"))
COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/content/ComfyUI")).expanduser().resolve()
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "").strip()
COMFYUI_INPUT_DIR = Path(os.getenv("COMFYUI_INPUT_DIR", str(COMFYUI_DIR / "input"))).expanduser().resolve()
COMFYUI_OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", str(COMFYUI_DIR / "output"))).expanduser().resolve()
MAX_WAIT_SECONDS = int(os.getenv("LTX23_MAX_WAIT_SECONDS", "3600"))
JOB_DIR = Path(os.getenv("LTX23_JOB_DIR", "/tmp/f5pdf_ltx23_jobs"))
JOB_DIR.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict[str, Any]] = {}
CACHED_IMAGE_NAME = ""
CACHED_VIDEO_NAME = ""

# Node layout of each supported workflow (IDs are from the exported API JSON).
WORKFLOWS: dict[str, dict[str, Any]] = {
    "ltx23": {
        "required": {"269": "LoadImage", "276": "LoadAudio", "341": "SaveVideo"},
        "save": "341",
        "negative": "340:314",          # CLIPTextEncode (negative) -> inputs["text"]
        "cfg": ("340:315", "cfg"),      # stage-1 CFGGuider; cfg 1 = negative prompt has no effect
    },
    "ltx25_pose": {
        "required": {"503": "LoadAudio", "506": "LoadVideo", "453": "SaveVideo"},
        "save": "453",
        "audio": "503", "video": "506", "duration": "505", "fps": "422",
        "prompt": "411", "seeds": ("433", "442"),
        "negative": "419",              # CLIPTextEncode (negative) -> inputs["text"]
        "cfg": ("435", "video_cfg"),    # stage-1 LTXVDualCFGGuider; cfg 1 = negative prompt has no effect
    },
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}

app = FastAPI(title="F5PdfStudio LTX Async ComfyUI Bridge", version="3.0.0")


def _auth(authorization: str | None) -> None:
    if BRIDGE_API_KEY:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, BRIDGE_API_KEY):
            raise HTTPException(status_code=401, detail="Invalid bridge API key")


def _safe_name(filename: str, suffix: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if not ext or len(ext) > 10 or any(ch not in ".abcdefghijklmnopqrstuvwxyz0123456789" for ch in ext):
        ext = suffix
    return f"f5pdf_{uuid.uuid4().hex}{ext}"


def _detect_mode(workflow: dict[str, Any]) -> str:
    for mode, spec in WORKFLOWS.items():
        if all(nid in workflow and workflow[nid].get("class_type") == cls for nid, cls in spec["required"].items()):
            return mode
    raise HTTPException(
        400,
        "Unrecognized workflow. Expected the LTX-2.3 image+audio workflow (nodes 269/276/341) or the "
        "LTX-2.5 pose-reference workflow (nodes 503/506/453).",
    )


def _probe_video(path: Path) -> tuple[float, float] | None:
    """Return (duration_seconds, fps) using ffprobe, or None if unavailable."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        data = json.loads(out)
        num, _, den = str(data["streams"][0]["r_frame_rate"]).partition("/")
        fps = float(num) / float(den or 1)
        return float(data["format"]["duration"]), fps
    except Exception:
        return None


def _find_video_outputs(history_entry: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for node_output in (history_entry.get("outputs") or {}).values():
        if not isinstance(node_output, dict):
            continue
        # SaveVideo reports its file under "images" (with animated=true) in current ComfyUI.
        for key in ("videos", "gifs", "files", "images"):
            items = node_output.get(key) or []
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if isinstance(item, dict) and Path(str(item.get("filename") or "")).suffix.lower() in VIDEO_SUFFIXES:
                    found.append({"filename": str(item["filename"]), "subfolder": str(item.get("subfolder") or ""),
                                  "type": str(item.get("type") or "output")})
    return found


def _history(prompt_id: str) -> dict[str, Any]:
    r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=20)
    r.raise_for_status()
    return r.json().get(prompt_id) or {}


def _finish_job(job_id: str, job: dict[str, Any], history_entry: dict[str, Any]) -> None:
    videos = _find_video_outputs(history_entry)
    if not videos:
        save_id = job.get("save_node", "")
        raise RuntimeError(f"ComfyUI finished but returned no video file. SaveVideo node output: "
                           f"{(history_entry.get('outputs') or {}).get(save_id, {})}")
    last_error = None
    for item in videos:
        try:
            r = requests.get(f"{COMFYUI_URL}/view", params={"filename": item["filename"], "subfolder": item["subfolder"],
                                                           "type": item["type"]}, timeout=300)
            r.raise_for_status()
            if len(r.content) < 1000:
                raise RuntimeError("ComfyUI returned a video smaller than 1 KB")
            out = JOB_DIR / f"{job_id}{Path(item['filename']).suffix or '.mp4'}"
            out.write_bytes(r.content)
            job.update(status="completed", video_path=str(out), video_filename=out.name, completed_at=time.time(), error="")
            return
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"Video was produced but could not be retrieved from ComfyUI: {last_error}")


@app.get("/health")
def health(authorization: str | None = Header(default=None)):
    _auth(authorization)
    reachable, detail = False, ""
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        reachable = r.ok
        if not r.ok:
            detail = f"ComfyUI HTTP {r.status_code}"
    except Exception as exc:
        detail = str(exc)
    return {
        "status": "ok" if reachable else "degraded",
        "backend": "Async ComfyUI LTX bridge",
        "comfyui": {"url": COMFYUI_URL, "reachable": reachable, "detail": detail},
        "workflow_support": "ltx23: image+audio (audio required, image optional/cached); "
                            "ltx25_pose: audio + reference video (audio required, video optional/cached)",
        "modes": list(WORKFLOWS),
        "ffprobe": bool(shutil.which("ffprobe")),
        "comfyui_dir_exists": COMFYUI_DIR.is_dir(),
        "input_dir_exists": COMFYUI_INPUT_DIR.is_dir(),
        "api_version": 3,
    }


def _pick_seed(seed: int) -> int:
    return secrets.randbelow(2**32 - 2) if int(seed) < 0 else int(seed)


def _compose_positive(base: str, extra: str) -> str:
    base, extra = str(base or "").strip(), str(extra or "").strip()
    if base and extra:
        return f"{base.rstrip()} {extra}"
    return base or extra


def _apply_negative_and_cfg(workflow: dict[str, Any], spec: dict[str, Any], extra_negative: str,
                            cfg: float, warnings: list[str]) -> None:
    """Append the optional extra negative prompt to the workflow's default one and optionally raise stage-1 CFG."""
    extra_negative = str(extra_negative or "").strip()
    neg_id = spec["negative"]
    if extra_negative and neg_id in workflow:
        node_inputs = workflow[neg_id].setdefault("inputs", {})
        base = str(node_inputs.get("text", "")).strip().rstrip(",").strip()
        node_inputs["text"] = f"{base}, {extra_negative}" if base else extra_negative
    cfg_id, cfg_field = spec["cfg"]
    cfg_value = max(1.0, min(8.0, float(cfg or 1.0)))
    if cfg_value > 1.0 and cfg_id in workflow:
        workflow[cfg_id].setdefault("inputs", {})[cfg_field] = round(cfg_value, 2)
        warnings.append(f"Stage-1 CFG set to {cfg_value:g}: negative prompt is active (slower, about 2x sampling cost).")
    elif extra_negative:
        warnings.append("Negative prompt has no effect while CFG is 1.0 (distilled model). "
                        "Raise 'Negative prompt strength (CFG)' above 1 to use it.")


@app.post("/generate")
async def generate(
    prompt: str = Form(""), duration: float = Form(5.0), fps: int = Form(24),
    width: int = Form(1280), height: int = Form(720), seed: int = Form(-1),
    extra_positive: str = Form(""), negative_prompt: str = Form(""), cfg: float = Form(1.0),
    workflow_json: str = Form(...), image: UploadFile | None = File(default=None),
    video: UploadFile | None = File(default=None),
    audio: UploadFile = File(...), authorization: str | None = Header(default=None),
):
    global CACHED_IMAGE_NAME, CACHED_VIDEO_NAME
    _auth(authorization)
    if not COMFYUI_INPUT_DIR.is_dir():
        raise HTTPException(500, f"ComfyUI input directory not found: {COMFYUI_INPUT_DIR}. Check COMFYUI_DIR.")
    try:
        workflow = json.loads(workflow_json)
    except Exception as exc:
        raise HTTPException(400, f"Invalid workflow_json: {exc}") from exc
    if not isinstance(workflow, dict):
        raise HTTPException(400, "workflow_json must be an API-format object")
    mode = _detect_mode(workflow)
    spec = WORKFLOWS[mode]
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=8)
        r.raise_for_status()
    except Exception as exc:
        raise HTTPException(502, f"Cannot reach ComfyUI at {COMFYUI_URL}: {exc}") from exc

    job_id = uuid.uuid4().hex
    created_files: list[Path] = []
    warnings: list[str] = []
    new_image = new_video = ""
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(400, "Uploaded audio file is empty")
        audio_name = _safe_name(audio.filename or "audio.wav", ".wav")
        audio_path = COMFYUI_INPUT_DIR / audio_name
        audio_path.write_bytes(audio_bytes)
        created_files.append(audio_path)
        dur = max(1.0, min(60.0, float(duration)))
        frame_rate = max(1, min(60, int(fps)))
        chosen_seed = _pick_seed(seed)

        if mode == "ltx23":
            workflow["276"].setdefault("inputs", {})["audio"] = audio_name
            if image is not None and image.filename:
                image_bytes = await image.read()
                if not image_bytes:
                    raise HTTPException(400, "Uploaded reference image is empty")
                image_name = _safe_name(image.filename or "reference.png", ".png")
                image_path = COMFYUI_INPUT_DIR / image_name
                image_path.write_bytes(image_bytes)
                created_files.append(image_path)
                new_image = image_name
            else:
                configured = str((workflow.get("269", {}).get("inputs") or {}).get("image") or "")
                chosen = CACHED_IMAGE_NAME if CACHED_IMAGE_NAME and (COMFYUI_INPUT_DIR / CACHED_IMAGE_NAME).is_file() else configured
                if not chosen or not (COMFYUI_INPUT_DIR / chosen).is_file():
                    raise HTTPException(400, "No reference image uploaded and no cached/default image exists. "
                                             "Upload an image once, then it can be omitted on later runs.")
                image_name = chosen
            workflow["269"].setdefault("inputs", {})["image"] = image_name

            if "340:319" in workflow:
                base_positive = str(prompt or "").strip() or str(workflow["340:319"].get("inputs", {}).get("value", ""))
                workflow["340:319"].setdefault("inputs", {})["value"] = _compose_positive(base_positive, extra_positive)
            if "340:331" in workflow:
                workflow["340:331"].setdefault("inputs", {})["value"] = dur
            if "340:323" in workflow:
                workflow["340:323"].setdefault("inputs", {})["value"] = frame_rate
            if "340:330" in workflow:
                workflow["340:330"].setdefault("inputs", {})["value"] = max(256, min(2048, int(width)))
            if "340:324" in workflow:
                workflow["340:324"].setdefault("inputs", {})["value"] = max(256, min(2048, int(height)))
            for node_id, node_seed in (("340:285", chosen_seed), ("340:286", chosen_seed + 1)):
                if node_id in workflow:
                    workflow[node_id].setdefault("inputs", {})["noise_seed"] = node_seed
            workflow["341"].setdefault("inputs", {})["filename_prefix"] = f"video/F5PdfStudio_LTX23_{job_id[:10]}"

        else:  # ltx25_pose
            workflow[spec["audio"]].setdefault("inputs", {})["audio"] = audio_name
            if video is not None and video.filename:
                video_bytes = await video.read()
                if not video_bytes:
                    raise HTTPException(400, "Uploaded reference video is empty")
                video_name = _safe_name(video.filename or "reference.mp4", ".mp4")
                video_path = COMFYUI_INPUT_DIR / video_name
                video_path.write_bytes(video_bytes)
                created_files.append(video_path)
                new_video = video_name
            else:
                configured = str((workflow.get(spec["video"], {}).get("inputs") or {}).get("file") or "")
                chosen = CACHED_VIDEO_NAME if CACHED_VIDEO_NAME and (COMFYUI_INPUT_DIR / CACHED_VIDEO_NAME).is_file() else configured
                if not chosen or not (COMFYUI_INPUT_DIR / chosen).is_file():
                    raise HTTPException(400, "No reference video uploaded and no cached/default one exists. "
                                             "Upload a reference video once, then it can be omitted on later runs.")
                video_name = chosen
            workflow[spec["video"]].setdefault("inputs", {})["file"] = video_name

            # Optional identity/start image: by default the workflow starts from the reference video's first
            # frame (node 510 -> 471). If an image is uploaded, node 471 is re-wired to the LoadImage node 452.
            if image is not None and image.filename:
                if "452" in workflow and "471" in workflow:
                    start_bytes = await image.read()
                    if not start_bytes:
                        raise HTTPException(400, "Uploaded start image is empty")
                    start_name = _safe_name(image.filename or "start.png", ".png")
                    start_path = COMFYUI_INPUT_DIR / start_name
                    start_path.write_bytes(start_bytes)
                    created_files.append(start_path)
                    workflow["452"].setdefault("inputs", {})["image"] = start_name
                    workflow["471"].setdefault("inputs", {})["input"] = ["452", 0]
                else:
                    warnings.append("This workflow file has no start-image node (452); using the reference video's first frame.")
            if workflow.get("471", {}).get("inputs", {}).get("input") != ["452", 0]:
                workflow.pop("452", None)  # start-image node unused: keep the prompt clean

            probe = _probe_video(COMFYUI_INPUT_DIR / video_name)
            if probe:
                ref_dur, ref_fps = probe
                if ref_dur + 0.25 < dur:
                    raise HTTPException(400, f"Reference video is {ref_dur:.1f}s but the requested duration is {dur:.1f}s. "
                                             "Use a longer reference video or a shorter duration.")
                if abs(ref_fps - frame_rate) > 0.5:
                    warnings.append(f"Reference video is {ref_fps:.2f} fps but the workflow runs at {frame_rate} fps; "
                                    "re-export the reference at the same fps for best pose timing.")
            else:
                warnings.append("ffprobe not available: reference length/fps were not checked.")

            workflow[spec["duration"]].setdefault("inputs", {})["value"] = dur
            workflow[spec["fps"]].setdefault("inputs", {})["value"] = frame_rate
            base_positive = str(prompt or "").strip() or str(workflow[spec["prompt"]].get("inputs", {}).get("text", ""))
            workflow[spec["prompt"]].setdefault("inputs", {})["text"] = _compose_positive(base_positive, extra_positive)
            for node_id, node_seed in zip(spec["seeds"], (chosen_seed, chosen_seed + 1)):
                workflow[node_id].setdefault("inputs", {})["noise_seed"] = node_seed
            workflow[spec["save"]].setdefault("inputs", {})["filename_prefix"] = f"video/F5PdfStudio_LTX25_{job_id[:10]}"

        _apply_negative_and_cfg(workflow, spec, negative_prompt, cfg, warnings)

        queued = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": f"f5pdfstudio-{job_id}"}, timeout=60)
        try:
            queue_data = queued.json()
        except Exception:
            queue_data = {"raw": queued.text[:4000]}
        if not queued.ok or not queue_data.get("prompt_id"):
            raise HTTPException(500, f"ComfyUI rejected workflow (HTTP {queued.status_code}): {queue_data}")
        # Remember the new reference only after ComfyUI accepted the job.
        if new_image:
            CACHED_IMAGE_NAME = new_image
        if new_video:
            CACHED_VIDEO_NAME = new_video
        JOBS[job_id] = {
            "job_id": job_id, "prompt_id": queue_data["prompt_id"], "status": "running", "mode": mode,
            "save_node": spec["save"], "created_at": time.time(), "deadline": time.time() + max(60, MAX_WAIT_SECONDS),
            "video_path": "", "error": "", "warnings": warnings, "input_files": [str(p) for p in created_files],
        }
        return {"job_id": job_id, "prompt_id": queue_data["prompt_id"], "status": "running", "mode": mode,
                "warnings": warnings, "status_url": f"/jobs/{job_id}", "video_url": f"/jobs/{job_id}/video",
                "message": "Queued in ComfyUI. Poll status_url; this request does not wait for video generation."}
    except HTTPException:
        for p in created_files:
            try: p.unlink(missing_ok=True)
            except OSError: pass
        raise
    except Exception as exc:
        for p in created_files:
            try: p.unlink(missing_ok=True)
            except OSError: pass
        raise HTTPException(500, f"LTX bridge error: {type(exc).__name__}: {exc}") from exc


@app.get("/jobs/{job_id}")
def job_status(job_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job ID. The Colab bridge may have restarted; resubmit the job.")
    if job["status"] == "running":
        if time.time() > job["deadline"]:
            job.update(status="failed", error=f"Timed out after {MAX_WAIT_SECONDS}s waiting for ComfyUI. Check its queue/log.")
        else:
            try:
                history_entry = _history(job["prompt_id"])
                if history_entry:
                    status = history_entry.get("status") or {}
                    status_str = str(status.get("status_str") or "").lower()
                    if status.get("completed") or status_str in {"success", "completed"}:
                        _finish_job(job_id, job, history_entry)
                    elif status_str in {"error", "failed"}:
                        job.update(status="failed", error=f"ComfyUI prompt failed: {status.get('messages') or history_entry}")
            except Exception as exc:
                job["last_poll_error"] = f"{type(exc).__name__}: {exc}"
    if job["status"] in {"completed", "failed"}:
        for raw in job.get("input_files", []):
            p = Path(raw)
            # Never delete the cached reference image/video.
            if p.name not in (CACHED_IMAGE_NAME, CACHED_VIDEO_NAME):
                try: p.unlink(missing_ok=True)
                except OSError: pass
        job["input_files"] = []
    return {"job_id": job_id, "prompt_id": job["prompt_id"], "mode": job.get("mode"), "status": job["status"],
            "error": job.get("error", ""), "last_poll_error": job.get("last_poll_error", ""),
            "warnings": job.get("warnings", []),
            "video_url": f"/jobs/{job_id}/video" if job["status"] == "completed" else None,
            "elapsed_seconds": round(time.time() - job["created_at"], 1)}


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job ID")
    if job["status"] != "completed" or not job.get("video_path"):
        raise HTTPException(409, f"Video is not ready. Current status: {job['status']}")
    path = Path(job["video_path"])
    if not path.is_file():
        raise HTTPException(404, "Completed video file is missing from bridge storage")
    return FileResponse(str(path), media_type="video/mp4", filename="ltx_output" + path.suffix)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT)
