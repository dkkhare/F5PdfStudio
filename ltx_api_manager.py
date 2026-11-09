"""SQLite-backed pool of up to five registered LTX API workers.

Each worker has its own URL and X-API-Key. LTX-2.3 and LTX-2.5 Pose
workflow IDs are registered independently on every worker.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "ltx_api_workers.db"
MAX_WORKERS = 5
HEALTH_FAILURE_LIMIT = 3
HEALTH_TIMEOUT = 15
POLL_INTERVAL = 5.0
_worker_db_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def init_worker_db() -> None:
    with _worker_db_lock, closing(_connect()) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS ltx_api_workers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                base_url TEXT NOT NULL UNIQUE,
                api_key TEXT NOT NULL,
                ltx23_workflow_id TEXT NOT NULL,
                ltx25_pose_workflow_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'FREE',
                current_job_id TEXT,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                last_health_check TEXT,
                last_used REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": str(api_key or "").strip()}


def _normalise_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    if not url:
        raise ValueError("Enter the LTX API URL.")
    if not url.startswith(("http://", "https://")):
        raise ValueError("LTX API URL must start with http:// or https://.")
    return url


def _load_workflow(path: str | Path) -> tuple[str, dict[str, Any]]:
    workflow_path = Path(str(path or "")).expanduser()
    if not workflow_path.is_file():
        raise FileNotFoundError(f"Workflow JSON file not found: {workflow_path}")
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError(f"Workflow JSON is empty or invalid: {workflow_path.name}")
    return workflow_path.name, workflow


def _register_one(base_url: str, api_key: str, path: str | Path) -> str:
    name, workflow = _load_workflow(path)
    response = requests.post(
        f"{base_url}/workflows/register",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json={"name": name, "workflow": workflow},
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Workflow registration failed for {name} (HTTP {response.status_code}): "
            f"{response.text[:2000]}"
        )
    body = response.json()
    workflow_id = str(body.get("workflow_id") or "").strip()
    if not body.get("success", True) or not workflow_id:
        raise RuntimeError(f"Worker did not return a valid workflow_id for {name}: {body}")
    return workflow_id


def register_worker(
    base_url: str,
    api_key: str,
    ltx23_workflow_path: str | Path,
    ltx25_pose_workflow_path: str | Path,
) -> str:
    """Health-check a server, register both workflows, then persist its credentials."""
    init_worker_db()
    base = _normalise_url(base_url)
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("Enter the API key for this LTX API worker.")

    health = requests.get(f"{base}/health", headers=_headers(key), timeout=HEALTH_TIMEOUT)
    health.raise_for_status()

    ltx23_id = _register_one(base, key, ltx23_workflow_path)
    pose_id = _register_one(base, key, ltx25_pose_workflow_path)

    with _worker_db_lock:
        db = _connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT id FROM ltx_api_workers WHERE base_url = ?", (base,)
            ).fetchone()
            count = db.execute("SELECT COUNT(*) AS n FROM ltx_api_workers").fetchone()["n"]
            if not existing and count >= MAX_WORKERS:
                db.rollback()
                raise ValueError(f"All {MAX_WORKERS} LTX API worker slots are already registered.")
            db.execute(
                """
                INSERT INTO ltx_api_workers
                    (base_url, api_key, ltx23_workflow_id, ltx25_pose_workflow_id,
                     status, current_job_id, consecutive_failures, last_health_check)
                VALUES (?, ?, ?, ?, 'FREE', NULL, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(base_url) DO UPDATE SET
                    api_key=excluded.api_key,
                    ltx23_workflow_id=excluded.ltx23_workflow_id,
                    ltx25_pose_workflow_id=excluded.ltx25_pose_workflow_id,
                    status=CASE WHEN ltx_api_workers.status='BUSY' THEN 'BUSY' ELSE 'FREE' END,
                    consecutive_failures=0, last_health_check=CURRENT_TIMESTAMP
                """,
                (base, key, ltx23_id, pose_id),
            )
            db.commit()
        finally:
            db.close()

    return (
        f"Registered LTX API worker: {base}\n"
        f"LTX-2.3 workflow ID: {ltx23_id}\n"
        f"LTX-2.5 Pose workflow ID: {pose_id}"
    )


def list_workers() -> list[dict[str, Any]]:
    init_worker_db()
    with _worker_db_lock, closing(_connect()) as db:
        rows = db.execute(
            """
            SELECT id, base_url, status, current_job_id, consecutive_failures,
                   last_health_check, ltx23_workflow_id, ltx25_pose_workflow_id
            FROM ltx_api_workers ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def workers_markdown() -> str:
    workers = list_workers()
    if not workers:
        return "No LTX API workers registered yet."
    lines = [
        "| ID | LTX API URL | Status | Current job | Failed checks | LTX-2.3 workflow ID | LTX-2.5 Pose workflow ID |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for worker in workers:
        url = worker["base_url"].replace("|", "%7C")
        job = worker.get("current_job_id") or "—"
        lines.append(
            f"| {worker['id']} | {url} | {worker['status']} | {job} | "
            f"{worker['consecutive_failures']} |"
        )
    lines.append("")
    lines.append("Workflow IDs are stored per worker for LTX-2.3 and LTX-2.5 Pose.")
    return "\n".join(lines)


def remove_worker(worker_id: int) -> str:
    init_worker_db()
    with _worker_db_lock, closing(_connect()) as db:
        cursor = db.execute("DELETE FROM ltx_api_workers WHERE id = ?", (int(worker_id),))
        if cursor.rowcount:
            return f"Removed LTX API worker {worker_id} from the database."
    return f"No registered LTX API worker found with ID {worker_id}."


def check_workers() -> str:
    """Check every saved worker; delete a row after three consecutive failures."""
    init_worker_db()
    workers = list_workers()
    messages: list[str] = []
    for worker in workers:
        worker_id = int(worker["id"])
        with _worker_db_lock, closing(_connect()) as db:
            row = db.execute(
                "SELECT base_url, api_key FROM ltx_api_workers WHERE id = ?", (worker_id,)
            ).fetchone()
        if not row:
            continue
        try:
            response = requests.get(
                f"{row['base_url']}/health",
                headers=_headers(row["api_key"]),
                timeout=HEALTH_TIMEOUT,
            )
            response.raise_for_status()
            with _worker_db_lock, closing(_connect()) as db:
                db.execute(
                    """
                    UPDATE ltx_api_workers
                    SET consecutive_failures=0,
                        status=CASE WHEN status='BUSY' THEN 'BUSY' ELSE 'FREE' END,
                        last_health_check=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (worker_id,),
                )
            messages.append(f"Worker {worker_id}: ONLINE")
        except Exception as exc:
            with _worker_db_lock:
                db = _connect()
                try:
                    db.execute("BEGIN IMMEDIATE")
                    current = db.execute(
                        "SELECT consecutive_failures FROM ltx_api_workers WHERE id=?",
                        (worker_id,),
                    ).fetchone()
                    if not current:
                        db.rollback()
                        continue
                    failures = int(current["consecutive_failures"] or 0) + 1
                    if failures >= HEALTH_FAILURE_LIMIT:
                        db.execute("DELETE FROM ltx_api_workers WHERE id=?", (worker_id,))
                        messages.append(
                            f"Worker {worker_id}: removed after {failures} failed health checks ({type(exc).__name__})."
                        )
                    else:
                        db.execute(
                            """
                            UPDATE ltx_api_workers SET status='OFFLINE',
                                consecutive_failures=?, last_health_check=CURRENT_TIMESTAMP
                            WHERE id=?
                            """,
                            (failures, worker_id),
                        )
                        messages.append(
                            f"Worker {worker_id}: OFFLINE ({failures}/{HEALTH_FAILURE_LIMIT} failures)."
                        )
                    db.commit()
                finally:
                    db.close()
    return "\n".join(messages) if messages else "No LTX API workers are registered."


def _reserve_worker(workflow_kind: str) -> dict[str, Any] | None:
    workflow_column = (
        "ltx25_pose_workflow_id" if workflow_kind == "pose" else "ltx23_workflow_id"
    )
    init_worker_db()
    with _worker_db_lock:
        db = _connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                f"""
                SELECT * FROM ltx_api_workers
                WHERE status='FREE' AND {workflow_column} IS NOT NULL
                  AND {workflow_column} <> ''
                ORDER BY last_used ASC, id ASC LIMIT 1
                """
            ).fetchone()
            if not row:
                db.rollback()
                return None
            cursor = db.execute(
                """
                UPDATE ltx_api_workers SET status='BUSY', last_used=?
                WHERE id=? AND status='FREE'
                """,
                (time.time(), row["id"]),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None
            result = dict(row)
            result["workflow_id"] = result[workflow_column]
            db.commit()
            return result
        finally:
            db.close()


def _set_job(worker_id: int, job_id: str) -> None:
    with _worker_db_lock, closing(_connect()) as db:
        db.execute(
            "UPDATE ltx_api_workers SET current_job_id=? WHERE id=?",
            (str(job_id), int(worker_id)),
        )


def _release_worker(worker_id: int) -> None:
    with _worker_db_lock, closing(_connect()) as db:
        db.execute(
            """
            UPDATE ltx_api_workers SET status='FREE', current_job_id=NULL
            WHERE id=? AND status='BUSY'
            """,
            (int(worker_id),),
        )


def _wait_for_worker(workflow_kind: str, wait_seconds: int, on_status=None) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, int(wait_seconds))
    check_workers()
    last_check = time.monotonic()
    while True:
        worker = _reserve_worker(workflow_kind)
        if worker:
            return worker
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("No registered LTX API worker became free before the wait timeout.")
        if now - last_check >= 30:
            check_workers()
            last_check = now
        if on_status:
            on_status("All registered LTX API workers are busy or offline. Waiting for a free worker...")
        time.sleep(min(5.0, max(0.5, deadline - now)))


def _file_binding(node_id: str, input_name: str) -> dict[str, str]:
    return {"node_id": str(node_id), "input_name": str(input_name), "upload_type": "input"}


def generate_ltx_api_video(
    *,
    workflow_kind: str,
    prompt: str,
    output_path: str | Path,
    audio_path: str | Path,
    image_path: str | Path | None = None,
    reference_video_path: str | Path | None = None,
    duration: float = 5,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
    seed: int = -1,
    extra_positive: str = "",
    negative_prompt: str = "",
    cfg: float = 1.0,
    timeout: int = 3600,
    poll_interval: float = POLL_INTERVAL,
    on_status=None,
) -> str:
    """Reserve one free worker, submit one job, poll it, download, then release."""
    kind = "pose" if workflow_kind == "pose" else "image"
    audio_file = Path(str(audio_path or "")).expanduser()
    image_file = Path(str(image_path)).expanduser() if image_path else None
    video_file = Path(str(reference_video_path)).expanduser() if reference_video_path else None
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if image_file is not None and not image_file.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_file}")
    if kind == "pose" and video_file is not None and not video_file.is_file():
        raise FileNotFoundError(f"Reference video not found: {video_file}")

    worker = _wait_for_worker(kind, wait_seconds=min(timeout, 1800), on_status=on_status)
    worker_id = int(worker["id"])
    base = worker["base_url"]
    key = worker["api_key"]
    job_id = ""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if kind == "pose":
        bindings = {"audio": _file_binding("506", "audio")}
        if video_file is not None:
            bindings["video"] = {
                "node_id": "503", "input_name": "video", "upload_type": "input"
            }
        if image_file is not None:
            bindings["image"] = {
                "node_id": "453", "input_name": "image", "upload_type": "image"
            }
    else:
        bindings = {
            "audio": _file_binding("276", "audio"),
        }
        if image_file is not None:
            bindings["image"] = {
                "node_id": "269", "input_name": "image", "upload_type": "image"
            }

    # StoryStudio worker API expects node-ID keyed input mappings.
    inputs = {
        "378": {"value": str(max(1.0, min(60.0, float(duration))))},
        "379": {"value": int(max(1, min(60, int(fps))))},
        "459": {"value": str(prompt or "").strip()},
    }
    fields = {
        "workflow_id": worker["workflow_id"],
        "inputs": json.dumps(inputs, ensure_ascii=False),
        "file_bindings": json.dumps(bindings, ensure_ascii=False),
        "timeout_seconds": str(int(timeout)),
    }

    try:
        if on_status:
            on_status(f"Submitting job to LTX API worker {worker_id}...")
        with ExitStack() as stack:
            files: dict[str, tuple[str, Any, str]] = {
                "audio": (audio_file.name, stack.enter_context(audio_file.open("rb")), "application/octet-stream")
            }
            if image_file is not None:
                files["image"] = (image_file.name, stack.enter_context(image_file.open("rb")), "application/octet-stream")
            if kind == "pose" and video_file is not None:
                files["video"] = (video_file.name, stack.enter_context(video_file.open("rb")), "application/octet-stream")
            response = requests.post(
                f"{base}/jobs",
                headers=_headers(key),
                data=fields,
                files=files,
                timeout=(30, 900),
            )
        if not response.ok:
            raise RuntimeError(f"LTX API worker {worker_id} submit HTTP {response.status_code}: {response.text[:4000]}")
        submitted = response.json()
        job_id = str(submitted.get("job_id") or "").strip()
        if not submitted.get("success", True) or not job_id:
            raise RuntimeError(f"LTX API worker {worker_id} did not return a valid job_id: {submitted}")
        _set_job(worker_id, job_id)
        if on_status:
            on_status(f"Worker {worker_id} accepted job {job_id[:10]}.")

        deadline = time.monotonic() + max(60, int(timeout))
        started = time.monotonic()
        last_status = "queued"
        while time.monotonic() < deadline:
            time.sleep(max(1.0, float(poll_interval)))
            status_response = requests.get(
                f"{base}/jobs/{job_id}", headers=_headers(key), timeout=45
            )
            status_response.raise_for_status()
            status_payload = status_response.json()
            status_data = status_payload.get("job") or status_payload
            last_status = str(status_data.get("status") or "unknown").lower()
            if on_status:
                on_status(
                    f"Worker {worker_id}: {last_status} "
                    f"({int(time.monotonic() - started)}s elapsed)"
                )
            if last_status in {"failed", "error", "cancelled"}:
                raise RuntimeError(
                    f"LTX API job {job_id} failed: "
                    f"{status_data.get('error') or status_data.get('detail') or status_data}"
                )
            if last_status in {"completed", "complete", "succeeded", "success"}:
                outputs_response = requests.get(
                    f"{base}/jobs/{job_id}/outputs",
                    headers=_headers(key),
                    timeout=60,
                )
                outputs_response.raise_for_status()
                outputs_payload = outputs_response.json()
                outputs = outputs_payload.get("outputs") or []
                if isinstance(outputs, dict):
                    outputs = [
                        {"node_id": node_id, **(value if isinstance(value, dict) else {"value": value})}
                        for node_id, value in outputs.items()
                    ]
                selected = next(
                    (item for item in outputs if str(item.get("node_id") or item.get("node") or "") == "400"),
                    None,
                )
                if selected is None and outputs:
                    selected = outputs[0]
                download_url = (
                    selected.get("download_url") if isinstance(selected, dict) else None
                )
                if not download_url:
                    download_url = f"{base}/jobs/{job_id}/download"
                download_response = requests.get(
                    urljoin(f"{base}/", str(download_url)),
                    headers=_headers(key),
                    timeout=300,
                )
                download_response.raise_for_status()
                if len(download_response.content) < 1000:
                    raise RuntimeError("LTX API returned an empty or incomplete video.")
                destination.write_bytes(download_response.content)
                if destination.stat().st_size < 1000:
                    raise RuntimeError("Downloaded LTX video is empty or incomplete.")
                return str(destination.resolve())
        raise TimeoutError(
            f"Timed out waiting for LTX API job {job_id}; last status: {last_status}. "
            "The remote job may still be running."
        )
    finally:
        _release_worker(worker_id)
