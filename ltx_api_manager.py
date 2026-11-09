"""SQLite-backed pool of up to five workers with dynamically discovered workflows.

Each worker has its own URL and X-API-Key. Workflow IDs and workflow metadata
are stored per worker without assuming model names or workflow filenames.
"""
from __future__ import annotations

import json
import os
import secrets
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
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def init_worker_db() -> None:
    with _worker_db_lock, closing(_connect()) as db:
        existing = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ltx_api_workers'"
        ).fetchone()
        if existing:
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(ltx_api_workers)").fetchall()
            }
            if "ltx23_workflow_id" in columns or "ltx25_pose_workflow_id" in columns:
                # Migrate the earlier two-workflow schema while preserving worker credentials/status.
                db.execute("DROP TABLE IF EXISTS ltx_api_worker_workflows")
                db.execute("ALTER TABLE ltx_api_workers RENAME TO ltx_api_workers_legacy")
                existing = None

        if not existing:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS ltx_api_workers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_url TEXT NOT NULL UNIQUE,
                    api_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'FREE',
                    current_job_id TEXT,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_health_check TEXT,
                    last_used REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            legacy = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ltx_api_workers_legacy'"
            ).fetchone()
            if legacy:
                db.execute(
                    """
                    INSERT INTO ltx_api_workers
                        (id, base_url, api_key, status, current_job_id,
                         consecutive_failures, last_health_check, last_used, created_at)
                    SELECT id, base_url, api_key, status, current_job_id,
                           consecutive_failures, last_health_check, last_used, created_at
                    FROM ltx_api_workers_legacy
                    """
                )
                db.execute("DROP TABLE ltx_api_workers_legacy")

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS ltx_api_worker_workflows (
                worker_id INTEGER NOT NULL,
                workflow_name TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                workflow_path TEXT NOT NULL,
                workflow_hash TEXT NOT NULL,
                PRIMARY KEY (worker_id, workflow_name),
                FOREIGN KEY(worker_id) REFERENCES ltx_api_workers(id) ON DELETE CASCADE
            )
            """
        )

def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": str(api_key or "").strip()}


def _assert_healthy(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    try:
        body = response.json()
    except Exception as exc:
        raise RuntimeError("Worker /health did not return valid JSON.") from exc
    status = str(body.get("status") or "").lower()
    if status in {"degraded", "offline", "error", "unhealthy", "failed"}:
        raise RuntimeError(f"Worker health status is {status}: {body}")
    comfy = body.get("comfyui")
    if isinstance(comfy, dict) and comfy.get("reachable") is False:
        raise RuntimeError(f"Worker is online but ComfyUI is unreachable: {comfy}")
    return body


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


def _workflow_candidates(directory: str | Path | None = None) -> list[Path]:
    """Find every valid ComfyUI API prompt-graph JSON without assuming filenames."""
    root = Path(directory or os.getenv("LTX_WORKFLOW_DIR") or ROOT).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Workflow directory does not exist: {root}")
    excluded = {".git", ".venv", "venv", "__pycache__", "data", "outputs", "input", "temp", "models"}
    found: list[Path] = []
    for path in root.rglob("*.json"):
        try:
            relative = path.relative_to(root)
            if any(part.startswith(".") or part.lower() in excluded for part in relative.parts):
                continue
            _, graph = _load_workflow(path)
            if graph and all(
                isinstance(node, dict) and isinstance(node.get("class_type"), str)
                and isinstance(node.get("inputs"), dict)
                for node in graph.values()
            ):
                found.append(path)
        except Exception:
            continue
    return sorted(found, key=lambda item: (item.name.lower(), str(item).lower()))


def discover_workflows(directory: str | Path | None = None) -> list[dict[str, str]]:
    """Return discovered workflow choices, using paths only internally."""
    result = []
    for path in _workflow_candidates(directory):
        root = Path(os.getenv("LTX_WORKFLOW_DIR") or ROOT).expanduser().resolve()
        result.append({"name": str(path.resolve().relative_to(root)), "path": str(path)})
    return result


def list_workflows() -> list[str]:
    return [item["name"] for item in discover_workflows()]


def _register_one(base_url: str, api_key: str, path: str | Path) -> tuple[str, str, str, str]:
    _, workflow = _load_workflow(path)
    workflow_root = Path(os.getenv("LTX_WORKFLOW_DIR") or ROOT).expanduser().resolve()
    name = str(Path(path).resolve().relative_to(workflow_root))
    raw = Path(path).read_bytes()
    import hashlib
    fingerprint = hashlib.sha256(raw).hexdigest()
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
    return name, workflow_id, str(Path(path).resolve()), fingerprint


def register_worker(base_url: str, api_key: str, *unused_workflow_paths: str | Path) -> str:
    """Register every valid workflow JSON discovered in the workflow directory."""
    init_worker_db()
    base = _normalise_url(base_url)
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("Enter the API key for this LTX API worker.")
    candidates = _workflow_candidates()
    if not candidates:
        folder = os.getenv("LTX_WORKFLOW_DIR") or str(ROOT)
        raise FileNotFoundError(
            f"No valid ComfyUI API workflow JSON files were found under {folder}. "
            "Set LTX_WORKFLOW_DIR to the folder containing your workflow JSON files."
        )

    with _worker_db_lock, closing(_connect()) as db:
        existing = db.execute(
            "SELECT id FROM ltx_api_workers WHERE base_url = ?", (base,)
        ).fetchone()
        count = db.execute("SELECT COUNT(*) AS n FROM ltx_api_workers").fetchone()["n"]
    if not existing and count >= MAX_WORKERS:
        raise ValueError(f"All {MAX_WORKERS} LTX API worker slots are already registered.")

    health = requests.get(f"{base}/health", headers=_headers(key), timeout=HEALTH_TIMEOUT)
    _assert_healthy(health)

    registered = [_register_one(base, key, path) for path in candidates]

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
                    (base_url, api_key, status, current_job_id,
                     consecutive_failures, last_health_check)
                VALUES (?, ?, 'FREE', NULL, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(base_url) DO UPDATE SET
                    api_key=excluded.api_key,
                    status=CASE WHEN ltx_api_workers.status='BUSY' THEN 'BUSY' ELSE 'FREE' END,
                    consecutive_failures=0, last_health_check=CURRENT_TIMESTAMP
                """,
                (base, key),
            )
            worker = db.execute(
                "SELECT id FROM ltx_api_workers WHERE base_url=?", (base,)
            ).fetchone()
            worker_id = int(worker["id"])
            db.execute("DELETE FROM ltx_api_worker_workflows WHERE worker_id=?", (worker_id,))
            db.executemany(
                """
                INSERT INTO ltx_api_worker_workflows
                    (worker_id, workflow_name, workflow_id, workflow_path, workflow_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(worker_id, name, workflow_id, path, fingerprint)
                 for name, workflow_id, path, fingerprint in registered],
            )
            db.commit()
        finally:
            db.close()

    names = "\n".join(f"  • {name} → {workflow_id}" for name, workflow_id, _, _ in registered)
    return f"Registered LTX API worker: {base}\nDiscovered and registered {len(registered)} workflow(s):\n{names}"

def list_workers() -> list[dict[str, Any]]:
    init_worker_db()
    with _worker_db_lock, closing(_connect()) as db:
        rows = db.execute(
            """
            SELECT w.id, w.base_url, w.status, w.current_job_id, w.consecutive_failures,
                   w.last_health_check,
                   COALESCE((SELECT group_concat(ww.workflow_name, ', ')
                             FROM ltx_api_worker_workflows ww WHERE ww.worker_id=w.id), '') AS workflows
            FROM ltx_api_workers w ORDER BY w.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def workers_markdown() -> str:
    workers = list_workers()
    if not workers:
        return "No LTX API workers registered yet."
    lines = [
        "| ID | LTX API URL | Status | Current job | Failed checks | Registered workflows |",
        "|---:|---|---|---|---:|---|",
    ]
    for worker in workers:
        url = worker["base_url"].replace("|", "%7C")
        job = worker.get("current_job_id") or "—"
        names = (worker.get("workflows") or "None").replace("|", "%7C")
        lines.append(
            f"| LTX API {worker['id']} | {url} | {worker['status']} | {job} | "
            f"{worker['consecutive_failures']} | {names} |"
        )
    lines.append("")
    lines.append("Workflow IDs are maintained separately for every discovered JSON workflow on each worker.")
    return "\n".join(lines)

def remove_worker(worker_id: int) -> str:
    init_worker_db()
    with _worker_db_lock, closing(_connect()) as db:
        db.execute("DELETE FROM ltx_api_worker_workflows WHERE worker_id = ?", (int(worker_id),))
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
            _assert_healthy(response)
            with _worker_db_lock, closing(_connect()) as db:
                db.execute(
                    """
                    UPDATE ltx_api_workers
                    SET consecutive_failures=0,
                        status=CASE WHEN current_job_id IS NOT NULL THEN 'BUSY' ELSE 'FREE' END,
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
                        db.execute("DELETE FROM ltx_api_worker_workflows WHERE worker_id=?", (worker_id,))
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


def _reserve_worker(workflow_name: str) -> dict[str, Any] | None:
    init_worker_db()
    with _worker_db_lock:
        db = _connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT w.*, ww.workflow_id, ww.workflow_path, ww.workflow_hash
                FROM ltx_api_workers AS w
                JOIN ltx_api_worker_workflows AS ww ON ww.worker_id=w.id
                WHERE w.status='FREE' AND ww.workflow_name=?
                ORDER BY w.last_used ASC, w.id ASC LIMIT 1
                """,
                (str(workflow_name),),
            ).fetchone()
            if not row:
                db.rollback()
                return None
            cursor = db.execute(
                "UPDATE ltx_api_workers SET status='BUSY', current_job_id=NULL, last_used=? WHERE id=? AND status='FREE'",
                (time.time(), row["id"]),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return None
            result = dict(row)
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
    if not list_workers():
        raise RuntimeError("No LTX API workers are registered. Register at least one worker first.")
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
            if not list_workers():
                raise RuntimeError("All LTX API worker registrations were removed after failed health checks.")
            last_check = now
        if on_status:
            on_status("All registered LTX API workers are busy or offline. Waiting for a free worker...")
        time.sleep(min(5.0, max(0.5, deadline - now)))


def _file_binding(node_id: str, input_name: str) -> dict[str, str]:
    return {"node_id": str(node_id), "input_name": str(input_name), "upload_type": "input"}



def _node_links(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if isinstance(value[0], (str, int)) and isinstance(value[1], int):
            return [str(value[0])]
    return []


def _workflow_job_payload(
    graph: dict[str, Any], *, prompt: str, negative_prompt: str,
    duration: float, fps: int, width: int, height: int, seed: int, cfg: float,
    image_path: Path | None, audio_path: Path, video_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Infer upload bindings and editable inputs from the workflow graph itself."""
    inputs: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    nodes = {str(k): v for k, v in graph.items() if isinstance(v, dict)}
    downstream: dict[str, list[tuple[str, str]]] = {}
    for target_id, node in nodes.items():
        for field, value in (node.get("inputs") or {}).items():
            for source_id in _node_links(value):
                downstream.setdefault(source_id, []).append((target_id, str(field).lower()))

    def ancestors(target_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [target_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            node = nodes.get(current, {})
            for value in (node.get("inputs") or {}).values():
                stack.extend(_node_links(value))
        return seen

    positive_sources: set[str] = set()
    negative_sources: set[str] = set()
    for target_id, node in nodes.items():
        cls = str(node.get("class_type") or "").lower()
        if "conditioning" in cls or "guider" in cls:
            for field, value in (node.get("inputs") or {}).items():
                field = str(field).lower()
                sources = _node_links(value)
                if field in {"positive", "positive_cond", "positive_conditioning"}:
                    for source in sources:
                        positive_sources.update(ancestors(source))
                elif field in {"negative", "negative_cond", "negative_conditioning"}:
                    for source in sources:
                        negative_sources.update(ancestors(source))

    positive_text = str(prompt or "").strip()
    negative_text = str(negative_prompt or "").strip()
    text_candidates: list[tuple[str, str, str]] = []
    output_node_id: str | None = None

    for node_id, node in nodes.items():
        cls = str(node.get("class_type") or "")
        lower_cls = cls.lower()
        node_inputs = node.get("inputs") or {}
        override: dict[str, Any] = {}

        # Identify media loaders by node class and its actual input names.
        if "loadimage" in lower_cls and image_path is not None:
            field = next((x for x in node_inputs if str(x).lower() in {"image", "file", "path"}), None)
            if field:
                bindings.setdefault("image", {"node_id": node_id, "input_name": str(field), "upload_type": "image"})
        elif "loadaudio" in lower_cls:
            field = next((x for x in node_inputs if str(x).lower() in {"audio", "file", "path"}), None)
            if field:
                bindings.setdefault("audio", _file_binding(node_id, str(field)))
        elif "loadvideo" in lower_cls and video_path is not None:
            field = next((x for x in node_inputs if str(x).lower() in {"file", "video", "path"}), None)
            if field:
                bindings.setdefault("video", _file_binding(node_id, str(field)))

        # Apply values only to inputs whose names express the setting.
        for field in node_inputs:
            key = str(field).lower()
            if key in {"noise_seed", "seed", "seed_value"}:
                override[field] = int(seed)
            elif key in {"width", "video_width"}:
                override[field] = int(width)
            elif key in {"height", "video_height"}:
                override[field] = int(height)
            elif key in {"duration", "seconds", "video_duration"}:
                override[field] = float(duration)
            elif key in {"fps", "frame_rate", "framerate"}:
                override[field] = int(fps)
            elif key in {"cfg", "video_cfg", "audio_cfg"}:
                override[field] = float(cfg)
            elif key in {"length", "num_frames", "frame_count"}:
                override[field] = max(1, int(round(float(duration) * int(fps))))

        # Text node candidates are resolved through conditioning graph links.
        for field, value in node_inputs.items():
            field_lower = str(field).lower()
            if field_lower not in {"text", "value", "prompt", "positive_prompt", "negative_prompt"}:
                continue
            if not isinstance(value, str):
                continue
            if "text" in lower_cls or "prompt" in lower_cls or "primitive" in lower_cls:
                text_candidates.append((node_id, str(field), lower_cls))

        if any(token in lower_cls for token in ("savevideo", "videocombine", "saveanimated", "savewebm")):
            output_node_id = node_id
            if "filename_prefix" in node_inputs:
                override["filename_prefix"] = f"video/F5PdfStudio_{secrets.token_hex(5)}"
        # Primitive widget nodes may hold the value that is connected to a downstream setting.
        if "primitive" in lower_cls and "value" in node_inputs:
            for _target_id, target_field in downstream.get(node_id, []):
                key = target_field.lower()
                if key in {"width", "video_width"}:
                    override["value"] = int(width)
                elif key in {"height", "video_height"}:
                    override["value"] = int(height)
                elif key in {"fps", "frame_rate", "framerate"}:
                    override["value"] = int(fps)
                elif key in {"duration", "seconds", "video_duration"}:
                    override["value"] = float(duration)
                elif key in {"length", "num_frames", "frame_count"}:
                    override["value"] = max(1, int(round(float(duration) * int(fps))))
                elif key in {"cfg", "video_cfg", "audio_cfg"}:
                    override["value"] = float(cfg)
                elif key in {"seed", "noise_seed"}:
                    override["value"] = int(seed)
        if override:
            inputs[node_id] = override

    # Only override prompt nodes when the graph identifies their role unambiguously.
    for node_id, field, _cls in text_candidates:
        if node_id in negative_sources and negative_text:
            inputs.setdefault(node_id, {})[field] = negative_text
        elif node_id in positive_sources and positive_text:
            inputs.setdefault(node_id, {})[field] = positive_text
    unresolved = [
        item for item in text_candidates
        if item[0] not in positive_sources and item[0] not in negative_sources
    ]
    if positive_text and not positive_sources and len(unresolved) == 1:
        node_id, field, _ = unresolved[0]
        inputs.setdefault(node_id, {})[field] = positive_text

    if audio_path and "audio" not in bindings:
        raise ValueError("The selected workflow has no detectable audio loader node.")
    if image_path and "image" not in bindings:
        raise ValueError("An image was supplied, but the selected workflow has no detectable image loader node.")
    if video_path and "video" not in bindings:
        raise ValueError("A reference video was supplied, but the selected workflow has no detectable video loader node.")
    return inputs, bindings, output_node_id

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
    workflow_name = str(workflow_kind or "").strip()
    if not workflow_name:
        raise ValueError("Select a discovered workflow before generating.")
    audio_file = Path(str(audio_path or "")).expanduser()
    image_file = Path(str(image_path)).expanduser() if image_path else None
    video_file = Path(str(reference_video_path)).expanduser() if reference_video_path else None
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if image_file is not None and not image_file.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_file}")
    if video_file is not None and not video_file.is_file():
        raise FileNotFoundError(f"Reference video not found: {video_file}")

    worker = _wait_for_worker(workflow_name, wait_seconds=min(timeout, 1800), on_status=on_status)
    worker_id = int(worker["id"])
    base = worker["base_url"]
    key = worker["api_key"]
    job_id = ""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    workflow_path = Path(worker["workflow_path"])
    if not workflow_path.is_file():
        raise FileNotFoundError(
            f"Workflow file is no longer available: {workflow_path}. Re-register workers after adding or moving workflow JSON files."
        )
    import hashlib
    current_hash = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
    if current_hash != worker["workflow_hash"]:
        raise RuntimeError(
            f"Workflow '{workflow_name}' changed after registration. Register the worker again to sync the updated JSON."
        )
    _, workflow_graph = _load_workflow(workflow_path)
    chosen_seed = int(seed)
    if chosen_seed < 0:
        chosen_seed = secrets.randbelow(2**32 - 1)
    positive_prompt = str(prompt or "").strip()
    if str(extra_positive or "").strip():
        positive_prompt = (positive_prompt + "\n" + str(extra_positive).strip()).strip()
    inputs, bindings, output_node_id = _workflow_job_payload(
        workflow_graph,
        prompt=positive_prompt,
        negative_prompt=str(negative_prompt or "").strip(),
        duration=float(max(0.1, min(600.0, float(duration)))),
        fps=int(max(1, min(120, int(fps)))),
        width=int(max(1, min(8192, int(width)))),
        height=int(max(1, min(8192, int(height)))),
        seed=chosen_seed,
        cfg=float(max(0.0, min(100.0, float(cfg or 1.0)))),
        image_path=image_file,
        audio_path=audio_file,
        video_path=video_file,
    )
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
            if video_file is not None:
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
                    (
                        item for item in outputs
                        if output_node_id and str(item.get("node_id") or item.get("node") or "") == output_node_id
                    ),
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
