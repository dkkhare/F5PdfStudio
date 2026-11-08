"""Project metadata and deterministic output-folder management."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def safe_folder_name(value: Any, fallback: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name or fallback)[:120].rstrip(" .")


def get_project_directory(project_name: Any, part_number: Any) -> Path:
    project = safe_folder_name(project_name, "audiobook")
    part = safe_folder_name(part_number, "1")
    part = re.sub(r"^(?:part|भाग)[\s_-]*", "", part, flags=re.I) or "1"
    directory = OUTPUTS / project / f"Part_{part}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def open_or_create_project(project_name: Any, book_name: Any,
                           part_number: Any, author_name: Any) -> tuple[str, str]:
    if not str(book_name or "").strip():
        raise ValueError("Enter the book name.")
    if not str(part_number or "").strip():
        raise ValueError("Enter the part number.")
    if not str(project_name or "").strip():
        raise ValueError("Enter the project name.")
    directory = get_project_directory(project_name, part_number)
    metadata = {
        "project_name": str(project_name or "").strip() or "audiobook",
        "book_name": str(book_name).strip(),
        "part_number": str(part_number).strip(),
        "author_name": str(author_name or "").strip(),
        "output_directory": str(directory),
    }
    temporary = directory / "project.tmp.json"
    destination = directory / "project.json"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return (
        f"Project ready.\nProject: {metadata['project_name']}\nBook: {metadata['book_name']}\n"
        f"Part: {metadata['part_number']}\nAuthor: {metadata['author_name']}\n"
        f"Output folder: {directory}",
        str(directory),
    )


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def list_existing_projects() -> list[str]:
    """List projects saved under outputs, including readable legacy metadata."""
    projects: dict[str, float] = {}
    if not OUTPUTS.exists():
        return []
    for metadata_file in OUTPUTS.glob("**/project.json"):
        metadata = _read_json(metadata_file)
        name = str(metadata.get("project_name") or "").strip()
        if not name:
            continue
        try:
            modified = metadata_file.stat().st_mtime
        except OSError:
            modified = 0.0
        projects[name] = max(projects.get(name, 0.0), modified)
    return [name for name, _modified in sorted(
        projects.items(), key=lambda item: (-item[1], item[0].casefold())
    )]


def find_latest_project_directory(project_name: Any) -> Path:
    requested = str(project_name or "").strip().casefold()
    if not requested:
        raise ValueError("Select an existing project.")
    candidates: list[tuple[int, float, Path]] = []
    for metadata_file in OUTPUTS.glob("**/project.json"):
        metadata = _read_json(metadata_file)
        if str(metadata.get("project_name") or "").strip().casefold() != requested:
            continue
        directory = metadata_file.parent
        state = _read_json(directory / "project_state.json")
        audio = _read_json(directory / "generation_checkpoint.json")
        video = _read_json(directory / "video" / "video_checkpoint.json")
        # Prefer a directory containing real project work over a newly-created
        # metadata-only directory, even if the latter project.json is newer.
        content_rank = 0
        if state:
            content_rank = 1
        if video:
            content_rank = 2
        if audio:
            content_rank = 3
        if str(state.get("extracted_text") or "").strip():
            content_rank = 4
        mtimes = []
        for candidate_file in (
            metadata_file,
            directory / "project_state.json",
            directory / "generation_checkpoint.json",
            directory / "video" / "video_checkpoint.json",
        ):
            try:
                mtimes.append(candidate_file.stat().st_mtime)
            except OSError:
                pass
        candidates.append((content_rank, max(mtimes, default=0.0), directory.resolve()))
    if not candidates:
        raise FileNotFoundError(f"Project not found: {project_name}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _latest_video_ui_values(project_name: Any) -> dict:
    """Find the newest Tab 5 values across legacy and canonical folders."""
    requested = str(project_name or "").strip().casefold()
    newest: dict[str, tuple[float, Any]] = {}
    state_keys = (
        "final_video_filename",
        "video_transition",
        "video_transition_duration",
    )
    checkpoint_keys = {
        "final_video_filename": "final_video_filename",
        "transition": "video_transition",
        "transition_duration": "video_transition_duration",
    }
    for metadata_file in OUTPUTS.glob("**/project.json"):
        metadata = _read_json(metadata_file)
        if str(metadata.get("project_name") or "").strip().casefold() != requested:
            continue
        directory = metadata_file.parent
        state_file = directory / "project_state.json"
        state = _read_json(state_file)
        try:
            state_time = state_file.stat().st_mtime
        except OSError:
            state_time = 0.0
        for key in state_keys:
            value = state.get(key)
            if value is not None and (key not in newest or state_time > newest[key][0]):
                newest[key] = (state_time, value)

        video_file = directory / "video" / "video_checkpoint.json"
        video = _read_json(video_file)
        settings = video.get("settings") or {}
        try:
            video_time = video_file.stat().st_mtime
        except OSError:
            video_time = 0.0
        if isinstance(settings, dict):
            for checkpoint_key, state_key in checkpoint_keys.items():
                value = settings.get(checkpoint_key)
                if value is not None and (
                    state_key not in newest or video_time > newest[state_key][0]
                ):
                    newest[state_key] = (video_time, value)
        final_video = video.get("final_video")
        if final_video and (
            "final_video_filename" not in newest
            or video_time > newest["final_video_filename"][0]
        ):
            newest["final_video_filename"] = (video_time, Path(str(final_video)).name)
    return {key: value for key, (_modified, value) in newest.items()}


def save_project_state(output_directory: Any, updates: dict) -> None:
    if not str(output_directory or "").strip():
        return
    directory = Path(str(output_directory)).expanduser().resolve()
    if not directory.exists():
        return
    state_file = directory / "project_state.json"
    state = _read_json(state_file)
    state.update({key: value for key, value in updates.items() if value is not None})
    state["updated"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    _atomic_json(state_file, state)


def _recover_generated_text(directory: Path, poetry_mode: bool) -> str:
    """Recover source text saved beside generated WAVs by older releases."""
    recovered_sections: list[str] = []
    for report_file in sorted((directory / "sections").glob("*.validation.json")):
        report = _read_json(report_file)
        expected_parts = [
            str(check.get("expected") or "").strip()
            for check in report.get("checks", [])
            if isinstance(check, dict) and str(check.get("expected") or "").strip()
        ]
        if expected_parts:
            separator = "\n" if poetry_mode else " "
            recovered_sections.append(separator.join(expected_parts))
    return "\n\n".join(recovered_sections).strip()


def load_project_state(project_name: Any) -> dict:
    """Load metadata, last UI values, checkpoints and generated artifacts."""
    directory = find_latest_project_directory(project_name)
    metadata = _read_json(directory / "project.json")
    state = _read_json(directory / "project_state.json")
    # A project may span an old book-name folder and the newer project-name
    # folder. Merge these particular UI values by modification time.
    state.update(_latest_video_ui_values(project_name))
    audio_checkpoint = _read_json(directory / "generation_checkpoint.json")
    video_checkpoint = _read_json(directory / "video" / "video_checkpoint.json")
    saved_video_settings = video_checkpoint.get("settings") or {}
    if isinstance(saved_video_settings, dict):
        for state_key, checkpoint_key in (
            ("sadtalker_directory", "sadtalker_directory"),
            ("sadtalker_environment", "environment_name"),
            ("preprocess_mode", "preprocess"),
            ("video_model_size", "size"),
            ("still_mode", "still"),
            ("output_resolution", "output_resolution"),
            ("burn_subtitles", "burn_subtitles"),
            ("subtitle_font", "subtitle_font"),
            ("subtitle_font_size", "subtitle_font_size"),
            ("video_transition", "transition"),
            ("video_transition_duration", "transition_duration"),
            ("final_video_filename", "final_video_filename"),
        ):
            if state_key not in state and checkpoint_key in saved_video_settings:
                state[state_key] = saved_video_settings[checkpoint_key]
    for state_key, checkpoint_key in (
        ("speech_speed", "speech_speed"),
        ("pause_duration", "section_pause_ms"),
        ("poetry_line_pause_duration", "poetry_line_pause_ms"),
        ("device_choice", "device_choice"),
        ("seed", "seed"),
        ("nfe_steps", "nfe_steps"),
        ("create_mp3", "create_mp3"),
    ):
        if state_key not in state and checkpoint_key in audio_checkpoint:
            state[state_key] = audio_checkpoint[checkpoint_key]

    assets_directory = directory / "assets"
    if not state.get("reference_audio") or not Path(str(state.get("reference_audio"))).is_file():
        saved_reference = next(iter(sorted(assets_directory.glob("reference_audio.*"))), None)
        if saved_reference and saved_reference.is_file():
            state["reference_audio"] = str(saved_reference.resolve())
    if not state.get("poet_image") or not Path(str(state.get("poet_image"))).is_file():
        saved_portrait = next(iter(sorted(assets_directory.glob("poet_portrait.*"))), None)
        if saved_portrait and saved_portrait.is_file():
            state["poet_image"] = str(saved_portrait.resolve())
        else:
            checkpoint_portrait = video_checkpoint.get("image")
            if checkpoint_portrait and Path(str(checkpoint_portrait)).is_file():
                state["poet_image"] = str(Path(str(checkpoint_portrait)).resolve())

    if not state.get("book_type"):
        state["book_type"] = audio_checkpoint.get("book_type") or "Poetry"
    poetry_mode = str(state.get("book_type") or "").strip().casefold() == "poetry"
    if not str(state.get("extracted_text") or "").strip():
        state["extracted_text"] = (
            audio_checkpoint.get("source_text")
            or _recover_generated_text(directory, poetry_mode)
        )
        if state["extracted_text"]:
            state["text_status"] = "Text restored from the saved audio-generation records."
            state["section_preview"] = state["extracted_text"]
    if not state.get("reference_audio"):
        reference_audio = audio_checkpoint.get("reference_audio")
        if reference_audio and Path(reference_audio).is_file():
            state["reference_audio"] = str(Path(reference_audio))
    if not str(state.get("reference_transcript") or "").strip():
        state["reference_transcript"] = audio_checkpoint.get("reference_text") or ""

    audio_total = int(audio_checkpoint.get("total_sections") or 0)
    audio_completed = len(audio_checkpoint.get("completed_sections") or [])
    video_total = int(video_checkpoint.get("total_sections") or 0)
    video_completed = len(video_checkpoint.get("completed_sections") or [])
    state.update(metadata)
    state["output_directory"] = str(directory)
    state["audio_progress"] = 100.0 * audio_completed / max(1, audio_total)
    state["video_progress"] = 100.0 * video_completed / max(1, video_total)
    state["audio_status"] = (
        f"Saved audio status: {audio_checkpoint.get('status', 'not started')}\n"
        f"Completed sections: {audio_completed}/{audio_total}\n"
        f"Output: {directory}"
    )
    state["video_status"] = (
        f"Saved video status: {video_checkpoint.get('status', 'not started')}\n"
        f"Completed sections: {video_completed}/{video_total}\n"
        f"Output: {directory / 'video'}"
    )
    state["wav_file"] = str(directory / "audiobook_complete.wav") if (directory / "audiobook_complete.wav").is_file() else None
    state["mp3_file"] = str(directory / "audiobook_complete.mp3") if (directory / "audiobook_complete.mp3").is_file() else None
    final_video = video_checkpoint.get("final_video")
    state["final_video"] = str(final_video) if final_video and Path(final_video).is_file() else None
    if final_video and not state.get("final_video_filename"):
        state["final_video_filename"] = Path(str(final_video)).name
    subtitle = video_checkpoint.get("subtitle_file")
    state["subtitle_file"] = str(subtitle) if subtitle and Path(subtitle).is_file() else None
    return state


# Compatibility helpers retained for older callers.
def safe_project_name(name: str) -> str:
    return safe_folder_name(name, "audiobook")


def get_project_paths(project_name: str):
    directory = OUTPUTS / safe_project_name(project_name)
    directory.mkdir(parents=True, exist_ok=True)
    return safe_project_name(project_name), directory, directory / "generation_checkpoint.json"


def load_checkpoint(checkpoint_file: Path):
    if not checkpoint_file.exists():
        return {"completed": [], "paused": False}
    return json.loads(checkpoint_file.read_text(encoding="utf-8"))


def save_checkpoint(checkpoint_file: Path, state: dict):
    temporary = checkpoint_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(checkpoint_file)
