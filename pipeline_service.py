"""Shared end-to-end pipeline helpers for F5PdfStudio.

One place for the rules that every tab must follow:

* file names      -> ``<book>_<part>_<section>`` for every audio and video file
* folders         -> all section audio of a project in ``<project>/sections`` and all
                     section videos in ``<project>/video/sections`` regardless of the tool
* sections        -> one canonical section list per project (``sections.json``) used by the
                     audio tools, the video tools and the burned text
* stage status    -> ``pipeline_status.json`` updated after every section / stage
* pause / stop    -> take effect only between units of work (a finished section) and the next
                     run resumes from the checkpoint
* video finishing -> the text of a section is burned on the whole section video, then the
                     section videos are combined into clips of at most N minutes without
                     ever cutting a section
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

DEFAULT_MAX_CLIP_SECONDS = 600          # 10 minutes
SECTION_DIGITS = 3

STAGES = [
    ("extract", "1. Extract text"),
    ("correct", "2. Correct text"),
    ("sections", "3. Divide into sections"),
    ("audio", "4. Audio"),
    ("video", "5. Video (section clips)"),
    ("combine", "6. Combine clips"),
]
STAGE_STATES = ("not started", "running", "paused", "stopped", "completed", "failed")


# =============================================================== file naming
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(value: Any, fallback: str = "untitled", max_length: int = 60) -> str:
    """File-system safe name that keeps Hindi/Unicode letters."""
    text = _ILLEGAL.sub("_", str(value or "").strip())
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    text = text[:max_length].rstrip("._ ")
    return text or fallback


def clean_part(part: Any) -> str:
    text = str(part or "").strip()
    text = re.sub(r"^(?:part|भाग)[\s_-]*", "", text, flags=re.I)
    return text or "1"


def project_prefix(book: Any, part: Any) -> str:
    """``<book>_<part>`` used in front of every section number."""
    return f"{safe_stem(book, 'book')}_{safe_stem(clean_part(part), '1', 20)}"


def section_stem(book: Any, part: Any, index: int) -> str:
    return f"{project_prefix(book, part)}_{int(index):0{SECTION_DIGITS}d}"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def project_identity(project_dir: Any) -> tuple[str, str]:
    """Return ``(book_name, part_number)`` of a project folder."""
    directory = Path(str(project_dir)).expanduser()
    meta = _read_json(directory / "project.json")
    book = str(meta.get("book_name") or "").strip()
    part = str(meta.get("part_number") or "").strip()
    if not part:
        match = re.match(r"Part_(.+)$", directory.name)
        part = match.group(1) if match else "1"
    if not book:
        book = directory.parent.name if directory.name.startswith("Part_") else directory.name
    return book, part


def prefix_for(project_dir: Any) -> str:
    book, part = project_identity(project_dir)
    return project_prefix(book, part)


# ------------------------------------------------------------------ audio files
def audio_directory(project_dir: Any) -> Path:
    directory = Path(str(project_dir)).expanduser() / "sections"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def audio_section_path(project_dir: Any, index: int) -> Path:
    book, part = project_identity(project_dir)
    return audio_directory(project_dir) / f"{section_stem(book, part, index)}.wav"


_LEGACY_AUDIO = re.compile(r"^section_(\d+)(\..+)$", re.I)


def migrate_legacy_audio(project_dir: Any) -> int:
    """Rename ``section_0001.wav`` (and sidecar files) to ``<book>_<part>_001.wav``."""
    directory = Path(str(project_dir)).expanduser() / "sections"
    if not directory.is_dir():
        return 0
    book, part = project_identity(project_dir)
    renamed = 0
    for file in sorted(directory.iterdir()):
        match = _LEGACY_AUDIO.match(file.name)
        if not match or not file.is_file():
            continue
        target = directory / f"{section_stem(book, part, int(match.group(1)))}{match.group(2)}"
        if target.exists():
            continue
        try:
            file.rename(target)
            renamed += 1
        except OSError:
            pass
    return renamed


def find_audio_section(project_dir: Any, index: int) -> Optional[Path]:
    """Return the section WAV (new name first, legacy name migrated) or ``None``."""
    root = Path(str(project_dir)).expanduser()
    migrate_legacy_audio(root)
    wanted = audio_section_path(root, index)
    if wanted.is_file():
        return wanted
    # A project root that only contains Part_* folders (older layout).
    for sibling in sorted(root.glob("Part_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if sibling.is_dir():
            migrate_legacy_audio(sibling)
            candidate = audio_section_path(sibling, index)
            if candidate.is_file():
                return candidate
    return None


# ------------------------------------------------------------------ video files
def video_directory(project_dir: Any) -> Path:
    directory = Path(str(project_dir)).expanduser() / "video"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def video_section_directory(project_dir: Any) -> Path:
    directory = video_directory(project_dir) / "sections"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def video_section_path(project_dir: Any, index: int) -> Path:
    book, part = project_identity(project_dir)
    return video_section_directory(project_dir) / f"{section_stem(book, part, index)}.mp4"


# ======================================================= canonical sections
def _fingerprint(text: str, book_type: str, max_characters: int) -> str:
    payload = f"{book_type}|{int(max_characters)}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_canonical_sections(
    project_dir: Any,
    text: str,
    book_type: str,
    max_characters: int,
    splitter: Callable[..., list[str]],
) -> list[str]:
    """Divide the text once; audio, video and burned text all use the same sections."""
    directory = Path(str(project_dir)).expanduser()
    fingerprint = _fingerprint(text, str(book_type), int(max_characters))
    stored = _read_json(directory / "sections.json")
    if stored.get("fingerprint") == fingerprint and isinstance(stored.get("sections"), list) and stored["sections"]:
        return [str(item) for item in stored["sections"]]
    sections = [str(item) for item in splitter(text, int(max_characters), book_type=book_type) if str(item).strip()]
    _atomic_json(directory / "sections.json", {
        "fingerprint": fingerprint,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "book_type": str(book_type),
        "max_characters": int(max_characters),
        "total": len(sections),
        "sections": sections,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return sections


def load_matching_sections(project_dir: Any, text: str) -> Optional[list[str]]:
    """Sections stored for exactly this text (whatever splitter settings made them)."""
    stored = _read_json(Path(str(project_dir)).expanduser() / "sections.json")
    if (stored.get("text_hash") == hashlib.sha256(text.encode("utf-8")).hexdigest()
            and isinstance(stored.get("sections"), list) and stored["sections"]):
        return [str(item) for item in stored["sections"]]
    return None


# =============================================================== stage status
_status_lock = threading.RLock()


def pipeline_file(project_dir: Any) -> Path:
    return Path(str(project_dir)).expanduser() / "pipeline_status.json"


def read_pipeline(project_dir: Any) -> dict:
    return _read_json(pipeline_file(project_dir))


def update_stage(
    project_dir: Any,
    stage: str,
    status: str,
    *,
    done: Optional[int] = None,
    total: Optional[int] = None,
    engine: Optional[str] = None,
    message: str = "",
    **extra: Any,
) -> dict:
    """Persist the status of one pipeline stage. Called after every section."""
    if not str(project_dir or "").strip():
        return {}
    directory = Path(str(project_dir)).expanduser()
    if not directory.exists():
        return {}
    with _status_lock:
        data = read_pipeline(directory)
        stages = data.setdefault("stages", {})
        entry = stages.setdefault(stage, {})
        entry.update({
            "status": status,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if done is not None:
            entry["done"] = int(done)
        if total is not None:
            entry["total"] = int(total)
        if engine:
            entry["engine"] = str(engine)
        if message:
            entry["message"] = str(message)[:400]
        entry.update({key: value for key, value in extra.items() if value is not None})
        data["updated"] = entry["updated"]
        data["last_stage"] = stage
        _atomic_json(pipeline_file(directory), data)
    try:  # mirror a one-line summary into project_state.json
        from project_service import save_project_state
        save_project_state(directory, {"pipeline_summary": format_pipeline_status(directory, short=True)})
    except Exception:
        pass
    return entry


def format_pipeline_status(project_dir: Any, short: bool = False) -> str:
    data = read_pipeline(project_dir) if str(project_dir or "").strip() else {}
    stages = data.get("stages", {})
    lines = []
    for key, label in STAGES:
        entry = stages.get(key) or {}
        status = entry.get("status", "not started")
        progress = ""
        if entry.get("total"):
            progress = f" {entry.get('done', 0)}/{entry['total']}"
        engine = f" [{entry['engine']}]" if entry.get("engine") else ""
        lines.append(f"{label}: {status}{progress}{engine}")
    if short:
        return " | ".join(lines)
    text = "\n".join(lines)
    if data.get("updated"):
        text += f"\n\nLast update: {data['updated']}"
    message = (stages.get(data.get("last_stage", "")) or {}).get("message")
    if message:
        text += f"\n{message}"
    return text


# ================================================================ controls
class StageStopped(RuntimeError):
    """Raised when Stop was requested; raised only between units of work."""


class StageControl:
    """Pause / resume / stop flags for text stages (extract, correct, Gemini)."""

    def __init__(self, name: str):
        self.name = name
        self._pause = threading.Event()
        self._stop = threading.Event()

    def pause(self) -> str:
        self._pause.set()
        return f"{self.name}: pause requested. It takes effect after the current unit finishes."

    def resume(self) -> str:
        self._pause.clear()
        return f"{self.name}: resumed."

    def stop(self) -> str:
        self._stop.set()
        self._pause.clear()
        return f"{self.name}: stop requested. Finished work is saved; run it again to resume."

    def clear(self) -> None:
        self._pause.clear()
        self._stop.clear()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def is_stop_requested(self) -> bool:
        return self._stop.is_set()

    def checkpoint(self, notify: Optional[Callable[[str], None]] = None) -> None:
        """Call between units of work: blocks while paused, raises when stopped."""
        if self._stop.is_set():
            raise StageStopped(f"{self.name} stopped by user.")
        announced = False
        while self._pause.is_set():
            if self._stop.is_set():
                raise StageStopped(f"{self.name} stopped by user.")
            if not announced and notify:
                try:
                    notify(f"{self.name} paused. Press Resume to continue.")
                except Exception:
                    pass
                announced = True
            time.sleep(0.25)


_CONTROLS: dict[str, StageControl] = {}


def get_control(name: str) -> StageControl:
    if name not in _CONTROLS:
        _CONTROLS[name] = StageControl(name)
    return _CONTROLS[name]


# ============================================================ media helpers
def _ffmpeg() -> str:
    try:
        from video_service import locate_ffmpeg
        found = locate_ffmpeg()
        if found:
            return found
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("FFmpeg was not found on PATH.")
    return found


def _ffprobe() -> str:
    try:
        from video_service import locate_ffprobe
        found = locate_ffprobe()
        if found:
            return found
    except Exception:
        pass
    found = shutil.which("ffprobe")
    if not found:
        raise RuntimeError("ffprobe was not found on PATH.")
    return found


def probe_media(path: Any) -> dict:
    """Duration, size, fps and whether the file has audio/video streams."""
    completed = subprocess.run(
        [_ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}:\n{completed.stderr}")
    data = json.loads(completed.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = 0.0
    if video and video.get("r_frame_rate") and "/" in str(video["r_frame_rate"]):
        num, den = str(video["r_frame_rate"]).split("/", 1)
        try:
            fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            fps = 0.0
    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    if not duration and audio and audio.get("duration"):
        duration = float(audio["duration"])
    return {
        "duration": duration,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "fps": fps,
    }


_subtitles_ok: Optional[bool] = None


def ffmpeg_has_subtitles() -> bool:
    global _subtitles_ok
    if _subtitles_ok is None:
        try:
            completed = subprocess.run([_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30)
            _subtitles_ok = bool(re.search(r"\bsubtitles\b", completed.stdout + completed.stderr))
        except Exception:
            _subtitles_ok = False
    return bool(_subtitles_ok)


# ----------------------------------------------------------- text paging
def _wrap_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def layout_for(width: int, height: int, font_size: int) -> tuple[int, int]:
    """Characters per line and lines per page that fit the frame at ``font_size``.

    ffmpeg renders SRT text at ``font_size`` / 288 of the frame height.
    """
    pixel_size = max(8.0, float(font_size) * float(height) / 288.0)
    chars = int(max(14, min(70, (float(width) * 0.92) / (pixel_size * 0.66))))
    lines = int(max(2, min(6, (float(height) * 0.42) / (pixel_size * 1.25))))
    return chars, lines


def paginate_text(text: str, max_chars: int = 36, max_lines: int = 3, preserve_lines: bool = False) -> list[str]:
    """Split the section text into pages that fit on screen.

    Poetry keeps its line breaks and shows a whole stanza (up to six lines) on one page.
    """
    raw = str(text or "").replace("\r", "")
    if preserve_lines:
        lines: list[str] = []
        for line in raw.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.extend(_wrap_words(line, max_chars) or [line])
        limit = max(max_lines, 6) if len(lines) <= 6 else max_lines
    else:
        flat = re.sub(r"\s+", " ", raw).strip()
        lines = _wrap_words(flat, max_chars)
        limit = max_lines
    pages = ["\n".join(lines[i:i + limit]) for i in range(0, len(lines), limit)]
    return [page for page in pages if page.strip()]


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def page_timings(pages: list[str], duration: float) -> list[tuple[float, float]]:
    """Each page is shown for a share of the clip proportional to its length; together
    the pages cover the entire section video."""
    weights = [max(1, len(re.sub(r"\s+", "", page))) for page in pages]
    total = float(sum(weights)) or 1.0
    cursor = 0.0
    result = []
    for position, weight in enumerate(weights):
        end = duration if position == len(weights) - 1 else cursor + duration * weight / total
        result.append((cursor, end))
        cursor = end
    return result


def write_srt(pages: list[str], duration: float, path: Path, offset: float = 0.0, first_number: int = 1) -> int:
    blocks = []
    for number, (page, (start, end)) in enumerate(zip(pages, page_timings(pages, duration)), start=first_number):
        blocks.append(f"{number}\n{_srt_time(start + offset)} --> {_srt_time(end + offset)}\n{page}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8-sig")
    return first_number + len(blocks)


# ----------------------------------------------- finishing one section video
def finalize_section_video(
    source: Any,
    destination: Any,
    text: str,
    *,
    audio_path: Any = None,
    mux_audio: bool = False,
    burn: bool = True,
    width: int = 1920,
    height: int = 1080,
    fps: int = 25,
    font_name: str = "Nirmala UI",
    font_size: int = 28,
    poetry: bool = False,
    work_dir: Any = None,
    preset: str = "medium",
) -> dict:
    """Normalise one engine output and burn the section text on its entire length.

    * the video is scaled/padded to ``width`` x ``height`` and re-encoded with identical
      parameters so that all clips can be joined without re-encoding;
    * ``mux_audio=True`` replaces/adds the section audio (for engines that produce silent
      video); the picture is looped to cover the whole narration;
    * the text is split into pages that are shown one after another so the complete text
      is on screen across the entire clip.
    """
    source, destination = Path(str(source)), Path(str(destination))
    info = probe_media(source)
    if not info["has_video"] or info["duration"] <= 0:
        raise RuntimeError(f"The engine output has no usable video: {source}")
    use_external = bool(audio_path) and Path(str(audio_path)).is_file() and (mux_audio or not info["has_audio"])
    audio_duration = probe_media(audio_path)["duration"] if use_external else 0.0
    duration = audio_duration if use_external and audio_duration > 0 else info["duration"]
    loop = use_external and info["duration"] + 0.05 < duration

    work = Path(str(work_dir)) if work_dir else destination.parent
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    srt_file: Optional[Path] = None
    pages: list[str] = []

    filters = [
        f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease",
        f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]
    burn_srt_name = None
    if burn and str(text or "").strip():
        if ffmpeg_has_subtitles():
            chars, lines = layout_for(width, height, font_size)
            pages = paginate_text(text, chars, lines, preserve_lines=poetry)
            burn_srt_name = f"burn_{destination.stem[-12:]}.srt".replace(" ", "_")
            srt_file = work / re.sub(r"[^A-Za-z0-9._-]", "_", burn_srt_name)
            write_srt(pages, duration, srt_file)
            style = (
                f"FontName={font_name},FontSize={int(font_size)},"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=42"
            )
            filters.append(f"subtitles=filename='{srt_file.name}':force_style='{style}'")
        else:
            warnings.append("This FFmpeg build has no subtitles/libass filter; the text was not burned.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    command = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error"]
    if loop:
        command += ["-stream_loop", "-1"]
    command += ["-i", str(source)]
    if use_external:
        command += ["-i", str(audio_path)]
    command += ["-map", "0:v:0"]
    if use_external:
        command += ["-map", "1:a:0"]
    elif info["has_audio"]:
        command += ["-map", "0:a:0"]
    command += [
        "-vf", ",".join(filters), "-r", str(int(fps)),
        "-c:v", "libx264", "-preset", preset, "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
    ]
    if loop or use_external:
        command += ["-t", f"{duration:.3f}"]
    command += ["-movflags", "+faststart", str(temporary)]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(work))
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg could not finish the section video:\n{completed.stderr[-1500:]}")
    temporary.replace(destination)
    final = probe_media(destination)
    return {
        "path": str(destination),
        "duration": final["duration"],
        "burned": bool(pages),
        "pages": len(pages),
        "srt": str(srt_file) if srt_file else None,
        "warnings": warnings,
    }


# ====================================================== combining the clips
def group_by_duration(durations: list[float], max_seconds: float) -> list[list[int]]:
    """Sequential groups whose total is at most ``max_seconds``; a section is never cut,
    so a single section longer than the limit becomes a group of its own."""
    groups: list[list[int]] = []
    current: list[int] = []
    total = 0.0
    for position, duration in enumerate(durations):
        if current and total + duration > max_seconds + 0.01:
            groups.append(current)
            current, total = [], 0.0
        current.append(position)
        total += duration
    if current:
        groups.append(current)
    return groups


def _concat_line(path: Path) -> str:
    value = Path(path).resolve().as_posix().replace("'", "'\\''")
    return f"file '{value}'"


def combine_clips(
    section_videos: list[Any],
    out_dir: Any,
    prefix: str,
    max_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    durations: Optional[list[float]] = None,
) -> list[dict]:
    """Join the section videos (in order) into clips of at most ``max_seconds``."""
    files = [Path(str(item)) for item in section_videos]
    if not files:
        raise RuntimeError("No section videos are available to combine.")
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if durations is None or len(durations) != len(files):
        durations = [probe_media(file)["duration"] for file in files]
    for old in out_dir.glob(f"{glob_escape(prefix)}_final_*.mp4"):
        try:
            old.unlink()
        except OSError:
            pass
    groups = group_by_duration(durations, float(max_seconds))
    results = []
    for number, group in enumerate(groups, start=1):
        destination = out_dir / f"{prefix}_final_{number:02d}.mp4"
        listing = out_dir / f"concat_{number:02d}.txt"
        listing.write_text("\n".join(_concat_line(files[i]) for i in group) + "\n", encoding="utf-8")
        temporary = destination.with_name(destination.stem + ".tmp.mp4")
        command = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                   "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(temporary)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:  # parameters differ: join with re-encoding instead
            command = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                       "-i", str(listing), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"FFmpeg could not combine the clips:\n{completed.stderr[-1500:]}")
        temporary.replace(destination)
        try:
            listing.unlink()
        except OSError:
            pass
        results.append({
            "path": str(destination),
            "sections": [i + 1 for i in group],
            "seconds": sum(durations[i] for i in group),
        })
    return results


def glob_escape(value: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", value)


# ============================================ engines with a maximum clip length
def silence_midpoints(audio: Any, noise_db: int = -35, min_silence: float = 0.25) -> list[float]:
    """Centre of every pause in the narration (used to cut long audio between words)."""
    completed = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-nostats", "-i", str(audio), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", completed.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", completed.stderr)]
    return [(a + b) / 2.0 for a, b in zip(starts, ends)]


def plan_audio_pieces(total: float, max_seconds: float, pauses: list[float]) -> list[tuple[float, float]]:
    """Cut points at pauses so that no piece is longer than ``max_seconds``."""
    pieces: list[tuple[float, float]] = []
    start = 0.0
    while total - start > max_seconds + 0.05:
        limit = start + max_seconds
        candidates = [p for p in pauses if start + max_seconds * 0.4 <= p <= limit]
        cut = max(candidates) if candidates else limit
        pieces.append((start, cut))
        start = cut
    pieces.append((start, total))
    return pieces


def split_audio(audio: Any, max_seconds: float, out_dir: Any, stem: str) -> list[Path]:
    """Split ``audio`` into WAV pieces of at most ``max_seconds``; one file when it is short."""
    total = probe_media(audio)["duration"]
    if total <= max_seconds + 0.05:
        return [Path(str(audio))]
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_audio_pieces(total, float(max_seconds), silence_midpoints(audio))
    files = []
    for number, (start, end) in enumerate(plan, start=1):
        target = out_dir / f"{stem}_p{number:02d}.wav"
        completed = subprocess.run(
            [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", str(audio), "-c:a", "pcm_s16le", str(target)],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg could not split the audio:\n{completed.stderr[-800:]}")
        files.append(target)
    return files


def concat_files(files: list[Any], destination: Any) -> Path:
    """Join clips produced by the same engine (stream copy, re-encode if needed)."""
    destination = Path(str(destination))
    listing = destination.with_suffix(".concat.txt")
    listing.write_text("\n".join(_concat_line(Path(str(f))) for f in files) + "\n", encoding="utf-8")
    base = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    completed = subprocess.run(base + ["-c", "copy", str(destination)], capture_output=True, text=True)
    if completed.returncode != 0:
        completed = subprocess.run(
            base + ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(destination)],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg could not join the pieces:\n{completed.stderr[-800:]}")
    try:
        listing.unlink()
    except OSError:
        pass
    return destination


def generate_in_pieces(
    piece_generator: Callable[[int, int, Path, Optional[Path], float], Any],
    raw_destination: Any,
    audio: Optional[Path],
    max_seconds: Optional[float],
    work_dir: Any,
    stem: str,
) -> None:
    """Run an engine that has a per-clip limit over the section audio.

    ``piece_generator(piece_number, piece_count, piece_raw, piece_audio, piece_seconds)``.
    The pieces are joined into ``raw_destination``.
    """
    raw_destination = Path(str(raw_destination))
    if audio is None or not max_seconds:
        piece_generator(1, 1, raw_destination, audio, float(max_seconds or 0))
        return
    pieces = split_audio(audio, float(max_seconds), work_dir, stem)
    if len(pieces) == 1:
        piece_generator(1, 1, raw_destination, pieces[0], probe_media(pieces[0])["duration"])
        return
    outputs = []
    for number, piece in enumerate(pieces, start=1):
        target = Path(str(work_dir)) / f"{stem}_p{number:02d}.mp4"
        piece_generator(number, len(pieces), target, piece, probe_media(piece)["duration"])
        outputs.append(target)
    concat_files(outputs, raw_destination)
    for item in outputs + [p for p in pieces if p != audio]:
        try:
            Path(item).unlink()
        except OSError:
            pass


# ====================================================== the video stage runner
def _file_fingerprint(path: Any) -> str:
    if not path:
        return ""
    file = Path(str(path))
    if not file.is_file():
        return str(path)
    stat = file.stat()
    if stat.st_size <= 50 * 1024 * 1024:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def video_signature(engine: str, sections: list[str], settings: dict, audio_files: list[Optional[Path]]) -> str:
    payload = {
        "engine": engine,
        "sections": [str(item) for item in sections],
        "settings": {k: v for k, v in settings.items() if not str(k).startswith("_")},
        "audio": [_file_fingerprint(path) if path else "" for path in audio_files],
    }
    for key in ("image", "reference_video"):
        if settings.get(key):
            payload[f"{key}_fingerprint"] = _file_fingerprint(settings[key])
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def run_video_stage(
    *,
    engine: str,
    project_dir: Any,
    sections: list[str],
    generator: Callable[[int, str, Path, Optional[Path]], Any],
    settings: dict,
    require_audio: bool = True,
    mux_audio: bool = False,
    burn: bool = True,
    width: int = 1920,
    height: int = 1080,
    fps: int = 25,
    font_name: str = "Nirmala UI",
    font_size: int = 28,
    poetry: bool = False,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    control_check: Optional[Callable[[], None]] = None,
    is_paused: Optional[Callable[[], bool]] = None,
    stop_exceptions: tuple = (),
    notify: Optional[Callable[..., None]] = None,
) -> dict:
    """Send the section audio to a video engine one section at a time.

    For every section: engine output -> burned text -> ``video/sections/<book>_<part>_<nnn>.mp4``
    -> checkpoint + stage status.  After the last section the clips are combined into files of
    at most ``max_clip_seconds`` (sections are never cut).
    """
    directory = Path(str(project_dir)).expanduser()
    total = len(sections)
    prefix = prefix_for(directory)
    migrate_legacy_audio(directory)
    audio_files = [find_audio_section(directory, index) for index in range(1, total + 1)]
    if require_audio:
        missing = [index + 1 for index, file in enumerate(audio_files) if file is None]
        if missing:
            example = audio_section_path(directory, missing[0]).name
            raise RuntimeError(
                f"Generate the audio first. {len(missing)} section audio file(s) are missing, "
                f"for example {example} (sections {missing[:10]})."
            )

    video_dir = video_directory(directory)
    section_dir = video_section_directory(directory)
    work_dir = video_dir / "work" / safe_stem(engine, "engine")
    previous_dir = video_dir / "previous"
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = video_dir / f"video_checkpoint_{safe_stem(engine, 'engine')}.json"
    compat_file = video_dir / "video_checkpoint.json"
    settings = dict(settings)
    settings.update({"burn": bool(burn), "width": int(width), "height": int(height), "font": font_name,
                     "font_size": int(font_size), "mux_audio": bool(mux_audio)})
    signature = video_signature(engine, sections, settings, audio_files)
    previous = _read_json(checkpoint_file)
    completed: dict[int, dict] = {}
    if previous.get("signature") == signature:
        for key, entry in (previous.get("sections") or {}).items():
            if str(key).isdigit() and Path(str(entry.get("path", ""))).is_file():
                completed[int(key)] = entry

    def save(status: str, extra: Optional[dict] = None) -> dict:
        state = {
            "version": 2, "engine": engine, "signature": signature, "status": status,
            "image": settings.get("image"),
            "completed_sections": sorted(completed), "total_sections": total,
            "last_completed_section": max(completed) if completed else 0,
            "sections": {str(k): v for k, v in sorted(completed.items())},
            "settings": settings, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            state.update(extra)
        _atomic_json(checkpoint_file, state)
        _atomic_json(compat_file, state)
        return state

    def say(message: str, done: Optional[int] = None) -> None:
        if notify:
            try:
                notify(message, len(completed) if done is None else done, total)
            except TypeError:
                notify(message)

    def check(after_section: bool) -> None:
        if control_check is None:
            return
        if is_paused and is_paused():
            update_stage(directory, "video", "paused", done=len(completed), total=total, engine=engine,
                         message=f"Paused after section {max(completed) if completed else 0}/{total}. Press Resume.")
            say("Video generation paused. Press Resume to continue.")
        control_check()
        update_stage(directory, "video", "running", done=len(completed), total=total, engine=engine)

    update_stage(directory, "video", "running", done=len(completed), total=total, engine=engine,
                 message=f"{engine}: {len(completed)}/{total} section videos already done.")
    state = save("running")
    stopped = False
    warnings: list[str] = []
    try:
        for index, text in enumerate(sections, start=1):
            check(after_section=False)
            destination = video_section_path(directory, index)
            if index in completed and destination.is_file():
                continue
            audio = audio_files[index - 1]
            raw = work_dir / f"{destination.stem}_raw.mp4"
            if raw.exists():
                raw.unlink()
            say(f"{engine}: generating section {index}/{total} ({destination.name})", len(completed))
            try:
                generator(index, str(text), raw, audio)
                if not raw.is_file() or raw.stat().st_size < 1000:
                    raise RuntimeError(f"{engine} finished section {index} but no usable MP4 was created.")
                if destination.is_file():  # keep the previous version when another engine redoes the section
                    previous_dir.mkdir(parents=True, exist_ok=True)
                    destination.replace(previous_dir / f"{destination.stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                say(f"Burning the section text on {destination.name}…", len(completed))
                result = finalize_section_video(
                    raw, destination, text, audio_path=audio, mux_audio=mux_audio, burn=burn,
                    width=width, height=height, fps=fps, font_name=font_name, font_size=font_size,
                    poetry=poetry, work_dir=work_dir,
                )
            except stop_exceptions:
                raise
            except Exception as exc:
                save("failed", {"failed_section": index, "error": str(exc)})
                update_stage(directory, "video", "failed", done=len(completed), total=total, engine=engine,
                             message=f"Failed at section {index}/{total}: {str(exc)[:200]}")
                raise
            warnings.extend(result.get("warnings", []))
            completed[index] = {"path": result["path"], "duration": result["duration"],
                                "burned": result["burned"], "pages": result["pages"], "engine": engine}
            for leftover in (raw,):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            state = save("running")
            update_stage(directory, "video", "running", done=len(completed), total=total, engine=engine,
                         message=f"Section {index}/{total} finished: {destination.name}")
            say(f"Completed video section {index}/{total}", len(completed))
            check(after_section=True)
    except stop_exceptions:
        stopped = True

    if stopped and len(completed) == total and total:
        stopped = False   # Stop arrived after the last section: everything is done, so finish normally
    ordered = [completed[i] for i in sorted(completed)]
    final_clips: list[dict] = []
    subtitle_file: Optional[str] = None
    if not stopped and len(completed) == total and total:
        update_stage(directory, "video", "completed", done=total, total=total, engine=engine,
                     message=f"All {total} section videos finished.")
        update_stage(directory, "combine", "running", done=0, total=1, engine=engine)
        say("Combining section videos into clips…", total)
        final_clips = combine_clips([e["path"] for e in ordered], video_dir, prefix, max_clip_seconds,
                                    [e["duration"] for e in ordered])
        if burn:
            subtitle_file = write_project_srt(directory, sections, ordered, prefix, font_size, width, height, poetry)
        update_stage(directory, "combine", "completed", done=len(final_clips), total=len(final_clips), engine=engine,
                     message=f"{len(final_clips)} clip(s) of at most {max_clip_seconds / 60:g} minutes: "
                             + ", ".join(Path(c["path"]).name for c in final_clips))
    elif stopped:
        update_stage(directory, "video", "stopped", done=len(completed), total=total, engine=engine,
                     message=f"Stopped after section {max(completed) if completed else 0}/{total}. Run again to resume.")
    state = save("stopped" if stopped else "completed", {
        "final_clips": final_clips, "final_video": final_clips[0]["path"] if final_clips else None,
        "subtitle_file": subtitle_file,
    })
    return {
        "status": "stopped" if stopped else "completed",
        "completed_sections": len(completed),
        "total_sections": total,
        "section_videos": [e["path"] for e in ordered],
        "final_clips": final_clips,
        "final_video": final_clips[0]["path"] if final_clips else None,
        "subtitle_file": subtitle_file,
        "checkpoint_file": str(checkpoint_file),
        "output_directory": str(video_dir),
        "warnings": warnings,
    }


def write_project_srt(project_dir: Any, sections: list[str], ordered: list[dict], prefix: str,
                      font_size: int, width: int, height: int, poetry: bool) -> str:
    """One SRT for the whole project (same pages as burned into the section videos)."""
    chars, lines = layout_for(width, height, font_size)
    offset = 0.0
    number = 1
    blocks: list[str] = []
    for text, entry in zip(sections, ordered):
        pages = paginate_text(text, chars, lines, preserve_lines=poetry)
        for page, (start, end) in zip(pages, page_timings(pages, entry["duration"])):
            blocks.append(f"{number}\n{_srt_time(start + offset)} --> {_srt_time(end + offset)}\n{page}")
            number += 1
        offset += entry["duration"]
    destination = video_directory(project_dir) / f"{prefix}_subtitles.srt"
    destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8-sig")
    return str(destination)


def combine_existing_videos(project_dir: Any, max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS) -> dict:
    """Combine the finished section videos of a project (also usable after a Stop)."""
    directory = Path(str(project_dir)).expanduser()
    section_dir = video_section_directory(directory)
    book, part = project_identity(directory)
    prefix = project_prefix(book, part)
    files = sorted(
        (f for f in section_dir.glob(f"{glob_escape(prefix)}_*.mp4") if re.search(r"_(\d+)\.mp4$", f.name)),
        key=lambda f: int(re.search(r"_(\d+)\.mp4$", f.name).group(1)),
    )
    if not files:
        raise RuntimeError("No finished section videos were found in " + str(section_dir))
    update_stage(directory, "combine", "running", done=0, total=1)
    clips = combine_clips(files, video_directory(directory), prefix, max_clip_seconds)
    update_stage(directory, "combine", "completed", done=len(clips), total=len(clips),
                 message=f"{len(clips)} clip(s) from {len(files)} section videos.")
    return {"final_clips": clips, "final_video": clips[0]["path"], "section_videos": [str(f) for f in files]}


__all__ = [
    "DEFAULT_MAX_CLIP_SECONDS", "STAGES", "StageControl", "StageStopped", "get_control",
    "safe_stem", "project_prefix", "section_stem", "project_identity", "prefix_for",
    "audio_directory", "audio_section_path", "find_audio_section", "migrate_legacy_audio",
    "video_directory", "video_section_directory", "video_section_path",
    "get_canonical_sections", "load_matching_sections", "update_stage", "read_pipeline", "format_pipeline_status",
    "probe_media", "paginate_text", "layout_for", "finalize_section_video", "group_by_duration",
    "combine_clips", "split_audio", "generate_in_pieces", "run_video_stage", "combine_existing_videos", "write_project_srt",
]
