from __future__ import annotations

import os
import queue
import hashlib
import json
import math
import re
import shutil
import threading
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr

import pipeline_service as pipeline

from document_service import (
    extract_document as extract_document_content,
)

from ollama_service import (
    check_ollama,
    correct_complete_text,
    detect_chapter_titles,
    format_chapter_report,
    get_last_correction_warnings,
    unload_ollama_model,
)

from text_service import (
    clean_text,
    detect_chapters,
    split_for_tts,
)

from tts_provider import (
    SARVAM_SPEAKERS,
    check_voicestudio,
    list_voicestudio_voices,
    check_voicebox,
    list_voicebox_profiles,
)

from tts_service import (
    clear_control_flags,
    generate_audiobook,
    request_pause,
    request_resume,
    request_stop,
    unload_model,
)
from project_service import (
    list_existing_projects,
    load_project_state,
    open_or_create_project,
    save_project_state,
)

from video_service import (
    DEFAULT_SADTALKER_DIR,
    DEFAULT_SADTALKER_ENV,
    check_sadtalker_installation,
    clear_video_control_flags,
    generate_talking_head_project,
    locate_ffmpeg,
    request_video_pause,
    request_video_resume,
    request_video_stop,
    VideoGenerationStopped,
    concatenate_videos,
)

from gemini_service import (
    DEFAULT_TTS_MODEL as GEMINI_DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE as GEMINI_DEFAULT_TTS_VOICE,
    DEFAULT_TEXT_CORRECTION_MODEL as GEMINI_DEFAULT_TEXT_CORRECTION_MODEL,
    DEFAULT_VEO_MODEL as GEMINI_DEFAULT_VEO_MODEL,
    DEFAULT_VEO_RESOLUTION as GEMINI_DEFAULT_VEO_RESOLUTION,
    DEFAULT_VEO_ASPECT_RATIO as GEMINI_DEFAULT_VEO_ASPECT_RATIO,
    check_gemini,
    generate_tts_sections,
    create_replicated_voice,
    correct_hindi_text,
    generate_veo_video,
    mux_external_audio,
)

from remote_colab_service import (
    DEFAULT_COLAB_URL,
    check_colab_server,
    list_fish_references,
    fish_speech_tts,
    generate_wan_remote,
    generate_ltx_remote,
)

from ltx23_comfyui_service import (
    DEFAULT_LTX23_BRIDGE_URL,
    DEFAULT_POSE_WORKFLOW_PATH,
    check_ltx23_bridge,
    generate_ltx23_video,
)


def refresh_voicestudio_voice_choices():
    try:
        voices = list_voicestudio_voices()
        choices = ["default"] + [v["id"] for v in voices if v.get("id") and v["id"] != "default"]
        return gr.Dropdown(choices=choices, value=choices[0], label="VoiceStudio voice/profile ID")
    except Exception as exc:
        return gr.Dropdown(choices=["default"], value="default", label=f"VoiceStudio voice/profile ID (API unavailable: {exc})")

def refresh_voicebox_profile_choices():
    try:
        profiles = list_voicebox_profiles()
        choices = [p["id"] for p in profiles if p.get("id")]
        if not choices:
            choices = [""]
        labels = {p["id"]: f"{p.get('name') or p['id']} [{p.get('language') or 'unknown'}]" for p in profiles if p.get("id")}
        return gr.Dropdown(choices=[(labels.get(x, x), x) for x in choices], value=choices[0], label="Voicebox voice profile ID")
    except Exception as exc:
        return gr.Dropdown(choices=[""], value="", label=f"Voicebox profile (API unavailable: {exc})", allow_custom_value=True)


# ================================================================
# Defaults
# ================================================================

DEFAULT_REFERENCE_TRANSCRIPT = (
    "बूढ़े हाथी ने यह देखा और बात उसकी समझ में आ गई, पर उसने किसी से कहा नहीं। जंगल में वह सबसे बड़ा था"
)

DEFAULT_TESSDATA_DIRECTORY = (
    r"C:\Program Files\Tesseract-OCR\tessdata"
)

DEFAULT_OLLAMA_MODEL = (
    "qwen3:1.7b"
)


def _clock(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timing(started, completed, total):
    elapsed = max(0.0, time.monotonic() - started)
    if completed > 0 and total > completed:
        remaining = elapsed / completed * (total - completed)
        finish = (datetime.now() + timedelta(seconds=remaining)).strftime("%I:%M %p")
        return _clock(elapsed), _clock(remaining), finish
    if completed >= total and total:
        return _clock(elapsed), "00:00:00", datetime.now().strftime("%I:%M %p")
    return _clock(elapsed), "Calculating after first section…", "Calculating…"


def _wav_seconds(path):
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / max(1, wav.getframerate())
    except Exception:
        return 0.0


def prepare_project(project_name, book_name, part_number, author_name):
    try:
        return open_or_create_project(project_name, book_name, part_number, author_name)
    except Exception as exc:
        raise gr.Error(f"Could not open/create project:\n{exc}") from exc


MIN_TTS_SECTION_CHARS = 300

def _safe_max_characters(value, default=1200):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = int(default)
    return max(MIN_TTS_SECTION_CHARS, min(value, 2000))


NEW_PROJECT_CHOICE = "➕ New Project"


def refresh_project_list(current_selection=None):
    choices = [NEW_PROJECT_CHOICE, *list_existing_projects()]
    selected = current_selection if current_selection in choices else NEW_PROJECT_CHOICE
    return gr.Dropdown(choices=choices, value=selected)


def open_selected_project(selection):
    if not selection or selection == NEW_PROJECT_CHOICE:
        return (
            "", "", "1", "", "New project selected. Enter its details and click Create Project.", "",
            "Poetry", "", "Upload a document to begin.", "", "", "", None,
            DEFAULT_REFERENCE_TRANSCRIPT, "Ready", None, None, None, 0,
            None, "Video not started.", None, None, None, 0, "poet_reciting_poem_1080p.mp4",
            1200, "f5tts", "default", "", "hi", "0.6B", "qwen", "shubh", "",
            "", DEFAULT_COLAB_URL, "", "auto", 0.95, 750, 450, 1234, 32, True,
            DEFAULT_SADTALKER_DIR, DEFAULT_SADTALKER_ENV, "crop", 256, True, "1920x1080",
            True, "Nirmala UI", 28, "Crossfade", 0.4,
            "Ready", "", "t2v-A14B", "", "", "480*832", 17, 8, -1, 1,
            "Ready", "t2v_i2v_single", "", 5.0, 24,
        )
    try:
        state = load_project_state(selection)
    except Exception as exc:
        raise gr.Error(f"Could not open project:\n{exc}") from exc
    wav_file = state.get("wav_file")
    final_video = state.get("final_video")
    project_message = (
        f"Existing project opened.\nProject: {state.get('project_name', selection)}\n"
        f"Book: {state.get('book_name', '')}\nPart: {state.get('part_number', '')}\n"
        f"Author: {state.get('author_name', '')}\nOutput: {state.get('output_directory', '')}"
    )
    return (
        state.get("project_name", selection), state.get("book_name", ""), state.get("part_number", "1"), state.get("author_name", ""),
        project_message, state.get("output_directory", ""), state.get("book_type", "Poetry"), state.get("extracted_text", ""),
        state.get("extraction_status", "No saved extraction status."), state.get("text_status", "No saved text-processing status."),
        state.get("ollama_status", "No saved Ollama status."), state.get("section_preview", ""), state.get("reference_audio"),
        state.get("reference_transcript") or DEFAULT_REFERENCE_TRANSCRIPT, state.get("audio_status", "Audio not started."), wav_file, wav_file,
        state.get("mp3_file"), float(state.get("audio_progress", 0)), state.get("poet_image"), state.get("video_status", "Video not started."),
        final_video, ([final_video] if final_video else None), state.get("subtitle_file"), float(state.get("video_progress", 0)), state.get("final_video_filename", "poet_reciting_poem_1080p.mp4"),
        _safe_max_characters(state.get("max_characters", 1200)), state.get("tts_engine", "f5tts"), state.get("voicestudio_voice", "default"),
        state.get("voicebox_profile", ""), state.get("voicebox_language", "hi"), state.get("voicebox_model_size", "0.6B"), state.get("voicebox_engine", "qwen"),
        state.get("sarvam_speaker", "shubh"), "", state.get("fish_reference_id", ""), state.get("colab_url", DEFAULT_COLAB_URL), "",
        state.get("device_choice", "auto"), float(state.get("speech_speed", 0.95)), int(state.get("pause_duration", 750)), int(state.get("poetry_line_pause_duration", 450)),
        int(state.get("seed", 1234)), int(state.get("nfe_steps", 32)), bool(state.get("create_mp3", True)), state.get("sadtalker_directory", DEFAULT_SADTALKER_DIR),
        state.get("sadtalker_environment", DEFAULT_SADTALKER_ENV), state.get("preprocess_mode", "crop"), int(state.get("video_model_size", 256)),
        bool(state.get("still_mode", True)), state.get("output_resolution", "1920x1080"), bool(state.get("burn_subtitles", True)),
        state.get("subtitle_font", "Nirmala UI"), int(state.get("subtitle_font_size", 28)), state.get("video_transition", "Crossfade"), float(state.get("video_transition_duration", 0.4)),
        "Ready", state.get("wan_checkpoint", ""), state.get("wan_task", "t2v-A14B"), state.get("wan_prompt", ""), state.get("wan_status", ""),
        state.get("wan_size", "480*832"), int(state.get("wan_frame_num", 17)), int(state.get("wan_steps", 8)), int(state.get("wan_seed", -1)), int(state.get("wan_audio_section", 1)),
        state.get("ltx_status", "Ready"), state.get("ltx_mode", "t2v_i2v_single"), state.get("ltx_prompt", ""), float(state.get("ltx_duration", 5.0)), int(state.get("ltx_fps", 24)),
    )


def persist_project_tabs(
    output_directory, extracted_text, book_type, extraction_status, text_status, ollama_status, section_preview, reference_audio, reference_transcript,
    poet_image, final_video_filename, max_characters, tts_engine, voicestudio_voice, voicebox_profile, voicebox_language, voicebox_model_size, voicebox_engine,
    sarvam_speaker, sarvam_api_key, fish_reference_id, colab_url, colab_token, device_choice, speech_speed, pause_duration, poetry_line_pause_duration,
    seed, nfe_steps, create_mp3, sadtalker_directory, sadtalker_environment, preprocess_mode, video_model_size, still_mode, output_resolution, burn_subtitles,
    subtitle_font, subtitle_font_size, video_transition, video_transition_duration, wan_checkpoint, wan_task, wan_prompt, wan_size, wan_frame_num, wan_steps, wan_seed, wan_audio_section,
    ltx_mode, ltx_prompt, ltx_duration, ltx_fps,
):
    updates = {
        "extracted_text": str(extracted_text or ""), "book_type": str(book_type or "Others"), "extraction_status": str(extraction_status or ""),
        "text_status": str(text_status or ""), "ollama_status": str(ollama_status or ""), "section_preview": str(section_preview or ""),
        "reference_audio": str(reference_audio) if reference_audio else None, "reference_transcript": str(reference_transcript or ""),
        "poet_image": str(poet_image) if poet_image else None, "final_video_filename": str(final_video_filename or ""), "max_characters": int(max_characters),
        "tts_engine": str(tts_engine or "f5tts"), "voicestudio_voice": str(voicestudio_voice or "default"), "voicebox_profile": str(voicebox_profile or ""),
        "voicebox_language": str(voicebox_language or "hi"), "voicebox_model_size": str(voicebox_model_size or "0.6B"), "voicebox_engine": str(voicebox_engine or "qwen"),
        "sarvam_speaker": str(sarvam_speaker or "shubh"), "sarvam_api_key": "", "fish_reference_id": str(fish_reference_id or ""),
        "colab_url": str(colab_url or DEFAULT_COLAB_URL), "colab_token": "", "device_choice": str(device_choice or "auto"),
        "speech_speed": float(speech_speed), "pause_duration": int(pause_duration), "poetry_line_pause_duration": int(poetry_line_pause_duration), "seed": int(seed),
        "nfe_steps": int(nfe_steps), "create_mp3": bool(create_mp3), "sadtalker_directory": str(sadtalker_directory or DEFAULT_SADTALKER_DIR),
        "sadtalker_environment": str(sadtalker_environment or DEFAULT_SADTALKER_ENV), "preprocess_mode": str(preprocess_mode or "crop"), "video_model_size": int(video_model_size),
        "still_mode": bool(still_mode), "output_resolution": str(output_resolution or "1920x1080"), "burn_subtitles": bool(burn_subtitles),
        "subtitle_font": str(subtitle_font or "Nirmala UI"), "subtitle_font_size": int(subtitle_font_size), "video_transition": str(video_transition or "None"),
        "video_transition_duration": float(video_transition_duration), "wan_checkpoint": str(wan_checkpoint or ""), "wan_task": str(wan_task or "t2v-A14B"),
        "wan_prompt": str(wan_prompt or ""), "wan_size": str(wan_size or "480*832"), "wan_frame_num": int(wan_frame_num), "wan_steps": int(wan_steps),
        "wan_seed": int(wan_seed), "wan_audio_section": int(wan_audio_section), "ltx_mode": str(ltx_mode or "t2v_i2v_single"), "ltx_prompt": str(ltx_prompt or ""),
        "ltx_duration": float(ltx_duration), "ltx_fps": int(ltx_fps),
    }
    if str(output_directory or "").strip():
        project_directory = Path(str(output_directory)).expanduser().resolve()
        if reference_audio:
            try: updates["reference_audio"] = _copy_project_asset(reference_audio, project_directory, "reference_audio")
            except (OSError, ValueError): pass
        if poet_image:
            try: updates["poet_image"] = _copy_project_asset(poet_image, project_directory, "poet_portrait")
            except (OSError, ValueError): pass
    save_project_state(output_directory, updates)


def _copy_project_asset(source, project_directory: Path, stem: str) -> str:
    source_path = Path(str(source)).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Uploaded asset was not found: {source_path}")
    suffix = source_path.suffix.lower() or ".bin"
    assets_directory = project_directory / "assets"
    assets_directory.mkdir(parents=True, exist_ok=True)
    destination = assets_directory / f"{stem}{suffix}"
    if source_path != destination.resolve():
        shutil.copy2(source_path, destination)
    return str(destination.resolve())


# ================================================================
# Text helpers
# ================================================================

def remove_page_markers(
    text: str,
) -> str:
    """
    Remove internal PDF/Word location markers before Ollama or TTS.
    """

    lines = []

    for line in str(text).splitlines():
        stripped = line.strip()

        if re.fullmatch(
            r"=+\s*(PDF PAGE|WORD BLOCK|"
            r"WORD EMBEDDED IMAGE)\s+\d+\s*=+",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue

        lines.append(line)

    return "\n".join(lines)


# ================================================================
# PDF and Word extraction
# ================================================================

def extract_uploaded_document(
    uploaded_file,
    selection_range,
    ocr_language,
    ocr_dpi,
    include_word_image_ocr,
    progress=gr.Progress(),
):
    if not uploaded_file:
        raise gr.Error(
            "Please upload a PDF or Word document."
        )

    document_path = Path(
        uploaded_file
    )

    if not document_path.exists():
        raise gr.Error(
            "The uploaded document was not found:\n"
            f"{document_path}"
        )

    extension = (
        document_path.suffix.lower()
    )

    if extension not in {
        ".pdf",
        ".docx",
        ".doc",
    }:
        raise gr.Error(
            "Upload a PDF, DOCX or DOC document."
        )

    control = pipeline.get_control("extract")
    control.clear()
    cache_file = _cache_path(
        "extract", _file_digest(document_path), selection_range or "", ocr_language or "hin+eng",
        int(ocr_dpi), bool(include_word_image_ocr),
    )
    saved = _cache_read(cache_file)
    results_done = list(saved.get("results") or [])
    done_numbers = {item.get("number") for item in results_done if item.get("number") is not None}
    new_results = []

    def extraction_progress(
        current,
        total,
        location,
    ):
        control.checkpoint()  # Pause / Stop act only between pages or blocks
        progress(
            current / max(1, total),
            desc=(
                f"Reading {location} "
                f"({current}/{total})"
            ),
        )

    def keep_result(entry):
        new_results.append(entry)
        _cache_write(cache_file, {"results": results_done + new_results})

    stopped = False
    fresh = []
    try:
        fresh = extract_document_content(
            document_path=document_path,
            selection_range=(
                selection_range or ""
            ),
            ocr_language=(
                ocr_language or "hin+eng"
            ),
            dpi=int(ocr_dpi),
            include_word_image_ocr=bool(
                include_word_image_ocr
            ),
            progress_callback=(
                extraction_progress
            ),
            skip_numbers=done_numbers or None,
            on_result=keep_result,
        )

    except pipeline.StageStopped:
        stopped = True

    except Exception as error:
        raise gr.Error(
            "Document extraction failed:\n"
            f"{error}"
        ) from error

    if stopped:
        finished = len(results_done) + len(new_results)
        return (
            gr.update(),
            f"Extraction stopped after the current page/block ({finished} finished). "
            "Progress is saved. Press 'Extract document text' again with the same file "
            "and settings to resume from the next page/block.",
        )

    results = results_done + list(fresh)
    _cache_clear(cache_file)

    if not results:
        raise gr.Error(
            "No text was extracted from the document."
        )

    extracted_parts = []
    method_counts = {}

    for result in results:
        location = str(
            result.get(
                "location",
                "Document content",
            )
        )

        method = str(
            result.get(
                "method",
                "Text extraction",
            )
        )

        content = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        if not content:
            continue

        method_counts[method] = (
            method_counts.get(
                method,
                0,
            )
            + 1
        )

        extracted_parts.append(
            f"===== {location.upper()} =====\n"
            f"{content}"
        )

    extracted_text = "\n\n".join(
        extracted_parts
    ).strip()

    if not extracted_text:
        raise gr.Error(
            "Document processing completed, but "
            "no readable text was found."
        )

    status_lines = [
        "Document extraction completed.",
        f"File type: {extension}",
        (
            f"Content blocks: "
            f"{len(extracted_parts)}"
        ),
        (
            f"Characters extracted: "
            f"{len(extracted_text):,}"
        ),
    ]

    for method, count in (
        method_counts.items()
    ):
        status_lines.append(
            f"{method}: {count}"
        )

    status_lines.append(
        "\nReview and correct the extracted text."
    )

    return (
        extracted_text,
        "\n".join(status_lines),
    )


# ================================================================
# Ordinary text processing
# ================================================================

def prepare_clean_text(
    extracted_text,
):
    if not extracted_text:
        raise gr.Error(
            "There is no text to clean."
        )

    if not str(
        extracted_text
    ).strip():
        raise gr.Error(
            "The extracted text is empty."
        )

    cleaned = remove_page_markers(
        extracted_text
    )

    cleaned = clean_text(
        cleaned
    )

    status = (
        "Text cleaned successfully.\n"
        f"Characters: {len(cleaned):,}\n"
        "PDF/Word location markers were removed."
    )

    return cleaned, status


def analyse_chapters(
    corrected_text,
    maximum_characters,
    book_type,
):
    if not corrected_text:
        raise gr.Error(
            "There is no text to analyse."
        )

    corrected_text = remove_page_markers(
        corrected_text
    )

    corrected_text = clean_text(
        corrected_text
    )

    if not corrected_text:
        raise gr.Error(
            "There is no usable text to analyse."
        )

    chapters = detect_chapters(
        corrected_text
    )

    all_sections = []
    preview_parts = []

    for chapter_number, chapter in enumerate(
        chapters,
        start=1,
    ):
        chapter_title = chapter.get(
            "title",
            f"Chapter {chapter_number}",
        )

        chapter_text = chapter.get(
            "text",
            "",
        )

        sections = split_for_tts(
            chapter_text,
            int(maximum_characters),
            book_type=book_type or "Others",
        )

        preview_parts.append(
            f"========== {chapter_title} =========="
        )

        for section in sections:
            all_sections.append(
                section
            )

            preview_parts.append(
                f"\n--- SECTION "
                f"{len(all_sections):04d} ---\n"
                f"{section}"
            )

    preview = "\n\n".join(
        preview_parts
    )

    largest_section = max(
        (
            len(section)
            for section in all_sections
        ),
        default=0,
    )

    status = (
        "Text analysis completed.\n"
        f"Chapters detected: {len(chapters)}\n"
        f"TTS sections prepared: "
        f"{len(all_sections)}\n"
        f"Largest section: "
        f"{largest_section} characters\n"
        + (
            f"Drama target: {int(maximum_characters)} characters; "
            "complete dialogues are preserved up to 2000 characters."
            if str(book_type or "Others").strip().casefold() == "drama"
            else f"Maximum characters per section: {int(maximum_characters)}"
        )
    )

    return preview, status


# ================================================================
# Ollama correction
# ================================================================

def check_ollama_from_ui():
    ready, message = check_ollama()
    return message


def ollama_correct_text(
    extracted_text,
    ollama_model,
    correction_language,
    book_type,
    correction_chunk_size=700,
    progress=gr.Progress(track_tqdm=False),
):
    text = (extracted_text or "").strip()

    if not text:
        raise gr.Error("There is no extracted text to correct.")

    model = (ollama_model or "qwen3:1.7b").strip()

    try:
        chunk_size = int(correction_chunk_size or 700)
    except (TypeError, ValueError):
        chunk_size = 700

    chunk_size = max(500, min(chunk_size, 1200))

    progress(
        0.01,
        desc="Releasing F5-TTS GPU memory…",
    )

    # The RTX 3050 has 4 GB VRAM.  Keeping F5-TTS loaded forces Ollama to CPU.
    try:
        unload_model()
    except Exception:
        pass

    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    progress(0.03, desc="Starting Ollama correction…")

    progress_state = {"pulse": 0.04}

    def update_progress(message, current=None, total=None):
        if current is not None and total:
            section_fraction = min(
                max(float(current) / float(total), 0.0),
                1.0,
            )
            fraction = 0.04 + (0.92 * section_fraction)
            progress_state["pulse"] = max(progress_state["pulse"], fraction)
        else:
            # Streaming Ollama calls report token activity without a known total.
            progress_state["pulse"] += 0.015
            if progress_state["pulse"] > 0.90:
                progress_state["pulse"] = 0.12
            fraction = progress_state["pulse"]

        progress(
            fraction,
            desc=str(message),
        )

    control = pipeline.get_control("correct")
    control.clear()
    cache_file = _cache_path("ollama", text, model, correction_language, book_type, chunk_size)
    saved = _cache_read(cache_file)
    resume_chunks = {int(k): v for k, v in (saved.get("chunks") or {}).items()}
    chunk_state = {"count": int(saved.get("total") or 0)}

    def save_chunk(index, count, corrected_text):
        resume_chunks[index] = corrected_text
        chunk_state["count"] = count
        _cache_write(cache_file, {"chunks": {str(k): v for k, v in resume_chunks.items()}, "total": count})

    try:
        corrected = correct_complete_text(
            text=text,
            model=model,
            language=correction_language or "Hindi/English",
            book_type=book_type or "Others",
            chunk_size=chunk_size,
            progress_callback=update_progress,
            timeout=180,
            continue_on_error=True,
            control=lambda: control.checkpoint(lambda message: progress(progress_state["pulse"], desc=message)),
            resume_chunks=resume_chunks,
            on_chunk=save_chunk,
        )
        _cache_clear(cache_file)

        warnings = get_last_correction_warnings()

        if warnings:
            warning_report = "\n".join(
                f"• {warning}" for warning in warnings
            )

            status = (
                "Correction completed with warnings.\n\n"
                f"{warning_report}"
            )
        else:
            status = "Ollama correction completed successfully."

        progress(
            1.0,
            desc="Correction completed.",
        )

        return corrected, status

    except pipeline.StageStopped:
        return gr.update(), (
            f"Correction stopped after the current section ({len(resume_chunks)}/{chunk_state['count'] or '?'} done). "
            "The text above is unchanged. Press the correct button again to resume from the next section."
        )

    except Exception as exc:
        raise gr.Error(
            "Ollama correction failed:\n"
            f"{exc}"
        ) from exc


# ================================================================
# Gemini text correction
# ================================================================

def gemini_correct_text_ui(text, model, language, book_type, chunk_size, api_key, progress=gr.Progress()):
    if not str(text or '').strip():
        raise gr.Error('There is no text to correct.')
    if not str(api_key or '').strip() and not os.getenv('GEMINI_API_KEY'):
        raise gr.Error('Enter a Gemini API key or set GEMINI_API_KEY in Windows environment variables.')
    def correction_progress(current, total, description):
        progress(current / max(1, total), desc=description)

    control = pipeline.get_control("correct")
    control.clear()
    cache_file = _cache_path("gemini", text, model, language, book_type, chunk_size)
    saved = _cache_read(cache_file)
    resume_chunks = {int(k): v for k, v in (saved.get("chunks") or {}).items()}
    chunk_state = {"count": int(saved.get("total") or 0)}

    def save_chunk(index, count, corrected_text):
        resume_chunks[index] = corrected_text
        chunk_state["count"] = count
        _cache_write(cache_file, {"chunks": {str(k): v for k, v in resume_chunks.items()}, "total": count})

    try:
        corrected, warnings = correct_hindi_text(
            text, model=str(model or GEMINI_DEFAULT_TEXT_CORRECTION_MODEL),
            language=str(language or 'Hindi/English'), book_type=str(book_type or 'Others'),
            chunk_size=int(chunk_size or 1200), api_key=str(api_key or '').strip() or None,
            progress=correction_progress,
            control=lambda: control.checkpoint(lambda message: progress(0.5, desc=message)),
            resume_chunks=resume_chunks, on_chunk=save_chunk,
        )
        _cache_clear(cache_file)
    except pipeline.StageStopped:
        return gr.update(), (
            f"Gemini correction stopped after the current section ({len(resume_chunks)}/{chunk_state['count'] or '?'} done). "
            "The text above is unchanged. Press the correct button again to resume."
        )
    except Exception as exc:
        raise gr.Error(f'Gemini text correction failed:\n{exc}') from exc
    status = 'Gemini correction completed successfully.' if not warnings else 'Gemini correction completed with warnings.\n\n' + '\n'.join('• '+w for w in warnings)
    return corrected, status

# ================================================================
# Ollama chapter detection
# ================================================================

def ollama_detect_chapters(
    corrected_text,
    ollama_model,
    book_type,
    progress=gr.Progress(),
):
    if not corrected_text:
        raise gr.Error(
            "There is no text to analyse."
        )

    text_for_analysis = remove_page_markers(
        corrected_text
    ).strip()

    if not text_for_analysis:
        raise gr.Error(
            "There is no usable text to analyse."
        )

    try:
        unload_model()

    except Exception:
        pass

    def update_progress(
        message,
        current=None,
        total=None,
    ):
        progress(
            (
                float(current) / max(1.0, float(total))
                if current is not None and total
                else 0.05
            ),
            desc=str(message),
        )

    try:
        chapter_data = detect_chapter_titles(
            text=text_for_analysis,
            model=ollama_model,
            chunk_size=1800,
            progress_callback=(
                update_progress
            ),
            book_type=book_type or "Others",
        )

        report = format_chapter_report(
            chapter_data
        )

    except Exception as error:
        raise gr.Error(
            "Ollama chapter detection failed:\n"
            f"{error}"
        ) from error

    return report


def unload_ollama_from_ui(
    ollama_model,
):
    _success, message = unload_ollama_model(
        ollama_model
    )
    return message


# ================================================================
# Gemini cloud audio/video
# ================================================================

def check_gemini_from_ui(api_key):
    return check_gemini(str(api_key or '').strip() or None)

def start_gemini_audio(corrected_text, project_name, book_name, part_number, author_name, book_type, api_key, tts_model, tts_voice, style_prompt, chunk_size, replicated_voice_id=""):
    """Gemini TTS, section by section, into the shared project audio folder (<book>_<part>_<nnn>.wav)."""
    if not str(corrected_text or '').strip():
        raise gr.Error('Corrected/extracted text is empty.')
    if not str(project_name or '').strip():
        raise gr.Error('Open/create a project first.')
    if not str(book_name or '').strip() or not str(part_number or '').strip():
        raise gr.Error('Book name and Part number are required.')
    if not str(api_key or '').strip() and not os.getenv('GEMINI_API_KEY'):
        raise gr.Error('Enter a Gemini API key or set GEMINI_API_KEY in Windows environment variables.')
    _msg, project_dir_text = open_or_create_project(project_name, book_name, part_number, author_name)
    project_dir = Path(project_dir_text)
    sections = _video_sections(project_dir, corrected_text, max(500, min(int(chunk_size or 1800), 3000)), book_type or 'Others')
    if not sections:
        raise gr.Error('No audio sections could be prepared.')
    pipeline.migrate_legacy_audio(project_dir)
    control = pipeline.get_control('gemini')
    control.clear()
    total = len(sections)
    events = queue.Queue()
    box = {}
    pipeline.update_stage(project_dir, 'audio', 'running', done=0, total=total, engine='gemini', message='Gemini TTS started.')

    def notify(message):
        events.put(str(message))

    def on_section(index, count, path):
        pipeline.update_stage(project_dir, 'audio', 'running', done=index, total=count, engine='gemini',
                              message=f'Section {index}/{count} finished: {Path(path).name}')
        events.put(f'Completed audio section {index}/{count}: {Path(path).name}')

    def worker():
        try:
            box['result'] = generate_tts_sections(
                sections, project_dir, voice=str(tts_voice or GEMINI_DEFAULT_TTS_VOICE),
                model=str(tts_model or GEMINI_DEFAULT_TTS_MODEL),
                style_prompt=str(style_prompt or '').strip() or 'Natural Hindi audiobook narration. Clear diction, warm, expressive, steady pace.',
                api_key=str(api_key or '').strip() or None, replicated_voice_id=str(replicated_voice_id or ''),
                progress=lambda index, count, message: events.put(str(message)),
                file_for=lambda index: pipeline.audio_section_path(project_dir, index),
                combined_path=project_dir / f'{pipeline.prefix_for(project_dir)}_gemini_audiobook.wav',
                control=lambda: control.checkpoint(notify), on_section=on_section)
        except pipeline.StageStopped:
            box['stopped'] = True
        except BaseException as exc:
            box['error'] = exc
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    started = time.monotonic()
    yield 'Gemini TTS started...', None
    while True:
        try:
            message = events.get(timeout=1.0)
        except queue.Empty:
            continue
        if message is None:
            break
        yield f"{message}\nElapsed: {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - started))}", gr.update()
    if 'error' in box:
        pipeline.update_stage(project_dir, 'audio', 'failed', total=total, engine='gemini', message=str(box['error'])[:300])
        raise gr.Error(f"Gemini TTS failed:\n{box['error']}") from box['error']
    if box.get('stopped'):
        done_files = sorted(p for p in pipeline.audio_directory(project_dir).glob(f'{pipeline.prefix_for(project_dir)}_*.wav'))
        pipeline.update_stage(project_dir, 'audio', 'stopped', done=len(done_files), total=total, engine='gemini',
                              message=f'Stopped after section {len(done_files)}/{total}. Generate again to resume.')
        yield (f'Gemini TTS stopped after the current section ({len(done_files)}/{total} done). '
               'Press Generate again to resume from the next section.'), None
        return
    section_files, combined_files = box['result']
    combined = combined_files[0] if combined_files else None
    pipeline.update_stage(project_dir, 'audio', 'completed', done=total, total=total, engine='gemini',
                          message=f'All {total} section audio files are in sections/.')
    status = f'Gemini TTS completed. {len(section_files)} section(s).\nOutput: {combined}'
    save_project_state(project_dir, {'gemini_audio_status': status, 'gemini_tts_model': str(tts_model or GEMINI_DEFAULT_TTS_MODEL), 'gemini_tts_voice': str(tts_voice or GEMINI_DEFAULT_TTS_VOICE), 'gemini_audio_file': str(combined) if combined else None})
    yield status, str(combined) if combined else None


def start_gemini_video(corrected_text, project_name, book_name, part_number, author_name, api_key, veo_model, resolution, aspect_ratio, prompt, reference_image, generated_audio=None,
                       burn_text=True, output_resolution="1920x1080", subtitle_font="Nirmala UI", subtitle_font_size=28, max_clip_minutes=10.0):
    """Gemini Veo, section by section; the section audio replaces Veo's audio, the text is burned in."""
    if not str(project_name or '').strip():
        raise gr.Error('Open/create a project first.')
    if not str(book_name or '').strip() or not str(part_number or '').strip():
        raise gr.Error('Book name and Part number are required.')
    if not str(api_key or '').strip() and not os.getenv('GEMINI_API_KEY'):
        raise gr.Error('Enter a Gemini API key or set GEMINI_API_KEY in Windows environment variables.')
    _msg, project_dir_text = open_or_create_project(project_name, book_name, part_number, author_name)
    project_dir = Path(project_dir_text)
    sections = _video_sections(project_dir, corrected_text)
    if not sections:
        raise gr.Error('No video sections could be prepared.')
    image_copy = _copy_project_asset(reference_image, project_dir, 'gemini_veo_reference') if reference_image else ''
    settings = {'model': str(veo_model or GEMINI_DEFAULT_VEO_MODEL), 'resolution': str(resolution or '720p'),
                'aspect_ratio': str(aspect_ratio or '16:9'), 'prompt': str(prompt or ''), 'image': image_copy,
                'generated_audio': str(generated_audio or '')}
    base_prompt = str(prompt or '').strip()

    def generate_one(index, section_text, destination, audio_file):
        prompt_text = base_prompt or ('Create a cinematic historical scene inspired by this book excerpt. No on-screen text; realistic movement and lighting.\n' + section_text[:2500])
        result = generate_veo_video(prompt_text, destination, model=str(veo_model or GEMINI_DEFAULT_VEO_MODEL), resolution=str(resolution or '720p'), aspect_ratio=str(aspect_ratio or '16:9'), api_key=str(api_key or '').strip() or None, image_path=image_copy or None, progress=lambda *args: None)
        if audio_file is None and generated_audio:  # no section audio: use the single uploaded audio instead
            muxed = destination.with_name(destination.stem + '_audio.mp4')
            mux_external_audio(result, str(generated_audio), muxed)
            muxed.replace(destination)
        return destination

    def runner(notify):
        return _remote_video_sections(
            engine='gemini_veo', project_dir=project_dir, sections=sections, settings=settings, generator=generate_one,
            mux_audio=True, require_audio=False, burn=burn_text, resolution=output_resolution,
            font_name=subtitle_font, font_size=subtitle_font_size, max_clip_minutes=max_clip_minutes, notify=notify)

    def finalize(result):
        preview, downloads = _video_outputs(result)
        status = _video_status_text('Gemini Veo', result, 'The section audio replaces Veo audio; Veo does not lip-sync to it.')
        save_project_state(project_dir, {'gemini_video_status': status, 'gemini_veo_model': str(veo_model or GEMINI_DEFAULT_VEO_MODEL), 'gemini_video_file': preview})
        return status, preview, downloads

    yield from _live_video_job('Gemini Veo', runner, finalize)


# ================================================================
# Installation check
# ================================================================

def check_installation():
    import sys

    results = [
        f"Python: {sys.version.split()[0]}"
    ]

    ffmpeg_path = locate_ffmpeg()

    results.append(
        (
            f"FFmpeg: Ready\n{ffmpeg_path}"
            if ffmpeg_path
            else "FFmpeg: NOT FOUND"
        )
    )

    _sadtalker_ready, sadtalker_message = (
        check_sadtalker_installation(
            sadtalker_directory=DEFAULT_SADTALKER_DIR,
            environment_name=DEFAULT_SADTALKER_ENV,
        )
    )
    results.append(
        "SadTalker: " + sadtalker_message
    )

    tesseract_path = shutil.which(
        "tesseract"
    )

    if not tesseract_path:
        default_tesseract = Path(
            r"C:\Program Files"
            r"\Tesseract-OCR"
            r"\tesseract.exe"
        )

        if default_tesseract.exists():
            tesseract_path = str(
                default_tesseract
            )

    results.append(
        (
            f"Tesseract: Ready\n"
            f"{tesseract_path}"
            if tesseract_path
            else "Tesseract: NOT FOUND"
        )
    )

    tessdata_path = Path(
        DEFAULT_TESSDATA_DIRECTORY
    )

    for language_file, label in [
        (
            "hin.traineddata",
            "Hindi OCR",
        ),
        (
            "eng.traineddata",
            "English OCR",
        ),
        (
            "osd.traineddata",
            "OCR orientation",
        ),
    ]:
        available = (
            tessdata_path
            / language_file
        ).exists()

        results.append(
            f"{label}: "
            + (
                "Ready"
                if available
                else "NOT FOUND"
            )
        )

    try:
        import fitz

        results.append(
            "PDF reader: Ready"
        )

    except Exception as error:
        results.append(
            f"PDF reader error: {error}"
        )

    try:
        import docx

        results.append(
            "Word DOCX reader: Ready"
        )

    except Exception as error:
        results.append(
            f"Word DOCX reader error: {error}"
        )

    try:
        import pytesseract
        from PIL import Image

        results.append(
            "Word image OCR: Ready"
        )

    except Exception as error:
        results.append(
            f"Word image OCR error: {error}"
        )

    libreoffice_candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        (
            r"C:\Program Files"
            r"\LibreOffice"
            r"\program"
            r"\soffice.exe"
        ),
        (
            r"C:\Program Files (x86)"
            r"\LibreOffice"
            r"\program"
            r"\soffice.exe"
        ),
    ]

    libreoffice_path = None

    for candidate in (
        libreoffice_candidates
    ):
        if not candidate:
            continue

        candidate_path = Path(
            candidate
        )

        if candidate_path.exists():
            libreoffice_path = (
                candidate_path
            )
            break

    if libreoffice_path:
        results.append(
            "Legacy DOC conversion: Ready\n"
            f"{libreoffice_path}"
        )
    else:
        results.append(
            "Legacy DOC conversion: LibreOffice "
            "not found. PDF and DOCX still work."
        )

    try:
        import torch

        results.append(
            f"PyTorch: {torch.__version__}"
        )

        cuda_available = (
            torch.cuda.is_available()
        )

        results.append(
            f"CUDA available: {cuda_available}"
        )

        if cuda_available:
            results.append(
                "GPU: "
                + torch.cuda.get_device_name(
                    0
                )
            )

            gpu_memory = (
                torch.cuda
                .get_device_properties(0)
                .total_memory
                / 1024**3
            )

            results.append(
                f"GPU memory: "
                f"{gpu_memory:.2f} GB"
            )

    except Exception as error:
        results.append(
            f"PyTorch error: {error}"
        )

    try:
        import f5_tts

        results.append(
            "F5-TTS: Installed"
        )

    except Exception as error:
        results.append(
            f"F5-TTS error: {error}"
        )

    try:
        results.append("Colab media server:\n" + check_colab_server(DEFAULT_COLAB_URL, ""))
    except Exception as error:
        results.append(f"Colab media server check error: {error}")

    try:
        ollama_ready, ollama_message = check_ollama()
        results.append(
            "Ollama: " + ollama_message
        )

    except Exception as error:
        results.append(
            f"Ollama check error: {error}"
        )

    # Keep the report resilient if a future service check returns a
    # non-string value.
    return "\n\n".join(str(item) for item in results)


# ================================================================
# F5-TTS audiobook generation
# ================================================================

def start_generation(
    corrected_text,
    book_type,
    book_name,
    part_number,
    author_name,
    reference_audio,
    reference_transcript,
    project_name,
    maximum_characters,
    tts_engine,
    voicestudio_voice,
    voicebox_profile,
    voicebox_language,
    voicebox_model_size,
    voicebox_engine,
    sarvam_speaker,
    sarvam_api_key,
    fish_reference_id,
    gemini_api_key,
    gemini_model,
    gemini_voice_id,
    colab_url,
    colab_token,
    speech_speed,
    pause_duration,
    poetry_line_pause_duration,
    device_choice,
    seed,
    nfe_steps,
    create_mp3,
    ollama_model,
):
    if not corrected_text:
        raise gr.Error(
            "Extract and correct the document text first."
        )

    selected_engine = str(tts_engine or "f5tts").strip().lower()
    if selected_engine == "f5tts":
        if not reference_audio:
            raise gr.Error("Upload the reference audio file for F5-Hindi.")
        if not str(reference_transcript or "").strip():
            raise gr.Error("Enter the exact reference audio transcript for F5-Hindi.")
    elif selected_engine == "voicestudio":
        if not reference_audio and not str(voicestudio_voice or "").strip():
            raise gr.Error("Upload a reference audio file or select a VoiceStudio profile ID.")
        if reference_audio and not str(reference_transcript or "").strip():
            raise gr.Error("Enter the exact reference audio transcript for VoiceStudio.")
    elif selected_engine == "voicebox":
        if not str(voicebox_profile or "").strip():
            raise gr.Error("Select a Voicebox voice profile ID.")
    elif selected_engine == "sarvam":
        if not str(sarvam_api_key or "").strip():
            raise gr.Error("Enter the Sarvam API key.")
    elif selected_engine == "fishspeech":
        if not str(colab_url or "").strip():
            raise gr.Error("Enter the Colab server URL for Fish Speech.")
    elif selected_engine == "gemini":
        if not str(gemini_api_key or '').strip() and not os.getenv('GEMINI_API_KEY'):
            raise gr.Error("Enter the Gemini API key.")
        if not str(gemini_voice_id or '').strip():
            raise gr.Error("Create/select a Gemini replicated voice ID first, or enter a prebuilt voice name such as Kore.")

    if not project_name:
        raise gr.Error(
            "Enter a project name."
        )
    if not str(book_name or "").strip() or not str(part_number or "").strip():
        raise gr.Error("Open/create a project with Book name and Part number first.")
    _project_message, project_directory_text = open_or_create_project(
        project_name, book_name, part_number, author_name
    )
    project_directory = Path(project_directory_text)
    if reference_audio:
        try:
            reference_audio = _copy_project_asset(
                reference_audio, project_directory, "reference_audio"
            )
        except Exception as exc:
            raise gr.Error(f"Could not save reference audio in the project:\n{exc}") from exc
    save_project_state(project_directory, {
        "extracted_text": str(corrected_text),
        "book_type": str(book_type or "Others"),
        "tts_engine": selected_engine,
        "voicestudio_voice": str(voicestudio_voice or "default"),
        "voicebox_profile": str(voicebox_profile or ""),
        "voicebox_language": str(voicebox_language or "hi"),
        "voicebox_model_size": str(voicebox_model_size or "0.6B"),
        "voicebox_engine": str(voicebox_engine or "qwen"),
        "sarvam_speaker": str(sarvam_speaker or "shubh"),
        "sarvam_api_key": "",  # API keys are never persisted in project JSON.
        "reference_audio": str(reference_audio) if reference_audio else None,
        "reference_transcript": str(reference_transcript),
        "gemini_tts_model": str(gemini_model or ''),
        "gemini_tts_voice_id": str(gemini_voice_id or ''),
        "audio_status": "Audio generation started.",
        "max_characters": int(maximum_characters),
        "device_choice": str(device_choice or "auto"),
        "speech_speed": float(speech_speed),
        "pause_duration": int(pause_duration),
        "poetry_line_pause_duration": int(poetry_line_pause_duration),
        "seed": int(seed),
        "nfe_steps": int(nfe_steps),
        "create_mp3": bool(create_mp3),
    })

    if (
        str(book_type or "Others").strip().casefold() == "poetry"
        and int(pause_duration) <= int(poetry_line_pause_duration)
    ):
        raise gr.Error(
            "For Poetry, silence between four-line sections must be greater "
            "than silence between individual lines. Recommended: 750 ms "
            "between sections and 450 ms between lines."
        )

    # One canonical section list per project: audio, video and the burned text all use it.
    sections = _video_sections(
        project_directory,
        corrected_text,
        maximum_characters,
        book_type,
    )

    if not sections:
        raise gr.Error(
            "No TTS sections could be prepared."
        )
    section_weights = [max(1, len(section)) for section in sections]

    largest_section = max(
        len(section)
        for section in sections
    )

    normalized_book_type = str(book_type or "Others").strip().casefold()
    if normalized_book_type == "drama":
        # In Drama mode 1200 is only a target. A complete character dialogue
        # must not be split merely because it exceeds the UI target.
        if largest_section > 2000:
            raise gr.Error(
                "A drama dialogue exceeded the 2000-character hard limit. "
                "It must be split at sentence/punctuation boundaries first."
            )
    elif normalized_book_type != "poetry" and largest_section > int(maximum_characters):
        raise gr.Error(
            "Text splitting failed.\n"
            f"Largest section: {largest_section} characters\n"
            f"Allowed: {int(maximum_characters)} characters"
        )

    clear_control_flags()

    # Free Ollama memory before F5-TTS starts.
    ollama_unload_message = (
        unload_ollama_model(
            ollama_model
        )
    )

    event_queue = queue.Queue()
    started_at = time.monotonic()
    timing_state = {"completed": 0}

    def progress_callback(
        information,
        current=None,
        total=None,
    ):
        message_text = (
            str(information.get("message", "")) if isinstance(information, dict)
            else str(information or "")
        )
        match = re.search(r"Completed section\s+(\d+)\s*/\s*(\d+)", message_text, re.I)
        if match:
            timing_state["completed"] = max(timing_state["completed"], int(match.group(1)))
        if isinstance(information, dict):
            progress_information = dict(information)
        else:
            try:
                current_number = int(current) if current is not None else 0
            except (TypeError, ValueError):
                current_number = 0
            try:
                total_number = int(total) if total is not None else len(sections)
            except (TypeError, ValueError):
                total_number = len(sections)

            percentage = (
                (100.0 * current_number / total_number)
                if total_number > 0
                else 0.0
            )
            progress_information = {
                "message": str(information or "Generating audio…"),
                "status": "running",
                "percentage": percentage,
                "completed": current_number,
                "remaining": max(0, total_number - current_number),
                "checkpoint": "Saved automatically after every section",
            }
        completed_weight = sum(section_weights[:timing_state["completed"]])
        elapsed, eta, finish = _timing(
            started_at, completed_weight, sum(section_weights)
        )
        progress_information.update({
            "percentage": 100.0 * timing_state["completed"] / max(1, len(sections)),
            "completed": timing_state["completed"],
            "remaining": max(0, len(sections) - timing_state["completed"]),
            "elapsed": elapsed,
            "eta": eta,
            "finish": finish,
        })

        event_queue.put({
            "type": "progress",
            "information": progress_information,
        })

    def generation_worker():
        try:
            result = generate_audiobook(
                sections=sections,
                reference_audio=(
                    reference_audio
                ),
                reference_text=(
                    reference_transcript
                ),
                project_name=(
                    project_name
                ),
                output_dir=str(project_directory),
                book_type=book_type or "Others",
                provider_engine=selected_engine,
                voicestudio_voice=str(voicestudio_voice or "default"),
                voicebox_profile=str(voicebox_profile or ""),
                voicebox_language=str(voicebox_language or "hi"),
                voicebox_model_size=str(voicebox_model_size or "0.6B"),
                voicebox_engine=str(voicebox_engine or "qwen"),
                sarvam_api_key=str(sarvam_api_key or ""),
                sarvam_speaker=str(sarvam_speaker or "shubh"),
                fish_reference_id=str(fish_reference_id or ""),
                gemini_api_key=str(gemini_api_key or ""),
                gemini_model=str(gemini_model or GEMINI_DEFAULT_TTS_MODEL),
                gemini_voice_id=str(gemini_voice_id or ""),
                colab_url=str(colab_url or DEFAULT_COLAB_URL),
                colab_token=str(colab_token or ""),
                speed=float(
                    speech_speed
                ),
                pause_ms=int(
                    pause_duration
                ),
                poetry_line_pause_ms=int(
                    poetry_line_pause_duration
                ),
                device_choice=(
                    device_choice
                ),
                resume=True,
                seed=int(seed),
                nfe_steps=int(
                    nfe_steps
                ),
                create_mp3=bool(
                    create_mp3
                ),
                progress_callback=(
                    progress_callback
                ),
            )

            event_queue.put({
                "type": "finished",
                "result": result,
            })

        except Exception as error:
            event_queue.put({
                "type": "error",
                "error": str(error),
            })

    worker = threading.Thread(
        target=generation_worker,
        daemon=True,
    )

    worker.start()

    last_status = (
        "Generation started.\n"
        f"Project: {project_name}\n"
        f"Book / part: {book_name} / {part_number}\n"
        f"Output: {project_directory}\n"
        f"Sections: {len(sections)}\n"
        f"Largest section: "
        f"{largest_section} characters\n"
        f"Target characters: {int(maximum_characters)}"
        + (" (Drama target; dialogues kept whole up to 2000 chars)" if normalized_book_type == "drama" else "")
        + "\n\n"
        f"{ollama_unload_message}"
    )

    yield (
        last_status,
        None,
        None,
        None,
        0,
    )

    while (
        worker.is_alive()
        or not event_queue.empty()
    ):
        try:
            event = event_queue.get(
                timeout=0.5
            )

        except queue.Empty:
            yield (
                last_status,
                None,
                None,
                None,
                100.0 * timing_state["completed"] / max(1, len(sections)),
            )
            continue

        event_type = event.get(
            "type"
        )

        if event_type == "progress":
            information = event[
                "information"
            ]

            last_status = (
                f"{information.get('message', '')}\n\n"
                f"Status: "
                f"{information.get('status', 'running')}\n"
                f"Progress: "
                f"{information.get('percentage', 0):.1f}%\n"
                f"Completed: "
                f"{information.get('completed', 0)}\n"
                f"Remaining: "
                f"{information.get('remaining', 0)}\n\n"
                f"Elapsed: {information.get('elapsed', '')}\n"
                f"Estimated remaining: {information.get('eta', '')}\n"
                f"Expected completion: {information.get('finish', '')}\n\n"
                f"Checkpoint:\n"
                f"{information.get('checkpoint', '')}"
            )

            yield (
                last_status,
                None,
                None,
                None,
                information.get("percentage", 0),
            )

        elif event_type == "finished":
            result = event[
                "result"
            ]

            status = result.get(
                "status",
                "unknown",
            )

            if status in {"complete", "completed"}:
                final_wav = (
                    result.get("final_wav")
                    or result.get("combined_wav")
                    or result.get("wav_file")
                )

                final_mp3 = (
                    result.get("final_mp3")
                    or result.get("combined_mp3")
                    or result.get("mp3_file")
                )

                completed_count = result.get(
                    "completed",
                    result.get("completed_sections", 0),
                )
                total_count = result.get(
                    "total",
                    result.get("total_sections", 0),
                )
                output_directory = result.get(
                    "output_directory",
                    result.get("output_dir", ""),
                )
                checkpoint_path = result.get(
                    "checkpoint",
                    result.get("checkpoint_file", ""),
                )

                last_status = (
                    "AUDIOBOOK COMPLETED\n\n"
                    f"Completed: "
                    f"{completed_count}/"
                    f"{total_count}\n\n"
                    f"Output folder:\n"
                    f"{output_directory}\n\n"
                    f"Checkpoint:\n"
                    f"{checkpoint_path}"
                )

                yield (
                    last_status,
                    final_wav,
                    final_wav,
                    final_mp3,
                    100,
                )

            elif status == "paused":
                last_status = (
                    "GENERATION PAUSED\n\n"
                    f"Completed: "
                    f"{result.get('completed', 0)}/"
                    f"{result.get('total', 0)}\n\n"
                    "Press Clear Pause, then "
                    "Generate / Resume."
                )

                yield (
                    last_status,
                    None,
                    None,
                    None,
                    100.0 * float(result.get('completed', 0)) / max(1, float(result.get('total', 1))),
                )

            elif status == "stopped":
                last_status = (
                    "GENERATION STOPPED\n\n"
                    f"Completed: "
                    f"{result.get('completed', 0)}/"
                    f"{result.get('total', 0)}\n\n"
                    "Press Generate / Resume "
                    "to continue."
                )

                yield (
                    last_status,
                    None,
                    None,
                    None,
                    100.0 * float(result.get('completed', 0)) / max(1, float(result.get('total', 1))),
                )

        elif event_type == "error":
            error_message = event.get(
                "error",
                "Unknown generation error",
            )

            raise gr.Error(
                "Audiobook generation failed.\n"
                "The checkpoint was saved.\n\n"
                f"{error_message}"
            )

        time.sleep(0.05)

    worker.join(
        timeout=1
    )


# ================================================================
# SadTalker talking-head video
# ================================================================

def check_sadtalker_from_ui(
    sadtalker_directory,
    environment_name,
):
    _ready, message = check_sadtalker_installation(
        sadtalker_directory=(
            sadtalker_directory
            or DEFAULT_SADTALKER_DIR
        ),
        environment_name=(
            environment_name
            or DEFAULT_SADTALKER_ENV
        ),
    )
    return message


def _project_audio_section(project_directory, index):
    """Section audio of the project (<book>_<part>_<nnn>.wav; older section_NNNN.wav files are migrated)."""
    found = pipeline.find_audio_section(project_directory, index)
    return found if found else pipeline.audio_section_path(project_directory, index)


def _project_state_dict(project_directory):
    try:
        return json.loads((Path(str(project_directory)) / "project_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _video_sections(project_directory, corrected_text, maximum_characters=None, book_type=None):
    """The ONE section list used by the audio tools, the video tools and the burned text."""
    text = clean_text(remove_page_markers(corrected_text or ""))
    stored = pipeline.load_matching_sections(project_directory, text)
    if stored:
        return stored
    state = _project_state_dict(project_directory)
    limit = int(maximum_characters or state.get("max_characters") or 1200)
    kind = str(book_type or state.get("book_type") or "Others")
    sections = pipeline.get_canonical_sections(project_directory, text, kind, limit, split_for_tts)
    pipeline.update_stage(
        project_directory, "sections", "completed", done=len(sections), total=len(sections),
        message=f"{len(sections)} sections ({kind}, up to {limit} characters each).",
    )
    return sections


def _is_poetry(project_directory):
    return str(_project_state_dict(project_directory).get("book_type") or "").strip().casefold() == "poetry"


def _parse_resolution(value, default=(1920, 1080)):
    match = re.match(r"\s*(\d+)\s*[xX*]\s*(\d+)", str(value or ""))
    return (int(match.group(1)), int(match.group(2))) if match else default


def _remote_video_sections(
    *, engine, project_dir, sections, settings, generator, mux_audio=False, require_audio=False,
    burn=True, resolution="1920x1080", font_name="Nirmala UI", font_size=28, max_clip_minutes=10.0,
    notify=None,
):
    """Run one video engine section by section (audio in -> video out), burn the section text,
    keep a checkpoint, and combine the section videos into clips of at most N minutes."""
    import inspect
    from video_service import _wait_for_video_controls, _video_pause_event

    width, height = _parse_resolution(resolution)
    takes_audio = len(inspect.signature(generator).parameters) >= 4

    def call(index, text, raw, audio):
        return generator(index, text, raw, audio) if takes_audio else generator(index, text, raw)

    clear_video_control_flags()
    result = pipeline.run_video_stage(
        engine=str(engine), project_dir=project_dir, sections=sections, generator=call, settings=dict(settings),
        require_audio=require_audio, mux_audio=mux_audio, burn=bool(burn), width=width, height=height, fps=25,
        font_name=str(font_name or "Nirmala UI"), font_size=int(font_size or 28), poetry=_is_poetry(project_dir),
        max_clip_seconds=float(max_clip_minutes or 10) * 60.0,
        control_check=lambda: _wait_for_video_controls(None), is_paused=_video_pause_event.is_set,
        stop_exceptions=(VideoGenerationStopped,), notify=notify,
    )
    result["checkpoint_file"] = result["checkpoint_file"]
    return result


def _video_outputs(result):
    """(preview, downloads) for the Gradio components: the combined clips, or the finished sections."""
    clips = [c["path"] for c in result.get("final_clips") or []]
    downloads = clips or list(result.get("section_videos") or [])
    preview = clips[0] if clips else (downloads[-1] if downloads else None)
    return preview, (downloads or None)


def _video_status_text(label, result, extra=""):
    clips = result.get("final_clips") or []
    text = (
        f"{label} {'stopped after the current section (checkpoint saved)' if result['status'] == 'stopped' else 'completed'}."
        f"\nCompleted sections: {result['completed_sections']}/{result['total_sections']}"
        f"\nSection videos: {result.get('output_directory', '')}/sections  (<book>_<part>_<nnn>.mp4, text burned in)"
    )
    if clips:
        text += "\nCombined clips (sections are never cut):\n" + "\n".join(
            f"  {Path(c['path']).name}  sections {c['sections'][0]}-{c['sections'][-1]}  {c['seconds'] / 60:.1f} min" for c in clips
        )
    if result.get("subtitle_file"):
        text += f"\nSubtitles: {result['subtitle_file']}"
    text += f"\nCheckpoint: {result['checkpoint_file']}"
    for warning in (result.get("warnings") or [])[:2]:
        text += f"\nNote: {warning}"
    return text + (("\n" + extra) if extra else "")


def _live_video_job(label, runner, finalize):
    """Run ``runner(notify)`` in a worker thread, yield live status lines, then ``finalize(result)``.

    ``finalize`` returns the final ``(status, preview, downloads)``.
    """
    events = queue.Queue()
    box = {}

    def notify(message, done=None, total=None):
        events.put(("progress", str(message), done, total))

    def work():
        try:
            box["result"] = runner(notify)
        except BaseException as exc:  # reported in the UI
            box["error"] = exc
        finally:
            events.put(("end", "", None, None))

    threading.Thread(target=work, daemon=True).start()
    started = time.monotonic()
    yield f"{label}: starting...", None, None
    while True:
        try:
            kind, message, done, total = events.get(timeout=1.0)
        except queue.Empty:
            continue
        if kind == "end":
            break
        progress = f"\nSections done: {done}/{total}" if done is not None and total else ""
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - started))
        yield f"{label}: {message}{progress}\nElapsed: {elapsed}", gr.update(), gr.update()
    if "error" in box:
        raise gr.Error(f"{label} failed:\n{box['error']}") from box["error"]
    yield finalize(box["result"])


# ---- small caches that let extraction / correction resume after Stop ----
_TEXT_CACHE = Path(__file__).resolve().parent / "outputs" / "_text_cache"


def _cache_path(kind, *parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]
    _TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    return _TEXT_CACHE / f"{kind}_{digest}.json"


def _cache_read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _cache_write(path, value):
    try:
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _cache_clear(path):
    try:
        Path(path).unlink()
    except OSError:
        pass


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_text_stage(stage, project_directory, status_text):
    """Write the outcome of a text stage (extract / correct) into the project's pipeline status."""
    if not str(project_directory or "").strip() or not Path(str(project_directory)).exists():
        return
    lowered = str(status_text or "").lower()
    state = "stopped" if "stopped" in lowered else ("completed" if "completed" in lowered else None)
    if state:
        first_line = str(status_text).strip().splitlines()[0][:200] if str(status_text).strip() else ""
        pipeline.update_stage(project_directory, stage, state, message=first_line)


def start_video_generation(
    corrected_text,
    project_name,
    maximum_characters,
    book_type,
    poet_image,
    sadtalker_directory,
    sadtalker_environment,
    preprocess_mode,
    video_model_size,
    still_mode,
    burn_subtitles,
    subtitle_font,
    subtitle_font_size,
    output_resolution,
    ollama_model,
    transition="None",
    transition_duration=0.4,
    video_book_name="",
    video_author_name="",
    video_part_number="",
    final_video_filename="poet_reciting_poem_1080p.mp4",
    max_clip_minutes=10.0,
):
    if not corrected_text or not str(corrected_text).strip():
        raise gr.Error(
            "Extract and review the poem text before generating video."
        )
    if not project_name or not str(project_name).strip():
        raise gr.Error("Enter the same project name used for audiobook generation.")
    if not str(video_book_name or "").strip() or not str(video_part_number or "").strip():
        raise gr.Error("Open/create a project with Book name and Part number first.")
    _project_message, project_directory_text = open_or_create_project(
        project_name, video_book_name, video_part_number, video_author_name
    )
    project_directory = Path(project_directory_text)
    if not poet_image:
        raise gr.Error("Upload a clear, front-facing photograph of the poet.")
    try:
        poet_image = _copy_project_asset(poet_image, project_directory, "poet_portrait")
    except Exception as exc:
        raise gr.Error(f"Could not save poet image in the project:\n{exc}") from exc
    save_project_state(project_directory, {
        "extracted_text": str(corrected_text),
        "book_type": str(book_type or "Others"),
        "poet_image": str(poet_image),
        "final_video_filename": str(final_video_filename or ""),
        "video_status": "Video generation started.",
        "max_characters": int(maximum_characters),
        "sadtalker_directory": str(sadtalker_directory or DEFAULT_SADTALKER_DIR),
        "sadtalker_environment": str(sadtalker_environment or DEFAULT_SADTALKER_ENV),
        "preprocess_mode": str(preprocess_mode or "crop"),
        "video_model_size": int(video_model_size or 256),
        "still_mode": bool(still_mode),
        "output_resolution": str(output_resolution or "1920x1080"),
        "burn_subtitles": bool(burn_subtitles),
        "subtitle_font": str(subtitle_font or "Nirmala UI"),
        "subtitle_font_size": int(subtitle_font_size or 28),
        "video_transition": str(transition or "None"),
        "video_transition_duration": float(transition_duration),
    })

    # The same section list the audio tools used (sections.json).
    sections = _video_sections(
        project_directory,
        corrected_text,
        maximum_characters,
        book_type,
    )
    if not sections:
        raise gr.Error("No subtitle/video sections could be prepared.")
    duration_weights = [
        _wav_seconds(_project_audio_section(project_directory, number))
        for number in range(1, len(sections) + 1)
    ]
    if not all(weight > 0 for weight in duration_weights):
        duration_weights = [max(1, len(section)) for section in sections]

    try:
        width_text, height_text = str(output_resolution or "1920x1080").lower().split("x", 1)
        output_width = int(width_text)
        output_height = int(height_text)
    except (TypeError, ValueError):
        output_width, output_height = 1920, 1080

    # Free the RTX 3050 before starting SadTalker in its separate environment.
    try:
        unload_model()
    except Exception:
        pass
    try:
        unload_ollama_model(ollama_model)
    except Exception:
        pass

    clear_video_control_flags()
    event_queue = queue.Queue()
    started_at = time.monotonic()
    timing_state = {"completed": 0}

    def progress_callback(
        information,
        current=None,
        total=None,
    ):
        message_text = str(information or "")
        match = re.search(r"Completed video section\s+(\d+)\s*/\s*(\d+)", message_text, re.I)
        if match:
            timing_state["completed"] = max(timing_state["completed"], int(match.group(1)))
        current_number = timing_state["completed"]
        try:
            total_number = int(total) if total is not None else len(sections)
        except (TypeError, ValueError):
            total_number = len(sections)
        percentage = (
            100.0 * current_number / max(1, len(sections))
        )
        elapsed, eta, finish = _timing(
            started_at,
            sum(duration_weights[:current_number]),
            sum(duration_weights),
        )
        event_queue.put({
            "type": "progress",
            "message": str(information),
            "current": current_number,
            "total": len(sections),
            "percentage": percentage,
            "elapsed": elapsed,
            "eta": eta,
            "finish": finish,
        })

    def video_worker():
        try:
            result = generate_talking_head_project(
                project_name=str(project_name),
                sections=sections,
                project_directory=str(project_directory),
                poet_image=poet_image,
                book_type=book_type or "Others",
                sadtalker_directory=(
                    sadtalker_directory
                    or DEFAULT_SADTALKER_DIR
                ),
                environment_name=(
                    sadtalker_environment
                    or DEFAULT_SADTALKER_ENV
                ),
                preprocess=str(preprocess_mode or "crop"),
                size=int(video_model_size or 256),
                still=bool(still_mode),
                burn_subtitles=bool(burn_subtitles),
                font_name=str(subtitle_font or "Nirmala UI"),
                font_size=int(subtitle_font_size or 28),
                output_width=output_width,
                output_height=output_height,
                transition=transition,
                transition_duration=float(transition_duration),
                resume=True,
                progress_callback=progress_callback,
                max_clip_minutes=float(max_clip_minutes or 10.0),
            )
            event_queue.put({
                "type": "finished",
                "result": result,
            })
        except Exception as error:
            event_queue.put({
                "type": "error",
                "error": str(error),
            })

    worker = threading.Thread(
        target=video_worker,
        daemon=True,
    )
    worker.start()

    last_status = (
        "Talking-head generation started.\n"
        f"Project: {project_name}\n"
        f"Book / part: {video_book_name} / {video_part_number}\n"
        f"Output: {project_directory}\n"
        f"Sections: {len(sections)}\n"
        f"SadTalker: {sadtalker_directory}\n"
        f"Environment: {sadtalker_environment}\n"
        "F5-TTS and Ollama were unloaded from GPU memory."
    )
    yield last_status, None, None, None, 0

    while worker.is_alive() or not event_queue.empty():
        try:
            event = event_queue.get(timeout=0.5)
        except queue.Empty:
            yield last_status, None, None, None, (
                100.0 * timing_state["completed"] / max(1, len(sections))
            )
            continue

        event_type = event.get("type")
        if event_type == "progress":
            last_status = (
                f"{event.get('message', '')}\n\n"
                f"Progress: {event.get('percentage', 0):.1f}%\n"
                f"Completed position: {event.get('current', 0)}/"
                f"{event.get('total', len(sections))}\n"
                f"Elapsed: {event.get('elapsed', '')}\n"
                f"Estimated remaining: {event.get('eta', '')}\n"
                f"Expected completion: {event.get('finish', '')}\n"
                "A checkpoint is saved after every completed MP4 section."
            )
            yield last_status, None, None, None, event.get("percentage", 0)
        elif event_type == "finished":
            result = event.get("result") or {}
            last_status = (
                f"{result.get('message', 'Video job finished.')}\n\n"
                f"Completed: {result.get('completed_sections', 0)}/"
                f"{result.get('total_sections', 0)}\n"
                f"Output directory:\n{result.get('output_directory', '')}\n\n"
                f"Checkpoint:\n{result.get('checkpoint_file', '')}"
            )
            final_video = result.get("final_video")
            final_clips = list(result.get("final_clips") or [])
            if not final_clips:
                final_clips = list(result.get("section_videos") or [])
            if result.get("final_clips"):
                last_status += "\n\nCombined clips (sections are never cut):\n" + "\n".join(
                    f"  {Path(clip).name}" for clip in result["final_clips"]
                )
            last_status += (
                "\n\nSection videos (text burned in): "
                f"{result.get('output_directory', '')}/sections/<book>_<part>_<nnn>.mp4"
            )
            subtitle_file = result.get("subtitle_file")
            yield (
                last_status,
                final_video or (final_clips[-1] if final_clips else None),
                final_clips or None,
                subtitle_file,
                100,
            )
            return
        elif event_type == "error":
            raise gr.Error(
                "SadTalker video generation failed:\n"
                + event.get("error", "Unknown error")
            )

    worker.join(timeout=1)



# ================================================================
# Colab remote media generation
# ================================================================

def check_colab_from_ui(colab_url, colab_token):
    return check_colab_server(colab_url or DEFAULT_COLAB_URL, colab_token or '')


def refresh_fish_reference_choices(colab_url, colab_token):
    try:
        references = list_fish_references(
            base_url=colab_url or DEFAULT_COLAB_URL,
            token=colab_token or "",
        )
        if not references:
            return gr.update(choices=[], value=None), "No Fish Speech reference voices found."
        preferred = "deven" if "deven" in references else references[0]
        return gr.update(choices=references, value=preferred), "Fish Speech references: " + ", ".join(references)
    except Exception as exc:
        return gr.update(), f"Could not refresh Fish Speech references: {type(exc).__name__}: {exc}"


def start_remote_wan_video_generation(
    corrected_text, project_name, book_name, author_name, part_number, poet_image, colab_url, colab_token,
    wan_task, wan_checkpoint, wan_prompt, wan_size, wan_frame_num, wan_steps, wan_seed, wan_audio_section,
    burn_text=True, output_resolution="1920x1080", subtitle_font="Nirmala UI", subtitle_font_size=28,
    max_clip_minutes=10.0,
):
    """Wan2.2 on Colab: section audio in -> section video out -> text burned -> clips of at most N minutes."""
    if not project_name or not str(project_name).strip():
        raise gr.Error('Enter the project name first.')
    if not str(book_name or '').strip() or not str(part_number or '').strip():
        raise gr.Error('Open/create the project with Book name and Part number first.')
    task = str(wan_task or 't2v-A14B')
    needs_image = task in {'ti2v-5B', 'i2v-A14B', 's2v-14B'}
    if needs_image and not poet_image:
        raise gr.Error('Upload the image first for this Wan2.2 mode.')
    _msg, project_dir_text = open_or_create_project(project_name, book_name, part_number, author_name)
    project_dir = Path(project_dir_text)
    sections = _video_sections(project_dir, corrected_text)
    if not sections:
        raise gr.Error('No video sections could be prepared.')
    image_copy = _copy_project_asset(poet_image, project_dir, 'wan_input_image') if poet_image else ''
    try:
        unload_model()
        unload_ollama_model('')
    except Exception:
        pass
    settings = {'task': task, 'checkpoint': str(wan_checkpoint or ''), 'size': str(wan_size or '1280*720'),
                'frame_num': int(wan_frame_num or 81), 'steps': int(wan_steps or 20), 'seed': int(wan_seed or -1),
                'prompt': str(wan_prompt or ''), 'image': image_copy}
    base_prompt = str(wan_prompt or '').strip()
    wan_fps = 24 if task == 'ti2v-5B' else 16
    piece_limit = max(2.0, int(wan_frame_num or 81) / wan_fps) if task == 's2v-14B' else None

    def generate_one(index, section_text, destination, audio_file):
        prompt = base_prompt or f'A realistic cinematic scene inspired by this narration: {section_text[:1200]}'

        def render(out, piece_audio):
            return generate_wan_remote(
                colab_url or DEFAULT_COLAB_URL, colab_token or '', prompt=prompt, image_path=image_copy,
                audio_path=str(piece_audio) if piece_audio else '', task=task, checkpoint_dir=wan_checkpoint or '',
                size=str(wan_size or '1280*720'), frame_num=int(wan_frame_num or 81), sample_steps=int(wan_steps or 20),
                seed=int(wan_seed or -1), output_path=out)

        if task != 's2v-14B':
            return render(destination, None)
        if audio_file is None:
            raise RuntimeError(f'Wan2.2 S2V needs the section audio for section {index}. Generate the audio first.')
        pipeline.generate_in_pieces(lambda n, c, out, piece_audio, secs: render(out, piece_audio),
                                    destination, audio_file, piece_limit, destination.parent, destination.stem)

    def runner(notify):
        return _remote_video_sections(
            engine='wan2.2', project_dir=project_dir, sections=sections, settings=settings, generator=generate_one,
            mux_audio=True, require_audio=(task == 's2v-14B'), burn=burn_text, resolution=output_resolution,
            font_name=subtitle_font, font_size=subtitle_font_size, max_clip_minutes=max_clip_minutes, notify=notify)

    def finalize(result):
        preview, downloads = _video_outputs(result)
        status = _video_status_text('Wan2.2 (Colab)', result)
        save_project_state(project_dir, {'wan_video': preview, 'wan_task': task, 'wan_prompt': base_prompt, 'wan_status': status})
        return status, preview, downloads

    yield from _live_video_job('Wan2.2 (Colab)', runner, finalize)


def start_remote_ltx_video_generation(
    corrected_text, project_name, book_name, author_name, part_number, poet_image, colab_url, colab_token,
    ltx_mode, ltx_prompt, ltx_duration, ltx_fps, ltx_audio=None,
    burn_text=True, output_resolution="1920x1080", subtitle_font="Nirmala UI", subtitle_font_size=28,
    max_clip_minutes=10.0,
):
    """LTX-2.5 on Colab: section audio in -> section video out -> text burned -> clips of at most N minutes."""
    if not project_name or not str(project_name).strip():
        raise gr.Error('Enter the project name first.')
    if not str(book_name or '').strip() or not str(part_number or '').strip():
        raise gr.Error('Open/create the project with Book name and Part number first.')
    mode = str(ltx_mode or 't2v_i2v_single')
    if mode != 't2v' and not poet_image:
        raise gr.Error('Upload the image first for this LTX-2.5 mode.')
    _msg, project_dir_text = open_or_create_project(project_name, book_name, part_number, author_name)
    project_dir = Path(project_dir_text)
    sections = _video_sections(project_dir, corrected_text)
    if not sections:
        raise gr.Error('No video sections could be prepared.')
    image_copy = _copy_project_asset(poet_image, project_dir, 'ltx_input_image') if poet_image else ''
    try:
        unload_model()
        unload_ollama_model('')
    except Exception:
        pass
    settings = {'mode': mode, 'duration': float(ltx_duration or 5), 'fps': int(ltx_fps or 24),
                'prompt': str(ltx_prompt or ''), 'image': image_copy, 'audio_input': str(ltx_audio or '')}
    base_prompt = str(ltx_prompt or '').strip()
    piece_limit = max(2.0, float(ltx_duration or 5)) if mode == 'a2vid' else None
    manual_audio = _copy_project_asset(ltx_audio, project_dir, 'ltx_input_audio') if ltx_audio else ''

    def generate_one(index, section_text, destination, audio_file):
        prompt = base_prompt or f'A cinematic realistic scene inspired by: {section_text[:1500]}'

        def render(out, piece_audio, seconds):
            return generate_ltx_remote(
                colab_url or DEFAULT_COLAB_URL, colab_token or '', prompt=prompt, image_path=image_copy,
                audio_path=str(piece_audio) if piece_audio else '', mode=mode,
                duration=float(seconds or ltx_duration or 5), fps=int(ltx_fps or 24), output_path=out)

        if mode != 'a2vid':
            return render(destination, None, float(ltx_duration or 5))
        source = audio_file or (Path(manual_audio) if manual_audio and Path(manual_audio).is_file() else None)
        if source is None:
            raise RuntimeError(f'LTX A2Vid needs the section audio for section {index}. Generate the audio first.')
        pipeline.generate_in_pieces(lambda n, c, out, piece_audio, secs: render(out, piece_audio, math.ceil(secs)),
                                    destination, Path(source), piece_limit, destination.parent, destination.stem)

    def runner(notify):
        return _remote_video_sections(
            engine='ltx2.5', project_dir=project_dir, sections=sections, settings=settings, generator=generate_one,
            mux_audio=True, require_audio=False, burn=burn_text, resolution=output_resolution,
            font_name=subtitle_font, font_size=subtitle_font_size, max_clip_minutes=max_clip_minutes, notify=notify)

    def finalize(result):
        preview, downloads = _video_outputs(result)
        status = _video_status_text('LTX 2.5 (Colab)', result)
        save_project_state(project_dir, {'ltx_video': preview, 'ltx_mode': mode, 'ltx_status': status})
        return status, preview, downloads

    yield from _live_video_job('LTX 2.5 (Colab)', runner, finalize)


def check_ltx23_from_ui(bridge_url, bridge_token):
    ok, message = check_ltx23_bridge(bridge_url or DEFAULT_LTX23_BRIDGE_URL, bridge_token or "")
    if not ok:
        return message
    return message


def start_remote_ltx23_video_generation(
    corrected_text, project_name, book_name, author_name, part_number,
    poet_image, ltx23_image, bridge_url, bridge_token, ltx23_prompt, ltx23_duration,
    ltx23_fps, ltx23_width, ltx23_height, ltx23_seed, ltx23_audio,
    ltx23_ref_video=None, ltx23_workflow="image",
    ltx23_extra_positive="", ltx23_extra_negative="", ltx23_cfg=1.0, ltx23_pose_use_image=True,
    burn_text=True, output_resolution="1920x1080", subtitle_font="Nirmala UI", subtitle_font_size=28,
    max_clip_minutes=10.0,
):
    """LTX ComfyUI workflows on Colab, section by section.

    ltx23_workflow == "image": LTX-2.3 image + audio.
    ltx23_workflow == "pose":  LTX-2.5 audio + reference video (head/hand/body motion from the video).
    The audio of every section is taken from the project's audio folder automatically.
    """
    if not project_name or not str(project_name).strip():
        raise gr.Error("Enter the project name first.")
    if not str(book_name or "").strip() or not str(part_number or "").strip():
        raise gr.Error("Open/create the project with Book name and Part number first.")
    if not str(bridge_url or DEFAULT_LTX23_BRIDGE_URL).strip():
        raise gr.Error("Enter the public URL for the LTX Colab bridge (port 8190).")
    use_pose = str(ltx23_workflow or "image") == "pose"
    has_ref_video = bool(ltx23_ref_video) and Path(str(ltx23_ref_video)).is_file()
    if use_pose and ltx23_ref_video and not has_ref_video:
        raise gr.Error("The reference video upload could not be found. Upload it again.")

    _msg, project_dir_text = open_or_create_project(project_name, book_name, part_number, author_name)
    project_dir = Path(project_dir_text)
    sections = _video_sections(project_dir, corrected_text)
    if not sections:
        raise gr.Error("No video sections could be prepared.")

    chosen_image = ltx23_image if ltx23_image and Path(str(ltx23_image)).is_file() else poet_image
    image_copy = (
        _copy_project_asset(chosen_image, project_dir, "ltx23_input_image")
        if (not use_pose or ltx23_pose_use_image) and chosen_image and Path(str(chosen_image)).is_file() else ""
    )
    # Optional manual audio: used only when a section has no generated audio.
    manual_audio_copy = (
        _copy_project_asset(ltx23_audio, project_dir, "ltx23_input_audio")
        if ltx23_audio and Path(str(ltx23_audio)).is_file() else ""
    )
    ref_video_copy = (
        _copy_project_asset(ltx23_ref_video, project_dir, "ltx23_ref_video") if use_pose and has_ref_video else ""
    )
    try:
        unload_model()
        unload_ollama_model("")
    except Exception:
        pass

    def _file_sha(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else ""
        except OSError:
            return ""

    max_piece = float(max(2, min(60, int(ltx23_duration or 20))))
    settings = {
        "mode": "audio+reference-video pose (LTX-2.5)" if use_pose else "image+audio-to-video",
        "reference_video": ref_video_copy,
        "reference_video_sha256": _file_sha(ref_video_copy),
        "pose_use_image": bool(use_pose and image_copy),
        "extra_positive": str(ltx23_extra_positive or "").strip(),
        "extra_negative": str(ltx23_extra_negative or "").strip(),
        "cfg": float(ltx23_cfg or 1.0),
        "max_seconds_per_render": max_piece,
        "fps": int(ltx23_fps or 24),
        "width": int(ltx23_width or 1280),
        "height": int(ltx23_height or 720),
        "seed": int(ltx23_seed if ltx23_seed is not None else -1),
        "prompt": str(ltx23_prompt or ""),
        "image": image_copy,
        "manual_audio_sha256": _file_sha(manual_audio_copy),
        "bridge_url": str(bridge_url or DEFAULT_LTX23_BRIDGE_URL),
    }
    base_prompt = str(ltx23_prompt or "").strip()
    reference = {"video": ref_video_copy or None}

    def generate_one(index, section_text, destination, audio_file):
        source = audio_file or (Path(manual_audio_copy) if manual_audio_copy and Path(manual_audio_copy).is_file() else None)
        if source is None:
            raise RuntimeError(f"No audio for section {index}. Generate the audio first (Tab 4).")

        def render(number, count, out, piece_audio, seconds):
            if use_pose:
                prompt = base_prompt  # blank: the bridge uses the positive prompt stored in the workflow JSON
            else:
                prompt = base_prompt or (
                    "Realistic audio-conditioned video. If a reference image is available, preserve the "
                    "subject's identity and appearance. Match visible movement to the uploaded audio, "
                    "with natural facial movement and accurate lip synchronization. Context: "
                    + str(section_text)[:3000]
                )
            pose_kwargs = (
                {"workflow_path": DEFAULT_POSE_WORKFLOW_PATH, "reference_video_path": reference["video"]}
                if use_pose else {}
            )
            output = generate_ltx23_video(
                prompt=prompt, output_path=out, image_path=image_copy or None, audio_path=piece_audio, **pose_kwargs,
                base_url=bridge_url or DEFAULT_LTX23_BRIDGE_URL, api_key=bridge_token or "",
                duration=float(max(1, min(60, math.ceil(seconds)))), fps=int(ltx23_fps or 24),
                width=int(ltx23_width or 1280), height=int(ltx23_height or 720),
                seed=int(ltx23_seed if ltx23_seed is not None else -1),
                extra_positive=str(ltx23_extra_positive or "").strip(),
                negative_prompt=str(ltx23_extra_negative or "").strip(),
                cfg=float(ltx23_cfg or 1.0), timeout=3600,
            )
            if use_pose:
                reference["video"] = None  # the bridge keeps the last reference video: upload it only once
            return output

        pipeline.generate_in_pieces(render, destination, Path(source), max_piece, destination.parent, destination.stem)

    engine_name = "ltx25_pose" if use_pose else "ltx2.3"
    label = "LTX-2.5 pose (ComfyUI / Colab)" if use_pose else "LTX-2.3 (ComfyUI / Colab)"

    def runner(notify):
        return _remote_video_sections(
            engine=engine_name, project_dir=project_dir, sections=sections, settings=settings, generator=generate_one,
            mux_audio=False, require_audio=not manual_audio_copy, burn=burn_text, resolution=output_resolution,
            font_name=subtitle_font, font_size=subtitle_font_size, max_clip_minutes=max_clip_minutes, notify=notify)

    def finalize(result):
        preview, downloads = _video_outputs(result)
        note = ""
        if str(ltx23_extra_negative or "").strip() and float(ltx23_cfg or 1.0) <= 1.0:
            note = ("Note: your additional negative prompt has no effect at CFG 1.0 (distilled model). "
                    "Raise 'Negative prompt strength (CFG)' to 1.5-3 to use it.")
        status = _video_status_text(label, result, note)
        save_project_state(project_dir, {"ltx23_video": preview, "ltx23_status": status})
        return status, preview, downloads

    yield from _live_video_job(label, runner, finalize)


# ================================================================
# Interface
# ================================================================

CUSTOM_CSS = """
:root {
    color-scheme: light;
    --f5-ink: #172235; --f5-line: #e7eaf0; --f5-paper: #f6f7fb;
    --f5-navy: #111b2c; --f5-accent: #6b63d9; --f5-accent-dark: #554cc5;
}
body, .gradio-container {
    background: var(--f5-paper) !important; color: var(--f5-ink) !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.gradio-container { max-width:1680px !important; padding:18px 24px 36px !important; }
#f5pdfstudio-app .f5-brandbar {
    display:flex; align-items:center; gap:13px; padding:12px 4px 20px;
    border-bottom:1px solid var(--f5-line); margin-bottom:20px;
}
#f5pdfstudio-app .f5-brandmark {
    width:43px; height:43px; border-radius:14px; display:grid; place-items:center;
    color:#fff; font-weight:900; letter-spacing:-1px;
    background:linear-gradient(145deg,#8e86f2,#5e56c9); box-shadow:0 7px 20px #655bd94d;
}
#f5pdfstudio-app .f5-brandcopy { display:grid; gap:1px; }
#f5pdfstudio-app .f5-brandcopy strong { color:#172235; font-size:17px; letter-spacing:-.4px; }
#f5pdfstudio-app .f5-brandcopy small { color:#8792a5; font-size:9px; font-weight:800; letter-spacing:1.5px; }
#f5pdfstudio-app .f5-local-pill {
    margin-left:auto; padding:7px 11px; border-radius:20px; background:#eaf7f2;
    color:#25856f; font-size:10px; font-weight:800; letter-spacing:.5px;
}
#f5pdfstudio-app .f5-welcome {
    overflow:hidden; margin:0 0 20px; padding:25px 30px; border-radius:17px; color:#fff;
    background:linear-gradient(112deg,#202b48 0%,#333b66 55%,#625eb1 100%);
    box-shadow:0 12px 28px #2731531c;
}
#f5pdfstudio-app .f5-welcome .eyebrow { color:#c5c2ff; font-size:10px; font-weight:800; letter-spacing:1.6px; }
#f5pdfstudio-app .f5-welcome h2 { margin:9px 0 7px; color:#fff; font-size:clamp(24px,3vw,34px); line-height:1.15; letter-spacing:-1px; }
#f5pdfstudio-app .f5-welcome p { margin:0; color:#d1d6e8; font-size:13px; }
#f5pdfstudio-app .tabs {
    display:grid !important; grid-template-columns:238px minmax(0,1fr) !important;
    align-items:start !important; gap:20px !important;
}
#f5pdfstudio-app .tab-nav {
    display:flex !important; flex-direction:column !important; align-items:stretch !important;
    gap:6px !important; padding:13px !important; margin:0 !important; position:sticky; top:14px;
    background:var(--f5-navy) !important; border:1px solid #24324a !important;
    border-radius:15px !important; box-shadow:0 8px 24px #17223512;
}
#f5pdfstudio-app .tab-nav button {
    width:100% !important; min-height:43px !important; justify-content:flex-start !important;
    text-align:left !important; padding:10px 13px !important; margin:0 !important;
    border:0 !important; border-radius:9px !important; background:transparent !important;
    color:#aeb9cc !important; font-size:12px !important; font-weight:650 !important;
    box-shadow:none !important; transition:background .18s,color .18s !important;
}
#f5pdfstudio-app .tab-nav button:hover { background:#ffffff0d !important; color:#fff !important; transform:none !important; }
#f5pdfstudio-app .tab-nav button.selected,
#f5pdfstudio-app .tab-nav button[aria-selected="true"] {
    background:#7169df30 !important; color:#fff !important; box-shadow:inset 3px 0 #8982f3 !important;
}
#f5pdfstudio-app .tab-nav button::after { display:none !important; }
#f5pdfstudio-app .tabitem {
    min-width:0 !important; padding:23px !important; margin:0 !important;
    background:#fff !important; border:1px solid var(--f5-line) !important;
    border-radius:15px !important; box-shadow:0 4px 18px #202b4208;
}
#f5pdfstudio-app input, #f5pdfstudio-app textarea, #f5pdfstudio-app select { border-radius:8px !important; }
#f5pdfstudio-app button { border-radius:8px !important; font-weight:650 !important; }
#f5pdfstudio-app button.primary { background:var(--f5-accent) !important; border-color:var(--f5-accent) !important; }
#f5pdfstudio-app button.primary:hover { background:var(--f5-accent-dark) !important; border-color:var(--f5-accent-dark) !important; }
#f5pdfstudio-app h1, #f5pdfstudio-app h2, #f5pdfstudio-app h3 { letter-spacing:-.35px; }
.status-output textarea { font-family:Consolas, "Cascadia Code", monospace !important; font-size:12px !important; }
@media (max-width:900px) {
    .gradio-container { padding:12px !important; }
    #f5pdfstudio-app .tabs { grid-template-columns:1fr !important; gap:12px !important; }
    #f5pdfstudio-app .tab-nav { position:static; flex-direction:row !important; overflow-x:auto !important; white-space:nowrap; padding:8px !important; }
    #f5pdfstudio-app .tab-nav button { width:auto !important; flex:0 0 auto !important; }
    #f5pdfstudio-app .tabitem { padding:15px !important; }
}
@media (prefers-reduced-motion:reduce) { #f5pdfstudio-app * { transition:none !important; scroll-behavior:auto !important; } }
"""




def _correction_engine_visibility(engine):
    engine = str(engine or "ollama").lower()
    return (gr.update(visible=engine == "ollama"), gr.update(visible=engine == "gemini"))

def _audio_engine_visibility(engine):
    engine = str(engine or "f5tts").lower()
    return (
        gr.update(visible=engine in {"f5tts", "voicestudio"}),
        gr.update(visible=engine == "f5tts"),
        gr.update(visible=engine == "voicestudio"),
        gr.update(visible=engine == "voicestudio"),
        gr.update(visible=engine == "voicebox"),
        gr.update(visible=engine == "voicebox"),
        gr.update(visible=engine == "sarvam"),
        gr.update(visible=engine == "fishspeech"),
        gr.update(visible=engine == "gemini"),
        gr.update(visible=engine == "fishspeech"),   # Colab connection: only Fish Speech needs it here
    )

def _video_engine_visibility(engine):
    engine = str(engine or "sadtalker").lower()
    return (
        gr.update(visible=engine == "sadtalker"),
        gr.update(visible=engine == "wan2.2"),
        gr.update(visible=engine == "ltx2.5"),
        gr.update(visible=engine == "ltx2.3"),
        gr.update(visible=engine in {"wan2.2", "ltx2.5"}),   # Colab media server: only Wan2.2 / LTX-2.5
    )

def refresh_pipeline_status(project_directory):
    if not str(project_directory or "").strip():
        return "Open or create a project to see its pipeline status."
    return pipeline.format_pipeline_status(project_directory)


def combine_finished_videos(project_directory, max_minutes):
    if not str(project_directory or "").strip():
        raise gr.Error("Open or create a project first.")
    try:
        result = pipeline.combine_existing_videos(project_directory, float(max_minutes or 10) * 60.0)
    except Exception as exc:
        raise gr.Error(f"Could not combine the section videos:\n{exc}") from exc
    clips = [clip["path"] for clip in result["final_clips"]]
    text = "Combined finished section videos (a section is never cut):\n" + "\n".join(
        f"  {Path(c['path']).name}  sections {c['sections'][0]}-{c['sections'][-1]}  {c['seconds'] / 60:.1f} min"
        for c in result["final_clips"]
    )
    return text, clips[0], clips


def _pause_stage(name, also_video=False):
    def action():
        message = pipeline.get_control(name).pause()
        if also_video:
            request_video_pause()
        return message
    return action


def _resume_stage(name, also_video=False):
    def action():
        message = pipeline.get_control(name).resume()
        if also_video:
            request_video_resume()
        return message
    return action


def _stop_stage(name, also_video=False):
    def action():
        message = pipeline.get_control(name).stop()
        if also_video:
            request_video_stop()
        return message
    return action


with gr.Blocks(title="F5PdfStudio — Local AI + Colab Media Studio", elem_id="f5pdfstudio-app") as demo:
    gr.Markdown("""
<div class="f5-brandbar">
  <div class="f5-brandmark">F5</div>
  <div class="f5-brandcopy">
    <strong>F5PdfStudio</strong>
    <small>DOCUMENT · AUDIOBOOK · VIDEO WORKSPACE</small>
  </div>
  <div class="f5-local-pill">● LOCAL WORKSPACE</div>
</div>
<div class="f5-welcome">
  <div class="eyebrow">YOUR AI BOOK PRODUCTION STUDIO</div>
  <h2>From document to audiobook and video.</h2>
  <p>Extract, refine, narrate and produce your book with your existing local and cloud AI tools.</p>
</div>
""")

    with gr.Tab("Overview / Projects"):
        gr.Markdown("### Project workspace\nOpen an existing book project or create a new one. Your saved sections, audio checkpoints and video outputs remain in their existing project folders.")
        with gr.Row():
            existing_project = gr.Dropdown(choices=[NEW_PROJECT_CHOICE, *list_existing_projects()], value=NEW_PROJECT_CHOICE, label="Existing project / New project")
            refresh_projects_button = gr.Button("Refresh project list")
        with gr.Row():
            project_name = gr.Textbox(label="Project name")
            book_name = gr.Textbox(label="Book name")
            part_number = gr.Textbox(value="1", label="Part number")
            author_name = gr.Textbox(label="Author name")
        project_button = gr.Button("Create / Save Project", variant="primary")
        project_status = gr.Textbox(label="Project status", lines=6, interactive=False)
        project_output_path = gr.Textbox(label="Project output folder", interactive=False)
        pipeline_status_box = gr.Textbox(
            label="Pipeline status (saved after every section and stage)", lines=9, interactive=False,
            value="Open or create a project to see its pipeline status.", elem_classes=["status-output"],
        )
        refresh_pipeline_button = gr.Button("Refresh pipeline status")
        pipeline_timer = gr.Timer(5) if hasattr(gr, "Timer") else None

    with gr.Tab("Document & OCR"):
        with gr.Row():
            document_file = gr.File(label="Upload PDF or Word document", file_types=[".pdf", ".docx", ".doc"], type="filepath")
            with gr.Column():
                selection_range = gr.Textbox(label="Content range", placeholder="PDF pages: 1-20,25. Word blocks: 1-30. Leave empty for all.")
                book_type = gr.Radio(choices=["Poetry", "Drama", "Novel", "Others"], value="Poetry", label="Type of Book")
                ocr_language = gr.Radio(choices=[("Hindi", "hin"), ("English", "eng"), ("Hindi + English", "hin+eng")], value="hin+eng", label="OCR language")
                ocr_dpi = gr.Slider(150, 400, value=300, step=25, label="OCR resolution (DPI)")
                word_image_ocr = gr.Checkbox(value=True, label="OCR images embedded in Word documents")
                extract_button = gr.Button("Extract document text", variant="primary")
        with gr.Row():
            pause_extract_button = gr.Button("Pause after current page/block")
            resume_extract_button = gr.Button("Resume")
            stop_extract_button = gr.Button("Stop after current page/block", variant="stop")
        extraction_status = gr.Textbox(label="Extraction status", value="Upload a document to begin.", lines=8, interactive=False, elem_classes=["status-output"])

    with gr.Tab("Text Editor"):
        with gr.Row():
            correction_engine = gr.Dropdown(choices=[("Ollama — local", "ollama"), ("Gemini API — cloud", "gemini")], value="ollama", label="Text correction engine", scale=1)
            max_characters = gr.Slider(300, 2000, value=1200, step=100, label="Target characters per section", scale=1)
        extracted_text = gr.Textbox(label="Extracted / corrected text", lines=25)
        with gr.Row():
            clean_button = gr.Button("Clean text")
            detect_button = gr.Button("Detect chapters and TTS sections")
        with gr.Row():
            pause_correct_button = gr.Button("Pause correction after current section")
            resume_correct_button = gr.Button("Resume")
            stop_correct_button = gr.Button("Stop after current section", variant="stop")

        with gr.Group(visible=True) as ollama_correction_group:
            with gr.Row():
                ollama_model = gr.Dropdown(choices=["llama3.2:latest", "qwen3:4b-instruct", "qwen3:1.7b", "qwen3:8b", "minimax-m3:cloud"], value=DEFAULT_OLLAMA_MODEL, allow_custom_value=True, label="Ollama model")
                ollama_chunk_size = gr.Slider(500, 1200, value=700, step=100, label="Characters per Ollama request")
                correction_language = gr.Dropdown(choices=["Hindi", "English", "Hindi/English"], value="Hindi/English", label="Correction language")
            with gr.Row():
                check_ollama_button = gr.Button("Check Ollama")
                ollama_correct_button = gr.Button("Correct Text with Ollama", variant="primary")
                ollama_chapters_button = gr.Button("Detect chapters with Ollama")
                unload_ollama_button = gr.Button("Unload Ollama model")

        with gr.Group(visible=False) as gemini_correction_group:
            with gr.Row():
                gemini_correction_model = gr.Dropdown(choices=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.8-flash"], value=GEMINI_DEFAULT_TEXT_CORRECTION_MODEL, allow_custom_value=True, label="Gemini correction model")
                gemini_correction_chunk_size = gr.Slider(500, 6000, value=1200, step=100, label="Characters per Gemini request")
                gemini_correction_api_key = gr.Textbox(type="password", label="Gemini API key (not saved)")
            gemini_correct_button = gr.Button("Correct Text with Gemini API", variant="primary")
            gemini_correction_status = gr.Textbox(label="Gemini correction status", lines=8, interactive=False, elem_classes=["status-output"])

        text_status = gr.Textbox(label="Text-processing status", lines=8, interactive=False, elem_classes=["status-output"])
        ollama_status = gr.Textbox(label="Ollama status", lines=10, interactive=False, elem_classes=["status-output"])
        section_preview = gr.Textbox(label="Chapter and section preview", lines=20, interactive=False)

    with gr.Tab("Audio Production"):
        gr.Markdown("## All audio engines are in this tab. Select one option for generation.")
        with gr.Row():
            tts_engine = gr.Dropdown(choices=[
                ("F5-Hindi — local GPU", "f5tts"),
                ("VoiceStudio — local API", "voicestudio"),
                ("Voicebox — local API", "voicebox"),
                ("Sarvam Bulbul v3 — cloud", "sarvam"),
                ("Fish Speech — Colab GPU", "fishspeech"),
                ("Gemini TTS — cloud + replicated voice", "gemini"),
            ], value="f5tts", label="Audio engine")
        with gr.Group(visible=True) as reference_audio_group:
            with gr.Row():
                reference_audio = gr.Audio(label="Reference audio", type="filepath")
                reference_transcript = gr.Textbox(label="Exact reference transcript", value=DEFAULT_REFERENCE_TRANSCRIPT, lines=7)
        with gr.Group(visible=True) as f5tts_audio_group:
            with gr.Row():
                device_choice = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="F5-TTS device")
                speech_speed = gr.Slider(0.70, 1.30, value=0.95, step=0.05, label="Speech speed")
                seed = gr.Number(value=1234, precision=0, label="Seed")
                nfe_steps = gr.Slider(16, 64, value=32, step=4, label="Inference steps")

        with gr.Group(visible=False) as voicestudio_audio_group:
            with gr.Row():
                voicestudio_voice = gr.Dropdown(choices=["default"], value="default", allow_custom_value=True, label="VoiceStudio voice/profile ID")

        with gr.Group(visible=False) as voicebox_audio_group:
            with gr.Row():
                voicebox_profile = gr.Dropdown(choices=[""], value="", allow_custom_value=True, label="Voicebox profile ID")
                voicebox_language = gr.Textbox(value="hi", label="Language")
                voicebox_model_size = gr.Dropdown(choices=["0.6B", "1.7B"], value="0.6B", label="Model size")
                voicebox_engine = gr.Dropdown(choices=["qwen", "qwen_custom_voice", "chatterbox", "chatterbox_turbo", "luxtts", "tada", "kokoro"], value="qwen", allow_custom_value=True, label="Voicebox engine")

        with gr.Group(visible=False) as sarvam_audio_group:
            with gr.Row():
                sarvam_speaker = gr.Dropdown(choices=SARVAM_SPEAKERS, value="deven", allow_custom_value=True, label="Speaker / cloned voice ID")
                sarvam_api_key = gr.Textbox(type="password", label="Sarvam API key (not saved)")

        with gr.Group(visible=False) as fishspeech_audio_group:
            with gr.Row():
                fish_reference_id = gr.Dropdown(choices=["deven"], value="deven", allow_custom_value=True, label="Fish Speech reference voice")
                refresh_fish_references_button = gr.Button("Refresh Fish voices")
            fish_reference_status = gr.Textbox(label="Fish Speech reference status", value="deven — registered reference voice", interactive=False, lines=2)
            gr.Markdown("Fish Speech uses the native `reference_id` API. Reference voices can be persisted on Google Drive by the Colab media server.")

        with gr.Group(visible=False) as gemini_audio_group:
            with gr.Row():
                gemini_audio_model = gr.Dropdown(choices=["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"], value=GEMINI_DEFAULT_TTS_MODEL, label="Gemini TTS model")
                gemini_voice_id = gr.Textbox(value="Kore", label="Voice / replicated voice ID", placeholder="voice_... or Kore/Puck/etc.")
                gemini_api_key_audio = gr.Textbox(type="password", label="Gemini API key (not saved)")
            with gr.Row():
                gemini_consent_audio = gr.Audio(label="Voice consent recording (10–30 sec)", type="filepath")
                create_gemini_voice_button = gr.Button("Create Gemini replicated voice")
            gemini_voice_status = gr.Textbox(label="Gemini voice status", lines=4, interactive=False, elem_classes=["status-output"])
        with gr.Group(visible=False) as colab_audio_group:
            gr.Markdown("### Colab media server connection (Fish Speech)")
            with gr.Row():
                colab_url = gr.Textbox(value=DEFAULT_COLAB_URL, label="Colab media server URL")
                colab_token = gr.Textbox(type="password", label="Colab bearer token (not saved)")
                check_colab_button = gr.Button("Check Colab server")
            colab_status = gr.Textbox(label="Colab status", lines=5, interactive=False, elem_classes=["status-output"])
        with gr.Row():
            pause_duration = gr.Slider(0, 2000, value=750, step=50, label="Silence between sections (ms)")
            poetry_line_pause_duration = gr.Slider(200, 1200, value=450, step=50, label="Poetry line pause (ms)")
            create_mp3 = gr.Checkbox(value=True, label="Create MP3")
        with gr.Group(visible=False) as voicestudio_refresh_group:
            refresh_vs_voices_button = gr.Button("Refresh VoiceStudio voices")
        with gr.Group(visible=False) as voicebox_refresh_group:
            refresh_voicebox_button = gr.Button("Refresh Voicebox profiles")
        with gr.Row():
            generate_button = gr.Button("Generate / Resume Audio", variant="primary")
            pause_button = gr.Button("Pause after current section")
            resume_button = gr.Button("Resume")
            stop_button = gr.Button("Stop after current section", variant="stop")
        progress_status = gr.Textbox(label="Audio generation progress", value="Ready", lines=12, interactive=False, elem_classes=["status-output"])
        audio_progress_bar = gr.Slider(0, 100, value=0, step=0.1, label="Audio generation progress (%)", interactive=False)
        final_audio = gr.Audio(label="Completed audiobook preview")
        with gr.Row():
            wav_download = gr.File(label="Download complete WAV")
            mp3_download = gr.File(label="Download complete MP3")

    with gr.Tab("Video Production"):
        gr.Markdown(
            "## Select a video engine. Only the settings of the selected engine are shown.\n"
            "For every section the section **audio** is sent to the video tool, the section **text** is burned on the "
            "whole video, and the clip is saved as `<book>_<part>_<nnn>.mp4` in the project's `video/sections` folder. "
            "When all sections are finished they are combined into clips of at most the length below; a section is never cut."
        )
        video_engine = gr.Dropdown(choices=[("SadTalker — local", "sadtalker"), ("Wan2.2 — Colab", "wan2.2"), ("LTX-2.5 — Colab", "ltx2.5"), ("LTX-2.3 / LTX-2.5 pose — ComfyUI / Colab GPU", "ltx2.3")], value="sadtalker", label="Video engine")
        poet_image = gr.Image(label="Portrait / character image", type="filepath", sources=["upload"])
        with gr.Row():
            burn_subtitles = gr.Checkbox(value=True, label="Burn the section text on every video")
            output_resolution = gr.Dropdown(choices=["1280x720", "1920x1080"], value="1920x1080", label="Output resolution")
            subtitle_font = gr.Textbox(value="Nirmala UI", label="Text font")
            subtitle_font_size = gr.Slider(18, 48, value=28, step=1, label="Text size")
            max_clip_minutes = gr.Slider(1, 60, value=10, step=1, label="Maximum length of a combined clip (minutes)")
        video_transition = gr.Dropdown(choices=["None", "Crossfade", "Fade through black"], value="Crossfade", label="Section transition", visible=False)
        video_transition_duration = gr.Slider(0.1, 1.0, value=0.4, step=0.1, label="Transition duration", visible=False)
        final_video_filename = gr.Textbox(value="poet_reciting_poem_1080p.mp4", label="Final filename", visible=False)
        with gr.Row():
            pause_video_button = gr.Button("Pause after current video section")
            resume_video_button = gr.Button("Resume video generation")
            stop_video_button = gr.Button("Stop after current video section", variant="stop")
            combine_now_button = gr.Button("Combine finished section videos now")
        video_control_status = gr.Textbox(label="Video controls", lines=3, interactive=False)
        with gr.Group(visible=False) as colab_video_group:
            gr.Markdown("### Colab media server connection (Wan2.2 / LTX-2.5)")
            with gr.Row():
                video_colab_url = gr.Textbox(value=DEFAULT_COLAB_URL, label="Colab media server URL")
                video_colab_token = gr.Textbox(type="password", label="Colab bearer token (not saved)")
                check_video_colab_button = gr.Button("Check Colab server")
            video_colab_status = gr.Textbox(label="Colab status", lines=4, interactive=False, elem_classes=["status-output"])

        with gr.Group(visible=True) as sadtalker_video_group:
            with gr.Row():
                sadtalker_directory = gr.Textbox(value=DEFAULT_SADTALKER_DIR, label="SadTalker directory")
                sadtalker_environment = gr.Textbox(value=DEFAULT_SADTALKER_ENV, label="SadTalker Conda environment")
                check_sadtalker_button = gr.Button("Check SadTalker")
            sadtalker_status = gr.Textbox(label="SadTalker status", lines=5, interactive=False, elem_classes=["status-output"])
            with gr.Row():
                preprocess_mode = gr.Dropdown(choices=["crop", "resize", "full"], value="crop", label="Face preprocessing")
                video_model_size = gr.Dropdown(choices=[256, 512], value=256, label="SadTalker model resolution")
                still_mode = gr.Checkbox(value=True, label="Restrained / still mode")
            generate_video_button = gr.Button("Generate / Resume Video", variant="primary")
            video_progress_status = gr.Textbox(label="SadTalker progress", value="Generate the audio first (Tab 4).", lines=10, interactive=False, elem_classes=["status-output"])
            video_progress_bar = gr.Slider(0, 100, value=0, step=0.1, label="SadTalker progress (%)", interactive=False)
            final_video_preview = gr.Video(label="Final video (first combined clip)")
            with gr.Row():
                final_video_download = gr.File(label="Download combined clips / section videos", file_count="multiple")
                subtitle_download = gr.File(label="Download Hindi SRT")

        with gr.Group(visible=False) as wan_video_group:
            gr.Markdown("## Wan2.2 — Colab option")
            with gr.Row():
                wan_task = gr.Dropdown(choices=[("T2V-A14B", "t2v-A14B"), ("I2V-A14B", "i2v-A14B"), ("TI2V-5B", "ti2v-5B"), ("S2V-14B", "s2v-14B")], value="t2v-A14B", label="Wan2.2 mode")
                wan_size = gr.Dropdown(choices=["480*832", "832*480", "1280*720", "720*1280"], value="480*832", label="Size")
                wan_frame_num = gr.Slider(17, 121, value=17, step=4, label="Frames (4n+1)")
                wan_steps = gr.Slider(4, 50, value=8, step=1, label="Sampling steps")
                wan_seed = gr.Number(value=-1, precision=0, label="Seed")
            wan_checkpoint = gr.Textbox(value="", label="Wan checkpoint override (optional)")
            wan_audio_section = gr.Number(value=1, precision=0, label="S2V audio section")
            wan_prompt = gr.Textbox(label="Wan2.2 prompt", lines=5)
            with gr.Row():
                check_wan_colab_button = gr.Button("Check Colab + Wan2.2")
                generate_wan_button = gr.Button("Generate / Resume Wan2.2", variant="primary")
            wan_status = gr.Textbox(label="Wan2.2 Colab status", lines=8, interactive=False, elem_classes=["status-output"])
            wan_preview = gr.Video(label="Wan2.2 result")
            wan_download = gr.File(label="Download combined clips / section videos", file_count="multiple")

        with gr.Group(visible=False) as ltx_video_group:
            gr.Markdown("## LTX-2.5 — Colab option")
            with gr.Row():
                ltx_mode = gr.Dropdown(choices=[("Text-to-video", "t2v"), ("Image-to-video", "t2v_i2v_single"), ("Image + generated audio → video (A2Vid)", "a2vid")], value="t2v_i2v_single", label="LTX-2.5 mode")
                ltx_duration = gr.Slider(1, 10, value=5, step=1, label="Duration (seconds)")
                ltx_fps = gr.Dropdown(choices=[24, 25, 30], value=24, label="FPS")
            ltx_prompt = gr.Textbox(label="LTX-2.5 prompt", lines=6)
            ltx_audio = gr.Audio(label="Optional fallback audio for A2Vid (section audio from Tab 4 is used automatically)", type="filepath")
            gr.Markdown("LTX-2.5 A2Vid is the audio-conditioned path for exact image + supplied-audio workflows; the Colab server preserves the supplied waveform in the output.")
            with gr.Row():
                check_ltx_colab_button = gr.Button("Check Colab + LTX-2.5")
                generate_ltx_button = gr.Button("Generate / Resume LTX-2.5", variant="primary")
            ltx_status = gr.Textbox(label="LTX-2.5 Colab status", lines=8, interactive=False, elem_classes=["status-output"])
            ltx_preview = gr.Video(label="LTX-2.5 result")
            ltx_download = gr.File(label="Download combined clips / section videos", file_count="multiple")

        with gr.Group(visible=False) as ltx23_video_group:
            gr.Markdown(
                "## LTX-2.3 — ComfyUI / Colab GPU\n"
                "The audio of every section is taken from the project's audio folder automatically (generate it in Tab 4 first). "
                "A reference image is optional; if you omit it, the bridge reuses its previously uploaded reference image.\n\n"
                "**LTX-2.5 pose workflow:** choose it above and upload a reference video; the head, hand and body "
                "motion are copied from it. By default the reference image is the start frame (your person and studio); "
                "untick the box to start from the reference video's first frame. The reference must be 24 fps and at least "
                "as long as the longest render."
            )
            ltx23_bridge_url = gr.Textbox(
                value=DEFAULT_LTX23_BRIDGE_URL,
                label="LTX-2.3 Colab bridge public URL (port 8190)",
                placeholder="https://your-public-tunnel.example",
            )
            ltx23_bridge_token = gr.Textbox(
                value=os.getenv("F5PDFSTUDIO_LTX23_API_KEY", ""),
                label="Optional bridge API key",
                type="password",
            )
            with gr.Row():
                ltx23_duration = gr.Slider(2, 60, value=20, step=1, label="Maximum seconds per render (longer section audio is cut at pauses, rendered in pieces and joined)")
                ltx23_fps = gr.Dropdown(choices=[16, 20, 24, 25, 30], value=24, label="FPS")
                ltx23_width = gr.Dropdown(choices=[512, 640, 768, 832, 1024, 1280, 1536], value=1280, label="Width")
                ltx23_height = gr.Dropdown(choices=[320, 384, 480, 512, 576, 720, 832, 1024], value=720, label="Height")
                ltx23_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            ltx23_workflow = gr.Radio(
                choices=[("LTX-2.3: image + audio", "image"), ("LTX-2.5 pose: audio + reference video", "pose")],
                value="image",
                label="Workflow",
            )
            ltx23_ref_video = gr.Video(
                label="Reference video for LTX-2.5 pose (24 fps, at least as long as the clip duration; empty = reuse the last one sent to Colab)",
                sources=["upload"],
            )
            ltx23_prompt = gr.Textbox(
                label="LTX-2.3 prompt (optional; blank uses each text section)",
                lines=5,
                placeholder="Describe the motion and scene; leave blank to include the current Hindi text section.",
            )
            with gr.Row():
                ltx23_extra_positive = gr.Textbox(
                    label="Additional positive prompt (optional, added after the prompt above)",
                    lines=2,
                    placeholder="e.g. soft warm studio light, subtle head nods, expressive hands",
                )
                ltx23_extra_negative = gr.Textbox(
                    label="Additional negative prompt (optional, added to the built-in negative prompt)",
                    lines=2,
                    placeholder="e.g. blurry, distorted hands, extra fingers, flicker, subtitles, watermark",
                )
            ltx23_pose_use_image = gr.Checkbox(
                value=True,
                label="Pose workflow: use the reference image as the start frame (identity + studio). "
                      "Untick to start from the reference video's first frame instead.",
            )
            ltx23_cfg = gr.Slider(
                1.0, 4.0, value=1.0, step=0.1,
                label="Negative prompt strength (CFG, first stage). 1.0 = negative prompt OFF (fastest); raise to 1.5-3 to use it (slower)",
            )
            ltx23_image = gr.Image(
                label="Optional reference image (reused by the Colab bridge when omitted later)",
                type="filepath",
                sources=["upload"],
            )
            ltx23_audio = gr.Audio(
                label="Optional fallback audio (the section audio from Tab 4 is used automatically)",
                type="filepath",
                sources=["upload"],
            )
            with gr.Row():
                check_ltx23_bridge_button = gr.Button("Check LTX-2.3 ComfyUI bridge")
                generate_ltx23_button = gr.Button("Generate / Resume LTX-2.3", variant="primary")
            ltx23_status = gr.Textbox(label="LTX-2.3 status", lines=8, interactive=False, elem_classes=["status-output"])
            ltx23_preview = gr.Video(label="LTX-2.3 result")
            ltx23_download = gr.File(label="Download combined clips / section videos", file_count="multiple")

    with gr.Tab("Cloud AI"):
        gr.Markdown("## Gemini API — separate cloud tab")
        with gr.Row():
            gemini_api_key = gr.Textbox(label="Gemini API key (not saved)", type="password")
            check_gemini_button = gr.Button("Check Gemini connection")
        gemini_status = gr.Textbox(label="Gemini status", lines=4, interactive=False, elem_classes=["status-output"])
        gr.Markdown("### Gemini TTS")
        with gr.Row():
            gemini_tts_model = gr.Dropdown(choices=["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-pro-preview-tts", "gemini-2.5-flash-preview-tts"], value=GEMINI_DEFAULT_TTS_MODEL, label="TTS model")
            gemini_tts_voice = gr.Dropdown(choices=["Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"], value=GEMINI_DEFAULT_TTS_VOICE, label="Voice")
            gemini_tts_chunk_size = gr.Slider(500, 3000, value=1800, step=100, label="TTS section target")
        gemini_style_prompt = gr.Textbox(value="Natural Hindi audiobook narration. Clear Hindi diction, warm and expressive voice, moderate pace, subtle emotion. Read the supplied text exactly.", label="TTS performance direction", lines=4)
        with gr.Row():
            pause_gemini_button = gr.Button("Pause after current section")
            resume_gemini_button = gr.Button("Resume")
            stop_gemini_button = gr.Button("Stop after current section", variant="stop")
        gemini_control_status = gr.Textbox(label="Gemini controls", lines=2, interactive=False)
        generate_gemini_audio_button = gr.Button("Generate / Resume Gemini TTS audiobook", variant="primary")
        gemini_audio_status = gr.Textbox(label="Gemini TTS status", lines=10, interactive=False, elem_classes=["status-output"])
        gemini_audio_download = gr.File(label="Gemini audiobook WAV")
        gr.Markdown("### Gemini Veo")
        gemini_veo_prompt = gr.Textbox(label="Veo scene prompt", lines=6)
        with gr.Row():
            gemini_veo_model = gr.Dropdown(choices=["veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-lite-generate-preview"], value=GEMINI_DEFAULT_VEO_MODEL, label="Veo model")
            gemini_veo_resolution = gr.Dropdown(choices=["720p", "1080p"], value=GEMINI_DEFAULT_VEO_RESOLUTION, label="Resolution")
            gemini_veo_aspect = gr.Dropdown(choices=["16:9", "9:16", "1:1"], value=GEMINI_DEFAULT_VEO_ASPECT_RATIO, label="Aspect ratio")
        gemini_veo_reference = gr.Image(label="Optional Veo reference image", type="filepath", sources=["upload"])
        gemini_veo_audio = gr.Audio(label="Optional generated audio to use as final audio (Veo does not lip-sync to external WAV)", type="filepath")
        gr.Markdown("For true image + generated-audio conditioning/lip-sync, use LTX-2.5 A2Vid in Tab 5. Gemini Veo can generate the visual clip from the image and then replace its native audio with this WAV.")
        generate_gemini_video_button = gr.Button("Generate / Resume Gemini Veo", variant="primary")
        gemini_video_status = gr.Textbox(label="Gemini Veo status", lines=8, interactive=False, elem_classes=["status-output"])
        gemini_video_preview = gr.Video(label="Gemini Veo video")
        gemini_video_download = gr.File(label="Gemini Veo clips / section videos", file_count="multiple")

    with gr.Tab("System Status"):
        check_button = gr.Button("Check complete installation", variant="primary")
        setup_report = gr.Textbox(label="Local installation report", lines=22, interactive=False, elem_classes=["status-output"])
        colab_setup_status = gr.Textbox(label="Colab media server", lines=8, interactive=False, elem_classes=["status-output"])
        unload_f5_button = gr.Button("Unload F5-TTS model from GPU")
        unload_f5_status = gr.Textbox(label="F5-TTS status", interactive=False)

    # -------------------- events --------------------
    extract_event = extract_button.click(fn=extract_uploaded_document, inputs=[document_file, selection_range, ocr_language, ocr_dpi, word_image_ocr], outputs=[extracted_text, extraction_status], concurrency_limit=1, concurrency_id="document_processing")
    clean_event = clean_button.click(fn=prepare_clean_text, inputs=extracted_text, outputs=[extracted_text, text_status])
    detect_event = detect_button.click(fn=analyse_chapters, inputs=[extracted_text, max_characters, book_type], outputs=[section_preview, text_status])
    check_ollama_button.click(fn=check_ollama_from_ui, inputs=[], outputs=ollama_status)
    gemini_correct_event = gemini_correct_button.click(fn=gemini_correct_text_ui, inputs=[extracted_text, gemini_correction_model, correction_language, book_type, gemini_correction_chunk_size, gemini_correction_api_key], outputs=[extracted_text, gemini_correction_status], concurrency_limit=1, concurrency_id="gemini_processing")
    ollama_correct_event = ollama_correct_button.click(fn=ollama_correct_text, inputs=[extracted_text, ollama_model, correction_language, book_type, ollama_chunk_size], outputs=[extracted_text, ollama_status], concurrency_limit=1, concurrency_id="ollama_processing")
    ollama_chapters_event = ollama_chapters_button.click(fn=ollama_detect_chapters, inputs=[extracted_text, ollama_model, book_type], outputs=ollama_status, concurrency_limit=1, concurrency_id="ollama_processing")
    unload_ollama_button.click(fn=unload_ollama_from_ui, inputs=ollama_model, outputs=ollama_status)
    project_event = project_button.click(fn=prepare_project, inputs=[project_name, book_name, part_number, author_name], outputs=[project_status, project_output_path])
    refresh_projects_button.click(fn=refresh_project_list, inputs=existing_project, outputs=existing_project)

    correction_engine.change(
        fn=_correction_engine_visibility, inputs=[correction_engine],
        outputs=[ollama_correction_group, gemini_correction_group]
    )
    tts_engine.change(
        fn=_audio_engine_visibility, inputs=[tts_engine],
        outputs=[reference_audio_group, f5tts_audio_group, voicestudio_audio_group, voicestudio_refresh_group, voicebox_audio_group, voicebox_refresh_group, sarvam_audio_group, fishspeech_audio_group, gemini_audio_group, colab_audio_group]
    )
    video_engine.change(
        fn=_video_engine_visibility, inputs=[video_engine],
        outputs=[sadtalker_video_group, wan_video_group, ltx_video_group, ltx23_video_group, colab_video_group]
    )

    state_outputs = [project_name, book_name, part_number, author_name, project_status, project_output_path, book_type, extracted_text, extraction_status, text_status, ollama_status, section_preview, reference_audio, reference_transcript, progress_status, final_audio, wav_download, mp3_download, audio_progress_bar, poet_image, video_progress_status, final_video_preview, final_video_download, subtitle_download, video_progress_bar, final_video_filename, max_characters, tts_engine, voicestudio_voice, voicebox_profile, voicebox_language, voicebox_model_size, voicebox_engine, sarvam_speaker, sarvam_api_key, fish_reference_id, colab_url, colab_token, device_choice, speech_speed, pause_duration, poetry_line_pause_duration, seed, nfe_steps, create_mp3, sadtalker_directory, sadtalker_environment, preprocess_mode, video_model_size, still_mode, output_resolution, burn_subtitles, subtitle_font, subtitle_font_size, video_transition, video_transition_duration, sadtalker_status, wan_checkpoint, wan_task, wan_prompt, wan_status, wan_size, wan_frame_num, wan_steps, wan_seed, wan_audio_section, ltx_status, ltx_mode, ltx_prompt, ltx_duration, ltx_fps]
    existing_project.change(fn=open_selected_project, inputs=existing_project, outputs=state_outputs).then(fn=refresh_pipeline_status, inputs=project_output_path, outputs=pipeline_status_box, concurrency_id="pipeline_status")

    persist_inputs = [project_output_path, extracted_text, book_type, extraction_status, text_status, ollama_status, section_preview, reference_audio, reference_transcript, poet_image, final_video_filename, max_characters, tts_engine, voicestudio_voice, voicebox_profile, voicebox_language, voicebox_model_size, voicebox_engine, sarvam_speaker, sarvam_api_key, fish_reference_id, colab_url, colab_token, device_choice, speech_speed, pause_duration, poetry_line_pause_duration, seed, nfe_steps, create_mp3, sadtalker_directory, sadtalker_environment, preprocess_mode, video_model_size, still_mode, output_resolution, burn_subtitles, subtitle_font, subtitle_font_size, video_transition, video_transition_duration, wan_checkpoint, wan_task, wan_prompt, wan_size, wan_frame_num, wan_steps, wan_seed, wan_audio_section, ltx_mode, ltx_prompt, ltx_duration, ltx_fps]
    project_saved_event = project_event.then(fn=persist_project_tabs, inputs=persist_inputs, outputs=[])
    for ev in (extract_event, clean_event, detect_event, ollama_correct_event, ollama_chapters_event, gemini_correct_event):
        ev.then(fn=persist_project_tabs, inputs=persist_inputs, outputs=[])

    refresh_vs_voices_button.click(fn=refresh_voicestudio_voice_choices, inputs=[], outputs=voicestudio_voice)
    refresh_voicebox_button.click(fn=refresh_voicebox_profile_choices, inputs=[], outputs=voicebox_profile)
    refresh_fish_references_button.click(fn=refresh_fish_reference_choices, inputs=[colab_url, colab_token], outputs=[fish_reference_id, fish_reference_status])
    check_colab_button.click(fn=check_colab_from_ui, inputs=[colab_url, colab_token], outputs=colab_status)
    check_wan_colab_button.click(fn=check_colab_from_ui, inputs=[video_colab_url, video_colab_token], outputs=wan_status)
    check_ltx_colab_button.click(fn=check_colab_from_ui, inputs=[video_colab_url, video_colab_token], outputs=ltx_status)
    check_video_colab_button.click(fn=check_colab_from_ui, inputs=[video_colab_url, video_colab_token], outputs=video_colab_status)
    colab_url.change(fn=lambda value: value, inputs=colab_url, outputs=video_colab_url)  # same server by default

    generate_button.click(fn=start_generation, inputs=[extracted_text, book_type, book_name, part_number, author_name, reference_audio, reference_transcript, project_name, max_characters, tts_engine, voicestudio_voice, voicebox_profile, voicebox_language, voicebox_model_size, voicebox_engine, sarvam_speaker, sarvam_api_key, fish_reference_id, gemini_api_key_audio, gemini_audio_model, gemini_voice_id, colab_url, colab_token, speech_speed, pause_duration, poetry_line_pause_duration, device_choice, seed, nfe_steps, create_mp3, ollama_model], outputs=[progress_status, final_audio, wav_download, mp3_download, audio_progress_bar], concurrency_limit=1, concurrency_id="generation")
    pause_button.click(fn=request_pause, inputs=[], outputs=progress_status, concurrency_id="generation_controls")
    resume_button.click(fn=request_resume, inputs=[], outputs=progress_status, concurrency_id="generation_controls")
    stop_button.click(fn=request_stop, inputs=[], outputs=progress_status, concurrency_id="generation_controls")

    check_sadtalker_button.click(fn=check_sadtalker_from_ui, inputs=[sadtalker_directory, sadtalker_environment], outputs=sadtalker_status)
    generate_video_button.click(fn=start_video_generation, inputs=[extracted_text, project_name, max_characters, book_type, poet_image, sadtalker_directory, sadtalker_environment, preprocess_mode, video_model_size, still_mode, burn_subtitles, subtitle_font, subtitle_font_size, output_resolution, ollama_model, video_transition, video_transition_duration, book_name, author_name, part_number, final_video_filename, max_clip_minutes], outputs=[video_progress_status, final_video_preview, final_video_download, subtitle_download, video_progress_bar], concurrency_limit=1, concurrency_id="video_generation")
    pause_video_button.click(fn=request_video_pause, inputs=[], outputs=video_progress_status, concurrency_id="video_controls")
    resume_video_button.click(fn=request_video_resume, inputs=[], outputs=video_progress_status, concurrency_id="video_controls")
    stop_video_button.click(fn=request_video_stop, inputs=[], outputs=video_progress_status, concurrency_id="video_controls")

    generate_wan_button.click(fn=start_remote_wan_video_generation, inputs=[extracted_text, project_name, book_name, author_name, part_number, poet_image, video_colab_url, video_colab_token, wan_task, wan_checkpoint, wan_prompt, wan_size, wan_frame_num, wan_steps, wan_seed, wan_audio_section, burn_subtitles, output_resolution, subtitle_font, subtitle_font_size, max_clip_minutes], outputs=[wan_status, wan_preview, wan_download], concurrency_limit=1, concurrency_id="video_generation")
    generate_ltx_button.click(fn=start_remote_ltx_video_generation, inputs=[extracted_text, project_name, book_name, author_name, part_number, poet_image, video_colab_url, video_colab_token, ltx_mode, ltx_prompt, ltx_duration, ltx_fps, ltx_audio, burn_subtitles, output_resolution, subtitle_font, subtitle_font_size, max_clip_minutes], outputs=[ltx_status, ltx_preview, ltx_download], concurrency_limit=1, concurrency_id="video_generation")
    check_ltx23_bridge_button.click(
        fn=check_ltx23_from_ui,
        inputs=[ltx23_bridge_url, ltx23_bridge_token],
        outputs=ltx23_status,
    )
    generate_ltx23_button.click(
        fn=start_remote_ltx23_video_generation,
        inputs=[
            extracted_text, project_name, book_name, author_name, part_number,
            poet_image, ltx23_image, ltx23_bridge_url, ltx23_bridge_token, ltx23_prompt,
            ltx23_duration, ltx23_fps, ltx23_width, ltx23_height, ltx23_seed,
            ltx23_audio, ltx23_ref_video, ltx23_workflow,
            ltx23_extra_positive, ltx23_extra_negative, ltx23_cfg, ltx23_pose_use_image,
            burn_subtitles, output_resolution, subtitle_font, subtitle_font_size, max_clip_minutes,
        ],
        outputs=[ltx23_status, ltx23_preview, ltx23_download],
        concurrency_limit=1,
        concurrency_id="video_generation",
    )

    def _create_gemini_voice_ui(reference_audio_path, consent_audio_path, api_key):
        if not reference_audio_path or not consent_audio_path:
            raise gr.Error('Provide the reference voice WAV and the required consent recording.')
        if not str(api_key or '').strip() and not os.getenv('GEMINI_API_KEY'):
            raise gr.Error('Enter the Gemini API key.')
        try:
            voice_id = create_replicated_voice(reference_audio_path, consent_audio_path, api_key=str(api_key or '').strip() or None, display_name='F5PdfStudio Replicated Voice')
        except Exception as exc:
            raise gr.Error(f'Gemini voice replication failed:\n{exc}') from exc
        return voice_id, f'Gemini replicated voice created: {voice_id}'

    create_gemini_voice_button.click(fn=_create_gemini_voice_ui, inputs=[reference_audio, gemini_consent_audio, gemini_api_key_audio], outputs=[gemini_voice_id, gemini_voice_status])
    check_gemini_button.click(fn=check_gemini_from_ui, inputs=gemini_api_key, outputs=gemini_status)
    generate_gemini_audio_button.click(fn=start_gemini_audio, inputs=[extracted_text, project_name, book_name, part_number, author_name, book_type, gemini_api_key, gemini_tts_model, gemini_tts_voice, gemini_style_prompt, gemini_tts_chunk_size], outputs=[gemini_audio_status, gemini_audio_download], concurrency_limit=1, concurrency_id="gemini_generation")
    generate_gemini_video_button.click(fn=start_gemini_video, inputs=[extracted_text, project_name, book_name, part_number, author_name, gemini_api_key, gemini_veo_model, gemini_veo_resolution, gemini_veo_aspect, gemini_veo_prompt, gemini_veo_reference, gemini_veo_audio, burn_subtitles, output_resolution, subtitle_font, subtitle_font_size, max_clip_minutes], outputs=[gemini_video_status, gemini_video_preview, gemini_video_download], concurrency_limit=1, concurrency_id="video_generation")

    # ---- pause / resume / stop on every tab (they act only after the current unit of work) ----
    pause_extract_button.click(fn=_pause_stage("extract"), inputs=[], outputs=extraction_status, concurrency_id="extract_controls")
    resume_extract_button.click(fn=_resume_stage("extract"), inputs=[], outputs=extraction_status, concurrency_id="extract_controls")
    stop_extract_button.click(fn=_stop_stage("extract"), inputs=[], outputs=extraction_status, concurrency_id="extract_controls")
    pause_correct_button.click(fn=_pause_stage("correct"), inputs=[], outputs=text_status, concurrency_id="correct_controls")
    resume_correct_button.click(fn=_resume_stage("correct"), inputs=[], outputs=text_status, concurrency_id="correct_controls")
    stop_correct_button.click(fn=_stop_stage("correct"), inputs=[], outputs=text_status, concurrency_id="correct_controls")
    pause_gemini_button.click(fn=_pause_stage("gemini", True), inputs=[], outputs=gemini_control_status, concurrency_id="gemini_controls")
    resume_gemini_button.click(fn=_resume_stage("gemini", True), inputs=[], outputs=gemini_control_status, concurrency_id="gemini_controls")
    stop_gemini_button.click(fn=_stop_stage("gemini", True), inputs=[], outputs=gemini_control_status, concurrency_id="gemini_controls")
    combine_now_button.click(fn=combine_finished_videos, inputs=[project_output_path, max_clip_minutes], outputs=[video_control_status, final_video_preview, final_video_download], concurrency_id="video_combine")

    # ---- pipeline status: refreshed on open/create, by the button and every 5 seconds ----
    refresh_pipeline_button.click(fn=refresh_pipeline_status, inputs=project_output_path, outputs=pipeline_status_box, concurrency_id="pipeline_status")
    project_event.then(fn=refresh_pipeline_status, inputs=project_output_path, outputs=pipeline_status_box, concurrency_id="pipeline_status")
    if pipeline_timer is not None:
        pipeline_timer.tick(fn=refresh_pipeline_status, inputs=project_output_path, outputs=pipeline_status_box, concurrency_id="pipeline_status", show_progress="hidden")
    extract_event.then(fn=lambda path, text: _record_text_stage("extract", path, text), inputs=[project_output_path, extraction_status], outputs=[])
    ollama_correct_event.then(fn=lambda path, text: _record_text_stage("correct", path, text), inputs=[project_output_path, ollama_status], outputs=[])
    gemini_correct_event.then(fn=lambda path, text: _record_text_stage("correct", path, text), inputs=[project_output_path, gemini_correction_status], outputs=[])

    check_button.click(fn=check_installation, inputs=[], outputs=setup_report)
    check_button.click(fn=check_colab_from_ui, inputs=[colab_url, colab_token], outputs=colab_setup_status)
    unload_f5_button.click(fn=unload_model, inputs=[], outputs=unload_f5_status)


# ================================================================
# Start
# ================================================================

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1,
        max_size=20,
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        css=CUSTOM_CSS,
    )
