"""F5-TTS audiobook generation service for F5PdfStudio.

Features
--------
* SPRINGLab/F5-Hindi-24KHz voice cloning.
* Safe sentence batching to avoid F5-TTS positional tensor overflows.
* Conservative OCR/newline cleanup before synthesis.
* Pause, resume and stop controls.
* Per-section WAV output and atomic JSON checkpoints.
* Combined WAV and optional MP3 output.
* Model caching and explicit GPU-memory unloading.

The public API intentionally accepts several legacy keyword aliases because
different F5PdfStudio ``app.py`` revisions used slightly different names.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import re
import hashlib
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from tts_provider import synthesize_sarvam, synthesize_voicestudio, synthesize_voicebox
from remote_colab_service import fish_speech_tts


SAMPLE_RATE = 24_000
DEFAULT_MODEL_REPO = "SPRINGLab/F5-Hindi-24KHz"
DEFAULT_CHECKPOINT_FILE = "model_2500000.safetensors"
DEFAULT_VOCAB_FILE = "vocab.txt"
DEFAULT_REFERENCE_TEXT = (
    "चींटियाँ ऊपर चढ़ने लगी थीं। बूढ़े हाथी ने यह देखा और बात उसकी "
    "समझ में आ गई, पर उसने किसी से कहा नहीं। जंगल में वह सबसे बड़ा था "
    "और सबसे भारी भी"
)

HINDI_MODEL_CONFIG: dict[str, Any] = {
    "dim": 768,
    "depth": 18,
    "heads": 12,
    "ff_mult": 2,
    "text_dim": 512,
    "text_mask_padding": False,
    "conv_layers": 4,
    "pe_attn_head": 1,
}


import pipeline_service as _pipeline


class GenerationStopped(RuntimeError):
    """Raised internally when the user presses Stop."""


@dataclass
class GenerationResult:
    status: str
    output_dir: str
    section_files: list[str]
    combined_wav: Optional[str]
    combined_mp3: Optional[str]
    checkpoint_file: str
    completed_sections: int
    total_sections: int
    stopped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_dir": self.output_dir,
            "section_files": self.section_files,
            "individual_wavs": self.section_files,
            "combined_wav": self.combined_wav,
            "wav_file": self.combined_wav,
            "combined_mp3": self.combined_mp3,
            "mp3_file": self.combined_mp3,
            "checkpoint_file": self.checkpoint_file,
            "completed_sections": self.completed_sections,
            "total_sections": self.total_sections,
            "stopped": self.stopped,
        }


_pause_event = threading.Event()
_stop_event = threading.Event()
_control_lock = threading.RLock()
_model_lock = threading.RLock()

# Set means paused.  It begins cleared (running).
_pause_event.clear()
_stop_event.clear()

_cached_model: Any = None
_cached_vocoder: Any = None
_cached_device: Optional[str] = None
_cached_checkpoint: Optional[str] = None
_cached_vocab: Optional[str] = None


def request_pause() -> str:
    with _control_lock:
        _pause_event.set()
    return "Pause requested. Generation will pause after the current audio batch."


def request_resume() -> str:
    with _control_lock:
        _pause_event.clear()
    return "Generation resumed."


def request_stop() -> str:
    with _control_lock:
        _stop_event.set()
        _pause_event.clear()  # release a paused worker so it can stop
    return "Stop requested. The checkpoint will be preserved."


def clear_control_flags() -> str:
    with _control_lock:
        _pause_event.clear()
        _stop_event.clear()
    return "Generation controls reset."


def is_paused() -> bool:
    return _pause_event.is_set()


def is_stop_requested() -> bool:
    return _stop_event.is_set()


def _check_controls(
    progress_callback: Optional[Callable[..., Any]] = None,
) -> None:
    if _stop_event.is_set():
        raise GenerationStopped("Generation stopped by user.")

    announced = False
    while _pause_event.is_set():
        if _stop_event.is_set():
            raise GenerationStopped("Generation stopped by user.")
        if not announced:
            _notify(progress_callback, "Generation paused. Press Resume to continue.")
            announced = True
        time.sleep(0.25)


def _notify(callback: Optional[Callable[..., Any]], message: str, *args: Any) -> None:
    if callback is None:
        return
    try:
        callback(message, *args)
    except TypeError:
        try:
            callback(message)
        except Exception:
            pass
    except Exception:
        pass


def prepare_text_for_tts(text: str) -> str:
    """Normalize OCR layout without changing the words or language."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Normalize ASCII poetry bars and Devanagari double danda.  The Hindi F5
    # model is more stable with one ordinary danda per line boundary.
    text = text.replace("||", "।")
    text = text.replace("|", "।")
    text = text.replace("॥", "।")

    # Protect paragraph breaks before joining OCR-wrapped single lines.
    paragraph_marker = " F5PDFSTUDIO_PARAGRAPH_BREAK "
    text = re.sub(r"\n[ \t]*\n+", paragraph_marker, text)
    # Join OCR-wrapped lines when the preceding line does not end a sentence.
    text = re.sub(r"(?<![।.!?;:])\n", " ", text)
    # A paragraph break receives one sentence boundary, not many seconds.
    text = text.replace(paragraph_marker, "। ")
    text = re.sub(r"\n+", " ", text)

    # Remove Markdown/OCR decoration while keeping ordinary hyphenated words.
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*[-_=*]{3,}\s*$", " ", text)

    # Pipes plus paragraph markers can produce "। ।".  F5-TTS may repeat the
    # preceding word when it encounters consecutive pause tokens.
    text = re.sub(r"(?:।\s*){2,}", "। ", text)
    text = re.sub(r"([,;:!?])\1+", r"\1", text)
    text = re.sub(r"\s+([,.;:!?।])", r"\1", text)
    text = re.sub(r"([,;:!?।])(?=\S)", r"\1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip(" \t\n।") + ("।" if text.strip() else "")


def split_text_for_inference(text: str, max_chars: int = 260) -> list[str]:
    """Create short synthesis batches at natural Hindi/English boundaries."""
    text = prepare_text_for_tts(text)
    if not text:
        return []
    max_chars = max(120, min(int(max_chars), 420))

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[।.!?])\s+", text)
        if item.strip()
    ]
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        clauses = [
            item.strip()
            for item in re.split(r"(?<=[,;:])\s+", sentence)
            if item.strip()
        ]
        current = ""
        for clause in clauses:
            candidate = f"{current} {clause}".strip()
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = clause
            elif len(clause) > max_chars:
                if current:
                    pieces.append(current)
                    current = ""
                # Word-safe fallback for an OCR sentence with no punctuation.
                words = clause.split()
                word_buffer = ""
                for word in words:
                    word_candidate = f"{word_buffer} {word}".strip()
                    if word_buffer and len(word_candidate) > max_chars:
                        pieces.append(word_buffer)
                        word_buffer = word
                    else:
                        word_buffer = word_candidate
                if word_buffer:
                    current = word_buffer
            else:
                current = candidate
        if current:
            pieces.append(current)

    # Do not merge adjacent sentences back together.  Keeping every danda-
    # terminated sentence in its own inference call prevents the Hindi model
    # from skipping words at poetry line boundaries.
    return pieces


def _resolve_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        candidate = Path(value).expanduser()
        return candidate
    # Gradio file values can expose .name or be dictionaries.
    if isinstance(value, dict):
        for key in ("path", "name"):
            if value.get(key):
                return Path(str(value[key])).expanduser()
    name = getattr(value, "name", None)
    return Path(str(name)).expanduser() if name else None


def _device_name(requested: Optional[str] = None) -> str:
    if requested and requested.lower() not in {"auto", "default"}:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _download_hindi_files(
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
) -> tuple[str, str]:
    checkpoint = _resolve_path(checkpoint_path)
    vocab = _resolve_path(vocab_path)
    if checkpoint and checkpoint.exists() and vocab and vocab.exists():
        return str(checkpoint.resolve()), str(vocab.resolve())

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    if not checkpoint or not checkpoint.exists():
        checkpoint = Path(
            hf_hub_download(DEFAULT_MODEL_REPO, DEFAULT_CHECKPOINT_FILE)
        )
    if not vocab or not vocab.exists():
        vocab = Path(hf_hub_download(DEFAULT_MODEL_REPO, DEFAULT_VOCAB_FILE))
    return str(checkpoint.resolve()), str(vocab.resolve())


def _call_supported(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Pass only keyword arguments supported by the installed F5-TTS version."""
    try:
        signature = inspect.signature(function)
        if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            return function(*args, **kwargs)
        allowed = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return function(*args, **allowed)
    except (TypeError, ValueError):
        return function(*args, **kwargs)


def load_hindi_model(
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[..., Any]] = None,
) -> tuple[Any, Any, str]:
    """Load and cache the Hindi F5-TTS model and Vocos vocoder."""
    global _cached_model, _cached_vocoder, _cached_device
    global _cached_checkpoint, _cached_vocab

    selected_device = _device_name(device)
    checkpoint, vocab = _download_hindi_files(checkpoint_path, vocab_path)

    with _model_lock:
        if (
            _cached_model is not None
            and _cached_vocoder is not None
            and _cached_device == selected_device
            and _cached_checkpoint == checkpoint
            and _cached_vocab == vocab
        ):
            return _cached_model, _cached_vocoder, selected_device

        unload_model()
        _notify(progress_callback, "Loading F5-TTS Hindi model…")
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            try:
                from f5_tts.model import DiT
            except ImportError:
                from f5_tts.model.backbones.dit import DiT
        except ImportError as exc:
            raise RuntimeError(
                "F5-TTS is not installed in this environment. Run: pip install f5-tts"
            ) from exc

        model = _call_supported(
            load_model,
            DiT,
            HINDI_MODEL_CONFIG,
            checkpoint,
            vocab_file=vocab,
            device=selected_device,
            mel_spec_type="vocos",
            use_ema=True,
        )
        vocoder = _call_supported(
            load_vocoder,
            vocoder_name="vocos",
            is_local=False,
            device=selected_device,
        )

        _cached_model = model
        _cached_vocoder = vocoder
        _cached_device = selected_device
        _cached_checkpoint = checkpoint
        _cached_vocab = vocab
        return model, vocoder, selected_device


def unload_model() -> str:
    """Release cached F5-TTS and Vocos objects and clear CUDA memory."""
    global _cached_model, _cached_vocoder, _cached_device
    global _cached_checkpoint, _cached_vocab
    with _model_lock:
        _cached_model = None
        _cached_vocoder = None
        _cached_device = None
        _cached_checkpoint = None
        _cached_vocab = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass
    return "F5-TTS model unloaded from memory."


def _to_numpy(audio: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for audio generation.") from exc

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32).squeeze()
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.isfinite(array).all():
        array = np.nan_to_num(array)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array.astype(np.float32, copy=False)


def _limit_long_silences(
    audio: Any,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -46.0,
    maximum_silence: float = 0.45,
) -> Any:
    """Cap only abnormally long silence; natural short pauses remain intact."""
    import numpy as np

    audio = _to_numpy(audio)
    if audio.size == 0:
        return audio
    threshold = 10.0 ** (threshold_db / 20.0)
    silent = np.abs(audio) < threshold
    max_samples = int(maximum_silence * sample_rate)
    if max_samples <= 0:
        return audio

    output: list[Any] = []
    start = 0
    index = 0
    length = len(audio)
    while index < length:
        if not silent[index]:
            index += 1
            continue
        silence_start = index
        while index < length and silent[index]:
            index += 1
        silence_length = index - silence_start
        output.append(audio[start:silence_start])
        if silence_length > max_samples:
            output.append(audio[silence_start : silence_start + max_samples])
        else:
            output.append(audio[silence_start:index])
        start = index
    output.append(audio[start:])
    return np.concatenate([part for part in output if len(part)])


def _trim_edge_silence(
    audio: Any,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -48.0,
    leading_ms: int = 50,
    trailing_ms: int = 70,
) -> Any:
    """Keep short natural margins while removing long model-start/end silence."""
    import numpy as np

    audio = _to_numpy(audio)
    if audio.size == 0:
        return audio
    threshold = 10.0 ** (threshold_db / 20.0)
    active = np.flatnonzero(np.abs(audio) >= threshold)
    if active.size == 0:
        return audio
    start = max(0, int(active[0]) - int(sample_rate * leading_ms / 1000))
    end = min(
        len(audio),
        int(active[-1]) + 1 + int(sample_rate * trailing_ms / 1000),
    )
    return audio[start:end]


def _write_wav(path: Path, audio: Any, sample_rate: int = SAMPLE_RATE) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    array = _to_numpy(audio)
    pcm = np.clip(array, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)


def _synthesize_batch(
    text: str,
    reference_audio: str,
    reference_text: str,
    model: Any,
    vocoder: Any,
    device: str,
    speed: float,
    seed: Optional[int],
) -> tuple[Any, int]:
    try:
        from f5_tts.infer.utils_infer import infer_process
    except ImportError as exc:
        raise RuntimeError("Cannot import F5-TTS infer_process.") from exc

    kwargs = {
        "mel_spec_type": "vocos",
        "speed": float(speed),
        "device": device,
        "seed": seed,
        "show_info": lambda *_args, **_kwargs: None,
        "progress": None,
    }
    result = _call_supported(
        infer_process,
        reference_audio,
        reference_text,
        text,
        model,
        vocoder,
        **kwargs,
    )
    if not isinstance(result, tuple):
        audio, sample_rate = result, SAMPLE_RATE
    elif len(result) >= 2:
        audio, sample_rate = result[0], result[1]
    else:
        audio, sample_rate = result[0], SAMPLE_RATE
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        sample_rate = SAMPLE_RATE
    return _to_numpy(audio), sample_rate


def generate_section_audio(
    text: str,
    reference_audio: Any,
    reference_text: str = DEFAULT_REFERENCE_TEXT,
    output_path: Optional[str | Path] = None,
    *,
    checkpoint_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    device: Optional[str] = None,
    speed: float = 1.0,
    seed: Optional[int] = None,
    max_batch_chars: int = 260,
    inter_batch_pause_ms: int = 120,
    progress_callback: Optional[Callable[..., Any]] = None,
) -> tuple[Any, int]:
    """Generate one logical section, internally split into safe short batches."""
    import numpy as np

    ref_path = _resolve_path(reference_audio)
    if not ref_path or not ref_path.exists():
        raise FileNotFoundError("Reference audio file was not found.")
    reference_text = prepare_text_for_tts(reference_text or DEFAULT_REFERENCE_TEXT)
    if not reference_text:
        raise ValueError("Reference transcript is empty.")

    batches = split_text_for_inference(text, max_batch_chars)
    if not batches:
        raise ValueError("Section text is empty after cleanup.")

    model, vocoder, selected_device = load_hindi_model(
        checkpoint_path, vocab_path, device, progress_callback
    )
    generated: list[Any] = []
    output_rate = SAMPLE_RATE
    for number, batch in enumerate(batches, start=1):
        # Pause/Stop are honoured only between sections (see generate_audiobook).
        _notify(
            progress_callback,
            f"Generating audio batch {number}/{len(batches)}…",
            number,
            len(batches),
        )
        try:
            audio, rate = _synthesize_batch(
                batch,
                str(ref_path.resolve()),
                reference_text,
                model,
                vocoder,
                selected_device,
                speed,
                seed,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "size of tensor" in message or "8192" in message:
                raise RuntimeError(
                    "F5-TTS received a text batch that was too long. Reduce "
                    "max_batch_chars to 180-220. Original error: " + message
                ) from exc
            raise
        output_rate = rate
        audio = _limit_long_silences(audio, rate)
        audio = _trim_edge_silence(audio, rate)
        generated.append(audio)
        if number < len(batches):
            generated.append(np.zeros(int(rate * inter_batch_pause_ms / 1000), dtype=np.float32))

    complete = np.concatenate(generated) if generated else np.zeros(0, dtype=np.float32)
    complete = _limit_long_silences(complete, output_rate)
    if output_path:
        _write_wav(Path(output_path), complete, output_rate)
    return complete, output_rate


def _normalise_sections(sections: Any) -> list[dict[str, Any]]:
    if isinstance(sections, str):
        raw_items: Sequence[Any] = [part for part in re.split(r"\n\s*\n+", sections) if part.strip()]
    elif isinstance(sections, dict):
        raw_items = list(sections.values())
    else:
        raw_items = list(sections or [])

    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("body") or "")
            title = str(item.get("title") or item.get("name") or f"Section {index}")
        else:
            text = str(item)
            title = f"Section {index}"
        cleaned = prepare_text_for_tts(text)
        if cleaned:
            result.append({"index": index, "title": title, "text": cleaned})
    return result


def _checkpoint_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def combine_wav_files(
    files: Iterable[str | Path],
    output_path: str | Path,
    gap_ms: int = 450,
) -> str:
    """Combine PCM WAV sections without requiring pydub."""
    import numpy as np

    arrays: list[Any] = []
    expected_rate: Optional[int] = None
    for file in files:
        path = Path(file)
        if not path.exists():
            continue
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise RuntimeError(f"Unsupported WAV format: {path}")
            rate = handle.getframerate()
            if expected_rate is None:
                expected_rate = rate
            elif rate != expected_rate:
                raise RuntimeError("Section WAV files have different sample rates.")
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            arrays.append(pcm.astype(np.float32) / 32767.0)
            arrays.append(np.zeros(int(rate * gap_ms / 1000), dtype=np.float32))
    if not arrays or expected_rate is None:
        raise RuntimeError("No completed section WAV files are available to combine.")
    if arrays:
        arrays.pop()  # no silence after final section
    output = np.concatenate(arrays)
    destination = Path(output_path)
    _write_wav(destination, output, expected_rate)
    return str(destination.resolve())


def convert_wav_to_mp3(
    wav_path: str | Path,
    mp3_path: str | Path,
    bitrate: str = "192k",
) -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if not executable:
        return None
    destination = Path(mp3_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate,
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return str(destination.resolve())


def _split_text_for_remote_tts(text: str, max_chars: int = 2400) -> list[str]:
    """Split remote-TTS text at natural boundaries while staying below API limits."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    limit = max(500, min(int(max_chars), 2400))
    if len(text) <= limit:
        return [text]

    units = [u.strip() for u in re.split(r"(?<=[।.!?])\s+|\n+", text) if u.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > limit:
            words = unit.split()
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and len(candidate) > limit:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _generate_remote_section(
    text: str,
    output_path: Path,
    *,
    provider_engine: str,
    voicestudio_voice: str,
    sarvam_api_key: str,
    sarvam_speaker: str,
    fish_reference_id: str,
    gemini_api_key: str,
    gemini_model: str,
    gemini_voice_id: str,
    voicebox_profile: str,
    voicebox_language: str,
    voicebox_model_size: str,
    voicebox_engine: str,
    colab_url: str,
    colab_token: str,
    speed: float,
    progress_callback: Optional[Callable[..., Any]],
    inter_batch_pause_ms: int = 120,
) -> None:
    """Generate one section through VoiceStudio or Sarvam and write one WAV."""
    import numpy as np

    engine = str(provider_engine or "").strip().lower()
    if engine not in {"voicestudio", "voicebox", "sarvam", "fishspeech", "gemini"}:
        raise ValueError(f"Unsupported remote TTS engine: {provider_engine}")

    chunks = _split_text_for_remote_tts(text, 2400 if engine == "sarvam" else (800 if engine == "gemini" else 1800))
    if not chunks:
        raise ValueError("Section text is empty after cleanup.")

    output_rate = 24000
    arrays: list[Any] = []
    for number, chunk in enumerate(chunks, start=1):
        # Pause/Stop are honoured only between sections (see generate_audiobook).
        _notify(progress_callback, f"Generating {engine} audio batch {number}/{len(chunks)}…", number - 1, len(chunks))
        temporary = output_path.with_suffix(f".remote_{number:04d}.wav")
        try:
            if engine == "sarvam":
                synthesize_sarvam(
                    chunk,
                    temporary,
                    api_key=sarvam_api_key,
                    speaker=sarvam_speaker,
                    language_code="hi-IN",
                    pace=float(speed),
                )
            elif engine == "voicestudio":
                synthesize_voicestudio(
                    chunk,
                    temporary,
                    voice=voicestudio_voice or "default",
                    response_format="wav",
                )
            elif engine == "voicebox":
                synthesize_voicebox(
                    chunk,
                    temporary,
                    profile_id=voicebox_profile,
                    language=voicebox_language or "hi",
                    model_size=voicebox_model_size or "0.6B",
                    engine=voicebox_engine or "qwen",
                )
            elif engine == "gemini":
                from gemini_service import generate_tts_wav
                generate_tts_wav(
                    chunk, temporary, model=gemini_model or "gemini-3.8-flash-tts",
                    voice=gemini_voice_id or "Kore", api_key=gemini_api_key or None,
                    replicated_voice_id=gemini_voice_id if str(gemini_voice_id).startswith("voice_") else "",
                    style_prompt="Natural Hindi audiobook narration. Clear Hindi diction, warm, expressive, steady pace.",
                )
            else:
                fish_speech_tts(
                    base_url=colab_url,
                    token=colab_token,
                    text=chunk,
                    output_path=temporary,
                    reference_id=fish_reference_id,
                    language="hi",
                )
            with wave.open(str(temporary), "rb") as handle:
                if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                    raise RuntimeError(f"Remote TTS returned unsupported WAV format: {temporary}")
                output_rate = handle.getframerate()
                pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
                arrays.append(pcm)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
        if number < len(chunks) and inter_batch_pause_ms > 0:
            arrays.append(np.zeros(int(output_rate * inter_batch_pause_ms / 1000), dtype=np.float32))

    if not arrays:
        raise RuntimeError("Remote TTS returned no audio.")
    _write_wav(output_path, np.concatenate(arrays), output_rate)


def generate_audiobook(
    sections: Any = None,
    reference_audio: Any = None,
    reference_text: str = DEFAULT_REFERENCE_TEXT,
    output_dir: str | Path = "outputs",
    checkpoint_file: Optional[str | Path] = None,
    resume: bool = True,
    output_format: str = "both",
    speed: float = 1.0,
    seed: Optional[int] = None,
    max_batch_chars: int = 240,
    inter_batch_pause_ms: int = 120,
    inter_section_pause_ms: int = 450,
    model_checkpoint: Optional[str] = None,
    vocab_path: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate and checkpoint a complete audiobook.

    Legacy aliases accepted through ``kwargs`` include ``texts``, ``ref_audio``,
    ``ref_text``, ``result_dir``, ``checkpoint_path`` and ``resume_generation``.
    """
    sections = sections if sections is not None else kwargs.get("texts", kwargs.get("section_texts"))
    reference_audio = reference_audio or kwargs.get("ref_audio") or kwargs.get("reference_audio_path")
    reference_text = kwargs.get("ref_text", reference_text)
    output_dir = kwargs.get("result_dir", kwargs.get("output_folder", output_dir))
    project_name = str(kwargs.get("project_name") or "").strip()
    if project_name and str(output_dir) == "outputs":
        safe_project_name = re.sub(r"[^A-Za-z0-9._-]+", "_", project_name).strip("._-")
        output_dir = Path(output_dir) / (safe_project_name or "audiobook")
    checkpoint_file = checkpoint_file or kwargs.get("checkpoint_path")
    if "resume_generation" in kwargs:
        resume = bool(kwargs["resume_generation"])
    model_checkpoint = model_checkpoint or kwargs.get("ckpt_path") or kwargs.get("model_path")
    vocab_path = vocab_path or kwargs.get("vocab_file")
    progress_callback = progress_callback or kwargs.get("status_callback") or kwargs.get("callback")
    if "pause_ms" in kwargs:
        inter_section_pause_ms = int(kwargs["pause_ms"])
    if "device_choice" in kwargs:
        device = kwargs["device_choice"]
    if "create_mp3" in kwargs:
        output_format = "both" if bool(kwargs["create_mp3"]) else "wav"

    provider_engine = str(kwargs.get("provider_engine") or "f5tts").strip().lower()
    voicestudio_voice = str(kwargs.get("voicestudio_voice") or "default").strip()
    sarvam_api_key = str(kwargs.get("sarvam_api_key") or "").strip()
    sarvam_speaker = str(kwargs.get("sarvam_speaker") or "shubh").strip().lower()
    fish_reference_id = str(kwargs.get("fish_reference_id") or "").strip()
    gemini_api_key = str(kwargs.get("gemini_api_key") or "").strip()
    gemini_model = str(kwargs.get("gemini_model") or "gemini-3.8-flash-tts").strip()
    gemini_voice_id = str(kwargs.get("gemini_voice_id") or "").strip()
    voicebox_profile = str(kwargs.get("voicebox_profile") or "").strip()
    voicebox_language = str(kwargs.get("voicebox_language") or "hi").strip()
    voicebox_model_size = str(kwargs.get("voicebox_model_size") or "0.6B").strip()
    voicebox_engine = str(kwargs.get("voicebox_engine") or "qwen").strip()
    colab_url = str(kwargs.get("colab_url") or "").strip()
    colab_token = str(kwargs.get("colab_token") or "").strip()

    normalised = _normalise_sections(sections)
    if not normalised:
        raise ValueError("No non-empty text sections were supplied for generation.")
    # Resume only when the exact section list is unchanged. If the user edits
    # the corrected text, starting from an old checkpoint could attach old WAVs
    # to new text, so invalidate that checkpoint safely.
    sections_fingerprint = hashlib.sha256(
        "\n\n".join(str(item.get("text") or "") for item in normalised).encode("utf-8")
    ).hexdigest()
    ref_path = _resolve_path(reference_audio)
    if provider_engine == "f5tts":
        if not ref_path or not ref_path.exists():
            raise FileNotFoundError("Upload a valid reference-audio WAV file for F5-Hindi.")
        reference_text = prepare_text_for_tts(reference_text or DEFAULT_REFERENCE_TEXT)
    else:
        reference_text = prepare_text_for_tts(reference_text or "")

    destination = Path(output_dir).expanduser().resolve()
    sections_dir = _pipeline.audio_directory(destination)       # one audio folder for every tool
    _pipeline.migrate_legacy_audio(destination)                 # section_0001.wav -> <book>_<part>_001.wav
    # Keep an independent resume checkpoint for every audio engine. This prevents
    # switching engines from destroying the previous engine's resume point.
    if checkpoint_file:
        checkpoint = Path(checkpoint_file).expanduser().resolve()
    else:
        checkpoint = destination / f"generation_checkpoint_{provider_engine}.json"

    previous = _checkpoint_data(checkpoint) if resume else {}
    previous_engine = str(previous.get("provider_engine") or provider_engine).strip().lower()
    previous_voice = str(previous.get("provider_voice") or "").strip().lower()
    if provider_engine == "f5tts":
        current_voice = str(ref_path or "")
    elif provider_engine == "voicestudio":
        current_voice = voicestudio_voice
    elif provider_engine == "voicebox":
        current_voice = f"{voicebox_profile}|{voicebox_language}|{voicebox_model_size}|{voicebox_engine}"
    elif provider_engine == "fishspeech":
        current_voice = fish_reference_id
    elif provider_engine == "gemini":
        current_voice = f"{gemini_model}|{gemini_voice_id}"
    else:
        current_voice = sarvam_speaker
    if previous and (previous_engine != provider_engine or previous_voice != current_voice.lower()):
        previous = {}
    if previous and str(previous.get("sections_fingerprint") or "") != sections_fingerprint:
        previous = {}
    completed = {
        int(value)
        for value in previous.get("completed_sections", [])
        if str(value).isdigit()
    }
    section_files: list[str] = []
    total = len(normalised)
    stopped = False
    clear_control_flags()
    _pipeline.update_stage(destination, "audio", "running", done=len(completed), total=total, engine=provider_engine,
                           message=f"Audio: {len(completed)}/{total} sections already done.")

    checkpoint_state: dict[str, Any] = {
        "version": 2,
        "status": "running",
        "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sections": total,
        "completed_sections": sorted(completed),
        "sections_fingerprint": sections_fingerprint,
        "last_completed_section": max(completed) if completed else 0,
        "reference_audio": str(ref_path.resolve()) if ref_path else None,
        "output_dir": str(destination),
        "provider_engine": provider_engine,
        "provider_voice": current_voice,
        "model": DEFAULT_MODEL_REPO if provider_engine == "f5tts" else ("voice-studio-local" if provider_engine == "voicestudio" else ("fish-speech-colab" if provider_engine == "fishspeech" else (gemini_model if provider_engine == "gemini" else "bulbul:v3"))),
    }
    _save_checkpoint(checkpoint, checkpoint_state)

    try:
        for ordinal, section in enumerate(normalised, start=1):
            section_file = _pipeline.audio_section_path(destination, ordinal)
            if resume and ordinal in completed and section_file.exists():
                section_files.append(str(section_file.resolve()))
                _notify(
                    progress_callback,
                    f"Using completed section {ordinal}/{total}",
                    ordinal,
                    total,
                )
                continue

            if is_paused():
                _pipeline.update_stage(destination, "audio", "paused", done=len(completed), total=total, engine=provider_engine,
                                       message=f"Paused after section {max(completed) if completed else 0}/{total}. Press Resume.")
            _check_controls(progress_callback)
            _pipeline.update_stage(destination, "audio", "running", done=len(completed), total=total, engine=provider_engine)
            _notify(
                progress_callback,
                f"Generating section {ordinal}/{total}: {section['title']}",
                ordinal - 1,
                total,
            )
            try:
                if provider_engine == "f5tts":
                    generate_section_audio(
                        section["text"],
                        str(ref_path),
                        reference_text,
                        section_file,
                        checkpoint_path=model_checkpoint,
                        vocab_path=vocab_path,
                        device=device,
                        speed=float(speed),
                        seed=seed,
                        max_batch_chars=int(max_batch_chars),
                        inter_batch_pause_ms=int(inter_batch_pause_ms),
                        progress_callback=progress_callback,
                    )
                else:
                    _generate_remote_section(
                        section["text"],
                        section_file,
                        provider_engine=provider_engine,
                        voicestudio_voice=voicestudio_voice,
                        sarvam_api_key=sarvam_api_key,
                        sarvam_speaker=sarvam_speaker,
                        fish_reference_id=fish_reference_id,
                        gemini_api_key=gemini_api_key,
                        gemini_model=gemini_model,
                        gemini_voice_id=gemini_voice_id,
                        voicebox_profile=voicebox_profile,
                        voicebox_language=voicebox_language,
                        voicebox_model_size=voicebox_model_size,
                        voicebox_engine=voicebox_engine,
                        colab_url=colab_url,
                        colab_token=colab_token,
                        speed=float(speed),
                        progress_callback=progress_callback,
                        inter_batch_pause_ms=int(inter_batch_pause_ms),
                    )
            except GenerationStopped:
                raise
            except Exception as exc:
                checkpoint_state.update({
                    "status": "failed",
                    "failed_section": ordinal,
                    "error": str(exc),
                    "completed_sections": sorted(completed),
                    "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                _save_checkpoint(checkpoint, checkpoint_state)
                _pipeline.update_stage(destination, "audio", "failed", done=len(completed), total=total, engine=provider_engine,
                                       message=f"Failed at section {ordinal}/{total}: {str(exc)[:200]}")
                raise RuntimeError(
                    f"Generation failed at section {ordinal}/{total}. "
                    f"The checkpoint was saved.\n{exc}"
                ) from exc

            completed.add(ordinal)
            section_files.append(str(section_file.resolve()))
            checkpoint_state.update({
                "status": "running",
                "completed_sections": sorted(completed),
                "last_completed_section": ordinal,
                "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            _save_checkpoint(checkpoint, checkpoint_state)
            _pipeline.update_stage(destination, "audio", "running", done=len(completed), total=total, engine=provider_engine,
                                   message=f"Section {ordinal}/{total} finished: {section_file.name}")
            _notify(
                progress_callback,
                f"Completed section {ordinal}/{total}",
                ordinal,
                total,
            )
    except GenerationStopped:
        stopped = True
        checkpoint_state.update({
            "status": "stopped",
            "completed_sections": sorted(completed),
            "last_completed_section": max(completed) if completed else 0,
            "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        _save_checkpoint(checkpoint, checkpoint_state)

    # Include valid completed files from earlier resumed work in numerical order.
    section_files = [
        str(_pipeline.audio_section_path(destination, number).resolve())
        for number in sorted(completed)
        if _pipeline.audio_section_path(destination, number).exists()
    ]

    combined_wav: Optional[str] = None
    combined_mp3: Optional[str] = None
    if section_files and (not stopped or len(completed) == total):
        _notify(progress_callback, "Combining completed WAV sections…", len(completed), total)
        combined_wav = combine_wav_files(
            section_files,
            destination / "audiobook_complete.wav",
            gap_ms=int(inter_section_pause_ms),
        )
        if str(output_format).lower() in {"mp3", "both", "wav+mp3", "all"}:
            combined_mp3 = convert_wav_to_mp3(
                combined_wav,
                destination / "audiobook_complete.mp3",
            )

    final_status = "stopped" if stopped else "completed"
    _pipeline.update_stage(
        destination, "audio", final_status, done=len(completed), total=total, engine=provider_engine,
        message=(f"Stopped after section {max(completed) if completed else 0}/{total}. Generate again to resume."
                 if stopped else f"All {total} section audio files are in {sections_dir.name}/ as <book>_<part>_<nnn>.wav."),
    )
    checkpoint_state.update({
        "status": final_status,
        "completed_sections": sorted(completed),
        "combined_wav": combined_wav,
        "combined_mp3": combined_mp3,
        "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_checkpoint(checkpoint, checkpoint_state)
    _notify(
        progress_callback,
        "Generation stopped; checkpoint saved." if stopped else "Audiobook generation completed.",
        len(completed),
        total,
    )

    return GenerationResult(
        status=final_status,
        output_dir=str(destination),
        section_files=section_files,
        combined_wav=combined_wav,
        combined_mp3=combined_mp3,
        checkpoint_file=str(checkpoint),
        completed_sections=len(completed),
        total_sections=total,
        stopped=stopped,
    ).as_dict()


# Compatibility aliases used by some earlier app.py versions.
generate_book_audio = generate_audiobook
generate_tts_section = generate_section_audio
reset_control_flags = clear_control_flags


__all__ = [
    "SAMPLE_RATE",
    "DEFAULT_REFERENCE_TEXT",
    "HINDI_MODEL_CONFIG",
    "GenerationStopped",
    "request_pause",
    "request_resume",
    "request_stop",
    "clear_control_flags",
    "reset_control_flags",
    "is_paused",
    "is_stop_requested",
    "prepare_text_for_tts",
    "split_text_for_inference",
    "load_hindi_model",
    "unload_model",
    "generate_section_audio",
    "generate_tts_section",
    "combine_wav_files",
    "convert_wav_to_mp3",
    "generate_audiobook",
    "generate_book_audio",
]
