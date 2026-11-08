from __future__ import annotations

import base64
import os
import re
import time
import wave
from pathlib import Path
from typing import Callable, Optional

try:
    from google.genai import types  # type: ignore
except Exception:
    types = None


DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_TTS_VOICE = "Kore"
DEFAULT_TEXT_CORRECTION_MODEL = "gemini-2.5-flash"
DEFAULT_VEO_MODEL = "veo-3.1-generate-preview"
DEFAULT_VEO_RESOLUTION = "720p"
DEFAULT_VEO_ASPECT_RATIO = "16:9"


def _client(api_key: Optional[str] = None):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK is not installed. Run: pip install -U google-genai"
        ) from exc

    key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Gemini API key is not configured. Set GEMINI_API_KEY in the "
            "environment or enter it in the Gemini tab."
        )
    return genai.Client(api_key=key)


def check_gemini(api_key: Optional[str] = None) -> str:
    """Validate SDK/key and perform a lightweight model listing check."""
    try:
        client = _client(api_key)
        # A lightweight authenticated call. The SDK may expose models.list as
        # an iterator; consuming one item avoids a large response.
        models = client.models.list()
        first = next(iter(models), None)
        name = getattr(first, "name", "authenticated") if first else "authenticated"
        return f"Gemini connection OK. {name}"
    except Exception as exc:
        return f"Gemini connection failed: {type(exc).__name__}: {exc}"


def _write_audio_wav(path: Path, audio_bytes: bytes, channels: int = 1, rate: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        path.write_bytes(audio_bytes)
        return
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio_bytes)


def _extract_audio_bytes(interaction) -> bytes:
    data = getattr(interaction, "output_audio", None)
    if data is not None:
        raw = getattr(data, "data", None)
        if raw:
            return base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)

    # SDK versions can expose the same audio in a generated-content shape.
    candidates = getattr(interaction, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            raw = getattr(inline, "data", None) if inline else None
            if raw:
                return base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)

    raise RuntimeError("Gemini returned no audio data.")


def generate_tts_wav(
    text: str,
    output_path: str | Path,
    *,
    voice: str = DEFAULT_TTS_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    style_prompt: str = "Natural Hindi audiobook narration. Clear diction, warm, expressive, steady pace.",
    api_key: Optional[str] = None,
    replicated_voice_id: str = "",
) -> Path:
    """Generate one Gemini TTS WAV file at 24 kHz mono PCM."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("Gemini TTS text is empty.")

    client = _client(api_key)
    prompt = f"{style_prompt}\n\nRead the following text exactly as written. Do not summarize or add words:\n{text}"
    selected_model = str(model or DEFAULT_TTS_MODEL).strip()
    selected_voice = str(replicated_voice_id or voice or DEFAULT_TTS_VOICE).strip()
    if replicated_voice_id and selected_model not in {"gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"}:
        selected_model = "gemini-3.8-flash-tts"

    # Current Gemini 3.1 TTS uses the Interactions API. Keep a fallback for
    # SDK/model combinations that still expose the generateContent interface.
    interaction_error = None
    try:
        interaction = client.interactions.create(
            model=selected_model,
            input=[{
                "type": "user_input",
                "content": [{
                    "type": "text",
                    "text": text,
                    "annotations": [{"type": "speech_metadata", "style": style_prompt}],
                }],
            }],
            response_format={"type": "audio"},
            generation_config={
                "speech_config": [
                    {"voice": selected_voice}
                ]
            },
        )
        pcm = _extract_audio_bytes(interaction)
    except Exception as exc:
        interaction_error = exc
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=selected_voice,
                            )
                        )
                    ),
                ),
            )
            pcm = _extract_audio_bytes(response)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Gemini TTS failed. Interactions error: "
                f"{interaction_error}; generateContent fallback error: {fallback_exc}"
            ) from fallback_exc

    destination = Path(output_path)
    _write_audio_wav(destination, pcm)
    return destination



def create_replicated_voice(reference_audio: str | Path, consent_audio: str | Path, *, api_key: Optional[str] = None, display_name: str = "F5PdfStudio Voice") -> str:
    """Create a persistent Gemini Voice Replication profile after consent verification."""
    source = Path(reference_audio)
    consent = Path(consent_audio)
    if not source.is_file():
        raise FileNotFoundError(f"Reference voice WAV not found: {source}")
    if not consent.is_file():
        raise FileNotFoundError(f"Consent WAV not found: {consent}")
    client = _client(api_key)
    source_b64 = base64.b64encode(source.read_bytes()).decode("utf-8")
    consent_b64 = base64.b64encode(consent.read_bytes()).decode("utf-8")
    voice = client.voices.create(
        store=True,
        voice={
            "model": "gemini-3.8-flash-tts",
            "type": "replicated",
            "display_name": str(display_name or "F5PdfStudio Voice"),
            "replicated": {
                "source_audio": {"mime_type": "audio/wav", "data": source_b64},
                "consent_audio": {"mime_type": "audio/wav", "data": consent_b64},
            },
        },
    )
    voice_id = getattr(voice, "id", None)
    if not voice_id:
        raise RuntimeError("Gemini Voice Replication succeeded but returned no voice ID.")
    return str(voice_id)


def list_replicated_voices(api_key: Optional[str] = None) -> list[dict]:
    client = _client(api_key)
    result = []
    for voice in client.voices.list():
        item = {
            "id": str(getattr(voice, "id", "") or ""),
            "name": str(getattr(voice, "display_name", "") or getattr(voice, "name", "") or ""),
            "type": str(getattr(voice, "type", "") or ""),
        }
        if item["id"]:
            result.append(item)
    return result


def _preserve_structure_ok(original: str, corrected: str) -> bool:
    """Reject corrections that materially alter paragraph/label structure."""
    def shape(text: str):
        lines = [x.strip() for x in str(text).splitlines() if x.strip()]
        labels = [x for x in lines if re.match(r"^(?:[\w\u0900-\u097F .'-]{1,40})\s*[:：]", x)]
        return len(lines), len(labels), [x.split(':', 1)[0].strip() for x in labels]
    ol, oc, on = shape(original)
    cl, cc, cn = shape(corrected)
    if ol and abs(cl - ol) > max(2, int(ol * 0.25)):
        return False
    if oc != cc and (oc or cc):
        return False
    if on and cn and on != cn:
        return False
    return True


def _is_gemini_rate_limit_error(exc: Exception) -> bool:
    """Return True for Gemini/API quota errors that are normally retryable."""
    message = str(exc or "")
    lowered = message.lower()
    return (
        "429" in lowered
        or "resource_exhausted" in lowered
        or "too many requests" in lowered
        or "rate limit" in lowered
        or "quota exceeded" in lowered
    )


def _gemini_retry_delay(exc: Exception, attempt: int) -> float:
    """Extract Gemini RetryInfo delay, falling back to bounded exponential backoff."""
    message = str(exc or "")
    patterns = (
        r"retryDelay\"\s*:\s*\"(\d+(?:\.\d+)?)s",
        r"retryDelay\s*[=:]\s*[\"']?(\d+(?:\.\d+)?)s",
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"retry after\s+(\d+(?:\.\d+)?)s",
        r"after\s+(\d+(?:\.\d+)?)\s*seconds",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            try:
                return max(1.0, min(float(match.group(1)), 300.0))
            except (TypeError, ValueError):
                pass
    # Gemini may omit RetryInfo. Keep the wait bounded and progressively longer.
    return float(min(30 * (2 ** max(0, attempt - 1)), 120))


def correct_hindi_text(text: str, *, model: str = DEFAULT_TEXT_CORRECTION_MODEL, language: str = "Hindi/English", book_type: str = "Others", chunk_size: int = 1200, api_key: Optional[str] = None, progress: Optional[Callable[[int, int, str], None]] = None, control: Optional[Callable[[], None]] = None, resume_chunks: Optional[dict] = None, on_chunk: Optional[Callable[[int, int, str], None]] = None) -> tuple[str, list[str]]:
    """Correct OCR/spelling/grammar while preserving drama/paragraph boundaries.

    Gemini HTTP 429/resource-exhausted responses are retried automatically up to
    three times, honoring the server-provided retry delay when available.
    Non-rate-limit errors are not retried.
    """
    source = str(text or "").strip()
    if not source:
        raise ValueError("Text is empty.")
    limit = max(500, min(int(chunk_size or 1200), 6000))
    paragraphs = source.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > limit:
            chunks.append(current); current = para
        else:
            current = candidate
    if current: chunks.append(current)
    client = _client(api_key)
    results, warnings = [], []
    for idx, chunk in enumerate(chunks, 1):
        if resume_chunks and idx in resume_chunks:
            results.append(str(resume_chunks[idx]))
            if progress: progress(idx, len(chunks), f"Using saved Gemini correction {idx}/{len(chunks)}")
            continue
        if control is not None:
            control()  # pause / stop only between chunks
        prompt = f"""You are a careful Hindi OCR correction editor.\nLanguage scope: {language}. Book type: {book_type}.\n\nCorrect only OCR mistakes, spelling, punctuation, and clear grammar errors. Preserve meaning and wording. Preserve every paragraph, line, scene heading, character label, dialogue boundary, and stage direction. Do not add, remove, summarize, modernize, or rewrite content. If uncertain, keep the original wording. Return ONLY the corrected text.\n\nTEXT:\n{chunk}"""
        corrected = ""
        last_exc: Optional[Exception] = None
        max_rate_limit_retries = 3
        for attempt in range(1, max_rate_limit_retries + 2):
            try:
                response = client.models.generate_content(
                    model=str(model or DEFAULT_TEXT_CORRECTION_MODEL),
                    contents=prompt,
                    config={"temperature": 0.1},
                )
                corrected = str(getattr(response, "text", "") or "").strip()
                if not corrected:
                    raise RuntimeError("Gemini returned empty correction.")
                break
            except Exception as exc:
                last_exc = exc
                if not _is_gemini_rate_limit_error(exc) or attempt > max_rate_limit_retries:
                    break
                delay = _gemini_retry_delay(exc, attempt)
                if progress:
                    progress(
                        max(0, idx - 1) / max(1, len(chunks)),
                        len(chunks),
                        f"Gemini rate limit (429) on section {idx}/{len(chunks)}; waiting {delay:.0f}s before retry {attempt}/{max_rate_limit_retries}...",
                    )
                time.sleep(delay)

        try:
            if not corrected:
                raise last_exc or RuntimeError("Gemini returned empty correction.")
            if not _preserve_structure_ok(chunk, corrected):
                warnings.append(f"Section {idx}/{len(chunks)} kept unchanged: Gemini changed paragraph/label boundaries.")
                corrected = chunk
        except Exception as exc:
            if _is_gemini_rate_limit_error(exc):
                warnings.append(
                    f"Section {idx}/{len(chunks)} kept unchanged after {max_rate_limit_retries} automatic 429 retries: {exc}"
                )
            else:
                warnings.append(f"Section {idx}/{len(chunks)} kept unchanged: {exc}")
            corrected = chunk
        results.append(corrected)
        if on_chunk is not None:
            on_chunk(idx, len(chunks), corrected)
        if progress: progress(idx, len(chunks), f"Gemini correction {idx}/{len(chunks)} complete")
    return "\n\n".join(results), warnings

def generate_tts_sections(
    sections: list[str],
    output_directory: str | Path,
    *,
    voice: str = DEFAULT_TTS_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    style_prompt: str = "Natural Hindi audiobook narration. Clear diction, warm, expressive, steady pace.",
    api_key: Optional[str] = None,
    replicated_voice_id: str = "",
    progress: Optional[Callable[[int, int, str], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    file_for: Optional[Callable[[int], Path]] = None,
    combined_path: Optional[str | Path] = None,
    control: Optional[Callable[[], None]] = None,
    on_section: Optional[Callable[[int, int, Path], None]] = None,
) -> tuple[list[Path], list[Path]]:
    """Generate section WAVs and a combined WAV. Existing section WAVs resume.

    ``file_for(index)`` supplies the shared project audio path (``<book>_<part>_<nnn>.wav``),
    ``control`` is called between sections (pause blocks, stop raises) and ``on_section``
    is called after every finished section.
    """
    out = Path(output_directory)
    sections_dir = out / "gemini_audio_sections"
    if file_for is None:
        sections_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for index, section in enumerate(sections, start=1):
        if control is not None:
            control()
        if stop_requested and stop_requested():
            break
        path = file_for(index) if file_for else sections_dir / f"section_{index:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size < 1000:
            generate_tts_wav(
                section,
                path,
                voice=voice,
                model=model,
                style_prompt=style_prompt,
                api_key=api_key,
                replicated_voice_id=replicated_voice_id,
            )
        generated.append(path)
        if on_section is not None:
            on_section(index, len(sections), path)
        if progress:
            progress(index, len(sections), f"Gemini audio section {index}/{len(sections)} complete")

    combined = Path(combined_path) if combined_path else out / "gemini_audiobook.wav"
    _combine_wavs(generated, combined)
    return generated, [combined] if combined.exists() else []


def _combine_wavs(paths: list[Path], output: Path):
    valid = [p for p in paths if p.exists()]
    if not valid:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(valid[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for path in valid[1:]:
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != params.nchannels
                or wf.getsampwidth() != params.sampwidth
                or wf.getframerate() != params.framerate
            ):
                raise RuntimeError(f"Incompatible WAV format in {path.name}.")
            frames.append(wf.readframes(wf.getnframes()))

    with wave.open(str(output), "wb") as out_wav:
        out_wav.setparams(params)
        for data in frames:
            out_wav.writeframes(data)


def generate_veo_video(
    prompt: str,
    output_path: str | Path,
    *,
    model: str = DEFAULT_VEO_MODEL,
    resolution: str = DEFAULT_VEO_RESOLUTION,
    aspect_ratio: str = DEFAULT_VEO_ASPECT_RATIO,
    api_key: Optional[str] = None,
    image_path: Optional[str | Path] = None,
    poll_seconds: int = 10,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Generate a Veo clip with bounded retry handling for transient backend failures.

    Retries only server-side/internal failures (gRPC/HTTP code 13 or equivalent
    internal-server wording). Invalid requests, authentication errors, and quota
    errors are surfaced immediately.
    """
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Veo prompt is empty.")

    client = _client(api_key)
    from google.genai import types

    kwargs = {
        "model": model,
        "prompt": prompt,
        "config": types.GenerateVideosConfig(
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            number_of_videos=1,
        ),
    }

    if image_path:
        if isinstance(image_path, dict):
            image_path = image_path.get("path") or image_path.get("name") or image_path.get("file")
        image_path = Path(str(image_path))
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Veo reference image not found: {image_path}")
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise ValueError("The reference image file is empty.")
        kwargs["image"] = types.Image(image_bytes=image_bytes, mime_type=mime_type)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        if progress:
            progress(f"Submitting Veo video generation job (attempt {attempt}/{max_attempts})…")
        try:
            operation = client.models.generate_videos(**kwargs)
            while not operation.done:
                if progress:
                    progress(f"Veo video is rendering… (attempt {attempt}/{max_attempts})")
                time.sleep(max(2, int(poll_seconds)))
                operation = client.operations.get(operation)

            error_obj = getattr(operation, "error", None)
            response = getattr(operation, "response", None)
            generated_videos = getattr(response, "generated_videos", None) or []
            if not generated_videos:
                error_text = str(error_obj or operation)
                lowered = error_text.lower()
                transient = (
                    "code': 13" in lowered
                    or '"code": 13' in lowered
                    or "code=13" in lowered
                    or "internal server" in lowered
                    or "internal_server" in lowered
                    or "status': 'internal" in lowered
                )
                if transient and attempt < max_attempts:
                    delay = 15 * attempt
                    if progress:
                        progress(f"Veo temporary internal-server failure. Retrying in {delay}s…")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Veo completed without a video: {operation}")

            video = generated_videos[0].video
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.files.download(file=video, destination=str(destination))
            if not destination.exists() or destination.stat().st_size < 1000:
                raise RuntimeError("Veo reported success but no usable MP4 was downloaded.")
            if progress:
                progress(f"Veo video saved: {destination}")
            return destination
        except Exception as exc:
            text = str(exc or "")
            lowered = text.lower()
            transient = (
                "code': 13" in lowered
                or '"code": 13' in lowered
                or "code=13" in lowered
                or "internal server" in lowered
                or "internal_server" in lowered
            )
            if transient and attempt < max_attempts:
                delay = 15 * attempt
                if progress:
                    progress(f"Veo temporary internal-server failure. Retrying in {delay}s…")
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("Veo generation failed after retries.")


def mux_external_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> Path:
    """Replace a Veo video's native audio with an externally generated WAV.
    This is muxing, not audio-conditioned lip-sync; LTX-2.5 A2Vid is the exact audio-conditioned option.
    """
    import subprocess, shutil
    video = Path(video_path); audio = Path(audio_path); output = Path(output_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to mux external Gemini/TTS audio.")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output

# ================================================================
# Portrait + supplied-WAV talking-head preflight / generation plan
# ================================================================

TALKING_HEAD_POEM = """खंड 1

मातृभूमि की करें वंदना,
आओ हम सब पहले मिलकर।

सुरभित फिर कर दें स्वदेश को,
बनें सुमन हम स्वयं विहँसकर।

खंड 2

त्याग, तपस्या और प्रेम की
धारा बहे राष्ट्र-जीवन में।

एक अमर संगीत गुंजरित
रहे सदा ही घर-आँगन में।

खंड 3

भारत फिर भारत बन जाए,
लौटे स्वर्ग इसी धरती पर।

मातृभूमि की करें वंदना,
आओ हम सब पहले मिलकर।

खंड 4

उत्तर में ऊँचा-सा हिमगिरि,
मातृभूमि का महिमा-मंडल।

गंगा-जमुना की जलधारा,
द्रवीभूत उसका करुण मन।

खंड 5

विंध्य-शृंखला बनी मेखला,
पूरब से पश्चिम को जाकर।

मातृभूमि की करें वंदना,
आओ हम सब शीश झुकाकर।"""

TALKING_HEAD_PROMPT = """Use the attached portrait as the only visual identity reference and the attached WAV file as the only voice reference.

The user confirms that they own these materials or have the depicted person's informed permission to animate his image and reproduce his voice.

Create a photorealistic talking-head video of the man reciting the complete Hindi poem provided below.

IMPORTANT AUDIO INSTRUCTIONS

1. First determine whether the attached WAV already contains the complete poem:
   - If it does, use that original recording directly as the video audio.
   - Do not regenerate or alter the voice unnecessarily.
2. If the WAV is only a voice sample, reproduce the same speaker's voice, Hindi pronunciation, accent, pitch, rhythm and mature vocal character.
3. Do not invent words or change the poem.
4. Use a dignified, patriotic and emotionally expressive delivery.
5. Pause naturally at commas, line endings and stanza boundaries.
6. Maintain consistent loudness and voice identity throughout.
7. Do not add music, sound effects or another speaker.

VISUAL INSTRUCTIONS

- Preserve the man's exact facial identity, age, complexion, white hair, clothing and podcast-studio setting.
- Preserve the microphone and desk arrangement.
- Keep the man seated and looking generally toward the camera.
- Animate only natural lip movement, blinking, breathing, slight eyebrow movement and subtle head movement.
- Use restrained expressions appropriate for a dignified patriotic poem.
- Maintain accurate Hindi lip synchronization.
- Prevent facial warping, lip jitter, changing teeth, wandering eyes, identity drift or exaggerated movement.
- Do not make the microphone cover the mouth.
- The source image is portrait-oriented. Extend the existing studio naturally on the left and right to produce a 16:9 frame. Do not stretch the image or crop the man's head, hands or body.
- Use a locked, eye-level camera without zooming or camera movement.
- Do not add captions, titles, logos or watermarks.
- Output: 1920x1080, 16:9, 25 or 30 fps, high-quality MP4.

LONG-VIDEO INSTRUCTIONS

If the platform cannot generate the whole poem in one operation:
- Divide it only at the numbered stanza boundaries below.
- Generate each stanza as a separate segment.
- Keep the same face, clothing, posture, framing, background, lighting and microphone position in every segment.
- Maintain identical voice tone, loudness and recording quality.
- Provide segments that can be joined seamlessly.
- Do not repeat or omit lines.

POEM

""" + TALKING_HEAD_POEM


def _normalize_hindi_for_comparison(text: str) -> str:
    import unicodedata
    value = unicodedata.normalize("NFC", str(text or ""))
    value = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.,!?;:'\"()\[\]{}|/\\—–-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _poem_word_count(text: str) -> int:
    return len(_normalize_hindi_for_comparison(text).split())


def _wav_quality_report(wav_path: str | Path) -> dict:
    import math
    import struct
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Voice WAV not found: {path}")
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        duration = frames / max(1, rate)
        raw = wf.readframes(frames)
    result = {
        "path": str(path.resolve()), "channels": channels, "sample_rate": rate,
        "sample_width": width, "frames": frames, "duration_seconds": duration,
        "duration": _clock(duration), "rms_db": None, "peak_db": None,
        "clipping_ratio": 0.0, "clean_enough": True, "quality_notes": [],
    }
    if width != 2 or not raw:
        result["quality_notes"].append("WAV is not 16-bit PCM; local quality metrics are limited.")
        return result
    samples = __import__('struct').unpack("<" + "h" * (len(raw) // 2), raw)
    if not samples:
        return result
    peak = max(abs(x) for x in samples)
    mean_square = sum(float(x) * float(x) for x in samples) / len(samples)
    rms = math.sqrt(mean_square)
    full_scale = 32768.0
    result["rms_db"] = 20.0 * math.log10(max(rms / full_scale, 1e-9))
    result["peak_db"] = 20.0 * math.log10(max(peak / full_scale, 1e-9))
    result["clipping_ratio"] = sum(1 for x in samples if abs(x) >= 32760) / len(samples)
    if result["clipping_ratio"] > 0.001:
        result["clean_enough"] = False
        result["quality_notes"].append("Noticeable digital clipping was detected.")
    if result["rms_db"] < -40:
        result["clean_enough"] = False
        result["quality_notes"].append("Recording level is very low.")
    if result["rms_db"] > -3:
        result["quality_notes"].append("Recording is close to full scale; check for aggressive loudness processing.")
    if duration < 2:
        result["clean_enough"] = False
        result["quality_notes"].append("Recording is too short for reliable voice/poem analysis.")
    return result


def _gemini_audio_transcript(wav_path: str | Path, api_key: Optional[str] = None) -> str:
    client = _client(api_key)
    uploaded = client.files.upload(file=str(Path(wav_path)))
    prompt = (
        "Transcribe this Hindi WAV exactly as spoken. Return ONLY the spoken words, "
        "with no commentary, no summary, no corrections and no invented words. "
        "Preserve repeated words if they are actually spoken."
    )
    try:
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type or "audio/wav"},
            ],
        )
        text = getattr(interaction, "output_text", None)
        if text:
            return str(text).strip()
    except Exception as interaction_error:
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    prompt,
                    types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type or "audio/wav"),
                ],
            )
            text = getattr(response, "text", None)
            if text:
                return str(text).strip()
        except Exception as fallback_error:
            raise RuntimeError(
                f"Gemini audio transcription failed: {interaction_error}; fallback: {fallback_error}"
            ) from fallback_error
    raise RuntimeError("Gemini returned no transcript for the supplied WAV.")


def _compare_transcript_to_poem(transcript: str, poem: str = TALKING_HEAD_POEM) -> dict:
    from difflib import SequenceMatcher
    expected = re.sub(r"(?m)^खंड\s+\d+\s*$", "", poem)
    expected_n = _normalize_hindi_for_comparison(expected)
    actual_n = _normalize_hindi_for_comparison(transcript)
    ratio = SequenceMatcher(None, actual_n, expected_n).ratio() if expected_n else 0.0
    expected_words = expected_n.split()
    actual_words = actual_n.split()
    word_ratio = SequenceMatcher(None, actual_words, expected_words).ratio() if expected_words else 0.0
    complete = word_ratio >= 0.97 and ratio >= 0.96
    matcher = SequenceMatcher(None, actual_words, expected_words)
    missing, extra = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace") and len(missing) < 12:
            missing.extend(expected_words[j1:j2])
        if tag in ("insert", "replace") and len(extra) < 12:
            extra.extend(actual_words[i1:i2])
    return {
        "complete": complete, "character_similarity": ratio, "word_similarity": word_ratio,
        "expected_words": len(expected_words), "spoken_words": len(actual_words),
        "missing_or_different_words": " ".join(missing[:12]),
        "extra_or_different_words": " ".join(extra[:12]),
    }


def _estimate_poem_duration_seconds(wav_info: dict, comparison: dict) -> float:
    if comparison.get("complete") and wav_info.get("duration_seconds"):
        return float(wav_info["duration_seconds"])
    words = max(1, int(comparison.get("expected_words") or _poem_word_count(TALKING_HEAD_POEM)))
    return (words / 130.0) * 60.0


def preflight_talking_head_video(
    portrait_path: str | Path,
    wav_path: str | Path,
    *,
    api_key: Optional[str] = None,
    poem: str = TALKING_HEAD_POEM,
) -> dict:
    """Run the requested five checks. No video generation occurs here."""
    portrait = Path(portrait_path)
    wav = Path(wav_path)
    if not portrait.is_file():
        raise FileNotFoundError(f"Portrait image not found: {portrait}")
    if not wav.is_file():
        raise FileNotFoundError(f"Voice WAV not found: {wav}")
    wav_info = _wav_quality_report(wav)
    transcript = _gemini_audio_transcript(wav, api_key=api_key)
    comparison = _compare_transcript_to_poem(transcript, poem)
    duration_seconds = _estimate_poem_duration_seconds(wav_info, comparison)
    pronunciation_check = "No obvious spelling problem detected by exact-text comparison."
    if comparison.get("missing_or_different_words") or comparison.get("extra_or_different_words"):
        pronunciation_check = (
            "The WAV differs from the supplied poem. Review possible differences: "
            f"expected/missing: {comparison.get('missing_or_different_words') or 'none'}; "
            f"spoken/extra: {comparison.get('extra_or_different_words') or 'none'}."
        )
    return {
        "portrait": str(portrait.resolve()), "wav": str(wav.resolve()),
        "wav_contains_complete_poem": bool(comparison["complete"]),
        "wav_role": "complete poem recording" if comparison["complete"] else "voice reference/sample",
        "wav_quality": wav_info, "transcript": transcript, "comparison": comparison,
        "pronunciation_check": pronunciation_check,
        "expected_duration_seconds": duration_seconds, "expected_duration": _clock(duration_seconds),
        "can_preserve_supplied_voice_directly": bool(comparison["complete"]),
        "complete_video_directly_with_gemini": False,
        "generation_blocker": (
            "Gemini TTS currently uses prebuilt voices and does not reproduce an arbitrary supplied WAV speaker identity. "
            "Veo 3.1 generates native audio and does not accept the supplied WAV as a lip-sync audio track. "
            "For exact supplied-voice preservation, use the original WAV as the master audio with a compatible "
            "lip-sync engine such as SadTalker; Gemini/Veo can provide visual segments."
        ),
        "prompt": TALKING_HEAD_PROMPT,
    }


def format_talking_head_preflight(report: dict) -> str:
    q = report.get("wav_quality") or {}
    rms = q.get("rms_db")
    peak = q.get("peak_db")
    return "\n".join([
        "GEMINI TALKING-HEAD PRE-GENERATION CHECK", "=" * 48, "",
        "1. WAV CONTENT",
        f"   Result: {report.get('wav_role', 'unknown')}.",
        f"   Complete poem detected: {'YES' if report.get('wav_contains_complete_poem') else 'NO'}.",
        f"   WAV duration: {q.get('duration', 'unknown')}.", "",
        "2. SPEECH QUALITY",
        f"   Clean enough for analysis/reproduction: {'YES' if q.get('clean_enough') else 'NO'}.",
        f"   Format: {q.get('channels', '?')} channel(s), {q.get('sample_rate', '?')} Hz, "
        f"{q.get('sample_width', '?') * 8 if q.get('sample_width') else '?'}-bit.",
        f"   RMS: {rms:.1f} dBFS" if isinstance(rms, (int, float)) else "   RMS: unavailable",
        f"   Peak: {peak:.1f} dBFS" if isinstance(peak, (int, float)) else "   Peak: unavailable",
        f"   Clipping ratio: {q.get('clipping_ratio', 0.0) * 100:.3f}%",
        f"   Notes: {' '.join(q.get('quality_notes') or []) or 'none'}", "",
        "3. PRONUNCIATION / TEXT CHECK",
        f"   {report.get('pronunciation_check', '')}",
        f"   Word similarity: {float(report.get('comparison', {}).get('word_similarity', 0)) * 100:.1f}%.", "",
        "4. EXPECTED VIDEO DURATION",
        f"   Approximately {report.get('expected_duration', 'unknown')}.",
        "   Complete recording: measured WAV duration is used.",
        "   Voice sample: duration estimated from the poem at ~130 words/minute.", "",
        "5. CAN GEMINI PRESERVE THE SUPPLIED VOICE AND GENERATE THE COMPLETE VIDEO DIRECTLY?",
        f"   Exact supplied voice preserved directly: {'YES' if report.get('can_preserve_supplied_voice_directly') else 'NO'}.",
        "   Complete talking-head video directly with Gemini/Veo: NO.",
        f"   Reason: {report.get('generation_blocker', '')}", "",
        "GENERATION GATE: No video generation should start until this report is reviewed.",
    ])


def split_poem_into_numbered_stanzas(poem: str = TALKING_HEAD_POEM) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^खंड\s+(\d+)\s*$", poem.strip()))
    if not matches:
        return [("1", poem.strip())]
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(poem)
        result.append((match.group(1), poem[match.end():end].strip()))
    return result


def generate_talking_head_veo_segments(
    portrait_path: str | Path,
    output_directory: str | Path,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_VEO_MODEL,
    resolution: str = "1080p",
    aspect_ratio: str = "16:9",
    poem: str = TALKING_HEAD_POEM,
    preflight_report: Optional[dict] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """Generate visual Veo segments after preflight.

    Veo cannot be given an arbitrary supplied WAV as its lip-sync audio track.
    The supplied WAV must therefore remain the master audio during final assembly.
    """
    if preflight_report is None:
        raise RuntimeError(
            "Generation is blocked until preflight_talking_head_video() has completed. "
            "Pass its returned report as preflight_report after reviewing the five checks."
        )
    required_keys = {
        "wav_contains_complete_poem", "wav_quality", "comparison",
        "expected_duration_seconds", "generation_blocker",
    }
    missing = sorted(required_keys - set(preflight_report))
    if missing:
        raise RuntimeError(
            "Invalid talking-head preflight report; missing: " + ", ".join(missing)
        )

    portrait = Path(portrait_path)
    if not portrait.is_file():
        raise FileNotFoundError(f"Portrait image not found: {portrait}")
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    for number, stanza in split_poem_into_numbered_stanzas(poem):
        prompt = (
            "Photorealistic locked-off talking-head shot. Use the supplied portrait as the ONLY visual "
            "identity reference. Preserve the man's exact face, age, complexion, white hair, clothing, "
            "podcast studio, microphone and desk. Seated, eye-level camera, looking generally toward camera. "
            "Only subtle natural blinking, breathing, eyebrow movement and restrained head movement. "
            "No camera movement, no zoom, no captions, no logos, no watermark, no identity drift, no facial "
            "warping, no lip jitter, no exaggerated expression. Extend the portrait studio naturally to 16:9 "
            "without stretching or cropping the man. Dignified patriotic delivery. Hindi lip movement should "
            "follow the stanza as closely as possible. Do not display the poem on screen.\n\n"
            f"Stanza {number}:\n{stanza}"
        )
        destination = out / f"stanza_{int(number):02d}_veo.mp4"
        if destination.exists() and destination.stat().st_size > 1000:
            generated.append(destination)
            continue
        if progress:
            progress(f"Generating Veo visual segment for stanza {number}...")
        generate_veo_video(
            prompt, destination, model=model, resolution=resolution,
            aspect_ratio=aspect_ratio, api_key=api_key, image_path=portrait,
            progress=progress,
        )
        generated.append(destination)
    return generated
