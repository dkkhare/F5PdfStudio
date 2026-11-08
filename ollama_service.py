"""Ollama support for F5PdfStudio.

This module is deliberately defensive because small local models sometimes:
* time out on long OCR documents;
* wrap answers in Markdown;
* add explanations instead of returning corrected text;
* stop at their output-token limit; or
* return malformed JSON during chapter detection.

Public functions used by app.py:
    get_installed_models
    check_ollama_status / check_ollama_connection
    call_ollama
    correct_ocr_chunk
    correct_complete_text
    save_as_docx
    detect_chapter_titles
    consume_correction_warnings

Only the third-party ``requests`` package is required, EXCEPT for
``save_as_docx``, which additionally needs ``python-docx`` (``pip install
python-docx``). If it isn't installed, ``save_as_docx`` raises a clear
RuntimeError rather than failing to import the whole module.
"""

from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests

try:
    from docx import Document
    from docx.shared import Pt
except ImportError:  # pragma: no cover - optional dependency
    Document = None
    Pt = None


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_TIMEOUT = 300
DEFAULT_CORRECTION_CHUNK_SIZE = 900
DEFAULT_CHAPTER_CHUNK_SIZE = 2800

# app.py may display these after correction.  Keep the same list object alive.
LAST_CORRECTION_WARNINGS: list[str] = []

# Path of the most recently saved .docx from correct_complete_text(save_docx=True),
# or None if no document has been saved yet in this process.
LAST_SAVED_DOCX_PATH: Optional[str] = None


@dataclass(frozen=True)
class TextChunk:
    number: int
    start: int
    text: str



# --- Compatibility imports retained from hindi_correct.py ---
import argparse
import hashlib
import os
import sys
import unicodedata


# --- Hindi corrector compatibility configuration ---
OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "llama3.1:8b"

BATCH_MAX_CHARS_BY_TYPE = {"poetry": 400, "drama": 400, "prose": 900}

NUM_CTX_BY_TYPE = {"poetry": 2560, "drama": 2560, "prose": 4096}

TEXT_TYPES = ["poetry", "drama", "prose"]

COMMON_RULES = """आप एक अनुभवी हिंदी भाषा संपादक हैं। यह पाठ एक पुराने दस्तावेज़ से आया है, जिसमें
टाइपिंग/कन्वर्ज़न के दौरान जगह-जगह वर्तनी (spelling), मात्रा (matras), और कभी-कभी कुछ भ्रष्ट
(corrupted) वर्ण भी रह गए हैं।

आपका मुख्य काम:
1. केवल वर्तनी और मात्राओं की स्पष्ट गलतियाँ ठीक करना।
2. शब्दों का चयन (word choice), वाक्य-रचना, अर्थ, और मूल शैली बिल्कुल न बदलें — आप संपादक हैं,
   लेखक नहीं।
3. केवल वही बदलें जो निश्चित रूप से गलत है; अनिश्चित होने पर मूल पाठ को वैसा ही रहने दें।
4. इनपुट में जितनी पंक्तियाँ (lines) हैं, आउटपुट में भी बिल्कुल उतनी ही पंक्तियाँ होनी चाहिए,
   उसी क्रम में, एक-एक करके। कोई पंक्ति न जोड़ें, न हटाएं, न आपस में मिलाएं।
5. आपका उत्तर केवल सुधारी हुई पंक्तियाँ होनी चाहिए — कोई विश्लेषण, व्याख्या, शीर्षक,
   मार्कडाउन (#, **, ---), या अतिरिक्त वाक्य नहीं। सिर्फ सुधारा हुआ पाठ, पंक्ति दर पंक्ति।
"""

TYPE_PROMPT_ADDITIONS = {
    "poetry": """
यह पाठ कविता (poetry) है।
6. यह बहुत ज़रूरी है: किसी पंक्ति के बीच में अल्पविराम (,) आना सामान्य बात है (कविता में यह
   पंक्ति के भीतर एक ठहराव/यति दिखाता है)। ऐसे अल्पविराम को देखकर पंक्ति को दो हिस्सों में मत
   तोड़ें — पूरी पंक्ति, अल्पविराम सहित, एक ही पंक्ति बनी रहनी चाहिए। नई पंक्ति केवल वहीं बनाएं
   जहां इनपुट में पहले से ही एक नई लाइन (newline) थी।
7. तुकांत (rhyme), छंद (meter), और लय बिल्कुल न बदलें।

उदाहरण इनपुट (2 पंक्तियाँ):
हँसता स्वर्ण विहान , फैलने को उजियाला
जनमन में जो व्याप्त , राम ने ली अंगड़ाई

उदाहरण आउटपुट (बिल्कुल 2 पंक्तियाँ — ध्यान दें अल्पविराम के बावजूद कोई पंक्ति नहीं टूटी):
हँसता स्वर्ण विहान, फैलने को उजियाला
जनमन में जो व्याप्त, राम ने ली अंगड़ाई
""",
    "drama": """
यह एक नाटक (drama) की स्क्रिप्ट है।
6. पात्रों के नाम (character names) और मंच निर्देश (stage directions) का प्रारूप न बदलें।
7. एक विशेष पुराना दोष पहचानना: कभी-कभी स्टेज-डायरेक्शन (मंच निर्देश, जैसे "(क्रोधित होकर)")
   गलती से बिना कोष्ठक के "य" से शुरू और "द्ध" पर खत्म होते मिलते हैं। जैसे:
   गलत: यक्रोधित से होकरद्ध रामसिंहःण्
   सही: (क्रोधित से होकर) रामसिंहः
   ऐसे मामलों में "य" को "(" से और "द्ध" को ")" से बदलें — लेकिन ध्यान रहे "य" अक्सर सामान्य
   शब्दों (यह, युद्ध, यही, याद) का हिस्सा भी होता है, इसलिए केवल तभी बदलें जब यह वाकई किसी
   कोष्ठक-जैसे मंच-निर्देश की शुरुआत लगे, सामान्य शब्दों को मत छुएं।

उदाहरण इनपुट (3 पंक्तियाँ):
राम: तुम कहाँ जा रहे हे?
श्याम: मे बाज़ार जा रहा हूँ।
यमुस्कुराते हुयेद्ध (श्याम बाहर जाता हे)

उदाहरण आउटपुट (बिल्कुल 3 पंक्तियाँ):
राम: तुम कहाँ जा रहे हो?
श्याम: मैं बाज़ार जा रहा हूँ।
(मुस्कुराते हुए) (श्याम बाहर जाता है)
""",
    "prose": """
यह गद्य (prose — उपन्यास/कहानी/निबंध) है, जिसमें वर्णन और संवाद दोनों हो सकते हैं।
6. वाक्यों को छोटा-बड़ा करने, दोबारा लिखने (rewrite), या पैराग्राफ को पुनर्गठित करने से बचें —
   केवल स्पष्ट गलतियाँ ठीक करें।
7. यदि कोई वाक्य टाइपिंग के कारण एक पंक्ति से अगली पंक्ति में टूटा हुआ (hard-wrapped) लगे, तब भी
   हर पंक्ति को अलग पंक्ति के रूप में ही सुधारें — पंक्तियों को आपस में न जोड़ें, चाहे वाक्य
   अधूरा ही क्यों न दिखे।

उदाहरण इनपुट (2 पंक्तियाँ):
वह धीरे धीरे चलता हुआ घर की और बढा, उसके मन मे
अनेक विचार आ रहे थे।

उदाहरण आउटपुट (बिल्कुल 2 पंक्तियाँ):
वह धीरे धीरे चलता हुआ घर की ओर बढ़ा, उसके मन में
अनेक विचार आ रहे थे।
""",
}

USER_TEMPLATE = "इन पंक्तियों में केवल वर्तनी और मात्रा ठीक करें। इसमें ठीक {n} पंक्तियाँ हैं — अपने उत्तर में भी ठीक {n} पंक्तियाँ दें, एक भी कम या ज़्यादा नहीं। किसी भी पंक्ति को न आपस में मिलाएं, न बीच के अल्पविराम (,) पर तोड़ें — भले ही पंक्ति के बीच में अल्पविराम/ठहराव हो, वह एक ही पंक्ति बनी रहे:\n\n{text}"

_INVALID_MARKERS = ["###", "**", "---"]

CHARACTER_CUE_RE = re.compile(r'^([^\n:ः]{1,25}[:ः])\s*')

STAGE_DIRECTION_RE = re.compile(r'\([^)]*\)')

_TRAILING_END_PUNCT_RE = re.compile(r'[,।॥.]+\s*$')

_RHYME_STRIP_RE = re.compile(r'[ंँ]')

MANUAL_OVERRIDES_FILENAME = "manual_overrides.json"

def _base_url(value: Optional[str] = None) -> str:
    return (value or OLLAMA_BASE_URL).strip().rstrip("/")


def _notify(callback: Optional[Callable[..., Any]], message: str, *args: Any) -> None:
    """Call different styles of Gradio/application progress callbacks safely."""
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


def get_installed_models(base_url: str = OLLAMA_BASE_URL, timeout: int = 15) -> list[str]:
    """Return locally installed Ollama model names."""
    try:
        response = requests.get(f"{_base_url(base_url)}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at http://127.0.0.1:11434. "
            "Start Ollama and confirm it with: ollama list"
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Ollama returned an invalid response from /api/tags.") from exc

    names: list[str] = []
    for item in payload.get("models", []):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
    return names


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Backward-compatible alias."""
    return get_installed_models(base_url=base_url)


def check_ollama_status(
    model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[bool, str]:
    """Return ``(available, user_friendly_message)``."""
    try:
        models = get_installed_models(base_url=base_url)
    except RuntimeError as exc:
        return False, str(exc)

    if model and model not in models:
        return (
            False,
            f"Ollama is running, but '{model}' is not installed. "
            f"Run: ollama pull {model}",
        )
    if not models:
        return False, "Ollama is running, but no model is installed."
    return True, "Ollama is ready. Installed models: " + ", ".join(models)


def check_ollama_connection(
    model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[bool, str]:
    """Backward-compatible alias."""
    return check_ollama_status(model=model, base_url=base_url)


def check_ollama(
    model: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[bool, str]:
    """Compatibility alias used by earlier F5PdfStudio ``app.py`` files."""
    return check_ollama_status(model=model, base_url=base_url)


def _contextual_error(exc: BaseException) -> RuntimeError:
    if isinstance(exc, requests.Timeout):
        return RuntimeError(
            "Ollama timed out. The input was too large for the selected model. "
            "Use a smaller chunk size (1200-1600 for correction) and close other GPU programs."
        )
    if isinstance(exc, requests.ConnectionError):
        return RuntimeError(
            "Cannot connect to Ollama. Start the Ollama application or run 'ollama serve'."
        )
    return RuntimeError(f"Ollama request failed: {exc}")


def _service_call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    system: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0.0,
    num_predict: int = 2048,
    num_ctx: int = 4096,
    options: Optional[dict[str, Any]] = None,
    format: Optional[str | dict[str, Any]] = None,
    retries: int = 1,
    keep_alive: str = "10m",
    think: bool = False,
    progress_callback: Optional[Callable[..., Any]] = None,
    **_: Any,
) -> str:
    """Call Ollama and return response text, streaming progress when requested.

    ``options`` and individual option arguments are both accepted so older and
    newer app.py versions can use this function.
    """
    prompt = str(prompt or "").strip()
    model = str(model or DEFAULT_MODEL).strip()
    if not prompt:
        raise ValueError("Ollama prompt is empty.")
    if not model:
        raise ValueError("Ollama model name is empty.")

    merged_options: dict[str, Any] = {
        "temperature": float(temperature),
        "num_predict": int(num_predict),
        "num_ctx": int(num_ctx),
        "top_p": 0.85,
        "repeat_penalty": 1.05,
    }
    if options:
        merged_options.update(options)

    effective_prompt = prompt
    if not think and not prompt.lstrip().lower().startswith("/no_think"):
        effective_prompt = "/no_think\n" + prompt

    payload: dict[str, Any] = {
        "model": model,
        "prompt": effective_prompt,
        "stream": True,
        "think": bool(think),
        "options": merged_options,
        "keep_alive": keep_alive,
    }
    if system:
        payload["system"] = system
    if format is not None:
        payload["format"] = format

    last_error: Optional[BaseException] = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            response = requests.post(
                f"{_base_url(base_url)}/api/generate",
                json=payload,
                timeout=(15, max(30, int(timeout))),
                stream=True,
            )
            response.raise_for_status()
            output_parts: list[str] = []
            response_pieces = 0
            final_data: dict[str, Any] = {}

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except (TypeError, json.JSONDecodeError):
                    continue
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                piece = str(data.get("response") or "")
                if piece:
                    output_parts.append(piece)
                    response_pieces += 1
                    if response_pieces == 1 or response_pieces % 8 == 0:
                        _notify(
                            progress_callback,
                            f"Ollama is generating corrected text… ({response_pieces} tokens)",
                        )
                if data.get("done") is True:
                    final_data = data

            output = "".join(output_parts).strip()

            if not output:
                raise RuntimeError("Ollama returned an empty response.")

            if final_data and final_data.get("done") is False:
                raise RuntimeError("Ollama stopped before completing its response.")
            if final_data.get("done_reason") == "length":
                raise RuntimeError("Ollama reached its output limit before returning the complete corrected text.")
            _notify(progress_callback, "Ollama response received; validating text…")
            return output
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < max(0, int(retries)):
                time.sleep(1.5 * (attempt + 1))

    assert last_error is not None
    if isinstance(last_error, RuntimeError):
        raise last_error
    raise _contextual_error(last_error) from last_error


def _normalize_input(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _split_text(text: str, maximum: int) -> list[TextChunk]:
    """Split on paragraph/sentence boundaries while preserving order."""
    maximum = max(500, int(maximum))
    text = _normalize_input(text)
    if not text:
        return []

    pieces = re.split(r"(\n\s*\n+)", text)
    units: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= maximum:
            units.append(piece)
            continue
        # Long OCR paragraph: sentence-aware split.
        sentences = re.split(r"(?<=[।.!?])\s+|\n+", piece)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= maximum:
                units.append(sentence)
            else:
                # Last-resort fixed split prevents context overflows.
                for pos in range(0, len(sentence), maximum):
                    units.append(sentence[pos : pos + maximum])

    chunks: list[TextChunk] = []
    buffer = ""
    start = 0
    cursor = 0
    for unit in units:
        separator = "" if not buffer else ("" if unit.startswith("\n") else "\n\n")
        candidate = buffer + separator + unit
        if buffer and len(candidate) > maximum:
            chunks.append(TextChunk(len(chunks) + 1, start, buffer.strip()))
            cursor += len(buffer)
            start = cursor
            buffer = unit
        else:
            buffer = candidate
    if buffer.strip():
        chunks.append(TextChunk(len(chunks) + 1, start, buffer.strip()))
    return chunks


def _split_drama_text(text: str, maximum: int) -> list[TextChunk]:
    """Split drama only between logical paragraphs.

    A dramatic dialogue is one logical paragraph even when OCR/page layout
    originally wrapped it across several physical lines.  Therefore we never
    split a character's dialogue merely because a line is long.
    """
    maximum = max(500, int(maximum))
    source = _normalize_input(text)
    if not source:
        return []
    paragraphs = _normalize_drama_paragraphs(source)
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    length = 0
    start = 0
    cursor = 0
    for para in paragraphs:
        para_len = len(para) + (2 if buffer else 0)
        # Never split a logical drama paragraph.  A single very long paragraph
        # is allowed to exceed the nominal chunk size so the speaker's dialogue
        # remains intact for Ollama.
        if buffer and length + para_len > maximum:
            chunk_text = "\n\n".join(buffer).strip()
            chunks.append(TextChunk(len(chunks) + 1, start, chunk_text))
            cursor += len(chunk_text) + 2
            start = cursor
            buffer = [para]
            length = len(para)
        else:
            buffer.append(para)
            length += para_len
    if buffer:
        chunks.append(TextChunk(len(chunks) + 1, start, "\n\n".join(buffer).strip()))
    return chunks


def _strip_model_wrappers(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:text|markdown|md|json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    prefixes = (
        "corrected text:", "corrected version:", "output:", "result:",
        "सुधारा हुआ पाठ:", "शुद्ध पाठ:", "संशोधित पाठ:",
    )
    lowered = text.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix.casefold()):
            text = text[len(prefix):].lstrip()
            break
    return text.strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w\u0900-\u097f]+", text, flags=re.UNICODE))


def _boundary_token(text: str, first: bool) -> str:
    """Return a conservative first/last lexical token for truncation checks."""
    tokens = re.findall(r"[A-Za-z0-9_\u0900-\u097F]+", str(text or ""), flags=re.UNICODE)
    if not tokens:
        return ""
    return tokens[0] if first else tokens[-1]


def validate_corrected_output(source: str, corrected: str) -> str:
    """Validate that Ollama returned the complete passage, not a shortened rewrite."""
    source = _normalize_input(source)
    corrected = _strip_model_wrappers(corrected)
    if not corrected:
        raise RuntimeError("Ollama returned empty corrected text.")

    lines = corrected.splitlines()
    if lines and re.match(
        r"^(?:here(?:'s| is)|sure[,!:]*|नीचे|यह रहा|मैंने)\b",
        lines[0].strip(), re.I,
    ) and len(lines) > 1:
        corrected = "\n".join(lines[1:]).strip()

    corrected = corrected.replace("**", "").replace("__", "")
    if not corrected:
        raise RuntimeError("Ollama returned empty corrected text after cleanup.")

    source_words = max(1, _word_count(source))
    output_words = _word_count(corrected)
    ratio = output_words / source_words
    if ratio < 0.55:
        raise RuntimeError(
            f"Ollama response appears incomplete ({output_words}/{source_words} words)."
        )
    if ratio > 1.55:
        raise RuntimeError("Ollama added too much new text instead of only correcting OCR.")

    # Do not require the first/last lexical token to be identical. OCR correction
    # may legitimately change the first or last word (for example ``पहिलाण्`` ->
    # ``पहिला``). Compare short content windows instead, so omitted/truncated
    # content is still rejected while legitimate spelling corrections survive.
    def _edge_similarity(a: str, b: str, first: bool) -> float:
        a = re.sub(r"\s+", " ", a).strip()
        b = re.sub(r"\s+", " ", b).strip()
        if first:
            a, b = a[:120], b[:120]
        else:
            a, b = a[-120:], b[-120:]
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()

    source_first = _boundary_token(source, True)
    output_first = _boundary_token(corrected, True)
    source_last = _boundary_token(source, False)
    output_last = _boundary_token(corrected, False)

    if source_first and output_first and source_first.casefold() != output_first.casefold():
        if _edge_similarity(source, corrected, True) < 0.48:
            raise RuntimeError(
                "Ollama changed or omitted the beginning of the passage; the original chunk will be retained."
            )
    if source_last and output_last and source_last.casefold() != output_last.casefold():
        if _edge_similarity(source, corrected, False) < 0.48:
            raise RuntimeError(
                "Ollama response appears truncated or changed the end of the passage; the original chunk will be retained."
            )

    return _normalize_input(corrected)


def validate_drama_output(source: str, corrected: str) -> str:
    """Validate drama while allowing logical paragraph reflow.

    Physical OCR line count is intentionally NOT validated for drama.  The
    desired output is one paragraph per speaker/scene direction, not one
    paragraph per scanned line.
    """
    raw = str(corrected or "").strip()
    report_patterns = (
        r"(?im)^\s*अनिवार्य\s+नियम",
        r"(?im)^\s*दिए\s+गए.+त्रुटि",
        r"(?im)^\s*निम्नलिखित.+(?:त्रुटि|सुधार)",
        r"(?im)^\s*\d+[.)]\s*(?:वर्तनी|व्याकरण|वाक्य[- ]?विन्यास|विराम[- ]?चिह्न)",
        r"(?im)^\s*[-*]\s*[\"“].+?[\"”]\s*(?:→|->)",
        r"(?im)^\s*(?:सुधारा हुआ पाठ|शुद्ध किया हुआ पाठ)\s*[:：]",
    )
    if "```" in raw or "**" in raw or "→" in raw or re.search(r"(?m)^\s*---+\s*$", raw):
        raise RuntimeError("Ollama returned Markdown or a correction report instead of the complete drama text.")
    for pattern in report_patterns:
        if re.search(pattern, raw):
            raise RuntimeError("Ollama returned an explanation/correction list instead of the complete drama text.")

    validated = validate_corrected_output(source, raw)
    source_paragraphs = _normalize_drama_paragraphs(source)
    output_paragraphs = _normalize_drama_paragraphs(validated)
    if source_paragraphs:
        # Normally each logical paragraph remains one paragraph.  Allow a small
        # tolerance for a stage direction becoming standalone, but reject major
        # omissions/additions.
        if not output_paragraphs:
            raise RuntimeError("Ollama returned no drama paragraphs.")
        source_count = len(source_paragraphs)
        output_count = len(output_paragraphs)
        if output_count > source_count + 2 or output_count < max(1, source_count - 2):
            raise RuntimeError(
                f"Ollama changed drama paragraph structure ({output_count} vs {source_count}); the original chunk will be retained."
            )

    label_pattern = re.compile(r"(?m)^\s*([\u0900-\u097FA-Za-z][\u0900-\u097FA-Za-z .]{0,35})\s*[:：]")
    source_labels = {
        re.sub(r"[^\w\u0900-\u097f]", "", label).casefold()
        for label in label_pattern.findall(source)
    }
    output_labels = {
        re.sub(r"[^\w\u0900-\u097f]", "", label).casefold()
        for label in label_pattern.findall(validated)
    }
    missing_labels = {label for label in source_labels if label and label not in output_labels}
    if source_labels and len(missing_labels) > max(1, len(source_labels) // 3):
        raise RuntimeError("Ollama removed or rewrote character-name labels; the original chunk will be retained.")
    return validated


def correct_ocr_chunk(
    text: str,
    model: str = DEFAULT_MODEL,
    language: str = "Hindi/English",
    book_type: str = "Others",
    *,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 2,
    progress_callback: Optional[Callable[..., Any]] = None,
    **_: Any,
) -> str:
    """Correct one small OCR chunk while preserving its meaning and structure."""
    source = _normalize_input(text)
    if not source:
        return ""

    normalized_book_type = str(book_type or "Others").strip().title()
    if normalized_book_type not in {"Poetry", "Drama", "Novel", "Others"}:
        normalized_book_type = "Others"

    type_rules = {
        "Poetry": (
            "Preserve verse line breaks, stanza breaks, rhyme, metre, refrains, "
            "intentional repetition and poetic vocabulary. Do not convert poetry "
            "into prose or modernize unusual but valid poetic words. Correct only "
            "spelling, matras, broken Devanagari characters, spacing and unmistakable "
            "OCR punctuation errors. Never correct grammar, word order, agreement, "
            "idiom, metre, rhyme or the poet's deliberate language. IMPORTANT: a comma "
            "in the middle of a verse line is normal (it marks a poetic pause/yati) — "
            "never split that line into two lines because of it; the full line, comma "
            "included, must stay on one line exactly as in the source."
        ),
        "Drama": (
            "लेखक की शैली, भाव, अर्थ, वाक्य-विन्यास, पात्रों की बोलचाल, क्षेत्रीय "
            "बोली, ग्रामीण भाषा और जानबूझकर प्रयुक्त अशुद्ध भाषा सुरक्षित रखें। "
            "पात्र-नाम, संवाद-क्रम, अंक, दृश्य, मंच-निर्देश, गीत, कविता और मूल पंक्ति तथा "
            "अनुच्छेद सीमाएँ न बदलें। संवाद को वर्णन में न बदलें।"
        ),
        "Novel": (
            "Preserve paragraphs, dialogue, narration, names, viewpoint and the "
            "author's style. Do not shorten or rewrite sentences. If a sentence "
            "appears to be cut off mid-line because of how the source was line-wrapped "
            "(common in OCR/typed text), still correct and keep each line exactly as "
            "its own line — do not merge it with the next line, even if the sentence "
            "looks incomplete on its own."
        ),
        "Others": (
            "Preserve headings, lists, paragraphs, terminology and the original "
            "document structure. If a sentence appears to be cut off mid-line because "
            "of how the source was line-wrapped, still keep each line exactly as its "
            "own line — do not merge it with the next line."
        ),
    }

    if normalized_book_type == "Drama":
        system = (
            "आप हिन्दी नाटक के अनुभवी संपादक और OCR-proofreader हैं। "
            "आपका काम पुराने/स्कैन किये हुए नाटक को पढ़कर शुद्ध, स्वाभाविक और साहित्यिक रूप में लौटाना है। "
            "सिर्फ corrected text दें; कोई टिप्पणी, explanation, analysis, Markdown या सूची नहीं।"
        )
        prompt = f"""इस हिन्दी नाटक के पाठ को ध्यान से पढ़कर OCR, spelling, matra, grammar और punctuation की स्पष्ट गलतियाँ सुधारें।

सबसे महत्वपूर्ण नियम:
1. अर्थ और मूल साहित्यिक आशय सुरक्षित रखें। आप लेखक नहीं हैं; अनावश्यक rewriting न करें।
2. OCR की स्पष्ट गलत पहचान को संदर्भ से ठीक करें। उदाहरण: वच→बच, राण→राणा, मेवाण/मेवाण्→मेवाड़, कानो→कानों, आखों→आँखों, प्राणो→प्राणों, उन्हे→उन्हें, हुये→हुए, लिये→लिए, सवसे→सबसे, नही→नहीं, कहॉ→कहाँ आदि। लेकिन केवल संदर्भ से स्पष्ट होने पर ही सुधार करें।
3. लंबे garbage runs जैसे ण्ण्ण्ण्ण्ण्ण, ण्, या ऐसे OCR markers को हटाकर वाक्य को स्वाभाविक बनाएं।
4. `घ्` जैसे OCR question-mark corruption को संदर्भ के अनुसार `?` में बदलें। `ए`/`ऐ` जैसे OCR punctuation corruption को संदर्भ के अनुसार सही comma/full stop आदि में बदलें।
5. `ययुद्ध` जैसे स्पष्ट duplicated OCR अक्षर को `युद्ध` करें।
6. एक ही पात्र का लगातार पूरा संवाद एक ही logical paragraph में रखें। OCR/page/column की physical line breaks को paragraph break न मानें। नया dialogue paragraph केवल speaker बदलने पर शुरू करें।
7. Character name के बाद `:` या `ः` cue सुरक्षित रखें। Character name को दूसरे शब्दों में न बदलें।
8. यदि dialogue के बीच `(क्रोधित होकर)`, `(धीरे से)`, `(आश्चर्य से)` जैसी action/emotion direction हो तो उसे छोटे brackets में रखें; उसे अलग speaker न बनाएं।
9. OCR में `यक्रोधित से होकरद्ध` जैसे स्पष्ट bracket-corruption को `(क्रोधित होकर)` करें। `यदोनों का प्रस्थानद्ध` को `(दोनों का प्रस्थान)` जैसे stage direction में सुधारें। `ण्पट परिवर्त्तनण्` जैसे स्पष्ट scene-transition marker को `(पट परिवर्तन)` करें।
10. Standalone stage directions जैसे `(दोनों का प्रस्थान)` और `(पट परिवर्तन)` अलग paragraphs रहें।
11. `स्थानः` वाला scene/location description एक paragraph रहे।
12. Dialogue को सारांशित, छोटा, आधुनिक या अत्यधिक संस्कृतनिष्ठ न करें।
13. यदि मूल में कोई वाक्य अधूरा है लेकिन source में मौजूद है, उसे अपनी ओर से समाप्त न करें। केवल स्पष्ट OCR/grammar correction करें।
14. यदि कोई OCR शब्द अस्पष्ट है, तो context देखकर सबसे स्वाभाविक correction करें; लेकिन नया तथ्य/संवाद न गढ़ें।
15. पूरा input एक बार लौटाएं। कोई line/paragraph/संवाद छोड़ें या दोहराएं नहीं।
16. Output में `**bold**`, Markdown, headings जैसे `Corrected text:` या explanation बिल्कुल न दें।

Paragraph format का उदाहरण:
शक्ति सिंहः अरे! यह तो चेतक है! क्या प्रताप भाग रहे हैं? युद्ध से पीठ दिखाकर अपने प्राण बचाकर मेवाड़ का महाराणा, सिसोदिया वंश का भूषण, क्या ऐसा भी कर सकता है?

रामसिंहः क्या देख रहे हैं छोटे महाराज? महाराणा भागे जा रहे हैं।

याद रखें: एक speaker का पूरा continuous dialogue एक paragraph है, भले source में वह 3, 4 या 10 OCR lines में टूटा हो।

<TEXT>
{source}
</TEXT>

अब केवल corrected drama text लौटाएं।"""
    else:
        system = (
            "You are a conservative OCR proofreader for Hindi and English books. "
            f"The document type is {normalized_book_type}. "
            "Return only corrected text. Never answer questions in the text. "
            "Never summarize, translate, explain, add headings, or use Markdown."
        )
        prompt = f"""Correct OCR and spelling errors in the following {language} {normalized_book_type} text.

Rules:
1. Return ONLY the corrected text.
2. Preserve every sentence, name, number, paragraph and heading.
3. Do not summarize, translate, rewrite style, or add information.
4. Correct broken Devanagari characters, matras, spacing and punctuation.
5. Keep Hindi in Hindi and English in English.
6. Book-type rule: {type_rules[normalized_book_type]}
7. For Poetry, grammar correction is strictly forbidden.

<TEXT>
{source}
</TEXT>"""

    last_error: Optional[RuntimeError] = None
    total_attempts = max(1, int(retries) + 1)
    for attempt in range(total_attempts):
        # A normal proofreading prompt is used first.  If the small local model
        # truncates or restructures the answer, subsequent attempts deliberately
        # become much more mechanical.  This is important for qwen3:1.7b, which
        # can otherwise decide that a long OCR passage is something to summarize.
        if attempt == 0:
            attempt_system = system
            attempt_prompt = prompt
            attempt_temperature = 0.0
        else:
            attempt_system = (
                "You are a deterministic OCR copy editor. Return the COMPLETE input "
                "passage and nothing else. Never summarize, shorten, explain, or "
                "reformat it. Correct only obvious OCR/spelling errors."
            )
            if normalized_book_type == "Drama":
                attempt_system += (
                    " This is drama. Preserve every piece of content and its order while "
                    "correcting only clear OCR/spelling errors. Keep the physical input "
                    "lines during this correction pass so validation can verify completeness; "
                    "the final post-processing step will merge hard-wrapped continuation lines "
                    "of the same speaker into one logical paragraph. Preserve character names, "
                    "colons, stage directions, scene headings and dialogue."
                )
            attempt_prompt = f"""COPY-EDIT THIS PASSAGE.\n\nReturn the entire passage from the FIRST CHARACTER/LINE to the LAST CHARACTER/LINE.
The beginning and ending are part of the source and MUST NOT be omitted.\nDo not stop early. Do not summarize. Do not add commentary.\nCorrect only clear OCR/spelling/matra/punctuation errors.\nPreserve every original content item and its order.\nIf a word is uncertain, keep the original word.\n\n""" \
                + ("For this drama, keep the same number of non-empty physical input lines in this correction pass. "
                   "Do not omit, invent, or duplicate content. Hard-wrapped continuation lines will be "
                   "merged later by the program when they belong to the same speaker.\n\n"
                   if normalized_book_type == "Drama" else "") \
                + f"INPUT:\n{source}\n\nOUTPUT ONLY THE CORRECTED PASSAGE."
            # A slightly larger output budget is useful on retry, while the
            # source itself remains deliberately small enough for the 1.7B model.
            attempt_temperature = 0.0

        try:
            output = call_ollama(
                attempt_prompt,
                model=model,
                system=attempt_system,
                base_url=base_url,
                timeout=timeout,
                temperature=attempt_temperature,
                num_predict=max(1536, min(6144, int(len(source) * 4.5) + 768)),
                num_ctx=8192,
                retries=0,
                think=False,
                progress_callback=progress_callback,
            )
            validated = (
                validate_drama_output(source, output)
                if normalized_book_type == "Drama"
                else validate_corrected_output(source, output)
            )
            if normalized_book_type == "Poetry":
                source_lines = source.splitlines()
                corrected_lines = validated.splitlines()
                if len(source_lines) != len(corrected_lines):
                    raise RuntimeError(
                        "Ollama changed the number of poetry lines; the original "
                        "chunk will be retained to protect verse boundaries."
                    )
            return validated
        except RuntimeError as exc:
            last_error = exc
            # On the first validation failure, immediately retry with the mechanical
            # copy-edit prompt rather than repeating the same high-level instruction.
            if attempt + 1 < total_attempts:
                _notify(
                    progress_callback,
                    f"Retrying section after validation issue (attempt {attempt + 2}/{total_attempts})…",
                )

    assert last_error is not None
    raise last_error


def correct_complete_text(
    text: str,
    model: str = DEFAULT_MODEL,
    language: str = "Hindi/English",
    book_type: str = "Others",
    chunk_size: int = DEFAULT_CORRECTION_CHUNK_SIZE,
    progress_callback: Optional[Callable[..., Any]] = None,
    *,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    continue_on_error: bool = True,
    save_docx: bool = False,
    docx_output_root: Optional[str] = None,
    docx_project_name: Optional[str] = None,
    docx_part_number: Any = 1,
    docx_filename: Optional[str] = None,
    docx_nested: bool = True,
    control: Optional[Callable[[], None]] = None,
    resume_chunks: Optional[dict] = None,
    on_chunk: Optional[Callable[[int, int, str], None]] = None,
    **kwargs: Any,
) -> str:
    """Correct a complete document safely in small sequential chunks.

    ``control`` is called before every chunk (it may block while paused or raise to stop),
    ``resume_chunks`` maps chunk number -> already corrected text and ``on_chunk`` receives
    ``(chunk_number, total_chunks, corrected_text)`` after every finished chunk.

    Failed chunks retain their original text. Their descriptions are added to
    ``LAST_CORRECTION_WARNINGS`` so a 300-page book is not lost due to one bad
    local-model response.

    If ``save_docx=True`` (with ``docx_output_root`` and ``docx_project_name``
    also given), the corrected text is additionally saved as a formatted
    .docx via ``save_as_docx`` once correction finishes. The saved path is
    recorded in ``LAST_SAVED_DOCX_PATH`` (see ``get_last_saved_docx_path``);
    a failure here is added to ``LAST_CORRECTION_WARNINGS`` rather than
    raised, so a docx-saving problem never discards already-corrected text.
    """
    LAST_CORRECTION_WARNINGS.clear()
    source = _normalize_input(text)
    if not source:
        return ""

    # Drama OCR needs the deterministic legacy cleanup before chunking, otherwise
    # the small local model spends its capacity trying to decode ``ण्`` garbage.
    if str(book_type or "Others").strip().title() == "Drama":
        source = fix_legacy_corruption(source)
        # Convert physical OCR line wraps into logical drama paragraphs BEFORE
        # sending text to Ollama.  This is the key distinction between the old
        # output and the desired literary-script output.
        source = "\n\n".join(_normalize_drama_paragraphs(source))

    effective_chunk_size = min(max(int(chunk_size), 500), 1200)
    if str(book_type or "Others").strip().title() == "Drama":
        chunks = _split_drama_text(source, effective_chunk_size)
    else:
        chunks = _split_text(source, effective_chunk_size)
    corrected_chunks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if resume_chunks and index in resume_chunks:
            corrected_chunks.append(str(resume_chunks[index]).strip())
            _notify(
                progress_callback,
                f"Using saved correction for section {index}/{len(chunks)}",
                index,
                len(chunks),
            )
            continue
        if control is not None:
            control()  # pause / stop only between chunks
        _notify(
            progress_callback,
            f"Correcting text: section {index}/{len(chunks)}",
            index - 1,
            len(chunks),
        )
        try:
            corrected = correct_ocr_chunk(
                chunk.text,
                model=model,
                language=language,
                book_type=book_type,
                base_url=base_url,
                timeout=timeout,
                retries=2,
                progress_callback=progress_callback,
                **kwargs,
            )
        except Exception as exc:
            message = f"Section {index}/{len(chunks)} was kept unchanged: {exc}"
            if not continue_on_error:
                raise RuntimeError(
                    f"Ollama correction failed at section {index}/{len(chunks)}.\n{exc}"
                ) from exc
            LAST_CORRECTION_WARNINGS.append(message)
            corrected = chunk.text
        corrected_chunks.append(corrected.strip())
        if on_chunk is not None:
            on_chunk(index, len(chunks), corrected.strip())
        _notify(
            progress_callback,
            f"Completed correction section {index}/{len(chunks)}",
            index,
            len(chunks),
        )

    _notify(progress_callback, "Text correction completed.", len(chunks), len(chunks))
    result = "\n\n".join(part for part in corrected_chunks if part)
    if str(book_type or "Others").strip().title() == "Drama":
        result = _postprocess_drama_text(result)

    global LAST_SAVED_DOCX_PATH
    if save_docx:
        if not docx_output_root or not docx_project_name:
            LAST_CORRECTION_WARNINGS.append(
                "Could not save .docx: docx_output_root and docx_project_name are required."
            )
        else:
            try:
                LAST_SAVED_DOCX_PATH = save_as_docx(
                    result,
                    output_root=docx_output_root,
                    project_name=docx_project_name,
                    part_number=docx_part_number,
                    book_type=book_type,
                    filename=docx_filename,
                    nested=docx_nested,
                )
                _notify(progress_callback, f"Saved Word document: {LAST_SAVED_DOCX_PATH}")
            except Exception as exc:
                LAST_CORRECTION_WARNINGS.append(f"Could not save .docx: {exc}")

    return result


def _sanitize_path_component(value: Any, default: str) -> str:
    """Make a string safe to use as a Windows/macOS/Linux folder or file name."""
    text = str(value if value is not None else "").strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or default



_DRAMA_SPEAKER_RE = re.compile(r"^\s*([^\n:：ः]{1,40})\s*[:：ः]\s*")
_DRAMA_STAGE_ONLY_RE = re.compile(r"^\s*\([^()\n]{1,300}\)\s*$")
_DRAMA_HEADING_RE = re.compile(r"^\s*(?:अंक|दृश्य|स्थान|मंच|पर्दा|प्रवेश|निर्गमन|दृश्य[- ]?परिवर्तन)\b", re.I)

def _normalize_drama_paragraphs(text: str) -> list[str]:
    """Convert OCR line wrapping into logical drama paragraphs.

    A character's continuous speech is ALWAYS kept in one paragraph. Physical
    OCR/page line breaks and blank lines do not split a dialogue. A new
    paragraph starts only at a new speaker, a scene/heading line, or a
    standalone parenthesized stage direction.
    """
    source = _normalize_input(text)
    if not source:
        return []

    raw_lines = source.splitlines()
    paragraphs: list[str] = []
    current: str | None = None
    current_is_speaker = False

    def flush() -> None:
        nonlocal current, current_is_speaker
        if current:
            value = re.sub(r"[ \\t]+", " ", current).strip()
            if value:
                paragraphs.append(value)
        current = None
        current_is_speaker = False

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            # Blank lines in OCR are unreliable. Never split a character's
            # dialogue because of one; for ordinary setting/prose they remain
            # a paragraph boundary.
            if current and not current_is_speaker:
                flush()
            continue

        is_speaker = bool(_DRAMA_SPEAKER_RE.match(line))
        is_stage = bool(_DRAMA_STAGE_ONLY_RE.match(line))
        is_heading = bool(_DRAMA_HEADING_RE.match(line))

        if is_speaker or is_stage or is_heading:
            flush()
            current = line
            current_is_speaker = is_speaker
            continue

        if current is None:
            current = line
            current_is_speaker = False
        else:
            # Continuation of the current logical paragraph. In particular,
            # never split a long character dialogue merely because OCR wrapped
            # it across several physical lines.
            current += " " + line

    flush()
    return paragraphs

_DOCX_CHARACTER_CUE_RE = re.compile(r"^\s*([^\n:：ः]{1,40}[:：ः])\s*")
_DOCX_STAGE_DIRECTION_RE = re.compile(r"\([^()\n]{1,300}\)")
_DOCX_LOCATION_RE = re.compile(r"^\s*(स्थानः|स्थान:)\s*.*$", re.UNICODE)
_DOCX_SCENE_HEADING_RE = re.compile(r"^\s*(?:अंक\s+[^:：]+|दृश्य\s+[^:：]+|पहला दृश्य|दूसरा दृश्य|तीसरा दृश्य|चौथा दृश्य)\s*$", re.UNICODE)


def _add_drama_paragraph(doc: "Document", line: str) -> None:
    """Add one logical drama paragraph with semantic Word formatting.

    Character cue -> bold.  Location/setting paragraph -> fully italic.
    Parenthesized acting/stage directions -> italic.  Standalone stage
    directions are therefore also fully italic because their entire content is
    inside parentheses.
    """
    paragraph = doc.add_paragraph()
    line = str(line or '').strip()
    if not line:
        return

    # A location/setting description is a stage-setting paragraph, so italicize
    # the complete paragraph (including ``स्थानः``).
    if _DOCX_LOCATION_RE.match(line) or _DOCX_STAGE_DIRECTION_RE.fullmatch(line):
        run = paragraph.add_run(line)
        run.italic = True
        return

    # Scene headings stay normal (not italic); only the location description is
    # italicized as requested.
    remaining = line
    cue_match = _DOCX_CHARACTER_CUE_RE.match(remaining)
    if cue_match:
        name_run = paragraph.add_run(cue_match.group(1) + " ")
        name_run.bold = True
        remaining = remaining[cue_match.end():]

    pos = 0
    for match in _DOCX_STAGE_DIRECTION_RE.finditer(remaining):
        if match.start() > pos:
            paragraph.add_run(remaining[pos:match.start()])
        italic_run = paragraph.add_run(match.group(0))
        italic_run.italic = True
        pos = match.end()
    if pos < len(remaining):
        paragraph.add_run(remaining[pos:])


def _populate_docx_body(doc: "Document", text: str, book_type: str) -> None:
    normalized_type = str(book_type or "Others").strip().title()
    if normalized_type == "Drama":
        for paragraph_text in _normalize_drama_paragraphs(_postprocess_drama_text(text)):
            _add_drama_paragraph(doc, paragraph_text)
    elif normalized_type == "Poetry":
        for line in text.splitlines():
            doc.add_paragraph(line)
    else:
        blocks = re.split(r"\n\s*\n+", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            reflowed = " ".join(line.strip() for line in block.splitlines() if line.strip())
            doc.add_paragraph(reflowed)

def save_as_docx(
    text: str,
    output_root: str,
    project_name: str,
    part_number: Any = 1,
    book_type: str = "Others",
    filename: Optional[str] = None,
    nested: bool = True,
) -> str:
    """Save corrected/formatted text as a Word (.docx) document.

    If ``nested`` is True (default), the file is saved to
    ``output_root/project_name/part_number/filename`` — any missing
    folders are created. Pass ``nested=False`` when ``output_root`` is
    ALREADY the exact target folder (e.g. a per-project-per-part folder
    your own app already resolved) — in that case the file is written
    directly into ``output_root`` with no extra subfolders, avoiding a
    double-nested path like .../ProjectName/ProjectName/1/1/file.docx.

    Formatting is applied per book type (see ``_populate_docx_body``):
    Drama gets bold character cues and italic stage directions; Poetry
    keeps every line as its own paragraph; Novel and Others reflow
    blank-line-separated blocks into normal paragraphs.

    Returns the full path of the saved file. Raises RuntimeError if
    python-docx is not installed, or ValueError if the text is empty.
    """
    if Document is None:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    source = _normalize_input(text)
    if not source:
        raise ValueError("Cannot save an empty document to .docx.")

    project_folder = _sanitize_path_component(project_name, "project")
    part_folder = _sanitize_path_component(part_number, "part_1")

    if nested:
        target_dir = Path(output_root) / project_folder / part_folder
    else:
        target_dir = Path(output_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = _sanitize_path_component(filename, "") if filename else ""
    if not safe_filename:
        safe_filename = f"{project_folder}_{part_folder}.docx"
    if not safe_filename.lower().endswith(".docx"):
        safe_filename += ".docx"

    doc = Document()
    _populate_docx_body(doc, source, book_type)

    target_path = target_dir / safe_filename
    doc.save(str(target_path))
    return str(target_path)


def consume_correction_warnings() -> list[str]:
    """Return and clear warnings generated by the latest correction run."""
    warnings = list(LAST_CORRECTION_WARNINGS)
    LAST_CORRECTION_WARNINGS.clear()
    return warnings


def get_last_correction_warnings(clear: bool = False) -> list[str]:
    """Return warnings from the most recent document-correction operation.

    This name is used by the current F5PdfStudio ``app.py``.  Warnings are not
    cleared by default, allowing the UI to read them more than once.  Pass
    ``clear=True`` when they should be consumed.
    """
    warnings = list(LAST_CORRECTION_WARNINGS)
    if clear:
        LAST_CORRECTION_WARNINGS.clear()
    return warnings


def get_last_saved_docx_path() -> Optional[str]:
    """Return the path of the most recently saved .docx file, if any.

    Set by ``correct_complete_text(save_docx=True, ...)``. Returns ``None``
    if no document has been saved yet in this process.
    """
    return LAST_SAVED_DOCX_PATH


def unload_ollama_model(
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 30,
) -> tuple[bool, str]:
    """Ask Ollama to release a loaded model from RAM and GPU memory.

    Ollama unloads a model when ``keep_alive`` is zero.  Returning a status and
    message matches ``check_ollama`` and is convenient for Gradio callbacks.
    """
    model = str(model or DEFAULT_MODEL).strip()
    if not model:
        return False, "No Ollama model was selected."

    try:
        response = requests.post(
            f"{_base_url(base_url)}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=(10, max(15, int(timeout))),
        )
        response.raise_for_status()
    except requests.Timeout:
        return False, f"Timed out while unloading Ollama model '{model}'."
    except requests.ConnectionError:
        return False, "Cannot connect to Ollama. Start Ollama and try again."
    except requests.RequestException as exc:
        return False, f"Could not unload Ollama model '{model}': {exc}"

    return True, f"Ollama model '{model}' was unloaded from memory."


def _clean_title(value: Any) -> str:
    value = str(value or "").strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n\"'`*#:-")


_EXPLICIT_HEADING = re.compile(
    r"^(?:अध्याय|पाठ|भाग|खंड|खण्ड|प्रकरण|सर्ग|कहानी|परिशिष्ट|"
    r"chapter|part|section|book|lesson|appendix)"
    r"\s*[-–—:।.]?\s*"
    r"(?:\d+|[ivxlcdm]+|प्रथम|द्वितीय|तृतीय|चतुर्थ|पंचम|षष्ठ|"
    r"सप्तम|अष्टम|नवम|दशम|एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस|"
    r"one|two|three|four|five|six|seven|eight|nine|ten)?"
    r"(?:\s*[-–—:।.]\s*.*)?$",
    re.I,
)


def _is_heading(line: str, previous_blank: bool, next_blank: bool,
                book_type: str = "Others") -> bool:
    line = _clean_title(line)
    if not line or len(line) > 140 or len(line.split()) > 16:
        return False
    if _EXPLICIT_HEADING.match(line):
        return True
    # A short isolated verse is not a chapter. Poetry accepts only explicit
    # structural labels (अध्याय, सर्ग, कविता, गीत, भाग, etc.).
    if str(book_type or "").strip().casefold() == "poetry":
        return bool(re.match(
            r"^(?:कविता|गीत|खंडकाव्य|महाकाव्य)(?:\s|[-–—:।.]|$)",
            line,
            re.I,
        ))
    if re.match(r"^(?:\d+|[ivxlcdm]+)\s*[.):।-]\s+\S+", line, re.I):
        return True
    return (
        previous_blank
        and next_blank
        and 2 <= len(line) <= 80
        and len(line.split()) <= 9
        and not re.search(r"[.!?।,;:]$", line)
    )


def _local_headings(text: str, offset: int = 0, line_offset: int = 0,
                    book_type: str = "Others") -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    results: list[dict[str, Any]] = []
    position = offset
    for index, raw in enumerate(lines):
        title = _clean_title(raw.rstrip("\r\n"))
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if _is_heading(title, previous_blank, next_blank, book_type):
            results.append({
                "title": title,
                "position": position,
                "line_number": line_offset + index + 1,
            })
        position += len(raw)
    return results


def _json_array(response: str) -> list[Any]:
    response = _strip_model_wrappers(response)
    try:
        value = json.loads(response)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("headings", "chapters", "results", "items"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    return candidate
        return []
    except json.JSONDecodeError:
        start, end = response.find("["), response.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            value = json.loads(response[start : end + 1])
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []


def _line_chunks(text: str, maximum: int) -> list[tuple[int, int, str]]:
    """Return ``(character_offset, line_offset, chunk_text)`` tuples."""
    maximum = max(800, min(int(maximum), 3500))
    chunks: list[tuple[int, int, str]] = []
    buffer: list[str] = []
    size = 0
    char_offset = 0
    line_offset = 0
    chunk_char_offset = 0
    chunk_line_offset = 0
    for line in text.splitlines(keepends=True):
        if buffer and size + len(line) > maximum:
            chunks.append((chunk_char_offset, chunk_line_offset, "".join(buffer)))
            char_offset += size
            line_offset += len(buffer)
            chunk_char_offset = char_offset
            chunk_line_offset = line_offset
            buffer, size = [], 0
        buffer.append(line)
        size += len(line)
    if buffer:
        chunks.append((chunk_char_offset, chunk_line_offset, "".join(buffer)))
    return chunks


def detect_chapter_titles(
    text: str,
    model: str = DEFAULT_MODEL,
    language: str = "Hindi/English",
    chunk_size: int = DEFAULT_CHAPTER_CHUNK_SIZE,
    progress_callback: Optional[Callable[..., Any]] = None,
    book_type: str = "Others",
    *,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 180,
    use_ollama: bool = True,
    **_: Any,
) -> list[dict[str, Any]]:
    """Find chapter headings without ever failing an entire document.

    Ollama returns only title and local line number. If it times out, reaches its
    token limit, or emits malformed JSON, conservative regex detection is used.
    """
    source = _normalize_input(text)
    if not source:
        return []
    chunks = _line_chunks(source, chunk_size)
    found: list[dict[str, Any]] = []

    for index, (char_offset, line_offset, section) in enumerate(chunks, start=1):
        _notify(
            progress_callback,
            f"Detecting chapters: section {index}/{len(chunks)}",
            index,
            len(chunks),
        )
        detected: list[dict[str, Any]] = []
        if use_ollama:
            poetry_rule = (
                "This is poetry. Ordinary verse lines and four-line stanzas are NEVER chapters. "
                "Return only lines explicitly labelled अध्याय, सर्ग, कविता, गीत, भाग or खंड."
                if str(book_type or "").strip().casefold() == "poetry"
                else "Do not classify ordinary sentences or paragraphs as headings."
            )
            prompt = f"""Find only chapter, part, lesson, story, section or major heading lines in this {language} {book_type} book extract.
{poetry_rule}
Return ONLY a compact JSON array. No Markdown and no explanation.
Each item must be: {{"title":"exact heading","line":12}}
The line number is 1-based within INPUT. Do not return paragraphs.
Return [] when there is no heading.

INPUT:
{section}"""
            try:
                response = call_ollama(
                    prompt,
                    model=model,
                    system="Return compact valid JSON only. Never reproduce the input document.",
                    base_url=base_url,
                    timeout=timeout,
                    temperature=0.0,
                    num_predict=512,
                    num_ctx=4096,
                    format="json",
                    retries=0,
                )
                section_lines = section.splitlines(keepends=True)
                for item in _json_array(response):
                    if not isinstance(item, dict):
                        continue
                    title = _clean_title(item.get("title"))
                    try:
                        local_line = int(item.get("line", 0))
                    except (TypeError, ValueError):
                        continue
                    if not title or not (1 <= local_line <= len(section_lines)):
                        continue
                    actual = _clean_title(section_lines[local_line - 1].strip())
                    previous_blank = local_line == 1 or not section_lines[local_line - 2].strip()
                    next_blank = local_line == len(section_lines) or not section_lines[local_line].strip()
                    # Reject hallucinated/reworded titles and all model-proposed
                    # lines that fail the same conservative local heading rule.
                    if title.casefold() != actual.casefold():
                        continue
                    if not _is_heading(actual, previous_blank, next_blank, book_type):
                        continue
                    local_position = sum(len(x) for x in section_lines[: local_line - 1])
                    detected.append({
                        "title": title,
                        "position": char_offset + local_position,
                        "line_number": line_offset + local_line,
                    })
            except Exception as exc:
                LAST_CORRECTION_WARNINGS.append(
                    f"Chapter section {index}/{len(chunks)} used local detection: {exc}"
                )

        # Always merge local explicit headings; this also covers invalid JSON.
        detected.extend(_local_headings(section, char_offset, line_offset, book_type))
        found.extend(detected)

    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in sorted(found, key=lambda row: int(row.get("position", 0))):
        title = _clean_title(item.get("title"))
        position = int(item.get("position", 0))
        key = (position, title.casefold())
        if title and key not in seen:
            seen.add(key)
            unique.append({
                "title": title,
                "position": position,
                "line_number": int(item.get("line_number", 0)),
            })
    _notify(progress_callback, f"Detected {len(unique)} chapter headings.")
    return unique


def format_chapter_titles(chapters: Iterable[dict[str, Any]]) -> str:
    """Optional UI helper: render chapter records as editable plain text."""
    lines: list[str] = []
    for index, chapter in enumerate(chapters, start=1):
        title = _clean_title(chapter.get("title"))
        line_number = int(chapter.get("line_number", 0))
        if title:
            lines.append(f"{index}. {title} (line {line_number})")
    return "\n".join(lines)


def format_chapter_report(chapters: Optional[Iterable[Any]]) -> str:
    """Format chapter-detection results for the Gradio report textbox.

    Older F5PdfStudio ``app.py`` versions import this name.  The helper accepts
    the current list-of-dictionaries result as well as older lists containing
    plain titles or ``(title, line_number)`` tuples.
    """
    if not chapters:
        return "No chapter headings were detected."

    rows: list[str] = []
    for index, chapter in enumerate(chapters, start=1):
        title = ""
        line_number = 0
        position: Optional[int] = None

        if isinstance(chapter, dict):
            title = _clean_title(
                chapter.get("title")
                or chapter.get("chapter")
                or chapter.get("name")
            )
            try:
                line_number = int(
                    chapter.get("line_number")
                    or chapter.get("line")
                    or 0
                )
            except (TypeError, ValueError):
                line_number = 0
            try:
                if chapter.get("position") is not None:
                    position = int(chapter.get("position"))
            except (TypeError, ValueError):
                position = None
        elif isinstance(chapter, (tuple, list)):
            if chapter:
                title = _clean_title(chapter[0])
            if len(chapter) > 1:
                try:
                    line_number = int(chapter[1])
                except (TypeError, ValueError):
                    line_number = 0
        else:
            title = _clean_title(chapter)

        if not title:
            continue

        location = ""
        if line_number > 0:
            location = f" — line {line_number}"
        elif position is not None:
            location = f" — character {position}"
        rows.append(f"{index}. {title}{location}")

    if not rows:
        return "No chapter headings were detected."
    return f"Detected {len(rows)} chapter heading(s):\n\n" + "\n".join(rows)

# --- Functions merged from hindi_correct.py ---
def build_system_prompt(text_type):
    return COMMON_RULES + TYPE_PROMPT_ADDITIONS.get(text_type, TYPE_PROMPT_ADDITIONS["prose"])

def detect_text_type(raw_lines):
    """Lightweight heuristic — good enough to pick a sensible default, with
    --type as a manual override for when it guesses wrong. Looks for:
    drama: a high ratio of "CharacterName:" cue lines
    poetry: numbered poem markers like "(1)", "(2)" (a strong, reliable
        signal for an anthology), OR many short lines — checked more
        broadly than just comma-containing lines, since plenty of real
        verse doesn't lean on a mid-line comma at all
    prose: falls back here otherwise (long-ish flowing paragraphs)"""
    non_blank = [l for l in raw_lines if l.strip()]
    if not non_blank:
        return "prose"

    cue_pattern = re.compile(r'^[^\s:ःघ्ण्]{1,20}[:ः]\s*\S')
    cue_ratio = sum(1 for l in non_blank if cue_pattern.match(l.strip())) / len(non_blank)

    marker_ratio = sum(1 for l in non_blank if is_passthrough_marker_line(l)) / len(non_blank)
    short_comma_ratio = sum(1 for l in non_blank if len(l.strip()) < 60 and "," in l) / len(non_blank)
    short_line_ratio = sum(1 for l in non_blank if len(l.strip()) < 70) / len(non_blank)
    avg_len = sum(len(l) for l in non_blank) / len(non_blank)

    if cue_ratio > 0.12:
        return "drama"
    if marker_ratio > 0.005 or (short_comma_ratio > 0.15 and avg_len < 80) or short_line_ratio > 0.6:
        return "poetry"
    return "prose"

def fix_legacy_corruption(text):
    """Repair recurring OCR corruption found in this Hindi drama corpus.

    This is intentionally conservative: deterministic fixes are limited to
    high-confidence OCR artifacts. Literary sentence reconstruction remains
    Ollama's job.
    """
    text = str(text or "")

    # Speaker/location cue artifacts.
    text = re.sub(r'(?m)^([^\n:ः]{1,40}[ः:])\s*(?:ण्)+', r'\1 ', text)

    # Repeated/isolated legacy OCR markers.
    text = re.sub(r'(?:ण्){2,}', ' ', text)
    text = text.replace('ण्', ' ')
    # OCR often duplicates anusvara/chandrabindu after a word (e.g. प्राणोंंं).
    text = re.sub(r'([ंँ])\1+', r'\1', text)
    # Known OCR split in this manuscript: ``शी? रता`` / ``घ् रता``.
    text = text.replace('शी? रता', 'शीघ्रता')
    text = text.replace('घ् रता', 'घ्रता')
    text = text.replace('शीघ्रता', 'शीघ्रता')
    text = text.replace('घ्', '?')

    # Common glued OCR punctuation and duplicated glyphs.
    for word in ["है", "थे", "थी", "गए", "हूं", "हूँ", "गया", "गई", "था", "थीं"]:
        text = text.replace(f"{word}ए", f"{word},")
    text = text.replace('ययुद्ध', 'युद्ध')

    # Broken parenthesized stage directions from this OCR source.
    stage_words = r'(?:क्रोधित(?: होकर)?|क्रोध से|धीरे(?: से)?|आश्चर्य(?: से)?|मुस्कराते(?: हुए)?|रोते(?: हुए)?|हँसते(?: हुए)?)'
    text = re.sub(
        rf'(?m)^[ \t]*य({stage_words})द्ध[ \t]*$',
        lambda m: '(' + m.group(1).strip() + ')',
        text,
    )
    # Inline OCR stage-direction corruption, including forms such as
    # ``शक्तिः यक्रोधित से होकरद्ध रामसिंह``.  The emotion belongs to the
    # current speaker and must remain inline, in parentheses.
    text = re.sub(
        r'य(?:क्रोधित|क्रोध)\s*(?:से\s*)?होकरद्ध',
        '(क्रोधित होकर)',
        text,
        flags=re.UNICODE,
    )
    text = re.sub(
        r'य(धीरे(?:\s+से)?|आश्चर्य(?:\s+से)?|मुस्कराते(?:\s+हुए)?|रोते(?:\s+हुए)?|हँसते(?:\s+हुए)?)द्ध',
        lambda m: '(' + re.sub(r'\s+', ' ', m.group(1)).strip() + ')',
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r'यदोनों का प्रस्थानद्ध', '(दोनों का प्रस्थान)', text)
    text = re.sub(r'ण्पट परिवर्त्तनण्', '(पट परिवर्तन)', text)
    text = re.sub(r'(?m)^\s*पट परिवर्त(?:्त|t)न\s*$', '(पट परिवर्तन)', text)
    text = re.sub(r'(?m)^\s*पट परिवर्तन\s*$', '(पट परिवर्तन)', text)

    replacements = {
        'वचाकर': 'बचाकर', 'वचाने': 'बचाने', 'वचाया': 'बचाया',
        'वच गये': 'बच गये', 'वच निकल': 'बच निकल', 'वच निकला': 'बच निकला',
        'वच निकले': 'बच निकले', 'बच निकल कर': 'बच निकलकर',
        'राण प्रताप': 'राणा प्रताप', 'माहारणा': 'महाराणा',
        'मेवाण': 'मेवाड़', 'मेवाड़': 'मेवाड़', 'मेवाण्': 'मेवाड़',
        'सीसौदिया वेश': 'सिसोदिया वंश', 'सिसौदिया वेश': 'सिसोदिया वंश',
        'कानो से': 'कानों से', 'आखों': 'आँखों', 'प्राणो': 'प्राणों',
        'उन्हे': 'उन्हें', 'हुये': 'हुए', 'लिये': 'लिए', 'सवसे': 'सबसे',
        'नही': 'नहीं', 'कहॉ': 'कहाँ', 'गये': 'गए', 'रहेंगें': 'रहेंगे',
        'रहेगें': 'रहेंगे', 'कर रहें': 'कर रहे', 'विताया': 'बिताया',
        'बलिचढ़ा': 'बलि चढ़ा', 'हंसते हंसते': 'हँसते-हँसते',
        'अंग अंग': 'अंग-अंग', 'रोम रोम': 'रोम-रोम',
        'राज चिन्ह': 'राजचिह्न', 'राज चिन्हो': 'राजचिह्नों',
        'राज चिन्हें': 'राजचिह्नों', 'राजचिह्नें': 'राजचिह्नों',
        'राजसिंहासन': 'राजसिंहासन', 'कर्त्तव्य': 'कर्तव्य',
        'छत्र चैवर': 'छत्र, चँवर', 'एंव': 'एवं', 'राजचिंह': 'राजचिह्न',
        'प्राणो के': 'प्राणों के', 'अनेको': 'अनेकों', 'बहरहा': 'बह रहा',
        'चारो और': 'चारों ओर', 'चारो ओर': 'चारों ओर', 'परन्तु': 'परंतु',
        'थिलिलता': 'शिथिलता', 'दवाव': 'दबाव', 'उन्होने': 'उन्होंने',
        'रूक कर': 'रुककर', 'माग रहा': 'माँग रहा', 'माग रहा हूं': 'माँग रहा हूँ',
        'माग रहा हूँ': 'माँग रहा हूँ', 'वलिदान': 'बलिदान', 'जन्म भुमि': 'जन्मभूमि',
        'जन्म भूमि': 'जन्मभूमि', 'इस का': 'इसका', 'इन राजचिह्नो': 'इन राजचिह्नों',
        'राजचिह्नो': 'राजचिह्नों', 'राजचिह्नें': 'राजचिह्नों',
        'पहिलाण्': 'पहिला', 'पहिला दृश्य': 'पहला दृश्य',
        'रहे है': 'रहे हैं', 'रहेगें': 'रहेंगे', 'रहेंगें': 'रहेंगे',
        'दोह करके': 'द्रोह करके', 'प्राणोंं': 'प्राणों', 'नहींंं': 'नहीं',
        'नहींं': 'नहीं', 'आँखोंं': 'आँखों', 'उन्हेंंं': 'उन्हें', 'उन्हेंं': 'उन्हें',
        'करेंए': 'करें।', 'दिएए': 'दिए।', 'लियाए': 'लिया।',
        'जीयेएमेवाड़': 'जीये, मेवाड़', 'गौख': 'गौरव', 'राण सलीम': 'राणा सलीम', 'ख ड ख ड': 'खण्ड-खण्ड',
        'प्रलय काल': 'प्रलयकाल',
        'राजचिह्नोंं': 'राजचिह्नों', 'राजचिह्नोंंं': 'राजचिह्नों',
        'राजचिह्नो': 'राजचिह्नों', 'राजचिह्नें': 'राजचिह्नों',
        'क्रोधित से होकर': 'क्रोधित होकर',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # High-confidence sentence repairs seen in this manuscript.
    exact_repairs = {
        'क्षेत्र युद्ध': 'क्षेत्र। युद्ध',
        'भाग रहे हैंं': 'भाग रहे हैं',
        'क्या प्रताप भाग रहे हैं': 'क्या प्रताप भाग रहे हैं',
        'अरे यह तो चेतक है': 'अरे! यह तो चेतक है!',
        'क्या देख रहें हैं': 'क्या देख रहे हैं',
        'हैंं': 'हैं', 'ँं': 'ँ', 'ंँ': 'ं',
        'स्वदेश का गौरव से जीये': 'स्वदेश के गौरव से जीये',
        'मैं ठीक कहता हूं': 'मैं ठीक कहता हूँ',
        'मैं ठीक कहता है': 'मैं ठीक कहता हूँ',
        'प्राणोंं': 'प्राणों',
        'नहींए': 'नहीं,',
        'बहुत बड़ा संग्राम लड़ने को': 'बहुत बड़ा संग्राम लड़ने को',
    }
    for bad, good in exact_repairs.items():
        text = text.replace(bad, good)

    # Repair common malformed stage-direction markup produced by small models.
    text = text.replace('*(क्रोधित होकरद)*', '(क्रोधित होकर)')
    text = text.replace('(क्रोधित होकरद)', '(क्रोधित होकर)')
    text = text.replace('(क्रोध से होकरद)', '(क्रोध से होकर)')
    text = re.sub(r'यक्रोधित\s*(?:से\s*)?होकरद्ध', '(क्रोधित होकर)', text)
    text = text.replace('*(दोनों का प्रस्थान)*', '(दोनों का प्रस्थान)')
    text = text.replace('*(पट परिवर्तन)*', '(पट परिवर्तन)')

    # ``द्ध`` appended after a completed sentence is a legacy closing marker.
    text = re.sub(r'(?<=[।!?])द्ध(?=\s|$)', '', text)

    # Remove accidental Markdown emphasis around drama text; Word formatting is
    # applied structurally by python-docx instead.
    text = text.replace('**', '').replace('__', '').replace('*', '')

    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Avoid duplicate punctuation introduced by OCR + model cleanup.
    text = re.sub(r'([!?।])\s*([!?।])+', r'\1', text)
    text = re.sub(r' +([,।!?])', r'\1', text)
    text = re.sub(r'([,!?])(?=\S)', r'\1 ', text)
    return text


def _postprocess_drama_text(text: str) -> str:
    """Final deterministic drama cleanup after Ollama."""
    text = fix_legacy_corruption(text)
    text = _normalize_input(text)

    paragraphs = _normalize_drama_paragraphs(text)
    cleaned: list[str] = []
    for para in paragraphs:
        para = re.sub(r'[ \t]{2,}', ' ', para).strip()
        if not para:
            continue
        # Exact adjacent duplicate paragraph from page/column OCR duplication.
        if cleaned and para == cleaned[-1]:
            continue
        cleaned.append(para)

    return "\n\n".join(cleaned)

def sanitize_text(text):
    text = "".join(ch for ch in text if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C")
    return unicodedata.normalize("NFC", text)

def is_valid_batch(original_lines, candidate_text):
    """Returns (is_valid, reason) — reason explains exactly why a batch was
    rejected, so failures are debuggable from the console log alone."""
    if not candidate_text.strip():
        return False, "empty response"
    hit_markers = [m for m in _INVALID_MARKERS if m in candidate_text]
    if hit_markers:
        return False, f"contains commentary/markdown marker(s): {hit_markers}"
    candidate_lines = candidate_text.split("\n")
    if len(candidate_lines) != len(original_lines):
        return False, f"line count mismatch (gave {len(original_lines)}, got {len(candidate_lines)} back)"
    orig_len = len("\n".join(original_lines))
    if len(candidate_text) > orig_len * 1.8 + 30:
        return False, f"response too long ({len(candidate_text)} chars vs {orig_len} input — likely added extra text)"
    return True, ""

def is_decorative_line(line):
    """Lines that are just repeated symbols (used as visual separators
    between poems/sections) — nothing here for the model to correct, and
    they can accidentally trip the markdown-lookalike validator. Skip the
    model entirely and pass these through unchanged, same as blank lines."""
    s = line.strip()
    if not s:
        return False
    return bool(re.fullmatch(r'[-—_=*~.·•\s]{4,}', s))

def is_passthrough_marker_line(line):
    """Lines that are just a poem/section number, like '(8)', '11.', '(11)'
    — no real language content, so nothing for the model to correct. Giving
    the model a lone number tends to produce hallucinated/expanded output
    since there's so little for it to work with, so route these around it
    entirely, same as blank/decorative lines."""
    s = line.strip()
    if not s:
        return False
    return bool(re.fullmatch(r'\(?\d{1,4}\)?[.)]?', s))

def is_skippable_line(line):
    return line.strip() == "" or is_decorative_line(line) or is_passthrough_marker_line(line)

def file_signature(lines):
    """Fingerprint the source content so a stale checkpoint from a
    different/edited file is never applied by mistake. Line-indexed
    checkpoints (below) don't need to track batching logic separately —
    they're keyed by original line number, which stays stable no matter
    how batching groups lines together."""
    h = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
    return h

def load_checkpoint(path, sig):
    """Returns a dict: {line_index: {"text": corrected_line, "ok": bool}}.
    Transparently migrates the old batch-indexed checkpoint format (from
    earlier versions of this script) into the new line-indexed format, so
    already-completed work is never lost when the batching logic changes."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("signature") != sig:
            print("  (source file changed since last run — starting fresh)")
            return {}
        if "lines" in data:
            return {int(k): v for k, v in data["lines"].items()}
        if "batches" in data:
            # Old format: reconstruct line indices by replaying the batch
            # sizes in order (safe because line order/content is unchanged).
            print("  (migrating old checkpoint format — keeping your progress)")
            line_cache = {}
            cursor = 0
            for bi in sorted(int(k) for k in data["batches"].keys()):
                b = data["batches"][str(bi)]
                corrected = b["corrected"]
                ok = b["ok"]
                for text in corrected:
                    line_cache[cursor] = {"text": text, "ok": ok}
                    cursor += 1
            return line_cache
        return {}
    except Exception:
        return {}

def save_checkpoint(path, sig, line_cache):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"signature": sig, "lines": {str(k): v for k, v in line_cache.items()}}, f, ensure_ascii=False)

def add_formatted_line(doc, line):
    """One paragraph per line. Bolds a leading 'Character:' cue, italicizes
    any (parenthetical) stage direction wherever it appears in the line,
    everything else stays plain. Blank/decorative/marker lines are added
    as plain paragraphs unchanged."""
    p = doc.add_paragraph()
    if line.strip() == "" or is_decorative_line(line) or is_passthrough_marker_line(line):
        p.add_run(line)
        return

    remaining = line
    cue_match = CHARACTER_CUE_RE.match(remaining)
    if cue_match:
        name_run = p.add_run(cue_match.group(1) + " ")
        name_run.bold = True
        remaining = remaining[cue_match.end():]

    pos = 0
    for m in STAGE_DIRECTION_RE.finditer(remaining):
        if m.start() > pos:
            p.add_run(remaining[pos:m.start()])
        italic_run = p.add_run(m.group(0))
        italic_run.italic = True
        pos = m.end()
    if pos < len(remaining):
        p.add_run(remaining[pos:])

def write_formatted_docx(corrected_lines, out_path):
    """Produces a Word doc from already-corrected lines: character names
    bold, stage directions italic and kept in their (brackets)."""
    if Document is None:
        print("  WARNING: python-docx not installed — skipping formatted .docx output.")
        print("  Run: pip install python-docx")
        return
    doc = Document()
    for paragraph_text in _normalize_drama_paragraphs("\n".join(corrected_lines)):
        add_formatted_line(doc, paragraph_text)
    doc.save(out_path)
    print(f"  Formatted Word doc: {out_path}")

def rhyme_key(line, tail):
    s = _TRAILING_END_PUNCT_RE.sub('', line.rstrip()).strip()
    s = _RHYME_STRIP_RE.sub('', s)
    return s[-tail:] if len(s) >= tail else s

def lines_rhyme(a, b):
    """Approximate rhyme check, not true phonetic analysis: true Hindi
    rhyme endings vary in length (e.g. '-ाता' is 3 characters, '-ाई' is
    only 2), so this checks a couple of common tail lengths and calls it
    a match if any agree. Will catch clear cases (विधाता/उद्गाता, बरसाई/
    दिखाई) but can miss or misjudge subtler ones — a reasonable first
    pass, not a final word; worth a human read-through afterward."""
    for tail in (2, 3, 4):
        ka, kb = rhyme_key(a, tail), rhyme_key(b, tail)
        if ka and kb and ka == kb:
            return True
    return False

def assign_rhyme_punctuation(lines):
    """For a stanza's worth of poetic lines: if a line rhymes with the
    next one, they form a couplet — the first gets a single danda (।),
    the second a double danda (॥), and BOTH are consumed (the second one
    is not then independently re-checked against the line after it,
    which was the earlier bug — a line that just closed a couplet
    shouldn't also be treated as opening a new one). A line that doesn't
    rhyme with the next one gets a comma (the thought carries on). The
    very last line of the stanza always closes with ॥, marking the end
    of the passage."""
    n = len(lines)
    endings = [None] * n
    i = 0
    while i < n:
        if i == n - 1:
            endings[i] = '॥'
            i += 1
        elif lines_rhyme(lines[i], lines[i + 1]):
            endings[i] = '।'
            endings[i + 1] = '॥'
            i += 2
        else:
            endings[i] = ','
            i += 1
    return endings

def reformat_poetry_lines(corrected_lines):
    """Structural reformatting layer for poetry, applied AFTER spelling
    correction (kept separate so the AI correction step stays simple and
    reliable). Rules:
      1. A line ending in a comma is joined with the following line(s)
         until one is reached that does NOT end in a comma — i.e. a
         comma means the thought continues onto the next line.
      2. Within each stanza, lines are end-punctuated by approximate rhyme
         matching (see assign_rhyme_punctuation): a line only gets a danda
         once it finds an actual rhyming partner nearby; otherwise it gets
         a comma, since the thought is still continuing — UNLESS it
         already ends in ? or !, which is left alone since forcing those
         to a plain danda would change a question/exclamation into a
         statement.
      3. A blank line is inserted after every 4 poetic lines.
    Blank lines, decorative separators, and poem-number markers already
    in the text act as hard breaks: they reset stanza grouping (rhyme
    pairing restarts fresh each stanza) and are never merged across.
    """
    # Step 1: merge comma-continued lines into logical units.
    units = []  # ('text', merged_string) or ('special', original_line)
    buffer = []
    for line in corrected_lines:
        if is_skippable_line(line):
            if buffer:
                units.append(('text', ' '.join(buffer)))
                buffer = []
            units.append(('special', line))
            continue
        buffer.append(line.strip())
        if not line.rstrip().endswith(','):
            units.append(('text', ' '.join(buffer)))
            buffer = []
    if buffer:
        units.append(('text', ' '.join(buffer)))

    # Step 2: group into stanzas (split at 'special' units), rhyme-punctuate
    # each stanza independently, then reassemble with the specials in place.
    output = []
    stanza_buffer = []

    def flush_stanza():
        if not stanza_buffer:
            return
        texts = [c for c in stanza_buffer]
        # Chunk into groups of up to 4 lines — each chunk is its own
        # independent rhyme/punctuation unit (so its last line always
        # closes with ॥), AND becomes one visual stanza separated by a
        # blank line. This keeps the "every 4 lines" rule and the rhyme
        # logic consistent with each other instead of operating on
        # different groupings.
        for start in range(0, len(texts), 4):
            chunk = texts[start:start + 4]
            forced = [t.rstrip().endswith('?') or t.rstrip().endswith('!') for t in chunk]
            to_punctuate_idx = [i for i, f in enumerate(forced) if not f]
            sub_lines = [chunk[i] for i in to_punctuate_idx]
            sub_endings = assign_rhyme_punctuation(sub_lines)
            endings_by_idx = dict(zip(to_punctuate_idx, sub_endings))

            for i, t in enumerate(chunk):
                if forced[i]:
                    finished = t.rstrip()
                else:
                    clean = _TRAILING_END_PUNCT_RE.sub('', t.rstrip())
                    finished = clean + endings_by_idx[i]
                output.append(finished)
            output.append('')
        stanza_buffer.clear()

    for kind, content in units:
        if kind == 'special':
            flush_stanza()
            output.append(content)
            continue
        stanza_buffer.append(content)
    flush_stanza()

    return output

def write_reformatted_poetry(corrected_lines, out_path):
    reformatted = reformat_poetry_lines(corrected_lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(reformatted))
    print(f"  Reformatted poetry: {out_path}")

def load_manual_overrides(output_dir):
    """A hand-maintained JSON file the user can add to: exact original
    line text -> exact corrected text. Checked BEFORE calling the model,
    so it's a guaranteed, permanent fix for any specific line the AI
    keeps mishandling (e.g. content that accidentally confuses the model
    about its own instructions), no further prompt tuning needed. Format:
    {"original line 1": "corrected line 1", "original line 2": "..."}"""
    path = os.path.join(output_dir, MANUAL_OVERRIDES_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = {k.strip(): v for k, v in data.items()}
        print(f"  (loaded {len(data)} manual override(s) from {MANUAL_OVERRIDES_FILENAME})")
        return data
    except Exception as e:
        print(f"  WARNING: couldn't read {MANUAL_OVERRIDES_FILENAME}: {e}")
        return {}

def process_txt_file(path, output_dir, report_lines, restart, overrides, type_arg):
    filename = os.path.basename(path)

    with open(path, "r", encoding="utf-8") as f:
        original_lines = [l.rstrip("\n") for l in f.readlines()]

    text_type = detect_text_type(original_lines) if type_arg == "auto" else type_arg
    detected_note = " (auto-detected)" if type_arg == "auto" else " (forced via --type)"
    print(f"Processing {filename} ... type: {text_type}{detected_note}")

    # The legacy-font corruption fix (ण्/घ्/य...द्ध substitutions) was
    # reverse-engineered from ONE specific corrupted drama script. Those
    # same characters (especially ण्) are also normal, legitimate parts of
    # everyday Hindi conjuncts (पुण्य, कण्ठ, अरण्य, खण्ड...), so applying
    # this fix outside of drama text risks silently corrupting real words.
    # Only apply it where the corruption pattern was actually confirmed.
    if text_type == "drama":
        raw_lines = [sanitize_text(fix_legacy_corruption(l)) for l in original_lines]
    else:
        raw_lines = [sanitize_text(l) for l in original_lines]

    system_prompt = build_system_prompt(text_type)
    batch_max_chars = BATCH_MAX_CHARS_BY_TYPE.get(text_type, 400)
    num_ctx = NUM_CTX_BY_TYPE.get(text_type, 4096)

    sig = file_signature(raw_lines)
    ckpt_path = os.path.join(output_dir, f".{filename}.progress.json")
    line_cache = {} if restart else load_checkpoint(ckpt_path, sig)
    if restart:
        line_cache = {}

    # Apply manual overrides first — these always win, skip the model entirely.
    # Matched on stripped text so incidental leading/trailing whitespace
    # differences don't silently prevent a match.
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if stripped in overrides:
            line_cache[i] = {"text": overrides[stripped], "ok": True}

    # Build batches as groups of (line_index, line_text)
    indexed = list(enumerate(raw_lines))
    batches = []
    current = []
    current_len = 0
    for idx, line in indexed:
        if is_skippable_line(line):
            if current:
                batches.append(current)
                current, current_len = [], 0
            batches.append([(idx, line)])
            continue
        if current_len + len(line) + 1 > batch_max_chars and current:
            batches.append(current)
            current, current_len = [], 0
        current.append((idx, line))
        current_len += len(line) + 1
    if current:
        batches.append(current)

    total = len(batches)
    for bi, batch in enumerate(batches):
        indices = [i for i, _ in batch]
        texts = [t for _, t in batch]

        # Skip only if EVERY line in this batch already succeeded before.
        if all(i in line_cache and line_cache[i]["ok"] for i in indices):
            continue

        if len(batch) == 1 and is_skippable_line(texts[0]):
            line_cache[indices[0]] = {"text": texts[0], "ok": True}
        else:
            start = time.time()
            fixed, ok = call_ollama(texts, system_prompt, num_ctx)
            elapsed = time.time() - start

            if not ok and len(texts) > 1:
                # The batch as a whole failed (usually a line-merge/split
                # issue). Fall back to correcting each line individually —
                # a single line can't be merged with a neighbor, so this
                # sidesteps the problem structurally instead of retrying
                # the same failure mode again.
                print(f"    (batch {bi+1} failed as a group — retrying its {len(texts)} lines individually)")
                fixed = []
                all_ok = True
                for single_text in texts:
                    s_fixed, s_ok = call_ollama([single_text], system_prompt, num_ctx)
                    fixed.append(s_fixed[0] if s_fixed else single_text)
                    all_ok = all_ok and s_ok
                ok = all_ok

            for i, t in zip(indices, fixed):
                line_cache[i] = {"text": t, "ok": ok}
            print(f"  batch {bi+1}/{total} done ({elapsed:.1f}s, {len(indices)} line(s))" + ("" if ok else "  [UNVERIFIED]"))

        save_checkpoint(ckpt_path, sig, line_cache)  # persist after every batch

    corrected_lines = [line_cache[i]["text"] if i in line_cache else raw_lines[i] for i in range(len(raw_lines))]
    failed_line_numbers = [i + 1 for i in range(len(raw_lines)) if i not in line_cache or not line_cache[i]["ok"]]

    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corrected_lines))

    if text_type == "drama":
        docx_path = os.path.splitext(out_path)[0] + ".docx"
        write_formatted_docx(corrected_lines, docx_path)

    if text_type == "poetry":
        formatted_path = os.path.splitext(out_path)[0] + "_formatted.txt"
        write_reformatted_poetry(corrected_lines, formatted_path)

    # Build a readable diff report
    lines_out = [f"=== {filename} ===\n"]
    if failed_line_numbers:
        lines_out.append(f"⚠ {len(failed_line_numbers)} line(s) could NOT be verified "
                          f"(left as original, review manually): {failed_line_numbers}\n")
    changed_any = False
    for i, (orig, corr) in enumerate(zip(raw_lines, corrected_lines), 1):
        if orig.strip() != corr.strip() and (i not in failed_line_numbers):
            changed_any = True
            lines_out.append(f"[Line {i}]  पहले: {orig}")
            lines_out.append(f"          बाद : {corr}")
    if not changed_any and not failed_line_numbers:
        lines_out.append("(कोई बदलाव नहीं / no changes)")
    report_lines.append("\n".join(lines_out))

    print(f"  Done: {filename} -> {out_path}")
    if failed_line_numbers:
        print(f"  ⚠ {len(failed_line_numbers)} line(s) unverified — see report.")

def main():
    global MODEL
    parser = argparse.ArgumentParser(description="Batch-correct a Hindi manuscript (poetry/drama/prose) via local Ollama")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--type", default="auto", choices=["auto"] + TEXT_TYPES,
                         help="Text type: auto (default, detected per file), poetry, drama, or prose")
    parser.add_argument("--restart", action="store_true", help="Ignore saved progress and reprocess from scratch")
    args = parser.parse_args()

    MODEL = args.model

    os.makedirs(args.output, exist_ok=True)

    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        print("ERROR: Cannot reach Ollama at http://localhost:11434 — is it running?")
        sys.exit(1)

    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=10).json()
        available = [m["name"] for m in tags.get("models", [])]
        if MODEL not in available:
            print(f"ERROR: Model '{MODEL}' not found. Available: {available}")
            print(f"Run: ollama pull {MODEL}")
            sys.exit(1)
    except requests.RequestException as e:
        print(f"WARNING: could not verify model list ({e}), continuing anyway...")

    files = [f for f in os.listdir(args.input) if f.lower().endswith(".txt")]
    if not files:
        print(f"No .txt files found in {args.input}")
        print("(Note: this version handles .txt files. Say the word and I'll add .docx support back in.)")
        sys.exit(0)

    overrides = load_manual_overrides(args.output)

    report_lines = []
    for f in files:
        process_txt_file(os.path.join(args.input, f), args.output, report_lines, args.restart, overrides, args.type)

    report_path = os.path.join(args.output, "correction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))

    print(f"\nAll done. Corrected files + correction_report.txt saved to: {args.output}")

def _legacy_call_ollama(lines, system_prompt, num_ctx=4096, retries=5):
    text = "\n".join(lines)
    # Tight budget: a short line should only ever get a short response.
    # The old floor of 50 gave tiny/single-word lines way more room than
    # they needed, which let the model wander into hallucinated extra
    # content instead of being cut off quickly.
    approx_tokens = max(len(text) // 2, 12)
    num_predict = min(int(approx_tokens * 2.2) + 15, 1200)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_TEMPLATE.format(text=text, n=len(lines))},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "num_thread": os.cpu_count() or 4,
        },
    }

    last_err = ""
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            candidate = resp.json()["message"]["content"].strip()
            valid, reason = is_valid_batch(lines, candidate)
            if valid:
                return candidate.split("\n"), True
            last_err = reason
        except Exception as e:
            body = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    body = e.response.text[:200]
                except Exception:
                    pass
            last_err = f"{e} | {body}"
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))

    preview = " / ".join(lines)[:150]
    print(f"    ! giving up on this batch: {last_err}")
    print(f"    ! input was: {preview}{'...' if len(' / '.join(lines)) > 150 else ''}")
    return lines, False

def call_ollama(prompt_or_lines, *args, **kwargs):
    """Unified Ollama API.

    Accepts both the modern F5PdfStudio signature (prompt string) and the
    legacy hindi_correct.py signature (list of lines, system prompt, ...).
    """
    if isinstance(prompt_or_lines, (list, tuple)):
        return _legacy_call_ollama(prompt_or_lines, *args, **kwargs)
    return _service_call_ollama(prompt_or_lines, *args, **kwargs)


__all__ = [
    "OLLAMA_BASE_URL", "DEFAULT_MODEL", "LAST_CORRECTION_WARNINGS",
    "LAST_SAVED_DOCX_PATH", "get_installed_models", "list_ollama_models",
    "check_ollama_status", "check_ollama_connection", "check_ollama",
    "call_ollama", "validate_corrected_output", "correct_ocr_chunk",
    "correct_complete_text", "save_as_docx", "consume_correction_warnings",
    "get_last_correction_warnings", "get_last_saved_docx_path",
    "unload_ollama_model", "detect_chapter_titles", "format_chapter_titles",
    "format_chapter_report",
    # hindi_correct.py compatibility API
    "OLLAMA_URL", "MODEL", "BATCH_MAX_CHARS_BY_TYPE", "NUM_CTX_BY_TYPE",
    "TEXT_TYPES", "COMMON_RULES", "TYPE_PROMPT_ADDITIONS", "USER_TEMPLATE",
    "build_system_prompt", "detect_text_type", "fix_legacy_corruption",
    "sanitize_text", "is_valid_batch", "is_decorative_line",
    "is_passthrough_marker_line", "is_skippable_line", "file_signature",
    "load_checkpoint", "save_checkpoint", "add_formatted_line",
    "write_formatted_docx", "rhyme_key", "lines_rhyme",
    "assign_rhyme_punctuation", "reformat_poetry_lines",
    "write_reformatted_poetry", "load_manual_overrides", "process_txt_file",
    "main",
]


if __name__ == "__main__":
    main()
