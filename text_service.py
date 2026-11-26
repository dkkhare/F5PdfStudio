import re


def clean_text(text: str):
    """
    Normalize extraction noise without changing the document's structure.

    Preserve heading markers (including Markdown Heading 1/2), paragraph
    boundaries, poetry line breaks, dialogue punctuation and all visible text.
    """
    if text is None:
        return ""

    text = str(text)

    # Normalize platform-specific line endings before touching whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize common invisible characters introduced by PDF/Word extraction.
    text = text.replace("\ufeff", "")   # byte-order mark
    text = text.replace("\u00a0", " ") # non-breaking space
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u2060", "")  # word joiner
    text = text.replace("\u00ad", "")  # soft hyphen

    # Clean horizontal whitespace only; never join separate text lines.
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    # Keep one blank line between paragraphs, headings and verse blocks.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_chapters(text: str):
    chapter_pattern = re.compile(
        r"(?im)^(अध्याय|भाग|खंड|chapter|part)"
        r"\s*[-–—:]?\s*[\d०-९A-Za-z]*.*$"
    )

    matches = list(chapter_pattern.finditer(text))

    if not matches:
        return [{
            "title": "Complete document",
            "text": text,
        }]

    chapters = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        chapters.append({
            "title": match.group(0).strip(),
            "text": text[start:end].strip(),
        })

    return chapters


def split_for_tts(
    text: str,
    max_characters: int = 180,
    book_type: str = "Others",
) -> list[str]:
    """
    Split text into sections that are always within max_characters.

    Splitting preference:
    paragraph → sentence → comma/semicolon → words → hard characters
    """

    text = clean_text(text)

    if not text:
        return []

    # Poetry only: Word Heading 1–5 are standalone title sections.
    # Other non-empty lines are verse; group them in pairs, flushing before
    # the next title and at end-of-book. A final single line is retained.
    if str(book_type or "Others").strip().casefold() == "poetry":
        sections: list[str] = []
        verse_buffer: list[str] = []

        def flush_verse_buffer() -> None:
            while verse_buffer:
                sections.append("\n".join(verse_buffer[:2]))
                del verse_buffer[:2]

        heading_pattern = re.compile(r"^#{1,5}\s+\S")

        def ensure_line_pause(value: str) -> str:
            """Add sentence-ending punctuation so TTS pauses between lines."""
            value = value.strip()
            if value and not re.search(r"[.!?।॥…]$", value):
                value += "."
            return value

        first_content_line = True
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Fallback for DOCX titles whose Heading 1–5 style marker was
            # lost during extraction/editing/proofreading: treat the first
            # non-empty line of a Poetry document as a standalone title.
            is_heading = bool(heading_pattern.match(line))
            if first_content_line and not is_heading:
                is_heading = True
            first_content_line = False

            if is_heading:
                flush_verse_buffer()
                title = re.sub(r"^#{1,5}\s+", "", line).strip()
                if title:
                    sections.append(ensure_line_pause(title))
                continue

            verse_buffer.append(ensure_line_pause(line))
            if len(verse_buffer) == 2:
                sections.append("\n".join(verse_buffer))
                verse_buffer.clear()

        flush_verse_buffer()
        return sections

    max_characters = max(
        80,
        int(max_characters),
    )

    def pack_words(
        value: str,
    ) -> list[str]:
        """
        Hard safety splitter for an oversized sentence.
        """

        words = value.split()
        results = []
        current = ""

        for word in words:
            # Handle malformed OCR strings having no spaces.
            if len(word) > max_characters:
                if current:
                    results.append(current)
                    current = ""

                for start in range(
                    0,
                    len(word),
                    max_characters,
                ):
                    results.append(
                        word[
                            start:
                            start + max_characters
                        ]
                    )

                continue

            candidate = (
                f"{current} {word}".strip()
            )

            if (
                current
                and len(candidate)
                > max_characters
            ):
                results.append(current)
                current = word
            else:
                current = candidate

        if current:
            results.append(current)

        return results

    def split_large_unit(
        value: str,
    ) -> list[str]:
        value = value.strip()

        if not value:
            return []

        if len(value) <= max_characters:
            return [value]

        # Prefer pauses at Hindi/English punctuation.
        clauses = re.split(
            r"(?<=[,;:।.!?])\s*",
            value,
        )

        results = []
        current = ""

        for clause in clauses:
            clause = clause.strip()

            if not clause:
                continue

            if len(clause) > max_characters:
                if current:
                    results.append(current)
                    current = ""

                results.extend(
                    pack_words(clause)
                )

                continue

            candidate = (
                f"{current} {clause}".strip()
            )

            if (
                current
                and len(candidate)
                > max_characters
            ):
                results.append(current)
                current = clause
            else:
                current = candidate

        if current:
            results.append(current)

        return results

    # First separate paragraphs.
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    small_units = []

    for paragraph in paragraphs:
        # Split at sentence-ending punctuation.
        sentences = re.split(
            r"(?<=[।.!?])\s+",
            paragraph,
        )

        for sentence in sentences:
            small_units.extend(
                split_large_unit(sentence)
            )

    # Pack small units without crossing the limit.
    sections = []
    current = ""

    for unit in small_units:
        unit = unit.strip()

        if not unit:
            continue

        candidate = (
            f"{current} {unit}".strip()
        )

        if (
            current
            and len(candidate)
            > max_characters
        ):
            sections.append(current)
            current = unit
        else:
            current = candidate

    if current:
        sections.append(current)

    # Final safety validation.
    oversized = [
        (index, len(section))
        for index, section in enumerate(
            sections,
            start=1,
        )
        if len(section) > max_characters
    ]

    if oversized:
        raise ValueError(
            "TTS splitting failed. Oversized sections: "
            + ", ".join(
                f"{index} ({length} characters)"
                for index, length in oversized
            )
        )

    return sections
