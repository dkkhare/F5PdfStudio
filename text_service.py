import re


def clean_text(text: str):
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
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

    # Preserve verse boundaries and group four lines into one logical section.
    # TTS still synthesizes each line independently and inserts line silence.
    if str(book_type or "Others").strip().casefold() == "poetry":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [
            "\n".join(lines[start:start + 4])
            for start in range(0, len(lines), 4)
        ]

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
