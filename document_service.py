from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import fitz
import pytesseract

from docx import Document
from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image


TESSERACT_DIRECTORY = Path(
    r"C:\Program Files\Tesseract-OCR"
)

TESSERACT_EXE = (
    TESSERACT_DIRECTORY / "tesseract.exe"
)

TESSDATA_DIRECTORY = (
    TESSERACT_DIRECTORY / "tessdata"
)

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_EXE
    )


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
}


def parse_selection_range(
    selection: str,
    total_items: int,
) -> list[int]:
    """
    Convert:

        1-10,15,20-25

    to zero-based item indexes.

    For PDFs, items are pages.
    For Word documents, items are text blocks.
    """

    if total_items <= 0:
        return []

    if not selection or not selection.strip():
        return list(range(total_items))

    selected = set()

    for item in selection.split(","):
        item = item.strip()

        if not item:
            continue

        if "-" in item:
            parts = item.split("-", 1)

            try:
                start = int(parts[0])
                end = int(parts[1])

            except ValueError as error:
                raise ValueError(
                    f"Invalid selection range: {item}"
                ) from error

            if start > end:
                start, end = end, start

            start = max(1, start)
            end = min(total_items, end)

            selected.update(
                range(start - 1, end)
            )

        else:
            try:
                number = int(item)

            except ValueError as error:
                raise ValueError(
                    f"Invalid selection number: {item}"
                ) from error

            if 1 <= number <= total_items:
                selected.add(number - 1)

    if not selected:
        raise ValueError(
            "The supplied range does not select "
            "any available content."
        )

    return sorted(selected)


def extract_pdf(
    pdf_path: str | Path,
    selection_range: str = "",
    ocr_language: str = "hin+eng",
    dpi: int = 300,
    progress_callback: Optional[
        Callable[[int, int, str], None]
    ] = None,
    skip_numbers: Optional[set] = None,
    on_result: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """
    Extract embedded PDF text and run OCR on pages
    containing almost no embedded text.

    ``skip_numbers`` (1-based page numbers already extracted) and ``on_result``
    (called after every finished page) let the caller pause/stop between pages
    and resume later.
    """

    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)

    try:
        selected_pages = parse_selection_range(
            selection_range,
            document.page_count,
        )

        results = []

        for position, page_index in enumerate(
            selected_pages,
            start=1,
        ):
            page = document[page_index]

            if skip_numbers and (page_index + 1) in skip_numbers:
                continue

            if progress_callback:
                progress_callback(
                    position,
                    len(selected_pages),
                    f"PDF page {page_index + 1}",
                )

            text = page.get_text(
                "text",
                sort=True,
            ).strip()

            method = "Embedded PDF text"

            # Use OCR if the page contains almost
            # no extractable text.
            visible_characters = len(
                re.sub(r"\s+", "", text)
            )

            if visible_characters < 25:
                if not TESSDATA_DIRECTORY.exists():
                    raise FileNotFoundError(
                        "Tesseract language-data folder "
                        "was not found:\n"
                        f"{TESSDATA_DIRECTORY}"
                    )

                text_page = page.get_textpage_ocr(
                    language=ocr_language,
                    dpi=int(dpi),
                    full=True,
                    tessdata=str(
                        TESSDATA_DIRECTORY
                    ),
                )

                text = page.get_text(
                    "text",
                    textpage=text_page,
                    sort=True,
                ).strip()

                method = "PDF OCR"

            entry = {
                "number": page_index + 1,
                "location": (
                    f"PDF page {page_index + 1}"
                ),
                "method": method,
                "text": text,
            }
            results.append(entry)
            if on_result:
                on_result(entry)

        return results

    finally:
        document.close()


def iter_word_blocks(
    document: DocumentType,
):
    """
    Yield Word paragraphs and tables in their
    original document order.
    """

    parent_element = document.element.body

    paragraph_map = {
        paragraph._p: paragraph
        for paragraph in document.paragraphs
    }

    table_map = {
        table._tbl: table
        for table in document.tables
    }

    for child in parent_element.iterchildren():
        if child in paragraph_map:
            yield paragraph_map[child]

        elif child in table_map:
            yield table_map[child]


def table_to_text(table: Table) -> str:
    """
    Convert a Word table into readable tab-separated
    text.
    """

    rows = []

    for row in table.rows:
        cells = []

        for cell in row.cells:
            value = " ".join(
                paragraph.text.strip()
                for paragraph in cell.paragraphs
                if paragraph.text.strip()
            )

            cells.append(value)

        if any(cells):
            rows.append(
                "\t".join(cells)
            )

    return "\n".join(rows)


def extract_word_images(
    document: DocumentType,
    ocr_language: str,
    progress_callback: Optional[
        Callable[[int, int, str], None]
    ] = None,
) -> list[dict]:
    """
    OCR raster images embedded inside a DOCX file.

    The OCR results are appended after the normal Word
    text because exact image placement inside Word runs
    is not always reliable.
    """

    image_parts = []

    for relationship in (
        document.part.rels.values()
    ):
        target = getattr(
            relationship,
            "target_part",
            None,
        )

        content_type = getattr(
            target,
            "content_type",
            "",
        )

        if content_type.startswith("image/"):
            image_parts.append(target)

    results = []

    for index, image_part in enumerate(
        image_parts,
        start=1,
    ):
        if progress_callback:
            progress_callback(
                index,
                len(image_parts),
                f"Word image {index}",
            )

        try:
            image = Image.open(
                io.BytesIO(image_part.blob)
            )

            # Improve OCR reliability for transparent
            # or palette-based images.
            if image.mode not in {
                "RGB",
                "L",
            }:
                image = image.convert("RGB")

            text = pytesseract.image_to_string(
                image,
                lang=ocr_language,
                config="--psm 6",
            ).strip()

        except Exception:
            continue

        # Ignore decorative images and logos that
        # produce almost no meaningful text.
        if len(re.sub(r"\s+", "", text)) < 10:
            continue

        results.append({
            "number": index,
            "location": (
                f"Word embedded image {index}"
            ),
            "method": "Word image OCR",
            "text": text,
        })

    return results


def extract_docx(
    docx_path: str | Path,
    selection_range: str = "",
    ocr_language: str = "hin+eng",
    include_image_ocr: bool = True,
    progress_callback: Optional[
        Callable[[int, int, str], None]
    ] = None,
    skip_numbers: Optional[set] = None,
    on_result: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """
    Extract paragraphs and tables from a DOCX document.

    ``skip_numbers`` / ``on_result`` allow pause, stop and resume between blocks.
    """

    docx_path = Path(docx_path)
    document = Document(docx_path)

    blocks = []

    for block in iter_word_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            block_type = "Word paragraph"

        elif isinstance(block, Table):
            text = table_to_text(block).strip()
            block_type = "Word table"

        else:
            continue

        if text:
            blocks.append({
                "type": block_type,
                "text": text,
            })

    selected_indexes = parse_selection_range(
        selection_range,
        len(blocks),
    )

    results = []

    for position, block_index in enumerate(
        selected_indexes,
        start=1,
    ):
        block = blocks[block_index]

        if skip_numbers and (block_index + 1) in skip_numbers:
            continue

        if progress_callback:
            progress_callback(
                position,
                len(selected_indexes),
                f"Word block {block_index + 1}",
            )

        entry = {
            "number": block_index + 1,
            "location": (
                f"Word block {block_index + 1}"
            ),
            "method": block["type"],
            "text": block["text"],
        }
        results.append(entry)
        if on_result:
            on_result(entry)

    if include_image_ocr and not (skip_numbers and "images" in skip_numbers):
        image_results = extract_word_images(
            document=document,
            ocr_language=ocr_language,
            progress_callback=progress_callback,
        )

        results.extend(image_results)
        if on_result:
            for item in image_results:
                on_result(item)
            on_result({"number": "images", "location": "Word images done", "method": "marker", "text": ""})

    return results


def find_libreoffice() -> Optional[Path]:
    """
    Locate LibreOffice for legacy DOC conversion.
    """

    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        (
            r"C:\Program Files\LibreOffice"
            r"\program\soffice.exe"
        ),
        (
            r"C:\Program Files (x86)"
            r"\LibreOffice\program\soffice.exe"
        ),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)

        if path.exists():
            return path

    return None


def convert_doc_to_docx(
    doc_path: str | Path,
) -> Path:
    """
    Convert a legacy .doc file to .docx using
    LibreOffice in headless mode.
    """

    doc_path = Path(doc_path)
    libreoffice = find_libreoffice()

    if libreoffice is None:
        raise RuntimeError(
            "Legacy .doc files require LibreOffice.\n"
            "Install LibreOffice or save the document "
            "as .docx using Microsoft Word."
        )

    output_directory = Path(
        tempfile.mkdtemp(
            prefix="f5_word_conversion_"
        )
    )

    command = [
        str(libreoffice),
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        str(output_directory),
        str(doc_path),
    ]

    creation_flags = 0

    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
        )

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )

    converted_file = (
        output_directory
        / f"{doc_path.stem}.docx"
    )

    if (
        completed.returncode != 0
        or not converted_file.exists()
    ):
        raise RuntimeError(
            "LibreOffice could not convert the DOC file.\n\n"
            f"Output:\n{completed.stdout}\n\n"
            f"Error:\n{completed.stderr}"
        )

    return converted_file


def extract_document(
    document_path: str | Path,
    selection_range: str = "",
    ocr_language: str = "hin+eng",
    dpi: int = 300,
    include_word_image_ocr: bool = True,
    progress_callback: Optional[
        Callable[[int, int, str], None]
    ] = None,
    skip_numbers: Optional[set] = None,
    on_result: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """
    Main document-extraction dispatcher.
    """

    document_path = Path(document_path)
    extension = document_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported document type. "
            "Upload PDF, DOCX or DOC."
        )

    if extension == ".pdf":
        return extract_pdf(
            pdf_path=document_path,
            selection_range=selection_range,
            ocr_language=ocr_language,
            dpi=dpi,
            progress_callback=progress_callback,
            skip_numbers=skip_numbers,
            on_result=on_result,
        )

    if extension == ".doc":
        converted_path = convert_doc_to_docx(
            document_path
        )

        return extract_docx(
            docx_path=converted_path,
            selection_range=selection_range,
            ocr_language=ocr_language,
            include_image_ocr=(
                include_word_image_ocr
            ),
            progress_callback=progress_callback,
            skip_numbers=skip_numbers,
            on_result=on_result,
        )

    return extract_docx(
        docx_path=document_path,
        selection_range=selection_range,
        ocr_language=ocr_language,
        include_image_ocr=(
            include_word_image_ocr
        ),
        progress_callback=progress_callback,
        skip_numbers=skip_numbers,
        on_result=on_result,
    )