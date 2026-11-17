
from pathlib import Path
import re

from docx import Document

from app.ingestion.schemas import (
    DocumentMetadata,
    ParsedBlock,
    ParsedDocument,
)


def parse_docx(
    file_path: Path,
    metadata: DocumentMetadata,
) -> ParsedDocument:
    """Parse ordered DOCX paragraphs and tables into traceable text blocks."""

    document = Document(file_path)
    blocks: list[ParsedBlock] = []
    section_stack: list[str] = []
    metadata_labels = {"document id", "department", "version", "effective date"}

    for block_index, item in enumerate(document.iter_inner_content()):
        if hasattr(item, "style") and hasattr(item, "text"):
            text = item.text.strip()
            if not text:
                continue
            style_name = item.style.name if item.style else ""
            if style_name.startswith("Heading"):
                try:
                    level = int(style_name.split()[-1])
                except ValueError:
                    level = 1
                section_stack = section_stack[: level - 1]
                section_stack.append(text)
            blocks.append(ParsedBlock(
                text=text,
                paragraph_index=block_index,
                section_path=section_stack.copy(),
            ))
            continue

        rows = [
            [cell.text.strip().replace("\n", " ") for cell in row.cells]
            for row in item.rows
        ]
        rows = [row for row in rows if any(row)]
        if not rows:
            continue
        if any(row[0].casefold().rstrip(":") in metadata_labels for row in rows):
            continue

        header = rows[0]
        table_lines = []
        if len(rows) > 1:
            for row in rows[1:]:
                fields = [
                    f"{header[i]}: {value}" if i < len(header) and header[i] else value
                    for i, value in enumerate(row)
                    if value
                ]
                if fields:
                    table_lines.append(" | ".join(fields))
        else:
            table_lines = [" | ".join(header)]

        if table_lines:
            blocks.append(ParsedBlock(
                text="Table: " + "\n".join(table_lines),
                paragraph_index=block_index,
                section_path=section_stack.copy(),
            ))

    return ParsedDocument(metadata=metadata, blocks=blocks)


def parse_pdf(file_path: Path, metadata: DocumentMetadata) -> ParsedDocument:
    """Extract PDF text and infer section headings from span typography."""

    import fitz
    import statistics

    blocks: list[ParsedBlock] = []
    section_stack: list[str] = []

    with fitz.open(file_path) as pdf:
        pages = [page.get_text("dict") for page in pdf]
        font_sizes = [
            span["size"]
            for page_data in pages
            for block in page_data.get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        body_size = statistics.median(font_sizes) if font_sizes else 12.0

        def flush_body(lines: list[str], page_number: int) -> None:
            text = " ".join(line.strip() for line in lines).strip()
            if text:
                blocks.append(ParsedBlock(
                    text=text,
                    page_number=page_number,
                    paragraph_index=len(blocks),
                    section_path=section_stack.copy(),
                ))
            lines.clear()

        for page_index, page_data in enumerate(pages):
            page_number = page_index + 1
            for text_block in page_data.get("blocks", []):
                if text_block.get("type") != 0:
                    continue
                body_lines: list[str] = []
                for line in text_block.get("lines", []):
                    spans = [
                        span for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ]
                    if not spans:
                        continue
                    text = re.sub(
                        r"\s+", " ",
                        "".join(span["text"] for span in spans),
                    ).strip()
                    if not text:
                        continue

                    font_size = max(span.get("size", body_size) for span in spans)
                    character_count = sum(len(span["text"].strip()) for span in spans)
                    bold_count = sum(
                        len(span["text"].strip())
                        for span in spans
                        if span.get("flags", 0) & (1 << 4)
                    )
                    mostly_bold = character_count > 0 and bold_count / character_count >= 0.7
                    short_heading = len(text) <= 120 and not re.search(r"[.!?]$", text)
                    is_heading = short_heading and (
                        font_size > body_size * 1.15
                        or (mostly_bold and font_size >= body_size * 0.95)
                    )

                    if is_heading:
                        flush_body(body_lines, page_number)
                        if font_size > body_size * 1.4:
                            level = 1
                        elif font_size > body_size * 1.15:
                            level = 2
                        else:
                            level = 3
                        section_stack[:] = section_stack[:level - 1]
                        section_stack.append(text)
                        blocks.append(ParsedBlock(
                            text=text,
                            page_number=page_number,
                            paragraph_index=len(blocks),
                            section_path=section_stack.copy(),
                        ))
                    else:
                        body_lines.append(text)
                flush_body(body_lines, page_number)

    return ParsedDocument(metadata=metadata, blocks=blocks)

def parse_markdown(file_path: Path, metadata: DocumentMetadata) -> ParsedDocument:
    """Parse Markdown headings, paragraphs, lists, and table rows."""

    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1:]

    blocks: list[ParsedBlock] = []
    section_stack: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            blocks.append(ParsedBlock(
                text=" ".join(line.strip() for line in paragraph_lines).strip(),
                paragraph_index=len(blocks),
                section_path=section_stack.copy(),
            ))
            paragraph_lines.clear()

    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            section_stack = section_stack[: level - 1]
            section_stack.append(heading.group(2).strip())
        elif line.strip():
            paragraph_lines.append(line)
        else:
            flush_paragraph()
    flush_paragraph()
    return ParsedDocument(metadata=metadata, blocks=blocks)
