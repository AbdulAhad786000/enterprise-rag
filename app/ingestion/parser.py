
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
    """Extract page text from a PDF and preserve page numbers."""

    import fitz

    blocks: list[ParsedBlock] = []
    with fitz.open(file_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            if not paragraphs:
                paragraphs = [text]
            for paragraph_index, paragraph in enumerate(paragraphs):
                blocks.append(ParsedBlock(
                    text=paragraph,
                    page_number=page_index + 1,
                    paragraph_index=paragraph_index,
                ))

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
