
from pathlib import Path

from app.ingestion.loader import discover_documents
from app.ingestion.metadata import extract_document_metadata, extract_docx_metadata
from app.ingestion.parser import parse_docx, parse_markdown, parse_pdf
from app.ingestion.schemas import ParsedDocument


def ingest_docx_directory(raw_dir: Path) -> list[ParsedDocument]:
    """Discover, extract metadata, and parse all DOCX documents."""

    files = discover_documents(raw_dir)

    docx_files = [
        file_path
        for file_path in files
        if file_path.suffix.lower() == ".docx"
    ]

    parsed_documents: list[ParsedDocument] = []

    for file_path in docx_files:
        metadata = extract_docx_metadata(file_path)
        parsed_document = parse_docx(file_path, metadata)

        parsed_documents.append(parsed_document)

    return parsed_documents


def ingest_directory(raw_dir: Path) -> list[ParsedDocument]:
    """Ingest every supported PDF, DOCX, and Markdown file in a directory."""

    parsed_documents: list[ParsedDocument] = []
    for file_path in discover_documents(raw_dir):
        metadata = extract_document_metadata(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            parsed = parse_docx(file_path, metadata)
        elif suffix == ".pdf":
            parsed = parse_pdf(file_path, metadata)
        elif suffix == ".md":
            parsed = parse_markdown(file_path, metadata)
        else:
            continue
        parsed_documents.append(parsed)
    return parsed_documents
