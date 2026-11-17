
from pathlib import Path

from docx import Document

from app.ingestion.schemas import DocumentMetadata


REQUIRED_FIELDS = {
    "document id": "document_id",
    "department": "department",
    "version": "version",
    "effective date": "effective_date",
}


def extract_docx_metadata(file_path: Path) -> DocumentMetadata:
    """Extract and validate metadata from a DOCX document."""

    document = Document(file_path)

    extracted: dict[str, str] = {}

    for table in document.tables:
        for row in table.rows:
            if len(row.cells) < 2:
                continue

            key = row.cells[0].text.strip().lower().rstrip(":")
            value = row.cells[1].text.strip()

            if key in REQUIRED_FIELDS:
                field_name = REQUIRED_FIELDS[key]
                extracted[field_name] = value

    missing_fields = [
        field_name
        for field_name in REQUIRED_FIELDS.values()
        if not extracted.get(field_name)
    ]

    if missing_fields:
        raise ValueError(
            f"Missing metadata in {file_path.name}: {missing_fields}"
        )

    title = file_path.stem.replace("_", " ")

    return DocumentMetadata(
        document_id=extracted["document_id"],
        title=title,
        department=extracted["department"],
        version=extracted["version"],
        effective_date=extracted["effective_date"],
        source_file=file_path.name,
        file_type=file_path.suffix.lower(),
    )

_METADATA_LABELS = {
    "document id": "document_id",
    "document_id": "document_id",
    "id": "document_id",
    "title": "title",
    "department": "department",
    "version": "version",
    "effective date": "effective_date",
    "effective_date": "effective_date",
}


def extract_file_metadata(file_path: Path) -> DocumentMetadata:
    """Read basic metadata from PDF/Markdown text, with safe stable defaults."""

    suffix = file_path.suffix.lower()
    if suffix == ".md":
        content = file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        import fitz

        with fitz.open(file_path) as pdf:
            content = "\n".join(page.get_text("text") for page in pdf)
            pdf_title = (pdf.metadata or {}).get("title", "").strip()
    else:
        raise ValueError(f"Unsupported metadata source type: {suffix}")

    values: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip().lstrip("#*- ")
        if ":" not in stripped:
            continue
        label, value = stripped.split(":", maxsplit=1)
        field_name = _METADATA_LABELS.get(label.strip().casefold())
        if field_name and value.strip():
            values[field_name] = value.strip().strip('"\'')

    title = values.get("title")
    if not title and suffix == ".md":
        title = next(
            (line.lstrip("# ").strip() for line in content.splitlines()
             if line.startswith("#") and line.lstrip("# ").strip()),
            None,
        )
    if not title:
        title = pdf_title if suffix == ".pdf" and pdf_title else file_path.stem.replace("_", " ")

    document_id = values.get("document_id") or file_path.stem
    return DocumentMetadata(
        document_id=document_id,
        title=title,
        department=values.get("department", "Unknown"),
        version=values.get("version", "Unknown"),
        effective_date=values.get("effective_date", "Unknown"),
        source_file=file_path.name,
        file_type=suffix,
    )


def extract_document_metadata(file_path: Path) -> DocumentMetadata:
    """Dispatch metadata extraction according to the file type."""

    if file_path.suffix.lower() == ".docx":
        return extract_docx_metadata(file_path)
    return extract_file_metadata(file_path)

