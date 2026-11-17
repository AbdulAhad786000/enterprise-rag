from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md"}


def discover_documents(raw_dir: Path) -> list[Path]:
    """
    Find supported documents inside the raw data directory.

    Returns:
        A sorted list of paths to PDF, DOCX, and Markdown files.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {raw_dir}")

    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {raw_dir}")

    documents = [
        file_path
        for file_path in raw_dir.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not file_path.name.startswith("~$")
        )
    ]

    return sorted(documents, key=lambda path: path.name.lower())