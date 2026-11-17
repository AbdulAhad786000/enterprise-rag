from app.ingestion.schemas import ChildChunk


def build_embedding_text(child: ChildChunk) -> str:
    """Format a self-contained, metadata-enriched document for embedding."""

    metadata = child.metadata
    context = [
        f"document ID: {metadata.document_id}",
        f"department: {metadata.department}",
        f"version: {metadata.version}",
        f"effective date: {metadata.effective_date}",
    ]
    if child.section_path:
        context.append(f"section: {' > '.join(child.section_path)}")
    if child.page_start is not None:
        page_range = (
            str(child.page_start)
            if child.page_start == child.page_end
            else f"{child.page_start}-{child.page_end}"
        )
        context.append(f"pages: {page_range}")
    context.append(f"content: {child.text.strip()}")
    return f"title: {metadata.title} | text: {' | '.join(context)}"
