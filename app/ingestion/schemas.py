from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    title: str
    department: str
    version: str
    effective_date: str
    source_file: str
    file_type: str


@dataclass
class ParsedBlock:
    text: str
    page_number: int | None = None
    paragraph_index: int | None = None
    section_path: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    metadata: DocumentMetadata
    blocks: list[ParsedBlock] = field(default_factory=list)


@dataclass
class ParentChunk:
    parent_id: str
    document_id: str
    text: str
    section_path: list[str]
    metadata: DocumentMetadata
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    text: str
    section_path: list[str]
    metadata: DocumentMetadata
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
