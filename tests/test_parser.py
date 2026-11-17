import unittest
from pathlib import Path

from docx import Document

from app.ingestion.loader import discover_documents
from app.ingestion.parser import parse_docx
from app.ingestion.schemas import DocumentMetadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def metadata_for(path: Path) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=path.stem.lower(),
        title=path.stem.replace("_", " "),
        department="test department",
        version="test version",
        source_file=str(path),
        file_type=path.suffix.lstrip(".").lower(),
    )


class DocxParserTests(unittest.TestCase):
    def test_all_known_documents_parse_with_text_and_metadata(self) -> None:
        paths = discover_documents(RAW_DIR)
        self.assertEqual(len(paths), 5)

        for path in paths:
            with self.subTest(document=path.name):
                metadata = metadata_for(path)
                parsed = parse_docx(path, metadata)

                self.assertIs(parsed.metadata, metadata)
                self.assertTrue(parsed.blocks)
                self.assertTrue(all(block.text.strip() for block in parsed.blocks))

    def test_paragraph_indexes_refer_to_original_docx_positions(self) -> None:
        for path in discover_documents(RAW_DIR):
            with self.subTest(document=path.name):
                source = Document(path)
                expected_indexes = [
                    index
                    for index, paragraph in enumerate(source.paragraphs)
                    if paragraph.text.strip()
                ]
                parsed = parse_docx(path, metadata_for(path))
                actual_indexes = [block.paragraph_index for block in parsed.blocks]

                self.assertEqual(actual_indexes, expected_indexes)

    def test_heading_is_attached_to_its_section_context(self) -> None:
        path = RAW_DIR / "HR_Employee_Handbook.docx"
        parsed = parse_docx(path, metadata_for(path))
        heading = next(
            block for block in parsed.blocks if block.text == "3. Annual Leave Policy"
        )

        self.assertEqual(heading.section_path, ["3. Annual Leave Policy"])

        heading_position = parsed.blocks.index(heading)
        following_block = parsed.blocks[heading_position + 1]
        self.assertEqual(following_block.section_path, heading.section_path)


if __name__ == "__main__":
    unittest.main()
