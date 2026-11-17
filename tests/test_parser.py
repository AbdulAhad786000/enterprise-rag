import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from docx import Document

from app.ingestion.loader import discover_documents
from app.ingestion.parser import parse_docx, parse_pdf
from app.ingestion.schemas import DocumentMetadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def metadata_for(path: Path) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=path.stem.lower(),
        title=path.stem.replace("_", " "),
        department="test department",
        version="test version",
        effective_date="2025-01-01",
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
                    for index, item in enumerate(source.iter_inner_content())
                    if hasattr(item, "style") and getattr(item, "text", "").strip()
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


    def test_pdf_typography_creates_section_paths_and_keeps_pages(self) -> None:
        import pymupdf

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "1. Leave Policy", fontsize=16)
            page.insert_text((72, 100), "Full-time employees receive 20 days.", fontsize=10)
            page.insert_text((72, 140), "Eligibility", fontsize=10, fontname="hebo")
            page.insert_text((72, 165), "Part-time terms follow the agreement.", fontsize=10)
            pdf.save(path)
            pdf.close()

            parsed = parse_pdf(path, metadata_for(path))

        heading = next(block for block in parsed.blocks if block.text == "1. Leave Policy")
        first_body = next(block for block in parsed.blocks if "receive 20 days" in block.text)
        subheading = next(block for block in parsed.blocks if block.text == "Eligibility")
        final_body = next(block for block in parsed.blocks if "follow the agreement" in block.text)

        self.assertEqual(heading.page_number, 1)
        self.assertEqual(first_body.section_path, ["1. Leave Policy"])
        self.assertEqual(subheading.section_path, ["1. Leave Policy", "Eligibility"])
        self.assertEqual(final_body.section_path, subheading.section_path)


if __name__ == "__main__":
    unittest.main()
