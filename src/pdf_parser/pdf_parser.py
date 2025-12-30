# src/pdf_parser/pdf_parser.py
import logging
from pathlib import Path
from typing import Union, Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)

try:
    # Optional OCR backend (Docling)
    from docling.document_converter import DocumentConverter

    _DOCLING_AVAILABLE = True
except ImportError:
    DocumentConverter = None
    _DOCLING_AVAILABLE = False


class PdfTextExtractor:

    @staticmethod
    def extract(
        pdf_path: Union[str, Path],
        enable_ocr: bool = True,
        ocr_max_chars: int | None = None,
    ) -> str:
        """
        Extract text from a PDF.

        The method follows a two-stage strategy:
          1. Use PyPDF's text extraction for each page.
          2. If no text is found and `enable_ocr=True`, attempt OCR using Docling.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.
        enable_ocr : bool, optional
            Whether to attempt OCR with Docling if no text is found using PyPDF.
        ocr_max_chars : int | None, optional
            Optional safety cap on OCR output length. If None, uses
            `PdfTextExtractor.DEFAULT_OCR_MAX_CHARS`.

        Returns
        -------
        str
            The extracted text (possibly from OCR).
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        logger.info("Starting PDF text extraction for file: %s", pdf_path)

        # Try standard text extraction via PyPDF
        full_text = PdfTextExtractor._extract_with_pypdf(pdf_path)

        if full_text:
            logger.info(
                "PyPDF extraction succeeded for %s (length=%d chars).",
                pdf_path,
                len(full_text),
            )
            return full_text

        logger.warning(
            "PyPDF extraction returned no text for %s. "
            "This is likely a scanned PDF without embedded text.",
            pdf_path,
        )

        # Optional OCR fallback via Docling
        if enable_ocr:
            if not _DOCLING_AVAILABLE:
                logger.error(
                    "OCR fallback requested, but 'docling' is not installed. "
                    "Run `pip install docling easyocr` to enable OCR."
                )
            else:
                logger.info(
                    "Attempting OCR extraction via Docling for file: %s",
                    pdf_path,
                )
                ocr_text = PdfTextExtractor._extract_with_docling(
                    pdf_path=pdf_path,
                    max_chars=ocr_max_chars,
                )
                if ocr_text:
                    logger.info(
                        "Docling OCR extraction succeeded for %s (length=%d chars).",
                        pdf_path,
                        len(ocr_text),
                    )
                    return ocr_text

                logger.warning(
                    "Docling OCR extraction did not produce any text for %s.",
                    pdf_path,
                )

        # If we reach here, nothing worked
        raise ValueError(
            "No extractable text found in PDF (either not OCR'ed, or OCR disabled / failed): "
            f"{pdf_path}"
        )


    @staticmethod
    def _extract_with_pypdf(pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        all_text_parts: list[str] = []

        for page_index, page in enumerate(reader.pages):
            try:
                page_text: Optional[str] = page.extract_text()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Could not extract text from page %s of %s: %s",
                    page_index,
                    pdf_path,
                    e,
                )
                page_text = ""

            if page_text:
                all_text_parts.append(page_text)

        full_text = "\n\n".join(all_text_parts).strip()
        return full_text

    @staticmethod
    def _extract_with_docling(
        pdf_path: Path,
        max_chars: int,
    ) -> str:
        if not _DOCLING_AVAILABLE or DocumentConverter is None:
            logger.error(
                "Docling is not available but _extract_with_docling was called."
            )
            return ""

        try:
            converter = DocumentConverter()
            # Docling accepts either a local path or a URL as the `source`.
            result = converter.convert(str(pdf_path))

            text = result.document.export_to_text()
            if not isinstance(text, str):
                logger.warning(
                    "Docling export_to_text() returned non-string type: %s",
                    type(text),
                )
                return ""

            text = text.strip()
            if not text:
                return ""

            if max_chars is not None and len(text) > max_chars:
                logger.info(
                    "Docling OCR output truncated from %d to %d chars for file %s.",
                    len(text),
                    max_chars,
                    pdf_path,
                )
                text = text[:max_chars]

            return text

        except Exception as e:  # pragma: no cover - defensive guard
            logger.exception(
                "Unexpected error during Docling OCR extraction for %s: %s",
                pdf_path,
                e,
            )
            return ""