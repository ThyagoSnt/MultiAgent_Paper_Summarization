# tests/test_graph_pipeline.py
import os
from pathlib import Path

import pytest

from src.pdf_parser.pdf_parser import PdfTextExtractor
from src.multi_agent_system.multi_agent_graph import run_pipeline


# Skip this whole module if there is no GROQ_API_KEY (LLM integration test)
if not os.getenv("GROQ_API_KEY"):
    pytest.skip(
        "GROQ_API_KEY not set; skipping integration tests that call the LLM.",
        allow_module_level=True,
    )


def _get_sample_pdf() -> Path:
    """
    Return a sample PDF to run through the pipeline.
    Prefer samples/input_article_1.pdf if it exists,
    otherwise fall back to one of the tech PDFs.
    """
    root_dir = Path(__file__).resolve().parents[1]
    candidates = [
        root_dir / "samples" / "input_article_1.pdf",
        root_dir / "pdf_database" / "tech" / "tech_1.pdf",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("No suitable sample PDF found for pipeline test.")


@pytest.mark.integration
def test_run_pipeline_on_sample_pdf():
    """
    Full pipeline test: classify -> extract JSON -> review.
    Ensures the output structure is consistent.
    """
    pdf_path = _get_sample_pdf()
    article_text = PdfTextExtractor.extract(pdf_path)

    result = run_pipeline(article_text)

    # Check basic structure
    assert isinstance(result, dict)
    assert "area" in result
    assert "extraction" in result
    assert "review" in result

    # Area should be a non-empty string
    area = result["area"]
    assert isinstance(area, str)
    assert len(area.strip()) > 0

    # Extraction JSON should have the expected keys
    extraction = result["extraction"]
    assert isinstance(extraction, dict)
    expected_keys = {
        "what problem does the artcle propose to solve?",
        "step by step on how to solve it",
        "conclusion",
    }
    assert expected_keys.issubset(extraction.keys())

    # Review should be non-empty markdown text
    review = result["review"]
    assert isinstance(review, str)
    assert len(review.strip()) > 0
