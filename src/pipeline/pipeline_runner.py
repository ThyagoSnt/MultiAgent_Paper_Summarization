# src/multi_agent_system/pipeline_runner.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from shutil import copy2
from typing import Dict, Any
from urllib.request import urlretrieve

from src.pdf_parser.pdf_parser import PdfTextExtractor
from src.multi_agent_system.multi_agent_graph import run_pipeline

logger = logging.getLogger(__name__)


class ArticleSampleManager:
    """
    Responsible for:
    - Managing the `samples/` directory.
    - Assigning incremental sample indices (1, 2, 3, ...).
    - Normalizing input sources (local paths or URLs) into files
      stored under `samples/`, such as:
        - samples/input_article_{N}.pdf
        - samples/input_article_{N}.txt
    """

    OUTPUT_PATTERN = re.compile(r"output_(\d+)\.json$")

    def __init__(self, samples_dir: Path | str = "samples") -> None:
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(exist_ok=True)

    def get_next_sample_index(self) -> int:
        """
        Inspect existing `output_N.json` files and return the next available N.
        """
        indices = []
        for path in self.samples_dir.glob("output_*.json"):
            match = self.OUTPUT_PATTERN.match(path.name)
            if match:
                indices.append(int(match.group(1)))

        if not indices:
            return 1
        return max(indices) + 1

    def _download_from_url(self, url: str, idx: int) -> Path:
        """
        Download a PDF from the given URL into:
            samples/input_article_{idx}.pdf
        """
        target_path = self.samples_dir / f"input_article_{idx}.pdf"
        logger.info("Downloading article from URL: %s -> %s", url, target_path)

        try:
            urlretrieve(url, target_path)
        except Exception as e:
            logger.error("Failed to download URL %s: %s", url, e)
            raise

        if not target_path.exists():
            raise FileNotFoundError(f"Downloaded file not found: {target_path}")

        return target_path

    def _copy_local_file(self, src: Path, idx: int) -> Path:
        """
        Copy a local file into the samples directory, normalizing the name to:
            samples/input_article_{idx}{ext}
        """
        if not src.exists():
            logger.error("Input file not found: %s", src)
            raise FileNotFoundError(f"Input file not found: {src}")

        ext = src.suffix or ".pdf"
        target_path = self.samples_dir / f"input_article_{idx}{ext}"

        if src.resolve() == target_path.resolve():
            logger.info("Input file already in samples/, skipping copy.")
            return target_path

        logger.info("Copying local article to samples: %s -> %s", src, target_path)
        try:
            copy2(src, target_path)
        except Exception as e:
            logger.error("Failed to copy input file %s to %s: %s", src, target_path, e)
            raise

        return target_path

    def resolve_input_source(self, raw_source: str) -> tuple[int, Path]:
        """
        Given a raw CLI argument (path or URL), resolve it to a normalized path
        inside `samples/`, and return:

            (sample_index, normalized_path)

        The sample_index will be used consistently for:
          - input_article_{index}.*
          - review_{index}.md
          - output_{index}.json
        """
        idx = self.get_next_sample_index()
        logger.info("Using sample index: %d", idx)

        # URL case
        if raw_source.startswith("http://") or raw_source.startswith("https://"):
            normalized_path = self._download_from_url(raw_source, idx)
        else:
            # Local file case
            normalized_path = self._copy_local_file(Path(raw_source), idx)

        return idx, normalized_path

    def get_review_path(self, idx: int) -> Path:
        return self.samples_dir / f"review_{idx}.md"

    def get_output_json_path(self, idx: int) -> Path:
        return self.samples_dir / f"output_{idx}.json"


class ArticlePipelineRunner:
    """
    High-level orchestrator for:
    - Loading article text (PDF, TXT, MD).
    - Running the multi-agent LangGraph pipeline.
    - Writing JSON + Markdown outputs.

    It delegates sample index management and input normalization to
    `ArticleSampleManager`.
    """

    def __init__(self, sample_manager: ArticleSampleManager) -> None:
        self.sample_manager = sample_manager

    @staticmethod
    def _load_article_text(input_path: Path) -> str:
        """
        Read article text from a PDF or plain text file.
        """
        suffix = input_path.suffix.lower()

        if suffix == ".pdf":
            return PdfTextExtractor.extract(input_path, enable_ocr=True)
        elif suffix in {".txt", ".md"}:
            return input_path.read_text(encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                "Please provide a .pdf, .txt or .md file."
            )

    def run_for_source(self, raw_source: str) -> Dict[str, Any]:
        """
        Main entry point for the pipeline, given either:
        - a local file path (PDF/TXT/MD), or
        - a remote URL (PDF)

        Returns a dict with metadata about the run:
        {
          "index": int,
          "area": str | None,
          "input_path": Path,
          "review_path": Path,
          "output_json_path": Path,
        }
        """
        # 1) Normalize input and assign index
        idx, normalized_path = self.sample_manager.resolve_input_source(raw_source)

        # 2) Load text
        logger.info("Reading article from normalized path: %s", normalized_path)
        article_text = self._load_article_text(normalized_path)

        # 3) Run multi-agent pipeline
        logger.info("Running multi-agent pipeline...")
        result = run_pipeline(article_text)

        area = result.get("area")
        extraction = result.get("extraction", {})
        review = result.get("review", "")

        # 4) Persist outputs
        review_path = self.sample_manager.get_review_path(idx)
        with review_path.open("w", encoding="utf-8") as f:
            f.write(review)

        final_payload = {
            "area": area,
            "extraction": extraction,
            "review_markdown": review,
        }

        output_json_path = self.sample_manager.get_output_json_path(idx)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)

        # 5) Log and return metadata
        logger.info("Area classification: %s", area)
        logger.info("Input article stored at: %s", normalized_path)
        logger.info("Review saved to: %s", review_path)
        logger.info("Final JSON saved to: %s", output_json_path)

        return {
            "index": idx,
            "area": area,
            "input_path": normalized_path,
            "review_path": review_path,
            "output_json_path": output_json_path,
        }
