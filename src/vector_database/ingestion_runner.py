# src/vector_database/ingestion_runner.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.vector_database.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


class VectorIndexBuilder:
    """
    High-level orchestrator for building (or rebuilding) the Chroma vector index
    based on the configuration in configuration/base.yaml.

    Responsibilities:
      - Locate and load the YAML configuration.
      - Resolve paths for PDF root and Chroma DB.
      - Instantiate VectorDatabase with the configured parameters.
      - Trigger the index build process.
    """

    def __init__(self, root_dir: Path, config_path: Path | None = None) -> None:
        """
        Parameters
        ----------
        root_dir : Path
            Project root directory (e.g., the folder containing `configuration/`).
        config_path : Path, optional
            Optional explicit path to the YAML config. If not provided,
            defaults to `root_dir / "configuration" / "base.yaml"`.
        """
        self.root_dir = root_dir
        self.config_path = config_path or (self.root_dir / "configuration" / "base.yaml")

        logger.debug("VectorIndexBuilder initialized with root_dir=%s", self.root_dir)
        logger.debug("Using configuration file: %s", self.config_path)

        if not self.config_path.exists():
            logger.error("Configuration file not found: %s", self.config_path)
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and return the full YAML configuration.
        """
        logger.info("Loading configuration from: %s", self.config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Configuration loaded successfully.")
        return config

    def _resolve_paths(self) -> tuple[Path, Path]:
        """
        Resolve PDF root and Chroma DB paths from the configuration.

        Returns
        -------
        (pdf_root_path, chroma_path) : tuple[Path, Path]
        """
        paths_cfg = self._config.get("paths") or {}

        pdf_root_path = self.root_dir / paths_cfg.get("pdf_root", "pdf_database")
        chroma_path = self.root_dir / paths_cfg.get("chroma_path", "chroma_db")

        logger.info("PDF root path: %s", pdf_root_path)
        logger.info("Chroma DB path: %s", chroma_path)

        return pdf_root_path, chroma_path

    def _resolve_vector_db_params(self) -> Dict[str, Any]:
        """
        Extract VectorDatabase-related parameters from the configuration.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - embedding_model
            - collection_name
            - chunk_size
            - chunk_overlap
        """
        vdb_cfg = self._config.get("vector_db") or {}

        embedding_model = vdb_cfg.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        collection_name = vdb_cfg.get("collection_name", "articles")
        chunk_size = int(vdb_cfg.get("chunk_size", 1000))
        chunk_overlap = int(vdb_cfg.get("chunk_overlap", 200))

        logger.info(
            "Vector DB config -> model=%s | collection=%s | chunk_size=%d | chunk_overlap=%d",
            embedding_model,
            collection_name,
            chunk_size,
            chunk_overlap,
        )

        return {
            "embedding_model": embedding_model,
            "collection_name": collection_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def create_vector_db(self) -> VectorDatabase:
        """
        Instantiate and return a configured VectorDatabase.

        This method does NOT build the index by itself; it only prepares the instance.
        """
        pdf_root_path, chroma_path = self._resolve_paths()
        vdb_params = self._resolve_vector_db_params()

        logger.debug(
            "Creating VectorDatabase with pdf_root_path=%s, chroma_path=%s",
            pdf_root_path,
            chroma_path,
        )

        return VectorDatabase(
            pdf_root_path=pdf_root_path,
            chroma_path=chroma_path,
            embedding_model=vdb_params["embedding_model"],
            collection_name=vdb_params["collection_name"],
            chunk_size=vdb_params["chunk_size"],
            chunk_overlap=vdb_params["chunk_overlap"],
        )

    def build_index(self) -> None:
        """
        Build (or rebuild) the Chroma vector index using the current configuration.
        """
        logger.info("Starting vector index build process...")
        vector_db = self.create_vector_db()

        logger.info("Building vector index (this may take a while on first run)...")
        vector_db.build_index()
        logger.info("Vector index built successfully.")
