# src/mcp_server/server.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys
import logging

import yaml
from fastmcp import FastMCP

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vector_database.vector_database import VectorDatabase
from src.mcp_server.schemas import (
    SearchArticlesResponse,
    ArticleSummary,
    ArticleContent,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT_DIR / "configuration" / "base.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config: Dict[str, Any] = yaml.safe_load(f) or {}

mcp_cfg: Dict[str, Any] = config.get("mcp", {}) or {}
paths_cfg: Dict[str, Any] = config.get("paths", {}) or {}
vdb_cfg: Dict[str, Any] = config.get("vector_db", {}) or {}

MCP_NAME: str = mcp_cfg.get("name", "ArticleVectorStore")
MCP_HOST: str = mcp_cfg.get("host", "0.0.0.0")
MCP_PORT: int = int(mcp_cfg.get("port", 8000))
MCP_PATH: str = mcp_cfg.get("path", "/mcp")

pdf_root_cfg: str = paths_cfg.get("pdf_root", "pdf_database")
chroma_path_cfg: str = paths_cfg.get("chroma_path", "chroma_db")

PDF_ROOT = ROOT_DIR / pdf_root_cfg
CHROMA_PATH = ROOT_DIR / chroma_path_cfg

EMBEDDING_MODEL: str = vdb_cfg.get(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
)
COLLECTION_NAME: str = vdb_cfg.get("collection_name", "articles")
CHUNK_SIZE: int = int(vdb_cfg.get("chunk_size", 1000))
CHUNK_OVERLAP: int = int(vdb_cfg.get("chunk_overlap", 200))

vector_db = VectorDatabase(
    pdf_root_path=PDF_ROOT,
    chroma_path=CHROMA_PATH,
    embedding_model=EMBEDDING_MODEL,
    collection_name=COLLECTION_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

mcp = FastMCP(name=MCP_NAME)


@mcp.tool
def search_articles(query: str, top_k: int = 5) -> SearchArticlesResponse:
    logger.info(
        "search_articles called | query_length=%d | top_k=%d",
        len(query),
        top_k,
    )

    try:
        raw_results = vector_db.search_articles(query=query, top_k=top_k)
    except Exception as e:
        logger.exception("Failed to search articles.")
        raise RuntimeError(str(e)) from e

    summaries = [ArticleSummary(**item) for item in raw_results]
    return SearchArticlesResponse(results=summaries)


@mcp.tool
def get_article_content(article_id: str) -> ArticleContent:
    logger.info("get_article_content called | article_id=%s", article_id)

    try:
        article = vector_db.get_article_content(article_id=article_id)
    except KeyError as e:
        logger.warning("Article not found: %s", article_id)
        raise ValueError("Article not found") from e
    except Exception as e:
        logger.exception("Failed to retrieve article content.")
        raise RuntimeError(str(e)) from e

    return ArticleContent(**article)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting FastMCP server '%s' via HTTP | host=%s port=%d path=%s",
        MCP_NAME,
        MCP_HOST,
        MCP_PORT,
        MCP_PATH,
    )
    logger.info("PDF root: %s | Chroma path: %s", PDF_ROOT, CHROMA_PATH)

    mcp.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        path=MCP_PATH,
    )
