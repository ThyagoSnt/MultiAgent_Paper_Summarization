# src/mcp_server/server.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys
import logging

import yaml
from fastapi import FastAPI, HTTPException
import uvicorn

# Make sure project root is on sys.path so imports like `src.vector_database` work.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vector_database.vector_database import VectorDatabase  # noqa: E402
from src.mcp_server.schemas import (  # noqa: E402
    SearchArticlesRequest,
    SearchArticlesResponse,
    ArticleSummary,
    GetArticleContentRequest,
    ArticleContent,
)

logger = logging.getLogger(__name__)

# Configuration loading
CONFIG_PATH = ROOT_DIR / "configuration" / "base.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config: Dict[str, Any] = yaml.safe_load(f) or {}

mcp_cfg: Dict[str, Any] = config.get("mcp", {}) or {}
paths_cfg: Dict[str, Any] = config.get("paths", {}) or {}
vdb_cfg: Dict[str, Any] = config.get("vector_db", {}) or {}

MCP_NAME: str = mcp_cfg.get("name", "ArticleVectorStore")
MCP_TRANSPORT: str = mcp_cfg.get("transport", "http")  # kept for logging only
MCP_HOST: str = mcp_cfg.get("host", "0.0.0.0")
MCP_PORT: int = int(mcp_cfg.get("port", 8000))

# Paths
pdf_root_cfg: str = paths_cfg.get("pdf_root", "pdf_database")
chroma_path_cfg: str = paths_cfg.get("chroma_path", "chroma_db")

PDF_ROOT = ROOT_DIR / pdf_root_cfg
CHROMA_PATH = ROOT_DIR / chroma_path_cfg

# Vector DB settings (kept in sync with database_ingestion.py)
EMBEDDING_MODEL: str = vdb_cfg.get(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
)
COLLECTION_NAME: str = vdb_cfg.get("collection_name", "articles")
CHUNK_SIZE: int = int(vdb_cfg.get("chunk_size", 1000))
CHUNK_OVERLAP: int = int(vdb_cfg.get("chunk_overlap", 200))

# Vector database instance
# Instantiate VectorDatabase pointing to the existing Chroma index.
# Note: VectorDatabase will *not* rebuild the index here, only open it.
vector_db = VectorDatabase(
    pdf_root_path=PDF_ROOT,
    chroma_path=CHROMA_PATH,
    embedding_model=EMBEDDING_MODEL,
    collection_name=COLLECTION_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# FastAPI application
app = FastAPI(
    title=MCP_NAME,
    description=(
        "HTTP API exposing search_articles and get_article_content "
        "over the local Chroma-based vector store."
    ),
    version="1.0.0",
)


@app.post("/search_articles", response_model=SearchArticlesResponse)
def search_articles_endpoint(payload: SearchArticlesRequest) -> SearchArticlesResponse:
    """
    HTTP endpoint mirroring the old MCP `search_articles` tool.

    Clients send:
        {
          "query": "...",
          "top_k": 5
        }

    and receive:
        {
          "results": [
            {"id": "...", "title": "...", "area": "...", "score": 0.99},
            ...
          ]
        }
    """
    logger.info(
        "HTTP /search_articles called: query_length=%d, top_k=%d",
        len(payload.query),
        payload.top_k,
    )

    try:
        raw_results = vector_db.search_articles(
            query=payload.query,
            top_k=payload.top_k,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Error while searching articles.")
        raise HTTPException(status_code=500, detail=str(e))

    summaries = [ArticleSummary(**item) for item in raw_results]
    return SearchArticlesResponse(results=summaries)


@app.post("/get_article_content", response_model=ArticleContent)
def get_article_content_endpoint(payload: GetArticleContentRequest) -> ArticleContent:
    """
    HTTP endpoint mirroring the old MCP `get_article_content` tool.

    Clients send:
        { "article_id": "med_med_1" }

    and receive:
        {
          "id": "...",
          "title": "...",
          "area": "...",
          "content": "full text..."
        }
    """
    logger.info(
        "HTTP /get_article_content called: article_id=%s",
        payload.article_id,
    )

    try:
        article = vector_db.get_article_content(article_id=payload.article_id)
    except KeyError:
        logger.warning("Article not found: %s", payload.article_id)
        raise HTTPException(status_code=404, detail="Article not found")
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Error while fetching article content.")
        raise HTTPException(status_code=500, detail=str(e))

    return ArticleContent(**article)


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting HTTP server '%s' transport=%s host=%s port=%d pdf_root=%s chroma_path=%s",
        MCP_NAME,
        MCP_TRANSPORT,
        MCP_HOST,
        MCP_PORT,
        PDF_ROOT,
        CHROMA_PATH,
    )

    uvicorn.run(
        app,
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="info",
    )
