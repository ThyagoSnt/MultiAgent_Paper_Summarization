# src/multi_agent_system/mcp_vector_client.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml
from pydantic import ValidationError

from src.mcp_server.schemas import (
    SearchArticlesRequest,
    SearchArticlesResponse,
    GetArticleContentRequest,
    ArticleContent,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configuration" / "base.yaml"


# Load MCP HTTP configuration
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    _config: Dict[str, Any] = yaml.safe_load(f) or {}

_mcp_cfg: Dict[str, Any] = _config.get("mcp", {}) or {}

_MCP_HOST: str = _mcp_cfg.get("host", "127.0.0.1")
_MCP_PORT: int = int(_mcp_cfg.get("port", 8000))
_DEFAULT_BASE_URL: str = _mcp_cfg.get(
    "base_url",
    f"http://{_MCP_HOST}:{_MCP_PORT}",
)


@dataclass
class MCPVectorStoreClient:
    """
    Thin HTTP client for the external vector-store server.

    This client does **not** start the server process.
    You are expected to run the server separately (e.g. `make mcp`
    inside the Docker container), and the client will talk to it
    over HTTP.

    Public API:
      - search_articles(query, top_k)
      - get_article_content(article_id)
    """

    base_url: str = _DEFAULT_BASE_URL

    # Internal helper to perform POST requests and handle basic errors
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        logger.info("HTTP POST %s", url)
        logger.debug("Payload: %s", payload)

        try:
            response = requests.post(url, json=payload, timeout=30)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("HTTP request to %s failed.", url)
            raise RuntimeError(f"Failed to call MCP HTTP server at {url}: {e}") from e

        if not response.ok:
            logger.error(
                "HTTP %s from %s: %s",
                response.status_code,
                url,
                response.text,
            )
            raise RuntimeError(
                f"HTTP {response.status_code} calling {url}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as e:  # pragma: no cover - defensive
            logger.exception("Failed to decode JSON from %s.", url)
            raise RuntimeError(f"Invalid JSON response from {url}: {e}") from e

        logger.debug("Response JSON from %s: %s", url, data)
        return data

    # Public API used by agents
    def search_articles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Call the HTTP `/search_articles` endpoint and return a list of
        plain dicts:

            { "id": str, "title": str, "area": str, "score": float }

        Pydantic is used both to validate the outgoing request and
        the incoming response.
        """
        logger.info(
            "Invoking HTTP search_articles with top_k=%d and query length=%d.",
            top_k,
            len(query),
        )

        # Validate outbound payload with Pydantic
        req_model = SearchArticlesRequest(query=query, top_k=top_k)

        # Perform HTTP request
        raw_json = self._post("/search_articles", req_model.model_dump())

        # Validate inbound response with Pydantic
        try:
            resp_model = SearchArticlesResponse.model_validate(raw_json)
        except ValidationError as e:
            logger.exception("Failed to validate SearchArticlesResponse.")
            raise RuntimeError(
                f"Invalid response schema from MCP HTTP server: {e}"
            ) from e

        # Convert back to a list of plain dicts to keep the rest of the
        # codebase unchanged (ClassifierAgent etc.).
        results = [article.model_dump() for article in resp_model.results]
        logger.debug("search_articles returned %d results.", len(results))
        return results

    def get_article_content(self, article_id: str) -> Dict[str, Any]:
        """
        Call the HTTP `/get_article_content` endpoint and return a single
        dict:

            { "id": str, "title": str, "area": str, "content": str }

        Pydantic is used both on the request and the response.
        """
        logger.info(
            "Invoking HTTP get_article_content for article_id=%s.",
            article_id,
        )

        # Validate outbound payload
        req_model = GetArticleContentRequest(article_id=article_id)

        # Perform HTTP request
        raw_json = self._post("/get_article_content", req_model.model_dump())

        # Validate inbound response
        try:
            article = ArticleContent.model_validate(raw_json)
        except ValidationError as e:
            logger.exception("Failed to validate ArticleContent response.")
            raise RuntimeError(
                f"Invalid response schema from MCP HTTP server: {e}"
            ) from e

        article_dict = article.model_dump()
        logger.debug(
            "get_article_content returned article with id=%s, area=%s.",
            article_dict.get("id"),
            article_dict.get("area"),
        )
        return article_dict
