from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

import yaml
from fastmcp import Client
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

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    _config: Dict[str, Any] = yaml.safe_load(f) or {}

_mcp_cfg: Dict[str, Any] = _config.get("mcp", {}) or {}
_MCP_HOST: str = _mcp_cfg.get("host", "127.0.0.1")
_MCP_PORT: int = int(_mcp_cfg.get("port", 8000))
_MCP_PATH: str = _mcp_cfg.get("path", "/mcp")

_DEFAULT_MCP_URL: str = _mcp_cfg.get(
    "url",
    f"http://{_MCP_HOST}:{_MCP_PORT}{_MCP_PATH}",
)


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "An asyncio event loop is already running. "
        "Use the async methods or an async context manager."
    )


def _to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_plain(v) for k, v in vars(obj).items()}
        except Exception:
            pass
    return obj


def _validate_model(model_cls, raw: Any):
    try:
        return model_cls.model_validate(raw)
    except ValidationError:
        return model_cls.model_validate(raw, from_attributes=True)


@dataclass
class MCPVectorStoreClient:
    url: str = _DEFAULT_MCP_URL
    _client: Optional[Client] = field(default=None, init=False, repr=False)

    def _make_client(self) -> Client:
        return Client(self.url)

    async def __aenter__(self) -> "MCPVectorStoreClient":
        if self._client is None:
            self._client = self._make_client()
            await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if self._client is None:
            async with self._make_client() as client:
                result = await client.call_tool(tool_name, args)
                data = getattr(result, "data", result)
                return _to_plain(data)

        result = await self._client.call_tool(tool_name, args)
        data = getattr(result, "data", result)
        return _to_plain(data)

    @tool("search_articles")
    def search_articles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        MCP tool to the /search_articles endpoint.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts: {id, title, area, score}.
        """
        return _run_sync(self.search_articles_async(query=query, top_k=top_k))


    async def search_articles_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info("Calling MCP search_articles | top_k=%d | query_length=%d", top_k, len(query))

        req = SearchArticlesRequest(query=query, top_k=top_k)
        raw_data = await self._call_tool("search_articles", req.model_dump())

        try:
            resp = _validate_model(SearchArticlesResponse, raw_data)
        except ValidationError as e:
            logger.exception("Invalid SearchArticlesResponse from MCP.")
            raise RuntimeError(
                f"Invalid response schema from MCP server: {e} | raw_type={type(raw_data).__name__}"
            ) from e

        return [article.model_dump() for article in resp.results]

    @tool("get_article_content")
    def get_article_content(self, article_id: str) -> Dict[str, Any]:
        """
        MCP tool to the /get_article_content endpoint.

        Args:
            article_id: Id of the article to search his full content.

        Returns:
            Dict of article content.
        """
        return _run_sync(self.get_article_content_async(article_id=article_id))

    async def get_article_content_async(self, article_id: str) -> Dict[str, Any]:
        logger.info("Calling MCP get_article_content | article_id=%s", article_id)

        req = GetArticleContentRequest(article_id=article_id)
        raw_data = await self._call_tool("get_article_content", req.model_dump())

        try:
            article = _validate_model(ArticleContent, raw_data)
        except ValidationError as e:
            logger.exception("Invalid ArticleContent from MCP.")
            raise RuntimeError(
                f"Invalid response schema from MCP server: {e} | raw_type={type(raw_data).__name__}"
            ) from e

        return article.model_dump()
