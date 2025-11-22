# src/mcp_server/schemas.py
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class SearchArticlesRequest(BaseModel):
    """Payload sent by clients when requesting a similarity search."""
    query: str = Field(..., description="Natural language query string.")
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of distinct articles to return.",
    )


class ArticleSummary(BaseModel):
    """Summary information for an article in the vector store."""
    id: str
    title: str
    area: str
    score: float


class SearchArticlesResponse(BaseModel):
    """Response for the /search_articles endpoint."""
    results: List[ArticleSummary]


class GetArticleContentRequest(BaseModel):
    """Payload sent by clients when requesting the full content of an article."""
    article_id: str


class ArticleContent(BaseModel):
    """Full article content as stored in the vector store."""
    id: str
    title: str
    area: str
    content: str
