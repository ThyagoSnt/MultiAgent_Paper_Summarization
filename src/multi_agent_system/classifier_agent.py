# src/multi_agent_system/classifier_agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .config_loader import MultiAgentConfig
from .mcp_vector_client import MCPVectorStoreClient

logger = logging.getLogger(__name__)

# Project root and PDF root discovery (used to infer available areas)
ROOT_DIR = Path(__file__).resolve().parents[2]
PDF_ROOT = ROOT_DIR / "pdf_database"


@lru_cache(maxsize=1)
def get_available_areas() -> List[str]:
    """
    Discover available areas dynamically by listing subfolders of pdf_database/.

    Example:
        pdf_database/
          economy/
          med/
          tech/

    If the directory is missing or empty, we fall back to a default list.
    """
    if not PDF_ROOT.exists():
        logger.warning(
            "PDF root directory %s does not exist; "
            "falling back to default areas ['economy', 'med', 'tech'].",
            PDF_ROOT,
        )
        return ["economy", "med", "tech"]

    areas = sorted(
        d.name
        for d in PDF_ROOT.iterdir()
        if d.is_dir()
    )

    if not areas:
        logger.warning(
            "No subdirectories found under %s; "
            "falling back to default areas ['economy', 'med', 'tech'].",
            PDF_ROOT,
        )
        return ["economy", "med", "tech"]

    logger.info("Discovered areas from pdf_database: %s", areas)
    return areas


@dataclass
class ClassifierAgent:
    """
    Object-oriented classifier agent.

    Responsibilities:
      - Call MCP vector store to get similar articles.
      - Ask the LLM to choose an area.
      - Normalize the raw LLM output into one of the known areas
        discovered from pdf_database/ subfolders.

    Parameters
    ----------
    max_article_chars : int
        Maximum number of characters from the article text sent to the LLM
        for classification.

    mcp_query_chars : int
        Maximum number of characters used as a query snippet to the MCP
        vector-store search.
    """

    max_article_chars: int = 4000
    mcp_query_chars: int = 800

    _cfg: MultiAgentConfig = field(default_factory=MultiAgentConfig, init=False)
    _mcp_client: MCPVectorStoreClient = field(
        default_factory=MCPVectorStoreClient,
        init=False,
    )

    # Internal helpers
    def _build_llm(self) -> ChatGroq:
        llm_cfg = self._cfg.get_llm_config()
        model = llm_cfg.get("model", "openai/gpt-oss-120b")
        temperature = float(llm_cfg.get("temperature", 0.0))

        logger.debug(
            "Building classifier LLM client with model=%s, temperature=%s",
            model,
            temperature,
        )

        return ChatGroq(
            model=model,
            temperature=temperature,
        )

    @staticmethod
    def _normalize_area(raw: str, candidates: List[str]) -> str:
        """
        Map the raw LLM output to one of the known candidate areas.

        Strategy:
          1) Exact match.
          2) Substring match (candidate in raw).
          3) Handle common synonym 'econ' -> any area starting with 'econ'.
          4) Fallback: first candidate (deterministic).
        """
        raw = (raw or "").strip().lower()
        if not candidates:
            logger.warning("No candidate areas provided; defaulting to 'tech'.")
            return "tech"

        # 1) Exact match
        for c in candidates:
            if raw == c.lower():
                return c

        # 2) Substring match
        for c in candidates:
            if c.lower() in raw:
                return c

        # 3) Special handling for 'econ' style abbreviations
        if "econ" in raw:
            for c in candidates:
                if c.lower().startswith("econ"):
                    return c

        # 4) Fallback deterministic choice
        logger.warning(
            "Could not confidently map raw area '%s' to known areas %s; "
            "falling back to '%s'.",
            raw,
            candidates,
            candidates[0],
        )
        return candidates[0]

    # Public API
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core classification logic.

        Receives:
          - state["article_text"]: full article text

        Produces:
          - state["area"]: one of the discovered areas from pdf_database/
        """
        article_text: str = state["article_text"]

        logger.info(
            "Classifier agent started. Article length=%d chars",
            len(article_text),
        )

        # 1) Truncate text for the LLM
        truncated_text = article_text[: self.max_article_chars]
        if len(article_text) > self.max_article_chars:
            logger.debug(
                "Article text truncated from %d to %d chars for classifier.",
                len(article_text),
                len(truncated_text),
            )

        # 2) Query MCP vector store with a smaller snippet
        query_snippet = article_text[: self.mcp_query_chars]
        logger.info(
            "Invoking MCP search_articles with query length=%d, top_k=5",
            len(query_snippet),
        )
        similar_articles = self._mcp_client.search_articles(
            query=query_snippet,
            top_k=5,
        )

        # Build compact textual context from MCP results
        context_lines = []
        for art in similar_articles:
            context_lines.append(
                f"- id={art['id']} | area={art['area']} | "
                f"title={art['title']} | score={art['score']:.3f}"
            )
        context_text = "\n".join(context_lines)

        logger.debug("MCP search_articles returned %d results.", len(similar_articles))
        for art in similar_articles:
            logger.debug("  MCP hit: %s", art)

        # 3) Call LLM with article + MCP context
        system_prompt = self._cfg.get_prompt("classifier")
        llm = self._build_llm()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Article to classify (truncated):\n\n"
                    f"{truncated_text}\n\n"
                    "Similar articles retrieved from the MCP vector store "
                    "via the `search_articles` tool:\n"
                    f"{context_text}\n\n"
                    "Based on the article and the retrieved context, respond with "
                    "ONLY the area name, exactly matching one of the known areas."
                )
            ),
        ]

        logger.debug("Sending classification request to LLM.")
        response = llm.invoke(messages)
        raw_area = response.content.strip()

        candidates = get_available_areas()
        area = self._normalize_area(raw_area, candidates)

        logger.info(
            "Classifier LLM raw output='%s' mapped to area='%s' (candidates=%s)",
            raw_area,
            area,
            candidates,
        )

        new_state = dict(state)
        new_state["area"] = area
        return new_state


# LangGraph-compatible wrapper
_classifier_agent = ClassifierAgent()

def classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: classify the article into one of the areas defined by
    the folders inside pdf_database/ (e.g., economy, med, tech).

    This is a thin wrapper over the ClassifierAgent.run(...) method
    to keep compatibility with graph.py.
    """
    return _classifier_agent.run(state)
