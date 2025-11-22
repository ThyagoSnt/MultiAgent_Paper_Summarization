# src/multi_agent_system/reviewer_agent.py
from __future__ import annotations

from typing import Dict, Any
import json
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .config_loader import MultiAgentConfig

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """
    Object-oriented wrapper around the reviewer LLM agent.

    This class is responsible for:
    - Loading the multi-agent configuration.
    - Building and holding a Groq LLM client.
    - Generating a Portuguese markdown review from the pipeline state.
    """

    def __init__(
        self,
        max_article_chars: int = 4000,
        config: MultiAgentConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_article_chars : int
            Maximum number of article characters to send to the LLM.
            This is used to truncate very long articles to avoid
            exceeding token limits.
        config : MultiAgentConfig | None
            Optional pre-loaded configuration object. If None, a new
            MultiAgentConfig will be created.
        """
        self._max_article_chars = max_article_chars
        self._cfg = config or MultiAgentConfig()
        self._llm = self._build_llm()

        logger.info(
            "ReviewerAgent initialized with max_article_chars=%d",
            self._max_article_chars,
        )

    def _build_llm(self) -> ChatGroq:
        """
        Build a Groq LLM client based on the multi_agent.llm config.
        """
        llm_cfg = self._cfg.get_llm_config()
        model = llm_cfg.get("model", "openai/gpt-oss-120b")
        temperature = float(llm_cfg.get("temperature", 0.0))

        logger.debug(
            "Building reviewer LLM client with model=%s, temperature=%s",
            model,
            temperature,
        )

        return ChatGroq(
            model=model,
            temperature=temperature,
        )

    def _build_human_message(
        self,
        area: str | None,
        extraction: Dict[str, Any],
        article_text: str,
    ) -> str:
        """
        Build the human message content sent to the LLM.
        """
        extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)

        # Truncate article text for safety
        truncated_article = article_text[: self._max_article_chars]
        if len(article_text) > self._max_article_chars:
            logger.debug(
                "Article text truncated from %d to %d chars for reviewer.",
                len(article_text),
                len(truncated_article),
            )

        return (
            "Use the information below to write a critical review in Brazilian Portuguese.\n\n"
            f"Area (chosen by a classifier among ['economy', 'med', 'tech']): {area}\n\n"
            "Extraction JSON:\n"
            f"{extraction_json}\n\n"
            "You may optionally use the article text to refine your review, "
            "but the JSON is the main structured summary.\n\n"
            "--- ARTICLE TEXT START (optional) ---\n"
            f"{truncated_article}\n"
            "--- ARTICLE TEXT END ---\n"
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reviewer step of the pipeline.

        It reads 'area', 'extraction' and 'article_text' from the state and
        writes a Portuguese markdown review into 'review'.
        """
        area: str | None = state.get("area")
        extraction: Dict[str, Any] = state.get("extraction", {}) or {}
        article_text: str = state.get("article_text", "")

        logger.info(
            "ReviewerAgent.run started. Area=%s, extraction_keys=%s, article_length=%d",
            area,
            list(extraction.keys()),
            len(article_text),
        )

        system_prompt = self._cfg.get_prompt("reviewer")
        human_content = self._build_human_message(area, extraction, article_text)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        logger.debug("Sending review generation request to LLM.")
        response = self._llm.invoke(messages)
        review_md = response.content.strip()

        logger.info(
            "ReviewerAgent.run finished. Review length=%d chars.",
            len(review_md),
        )

        new_state = dict(state)
        new_state["review"] = review_md
        return new_state


# Global instance used by LangGraph node wrapper
_reviewer_agent = ReviewerAgent()

def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible wrapper that delegates to the ReviewerAgent instance.

    This keeps the existing function-based API used in graph.py while
    allowing the implementation to be fully object-oriented.
    """
    return _reviewer_agent.run(state)
