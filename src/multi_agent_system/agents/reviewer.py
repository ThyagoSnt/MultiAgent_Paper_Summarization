# src/multi_agent_system/reviewer_agent.py
from __future__ import annotations

from typing import Dict, Any, List
import json
import logging

import tiktoken
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from ..config_loader import MultiAgentConfig

logger = logging.getLogger(__name__)


class ReviewerAgent:
    def __init__(
        self,
        max_article_tokens: int = 6000,
        token_encoding: str = "cl100k_base",
        config: MultiAgentConfig | None = None,
    ) -> None:
        self._max_article_tokens = int(max_article_tokens)
        self._token_encoding = token_encoding
        self._cfg = config or MultiAgentConfig()
        self._llm = self._build_llm()

        logger.info(
            "ReviewerAgent initialized with max_article_tokens=%d, token_encoding=%s",
            self._max_article_tokens,
            self._token_encoding,
        )

    def _build_llm(self) -> ChatGroq:
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

    def _truncate_by_tokens(self, text: str) -> str:
        if not text:
            return ""

        if self._max_article_tokens <= 0:
            return ""

        enc = tiktoken.get_encoding(self._token_encoding)
        tokens: List[int] = enc.encode(text)

        if len(tokens) <= self._max_article_tokens:
            return text

        truncated = enc.decode(tokens[: self._max_article_tokens])
        logger.debug(
            "Article text truncated by tokens from %d to %d tokens for reviewer.",
            len(tokens),
            self._max_article_tokens,
        )
        return truncated

    def _build_human_message(
        self,
        area: str | None,
        extraction: Dict[str, Any],
        article_text: str,
    ) -> str:
        extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
        truncated_article = self._truncate_by_tokens(article_text)

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


_reviewer_agent = ReviewerAgent()


def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return _reviewer_agent.run(state)
