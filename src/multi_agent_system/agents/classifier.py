# src/multi_agent_system/classifier_agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List
import logging
import json

import tiktoken
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from ..config_loader import MultiAgentConfig
from ..mcp_vector_client import MCPVectorStoreClient

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PDF_ROOT = ROOT_DIR / "pdf_database"

_mcp_client_singleton = MCPVectorStoreClient()


def _truncate_by_tokens(text: str, max_tokens: int, encoding_name: str) -> str:
    if not text:
        return ""

    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        return ""

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    return enc.decode(tokens[:max_tokens])


@lru_cache(maxsize=1)
def get_available_areas() -> List[str]:
    areas = sorted(d.name for d in PDF_ROOT.iterdir() if d.is_dir())

    if not areas:
        logger.warning(
            "No subdirectories found under %s; "
            "falling back to default areas ['economy', 'med', 'tech'].",
            PDF_ROOT,
        )
        return ["error while loading areas from pdf database"]

    logger.info("Discovered areas from pdf_database: %s", areas)
    return areas


@dataclass
class ClassifierAgent:
    max_article_tokens: int = 6000
    mcp_query_tokens: int = 1500
    token_encoding: str = "cl100k_base"

    _cfg: MultiAgentConfig = field(default_factory=MultiAgentConfig, init=False)

    def _build_llm(self) -> ChatGroq:
        llm_cfg = self._cfg.get_llm_config()
        model = llm_cfg.get("model", "openai/gpt-oss-120b")
        temperature = float(llm_cfg.get("temperature", 0.0))

        logger.debug(
            "Building classifier LLM client with model=%s, temperature=%s",
            model,
            temperature,
        )

        return ChatGroq(model=model, temperature=temperature)

    @staticmethod
    def _normalize_area(raw: str, candidates: List[str]) -> str:
        raw = (raw or "").strip().lower()
        if not candidates:
            logger.warning("No candidate areas provided; defaulting to 'tech'.")
            return "tech"

        for c in candidates:
            if raw == c.lower():
                return c

        for c in candidates:
            if c.lower() in raw:
                return c

        if "econ" in raw:
            for c in candidates:
                if c.lower().startswith("econ"):
                    return c

        logger.warning(
            "Could not confidently map raw area '%s' to known areas %s; "
            "falling back to '%s'.",
            raw,
            candidates,
            candidates[0],
        )
        return candidates[0]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        article_text: str = state["article_text"]

        logger.info(
            "Classifier agent started. Article length=%d chars",
            len(article_text),
        )

        article_for_llm = _truncate_by_tokens(
            article_text,
            max_tokens=self.max_article_tokens,
            encoding_name=self.token_encoding,
        )

        query_for_mcp = _truncate_by_tokens(
            article_text,
            max_tokens=self.mcp_query_tokens,
            encoding_name=self.token_encoding,
        )

        candidates = get_available_areas()

        system_prompt = self._cfg.get_prompt("classifier")
        llm = self._build_llm()
        llm_with_tools = llm.bind_tools([_mcp_client_singleton.search_articles])

        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "You are classifying the scientific area of the paper.\n"
                    f"Valid areas (choose exactly one): {candidates}\n\n"
                    "You MAY call the tool `search_articles` to retrieve similar papers.\n"
                    "If you call it, call it at most once with top_k=5.\n"
                    "If you do NOT call it, that's fine—just answer with the area.\n\n"
                    "Output format: respond with ONLY the area name, exactly matching one of the valid areas.\n\n"
                    "Paper (token-truncated):\n"
                    f"{article_for_llm}\n\n"
                    "Snippet you can use as tool query (token-truncated):\n"
                    f"{query_for_mcp}\n"
                )
            ),
        ]

        logger.info("Sending classification request to LLM (tool-calling enabled).")
        first = llm_with_tools.invoke(messages)

        tool_calls = getattr(first, "tool_calls", None) or []

        # If the model chooses NOT to call tools, we accept its choice and use its content directly.
        if not tool_calls:
            raw_area = (first.content or "").strip()
            area = self._normalize_area(raw_area, candidates)

            logger.info(
                "Classifier returned without tool call. raw='%s' mapped='%s' (candidates=%s)",
                raw_area,
                area,
                candidates,
            )

            new_state = dict(state)
            new_state["area"] = area
            return new_state

        # If the model DOES call tools, we execute them and do a second pass.
        messages.append(first)

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or "search_articles_call"

            if name != "search_articles":
                logger.warning("Ignoring unexpected tool call name=%s", name)
                continue

            query = args.get("query", query_for_mcp)
            top_k = int(args.get("top_k", 5))

            logger.info("Executing tool call: search_articles(top_k=%d)", top_k)
            results = _mcp_client_singleton.search_articles({"query": query, "top_k": top_k})

            messages.append(
                ToolMessage(
                    content=json.dumps(results, ensure_ascii=False),
                    tool_call_id=call_id,
                )
            )

        logger.debug("Sending second pass to LLM to output ONLY the final area.")
        final = llm_with_tools.invoke(messages)
        raw_area = (final.content or "").strip()

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


_classifier_agent = ClassifierAgent()


def classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return _classifier_agent.run(state)
