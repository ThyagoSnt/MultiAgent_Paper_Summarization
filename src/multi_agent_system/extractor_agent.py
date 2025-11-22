# src/multi_agent_system/extractor_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import json
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .config_loader import MultiAgentConfig


logger = logging.getLogger(__name__)


@dataclass
class ExtractorAgent:
    """
    Object-oriented extractor agent.

    Responsibilities:
      - Build the LLM client using multi_agent.llm config.
      - Ask the LLM to fill the JSON schema.
      - Robustly parse and normalize the JSON returned by the LLM.

    Parameters
    ----------
    max_article_chars : int
        Maximum number of characters from the article to send to the LLM.
        This replaces the old hardcoded MAX_EXTRACTOR_CHARS constant.
    """

    max_article_chars: int = 6000

    def __post_init__(self) -> None:
        # Load configuration once
        self._cfg = MultiAgentConfig()


    # Internal helpers
    def _build_llm(self) -> ChatGroq:
        """
        Build a Groq LLM client based on the multi_agent.llm config.
        """
        llm_cfg = self._cfg.get_llm_config()
        model = llm_cfg.get("model", "openai/gpt-oss-120b")
        temperature = float(llm_cfg.get("temperature", 0.0))

        logger.debug(
            "Building extractor LLM client with model=%s, temperature=%s",
            model,
            temperature,
        )

        return ChatGroq(
            model=model,
            temperature=temperature,
        )

    def _extract_json_from_response(self, raw_content: str) -> Dict[str, Any]:
        """
        Try to parse a JSON object from the LLM response.
        Handles both raw JSON and ```json ... ``` fenced blocks.
        """
        logger.debug(
            "Attempting to parse JSON from LLM response (length=%d chars)",
            len(raw_content),
        )

        text = raw_content.strip()

        # Try to strip ```json ... ```
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    candidate = part[len("json"):].strip()
                    text = candidate
                    break
                if part.startswith("{"):
                    text = part
                    break

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not an object.")
            logger.debug("Successfully parsed JSON object from LLM response.")
            return parsed
        except Exception as e:
            logger.exception("Failed to parse JSON from LLM response.")
            raise ValueError(
                f"Failed to parse JSON from LLM response: {e}\nRaw: {raw_content}"
            )

    def _normalize_extraction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the extraction dict has exactly the expected keys and types.
        """
        expected_problem_key = "what problem does the artcle propose to solve?"
        expected_steps_key = "step by step on how to solve it"
        expected_conclusion_key = "conclusion"

        logger.debug(
            "Normalizing extraction payload. Keys present: %s",
            list(payload.keys()),
        )

        extraction: Dict[str, Any] = {
            expected_problem_key: "",
            expected_steps_key: [],
            expected_conclusion_key: "",
        }

        if isinstance(payload.get(expected_problem_key), str):
            extraction[expected_problem_key] = payload[expected_problem_key]

        steps = payload.get(expected_steps_key)
        if isinstance(steps, list):
            extraction[expected_steps_key] = [
                str(item) for item in steps if isinstance(item, (str, int, float))
            ]

        if isinstance(payload.get(expected_conclusion_key), str):
            extraction[expected_conclusion_key] = payload[expected_conclusion_key]

        logger.debug(
            "Normalized extraction: problem_len=%d, steps=%d, conclusion_len=%d",
            len(extraction[expected_problem_key]),
            len(extraction[expected_steps_key]),
            len(extraction[expected_conclusion_key]),
        )

        return extraction

    # Public API
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core logic for the extractor agent.

        Receives:
          - state["article_text"]: full article text
          - optional state["area"]: area classified by the classifier agent

        Produces:
          - state["extraction"]: normalized JSON with problem, steps and conclusion
        """
        article_text: str = state["article_text"]
        area: str | None = state.get("area")

        logger.info(
            "Extractor agent started. Area=%s, article_length=%d chars",
            area,
            len(article_text),
        )

        # Truncate to keep the request within token limits
        truncated_text = article_text[: self.max_article_chars]
        if len(article_text) > self.max_article_chars:
            logger.debug(
                "Article text truncated from %d to %d chars for extraction.",
                len(article_text),
                len(truncated_text),
            )

        system_prompt = self._cfg.get_prompt("extractor")
        llm = self._build_llm()

        user_instructions = [
            "You must read the following article text and fill the JSON schema exactly.",
            "Return ONLY the JSON object, nothing else.",
            f"The article content is truncated to the first {self.max_article_chars} characters.",
        ]
        if area:
            user_instructions.append(
                f"The article was classified in the area: {area}."
            )

        human_content = (
            "\n".join(user_instructions)
            + "\n\n--- ARTICLE TEXT START ---\n"
            + truncated_text
            + "\n--- ARTICLE TEXT END ---\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]

        logger.debug("Sending extraction request to LLM.")
        response = llm.invoke(messages)
        raw_content = response.content

        logger.debug(
            "Received extraction response from LLM (length=%d chars).",
            len(raw_content),
        )

        parsed = self._extract_json_from_response(raw_content)
        extraction = self._normalize_extraction(parsed)

        new_state = dict(state)
        new_state["extraction"] = extraction

        logger.info("Extractor agent finished. Extraction object populated.")
        return new_state


# LangGraph-compatible wrapper
# Global instance used by LangGraph node wrapper
_extractor_agent = ExtractorAgent()


def extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible wrapper that delegates to the ExtractorAgent instance.

    This keeps the existing function-based API used in graph.py while
    allowing the implementation to be fully object-oriented.
    """
    return _extractor_agent.run(state)
