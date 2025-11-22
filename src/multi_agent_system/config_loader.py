# src/multi_agent_system/config_loader.py
from pathlib import Path
from typing import Any, Dict
import logging

import yaml


logger = logging.getLogger(__name__)


class MultiAgentConfig:
    """
    OO wrapper around the multi_agent section of configuration/base.yaml.

    Usage:
        cfg = MultiAgentConfig()  # loads once
        system_prompt = cfg.get_prompt("classifier")
        llm_cfg = cfg.get_llm_config()
    """

    def __init__(self, config_path: Path | None = None) -> None:
        # Discover project root: .../most
        root_dir = Path(__file__).resolve().parents[2]
        logger.debug("Resolved project root for MultiAgentConfig: %s", root_dir)

        # Default config path
        self.config_path = config_path or (root_dir / "configuration" / "base.yaml")
        logger.info("Loading multi-agent configuration from: %s", self.config_path)

        if not self.config_path.exists():
            logger.error("Configuration file not found: %s", self.config_path)
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        # Load the full config once
        with self.config_path.open("r", encoding="utf-8") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f) or {}

        logger.debug(
            "Full configuration loaded. Top-level keys: %s",
            list(self._config.keys()),
        )

        # Cache the multi_agent block
        self._multi_agent: Dict[str, Any] = (
            self._config.get("multi_agent", {}) or {}
        )

        logger.debug(
            "`multi_agent` block loaded. Keys: %s",
            list(self._multi_agent.keys()),
        )

    # Public API

    def get_multi_agent_config(self) -> Dict[str, Any]:
        """
        Returns the entire `multi_agent` config block.
        """
        logger.debug("get_multi_agent_config called.")
        return self._multi_agent

    def get_prompt(self, agent_name: str) -> str:
        """
        agent_name: 'classifier' | 'extractor' | 'reviewer'
        Returns the system prompt string for the given agent.
        """
        logger.info("Fetching system prompt for agent: %s", agent_name)

        prompts = self._multi_agent.get("prompts", {}) or {}
        agent_cfg = prompts.get(agent_name, {}) or {}
        system_prompt = agent_cfg.get("system")

        if not system_prompt:
            logger.error(
                "No system prompt configured for agent '%s' in %s",
                agent_name,
                self.config_path,
            )
            raise ValueError(
                f"No system prompt configured for agent '{agent_name}' "
                f"in {self.config_path}."
            )

        logger.debug(
            "System prompt for agent '%s' loaded (length=%d).",
            agent_name,
            len(system_prompt),
        )
        return system_prompt


    def get_llm_config(self) -> Dict[str, Any]:
        llm_cfg = self._multi_agent.get("llm", {}) or {}
        logger.info("Loaded LLM config: %s", llm_cfg)
        return llm_cfg
