from typing import Any, Dict, TypedDict
import logging

from langgraph.graph import StateGraph, END

from .classifier_agent import classifier_node
from .extractor_agent import extractor_node
from .reviewer_agent import reviewer_node


logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph pipeline.
    """
    article_text: str
    area: str
    extraction: Dict[str, Any]
    review: str


def build_graph():
    """
    Build and compile the multi-agent pipeline graph.
    Flow:
        classifier -> extractor -> reviewer -> END
    """
    logger.info("Building LangGraph pipeline: classifier -> extractor -> reviewer -> END")

    graph = StateGraph(PipelineState)

    # Register nodes
    logger.debug("Adding node: classifier")
    graph.add_node("classifier", classifier_node)

    logger.debug("Adding node: extractor")
    graph.add_node("extractor", extractor_node)

    logger.debug("Adding node: reviewer")
    graph.add_node("reviewer", reviewer_node)

    # Define edges
    logger.debug("Setting entry point: classifier")
    graph.set_entry_point("classifier")

    logger.debug("Adding edge: classifier -> extractor")
    graph.add_edge("classifier", "extractor")

    logger.debug("Adding edge: extractor -> reviewer")
    graph.add_edge("extractor", "reviewer")

    logger.debug("Adding edge: reviewer -> END")
    graph.add_edge("reviewer", END)

    compiled = graph.compile()
    logger.info("LangGraph pipeline compiled successfully.")
    return compiled


def run_pipeline(article_text: str) -> Dict[str, Any]:
    """
    Convenience helper to run the full pipeline from raw article text.
    Returns a dict with:
        {
          "area": str,
          "extraction": {...},
          "review": str,
        }
    """
    logger.info(
        "Running multi-agent pipeline for article (length=%d chars).",
        len(article_text),
    )

    workflow = build_graph()

    initial_state: PipelineState = {
        "article_text": article_text,
    }

    logger.debug("Invoking LangGraph workflow with initial state keys: %s", list(initial_state.keys()))
    final_state: PipelineState = workflow.invoke(initial_state)

    logger.debug(
        "Pipeline finished. Final state keys: %s",
        list(final_state.keys()),
    )

    result = {
        "area": final_state.get("area"),
        "extraction": final_state.get("extraction"),
        "review": final_state.get("review"),
    }

    logger.info(
        "Pipeline result summary: area=%s, has_extraction=%s, review_length=%s",
        result["area"],
        result["extraction"] is not None,
        len(result["review"]) if result["review"] else 0,
    )

    return result
