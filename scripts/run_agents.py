# scripts/run_agents.py
import logging
import sys

from src.pipeline.pipeline_runner import (
    ArticleSampleManager,
    ArticlePipelineRunner,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        logger.error(
            "Usage: python -m scripts.run_agents "
            "<path_or_url_to_article.(pdf|txt|md)>"
        )
        sys.exit(1)

    raw_source = sys.argv[1]

    sample_manager = ArticleSampleManager(samples_dir="samples")
    runner = ArticlePipelineRunner(sample_manager=sample_manager)

    try:
        runner.run_for_source(raw_source)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
