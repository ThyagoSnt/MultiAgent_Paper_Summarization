# scripts/database_ingestion.py
import logging
from pathlib import Path

from src.vector_database.ingestion_runner import VectorIndexBuilder

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting database ingestion (vector index build)...")

    # Discover project root (folder above scripts/)
    root_dir = Path(__file__).resolve().parents[1]
    logger.debug("Resolved project root directory: %s", root_dir)

    builder = VectorIndexBuilder(root_dir=root_dir)
    builder.build_index()


if __name__ == "__main__":
    # Basic logging setup for this script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
