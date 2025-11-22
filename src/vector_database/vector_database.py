import logging
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.pdf_parser.pdf_parser import PdfTextExtractor

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Simple wrapper around ChromaDB + SentenceTransformers for
    building and querying an article vector store.
    """

    def __init__(
        self,
        pdf_root_path: Path,
        chroma_path: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "articles",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.pdf_root_path = pdf_root_path
        self.chroma_path = chroma_path

        # Configurable parameters
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Lazily initialized components
        self._model = None
        self._client = None
        self._collection = None

    # Internal helpers
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Very simple sliding-window chunking.

        :param text: Raw text to be split into chunks.
        :param chunk_size: Maximum number of characters per chunk.
        :param overlap: Number of characters that overlap between consecutive chunks.
        :return: List of text chunks.
        """
        if not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Move the window forward, keeping an overlap
            start += chunk_size - overlap

        return chunks

    def _ensure_model(self) -> None:
        """
        Lazily load the embedding model.
        """
        if self._model is None:
            logger.info("Loading embedding model: %s ...", self.embedding_model)
            self._model = SentenceTransformer(self.embedding_model)

    def _ensure_collection(self) -> None:
        """
        Lazily create/get the Chroma collection.
        """
        if self._client is None or self._collection is None:
            logger.info("Initializing Chroma at %s ...", self.chroma_path)
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )

    # Index building
    def build_index(self) -> None:
        """
        Build (or rebuild) the vector index from the PDFs.

        It walks through the pdf_root_path, expecting a structure like:

            pdf_root_path/
              economy/
                eco_1.pdf
                ...
              med/
                med_1.pdf
                ...
              tech/
                tech_1.pdf
                ...

        For each PDF it:
        - Extracts full text.
        - Splits into overlapping chunks.
        - Stores each chunk with metadata including area, article_id, title and chunk_index.
        """
        self._ensure_model()
        self._ensure_collection()

        all_ids: List[str] = []
        all_texts: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []

        logger.info("Reading PDFs from %s ...", self.pdf_root_path)
        if not self.pdf_root_path.exists():
            logger.error("PDF root path does not exist: %s", self.pdf_root_path)
            return

        for area_dir in sorted(self.pdf_root_path.iterdir()):
            if not area_dir.is_dir():
                continue

            area = area_dir.name  # e.g. "tech", "med", "economy"
            logger.info("Processing area: %s", area)

            for pdf_path in sorted(area_dir.glob("*.pdf")):
                logger.info("  [PDF] %s", pdf_path)

                try:
                    text = PdfTextExtractor.extract(pdf_path)
                except Exception as e:
                    # This will catch "no extractable text" and any other extraction error
                    logger.error(
                        "  [ERROR] Failed to extract text from %s: %s",
                        pdf_path,
                        e,
                    )
                    continue

                chunks = self.chunk_text(
                    text,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )

                if not chunks:
                    logger.warning(
                        "  [WARN] No chunks generated from %s, skipping.",
                        pdf_path,
                    )
                    continue

                # A logical article identifier based on area + file stem
                # Example: "tech_tech_1"
                article_id = f"{area}_{pdf_path.stem}"

                # Simple title derived from filename; can be improved later
                title = pdf_path.stem.replace("_", " ")

                for idx, chunk in enumerate(chunks):
                    # Unique chunk id, including the chunk index
                    doc_id = f"{article_id}_{idx}"
                    all_ids.append(doc_id)
                    all_texts.append(chunk)
                    all_metadatas.append(
                        {
                            "area": area,
                            "source_pdf": pdf_path.name,
                            "chunk_index": idx,
                            "article_id": article_id,
                            "title": title,
                        }
                    )

        if not all_texts:
            logger.warning("No chunks to index. Please check your PDFs.")
            return

        logger.info("Generating embeddings for %d chunks...", len(all_texts))
        embeddings = self._model.encode(all_texts, show_progress_bar=True).tolist()

        logger.info("Writing to '%s' collection...", self.collection_name)
        self._collection.add(
            ids=all_ids,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )

        logger.info("Chroma index built successfully with %d documents.", len(all_ids))

    # Public API for MCP tools
    def search_articles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search over the collection and return aggregated results
        at the article level (not chunk level).

        The MCP tool can expose this with the expected contract:
        - input: { "query": string }
        - output: list of objects { "id", "title", "area", "score" }

        :param query: Natural language query string.
        :param top_k: Maximum number of distinct articles to return.
        :return: List of article descriptors sorted by descending score.
        """
        if not query.strip():
            logger.error("Query must not be empty.")
            raise ValueError("Query must not be empty.")

        self._ensure_model()
        self._ensure_collection()

        logger.debug("Running vector search for query: %s", query)
        # Embed the query with the same model used to build the index
        query_embedding = self._model.encode([query]).tolist()

        # Ask Chroma for similar chunks; we request more than top_k chunks
        # and then aggregate them by article_id.
        raw = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 3,
            include=["metadatas", "distances"],
        )

        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        if not metadatas:
            logger.info("No results returned from vector search.")
            return []

        # Transform distances into scores (smaller distance -> higher score).
        # Here we use a simple heuristic: score = 1 / (1 + distance).
        scored_chunks: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas, distances):
            score = 1.0 / (1.0 + float(dist))
            scored_chunks.append(
                {
                    "article_id": meta["article_id"],
                    "title": meta.get("title") or meta.get("source_pdf"),
                    "area": meta["area"],
                    "score": score,
                }
            )

        # Aggregate by article_id: keep the best score per article.
        by_article: Dict[str, Dict[str, Any]] = {}
        for chunk in scored_chunks:
            aid = chunk["article_id"]
            current = by_article.get(aid)
            if current is None or chunk["score"] > current["score"]:
                by_article[aid] = {
                    "id": aid,
                    "title": chunk["title"],
                    "area": chunk["area"],
                    "score": chunk["score"],
                }

        # Sort by score (descending) and limit to top_k.
        results = sorted(by_article.values(), key=lambda x: x["score"], reverse=True)
        logger.debug("Vector search returned %d aggregated articles.", len(results))
        return results[:top_k]

    def get_article_content(self, article_id: str) -> Dict[str, Any]:
        """
        Retrieve the full content (concatenated chunks) for a given article_id.

        The MCP tool can expose this with the expected contract:
        - input: { "id": string }
        - output: { "id", "title", "area", "content" }

        :param article_id: Logical article identifier (area + filename stem).
        :return: Dict with article id, title, area and concatenated content.
        """
        self._ensure_collection()

        logger.debug("Fetching article content for id=%s", article_id)
        res = self._collection.get(
            where={"article_id": article_id},
            include=["documents", "metadatas"],
        )

        documents = res.get("documents", [])
        metadatas = res.get("metadatas", [])

        if not documents:
            logger.error("Article '%s' not found in vector store.", article_id)
            raise ValueError(f"Article '{article_id}' not found in vector store.")

        # Each entry is (document_text, metadata)
        combined = sorted(
            zip(documents, metadatas),
            key=lambda x: x[1].get("chunk_index", 0),
        )

        full_text = "\n".join(doc for doc, _ in combined)
        first_meta = combined[0][1]

        return {
            "id": article_id,
            "title": first_meta.get("title") or first_meta.get("source_pdf"),
            "area": first_meta["area"],
            "content": full_text,
        }
