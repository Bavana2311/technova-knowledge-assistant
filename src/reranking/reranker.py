import yaml
from typing import List, Tuple

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CrossEncoderReranker:

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)

        model_name = self.config["reranking"]["cross_encoder_model"]

        self.model = CrossEncoder(model_name)

        self.max_chars = int(
            self.config["reranking"].get("max_chars", 1200)
        )

    def _clip(self, text: str) -> str:
        """Limit document length before reranking."""
        if self.max_chars and len(text) > self.max_chars:
            return text[:self.max_chars]

        return text

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:

        if not docs:
            return []

        # Create query-document pairs
        pairs = [
            (query, self._clip(doc.page_content))
            for doc in docs
        ]

        # Calculate relevance scores
        scores = self.model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Combine documents with scores
        ranked_docs = sorted(
            zip(docs, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k documents
        return ranked_docs[:top_k]