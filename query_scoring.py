from dataclasses import dataclass
from typing import List
import numpy as np
import re
from langchain_core.documents import Document
from config import Config
from data_ingestor import create_embeddings

@dataclass
class QueryScore:
    score: float
    semantic: float
    lexical: float
    num_docs: int
    label: str

class QueryScorer:
    """
    Computes retrieval quality score using:
    - Semantic similarity (query vs retrieved docs)
    - Lexical coverage (query keyword overlap)
    """
    def __init__(self, embedder=None):
        self.embedder = embedder or create_embeddings()
        self.top_k = Config.Chatbot.QUERY_SCORE_TOP_K
        self.weights = Config.Chatbot.QUERY_SCORE_WEIGHTS
        self.thresholds = Config.Chatbot.QUERY_SCORE_THRESHOLDS

    def _lexical_score(self, query: str, docs: List[Document]) -> float:
        tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        tokens = [t for t in tokens if len(t) > 2]
        if not tokens:
            return 0.0

        doc_text = " ".join(d.page_content for d in docs).lower()
        matched = sum(1 for t in set(tokens) if t in doc_text)
        return matched / max(1, len(set(tokens)))

    def score(self, query: str, docs: List[Document]) -> QueryScore:
        if not docs:
            return QueryScore(0.0, 0.0, 0.0, 0, 'low')

        docs = docs[: self.top_k]
        q_emb = np.array(self.embedder.embed_query(query))
        d_embs = np.array(
            self.embedder.embed_documents([d.page_content[:1000] for d in docs])
        )

        # cosine similarity
        q_norm = np.linalg.norm(q_emb) + 1e-9
        d_norms = np.linalg.norm(d_embs, axis=1) + 1e-9
        sims = (d_embs @ q_emb) / (d_norms * q_norm)
        semantic = float(np.mean(sims))

        lexical = self._lexical_score(query, docs)
        score = (
            self.weights["semantic"] * semantic
            + self.weights["lexical"] * lexical
        )

        if score >= self.thresholds["high"]:
            label = "high"
        elif score >= self.thresholds["medium"]:
            label = "medium"
        else:
            label = "low"

        return QueryScore(score, semantic, lexical, len(docs), label)
        