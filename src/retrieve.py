"""Find the papers most relevant to a question.

Three stages, each switchable so the evaluation can measure what it adds:
  1. dense   - embedding similarity (meaning, catches paraphrases)
  2. bm25    - keyword match (exact terms, acronyms, model names)
     hybrid  - both lists merged with Reciprocal Rank Fusion
  3. rerank  - a cross-encoder rescores the top candidates by reading
               the question and each paper together
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.index import DEFAULT_MODEL, doc_text, emb_path, load_papers

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Some embedding models are trained to expect an instruction before queries.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

CANDIDATES = 100  # how deep each first-stage list goes before fusing/reranking
RRF_K = 60  # standard constant from the original RRF paper (Cormack et al., 2009)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings: list[list[int]]) -> list[int]:
    """Reciprocal Rank Fusion: merge ranked lists, rewarding items ranked high in any of them."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, rerank: bool = True):
        self.papers = load_papers()
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        # Brute-force numpy search: 10k x 384 floats is one matrix multiply, a few milliseconds.
        # Switch to FAISS if the corpus grows past a few hundred thousand papers.
        self.vectors = np.load(emb_path(model_name))
        self.bm25 = BM25Okapi([tokenize(doc_text(p)) for p in self.papers])
        self.reranker = CrossEncoder(RERANKER) if rerank else None

    def dense(self, query: str, n: int = CANDIDATES) -> list[int]:
        q = self.embedder.encode(
            QUERY_PREFIX.get(self.model_name, "") + query, normalize_embeddings=True
        )
        return np.argsort(-(self.vectors @ q))[:n].tolist()

    def keyword(self, query: str, n: int = CANDIDATES) -> list[int]:
        return np.argsort(-self.bm25.get_scores(tokenize(query)))[:n].tolist()

    def hybrid(self, query: str, n: int = CANDIDATES) -> list[int]:
        return rrf([self.dense(query), self.keyword(query)])[:n]

    def rerank(self, query: str, candidates: list[int], n: int) -> list[int]:
        pool = candidates[:50]
        scores = self.reranker.predict([(query, doc_text(self.papers[i])) for i in pool])
        return [pool[i] for i in np.argsort(-scores)[:n]]

    def search(self, query: str, k: int = 10, mode: str = "hybrid", rerank: bool = True) -> list[dict]:
        ranked = {"dense": self.dense, "bm25": self.keyword, "hybrid": self.hybrid}[mode](query)
        if rerank and self.reranker:
            ranked = self.rerank(query, ranked, k)
        return [self.papers[i] for i in ranked[:k]]
