"""Score retrieval on the benchmark and compare configurations (ablation).

Each benchmark line is {"query": ..., "relevant": [arXiv IDs]}.

  Recall@k  - share of queries where a relevant paper appears in the top k
  MRR       - mean of 1/rank of the first relevant paper (1.0 = always first)

    python -m src.evaluate
"""
import json
from pathlib import Path

from src.retrieve import Retriever

BENCHMARK = Path("eval/benchmark.jsonl")
RESULTS = Path("docs/results.md")
EMBEDDERS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
CONFIGS = [("bm25", False), ("dense", False), ("dense", True), ("hybrid", False), ("hybrid", True)]
K = 10


def score(retriever: Retriever, benchmark: list[dict], mode: str, rerank: bool) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    for item in benchmark:
        ids = [p["id"] for p in retriever.search(item["query"], k=K, mode=mode, rerank=rerank)]
        rank = next((i + 1 for i, pid in enumerate(ids) if pid in item["relevant"]), None)
        if rank:
            rr += 1 / rank
            for k in hits:
                hits[k] += rank <= k
    n = len(benchmark)
    return {f"Recall@{k}": v / n for k, v in hits.items()} | {"MRR": rr / n}


def main() -> None:
    benchmark = [json.loads(line) for line in BENCHMARK.open()]
    rows = []
    for embedder in EMBEDDERS:
        retriever = Retriever(embedder, rerank=True)
        for mode, rerank in CONFIGS:
            if mode == "bm25" and embedder != EMBEDDERS[0]:
                continue  # BM25 ignores the embedder; report it once
            label = "bm25" if mode == "bm25" else f"{mode}{' + rerank' if rerank else ''} ({embedder.split('/')[-1]})"
            rows.append((label, score(retriever, benchmark, mode, rerank)))
            print(label, rows[-1][1])

    cols = list(rows[0][1])
    lines = [
        f"# Retrieval results ({len(benchmark)} benchmark queries)\n",
        "| Configuration | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ] + [f"| {label} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |" for label, m in rows]
    RESULTS.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
