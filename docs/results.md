# Retrieval results (40 benchmark queries)

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.42 | 0.55 | 0.65 | 0.49 |
| dense (all-MiniLM-L6-v2) | 0.47 | 0.85 | 0.88 | 0.62 |
| dense + rerank (all-MiniLM-L6-v2) | 0.72 | 0.88 | 0.93 | 0.78 |
| hybrid (all-MiniLM-L6-v2) | 0.57 | 0.78 | 0.88 | 0.67 |
| hybrid + rerank (all-MiniLM-L6-v2) | 0.75 | 0.88 | 0.95 | 0.80 |
| dense (bge-small-en-v1.5) | 0.65 | 0.88 | 0.97 | 0.75 |
| dense + rerank (bge-small-en-v1.5) | 0.72 | 0.88 | 0.95 | 0.79 |
| hybrid (bge-small-en-v1.5) | 0.62 | 0.82 | 0.90 | 0.71 |
| hybrid + rerank (bge-small-en-v1.5) | 0.75 | 0.88 | 0.95 | 0.80 |
