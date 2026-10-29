"""Embed every paper (title + abstract) and save the vectors next to the data.

    python -m src.index --model BAAI/bge-small-en-v1.5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA = Path("data")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_papers() -> list[dict]:
    with (DATA / "papers.jsonl").open() as f:
        return [json.loads(line) for line in f]


def doc_text(paper: dict) -> str:
    return f"{paper['title']}. {paper['abstract']}"


def emb_path(model_name: str) -> Path:
    return DATA / f"emb_{model_name.replace('/', '__')}.npy"


def main(model_name: str) -> None:
    papers = load_papers()
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        [doc_text(p) for p in papers],
        batch_size=64,
        normalize_embeddings=True,  # unit vectors, so dot product = cosine similarity
        show_progress_bar=True,
    )
    np.save(emb_path(model_name), vecs.astype(np.float32))
    print(f"saved {vecs.shape} to {emb_path(model_name)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    main(ap.parse_args().model)
