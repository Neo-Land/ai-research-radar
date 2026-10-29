# Build log

What was built, in the order it was built, including what went wrong and how it was fixed.
Each step lists the task, how it was done, and the outcome.

## 1. Scope and setup

**Task:** build a question-answering tool over recent AI research that produces cited, plain-language
briefs, and *prove* it works with measured results rather than demos alone.

**How:**
- Python 3.12 with [uv](https://docs.astral.sh/uv/) for a fast, reproducible virtual environment.
- No LangChain or LlamaIndex. Each step is a small module (`src/ingest.py` → `index.py` → `retrieve.py`
  → `generate.py`), which keeps the pipeline short enough to read end to end.
- `data/`, `.venv/` and `.env` are git-ignored, and the scripts recreate the data.

## 2. Data ingestion: [src/ingest.py](../src/ingest.py)

**Task:** collect recent AI papers.

**How:** the public arXiv API, newest cs.AI papers first, 1,000 per request with the 3-second pause arXiv
asks for. Each paper is stored as one JSON line: ID, title, abstract, authors, categories, date and URL.

**Problems and fixes:**
- *The first run lost 9,000 papers.* arXiv started refusing connections partway through, and the script
  only wrote its file at the end. **Fix:** retry failed requests with increasing waits (30s, 60s, ...),
  and save after every page so a late failure loses nothing.
- *arXiv stops serving a single query after 10,000 results.* The second run stopped there.
  **Decision:** 10,000 papers is plenty for this project, and date-range queries can extend it later.

**Outcome:** 10,000 papers from Sep 10 – Oct 26, 2026, 18 MB.

## 3. Indexing: [src/index.py](../src/index.py)

**Task:** make papers searchable by meaning.

**How:** embed `title + abstract` with two sentence-transformer models so they can be compared
(`all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`, both 384 dimensions). Vectors are normalized, so a
dot product equals cosine similarity, and saved as `.npy` files.

**Decision:** no vector database. At 10,000 papers a NumPy matrix multiply takes milliseconds.

## 4. Retrieval: [src/retrieve.py](../src/retrieve.py)

**Task:** return the most relevant papers for a question.

**How:**
1. **Dense search:** embed the question and take the 100 nearest papers.
2. **BM25 keyword search:** the top 100 by term overlap, which catches exact names and acronyms.
3. **Reciprocal Rank Fusion:** merge both lists by rank (k = 60, the standard constant).
4. **Cross-encoder rerank:** `ms-marco-MiniLM-L-6-v2` reads the question and each of the top 50 papers
   together, and reorders them.

Each stage can be switched on or off, which the evaluation uses to measure what each one adds.

## 5. Retrieval benchmark: [src/evaluate.py](../src/evaluate.py)

**Task:** measure retrieval quality instead of eyeballing it.

**How:** 40 random papers, one plain-language question each, written to avoid the paper's own terms.
Scored with Recall@1/5/10 and MRR across 9 configurations. Full method in [EVALUATION.md](EVALUATION.md).

**Outcome:** the final pipeline reaches Recall@1 0.75 and MRR 0.80, up from 0.42 and 0.49 for keyword
search. The ablation also showed something unexpected: adding BM25 to bge *without* the reranker made
results slightly worse on this test set. The final configuration keeps hybrid + rerank, because it
ties for best and still covers exact-term queries.

## 6. Generation: [src/generate.py](../src/generate.py)

**Task:** turn the top 8 papers into a brief a non-specialist can use.

**How:**
- The LLM must return JSON matching a Pydantic schema: `summary`, `key_findings`, `implications`
  (with who is affected), `maturity` (early research / emerging / ready to pilot) and `risks_and_limitations`.
- The prompt requires using only the given papers and citing an arXiv ID for every claim.
- `invalid_citations()` flags any cited ID that wasn't retrieved, which catches invented references.
- `call_llm()` puts Gemini and Claude behind one function. `LLM_PROVIDER` in `.env` picks one.

**Problems and fixes:**
- *Claude API credits couldn't be purchased at first* (the card was declined before reaching the bank,
  likely a payment fraud check while on a mobile hotspot). **Fix:** added Google Gemini's free tier as a
  second provider so work could continue. Claude credits went through later, and both providers remain.
- *The newest Gemini Flash models returned "503: high demand"* on the free tier. **Fix:** sent a one-word
  test to each available model and pinned `gemini-3.5-flash`, the newest one responding. Calls also
  retry automatically on 503 and 429 errors.

**Outcome:** the first end-to-end brief took about 17 seconds, with all 8 citations valid.

## 7. Brief-quality evaluation: [src/evaluate_briefs.py](../src/evaluate_briefs.py)

**Task:** measure whether briefs are accurate, not just whether retrieval works.

**How:** for 15 open questions, generate a brief, then have a judge model rule on each claim using only
the abstracts that claim cites. Reports citation validity, faithfulness (split into key findings and
implications) and usefulness, and saves every brief and verdict for human review.

**Problems and fixes:**
- *Gemini's free tier allows 20 requests per model per day,* and each question needs 2 (write + judge).
  Runs kept stopping partway. **Fix:** results are saved per question and reused, so a rerun resumes
  where it stopped. Once Claude credits worked, the evaluation moved to Claude, which removed the quota limit.
- *The first judge was too lenient.* With Gemini grading its own briefs, faithfulness was 98%. With
  Claude Opus 5.5 as the judge it was 73%, and only 50% for implications. **Fix:** `JUDGE_PROVIDER` /
  `JUDGE_MODEL` let the judge be a different (and stronger) model than the writer.

## 8. Prompt iteration driven by the evaluation

**Task:** fix the overclaiming the stronger judge found.

**How:** the judge's reasons showed three patterns: one study stated as a universal rule ("must"),
unrelated papers merged into one recommendation, and numbers stripped of their scope. Four rules
targeting exactly those were added to the writer prompt as `v2`. The original prompt is kept as `v1`,
so both can be measured on the same 15 questions with the same writer (Claude Haiku 5.5) and judge
(Claude Opus 5.5).

**Outcome:** faithfulness rose from 85% to 92%, and implications from 71% to 84%. Partially-supported
claims halved (58 → 30), and usefulness held at about 4/5. v2 is now the default. Full table in
[EVALUATION.md](EVALUATION.md). Estimated cost of all Claude evaluation runs: about $3.

## 9. App: [app.py](../app.py)

**Task:** an interface for asking questions.

**How:** Streamlit. A question box, sidebar controls for the number of papers and the retrieval mode,
the brief on the left, expandable source abstracts with arXiv links on the right, a warning if any
citation is invalid, and a **Download draft brief (.md)** button.

## Next

- Hosted demo
- Confirm the v2 prompt gain on a fresh set of questions
- Full text for the top-ranked papers, not just abstracts
- Weekly digest of new papers on chosen topics
