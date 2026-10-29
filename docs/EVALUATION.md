# Evaluation

The project is measured at two levels: **did we find the right papers** (retrieval), and **did the
brief describe them accurately** (generation). Both evaluations are scripts, so every number in the
README can be reproduced.

## 1. Retrieval

**Script:** [src/evaluate.py](../src/evaluate.py). **Data:** [eval/benchmark.jsonl](../eval/benchmark.jsonl). **Output:** [results.md](results.md).

### Test set

40 papers were sampled at random (fixed seed) from the 10,000-paper corpus. For each one, a question was
written that the paper answers, phrased the way a non-specialist would ask it and **avoiding the paper's
own key terms**. Example:

> Paper: *PerMIT: Securing Persistent Memory for LLM Agents via Machine-Checked Integrity Theorems...*
> Question: "Protecting an AI assistant's long-term memory from being poisoned by malicious injected content"

The paper the question was written from counts as the correct answer. This is a *known-item* benchmark.

### Metrics

| Metric | Meaning |
|---|---|
| Recall@k | Share of questions where the correct paper is in the top k results |
| MRR | Mean of 1 / (rank of the correct paper). 1.0 means always first; 0.5 means second on average |

### Results

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.42 | 0.55 | 0.65 | 0.49 |
| dense (all-MiniLM-L6-v2) | 0.47 | 0.85 | 0.88 | 0.62 |
| dense + rerank (all-MiniLM-L6-v2) | 0.72 | 0.88 | 0.93 | 0.78 |
| hybrid (all-MiniLM-L6-v2) | 0.57 | 0.78 | 0.88 | 0.67 |
| hybrid + rerank (all-MiniLM-L6-v2) | 0.75 | 0.88 | 0.95 | 0.80 |
| dense (bge-small-en-v1.5) | 0.65 | 0.88 | 0.97 | 0.75 |
| dense + rerank (bge-small-en-v1.5) | 0.72 | 0.88 | 0.95 | 0.79 |
| hybrid (bge-small-en-v1.5) | 0.62 | 0.82 | 0.90 | 0.71 |
| **hybrid + rerank (bge-small-en-v1.5)** | **0.75** | **0.88** | **0.95** | **0.80** |

### What it shows

- **The reranker is the biggest single gain.** It lifts Recall@1 by about 0.10–0.28 in every configuration.
- **Embedding choice matters.** bge-small beats MiniLM by 0.18 Recall@1 at the same size and speed.
- **BM25 alone is weak here, and adding it to bge without reranking slightly hurts.** That is partly
  by design, since the questions avoid the papers' own words. Real queries often contain exact terms
  (model names, acronyms) where keyword search helps, so the final system keeps it.
- **A small trade-off:** bge alone has the best Recall@10 (0.97). The reranker only rescores the top
  50, so it occasionally pushes a correct paper out of the top 10. The gain at rank 1 is worth it.

### Caveats

- 40 questions means each one is worth 0.025, so gaps under about 0.1 are within noise.
- Only one paper is marked correct per question, but other papers may also answer it well. The
  scores are therefore a lower bound.
- The questions were written by an AI assistant from the sampled abstracts, not by independent users.

## 2. Brief quality

**Script:** [src/evaluate_briefs.py](../src/evaluate_briefs.py). **Data:** [eval/brief_questions.jsonl](../eval/brief_questions.jsonl). **Output:** [brief_results_v1.md](brief_results_v1.md), [brief_results_v2.md](brief_results_v2.md), and one JSON file per brief in [eval/briefs/](../eval/briefs/).

### Method

For each of 15 open questions (for example "What are the main security risks of deploying autonomous AI agents?"):

1. Retrieve the top 8 papers with the final retrieval pipeline.
2. Generate a brief with the **writer** model.
3. Send every claim (key findings and implications) to a **judge** model, together with **only the
   abstracts that claim cites**. The judge rules each claim *supported*, *partially supported* or
   *not supported*, then rates the whole brief's usefulness from 1 to 5.

Giving the judge only the cited abstracts is what makes this strict. A claim can be true in general and
still fail if its own sources don't back it up.

### Metrics

| Metric | How it is computed |
|---|---|
| Citation validity | Share of cited arXiv IDs that were among the retrieved papers. Exact check, no LLM |
| Faithfulness | Supported = 1, partially = 0.5, not supported = 0, averaged over claims. Reported for all claims, key findings only and implications only |
| Usefulness | Judge's 1–5 rating for a tech-curious non-specialist reader |

### How the evaluation evolved

**Run 1, a pilot with the writer grading itself.** Gemini wrote and Gemini judged, 7 briefs.
Faithfulness came out at **98%**.

**Run 2, a stronger judge from a different model family.** Gemini still wrote, but Claude Opus 5.5
judged, on the 11 questions completed before Gemini's free-tier quota ran out. Faithfulness fell to **73%**:

| | Key findings | Implications | All claims |
|---|---|---|---|
| Faithfulness (91 claims) | 85% | **50%** | 73% |

The 98% had mostly been a lenient grader. The stronger judge showed the problem was concentrated in
the implications, where the writer:

- turned a single study into a universal rule ("organizations **must** implement...");
- merged two unrelated papers into one recommendation;
- dropped the scope a number came with (a result for "the best agent" stated for agents in general;
  a simulation result stated as real-world).

**Run 3, the prompt fix, measured before and after.** Four rules were added to the writer prompt
(`FAITHFULNESS_RULES` in [src/generate.py](../src/generate.py)): keep each result's scope, label
position papers as such, phrase implications as possibilities rather than requirements, and don't
generalize from one study or merge unrelated papers. Both prompts were then run on all 15 questions,
with Claude Haiku 5.5 as the writer and Claude Opus 5.5 as the judge:

| Metric | v1 (original prompt) | v2 (+ faithfulness rules) |
|---|---|---|
| Citation validity | 100% | 100% |
| Faithfulness, all claims | 85% | **92%** |
| Faithfulness, key findings | 94% | 97% |
| Faithfulness, implications | 71% | **84%** |
| Claims partially supported | 58 of 195 | 30 of 191 |
| Claims not supported | 1 | 0 |
| Usefulness (1–5) | 4.0 | 4.1 |

The rules halved the partially-supported claims without making briefs less useful. v2 is the default
prompt in the app.

### Caveats

- **The v2 rules were written after reading the v1 verdicts on these same 15 questions,** so the gain
  may be partly fitted to this set. Confirming it on new questions is the next step.
- An LLM judge is a proxy for human review, not a replacement. Every verdict is saved in `eval/briefs/`
  so it can be spot-checked by hand.
- In run 3 the writer (Haiku) and judge (Opus) are different models from the same family. Run 2 is the
  cross-family check.
- Faithfulness is measured against abstracts only, so it can't catch a claim that matches the abstract
  but not the full paper.
- One run per prompt. LLM outputs vary between runs, so differences of a few points are within noise.
