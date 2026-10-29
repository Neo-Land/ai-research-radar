"""Score the quality of generated briefs, claim by claim.

For each benchmark question: retrieve papers, generate a brief, then a separate
LLM "judge" call checks every claim against ONLY the abstracts it cites.

  Citation validity  - share of cited IDs that were actually retrieved (exact check)
  Faithfulness       - share of claims the judge finds supported by their citations
                       (supported = 1, partially = 0.5, not supported = 0)
  Usefulness         - judge's 1-5 rating of how clear and informative the brief is for a non-specialist reader

Every brief and verdict is saved to eval/briefs/<prompt>/ for human review. Saved results
are reused, so delete that folder to re-score after changing prompts or models.

    python -m src.evaluate_briefs --prompt v1   # original prompt
    python -m src.evaluate_briefs --prompt v2   # with faithfulness rules
"""
import argparse
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.generate import Brief, call_llm, generate_brief
from src.retrieve import Retriever

QUESTIONS = Path("eval/brief_questions.jsonl")
OUT_DIR = Path("eval/briefs")
K = 8

JUDGE_INSTRUCTIONS = """You are a strict fact-checker reviewing an AI-written research brief.
For each numbered claim you are given the abstracts of the papers it cites.
Judge the claim ONLY against those abstracts, not your own knowledge:
- "supported": the abstracts clearly state or directly imply it
- "partially supported": the gist is there but the claim overstates, generalises
  or adds details (numbers, scope, causes) the abstracts do not give
- "not supported": the abstracts do not back it up, or it cites no valid paper
Then rate the whole brief's usefulness to a tech-curious non-specialist from 1
(vague, generic) to 5 (specific, concrete, explains what is new and who it affects).
"""

SCORE = {"supported": 1.0, "partially supported": 0.5, "not supported": 0.0}


class ClaimVerdict(BaseModel):
    claim_number: int
    verdict: Literal["supported", "partially supported", "not supported"]
    reason: str


class Judgement(BaseModel):
    verdicts: list[ClaimVerdict]
    usefulness: int
    usefulness_reason: str


def claims_of(brief: Brief) -> list[tuple[str, list[str]]]:
    return [(f.claim, f.citations) for f in brief.key_findings] + [
        (i.implication, i.citations) for i in brief.implications
    ]


def judge_prompt(question: str, brief: Brief, papers: list[dict]) -> str:
    by_id = {p["id"]: p for p in papers}
    parts = [f"Question the brief answers: {question}\n"]
    for n, (claim, cites) in enumerate(claims_of(brief), 1):
        sources = "\n".join(
            f"  [{c}] {by_id[c]['title']}: {by_id[c]['abstract']}" if c in by_id else f"  [{c}] (not a retrieved paper)"
            for c in cites
        ) or "  (no citations)"
        parts.append(f"Claim {n}: {claim}\nCited abstracts:\n{sources}\n")
    return "\n".join(parts)


def evaluate_one(retriever: Retriever, question: str, prompt_version: str) -> dict:
    papers = retriever.search(question, k=K)
    brief = generate_brief(question, papers, prompt_version)
    # A different model judges than writes, so the writer does not grade its own work.
    judgement = call_llm(
        judge_prompt(question, brief, papers), JUDGE_INSTRUCTIONS, Judgement,
        provider=os.getenv("JUDGE_PROVIDER"), model=os.getenv("JUDGE_MODEL"),
    )

    retrieved = {p["id"] for p in papers}
    cites = [c for _, cs in claims_of(brief) for c in cs]
    scores = [SCORE[v.verdict] for v in judgement.verdicts]
    n_findings = len(brief.key_findings)
    return {
        "question": question,
        "papers": [p["id"] for p in papers],
        "brief": brief.model_dump(),
        "judgement": judgement.model_dump(),
        "n_claims": len(claims_of(brief)),
        "citation_validity": sum(c in retrieved for c in cites) / max(len(cites), 1),
        "faithfulness": sum(scores) / max(len(scores), 1),
        # Per-claim scores, split by type: claims 1..n are findings, the rest implications.
        "finding_scores": scores[:n_findings],
        "implication_scores": scores[n_findings:],
        "usefulness": judgement.usefulness,
    }


def main(prompt_version: str) -> None:
    out_dir = OUT_DIR / prompt_version
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever = Retriever()
    questions = [json.loads(line)["question"] for line in QUESTIONS.open()]
    rows = []
    for n, q in enumerate(questions, 1):
        # Reuse saved results so a run cut off by API rate limits resumes where it stopped.
        saved = out_dir / f"{n:02d}.json"
        r = json.loads(saved.read_text()) if saved.exists() else evaluate_one(retriever, q, prompt_version)
        saved.write_text(json.dumps(r, indent=2))
        rows.append(r)
        print(f"{n:>2}. faith {r['faithfulness']:.2f}  cites {r['citation_validity']:.2f}  useful {r['usefulness']}  {q[:60]}")

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    findings = [x for r in rows for x in r["finding_scores"]]
    implications = [x for r in rows for x in r["implication_scores"]]
    counts = {v: sum(x["verdict"] == v for r in rows for x in r["judgement"]["verdicts"]) for v in SCORE}
    writer = os.getenv("CLAUDE_MODEL") if os.getenv("LLM_PROVIDER") == "claude" else os.getenv("GEMINI_MODEL")
    results = Path(f"docs/brief_results_{prompt_version}.md")
    lines = [
        f"# Brief quality: prompt {prompt_version} ({len(rows)} questions, {len(findings) + len(implications)} claims)\n",
        f"Writer: `{writer}`. Judge: `{os.getenv('JUDGE_MODEL')}`. Faithfulness is averaged over claims.\n",
        "| Metric | Score |",
        "|---|---|",
        f"| Citation validity | {mean([r['citation_validity'] for r in rows]):.0%} |",
        f"| Faithfulness, all claims | {mean(findings + implications):.0%} |",
        f"| Faithfulness, key findings | {mean(findings):.0%} |",
        f"| Faithfulness, implications | {mean(implications):.0%} |",
        f"| Usefulness (1-5) | {mean([r['usefulness'] for r in rows]):.1f} |",
        "",
        "Claim verdicts: " + ", ".join(f"{v} {c}" for v, c in counts.items()),
        "",
        "| # | Question | Claims | Faithfulness | Citations | Usefulness |",
        "|---|---|---|---|---|---|",
    ] + [
        f"| {n} | {r['question']} | {r['n_claims']} | {r['faithfulness']:.0%} | {r['citation_validity']:.0%} | {r['usefulness']} |"
        for n, r in enumerate(rows, 1)
    ]
    results.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {results} and {len(rows)} briefs to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v2", help="which writer prompt to evaluate")
    main(ap.parse_args().prompt)
