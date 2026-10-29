"""Turn retrieved papers into a cited, plain-language research brief.

The LLM returns structured JSON (validated by Pydantic), and every citation is
checked against the papers we actually gave it, so a made-up arXiv ID is caught.
"""
import os
import time
from typing import Literal, TypeVar

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

T = TypeVar("T", bound=BaseModel)

BASE_INSTRUCTIONS = """You are a research analyst who helps people keep up with AI research.
Your reader is smart and tech-curious but does not read papers: explain what is
new, why it matters in practice, and how far it is from real-world use.

Rules:
- Use ONLY the papers provided. Do not rely on outside knowledge.
- Every finding and implication must cite the arXiv IDs it is based on.
- If the papers do not answer the question well, say so in the summary rather
  than stretching them.
- Prefer concrete numbers and results from the abstracts over vague claims.
- Implications should name who is affected (people, roles or industries) and what
  could change for them. Avoid hype.
"""

# v2 adds rules aimed at the overclaiming the evaluation found in v1
# (see docs/EVALUATION.md): implications that turned one study into a universal
# rule, merged unrelated papers, or dropped the scope a number came with.
FAITHFULNESS_RULES = """- Keep each result's scope: copy numbers exactly and keep the setting they come
  from (which benchmark, simulation only, a single case study, best model only).
- If a paper is a position paper, proposal or vision rather than an experiment,
  say so instead of presenting its ideas as results.
- Implications must follow directly from the papers they cite. Phrase them as
  possibilities ("could", "may"), never as requirements ("must", "should").
- Do not turn one study into a general rule, and only combine papers in one
  implication when they address the same point.
"""

PROMPTS = {"v1": BASE_INSTRUCTIONS, "v2": BASE_INSTRUCTIONS + FAITHFULNESS_RULES}


class Finding(BaseModel):
    claim: str = Field(description="One technical finding, in plain language")
    citations: list[str] = Field(description="arXiv IDs supporting it, e.g. 2410.12345")


class Implication(BaseModel):
    implication: str = Field(description="What this means in practice, and what could change")
    who_it_affects: str
    citations: list[str]


class Brief(BaseModel):
    summary: str = Field(description="2-3 sentence answer to the question")
    key_findings: list[Finding]
    implications: list[Implication]
    maturity: Literal["early research", "emerging", "ready to pilot"]
    maturity_rationale: str
    risks_and_limitations: list[str]


def format_sources(papers: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['id']}] {p['title']} ({p['published']})\n{p['abstract']}" for p in papers
    )


def invalid_citations(brief: Brief, papers: list[dict]) -> set[str]:
    """Citation IDs in the brief that were not among the retrieved papers."""
    allowed = {p["id"] for p in papers}
    cited = {c for item in brief.key_findings + brief.implications for c in item.citations}
    return cited - allowed


def generate_brief(question: str, papers: list[dict], prompt_version: str | None = None) -> Brief:
    """prompt_version is "v1" or "v2" (default: PROMPT_VERSION in .env, else v2)."""
    prompt = f"Question: {question}\n\nPapers:\n\n{format_sources(papers)}"
    return call_llm(prompt, PROMPTS[prompt_version or os.getenv("PROMPT_VERSION", "v2")], Brief)


def call_llm(prompt: str, system: str, schema: type[T], provider: str | None = None, model: str | None = None) -> T:
    """Ask the LLM for JSON matching `schema`.

    Provider is LLM_PROVIDER in .env ("claude" or "gemini"); model defaults to
    CLAUDE_MODEL / GEMINI_MODEL for that provider.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "claude")
    return {"claude": _claude, "gemini": _gemini}[provider](prompt, system, schema, model)


def _claude(prompt: str, system: str, schema: type[T], model: str | None) -> T:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment / .env
    model = model or os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
    # If a safety classifier declines, retry on a fallback model instead of failing.
    # Haiku has no server-side fallback, so it only gets the refusal check below.
    fallback = {} if "haiku" in model else {"fallbacks": "default", "betas": ["server-side-fallback-2026-07-01"]}
    resp = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,  # response is validated against the schema
        output_config={"effort": "medium"},
        **fallback,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("The model declined this request; try rephrasing the question.")
    return resp.parsed_output


def _gemini(prompt: str, system: str, schema: type[T], model: str | None) -> T:
    client = genai.Client()  # reads GEMINI_API_KEY from the environment / .env
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,  # response is validated against the schema
    )
    # The free tier often answers "busy" (503) or "slow down" (429); wait and retry.
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash"), contents=prompt, config=config
            )
            break
        except genai_errors.APIError as err:
            if err.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(15 * (attempt + 1))
    if resp.parsed is None:
        raise RuntimeError("Gemini returned no valid response; try rephrasing the question.")
    return resp.parsed
