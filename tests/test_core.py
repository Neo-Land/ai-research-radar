"""Checks for the pure logic: no models, network or API keys needed.

    python -m tests.test_core      (or: pytest tests/)
"""
from src.evaluate_briefs import SCORE, claims_of
from src.generate import Brief, Finding, Implication, invalid_citations
from src.retrieve import rrf, tokenize


def make_brief(finding_cites: list[str], implication_cites: list[str]) -> Brief:
    return Brief(
        summary="s",
        key_findings=[Finding(claim="f", citations=finding_cites)],
        implications=[Implication(implication="i", who_it_affects="w", citations=implication_cites)],
        maturity="emerging",
        maturity_rationale="r",
        risks_and_limitations=[],
    )


def test_invalid_citations_flags_ids_that_were_not_retrieved():
    papers = [{"id": "2610.00001"}, {"id": "2610.00002"}]
    brief = make_brief(["2610.00001"], ["2610.00002", "9999.99999"])
    assert invalid_citations(brief, papers) == {"9999.99999"}
    assert invalid_citations(make_brief(["2610.00001"], []), papers) == set()


def test_rrf_rewards_agreement_between_rankings():
    # 2 is second in both lists, so it beats 1 and 3, which each appear in only one.
    assert rrf([[1, 2], [3, 2]])[0] == 2
    assert rrf([[1, 2], [4, 5]])[:2] in ([1, 4], [4, 1])  # ties at rank 1 in each list
    assert set(rrf([[1, 2], [2, 3]])) == {1, 2, 3}


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("GPT-4o beats LoRA, 2x!") == ["gpt", "4o", "beats", "lora", "2x"]


def test_claims_are_findings_then_implications():
    brief = make_brief(["a"], ["b"])
    assert claims_of(brief) == [("f", ["a"]), ("i", ["b"])]


def test_faithfulness_scores():
    assert SCORE == {"supported": 1.0, "partially supported": 0.5, "not supported": 0.0}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
