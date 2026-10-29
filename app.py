"""Streamlit front end: ask a question about AI research, get a cited brief.

    streamlit run app.py
"""
import streamlit as st

from src.generate import Brief, generate_brief, invalid_citations
from src.retrieve import Retriever

st.set_page_config(page_title="AI Research Radar", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


def to_markdown(question: str, brief: Brief, papers: list[dict]) -> str:
    """Draft brief for human review, not a publication-ready document."""
    lines = [f"# Research brief: {question}", "", "*DRAFT for review*", "", brief.summary, "", "## Key findings"]
    lines += [f"- {f.claim} [{', '.join(f.citations)}]" for f in brief.key_findings]
    lines += ["", "## Why it matters"]
    lines += [f"- **{i.who_it_affects}:** {i.implication} [{', '.join(i.citations)}]" for i in brief.implications]
    lines += ["", f"## Maturity: {brief.maturity}", brief.maturity_rationale, "", "## Risks and limitations"]
    lines += [f"- {r}" for r in brief.risks_and_limitations]
    lines += ["", "## Sources"] + [f"- [{p['id']}] {p['title']} ({p['published']}) {p['url']}" for p in papers]
    return "\n".join(lines)


st.title("AI Research Radar")
st.caption("Ask about recent AI research (arXiv cs.AI). Answers cite their sources and need human review before use.")

with st.sidebar:
    k = st.slider("Papers to use", 3, 15, 8)
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    rerank = st.checkbox("Rerank with cross-encoder", value=True)

# ?q=... in the URL pre-fills the question, so a brief can be shared as a link.
question = st.text_input("Question", value=st.query_params.get("q", ""), placeholder="What is new in making AI agents safer?")
if not question:
    st.stop()

with st.spinner("Searching papers..."):
    papers = get_retriever().search(question, k=k, mode=mode, rerank=rerank)
with st.spinner("Writing brief..."):
    brief = generate_brief(question, papers)

bad = invalid_citations(brief, papers)
if bad:
    st.warning(f"The model cited papers it was not given: {', '.join(sorted(bad))}. Treat those claims as unverified.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Summary")
    st.write(brief.summary)
    st.subheader("Key findings")
    for f in brief.key_findings:
        st.markdown(f"- {f.claim} `{', '.join(f.citations)}`")
    st.subheader("Why it matters")
    for i in brief.implications:
        st.markdown(f"- **{i.who_it_affects}:** {i.implication} `{', '.join(i.citations)}`")
    st.subheader(f"Maturity: {brief.maturity}")
    st.write(brief.maturity_rationale)
    st.subheader("Risks and limitations")
    for r in brief.risks_and_limitations:
        st.markdown(f"- {r}")
    st.download_button("Download draft brief (.md)", to_markdown(question, brief, papers), "brief.md")

with right:
    st.subheader("Sources")
    for p in papers:
        with st.expander(f"[{p['id']}] {p['title']}"):
            st.caption(f"{p['published']} · {', '.join(p['authors'][:3])}{' et al.' if len(p['authors']) > 3 else ''}")
            st.write(p["abstract"])
            st.markdown(f"[Open on arXiv]({p['url']})")
