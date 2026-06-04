"""HybridRAG — Streamlit UI.

A thin browser front-end over the same retrieve + answer functions the
FastAPI app uses. Calls them directly (no HTTP layer) so there's one
process to run instead of two.

Run:
    venv/bin/streamlit run streamlit_app.py
"""

import streamlit as st

from llm import answer
from retrieve import dense, hybrid, hybrid_rerank, sparse


RETRIEVERS = {
    "hybrid_rerank": hybrid_rerank,
    "hybrid":        hybrid,
    "dense":         dense,
    "sparse":        sparse,
}


st.set_page_config(page_title="HybridRAG", page_icon="📚", layout="centered")
st.title("HybridRAG")
st.caption("Hybrid-search RAG over technical docs — FastAPI, Pydantic, HuggingFace, LangChain, LangGraph, Anthropic.")

with st.sidebar:
    st.header("Retrieval settings")
    mode = st.selectbox("Mode", list(RETRIEVERS.keys()), index=0)
    k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5)
    st.caption(
        "**dense** — vector similarity\n\n"
        "**sparse** — Postgres full-text search\n\n"
        "**hybrid** — RRF fusion of the two\n\n"
        "**hybrid_rerank** — hybrid + Cohere rerank-3.5"
    )

query = st.text_input("Question", placeholder="How do I add CORS in FastAPI?")

if st.button("Ask", type="primary", disabled=not query):
    with st.spinner(f"Retrieving with `{mode}` …"):
        chunks = RETRIEVERS[mode](query, k=k)

    if not chunks:
        st.warning("No chunks matched this query.")
    else:
        with st.spinner("Generating answer …"):
            result = answer(query, chunks)

        st.markdown("### Answer")
        st.markdown(result["answer"])

        if result["sources"]:
            st.markdown("### Sources")
            for s in result["sources"]:
                title = s["title"] or "(no title)"
                st.markdown(f"- **{s['source']}/{s['doc_id']}** — {title}  `[chunk_{s['id']}]`")
        else:
            st.info("The answer didn't cite any specific chunks.")
