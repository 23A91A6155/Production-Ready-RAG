"""
app.py
======
Streamlit UI for the Production-Ready RAG Pipeline.

Run with:
  streamlit run app.py
"""

import sys
import time
import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import yaml

# ── page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Production RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d1117;
}

/* Header banner */
.rag-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f3460 50%, #16213e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.rag-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: #7d8590;
    font-size: 1rem;
    margin: 0;
}
.rag-header .badge {
    display: inline-block;
    background: rgba(88, 166, 255, 0.12);
    border: 1px solid rgba(88, 166, 255, 0.3);
    color: #58a6ff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 12px;
}

/* Answer card */
.answer-card {
    background: linear-gradient(135deg, rgba(22,33,62,0.9), rgba(15,52,96,0.6));
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 14px;
    padding: 28px 32px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
    animation: fadeIn 0.4s ease;
}
.answer-card h3 {
    color: #58a6ff;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 0 0 14px 0;
}
.answer-card p {
    color: #e6edf3;
    font-size: 1.05rem;
    line-height: 1.75;
    margin: 0;
}

/* Source chunk card */
.chunk-card {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.chunk-card:hover {
    border-color: #58a6ff;
}
.chunk-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.chunk-source {
    color: #58a6ff;
    font-weight: 600;
    font-size: 0.85rem;
}
.chunk-score {
    font-size: 0.78rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
}
.score-high   { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
.score-medium { background: rgba(210,153,34,0.15); color: #d2a022; border: 1px solid rgba(210,153,34,0.3); }
.score-low    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.chunk-text {
    color: #8b949e;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* Pipeline step badges */
.pipeline-step {
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #8b949e;
}
.pipeline-step span {
    color: #58a6ff;
    font-weight: 600;
}

/* Stat boxes */
.stat-box {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #58a6ff;
}
.stat-label {
    font-size: 0.78rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8b949e;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(88,166,255,0.15) !important;
    color: #58a6ff !important;
}

/* Input */
.stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #0d4a9e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(31,111,235,0.4) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Hide Streamlit default header */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def get_vector_store(persist_dir, collection_name):
    from src.vector_store import VectorStore
    return VectorStore(persist_dir=persist_dir, collection_name=collection_name)


def score_class(sim: float) -> str:
    if sim >= 0.75:
        return "score-high"
    elif sim >= 0.50:
        return "score-medium"
    return "score-low"


def score_label(sim: float) -> str:
    if sim >= 0.75:
        return f"✦ {sim:.2%} relevance"
    elif sim >= 0.50:
        return f"◆ {sim:.2%} relevance"
    return f"◇ {sim:.2%} relevance"


# ── sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(config, store):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0 8px 0;'>
            <div style='font-size:2.4rem'>🔍</div>
            <div style='color:#e6edf3; font-weight:700; font-size:1.1rem; margin-top:6px;'>RAG Pipeline</div>
            <div style='color:#7d8590; font-size:0.78rem;'>Production Ready</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Stats
        chunk_count = store.count() if store else 0
        st.markdown(f"""
        <div class="stat-box" style="margin-bottom:10px;">
            <div class="stat-number">{chunk_count}</div>
            <div class="stat-label">Chunks Indexed</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="font-size:1.2rem">{config.get('top_k', 4)}</div>
                <div class="stat-label">Top-K</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="font-size:1.2rem">{config.get('chunk_size', 400)}</div>
                <div class="stat-label">Chunk Size</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Architecture flow
        st.markdown("<div style='color:#58a6ff; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Pipeline Architecture</div>", unsafe_allow_html=True)

        steps = [
            ("1", "User Query"),
            ("2", "Query Embedding"),
            ("3", "Vector Search"),
            ("4", "Chunk Retrieval"),
            ("5", "Prompt Builder"),
            ("6", "LLM Generation"),
            ("7", "Answer + Citations"),
        ]
        for num, label in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span>Step {num}</span> — {label}
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Models
        st.markdown("<div style='color:#58a6ff; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Models</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8b949e; font-size:0.82rem;'>🧠 <b style='color:#e6edf3;'>Embedding</b><br/>{config.get('embedding_model','—')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8b949e; font-size:0.82rem; margin-top:10px;'>⚡ <b style='color:#e6edf3;'>LLM</b><br/>{config.get('llm_model','—')}</div>", unsafe_allow_html=True)


# ── query tab ────────────────────────────────────────────────────────────────

def render_query_tab(config, store):
    st.markdown("""
    <div class="rag-header">
        <h1>🔍 Ask Your Documents</h1>
        <p>Retrieval-Augmented Generation — powered by ChromaDB + Groq LLM</p>
        <div class="badge">Production-Ready Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    if store.is_empty():
        st.warning("⚠️ Vector store is empty. Go to the **Ingestion** tab and click **Run Ingestion** first.")
        return

    # Suggested questions
    st.markdown("<div style='color:#7d8590; font-size:0.82rem; margin-bottom:8px;'>💡 Try a sample question:</div>", unsafe_allow_html=True)
    sample_questions = [
        "What is RAG and how does it work?",
        "What are the benefits of vector databases?",
        "How does chunking affect retrieval quality?",
    ]
    cols = st.columns(len(sample_questions))
    for col, q in zip(cols, sample_questions):
        with col:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state["prefill_question"] = q

    # Text area
    default_q = st.session_state.pop("prefill_question", "")
    question = st.text_area(
        "Your Question",
        value=default_q,
        placeholder="Type your question here...",
        height=100,
        label_visibility="collapsed",
    )

    ask_col, _ = st.columns([1, 4])
    with ask_col:
        ask_clicked = st.button("🚀 Ask", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("🔄 Retrieving context and generating answer..."):
            try:
                from src.retrieval_pipeline import retrieve_top_k_chunks
                from src.prompt_builder import build_prompt
                from src.llm_generator import generate_answer

                t0 = time.time()
                chunks = retrieve_top_k_chunks(
                    query=question,
                    vector_store=store,
                    top_k=config["top_k"],
                    embedding_model=config["embedding_model"],
                )

                if not chunks:
                    st.warning("No relevant chunks found. Try rephrasing your question.")
                    return

                prompt = build_prompt(query=question, retrieved_chunks=chunks)
                answer = generate_answer(prompt=prompt, model=config["llm_model"])
                elapsed = time.time() - t0

            except EnvironmentError as e:
                st.error(f"API Key Error: {e}")
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return

        # Answer card
        st.markdown(f"""
        <div class="answer-card">
            <h3>Generated Answer</h3>
            <p>{answer.replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='color:#3fb950; font-size:0.82rem; text-align:right; margin-top:-12px; margin-bottom:20px;'>✓ Generated in {elapsed:.2f}s using {len(chunks)} source chunks</div>", unsafe_allow_html=True)

        # Sources
        st.markdown("---")
        st.markdown(f"<div style='color:#e6edf3; font-weight:600; font-size:1rem; margin-bottom:14px;'>📚 Sources Used ({len(chunks)} chunks)</div>", unsafe_allow_html=True)

        for i, chunk in enumerate(chunks, 1):
            sim = chunk["similarity"]
            sc = score_class(sim)
            sl = score_label(sim)
            preview = chunk["text"][:300].strip()
            if len(chunk["text"]) > 300:
                preview += "..."

            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-meta">
                    <div class="chunk-source">📄 {chunk['source']} &nbsp;·&nbsp; Chunk #{chunk['chunk_index']}</div>
                    <div class="chunk-score {sc}">{sl}</div>
                </div>
                <div class="chunk-text">{preview}</div>
            </div>
            """, unsafe_allow_html=True)

        # Expander: full prompt
        with st.expander("🔬 View Full Prompt Sent to LLM"):
            st.code(prompt, language="text")


# ── ingestion tab ─────────────────────────────────────────────────────────────

def render_ingestion_tab(config, store):
    st.markdown("""
    <div class="rag-header">
        <h1>📥 Document Ingestion</h1>
        <p>Load → Chunk → Embed → Store in ChromaDB</p>
        <div class="badge">One-time setup · Re-run to refresh</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    docs_dir = Path(config.get("documents_dir", "./documents"))
    doc_files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (str(len(doc_files)),       "Documents Found"),
        (str(store.count()),        "Chunks Indexed"),
        (str(config["chunk_size"]), "Chunk Size (tokens)"),
        (str(config["chunk_overlap"]), "Overlap (tokens)"),
    ]
    for col, (num, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Document list
    if doc_files:
        st.markdown(f"<div style='color:#e6edf3; font-weight:600; margin-bottom:10px;'>📁 Documents in <code>{docs_dir}</code></div>", unsafe_allow_html=True)
        for f in doc_files:
            size_kb = f.stat().st_size / 1024
            st.markdown(f"<div class='pipeline-step'>📄 <span>{f.name}</span> &nbsp;&nbsp; {size_kb:.1f} KB</div>", unsafe_allow_html=True)
    else:
        st.warning(f"No .txt or .md files found in `{docs_dir}`. Add documents and re-run.")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        force = st.checkbox("Force re-ingest", value=False, help="Re-ingest even if vector store already has data")
    with col_b:
        run_btn = st.button("⚡ Run Ingestion", use_container_width=False)

    if run_btn:
        log_box = st.empty()
        log_lines = []

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            from ingest import run_ingestion
            run_ingestion(config=config, force=force)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Ingestion failed: {e}")
            return
        finally:
            sys.stdout = old_stdout
            output = buffer.getvalue()

        st.success("✅ Ingestion complete!")
        st.code(output, language="text")

        # Refresh store stats
        st.cache_resource.clear()
        st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    try:
        store = get_vector_store(
            persist_dir=config["vector_store_dir"],
            collection_name=config["collection_name"],
        )
    except Exception as e:
        st.error(f"Could not connect to vector store: {e}")
        return

    render_sidebar(config, store)

    tab_query, tab_ingest = st.tabs(["🔍  Query", "📥  Ingestion"])

    with tab_query:
        render_query_tab(config, store)

    with tab_ingest:
        render_ingestion_tab(config, store)


if __name__ == "__main__":
    main()
