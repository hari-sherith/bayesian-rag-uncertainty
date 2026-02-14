import sys
import os

# Path setup so we can import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from app.config import AppConfig
from app.utils.demo_data import get_sample_queries, generate_synthetic_calibration_data
from app.components.query_interface import QueryInterface
from app.components.retrieval_viz import RetrievalVisualizer
from app.components.generation_viz import GenerationVisualizer
from app.components.calibration_viz import CalibrationVisualizer

from src.retrieval.document_loader import DocumentLoader
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStore
from src.retrieval.bayesian_retriever import BayesianRetriever
from src.generation.mc_dropout import MCDropoutModel
from src.generation.generator import UncertainGenerator
from src.calibration.calibrator import RAGCalibrator


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bayesian RAG — Uncertainty Demo",
    page_icon="🎯",
    layout="wide",
)

st.title("Bayesian RAG with Uncertainty Quantification")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render sidebar controls and return an updated AppConfig."""
    cfg = AppConfig()

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Retrieval")
        cfg.top_k = st.slider("Top-K documents", 1, 10, cfg.top_k)
        cfg.prior_alpha = st.slider("Prior alpha (Beta)", 0.5, 10.0, cfg.prior_alpha, 0.5)
        cfg.prior_beta = st.slider("Prior beta (Beta)", 0.5, 10.0, cfg.prior_beta, 0.5)

        st.subheader("Generation")
        cfg.n_samples = st.slider("MC Dropout samples", 2, 20, cfg.n_samples)
        cfg.max_new_tokens = st.slider("Max new tokens", 10, 150, cfg.max_new_tokens, 10)
        cfg.temperature = st.slider("Sampling temperature", 0.1, 2.0, cfg.temperature, 0.1)
        st.caption("Dropout rate is fixed at p=0.1 (distilgpt2 default).")

        st.subheader("Calibration")
        cfg.n_bins = st.slider("Calibration bins", 5, 20, cfg.n_bins)

    return cfg


config = render_sidebar()


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_retriever(prior_alpha, prior_beta):
    """Load documents and build the Bayesian retriever (cached)."""
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    embeddings = EmbeddingManager(model_name="all-MiniLM-L6-v2", cache_dir="data/embeddings")
    store = VectorStore(dimension=384)
    retriever = BayesianRetriever(loader, embeddings, store, prior_alpha=prior_alpha, prior_beta=prior_beta)
    retriever.index_documents("data/raw")
    return retriever


@st.cache_resource
def load_mc_model():
    """Load the MC Dropout model (cached)."""
    return MCDropoutModel(model_name="distilgpt2")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_query, tab_retrieval, tab_generation, tab_calibration = st.tabs(
    ["Query & Answer", "Retrieval Uncertainty", "Generation Uncertainty", "Calibration"]
)


# ---------------------------------------------------------------------------
# Query & Answer tab
# ---------------------------------------------------------------------------
with tab_query:
    sample_queries = get_sample_queries()
    query = QueryInterface.render_input(sample_queries)

    if st.button("Submit", type="primary") and query:
        with st.spinner("Retrieving documents..."):
            retriever = load_retriever(config.prior_alpha, config.prior_beta)
            docs = retriever.get_relevant_docs(query, top_k=config.top_k)
            st.session_state["docs"] = docs

        # Convert LangChain Document objects to dicts for the generator
        docs_for_gen = []
        for d in docs:
            docs_for_gen.append({
                "document": d["document"].page_content,
                "posterior_std": d["posterior_std"],
                "entropy": d["entropy"],
                "posterior_mean": d["posterior_mean"],
            })

        with st.spinner("Generating answer with MC Dropout..."):
            mc_model = load_mc_model()
            generator = UncertainGenerator(
                mc_model=mc_model,
                default_samples=config.n_samples,
                max_new_tokens=config.max_new_tokens,
            )
            gen_result = generator.generate(
                query, docs_for_gen, n_samples=config.n_samples
            )
            st.session_state["gen_result"] = gen_result

    # Display results if available
    if "gen_result" in st.session_state:
        QueryInterface.render_answer(st.session_state["gen_result"])


# ---------------------------------------------------------------------------
# Retrieval Uncertainty tab
# ---------------------------------------------------------------------------
with tab_retrieval:
    if "docs" not in st.session_state:
        st.info("Submit a query in the first tab to see retrieval uncertainty.")
    else:
        docs = st.session_state["docs"]
        st.header("Retrieval Uncertainty Analysis")

        RetrievalVisualizer.render_doc_uncertainty_bars(docs)

        col1, col2 = st.columns(2)
        with col1:
            RetrievalVisualizer.render_credible_intervals(docs)
        with col2:
            st.empty()

        RetrievalVisualizer.render_doc_cards(docs)


# ---------------------------------------------------------------------------
# Generation Uncertainty tab
# ---------------------------------------------------------------------------
with tab_generation:
    if "gen_result" not in st.session_state:
        st.info("Submit a query in the first tab to see generation uncertainty.")
    else:
        gen_result = st.session_state["gen_result"]
        mc_model = load_mc_model()

        st.header("Generation Uncertainty Analysis")

        GenerationVisualizer.render_token_entropy_heatmap(
            gen_result["uncertainty"],
            mc_model.tokenizer,
            gen_result["prompt"],
        )

        GenerationVisualizer.render_epistemic_aleatoric_breakdown(
            gen_result["uncertainty"]
        )

        GenerationVisualizer.render_sample_diversity(gen_result["samples"])


# ---------------------------------------------------------------------------
# Calibration tab
# ---------------------------------------------------------------------------
with tab_calibration:
    calibrator = RAGCalibrator(n_bins=config.n_bins)
    synthetic_data = generate_synthetic_calibration_data(n_samples=200)
    CalibrationVisualizer.render_full_calibration_tab(calibrator, synthetic_data)
