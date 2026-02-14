import math
import streamlit as st


class QueryInterface:
    """UI components for query input and answer display."""

    @staticmethod
    def render_input(sample_queries):
        """Render text input and example query buttons in a 3-column grid."""
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("query_text", ""),
            key="query_input",
        )

        st.caption("Or try an example:")
        cols = st.columns(3)
        for i, sq in enumerate(sample_queries):
            with cols[i % 3]:
                if st.button(sq[:50] + "..." if len(sq) > 50 else sq, key=f"sample_{i}"):
                    st.session_state["query_text"] = sq
                    st.rerun()

        return query

    @staticmethod
    def render_confidence_badge(label, value):
        """Render a colored pill badge: green >0.7, amber 0.3-0.7, red <0.3."""
        if value > 0.7:
            color, bg = "#1b5e20", "#c8e6c9"
        elif value > 0.3:
            color, bg = "#e65100", "#ffe0b2"
        else:
            color, bg = "#b71c1c", "#ffcdd2"

        st.markdown(
            f'<span style="background:{bg};color:{color};padding:4px 12px;'
            f'border-radius:12px;font-weight:600;font-size:0.85em;">'
            f"{label}: {value:.2f}</span>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_answer(generation_result):
        """Display best answer and confidence badges."""
        st.subheader("Answer")
        st.write(generation_result["best_answer"])

        uncertainty = generation_result["uncertainty"]
        retrieval_unc = generation_result.get("retrieval_uncertainty", {})

        # Normalize sequence entropy to [0, 1] using log(vocab_size) as ceiling
        vocab_size = 50257  # distilgpt2
        max_entropy = math.log(vocab_size)
        raw_entropy = uncertainty.get("sequence_entropy", 0.0)
        generation_conf = max(0.0, 1.0 - raw_entropy / max_entropy)

        # Epistemic: mutual information normalized similarly
        raw_mi = uncertainty.get("mean_mi", 0.0)
        epistemic_conf = max(0.0, 1.0 - raw_mi / max_entropy)

        # Retrieval confidence: 1 - mean_posterior_std (lower std = higher confidence)
        mean_std = retrieval_unc.get("mean_posterior_std")
        retrieval_conf = 1.0 - mean_std if mean_std is not None else 0.5

        cols = st.columns(3)
        with cols[0]:
            QueryInterface.render_confidence_badge("Generation", generation_conf)
        with cols[1]:
            QueryInterface.render_confidence_badge("Epistemic", epistemic_conf)
        with cols[2]:
            QueryInterface.render_confidence_badge("Retrieval", retrieval_conf)
