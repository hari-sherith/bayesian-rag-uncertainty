import streamlit as st
import plotly.graph_objects as go


class RetrievalVisualizer:
    """Visualization components for retrieval uncertainty."""

    @staticmethod
    def _doc_label(doc, i):
        """Extract a human-readable label from a document."""
        meta = doc["document"].metadata if hasattr(doc["document"], "metadata") else {}
        source = meta.get("source", "doc")
        # Strip path to just filename
        if "/" in source:
            source = source.rsplit("/", 1)[-1]
        chunk_id = meta.get("chunk_id", i)
        return f"{source} #{chunk_id}"

    @staticmethod
    def render_doc_uncertainty_bars(docs):
        """Horizontal bar chart: posterior_mean with CI error bars, color-coded by entropy."""
        labels = [RetrievalVisualizer._doc_label(d, i) for i, d in enumerate(docs)]
        means = [d["posterior_mean"] for d in docs]
        lowers = [d["ci_lower"] for d in docs]
        uppers = [d["ci_upper"] for d in docs]
        entropies = [d["entropy"] for d in docs]

        error_minus = [m - lo for m, lo in zip(means, lowers)]
        error_plus = [hi - m for m, hi in zip(means, uppers)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels,
            x=means,
            orientation="h",
            error_x=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus),
            marker=dict(
                color=entropies,
                colorscale="RdYlGn_r",
                colorbar=dict(title="Entropy"),
            ),
        ))
        fig.update_layout(
            title="Document Posterior Means with Credible Intervals",
            xaxis_title="Posterior Mean",
            yaxis_title="Document",
            height=max(300, len(docs) * 60),
        )
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def render_credible_intervals(docs):
        """Forest plot: connected scatter showing CI lower → mean → upper per doc."""
        labels = [RetrievalVisualizer._doc_label(d, i) for i, d in enumerate(docs)]

        fig = go.Figure()
        for i, doc in enumerate(docs):
            fig.add_trace(go.Scatter(
                x=[doc["ci_lower"], doc["posterior_mean"], doc["ci_upper"]],
                y=[labels[i]] * 3,
                mode="lines+markers",
                marker=dict(size=[6, 10, 6], color=["#1976d2", "#d32f2f", "#1976d2"]),
                line=dict(color="#90a4ae", width=2),
                showlegend=False,
            ))
        fig.update_layout(
            title="Credible Intervals (Forest Plot)",
            xaxis_title="Relevance Score",
            height=max(300, len(docs) * 60),
        )
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def render_doc_cards(docs):
        """Expandable cards per document with text content and metrics."""
        for i, doc in enumerate(docs):
            label = RetrievalVisualizer._doc_label(doc, i)
            content = doc["document"].page_content if hasattr(doc["document"], "page_content") else str(doc["document"])
            with st.expander(f"**{label}** — similarity {doc['similarity']:.3f}"):
                st.write(content[:500] + ("..." if len(content) > 500 else ""))
                cols = st.columns(4)
                cols[0].metric("Similarity", f"{doc['similarity']:.3f}")
                cols[1].metric("Posterior Mean", f"{doc['posterior_mean']:.3f}")
                cols[2].metric("Posterior Std", f"{doc['posterior_std']:.3f}")
                cols[3].metric("Entropy", f"{doc['entropy']:.3f}")
