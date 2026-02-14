import streamlit as st
import plotly.graph_objects as go
import numpy as np


class GenerationVisualizer:
    """Visualization components for generation uncertainty."""

    @staticmethod
    def render_token_entropy_heatmap(uncertainty, tokenizer, prompt):
        """3-row heatmap: predictive entropy, expected entropy, MI per token.

        Capped at 30 tokens for readability.
        """
        pred_ent = uncertainty.get("predictive_entropy", [])
        exp_ent = uncertainty.get("expected_entropy", [])
        mi = uncertainty.get("mutual_information", [])

        if not pred_ent:
            st.info("No token-level uncertainty data available.")
            return

        max_tokens = 30
        pred_ent = pred_ent[:max_tokens]
        exp_ent = exp_ent[:max_tokens]
        mi = mi[:max_tokens]
        n_tokens = len(pred_ent)

        # Decode token labels from tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        token_ids = input_ids[:n_tokens].tolist()
        token_labels = [tokenizer.decode([tid]).strip() or f"[{tid}]" for tid in token_ids]

        z = np.array([pred_ent, exp_ent, mi])
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=token_labels,
            y=["Predictive Entropy", "Expected Entropy", "Mutual Information"],
            colorscale="YlOrRd",
            colorbar=dict(title="Nats"),
        ))
        fig.update_layout(
            title="Token-Level Uncertainty Heatmap",
            xaxis_title="Token",
            height=250,
        )
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def render_sample_diversity(samples):
        """Display all MC dropout samples in styled containers."""
        st.subheader("MC Dropout Samples")
        for i, sample in enumerate(samples):
            st.markdown(
                f'<div style="background:#f5f5f5;border-left:3px solid #1976d2;'
                f'padding:8px 12px;margin:4px 0;border-radius:4px;font-size:0.9em;">'
                f"<strong>Sample {i + 1}:</strong> {sample}</div>",
                unsafe_allow_html=True,
            )

    @staticmethod
    def render_epistemic_aleatoric_breakdown(uncertainty):
        """Metric cards + stacked bar for epistemic/aleatoric decomposition."""
        seq_entropy = uncertainty.get("sequence_entropy", 0.0)
        mean_mi = uncertainty.get("mean_mi", 0.0)
        aleatoric = max(0.0, seq_entropy - mean_mi)

        # Metric cards
        cols = st.columns(3)
        cols[0].metric("Total Uncertainty", f"{seq_entropy:.4f}")
        cols[1].metric("Aleatoric", f"{aleatoric:.4f}")
        cols[2].metric("Epistemic (MI)", f"{mean_mi:.4f}")

        # Stacked bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Aleatoric",
            x=["Uncertainty"],
            y=[aleatoric],
            marker_color="#1976d2",
        ))
        fig.add_trace(go.Bar(
            name="Epistemic",
            x=["Uncertainty"],
            y=[mean_mi],
            marker_color="#d32f2f",
        ))
        fig.update_layout(
            barmode="stack",
            title="Uncertainty Decomposition",
            yaxis_title="Nats",
            height=300,
        )
        st.plotly_chart(fig, width="stretch")

        # Diversity metrics
        unique_ratio = uncertainty.get("unique_ratio", 0.0)
        pairwise_div = uncertainty.get("pairwise_diversity", 0.0)
        cols = st.columns(2)
        cols[0].metric("Unique Sample Ratio", f"{unique_ratio:.2f}")
        cols[1].metric("Pairwise Diversity", f"{pairwise_div:.2f}")
