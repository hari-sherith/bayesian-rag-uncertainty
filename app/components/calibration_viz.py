import streamlit as st
import plotly.graph_objects as go
import numpy as np


class CalibrationVisualizer:
    """Visualization components for calibration analysis."""

    @staticmethod
    def render_reliability_diagram(report, title="Reliability Diagram"):
        """Grouped bar chart (before/after) with dashed y=x diagonal."""
        before = report["before"]["diagram"]
        after = report["after"]["diagram"]

        midpoints = before["bin_midpoints"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=midpoints,
            y=before["bin_accuracies"],
            name="Before Calibration",
            marker_color="#ef5350",
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=midpoints,
            y=after["bin_accuracies"],
            name="After Calibration",
            marker_color="#66bb6a",
            opacity=0.7,
        ))
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(dash="dash", color="black", width=1),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Mean Predicted Confidence",
            yaxis_title="Fraction of Positives",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def render_confidence_histogram(report, title="Confidence Histogram"):
        """Overlayed bar histograms for before/after confidence distributions."""
        before = report["before"]["diagram"]
        after = report["after"]["diagram"]

        midpoints = before["bin_midpoints"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=midpoints,
            y=before["bin_counts"],
            name="Before",
            marker_color="#ef5350",
            opacity=0.5,
        ))
        fig.add_trace(go.Bar(
            x=midpoints,
            y=after["bin_counts"],
            name="After",
            marker_color="#66bb6a",
            opacity=0.5,
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Confidence",
            yaxis_title="Count",
            barmode="overlay",
            height=350,
        )
        st.plotly_chart(fig, width="stretch")

    @staticmethod
    def render_metric_cards(report):
        """Metric cards with delta (improvements show green via delta_color='inverse')."""
        before = report["before"]
        after = report["after"]

        metrics = []
        for key in ["ece", "mce", "brier"]:
            if key in before and key in after:
                metrics.append((key.upper(), before[key], after[key]))

        cols = st.columns(len(metrics)) if metrics else []
        for col, (name, val_before, val_after) in zip(cols, metrics):
            delta = val_after - val_before
            col.metric(
                label=f"{name}",
                value=f"{val_after:.4f}",
                delta=f"{delta:+.4f}",
                delta_color="inverse",  # decrease = green
            )

    @staticmethod
    def render_full_calibration_tab(calibrator, synthetic_data):
        """Orchestrate the entire calibration tab with synthetic data."""
        st.header("Calibration Analysis")
        st.caption(
            "Using synthetic overconfident data to demonstrate calibration. "
            "Before/after comparison shows the effect of post-hoc calibration."
        )

        # Run calibration
        retrieval_report = calibrator.calibrate_retrieval(
            synthetic_data["retrieval_confidences"],
            synthetic_data["relevance_labels"],
        )
        generation_report = calibrator.calibrate_generation(
            synthetic_data["generation_logits"],
            synthetic_data["generation_labels"],
        )

        # --- Retrieval calibration ---
        st.subheader("Retrieval Calibration (Platt Scaling)")
        CalibrationVisualizer.render_metric_cards(retrieval_report)

        col1, col2 = st.columns(2)
        with col1:
            CalibrationVisualizer.render_reliability_diagram(
                retrieval_report, "Retrieval Reliability"
            )
        with col2:
            CalibrationVisualizer.render_confidence_histogram(
                retrieval_report, "Retrieval Confidence Distribution"
            )

        st.divider()

        # --- Generation calibration ---
        st.subheader(
            f"Generation Calibration (Temperature Scaling, T={generation_report['temperature']:.2f})"
        )
        CalibrationVisualizer.render_metric_cards(generation_report)

        col1, col2 = st.columns(2)
        with col1:
            CalibrationVisualizer.render_reliability_diagram(
                generation_report, "Generation Reliability"
            )
        with col2:
            CalibrationVisualizer.render_confidence_histogram(
                generation_report, "Generation Confidence Distribution"
            )
