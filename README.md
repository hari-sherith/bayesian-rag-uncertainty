# Bayesian Uncertainty Estimation for RAG Systems

Quantifying confidence in retrieval and generation using Bayesian methods.

## Overview

Retrieval-Augmented Generation (RAG) systems retrieve relevant documents and use them to ground LLM responses. However, standard RAG pipelines provide no measure of how confident the system is in its output. This project adds uncertainty quantification at two levels:

1. **Retrieval uncertainty** - How confident is the system that it retrieved the right documents?
2. **Generation uncertainty** - How confident is the model in its generated answer given the retrieved context?

By combining Bayesian inference with RAG, users get not just an answer but a calibrated confidence score indicating when the system might be wrong.

## Tech Stack

- **Python** - Core language
- **PyTorch** - Model inference and Monte Carlo dropout
- **LangChain** - RAG pipeline orchestration
- **FAISS** - Vector similarity search
- **Streamlit** - Interactive demo UI
- **SciPy** - Statistical computations

## Methods

- Bayesian inference over retrieval scores
- Monte Carlo dropout for generation uncertainty
- Temperature scaling for calibration
- Expected Calibration Error (ECE) evaluation

## Project Structure

```
bayesian-rag-uncertainty/
├── src/
│   ├── retrieval/       # Document retrieval with uncertainty
│   ├── generation/      # Text generation with MC dropout
│   ├── calibration/     # Calibration and evaluation
│   └── utils/           # Shared utilities
├── tests/               # Unit tests
├── data/                # Datasets
├── notebooks/           # Experiments and analysis
└── CLAUDE.md            # Development guide
```

## Getting Started

```bash
pip install -r requirements.txt
```
