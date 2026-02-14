from dataclasses import dataclass


@dataclass
class AppConfig:
    """Default settings for the Streamlit demo app."""

    # Model
    model_name: str = "distilgpt2"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Generation
    n_samples: int = 5
    max_new_tokens: int = 50
    temperature: float = 1.0

    # Calibration
    n_bins: int = 10

    # UI
    data_dir: str = "data/raw"
    cache_dir: str = "data/embeddings"
