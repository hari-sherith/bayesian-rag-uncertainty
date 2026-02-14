import numpy as np

from .mc_dropout import MCDropoutModel
from .uncertainty import UncertaintyEstimator


class UncertainGenerator:
    """Orchestrates MC Dropout generation with uncertainty quantification.

    Ties together the dropout model, uncertainty math, and retrieval results
    into one generate() call that returns everything you need.
    """

    def __init__(self, mc_model=None, estimator=None, default_samples=10, max_new_tokens=50):
        self.mc_model = mc_model or MCDropoutModel()
        self.estimator = estimator or UncertaintyEstimator()
        self.default_samples = default_samples
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, query, docs, max_doc_chars=500):
        """Format retrieved docs + query into a prompt string."""
        context_parts = []
        for i, doc in enumerate(docs):
            text = doc.get("document", doc.get("text", ""))
            # truncate long docs so we don't blow up the context
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars] + "..."
            context_parts.append(f"[{i+1}] {text}")

        context = "\n".join(context_parts)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return prompt

    def generate(self, query, retrieved_docs, n_samples=None):
        """Full pipeline: prompt -> MC generate -> uncertainty -> result dict."""
        n = n_samples or self.default_samples
        prompt = self._build_prompt(query, retrieved_docs)

        # get MC samples
        gen_result = self.mc_model.generate(
            prompt, n_samples=n, max_new_tokens=self.max_new_tokens
        )
        texts = gen_result["texts"]

        # get logits for uncertainty math on the prompt itself
        input_ids = self.mc_model.tokenizer(prompt, return_tensors="pt")["input_ids"]
        logits = self.mc_model.forward_pass(input_ids, n_samples=n)

        # compute all uncertainty metrics
        uncertainty = self.estimator.compute_all(logits, texts)

        # pull retrieval-side uncertainty if available
        retrieval_summary = self._summarize_retrieval(retrieved_docs)

        return {
            "query": query,
            "prompt": prompt,
            "samples": texts,
            "best_answer": texts[0],  # just pick the first for now
            "uncertainty": uncertainty,
            "retrieval_uncertainty": retrieval_summary,
        }

    def _summarize_retrieval(self, docs):
        """Extract mean uncertainty stats from retrieval results."""
        if not docs:
            return {"mean_posterior_std": None, "mean_entropy": None}

        stds = [d["posterior_std"] for d in docs if "posterior_std" in d]
        ents = [d["entropy"] for d in docs if "entropy" in d]

        return {
            "mean_posterior_std": float(np.mean(stds)) if stds else None,
            "mean_entropy": float(np.mean(ents)) if ents else None,
        }
