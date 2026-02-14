import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class MCDropoutModel:
    """Wraps a HuggingFace causal LM with MC Dropout for uncertainty estimation.

    Uses distilgpt2 by default -- small enough for CPU, and actually has
    dropout layers we can exploit.
    """

    def __init__(self, model_name="distilgpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # distilgpt2 tokenizer doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _enable_dropout(self):
        """Turn on dropout layers while keeping everything else in eval mode."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _disable_dropout(self):
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, n_samples=10, max_new_tokens=50, temperature=1.0):
        """Run N stochastic forward passes and collect generated texts + scores.

        Returns dict with 'texts' list and 'all_scores' (list of score tuples).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._enable_dropout()

        texts = []
        all_scores = []

        for _ in range(n_samples):
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # decode only the new tokens
            new_tokens = out.sequences[0, inputs["input_ids"].shape[1]:]
            txt = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            texts.append(txt)
            all_scores.append(out.scores)  # tuple of (vocab_size,) tensors

        self._disable_dropout()
        return {"texts": texts, "all_scores": all_scores}

    @torch.no_grad()
    def forward_pass(self, input_ids, n_samples=10):
        """Get raw logits for a fixed input across N dropout masks.

        Returns tensor of shape (n_samples, seq_len, vocab_size).
        """
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(self.device)

        self._enable_dropout()
        logits_list = []
        for _ in range(n_samples):
            out = self.model(input_ids)
            logits_list.append(out.logits)  # (1, seq_len, vocab)

        self._disable_dropout()
        # stack along new dim -> (n_samples, seq_len, vocab_size)
        return torch.cat(logits_list, dim=0)
