from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.

    Args:
        temperature: Controls randomness. Higher values (e.g., 1.0) make output more random,
            lower values (e.g., 0.6) make it more deterministic. 0 means greedy sampling.
        max_tokens: Maximum number of tokens to generate.
        ignore_eos: If True, ignore the end-of-sequence token and continue generating.
        top_p: Nucleus sampling parameter. Keeps only tokens with cumulative probability
            below this threshold. 1.0 means no filtering.
        top_k: Keeps only the top k tokens with highest probability. 0 means no filtering.
        repetition_penalty: Penalty for repeated tokens. 1.0 means no penalty.
            Values > 1.0 penalize repetition.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k >= 0, "top_k must be >= 0"
        assert self.repetition_penalty > 0, "repetition_penalty must be > 0"
