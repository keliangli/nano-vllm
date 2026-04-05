import os
import torch
from torch import nn

DISABLE_TORCH_COMPILE = os.getenv("NANOVLLM_DISABLE_TORCH_COMPILE", "").lower() in {"1", "true", "yes", "on"}


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def _forward_impl(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Greedy decoding: temperature == 0 means take argmax directly
        greedy_mask = temperatures == 0
        if greedy_mask.all():
            # All requests use greedy decoding
            return logits.argmax(dim=-1)

        logits = logits.float()

        if greedy_mask.any():
            # Mixed: some greedy, some sampling
            # For greedy requests, use very low temperature to approximate argmax
            temperatures = temperatures.clone()
            temperatures[greedy_mask] = 1.0
            logits_for_greedy = logits[greedy_mask].argmax(dim=-1)

        # Temperature-scaled softmax with Gumbel-max trick for sampling
        logits = logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        if greedy_mask.any():
            sample_tokens[greedy_mask] = logits_for_greedy

        return sample_tokens

    if DISABLE_TORCH_COMPILE:
        forward = _forward_impl
    else:
        forward = torch.compile(_forward_impl)
