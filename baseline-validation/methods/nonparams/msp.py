import torch

from methods.base import BaseMethod


class MSPMethod(BaseMethod):
    name = "msp"

    def score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        return probs.max(dim=1).values
