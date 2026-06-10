import torch

from methods.base import BaseMethod


class EnergyMethod(BaseMethod):
    name = "energy"

    def __init__(self, model: torch.nn.Module, device: torch.device, config: object | None = None) -> None:
        super().__init__(model, device, config)
        self.temperature = 1.0

    def score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
