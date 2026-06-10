import torch
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from utils import extract_features


class ViMMethod(BaseMethod):
    name = "vim"

    def __init__(self, model: torch.nn.Module, device: torch.device, config: object | None = None) -> None:
        super().__init__(model, device, config)
        self.principal_dim = 256
        self.eps = 1e-12
        self.alpha: torch.Tensor | None = None
        self.origin: torch.Tensor | None = None
        self.residual_basis: torch.Tensor | None = None

    @torch.inference_mode()
    def fit(self, train_loader: DataLoader) -> None:
        self.model.eval()
        features = []
        logits = []

        for inputs, _ in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            batch_features = extract_features(self.model, inputs)
            batch_logits = self.model.fc(batch_features)
            features.append(batch_features.cpu())
            logits.append(batch_logits.cpu())

        features_tensor = torch.cat(features, dim=0).float()
        logits_tensor = torch.cat(logits, dim=0).float()

        origin = self._compute_origin().cpu()
        centered_features = features_tensor - origin

        covariance = centered_features.T @ centered_features / max(centered_features.size(0) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)

        feature_dim = centered_features.size(1)
        residual_dim = max(feature_dim - self.principal_dim, 1)
        residual_basis = eigenvectors[:, :residual_dim]
        residual_norm = torch.linalg.norm(centered_features @ residual_basis, dim=1)

        max_logits = logits_tensor.max(dim=1).values
        alpha = max_logits.mean() / residual_norm.mean().clamp_min(self.eps)

        self.alpha = alpha.to(self.device)
        self.origin = origin.to(self.device)
        self.residual_basis = residual_basis.to(self.device)

    def score_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.alpha is None or self.origin is None or self.residual_basis is None:
            raise RuntimeError("ViMMethod.fit(train_loader) must be called before evaluation.")

        features = extract_features(self.model, inputs).float()
        logits = self.model.fc(features)
        residual_norm = torch.linalg.norm((features - self.origin) @ self.residual_basis, dim=1)
        energy = torch.logsumexp(logits, dim=1)
        return energy - self.alpha * residual_norm

    def _compute_origin(self) -> torch.Tensor:
        weight = self.model.fc.weight.detach().cpu().float()
        if self.model.fc.bias is None:
            bias = torch.zeros(weight.size(0), dtype=weight.dtype)
        else:
            bias = self.model.fc.bias.detach().cpu().float()
        return -torch.linalg.pinv(weight) @ bias
