import itertools

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from methods.params.common import ClassFeatureQueue, adapt_queue_sample_number
from utils import extract_features


class VOSMethod(BaseMethod):
    name = "vos"

    def __init__(self, model: torch.nn.Module, device: torch.device, config: object | None = None) -> None:
        super().__init__(model, device, config)
        self.loss_weight = 0.1
        self.default_sample_number = 1000
        self.sample_number = self.default_sample_number
        self.start_epoch = 1
        self.select = 1
        self.sample_from = 10000
        self.covariance_eps = 1e-4
        self.eps = 1e-12

        self.num_classes = model.fc.out_features
        self.feature_dim = model.fc.in_features
        self.weight_energy = nn.Linear(self.num_classes, 1, bias=False).to(device)
        nn.init.uniform_(self.weight_energy.weight)
        self.logistic_regression = nn.Linear(1, 2).to(device)
        self.feature_queue = ClassFeatureQueue(self.num_classes, self.sample_number, self.feature_dim, device)

    def fit(self, train_loader: DataLoader) -> None:
        adapt_queue_sample_number(self, train_loader)

    def parameters(self):
        return itertools.chain(
            self.model.parameters(),
            self.weight_energy.parameters(),
            self.logistic_regression.parameters(),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "weight_energy": self.weight_energy.state_dict(),
            "logistic_regression": self.logistic_regression.state_dict(),
            "feature_queue": self.feature_queue.state_dict(),
        }

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
    ) -> tuple[float, float]:
        adapt_queue_sample_number(self, train_loader)
        self.model.train()
        self.weight_energy.train()
        self.logistic_regression.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            features = extract_features(self.model, inputs)
            logits = self.model.fc(features)

            queue_was_ready = self.feature_queue.ready()
            self.feature_queue.update(features, targets)
            reg_loss = logits.new_zeros(())
            if queue_was_ready and epoch >= self.start_epoch:
                ood_samples = self._synthesize_outliers()
                if ood_samples.numel() > 0:
                    id_energy = self._weighted_logsumexp(logits)
                    ood_logits = self.model.fc(ood_samples)
                    ood_energy = self._weighted_logsumexp(ood_logits)
                    lr_inputs = torch.cat([id_energy, ood_energy], dim=0).view(-1, 1)
                    lr_labels = torch.cat(
                        [
                            torch.ones(id_energy.size(0), device=self.device),
                            torch.zeros(ood_energy.size(0), device=self.device),
                        ],
                        dim=0,
                    ).long()
                    reg_loss = self.criterion(self.logistic_regression(lr_inputs), lr_labels)

            loss = self.criterion(logits, targets) + self.loss_weight * reg_loss
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    def score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=1)

    def _weighted_logsumexp(self, logits: torch.Tensor) -> torch.Tensor:
        weights = F.relu(self.weight_energy.weight).squeeze(0).clamp_min(self.eps)
        max_logits = logits.max(dim=1, keepdim=True).values
        weighted_sum = (weights * torch.exp(logits - max_logits)).sum(dim=1).clamp_min(self.eps)
        return max_logits.squeeze(1) + torch.log(weighted_sum)

    @torch.no_grad()
    def _synthesize_outliers(self) -> torch.Tensor:
        queues = self.feature_queue.data
        means = queues.mean(dim=1)
        centered = queues - means[:, None, :]
        centered = centered.reshape(-1, self.feature_dim)
        covariance = centered.T @ centered / max(centered.size(0), 1)
        covariance = covariance + self.covariance_eps * torch.eye(self.feature_dim, device=self.device)
        cholesky = self._cholesky(covariance)
        precision = torch.cholesky_inverse(cholesky)

        ood_samples = []
        for class_idx in range(self.num_classes):
            noise = torch.randn(self.sample_from, self.feature_dim, device=self.device)
            samples = means[class_idx] + noise @ cholesky.T
            diff = samples - means[class_idx]
            mahalanobis = (diff @ precision * diff).sum(dim=1)
            selected = torch.topk(mahalanobis, min(self.select, samples.size(0)), largest=True).indices
            ood_samples.append(samples[selected])
        return torch.cat(ood_samples, dim=0)

    def _cholesky(self, covariance: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(covariance.size(0), device=covariance.device, dtype=covariance.dtype)
        jitter = self.covariance_eps
        for _ in range(5):
            try:
                return torch.linalg.cholesky(covariance + jitter * eye)
            except RuntimeError:
                jitter *= 10
        return torch.linalg.cholesky(covariance + jitter * eye)
