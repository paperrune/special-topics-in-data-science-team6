import torch
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from utils import extract_features


class MahalanobisMethod(BaseMethod):
    name = "mahalanobis"

    def __init__(self, model: torch.nn.Module, device: torch.device, config: object | None = None) -> None:
        super().__init__(model, device, config)
        self.eps = 1e-5
        self.class_means: torch.Tensor | None = None
        self.precision: torch.Tensor | None = None

    @torch.inference_mode()
    def fit(self, train_loader: DataLoader) -> None:
        self.model.eval()
        features = []
        targets = []

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            batch_features = extract_features(self.model, inputs).cpu()
            features.append(batch_features)
            targets.append(labels.cpu())

        features_tensor = torch.cat(features, dim=0).float()
        targets_tensor = torch.cat(targets, dim=0)
        num_classes = self.model.fc.out_features

        class_means = []
        centered_features = []
        for class_idx in range(num_classes):
            class_features = features_tensor[targets_tensor == class_idx]
            if class_features.numel() == 0:
                raise ValueError(f"Cannot fit Mahalanobis statistics: class {class_idx} has no samples.")

            class_mean = class_features.mean(dim=0)
            class_means.append(class_mean)
            centered_features.append(class_features - class_mean)

        centered_tensor = torch.cat(centered_features, dim=0)
        denominator = max(centered_tensor.size(0) - num_classes, 1)
        covariance = centered_tensor.T @ centered_tensor / denominator
        covariance = covariance + self.eps * torch.eye(covariance.size(0), dtype=covariance.dtype)

        self.class_means = torch.stack(class_means).to(self.device)
        self.precision = torch.linalg.pinv(covariance).to(self.device)

    def score_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.class_means is None or self.precision is None:
            raise RuntimeError("MahalanobisMethod.fit(train_loader) must be called before evaluation.")

        features = extract_features(self.model, inputs).float()
        diff = features[:, None, :] - self.class_means[None, :, :]
        distances = torch.einsum("bcd,de,bce->bc", diff, self.precision, diff)
        min_distances = distances.min(dim=1).values
        return -min_distances
