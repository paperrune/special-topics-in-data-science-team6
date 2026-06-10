import torch
from torch.utils.data import Subset


class ClassFeatureQueue:
    def __init__(self, num_classes: int, sample_number: int, feature_dim: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.sample_number = sample_number
        self.feature_dim = feature_dim
        self.device = device
        self.data = torch.zeros(num_classes, sample_number, feature_dim, device=device)
        self.counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    def ready(self) -> bool:
        return bool(torch.all(self.counts >= self.sample_number).item())

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        features = features.detach()
        targets = targets.detach()

        for feature, target in zip(features, targets):
            class_idx = int(target.item())
            count = int(self.counts[class_idx].item())
            if count < self.sample_number:
                self.data[class_idx, count] = feature
                self.counts[class_idx] += 1
            else:
                self.data[class_idx, :-1] = self.data[class_idx, 1:].clone()
                self.data[class_idx, -1] = feature

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "data": self.data.detach().cpu(),
            "counts": self.counts.detach().cpu(),
        }


def class_counts(dataset, num_classes: int) -> torch.Tensor:
    targets = _dataset_targets(dataset)
    if not targets:
        raise ValueError("Cannot infer class counts from an empty dataset.")
    return torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes)


def adapt_queue_sample_number(method, train_loader) -> None:
    counts = class_counts(train_loader.dataset, method.num_classes)
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot adapt feature queue: missing train samples for classes {missing}.")

    sample_number = min(method.default_sample_number, int(counts.min().item()))
    if sample_number < 1:
        raise ValueError("Feature queue sample_number must be at least 1.")
    if sample_number == method.sample_number:
        return

    method.sample_number = sample_number
    method.feature_queue = ClassFeatureQueue(method.num_classes, sample_number, method.feature_dim, method.device)


def _dataset_targets(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_targets = _dataset_targets(dataset.dataset)
        return [parent_targets[index] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(target) for _, target in dataset.samples]
    if hasattr(dataset, "imgs"):
        return [int(target) for _, target in dataset.imgs]
    raise ValueError(f"Cannot infer targets from dataset type {type(dataset).__name__}.")
