import itertools

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from methods.params.common import ClassFeatureQueue, adapt_queue_sample_number
from utils import extract_features


class NPOSMethod(BaseMethod):
    name = "npos"

    def __init__(self, model: torch.nn.Module, device: torch.device, config: object | None = None) -> None:
        super().__init__(model, device, config)
        self.loss_weight = 0.1
        self.default_sample_number = 1000
        self.sample_number = self.default_sample_number
        self.start_epoch = 1
        self.select = 200
        self.sample_from = 600
        self.k = 300
        self.cov_mat = 0.1
        self.id_points_num = 2
        self.pick_nums = 2

        self.num_classes = model.fc.out_features
        self.feature_dim = model.fc.in_features
        self.feature_queue = ClassFeatureQueue(self.num_classes, self.sample_number, self.feature_dim, device)
        self.ood_head = nn.Linear(self.feature_dim, 1).to(device)
        self.outlier_criterion = nn.BCEWithLogitsLoss()
        self.eval_bank: torch.Tensor | None = None

    def parameters(self):
        return itertools.chain(self.model.parameters(), self.ood_head.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "ood_head": self.ood_head.state_dict(),
            "feature_queue": self.feature_queue.state_dict(),
        }

    @torch.inference_mode()
    def fit(self, train_loader: DataLoader) -> None:
        adapt_queue_sample_number(self, train_loader)
        self.model.eval()
        features = []
        for inputs, _ in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            batch_features = extract_features(self.model, inputs).float()
            features.append(F.normalize(batch_features, dim=1))
        self.eval_bank = torch.cat(features, dim=0)

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
    ) -> tuple[float, float]:
        adapt_queue_sample_number(self, train_loader)
        self.model.train()
        self.ood_head.train()
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
                    id_logits = self.ood_head(features).squeeze(1)
                    ood_logits = self.ood_head(ood_samples).squeeze(1)
                    head_logits = torch.cat([id_logits, ood_logits], dim=0)
                    head_labels = torch.cat(
                        [
                            torch.ones_like(id_logits),
                            torch.zeros_like(ood_logits),
                        ],
                        dim=0,
                    )
                    reg_loss = self.outlier_criterion(head_logits, head_labels)

            loss = self.criterion(logits, targets) + self.loss_weight * reg_loss
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    def score_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.eval_bank is None:
            raise RuntimeError("NPOSMethod.fit(train_loader) must be called before evaluation.")

        features = F.normalize(extract_features(self.model, inputs).float(), dim=1)
        distances = self._squared_l2(features, self.eval_bank)
        k = min(self.k, distances.size(1))
        kth_distance = torch.topk(distances, k, largest=False, dim=1).values[:, -1]
        return -kth_distance

    def score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("NPOS scores are computed from features, not logits.")

    @torch.no_grad()
    def _synthesize_outliers(self) -> torch.Tensor:
        noise = torch.randn(self.sample_from, self.feature_dim, device=self.device)
        samples = []
        for class_idx in range(self.num_classes):
            class_samples = self._generate_class_outliers(self.feature_queue.data[class_idx], noise)
            if class_samples.numel() > 0:
                samples.append(class_samples)
        if not samples:
            return torch.empty(0, self.feature_dim, device=self.device)
        return torch.cat(samples, dim=0)

    def _generate_class_outliers(self, id_features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        normalized_id = F.normalize(id_features, dim=1)
        k = min(self.k, id_features.size(0))
        select = min(self.select, id_features.size(0))
        pick_nums = min(self.pick_nums, select)
        if k < 1 or select < 1 or pick_nums < 1:
            return torch.empty(0, self.feature_dim, device=self.device)

        id_distances = self._squared_l2(normalized_id, normalized_id)
        kth_id_distances = torch.topk(id_distances, k, largest=False, dim=1).values[:, -1]
        boundary_indices = torch.topk(kth_id_distances, select, largest=True).indices
        picked = boundary_indices[torch.randperm(select, device=self.device)[:pick_nums]]

        base_points = id_features[picked]
        candidates = base_points[:, None, :] + self.cov_mat * noise[None, :, :]
        flat_candidates = candidates.reshape(-1, self.feature_dim)
        normalized_candidates = F.normalize(flat_candidates, dim=1)
        candidate_distances = self._squared_l2(normalized_candidates, normalized_id)
        kth_candidate_distances = torch.topk(candidate_distances, k, largest=False, dim=1).values[:, -1]
        kth_candidate_distances = kth_candidate_distances.view(pick_nums, self.sample_from)

        points_per_id = min(self.id_points_num, self.sample_from)
        selected_noise = torch.topk(kth_candidate_distances, points_per_id, largest=True, dim=1).indices
        flat_indices = torch.arange(pick_nums, device=self.device)[:, None] * self.sample_from + selected_noise
        return flat_candidates[flat_indices.reshape(-1)]

    def _squared_l2(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (2.0 - 2.0 * left @ right.T).clamp_min(0.0)
