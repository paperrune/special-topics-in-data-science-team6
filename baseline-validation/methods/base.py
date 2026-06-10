import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from utils import evaluate_accuracy, fpr_at_tpr


class BaseMethod:
    name = "base"

    def __init__(self, model: nn.Module, device: torch.device, config: object | None = None) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_loader: DataLoader) -> None:
        pass

    def parameters(self):
        return self.model.parameters()

    def state_dict(self) -> dict[str, object]:
        return {}

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            correct += logits.argmax(dim=1).eq(targets).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    def score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return self.score_logits(logits)

    @torch.inference_mode()
    def collect_scores(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        scores = []

        for inputs, _ in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            scores.append(self.score_batch(inputs).cpu().numpy())

        return np.concatenate(scores)

    def evaluate(self, id_loader: DataLoader, ood_loader: DataLoader) -> dict[str, float]:
        id_scores = self.collect_scores(id_loader)
        ood_scores = self.collect_scores(ood_loader)

        labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        scores = np.concatenate([id_scores, ood_scores])

        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores)),
            "fpr95": fpr_at_tpr(id_scores, ood_scores),
            "id_score_mean": float(id_scores.mean()),
            "ood_score_mean": float(ood_scores.mean()),
        }

    def evaluate_epoch(
        self,
        epoch: int,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        train_loss: float | None = None,
        train_acc: float | None = None,
    ) -> dict[str, float | int | None]:
        id_loss, id_acc = evaluate_accuracy(self.model, id_loader, self.device)
        metrics = self.evaluate(id_loader, ood_loader)
        return {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "id_loss": id_loss,
            "id_acc": id_acc,
            **metrics,
        }
