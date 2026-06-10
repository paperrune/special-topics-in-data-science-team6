import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from dataset_registry import DatasetInfo, build_id_datasets


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "ResNet-18"
PRETRAINED_WEIGHTS = "ImageNet-1K (torchvision.models.ResNet18_Weights.IMAGENET1K_V1)"
OOD_DATASET_NAME = "ImageNet-1K validation"
RESULT_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "id_loss",
    "id_acc",
    "auroc",
    "aupr",
    "fpr95",
    "id_score_mean",
    "ood_score_mean",
]


@dataclass
class Config:
    dataset: str
    data_root: str
    imagenet_root: str
    output_dir: str
    method: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    seed: int
    pretrained: bool
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, test_transform


def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    train_transform, test_transform = build_transforms()
    id_bundle = build_id_datasets(config.dataset, config.data_root, train_transform, test_transform)
    ood_root = Path(config.imagenet_root) / "val"

    ood_test_set = datasets.ImageFolder(ood_root, transform=test_transform)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
    }
    train_loader = DataLoader(id_bundle.train_set, shuffle=True, drop_last=False, **loader_kwargs)
    id_loader = DataLoader(id_bundle.eval_set, shuffle=False, drop_last=False, **loader_kwargs)
    ood_loader = DataLoader(ood_test_set, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, id_loader, ood_loader, id_bundle.info


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def extract_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    return torch.flatten(x, 1)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def dataset_summary(name: str, loader: DataLoader) -> str:
    dataset = loader.dataset
    class_names = getattr(dataset, "classes", None)
    num_classes = len(class_names) if class_names is not None else "unknown"
    return f"{name}: {type(dataset).__name__} | samples={len(dataset)} | classes={num_classes}"


def pretrained_summary(pretrained: bool) -> str:
    if pretrained:
        return f"yes | weights={PRETRAINED_WEIGHTS}"
    return "no | weights=random initialization"


def print_experiment_info(
    config: Config,
    model: nn.Module,
    train_loader: DataLoader,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    dataset_info: DatasetInfo,
    method_name: str,
) -> None:
    print("=" * 80, flush=True)
    print("Experiment", flush=True)
    print(f"Method: {method_name}", flush=True)
    print(f"ID Dataset: {dataset_info.display_name}", flush=True)
    print(f"ID Dataset root: {dataset_info.root}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Pretrained: {pretrained_summary(config.pretrained)}", flush=True)
    print(f"Classifier head: Linear({model.fc.in_features}, {model.fc.out_features})", flush=True)
    print(f"Parameters: total={count_parameters(model):,} trainable={count_parameters(model, True):,}", flush=True)
    print("", flush=True)
    print("Fine-tuning ID dataset", flush=True)
    print(f"- {dataset_summary(dataset_info.train_split, train_loader)}", flush=True)
    print(f"- {dataset_summary(dataset_info.eval_split, id_loader)}", flush=True)
    print("", flush=True)
    print("OOD evaluation datasets", flush=True)
    print(
        f"- ID evaluation: {dataset_info.display_name} {dataset_info.eval_split} | "
        f"samples={len(id_loader.dataset)} | classes={dataset_info.num_classes}",
        flush=True,
    )
    print(f"- OOD validation: {dataset_summary(OOD_DATASET_NAME, ood_loader)}", flush=True)
    print("", flush=True)
    print("Transforms", flush=True)
    print("- train: Resize(256), RandomCrop(224), RandomHorizontalFlip, ImageNet normalize", flush=True)
    print("- test: Resize(256), CenterCrop(224), ImageNet normalize", flush=True)
    print("", flush=True)
    print("Training", flush=True)
    print(f"- optimizer=Adam lr={config.lr}", flush=True)
    print(f"- epochs={config.epochs} batch_size={config.batch_size} num_workers={config.num_workers}", flush=True)
    print(f"- device={config.device} seed={config.seed}", flush=True)
    print("", flush=True)
    print("Outputs", flush=True)
    print(f"- checkpoints={Path(config.output_dir) / 'models'}", flush=True)
    print(f"- results={Path(config.output_dir) / 'results' / f'metrics.csv'}", flush=True) # [TO LLM]: 수정하지마
    print("=" * 80, flush=True)


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    threshold = np.quantile(id_scores, 1.0 - target_tpr)
    return float(np.mean(ood_scores >= threshold))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    method: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if method is not None:
        method_state = method.state_dict()
        if method_state:
            checkpoint["method"] = method_state
    torch.save(checkpoint, path)


def write_results(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_result(row: dict[str, float | int | None]) -> None:
    formatted = []
    for key in RESULT_FIELDS:
        value = row[key]
        if isinstance(value, float):
            formatted.append(f"{key}={value:.4f}")
        else:
            formatted.append(f"{key}={value}")
    print(" ".join(formatted))
