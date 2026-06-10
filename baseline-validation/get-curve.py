import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from methods import METHODS
from utils import build_dataloaders, build_model, fpr_at_tpr, set_seed


DATASET_DISPLAY = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cub200": "CUB-200",
    "oxfordpets": "Oxford-IIIT Pets",
    "stanfordcars": "Stanford Cars",
}
RUN_DISPLAY = {
    "pretrained": "Pretrained",
    "no-pretrained": "Scratch",
}
METHOD_DISPLAY = {
    "msp": "MSP",
    "energy": "Energy",
    "mahalanobis": "Mahalanobis",
    "vim": "ViM",
    "vos": "VOS",
    "npos": "NPOS",
}
METHOD_ORDER = {"msp": 0, "energy": 1, "mahalanobis": 2, "vim": 3, "vos": 4, "npos": 5}
RUN_ORDER = {"pretrained": 0, "no-pretrained": 1}


@dataclass(frozen=True)
class BaselineSpec:
    dataset: str
    run: str
    method: str
    epoch: int

    @property
    def run_dir(self) -> Path:
        return Path("saved") / self.dataset / self.method / self.run

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "models" / f"epoch-{self.epoch}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ROC and precision-recall curve coordinates for CIFAR-10 baseline checkpoints."
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"], help="Dataset to export.")
    parser.add_argument("--saved-root", default="saved", help="Root directory containing saved baseline runs.")
    parser.add_argument("--data-root", default="data", help="Root directory for CIFAR datasets.")
    parser.add_argument("--imagenet-root", default="/home/yang/data/imagenet1k", help="ImageNet-1K root with val folder.")
    parser.add_argument("--output-dir", default="outputs/cifar10_curve_data", help="Directory for exported CSV files.")
    parser.add_argument(
        "--epoch-policy",
        default="best-auroc",
        choices=["best-auroc", "best-fpr95", "final"],
        help="Checkpoint selection policy for each baseline run. best-* excludes epoch 0.",
    )
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-id-samples", default=None, type=int, help="Optional debug limit for ID samples.")
    parser.add_argument("--max-ood-samples", default=None, type=int, help="Optional debug limit for OOD samples.")
    return parser.parse_args()


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def display_run(run: str) -> str:
    return RUN_DISPLAY.get(run, run)


def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def discover_specs(args: argparse.Namespace) -> list[BaselineSpec]:
    dataset_dir = Path(args.saved_root) / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing saved dataset directory: {dataset_dir}")

    specs = []
    for method_dir in sorted(dataset_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name not in METHODS:
            continue
        for run_dir in sorted(method_dir.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "models").is_dir():
                continue
            epoch = select_epoch(run_dir, method_dir.name, args.epoch_policy)
            specs.append(BaselineSpec(args.dataset, run_dir.name, method_dir.name, epoch))

    return sorted(specs, key=lambda spec: (RUN_ORDER.get(spec.run, 99), METHOD_ORDER.get(spec.method, 99), spec.method))


def select_epoch(run_dir: Path, method: str, epoch_policy: str) -> int:
    checkpoint_epochs = []
    for path in (run_dir / "models").glob("epoch-*.pth"):
        try:
            checkpoint_epochs.append(int(path.stem.split("-")[-1]))
        except ValueError:
            pass
    if not checkpoint_epochs:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'models'}")

    if epoch_policy == "final":
        return max(checkpoint_epochs)

    result_path = run_dir / "results" / f"{method}_results.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result CSV for {epoch_policy}: {result_path}")

    metric_name = "auroc" if epoch_policy == "best-auroc" else "fpr95"
    best_epoch = None
    best_value = None
    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row["epoch"]))
            if epoch == 0:
                continue
            if epoch not in checkpoint_epochs:
                continue
            value = float(row[metric_name])
            if best_value is None:
                best_epoch = epoch
                best_value = value
                continue
            if metric_name == "auroc" and value > best_value:
                best_epoch = epoch
                best_value = value
            elif metric_name == "fpr95" and value < best_value:
                best_epoch = epoch
                best_value = value

    if best_epoch is None:
        raise ValueError(f"No trained checkpoints found in {result_path}")
    return best_epoch


def make_config(args: argparse.Namespace, spec: BaselineSpec, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=spec.dataset,
        data_root=args.data_root,
        imagenet_root=args.imagenet_root,
        output_dir=str(output_dir),
        method=spec.method,
        epochs=0,
        batch_size=args.batch_size,
        lr=1e-3,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=(spec.run == "pretrained"),
        device=args.device,
    )


def torch_load(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_method_state(method, method_state: dict[str, object] | None) -> None:
    if not method_state:
        return
    for attr in ("weight_energy", "logistic_regression", "ood_head"):
        if hasattr(method, attr) and attr in method_state:
            getattr(method, attr).load_state_dict(method_state[attr])


def build_method(args: argparse.Namespace, spec: BaselineSpec, train_loader: DataLoader, num_classes: int, output_dir: Path):
    checkpoint_path = Path(args.saved_root) / spec.dataset / spec.method / spec.run / "models" / f"epoch-{spec.epoch}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device(args.device)
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    config = make_config(args, spec, output_dir)
    method = METHODS[spec.method](model, device, config)
    load_method_state(method, checkpoint.get("method"))

    # Feature-space methods such as Mahalanobis and ViM reconstruct their
    # statistics from the saved model and the ID training split.
    method.fit(train_loader)
    return method, checkpoint_path


@torch.inference_mode()
def collect_scores(method, loader: DataLoader, device: torch.device, max_samples: int | None = None):
    method.model.eval()
    scores = []
    targets = []
    total = 0

    for inputs, labels in loader:
        if max_samples is not None and total >= max_samples:
            break
        if max_samples is not None:
            remaining = max_samples - total
            inputs = inputs[:remaining]
            labels = labels[:remaining]

        inputs = inputs.to(device, non_blocking=True)
        scores.append(method.score_batch(inputs).detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
        total += inputs.size(0)

    return np.concatenate(scores), np.concatenate(targets)


def write_raw_scores(writer, spec: BaselineSpec, split: str, curve_label: int, scores: np.ndarray, targets: np.ndarray) -> None:
    for index, (score, target) in enumerate(zip(scores, targets)):
        writer.writerow(
            {
                "dataset": spec.dataset,
                "dataset_display": display_dataset(spec.dataset),
                "run": spec.run,
                "run_display": display_run(spec.run),
                "method": spec.method,
                "method_display": display_method(spec.method),
                "epoch": spec.epoch,
                "split": split,
                "sample_index": index,
                "target": int(target),
                "curve_label": curve_label,
                "id_score": float(score),
            }
        )


def write_roc_points(writer, spec: BaselineSpec, labels: np.ndarray, scores: np.ndarray) -> None:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    for index, (x_fpr, y_tpr, threshold) in enumerate(zip(fpr, tpr, thresholds)):
        writer.writerow(
            {
                "dataset": spec.dataset,
                "dataset_display": display_dataset(spec.dataset),
                "run": spec.run,
                "run_display": display_run(spec.run),
                "method": spec.method,
                "method_display": display_method(spec.method),
                "epoch": spec.epoch,
                "point_index": index,
                "x_fpr": float(x_fpr),
                "y_tpr": float(y_tpr),
                "threshold": float(threshold),
            }
        )


def write_pr_points(
    writer,
    spec: BaselineSpec,
    labels: np.ndarray,
    scores: np.ndarray,
    positive_class: str,
) -> None:
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    for index, (x_recall, y_precision) in enumerate(zip(recall, precision)):
        threshold = float(thresholds[index]) if index < len(thresholds) else ""
        writer.writerow(
            {
                "dataset": spec.dataset,
                "dataset_display": display_dataset(spec.dataset),
                "run": spec.run,
                "run_display": display_run(spec.run),
                "method": spec.method,
                "method_display": display_method(spec.method),
                "epoch": spec.epoch,
                "positive_class": positive_class,
                "point_index": index,
                "x_recall": float(x_recall),
                "y_precision": float(y_precision),
                "threshold": threshold,
            }
        )


def write_metric_row(writer, spec: BaselineSpec, id_scores: np.ndarray, ood_scores: np.ndarray, checkpoint_path: Path) -> None:
    labels_in = np.concatenate([np.ones_like(id_scores, dtype=int), np.zeros_like(ood_scores, dtype=int)])
    scores_in = np.concatenate([id_scores, ood_scores])
    labels_out = 1 - labels_in
    scores_out = -scores_in

    writer.writerow(
        {
            "dataset": spec.dataset,
            "dataset_display": display_dataset(spec.dataset),
            "run": spec.run,
            "run_display": display_run(spec.run),
            "method": spec.method,
            "method_display": display_method(spec.method),
            "epoch": spec.epoch,
            "checkpoint": str(checkpoint_path),
            "num_id": len(id_scores),
            "num_ood": len(ood_scores),
            "auroc": float(roc_auc_score(labels_in, scores_in)),
            "aupr_in": float(average_precision_score(labels_in, scores_in)),
            "aupr_out": float(average_precision_score(labels_out, scores_out)),
            "fpr95": fpr_at_tpr(id_scores, ood_scores),
            "id_score_mean": float(id_scores.mean()),
            "ood_score_mean": float(ood_scores.mean()),
        }
    )


def open_writer(path: Path, fieldnames: list[str]):
    f = path.open("w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    specs = discover_specs(args)
    if not specs:
        raise RuntimeError(f"No baseline runs found under {Path(args.saved_root) / args.dataset}")

    raw_file, raw_writer = open_writer(
        output_dir / f"{args.dataset}_baseline_raw_scores.csv",
        [
            "dataset",
            "dataset_display",
            "run",
            "run_display",
            "method",
            "method_display",
            "epoch",
            "split",
            "sample_index",
            "target",
            "curve_label",
            "id_score",
        ],
    )
    roc_file, roc_writer = open_writer(
        output_dir / f"{args.dataset}_baseline_roc_points.csv",
        [
            "dataset",
            "dataset_display",
            "run",
            "run_display",
            "method",
            "method_display",
            "epoch",
            "point_index",
            "x_fpr",
            "y_tpr",
            "threshold",
        ],
    )
    pr_in_file, pr_in_writer = open_writer(
        output_dir / f"{args.dataset}_baseline_pr_in_points.csv",
        [
            "dataset",
            "dataset_display",
            "run",
            "run_display",
            "method",
            "method_display",
            "epoch",
            "positive_class",
            "point_index",
            "x_recall",
            "y_precision",
            "threshold",
        ],
    )
    pr_out_file, pr_out_writer = open_writer(
        output_dir / f"{args.dataset}_baseline_pr_out_points.csv",
        [
            "dataset",
            "dataset_display",
            "run",
            "run_display",
            "method",
            "method_display",
            "epoch",
            "positive_class",
            "point_index",
            "x_recall",
            "y_precision",
            "threshold",
        ],
    )
    metric_file, metric_writer = open_writer(
        output_dir / f"{args.dataset}_baseline_metrics.csv",
        [
            "dataset",
            "dataset_display",
            "run",
            "run_display",
            "method",
            "method_display",
            "epoch",
            "checkpoint",
            "num_id",
            "num_ood",
            "auroc",
            "aupr_in",
            "aupr_out",
            "fpr95",
            "id_score_mean",
            "ood_score_mean",
        ],
    )

    files = [raw_file, roc_file, pr_in_file, pr_out_file, metric_file]
    try:
        for index, spec in enumerate(specs, start=1):
            set_seed(args.seed + index)
            config = make_config(args, spec, output_dir)
            train_loader, id_loader, ood_loader, dataset_info = build_dataloaders(config)
            print(
                f"[{index}/{len(specs)}] {display_dataset(spec.dataset)} / "
                f"{display_run(spec.run)} / {display_method(spec.method)} / epoch {spec.epoch}",
                flush=True,
            )

            method, checkpoint_path = build_method(args, spec, train_loader, dataset_info.num_classes, output_dir)
            id_scores, id_targets = collect_scores(method, id_loader, device, args.max_id_samples)
            ood_scores, ood_targets = collect_scores(method, ood_loader, device, args.max_ood_samples)

            labels_in = np.concatenate([np.ones_like(id_scores, dtype=int), np.zeros_like(ood_scores, dtype=int)])
            scores_in = np.concatenate([id_scores, ood_scores])
            labels_out = 1 - labels_in
            scores_out = -scores_in

            write_raw_scores(raw_writer, spec, "id", 1, id_scores, id_targets)
            write_raw_scores(raw_writer, spec, "ood", 0, ood_scores, ood_targets)
            write_roc_points(roc_writer, spec, labels_in, scores_in)
            write_pr_points(pr_in_writer, spec, labels_in, scores_in, "id")
            write_pr_points(pr_out_writer, spec, labels_out, scores_out, "ood")
            write_metric_row(metric_writer, spec, id_scores, ood_scores, checkpoint_path)

        print(f"Saved curve data to {output_dir}", flush=True)
    finally:
        for f in files:
            f.close()


if __name__ == "__main__":
    main()
