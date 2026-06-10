import argparse
from pathlib import Path

import torch

from dataset_registry import DATASET_NAMES
from methods import METHODS
from utils import (
    Config,
    build_dataloaders,
    build_model,
    print_experiment_info,
    print_result,
    save_checkpoint,
    set_seed,
    write_results,
)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="OOD detection baselines with ResNet-18.")
    parser.add_argument("--dataset", default="cifar10", choices=DATASET_NAMES)
    parser.add_argument("--data-root", default="data", help="Path containing the selected ID dataset.")
    parser.add_argument("--imagenet-root", default="/home/yang/data/imagenet1k", help="Path containing ImageNet train/val folders.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--method", default="msp", choices=sorted(METHODS))
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--pretrained",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use ImageNet-1K pretrained ResNet-18 weights.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return Config(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Experiment already completed: output-dir exists at {output_dir}", flush=True)
        return

    set_seed(config.seed)

    model_dir = output_dir / "models"
    result_dir = output_dir / "results"
    result_path = result_dir / f"{config.method}_results.csv"

    device = torch.device(config.device)
    train_loader, id_loader, ood_loader, dataset_info = build_dataloaders(config)
    model = build_model(num_classes=dataset_info.num_classes, pretrained=config.pretrained).to(device)
    method = METHODS[config.method](model, device, config)
    print_experiment_info(config, model, train_loader, id_loader, ood_loader, dataset_info, method.name)

    optimizer = torch.optim.Adam(method.parameters(), lr=config.lr)
    results = []

    save_checkpoint(model_dir / "epoch-0.pth", model, optimizer, epoch=0, method=method)
    method.fit(train_loader)
    row = method.evaluate_epoch(0, id_loader, ood_loader)
    results.append(row)
    write_results(result_path, results)
    print_result(row)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = method.train_one_epoch(train_loader, optimizer, epoch=epoch)
        save_checkpoint(model_dir / f"epoch-{epoch}.pth", model, optimizer, epoch=epoch, method=method)

        method.fit(train_loader)
        row = method.evaluate_epoch(epoch, id_loader, ood_loader, train_loss, train_acc)
        results.append(row)
        write_results(result_path, results)
        print_result(row)
    
    print(f"Experiment completed: checkpoints and metrics saved at {output_dir}\n\n", flush=True)


if __name__ == "__main__":
    main()
