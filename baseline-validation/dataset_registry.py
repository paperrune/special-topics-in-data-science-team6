from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets


DEFAULT_DATA_ROOT = "data"
DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "cub200": "cub200",
    "stanfordcars": "stanfordcars",
    "stanford_cars": "stanfordcars",
    "oxfordpets": "oxfordpets",
    "oxford_pets": "oxfordpets",
}
DATASET_NAMES = tuple(sorted(set(DATASET_ALIASES.values())))
DEFAULT_ROOTS = {
    "cifar10": "data",
    "cifar100": "data",
    "cub200": "/home/yang/data/cub200",
    "stanfordcars": "/home/yang/data/stanfordcars",
    "oxfordpets": "/home/yang/data/oxfordpets",
}
DISPLAY_NAMES = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cub200": "CUB-200",
    "stanfordcars": "Stanford Cars",
    "oxfordpets": "Oxford-IIIT Pets",
}


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    display_name: str
    root: Path
    num_classes: int
    train_split: str
    eval_split: str


@dataclass(frozen=True)
class DatasetBundle:
    train_set: Dataset
    eval_set: Dataset
    info: DatasetInfo


def canonical_dataset_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        choices = ", ".join(DATASET_NAMES)
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {choices}")
    return DATASET_ALIASES[key]


def resolve_dataset_root(dataset_name: str, data_root: str) -> Path:
    dataset_name = canonical_dataset_name(dataset_name)
    if data_root == DEFAULT_DATA_ROOT:
        return Path(DEFAULT_ROOTS[dataset_name])
    return Path(data_root)


def build_id_datasets(dataset_name: str, data_root: str, train_transform, eval_transform) -> DatasetBundle:
    dataset_name = canonical_dataset_name(dataset_name)
    root = resolve_dataset_root(dataset_name, data_root)

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(root, train=True, transform=train_transform, download=False)
        eval_set = datasets.CIFAR10(root, train=False, transform=eval_transform, download=False)
        return _bundle(dataset_name, root, train_set, eval_set, "train", "test")

    if dataset_name == "cifar100":
        train_set = datasets.CIFAR100(root, train=True, transform=train_transform, download=False)
        eval_set = datasets.CIFAR100(root, train=False, transform=eval_transform, download=False)
        return _bundle(dataset_name, root, train_set, eval_set, "train", "test")

    train_set = datasets.ImageFolder(root / "train", transform=train_transform)
    eval_split = "val" if (root / "val").is_dir() else "test"
    eval_set = datasets.ImageFolder(root / eval_split, transform=eval_transform)
    if train_set.classes != eval_set.classes:
        raise ValueError(f"{DISPLAY_NAMES[dataset_name]} train and {eval_split} class folders do not match.")
    return _bundle(dataset_name, root, train_set, eval_set, "train", eval_split)


def _bundle(
    dataset_name: str,
    root: Path,
    train_set: Dataset,
    eval_set: Dataset,
    train_split: str,
    eval_split: str,
) -> DatasetBundle:
    classes = getattr(train_set, "classes", None)
    if classes is None:
        raise ValueError(f"Dataset {dataset_name} does not expose a classes attribute.")
    info = DatasetInfo(
        name=dataset_name,
        display_name=DISPLAY_NAMES[dataset_name],
        root=root,
        num_classes=len(classes),
        train_split=train_split,
        eval_split=eval_split,
    )
    return DatasetBundle(train_set=train_set, eval_set=eval_set, info=info)
