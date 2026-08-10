"""Fashion-MNIST の取得・分割・DataLoader 作成。

参照元:
    notebooks/20_fashion_mnist/mlp/01_mlp_baseline_early_stopping.ipynb
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..utils.seed import make_torch_generator


def get_fashion_mnist_datasets(
    data_dir: Path | str,
    *,
    transform=None,
    download: bool = True,
):
    """Fashion-MNIST の学習用全データとテストデータを取得する。

    指定がなければ元Notebookと同じ ``ToTensor`` を適用する。画像は
    PIL Image から値域 [0, 1] の ``(1, 28, 28)`` Tensor へ変換される。
    """
    from torchvision import datasets, transforms

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    return full_train_dataset, test_dataset


def split_fashion_mnist_dataset(
    dataset: Dataset,
    split_lengths: tuple[int, ...],
    *,
    seed: int,
):
    """学習データを指定サイズへ再現可能に分割する。

    元Notebookの 50,000 / 5,000 / 5,000 分割に対応する。最後の部分集合は
    rank選択専用にし、Early Stopping用 validation と分けている。
    """
    return random_split(
        dataset,
        split_lengths,
        generator=make_torch_generator(seed),
    )


def make_fashion_mnist_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    *,
    train_batch_size: int,
    validation_batch_size: int,
    test_batch_size: int,
    rank_validation_dataset: Dataset | None = None,
    train_generator: torch.Generator | None = None,
):
    """元Notebookと同じ shuffle 方針で DataLoader を作成する。

    学習用だけを shuffle し、validation・rank選択・test は順序を固定する。
    ``train_generator`` を渡すと学習バッチの並びも再現できる。
    """
    loaders = {
        "train_loader": DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=train_generator,
        ),
        "validation_loader": DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
        ),
        "test_loader": DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
        ),
    }

    if rank_validation_dataset is not None:
        loaders["validation_loader_rank"] = DataLoader(
            rank_validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
        )

    return loaders
