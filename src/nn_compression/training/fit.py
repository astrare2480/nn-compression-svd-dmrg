"""Early Stopping 付きの基本学習処理。

参照元:
    notebooks/20_fashion_mnist/mlp/01_mlp_baseline_early_stopping.ipynb
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from .loops import evaluate, train_one_epoch


def non_shuffling_loader(loader: DataLoader) -> DataLoader:
    """同じ Dataset を shuffle せず走査する DataLoader を返す。

    学習用 ``shuffle=True`` loader を評価で再走査すると Generator が進み、
    次 epoch や次候補の mini-batch 順が変わる。train 指標はこちらで測る。
    """
    kwargs = {
        "dataset": loader.dataset,
        "batch_size": loader.batch_size,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": False,
        "timeout": loader.timeout,
    }
    if loader.num_workers > 0:
        kwargs["worker_init_fn"] = loader.worker_init_fn
        kwargs["multiprocessing_context"] = loader.multiprocessing_context
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers
    return DataLoader(**kwargs)


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    *,
    reevaluate_train: bool = True,
    log_every_epoch: bool = False,
    train_eval_loader: DataLoader | None = None,
):
    """Early Stopping付きで学習し、最良validation loss時の重みへ戻す。

    各epochは「学習 → validation評価」のあと、必要なら train 指標を
    別 loader で取り直す。

    train 再評価に ``train_loader``（shuffle=True）を使わない。
    ``train_eval_loader`` があればそれを使う（このときは
    ``reevaluate_train=True`` が必要）。なければ
    ``reevaluate_train=True`` のときだけ、同じ Dataset の
    shuffle=False loader を内部で作る。どちらも無い /
    ``reevaluate_train=False`` なら ``train_one_epoch`` の戻り値を履歴に使う。

    ``min_delta`` を超えて validation loss が改善した時だけ最良重みを更新し、
    連続 ``patience`` 回改善しなければ停止する。学習率・epoch数は引数。
    """
    best_validation_loss = float("inf")
    best_epoch = None
    best_model_state = None
    no_improvement_count = 0
    history = []

    if train_eval_loader is not None and not reevaluate_train:
        raise ValueError(
            "train_eval_loader を渡すときは reevaluate_train=True にしてください。"
            " train 再評価をしない場合は train_eval_loader を省略し、"
            " reevaluate_train=False を指定してください。"
        )

    if train_eval_loader is not None:
        train_metric_loader = train_eval_loader
    elif reevaluate_train:
        train_metric_loader = non_shuffling_loader(train_loader)
    else:
        train_metric_loader = None

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        if train_metric_loader is not None:
            # shuffle 付き train_loader は再走査しない（学習順を変えない）。
            train_loss, train_acc = evaluate(
                model,
                train_metric_loader,
                criterion,
                device,
            )

        # 元Notebookと同じ厳密な改善判定。
        if best_validation_loss - min_delta > validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        history.append(
            {
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_acc": validation_acc,
            }
        )

        if log_every_epoch:
            print(
                f"epoch={epoch + 1:02d} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"val_acc={validation_acc:.4f}"
            )

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "train_loss_history": [row["train_loss"] for row in history],
        "validation_loss_history": [
            row["validation_loss"] for row in history
        ],
        "history": history,
    }
