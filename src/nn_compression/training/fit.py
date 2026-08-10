"""Early Stopping 付きの基本学習処理。

参照元:
    notebooks/20_fashion_mnist/mlp/01_mlp_baseline_early_stopping.ipynb
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

import copy

from .loops import evaluate, train_one_epoch


def fit_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    max_epochs,
    patience,
    min_delta,
):
    """Early Stopping付きで学習し、最良validation loss時の重みへ戻す。

    元Notebookと同じく各epochで「学習 → validation評価 → 学習データの
    再評価」を行う。``min_delta`` を超えて validation loss が改善した時だけ
    最良重みを更新し、連続 ``patience`` 回改善しなければ停止する。
    """
    best_validation_loss = float("inf")
    best_epoch = None
    best_model_state = None
    no_improvement_count = 0
    history = []

    for epoch in range(max_epochs):
        train_one_epoch(
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

        train_eval_loss, train_eval_acc = evaluate(
            model,
            train_loader,
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
                "train_acc": train_eval_acc,
                "train_loss": train_eval_loss,
                "validation_loss": validation_loss,
                "validation_acc": validation_acc,
            }
        )

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

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
