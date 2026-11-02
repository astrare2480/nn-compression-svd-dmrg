"""loader 全体の loss と accuracy を、重みを更新せずに評価する。"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.module_state import _preserve_module_training_modes


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    """重みを更新せず、loss と accuracy をデータローダ全体で評価する。

    ``model.eval()`` と ``torch.no_grad()`` により、推論モードで
    BatchNorm / Dropout の挙動を固定し、勾配計算を省略する。
    呼び出し前の ``model.training`` は、root だけでなく全 submodule
    についても個別に保存し、正常終了時・例外時のどちらでも元の状態へ
    復元する（``model.train(state)`` の再帰的な副作用で frozen
    BatchNorm 等の個別状態を壊さないようにするため）。
    """
    with _preserve_module_training_modes(model):
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        if total == 0:
            raise ValueError("空の DataLoader では評価できません。")

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy
