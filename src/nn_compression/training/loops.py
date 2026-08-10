"""MLP 実験で共通利用する学習・評価ループ。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
"""

import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """全学習バッチで順伝播・誤差逆伝播・更新を1回ずつ行う。

    損失はバッチ平均の加重平均、accuracy は全サンプルに対する正解率として
    返す。これは元Notebookの「学習 1 epoch」セルを関数化したもの。
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """重みを更新せず、loss と accuracy をデータローダ全体で評価する。

    ``model.eval()`` と ``torch.no_grad()`` により、推論モードで
    BatchNorm / Dropout の挙動を固定し、勾配計算を省略する。
    """
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

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
