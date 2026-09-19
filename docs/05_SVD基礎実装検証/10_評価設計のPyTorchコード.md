# 10 評価設計のPyTorchコード

数式と意味は各例にリンクした基礎理論ノートを参照する。ここにはPyTorchで確認するコードだけを置く。複数の例は独立実行を前提とせず、各例の前提条件を理論ノートで確認する。

---

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]（学習可能パラメータ数）

```python
sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]（11. test accuracy）

```python
predicted = torch.argmax(
    outputs,
    dim=1,
)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]（11. test accuracy）

```python
correct += (
    predicted == labels
).sum().item()
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]（11. test accuracy）

```python
total += labels.size(0)
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]（loss集計）

```python
total_loss += loss.item() * images.size(0)
```

## PyTorch確認-006

対応する数式・説明：[[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]（14. optimizerと候補モデルを独立させる）

```python
pareto_model = copy.deepcopy(source_model)
```

