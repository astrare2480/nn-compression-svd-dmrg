# `GpuResidentTensorBatches`

**Stability:** A  
**定義:** `src/nn_compression/training/gpu_resident_batches.py`
**参照Notebook:** `notebooks/40_dmrg/10_fashion_mnist_mlp/05_update_scope_ablation_gpu_resident.ipynb`

## 責務

すでに device 上にある入力とラベルを、sample 順の連続 slice として繰り返し供給する。

## Signature

```python
GpuResidentTensorBatches(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> None
```

## 引数

- `images`: sample 次元が先頭の入力。contiguous で、`requires_grad=False`。
- `labels`: `images` と同じ sample 数の1次元ラベル。同じ device。

## 戻り値

batch 供給オブジェクト。`batches(batch_size)` は、呼ぶたびに先頭から新しい走査を返す。

## 使用場面

- 前処理済みの学習データを一度だけ device へ置き、複数回の全データ走査で再利用するとき。
- DataLoader の CPU 転送を、勾配集計のたびに繰り返さないとき。

## 処理概要

1. sample 数、device、contiguous、勾配不要を確認する。
2. `batches` は `batch_size` ごとの半開区間を作る。
3. 最後の区間が `batch_size` より小さくても、残りの sample を返す。
4. 区間は欠落も重複もなく、`0` から sample 数までを覆う。

## 主なcontract / 注意事項

- 走査済み iterator はオブジェクトに保存しない。`batches` を再度呼べば先頭から走査できる。
- このオブジェクトは tensor を device 間でコピーしない。配置は呼び出し側が一度だけ行う。
- sample 数も core 数も固定しない。Fashion-MNIST 専用ではない。
- 既存の DataLoader 経路は残る。このクラスはその代替供給源。

## 関連API

[[90_src設計/05_Core_API_v1/training/accumulate_scoped_full_batch_gradients]]、[[90_src設計/05_Core_API_v1/training/train_scoped_full_batch_budget]]、[[90_src設計/05_Core_API_v1/Internal_API]]
