# `tt_linear_direct_contraction_macs`

**Stability:** A
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

`tt_linear_forward` の左から右への `torch.tensordot` 縮約に対応する、1 sample あたりの site 別理論 MACs を計算する。

## Signature

```python
tt_linear_direct_contraction_macs(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
) -> tuple[int, ...]
```

## 引数

- `out_modes`: 出力側 physical modes `(m_1, ..., m_d)`。
- `in_modes`: 入力側 physical modes `(n_1, ..., n_d)`。
- `ranks`: 境界を含む bond `(r_0, ..., r_d)`。`r_0` と `r_d` は 1。

## 戻り値

site 1 から site `d` までの MACs。site `k` は `(prod_{ell < k} m_ell) * (prod_{ell > k} n_ell) * m_k * r_{k-1} * n_k * r_k`。

## 使用場面

- 実際の TT core rank から TTLinear の理論計算量を出すとき。
- Dense Linear の MACs と足して MLP 全体を比べるとき。

## 処理概要

1. mode と rank を正の整数にする。0、負、bool、Tensor は拒否する。
2. mode 列の長さと、rank 列が mode 数より 1 だけ長いことを確認する。
3. 両端の rank が 1 であることを確認する。
4. 空の積は 1 として、site ごとに積和回数を整数で返す。

## 主なcontract / 注意事項

- bias 加算、ReLU、reshape、permute、movedim、コピー、backward、SVD、optimizer 更新は含めない。
- 低ランク 2 層分解の `factorized_linear_macs` とは別の計算量である。
- rank が学習中に変わった場合は、その時点の actual ranks を渡して再計算する。

## 関連API

- `linear_macs`
- `tt_linear_forward`
