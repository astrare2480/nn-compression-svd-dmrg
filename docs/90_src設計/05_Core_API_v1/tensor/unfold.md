# `unfold`

**Stability:** A  
**定義:** `src/nn_compression/tensor/operations.py`

## 責務

Tensorの指定mode（軸）を行方向へ移し、mode-n unfolding（mode-n展開）の2次元行列を作る。

## Signature

```python
unfold(X: torch.Tensor, mode: int) -> torch.Tensor
```

## 引数

- `X`: 展開対象Tensor。2階以上で、各dimensionは正である必要がある。
- `mode`: 行方向に置くaxis。bool以外のintで `0 <= mode < X.ndim`。

## 戻り値

shapeが次の2次元Tensor。

```text
(X.shape[mode], product(X.shape[m] for m != mode))
```

指定したmodeの大きさを行数にし、それ以外のmodeをすべて列方向へまとめた行列になる。

## 使用場面

- HOSVDでmodeごとの特異ベクトルを求める前処理。
- HOOIで更新対象modeのSVDを行う前処理。
- `mode_dot()`でTensorと行列のmode-n積を計算するとき。

## 処理概要

1. **入力Tensorのshapeを検証する。**  
   `X`が2階以上であること、0を含むdimensionがないことを確認する。Tensor分解の途中で不正shapeを黙って通さないための入口チェック。
2. **`mode`が有効な軸番号か確認する。**  
   boolを整数として扱わず、`0 <= mode < X.ndim` を満たすことを確認する。
3. **展開対象modeを先頭axisへ移動する。**  
   `X.movedim(mode, 0)` により、対象modeをaxis 0へ移す。他のaxis同士の相対的な順序は維持される。
4. **対象mode以外を列方向へまとめる。**  
   `flatten(start_dim=1)` でaxis 1以降を1次元へまとめ、2次元行列へ変換する。
5. **mode-n unfolding行列を返す。**  
   行数は対象modeのdimension、列数は残りdimensionの積になる。

## 主なcontract / 注意事項

- `torch.Tensor.unfold()` のsliding-window処理とは別物。
- `fold()` とaxis規約を必ず一致させる。
- データを数学的に展開するだけで、SVDやrank選択は行わない。

## 関連API

`fold`, `mode_dot`, `hosvd`, `hooi_sweep`
