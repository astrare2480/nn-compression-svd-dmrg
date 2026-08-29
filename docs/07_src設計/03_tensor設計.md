---
title: tensor設計
---

# tensor設計

## 1. 役割

`nn_compression.tensor` は、Tucker/HOOI等の分解アルゴリズムより下位にある**mode-nテンソル演算層**である。

```text
src/nn_compression/tensor/
├─ operations.py
└─ validation.py
```

Public API：

```python
unfold
fold
mode_dot
```

この層は `compression` に依存しない。

---

## 2. 共通shape contract

Public tensor APIで扱うTensorは原則として、

```text
ndim >= 2
各dimension > 0
```

を要求する。

`validate_tensor_shape()` がこのcontractを実装する。

0次元scalar、1次元vector、0-sized dimensionを含むTensorは、Tucker/HOOIでのmode演算対象として曖昧さが生じるため明示的に拒否する。

---

## 3. mode contract

modeはaxis番号を表す。

```text
boolではないint
0 <= mode < ndim
```

Pythonでは `bool` が `int` のsubclassなので、`True == 1` として誤受理しないよう明示的に拒否する。

このmode規約は、

```text
unfold
fold
mode_dot
Tucker ranks key
HOOI factor key
```

で統一する。

---

## 4. `unfold`

```python
unfold(X: torch.Tensor, mode: int) -> torch.Tensor
```

### 目的

N階Tensorの指定modeを行方向へ移し、残りのmodeを列方向へまとめる。

入力：

$$
\mathcal{X}\in\mathbb{R}^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

mode $n$ の出力shape：

$$
\boxed{
I_n
\times
\prod_{m\neq n} I_m
}
$$

実装は、

```python
X.movedim(mode, 0).flatten(start_dim=1)
```

を使う。

### 重要な規約

ここでいうunfoldingは `torch.Tensor.unfold()` のスライディングウィンドウ抽出とは別物である。

---

## 5. `fold`

```python
fold(
    unfolded: torch.Tensor,
    mode: int,
    shape: tuple[int, ...],
) -> torch.Tensor
```

### 目的

`unfold` された2次元行列を元のaxis配置へ戻す。

### validation

- `shape` は2次元以上
- `shape` の各dimensionはboolではない正のint
- `mode` は有効範囲
- `unfolded.ndim == 2`
- row数が `shape[mode]` と一致
- 総要素数が `prod(shape)` と一致
- column数が期待値と一致

最後のcolumn数確認により、要素数だけ同じ転置行列を黙ってreshapeしない。

### inverse contract

validな入力について、

```python
fold(unfold(X, mode), mode, tuple(X.shape))
```

はXと同じshape・axis順序へ戻る。

---

## 6. `mode_dot`

```python
mode_dot(
    X: torch.Tensor,
    matrix: torch.Tensor,
    mode: int,
) -> torch.Tensor
```

n-mode product、

$$
\mathcal{Y}=\mathcal{X}\times_n A
$$

を実装する。

行列shapeは、

$$
A\in\mathbb{R}^{J\times I_n}
$$

とし、

```text
matrix.shape[1] == X.shape[mode]
```

を要求する。

出力shapeは、

```text
X.shape
```

のmode $n$ のdimensionだけを

```text
I_n → J
```

へ置き換える。

実装上は、

```text
unfold
→ matrix @ unfolded
→ fold
```

の順に処理する。

したがって `unfold / fold / mode_dot` のaxis conventionはセットで固定する。

---

## 7. dtype / device / autograd

`tensor` packageは原則としてTensorをコピーしてCPUへ移したり、float32へcastしたりしない。

`movedim / flatten / reshape / matmul` のPyTorch semanticsに従うため、validなdtype/deviceの組合せではそのまま保持される。

`mode_dot` 自体は `detach()` しない。

そのため、

```text
X.requires_grad=True
matrix.requires_grad=True
```

の計算グラフをTensor-level APIで保持できる。

HOSVD/HOOIのModule構築時にどこでgraphを切るかは `compression` 側の責務とする。

---

## 8. validation moduleの位置付け

`tensor/validation.py` は現在Public APIとしてexportしていない。

内部helper：

```text
validate_tensor_ndim_at_least_2
validate_tensor_shape
validate_mode_index
validate_positive_shape
```

これらはCore API v1の直接固定対象ではない。

Public APIの挙動を維持できるなら、将来TT/MPS追加時にvalidation配置を整理・共通化してよい。

---

## 9. `compression/tucker_validation.py`との重複

現在、

```text
tensor/validation.py
compression/tucker_validation.py
```

に `validate_mode_index` / `validate_positive_shape` 相当の処理が重複している。

これは既知の技術的負債だが、今回のCore API v1固定では大規模な依存再編をしない。

将来共通化する場合も、

```text
tensor → compression
```

の逆依存を作らないことを優先する。

候補は、より下位のshared validation moduleへ移すことだが、Public API contractが変わらない限り内部変更として扱う。

---

## 10. TT/MPSへの再利用方針

TT/MPSではreshapeや逐次SVDが中心になるため、必ずしもTuckerと同じunfold conventionだけで全処理を表す必要はない。

ただし、既存のmode演算が必要な場合はこのAPIを再利用する。

新しいtensor operationを追加するときは、

```text
既存unfold/fold/mode_dotの意味を変える
```

のではなく、別名の新APIを追加する。

例：

```text
tensorize
matricize_chain_split
merge_adjacent_modes
```

等が必要になっても、既存mode-n conventionは保持する。

---

## 11. testで固定する性質

Core API v1では少なくとも次を回帰テストで維持する。

- non-square Tensor
- 3階以上のTensor
- `fold(unfold(X))` のinverse
- mode別shape
- invalid mode
- bool mode
- 1階Tensor拒否
- zero-sized dimension拒否
- unfolded row / column / numel不一致拒否
- `mode_dot` のdimension不一致拒否

Tensor演算の変更時は、Tucker/HOOI testが通るだけでなく、この低位contract自体を確認する。
