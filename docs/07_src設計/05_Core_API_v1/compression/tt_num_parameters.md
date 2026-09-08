# `tt_num_parameters`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb`, `notebooks/30_tt_mps/00_fundamentals/02_truncation_error_tradeoff.ipynb`

## 責務

TT core列が保持する総要素数を数え、TT表現の保存parameter数をPython `int`として返す。

$d$ 個のcore

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k}
$$

に対して、

$$
P_{\mathrm{TT}}
=
\sum_{k=1}^{d}
r_{k-1}n_kr_k
$$

を計算する。

## Signature

```python
tt_num_parameters(
    cores: list[torch.Tensor],
) -> int
```

## 引数

- `cores`: parameter数を数えるTT core列。`tt_reconstruct`と同じTT構造contractを要求する。public type hintはlistだが、runtime validatorではlist / tupleを受理する。

## 戻り値

全coreの`numel()`の総和をPython `int`として返す。

$$
\boxed{
\texttt{tt\_num\_parameters(cores)}
=
\sum_k \texttt{cores[k].numel()}
}
$$

値が小さいほどTT表現が保持するscalar数は少ない。ただし、この関数単体ではdense Tensorとの比率や圧縮率までは計算しない。

## 使用場面

- TT rank sweepにおける保存量比較。
- dense Tensorの`X.numel()`とのparameter ratio計算。
- rank上限`max_rank`を変えたときのstorage trade-off確認。
- 後続のTT/MPSモデル圧縮で、core storageの基礎指標を作るとき。

## 処理概要

1. `validate_tt_cores(cores)`でcore列の構造を検証する。
2. 各coreについて`core.numel()`を取得する。
3. `sum(...)`で全coreの要素数を合計する。
4. Python `int`を返す。

## 主なcontract / 注意事項

- `cores=[]`は`ValueError`とする。
- 不正なbond接続、境界bond、dtype/device不一致なども`validate_tt_cores`で拒否する。
- 数えているのは**TT coreに実際に保存されるscalar数**であり、byte数ではない。
- dtypeが`float32`でも`float64`でも、同じshapeならparameter数は同じ。memory byte数を知りたい場合はdtype sizeを別途掛ける必要がある。
- gauge freedomによる表現の非一意性を差し引いた「独立自由度」を数える関数ではない。単純にcore storageの総要素数を返す。
- dense側との比較では、例えば

  $$
  \text{parameter ratio}
  =
  \frac{P_{\mathrm{TT}}}{\prod_k n_k}
  $$

  を別途計算する。ratioが1を超える場合、TT表現の方がdenseより大きい。

## 関連API

`tt_svd_exact`, `tt_svd`, `tt_reconstruct`, `validate_tt_cores`
