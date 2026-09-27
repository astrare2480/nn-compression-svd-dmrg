# `tt_reconstruct`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/00_tt_svd_3way_basics.ipynb`, `notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb`

## 責務

TT core列を左から順にbond縮約し、対応するdense Tensorを再構成する。

入力coreが`tt_svd_exact`や`tt_svd`由来かどうかには依存せず、TT構造contractを満たす任意のcore列を受理する。

## Signature

```python
tt_reconstruct(
    cores: list[torch.Tensor],
) -> torch.Tensor
```

## 引数

- `cores`: TT core列。public type hintは`list[torch.Tensor]`で、各coreは3階Tensorである必要がある。runtime validationではlist / tupleを受理する。

  $d$ 個のcoreを

  $$
  G^{(k)}
  \in
  \mathbb R^{r_{k-1}\times n_k\times r_k}
  $$

  とすると、次の構造を要求する。

  - core列が空でない。
  - 全coreが`torch.Tensor`かつ3階。
  - 全dimensionが正。
  - 左端bondが1。
  - 右端bondが1。
  - 隣接core間でright bondと次coreのleft bondが一致する。
  - 全coreでdtypeとdeviceが一致する。

## 戻り値

境界bondを除いたdense Tensorを返す。

$d$ 個のcoreのphysical dimensionが

$$
n_1,\ldots,n_d
$$

なら、戻り値のshapeは

$$
(n_1,\ldots,n_d)
$$

となる。

入力coreのdtype / deviceを維持する。

## 使用場面

- `tt_svd_exact`の復元誤差確認。
- `tt_svd`でrank制限した近似Tensorの構築。
- TT coreとdense Tensorの数値一致テスト。
- 後続TT/MPS演算の基準実装として、dense側の正解を作るとき。

## 処理概要

1. `validate_tt_cores(cores)`でTT構造contractを確認する。
2. `result = cores[0]`として左端coreから開始する。
3. 2個目以降の各coreについて、

   ```python
   torch.tensordot(result, core, dims=([-1], [0]))
   ```

   を実行する。
4. 毎回、現在の`result`の最後のaxisであるright bondと、次coreの先頭axisであるleft bondを縮約する。
5. 全coreを縮約するとshapeは

   $$
   (1,n_1,\ldots,n_d,1)
   $$

   になる。
6. `squeeze(0).squeeze(-1)`で**境界bondだけ**を除去して返す。

### 縮約構造

```text
G1: (1, n1, r1)
        |
        r1
        |
G2: (r1, n2, r2)
        |
        r2
        |
...
        |
Gd: (r_{d-1}, nd, 1)

        ↓ bondを順次縮約

(1, n1, n2, ..., nd, 1)
        ↓ 境界1だけsqueeze
(n1, n2, ..., nd)
```

## 主なcontract / 注意事項

- `cores=[]`は`ValueError`とする。
- 各coreのshapeだけでなく、**隣接bondの一致**を事前に検証する。
- core間のdtype不一致は`TypeError`、device不一致は`ValueError`として拒否する。
- `squeeze()`を引数なしでは使わない。physical dimensionに`n_k=1`があっても、そのaxisを消さないため、左端と右端の境界axisだけを明示的に除去する。
- SVD由来のcoreである必要はない。TT構造を満たせば再構成する。
- `tt_svd`由来のcoreでは、元Tensorそのものではなくrank制限後の近似Tensorが得られる。
- 1 coreだけの`(1,n_1,1)`も現行validator上は構造的に受理され、戻り値は1次元Tensorになる。通常の`tt_svd_exact` / `tt_svd`は入力`ndim >= 2`のため2 core以上を生成する。

## 関連API

`tt_svd_exact`, `tt_svd`, `tt_num_parameters`, `validate_tt_cores`, `torch.tensordot`
